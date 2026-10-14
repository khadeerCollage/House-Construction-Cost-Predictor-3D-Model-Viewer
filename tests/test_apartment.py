"""
Apartment floor plate (floorplan_generator/apartment.py via LayoutEngine for 'Apartment').

Covers: 1 / 2 / 4 flats per floor only where they fit; every flat's main door opens onto the
common lobby / gallery / corridor (never through another home); the stair opens onto it and the
lift door faces it; 230 mm walls between homes and around the core; windows only to open air;
flats of one type are identical and front flats are planned for their own facing; a Smart
Re-Plan wish changes every flat of the type; houses are untouched; narrow plots fall back.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from floorplan_generator.layout_engine import LayoutEngine  # noqa: E402
from floorplan_generator import apartment as ap  # noqa: E402

APT = "Apartment / Multi-Storey Building"
_CACHE = {}


def _apt(w, h, bhk="2BHK", facing="N", floors=4, n="auto", lift=True):
    key = (w, h, bhk, facing, floors, n, lift)
    if key not in _CACHE:
        _CACHE[key] = LayoutEngine(w, h, bhk, True, facing, num_floors=floors, building_type=APT,
                                   has_lift=lift, flats_per_floor=n).generate()
    return _CACHE[key]


def _flat(name):
    return name.split(" · ")[0] if " · " in name else ""


def _room_at(plan, x, y):
    for r in plan.rooms:
        if r.category in ("PARKING", "BIKE_PARKING", "GARDEN", "POOL", "WATER_TANK", "GENERATOR", "SECURITY"):
            continue
        if r.x < x < r.right and r.y < y < r.top:
            return r
    return None


def _door_sides(plan, d):
    if d.wall_side in ("N", "S"):
        mx, my = d.x + d.width / 2, d.y
        a, b = (mx, my + 0.3), (mx, my - 0.3)
    else:
        mx, my = d.x, d.y + d.width / 2
        a, b = (mx + 0.3, my), (mx - 0.3, my)
    return _room_at(plan, *a), _room_at(plan, *b)


def _on_edge(o, r):
    mx, my = (o[0] + o[2]) / 2, (o[1] + o[3]) / 2
    on_h = abs(o[1] - o[3]) < 1e-6 and (abs(my - r.y) < 0.02 or abs(my - r.top) < 0.02) and r.x - 0.01 <= mx <= r.right + 0.01
    on_v = abs(o[0] - o[2]) < 1e-6 and (abs(mx - r.x) < 0.02 or abs(mx - r.right) < 0.02) and r.y - 0.01 <= my <= r.top + 0.01
    return on_h or on_v


class TestApartment(unittest.TestCase):

    def test_01_flats_per_floor_only_where_they_fit(self):
        big = ap.flats_options(30.0, 30.0, "N", "2BHK", has_lift=True)
        self.assertTrue(all(big[n]["fits"] for n in (1, 2, 4)), big)
        small = ap.flats_options(12.19, 18.29, "N", "2BHK", has_lift=True)
        self.assertTrue(small[1]["fits"])
        self.assertFalse(small[4]["fits"])
        for n in (1, 2, 4):
            plan = _apt(30.0, 30.0, n=n)
            self.assertEqual(plan.engineering["apartment"]["flats_per_floor"], n)
            labels = {_flat(r.name) for r in plan.rooms if r.role != "COMMON" and r.category not in ("STAIRS", "LIFT")
                      and r.category not in ("PARKING", "GARDEN", "BIKE_PARKING", "WATER_TANK", "POOL", "GENERATOR", "SECURITY")}
            self.assertEqual(len(labels), n, labels)

    def test_02_every_flat_opens_onto_the_common_space(self):
        for args in ((12.19, 18.29, "2BHK", "N", 4, 1), (18.0, 24.0, "2BHK", "E", 5, 2), (30.0, 30.0, "2BHK", "N", 6, 4)):
            plan = _apt(*args[:5], n=args[5])
            common = {r.name for r in plan.rooms if r.role == "COMMON"}
            per_flat = {}
            for d in plan.doors:
                r1, r2 = _door_sides(plan, d)
                if not (r1 and r2):
                    continue
                names = {r1.name, r2.name}
                if names & common:
                    other = (names - common).pop() if names - common else None
                    if other and other not in ("Staircase", "Lift / Elevator"):
                        per_flat.setdefault(_flat(other), []).append(d)
                # never a door straight from one home into another
                f1, f2 = _flat(r1.name), _flat(r2.name)
                if r1.name not in common and r2.name not in common and r1.category not in ("STAIRS", "LIFT") \
                        and r2.category not in ("STAIRS", "LIFT"):
                    self.assertEqual(f1, f2, (args, r1.name, r2.name))
            self.assertEqual(len(per_flat), args[5], (args, list(per_flat)))
            self.assertTrue(all(len(v) == 1 for v in per_flat.values()), args)
            lobby = next(r for r in plan.rooms if r.role == "COMMON" and "Duct" not in r.name)
            lift_door = next(d for d in plan.doors if d.room_name == "Lift / Elevator")
            self.assertIn(lobby.name, {r.name for r in _door_sides(plan, lift_door) if r})
            stair = next(r for r in plan.rooms if r.category == "STAIRS")
            touches = [o for o in plan.openings if _on_edge(o, stair)]
            self.assertTrue(touches, f"{args}: the stair has no opening onto the lobby")
            main = [d for d in plan.doors if d.is_main_entrance]
            self.assertEqual(len(main), 1)
            self.assertIn(main[0].room_name, common)

    def test_03_party_walls_and_windows_to_open_air(self):
        plan = _apt(18.0, 24.0, "2BHK", "E", 5, 2)
        fx, fy, fw, fd = plan.building_footprint
        homes = {}
        for r in plan.rooms:
            homes[r.name] = "common" if r.role == "COMMON" or r.category in ("STAIRS", "LIFT") else _flat(r.name)
        thick = 0
        for w in plan.walls:
            if w.wall_type != "internal":
                continue
            mx, my = (w.x1 + w.x2) / 2, (w.y1 + w.y2) / 2
            nx, ny = (0, 0.2) if abs(w.y1 - w.y2) < 1e-6 else (0.2, 0)
            a, b = _room_at(plan, mx + nx, my + ny), _room_at(plan, mx - nx, my - ny)
            if a and b and homes.get(a.name) != homes.get(b.name):
                self.assertAlmostEqual(w.thickness, 0.23, msg=(a.name, b.name))
                thick += 1
        self.assertGreater(thick, 2)
        for wd in plan.windows:
            ox, oy = {"N": (0, 1), "S": (0, -1), "E": (1, 0), "W": (-1, 0)}[wd.wall_side]
            px, py = wd.x + ox * 0.35, wd.y + oy * 0.35
            if not (fx < px < fx + fw and fy < py < fy + fd):
                continue
            host = _room_at(plan, px, py)
            self.assertTrue(host is not None and ("Duct" in host.name or host.name == "Common Gallery"),
                            (wd.room_name, host.name if host else None))

    def test_04_flats_identical_and_vastu_for_their_own_facing(self):
        plan = _apt(18.0, 24.0, "2BHK", "E", 5, 2)
        a = sorted((r.name[4:], round(r.w, 2), round(r.h, 2)) for r in plan.rooms if r.name.startswith("A · "))
        b = sorted((r.name[4:], round(r.w, 2), round(r.h, 2)) for r in plan.rooms if r.name.startswith("B · "))
        self.assertEqual(a, b)
        self.assertGreaterEqual(plan.vastu_score, 70)
        plan4 = _apt(30.0, 30.0, "2BHK", "N", 6, 4)
        c = sorted((r.name[4:], round(r.w, 2), round(r.h, 2)) for r in plan4.rooms if r.name.startswith("C · "))
        d = sorted((r.name[4:], round(r.w, 2), round(r.h, 2)) for r in plan4.rooms if r.name.startswith("D · "))
        self.assertEqual(c, d)
        self.assertEqual(plan4.engineering["apartment"]["turned"], [False, False, True, True])
        self.assertIn("front flats", " ".join(plan4.design_notes))

    def test_05_replan_changes_every_flat_of_the_type(self):
        plan = _apt(18.0, 24.0, "2BHK", "E", 5, 2)
        before = {r.name: r.w * r.h for r in plan.rooms if "Living" in r.name}
        opts = LayoutEngine(**plan.generation_params, design_intents=[{"type": "size", "room": "living", "factor": 1.2}],
                            base_layout=plan.layout_state).generate_alternatives(2)
        self.assertTrue(opts)
        after = [r for r in opts[0].rooms if "Living" in r.name]
        self.assertEqual(len(after), 2)
        self.assertEqual(round(after[0].w * after[0].h, 2), round(after[1].w * after[1].h, 2))
        self.assertGreater(after[0].w * after[0].h, max(before.values()))
        self.assertEqual(opts[0].engineering["apartment"]["flats_per_floor"], 2)

    def test_06_houses_unchanged_and_narrow_plots_fall_back(self):
        house = LayoutEngine(12.0, 16.0, "2BHK", True, "N", num_floors=1).generate()
        self.assertNotIn("apartment", house.engineering)
        self.assertNotIn("flats_per_floor", house.generation_params)
        g1 = LayoutEngine(12.0, 16.0, "2BHK", True, "N", num_floors=2).generate()     # a home on each floor
        self.assertEqual(g1.engineering["apartment"]["kind"], "house")
        self.assertNotIn("flats_per_floor", g1.generation_params)
        narrow = LayoutEngine(5.0, 16.0, "1BHK", True, "N", num_floors=3, building_type=APT, has_lift=True).generate()
        self.assertTrue(any("too narrow" in n for n in narrow.design_notes))
        self.assertNotIn("apartment", narrow.engineering)          # one home with its own stair (old behaviour)


if __name__ == "__main__":
    unittest.main()
