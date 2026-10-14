"""
Houses of several floors, and duplexes.

A house of 2+ floors is a home on every floor: the stair (and a lift inside) sit in a front band with a
stair lobby; every floor's front door opens off the lobby, so each floor is reached by the stair → lobby →
front door and nobody walks through another floor's home (a lift tower outside lands on the lobby too).
A duplex is one home on two floors: living, dining, kitchen (and the elders' room) below, the bedrooms and a
family lounge above, the stair rising from the foyer and landing on the same spot — and in 3D the first floor
is its own plan, walked to by the stair and visited by the guided tour.
"""

import base64
import os
import sys
import unittest
from collections import deque

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from floorplan_generator.layout_engine import LayoutEngine  # noqa: E402
from floorplan_generator.family_analyzer import FamilyProfile  # noqa: E402
from floorplan_renderer import walkthrough as wt  # noqa: E402
from floorplan_renderer.svg_renderer import SVGRenderer  # noqa: E402

FAM = FamilyProfile(adults=2, children=2, elderly=0, needs_pooja_room=True)
ELDERS = FamilyProfile(adults=2, children=1, elderly=2, needs_pooja_room=True)
_CACHE = {}


def _gen(w, h, bhk, facing, floors, btype="Independent House", fam=FAM, **kw):
    key = (w, h, bhk, facing, floors, btype, id(fam), tuple(sorted(kw.items())))
    if key not in _CACHE:
        _CACHE[key] = LayoutEngine(w, h, bhk, True, facing, "Standard", num_floors=floors, building_type=btype,
                                   family_profile=fam, **kw).generate()
    return _CACHE[key]


def _room_at(plan, x, y):
    for r in plan.rooms:
        if r.category in ("PARKING", "BIKE_PARKING", "GARDEN", "POOL", "WATER_TANK", "GENERATOR", "SECURITY"):
            continue
        if r.x < x < r.right and r.y < y < r.top:
            return r
    return None


def _sides(plan, d):
    if d.wall_side in ("N", "S"):
        mx, my = d.x + d.width / 2, d.y
        return _room_at(plan, mx, my + 0.3), _room_at(plan, mx, my - 0.3)
    mx, my = d.x, d.y + d.width / 2
    return _room_at(plan, mx + 0.3, my), _room_at(plan, mx - 0.3, my)


def _grid(nav):
    bits = np.unpackbits(np.frombuffer(base64.b64decode(nav["bits"]), dtype=np.uint8))[: nav["nx"] * nav["ny"]]
    return bits.reshape(nav["ny"], nav["nx"]).astype(bool)


def _reach(nav, a, b) -> bool:
    g = _grid(nav)

    def near(x, y, r=1.5):
        i0, j0 = int((x - nav["x0"]) / nav["cell"]), int((y - nav["y0"]) / nav["cell"])
        k, best = int(r / nav["cell"]), None
        for dj in range(-k, k + 1):
            for di in range(-k, k + 1):
                i, j = i0 + di, j0 + dj
                if 0 <= i < nav["nx"] and 0 <= j < nav["ny"] and g[j, i] and (best is None or di * di + dj * dj < best[0]):
                    best = (di * di + dj * dj, (i, j))
        return best[1] if best else None
    s, t = near(*a), near(*b)
    if s is None or t is None:
        return False
    seen, q = {s}, deque([s])
    while q:
        c = q.popleft()
        if c == t:
            return True
        for n in ((c[0] + 1, c[1]), (c[0] - 1, c[1]), (c[0], c[1] + 1), (c[0], c[1] - 1)):
            if 0 <= n[0] < nav["nx"] and 0 <= n[1] < nav["ny"] and g[n[1], n[0]] and n not in seen:
                seen.add(n)
                q.append(n)
    return False


class TestFloors(unittest.TestCase):

    def test_01_a_home_on_every_floor(self):
        for facing in "NESW":
            plan = _gen(12.33, 16.4, "3BHK", facing, 3)
            info = plan.engineering["apartment"]
            self.assertEqual((info["kind"], info["flats_total"]), ("house", 3), facing)
            lobby = next(r for r in plan.rooms if r.name == "Stair Lobby")
            stair = next(r for r in plan.rooms if r.category == "STAIRS")
            fx, fy, fw, fd = plan.building_footprint
            self.assertAlmostEqual(lobby.top, fy + fd, places=2)            # at the front
            self.assertAlmostEqual(stair.top, fy + fd, places=2)
            main = [d for d in plan.doors if d.is_main_entrance]
            self.assertEqual([d.room_name for d in main], ["Stair Lobby"], facing)
            homes = [d for d in plan.doors if lobby.name in {r.name for r in _sides(plan, d) if r} and not d.is_main_entrance
                     and d.room_name != "Lift / Elevator"]
            self.assertEqual(len(homes), 1, facing)                         # the home's own front door
            self.assertTrue(any(abs(o[0] - o[2]) < 1e-6 and stair.y - 0.01 <= min(o[1], o[3]) <= stair.top
                                and abs(o[0] - (stair.right if stair.x < lobby.x else stair.x)) < 0.02
                                for o in plan.openings), facing)          # the stair is open to the lobby
            self.assertTrue(any("home on every floor" in n for n in plan.design_notes), facing)
            self.assertGreaterEqual(plan.vastu_score, 60, facing)

    def test_02_the_floors_are_walked_the_same_way(self):
        plan = _gen(12.33, 16.4, "3BHK", "N", 3)
        sc = wt.build_scene(plan)
        self.assertEqual([l["label"] for l in sc["levels_list"]], ["Ground floor", "Floor 1", "Floor 2", "Terrace"])
        lob = next(r for r in sc["rooms"] if r["name"] == "Stair Lobby")["r"]
        home = next(r for r in sc["rooms"] if r["cat"] == "BEDROOM")["r"]
        lc = ((lob[0] + lob[2]) / 2, (lob[1] + lob[3]) / 2)
        hc = ((home[0] + home[2]) / 2, (home[1] + home[3]) / 2)
        self.assertTrue(_reach(sc["nav"], lc, hc))                          # lobby → front door → a bedroom
        self.assertTrue(_reach(sc["nav_typ"], lc, hc))                      # … and the same upstairs
        st = next(s for s in sc["stairs"] if not s["external"])
        self.assertAlmostEqual(st["top"], sc["levels_list"][1]["z"], delta=0.02)
        ex = st["exit"]
        mid = ((ex[0] + ex[2]) / 2, (ex[1] + ex[3]) / 2)
        self.assertTrue(lob[0] - 0.05 <= mid[0] <= lob[2] + 0.05 and lob[1] - 0.05 <= mid[1] <= lob[3] + 0.05,
                        "upstairs you step out of the stair into the lobby")
        main = next(d for d in sc["doors"] if d["kind"] == "main")
        self.assertEqual(main["upper"], "parapet")
        self.assertTrue(sc["walls"]["parapet"])                             # the lobby's open front

    def test_03_lifts_in_a_house_of_floors(self):
        lobby = _gen(12.33, 16.4, "3BHK", "N", 3, has_lift=True)
        lr = next(r for r in lobby.rooms if r.category == "LIFT")
        ld = next(d for d in lobby.doors if d.room_name == lr.name)
        self.assertIn("Stair Lobby", {r.name for r in _sides(lobby, ld) if r})     # opens on the lobby
        self.assertEqual(lobby.engineering["lift"]["position"], "lobby")
        ent = _gen(12.33, 16.4, "3BHK", "N", 3, has_lift=True, lift_position="entrance")
        self.assertEqual(ent.engineering["lift"]["position"], "entrance")
        sc = wt.build_scene(ent)
        main = next(d for d in sc["doors"] if d["kind"] == "main")
        self.assertEqual(main["upper"], "balcony")                          # upstairs: the tower's landing
        side = _gen(15.0, 20.0, "3BHK", "W", 3, has_lift=True, lift_position="side")
        if side.engineering["lift"].get("entry") == "side door":
            fx, fy, fw, fd = side.building_footprint
            entry = [d for d in side.doors if d.wall_side in ("E", "W") and (abs(d.x - fx) < 0.02 or abs(d.x - fx - fw) < 0.02)
                     and "Lift" not in d.room_name]
            self.assertTrue(entry)
            self.assertEqual(entry[0].room_name, "Stair Lobby")             # the side door is the lobby's, never a home's

    def test_04_duplex_is_one_home_on_two_floors(self):
        plan = _gen(12.33, 16.4, "3BHK", "N", 2, "Duplex Home")
        up = plan.upper_plan
        g_cats = [r.category for r in plan.rooms]
        u_cats = [r.category for r in up.rooms]
        self.assertEqual(g_cats.count("KITCHEN") + u_cats.count("KITCHEN"), 1)   # one home, one kitchen
        self.assertIn("KITCHEN", g_cats)
        self.assertIn("LIVING", g_cats)
        self.assertTrue(any(r.role == "MASTER_BEDROOM" for r in up.rooms))
        self.assertFalse(any(r.role == "MASTER_BEDROOM" for r in plan.rooms))
        self.assertTrue(any(r.name == "Family Lounge" for r in up.rooms))
        self.assertEqual(plan.num_floors, 2)
        self.assertEqual([d.room_name for d in plan.doors if d.is_main_entrance], ["Entrance Foyer"])
        self.assertFalse(any(d.is_main_entrance for d in up.doors))
        st_g = [(s.x, s.y, s.w, s.h) for s in plan.staircases]
        st_u = [(s.x, s.y, s.w, s.h) for s in up.staircases]
        self.assertEqual(st_g, st_u)                                         # the stair lands on the same spot
        self.assertIn("duplex", plan.engineering)
        self.assertNotIn("apartment", plan.engineering)
        self.assertGreater(len(SVGRenderer().render_to_string(up, mode="blueprint")), 1000)
        self.assertIn("First floor", up.project_name)
        old = _gen(12.33, 16.4, "3BHK", "N", 2, "Duplex Home", ELDERS)      # elders never climb
        self.assertTrue(any(r.role == "ELDER_BEDROOM" for r in old.rooms))
        self.assertFalse(any(r.role == "ELDER_BEDROOM" for r in old.upper_plan.rooms))
        lifted = _gen(12.33, 16.4, "3BHK", "N", 2, "Duplex Home", has_lift=True, lift_position="side")
        lifts_g = [(r.x, r.y) for r in lifted.rooms if r.category == "LIFT"]
        lifts_u = [(r.x, r.y) for r in lifted.upper_plan.rooms if r.category == "LIFT"]
        self.assertTrue(lifts_g and lifts_g == lifts_u)                      # the lift stays inside the home
        self.assertFalse(any("External" in r.name for r in lifted.rooms))

    def test_05_duplex_in_3d(self):
        plan = _gen(12.33, 16.4, "3BHK", "N", 2, "Duplex Home")
        sc = wt.build_scene(plan)
        self.assertTrue(sc["upper"] and sc["duplex"])
        self.assertEqual([l.get("plan") for l in sc["levels_list"]], [0, 1, 0])
        names = {r["name"] for r in sc["upper"]["rooms"]}
        self.assertIn("Family Lounge", names)
        self.assertNotIn("Kitchen", names)
        self.assertTrue(sc["upper"]["furniture"])
        up_stops = [s for s in sc["tour"] if s.get("level") == 1]
        self.assertTrue(any("Master" in s["title"] for s in up_stops))
        self.assertTrue(all(s["target"] for s in up_stops))
        # upstairs walking: from where the stair arrives to the master bedroom
        st = next(s for s in sc["stairs"] if not s["external"])
        ex = st["exit"]
        m = next(r for r in sc["upper"]["rooms"] if "Master" in r["name"])["r"]
        self.assertTrue(_reach(sc["nav_typ"] or sc["nav1"], ((ex[0] + ex[2]) / 2, (ex[1] + ex[3]) / 2),
                               ((m[0] + m[2]) / 2, (m[1] + m[3]) / 2)))
        foyer = next(r for r in sc["rooms"] if r["name"] == "Entrance Foyer")["r"]
        arch = next(a for a in sc["arches"] if abs(a["a"][1] - foyer[1]) < 0.02 and abs(a["b"][1] - foyer[1]) < 0.02)
        inside = ((arch["a"][0] + arch["b"][0]) / 2, foyer[1] - 0.6)
        self.assertEqual(next(r for r in sc["rooms"] if r["r"][0] < inside[0] < r["r"][2]
                              and r["r"][1] < inside[1] < r["r"][3])["cat"], "LIVING")
        self.assertTrue(_reach(sc["nav"], ((foyer[0] + foyer[2]) / 2, (foyer[1] + foyer[3]) / 2), inside))


    def test_06_duplex_replan_and_single_floor_houses_unchanged(self):
        plan = _gen(12.33, 16.4, "3BHK", "N", 2, "Duplex Home")
        opts = LayoutEngine(**plan.generation_params, design_intents=[{"type": "size", "room": "kitchen", "factor": 1.2}],
                            base_layout=plan.layout_state).generate_alternatives(1)
        self.assertTrue(opts and getattr(opts[0], "upper_plan", None) is not None)
        k0 = next(r for r in plan.rooms if r.category == "KITCHEN")
        k1 = next(r for r in opts[0].rooms if r.category == "KITCHEN")
        self.assertGreater(k1.w * k1.h, k0.w * k0.h)
        one = LayoutEngine(12.0, 15.0, "2BHK", True, "N").generate()
        self.assertNotIn("apartment", one.engineering)
        self.assertIsNone(getattr(one, "upper_plan", None))


if __name__ == "__main__":
    unittest.main()
