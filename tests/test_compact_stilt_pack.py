"""
2026-10-16: compact homes never fall back to the legacy layout · a stilt building's core at the back with
columns set out for the cars · the front elevation · the builder & bank pack (PDF).
"""

import os
import re
import sys
import unittest
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from floorplan_generator import gates as gt
from floorplan_generator.family_analyzer import FamilyProfile
from floorplan_generator.land_units import LandUnitsEngine
from floorplan_generator.layout_engine import LayoutEngine, _is_site_room
from floorplan_renderer.elevation import front_elevation_svg
from floorplan_renderer.walkthrough import build_scene

FAM = FamilyProfile(adults=2, children=2)
STILT3 = {"car_parking": {"enabled": True, "slots": 3, "mode": "stilt"}}
_CACHE = {}


def _plot(cents):
    p = LandUnitsEngine.solve_parcel(cents, "Rectangle", 1.33)
    return p.width_m, p.length_m


def _gen(key, *args, **kw):
    if key not in _CACHE:
        _CACHE[key] = LayoutEngine(*args, **kw).generate()
    return _CACHE[key]


def _apt(cents, bhk, facing, comps=STILT3, floors=4):
    W, H = _plot(cents)
    return _gen(("apt", cents, bhk, facing, str(comps), floors), W, H, bhk, True, facing, "Standard",
                building_type="Apartment", num_floors=floors, has_lift=True, family_profile=FAM, additional_components=comps)


class TestCompactHomes(unittest.TestCase):
    def test_01_tight_plots_get_a_real_layout(self):
        """The 'tight' BHKs the form offers used to fall back to the legacy subdivision (no Smart Re-Plan, a
        bedroom in the Brahmasthan); now an architect's compact steps keep every bedroom in a band layout."""
        for cents, bhk, facing, step in ((2.0, "1BHK", "N", "One bathroom for the home"),
                                         (3.0, "3BHK", "E", "shares the common bathroom"),
                                         (4.0, "4BHK", "W", "shares the common bathroom")):
            W, H = _plot(cents)
            plan = _gen(("house", cents, bhk, facing), W, H, bhk, True, facing, "Standard", family_profile=FAM)
            self.assertTrue(plan.layout_state, (cents, bhk, facing))                   # not the legacy fallback
            self.assertEqual(sum(1 for r in plan.rooms if r.category == "BEDROOM"), int(bhk[0]), (cents, bhk))
            self.assertTrue(any(step in n for n in plan.design_notes), plan.design_notes)
            self.assertFalse(any("functional subdivision" in n for n in plan.design_notes))
            self.assertGreaterEqual(plan.layout_state.get("relax_level", 0), 5)
            again = LayoutEngine(**plan.generation_params).generate()                  # rebuilds bit for bit
            self.assertEqual(again.layout_state.get("relax_level"), plan.layout_state.get("relax_level"))
            self.assertEqual([(r.name, r.x, r.y, r.w, r.h) for r in again.rooms],
                             [(r.name, r.x, r.y, r.w, r.h) for r in plan.rooms])
        # a re-plan replays the same relaxations (the program matches the base)
        W, H = _plot(3.0)
        plan = _CACHE[("house", 3.0, "3BHK", "E")]
        opts = LayoutEngine(**plan.generation_params, design_intents=[{"type": "lock", "room": "kitchen"}],
                            base_layout=plan.layout_state).generate_alternatives(1)
        self.assertTrue(opts and opts[0].layout_state)

    def test_02_roomy_plots_unchanged(self):
        """Plots that solved before take none of the compact steps."""
        W, H = _plot(5.0)
        plan = _gen(("house", 5.0, "3BHK", "N"), W, H, "3BHK", True, "N", "Standard", family_profile=FAM)
        self.assertLessEqual(plan.layout_state.get("relax_level", 0), 4)
        self.assertFalse(any("shares the common bathroom" in n or "pooja shelf" in n for n in plan.design_notes))


class TestStiltCore(unittest.TestCase):
    def test_03_three_cars_under_a_5_cent_apartment(self):
        for bhk in ("2BHK", "3BHK"):
            for facing in "NE":
                plan = _apt(5.0, bhk, facing)
                eng = plan.engineering
                ap, st = eng["apartment"], eng["stilt"]
                self.assertEqual(ap["core_at"], "rear", (bhk, facing))
                self.assertEqual((st["fitted"], st["wanted"]), (3, 3), (bhk, facing, st))
                self.assertTrue(eng.get("grid_x"))
                shafts, lobby, cols = gt.stilt_context(plan.rooms, plan.building_footprint, _is_site_room,
                                                       eng["grid_x"], ap["walkways"])
                car = gt.CARS[st["car"]]
                for b in st["bays"]:
                    cx = (b[0] + b[2]) / 2
                    body = (cx - car["W"] / 2, b[3] - 0.15 - car["L"], cx + car["W"] / 2, b[3] - 0.15)
                    for o in shafts + cols:
                        self.assertFalse(body[0] < o[2] and o[0] < body[2] and body[1] < o[3] and o[1] < body[3],
                                         (bhk, facing, b, o))
                    for w in ap["walkways"]:                    # the walkway to the lift stays clear
                        self.assertFalse(b[0] < w[0] + w[2] - 1e-6 and w[0] < b[2] - 1e-6 and
                                         b[1] < w[1] + w[3] - 1e-6 and w[1] < b[3] - 1e-6, (b, w))
                # the core is at the back, the flat in front; no building door on the typical floor (you come
                # in on the stilt, along the walkway), and the flat's own door opens onto the lobby
                fx, fy, fw, fd = plan.building_footprint
                stair = next(r for r in plan.rooms if r.category == "STAIRS")
                self.assertAlmostEqual(stair.y, fy, places=2)
                self.assertFalse(any(d.is_main_entrance for d in plan.doors))
                lob = next(r for r in plan.rooms if r.name == ap["lobby"])
                self.assertTrue(any(abs(d.y - lob.top) < 0.05 and lob.x - 0.05 <= d.x <= lob.right + 0.05
                                    for d in plan.doors if d.wall_side in ("N", "S")), "flat door on the lobby")
                self.assertTrue(plan.layout_state)                                   # Smart Re-Plan still works

    def test_04_honest_when_fewer_fit_and_front_core_otherwise(self):
        plan = _apt(4.0, "2BHK", "N")
        st = plan.engineering["stilt"]
        self.assertLess(st["fitted"], 3)
        self.assertTrue(any("fit under the building" in n for n in plan.design_notes))
        no_stilt = _apt(5.0, "3BHK", "N", comps={})                                   # no stilt: the core stays in front
        self.assertEqual(no_stilt.engineering["apartment"]["core_at"], "front")
        self.assertNotIn("grid_x", no_stilt.engineering)

    def test_05_walk_from_the_gate_to_the_lift(self):
        """In 3D the stilt keeps a path from the walking gate to the stair and lift past the parked cars."""
        plan = _apt(5.0, "2BHK", "N")
        sc = build_scene(plan)
        self.assertIsNone(sc.get("talk_error"))
        kinds = [s["kind"] for s in sc["tour"]]
        self.assertEqual(kinds[:4], ["start", "gate", "parking", "door"])         # the journey reached the core
        self.assertIn("terrace", kinds)
        self.assertEqual(len([c for c in sc["cars"] if c.get("stilt")]), 3)


class TestElevationAndPack(unittest.TestCase):
    def test_06_front_elevation(self):
        W, H = _plot(5.0)
        house = _gen(("house2", 5.0), W, H, "3BHK", True, "E", "Standard", num_floors=2, family_profile=FAM,
                     additional_components={"car_parking": {"enabled": True, "slots": 1}})
        sc = build_scene(house)
        svg = front_elevation_svg(house, sc)
        ET.fromstring(svg)                                                            # well-formed
        self.assertIn("FRONT ELEVATION", svg)
        for label in ("GL ±0.000", "TERRACE", "PARAPET", "FLOOR 1"):
            self.assertIn(label, svg)
        # the road-side view mirrors the plan: the walking gate is drawn at u = W − x
        gate = next(g for g in sc["gates"] if g["kind"] == "walk")
        u_mid = W - (gate["x0"] + gate["x1"]) / 2
        xs = [float(m) for m in re.findall(r'<text x="([\d.]+)"[^>]*>G\d', svg)]
        self.assertTrue(any(abs(x / 50 - 1.2 - u_mid) < 0.6 for x in xs), (u_mid, xs))
        # every front window is drawn on every floor
        yf = sc["footprint"][1] + sc["footprint"][3]
        front = [w for w in sc["windows"] if w["out"][1] > 0.5 and abs(w["a"][1] - yf) < 0.3]
        self.assertGreater(len(front), 0)
        self.assertGreaterEqual(svg.count('fill="#BFDBFE"'), len(front) * 2)
        blue = front_elevation_svg(house, sc, mode="blueprint")
        self.assertIn("#0A2342", blue)
        # a stilt building: the front columns of the parking floor
        apt = _apt(5.0, "3BHK", "N")
        s2 = front_elevation_svg(apt)
        self.assertIn("STILT PARKING", s2)
        self.assertIn("FLOOR 1", s2)

    def test_07_builder_and_bank_pack(self):
        from model_folder.builder_pack import build_pack
        from model_folder.cost_engine import CostEngine
        for plan in (_apt(5.0, "3BHK", "N"), _CACHE.get(("house2", 5.0)) or _apt(5.0, "2BHK", "E")):
            est = CostEngine.estimate_for_plan(plan, city_tier="Tier-2", quality_level="Standard (Medium)",
                                               num_floors=plan.num_floors)
            pdf = build_pack(plan, est, "Standard (Medium)")
            self.assertTrue(pdf.startswith(b"%PDF"))
            pages = len(re.findall(rb"/Type\s*/Page[^s]", pdf))
            self.assertGreaterEqual(pages, 7, pages)                 # cover, plan, elevation, schedules, cost, Vastu, notes
            self.assertGreater(len(pdf), 40_000)
            self.assertIn(b"svglib", open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "requirements.txt"),
                                          "rb").read())


if __name__ == "__main__":
    unittest.main()
