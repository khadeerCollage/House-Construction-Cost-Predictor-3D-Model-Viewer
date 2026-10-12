"""
House report (floorplan_generator/house_report.py, floorplan_generator/takeoff.py,
CostEngine.boq_for_plan) — the four result tabs, measured from the drawing.

Covers: the Vastu ledger reproduces plan.vastu_score exactly, the room schedule's inside sizes
match the carpet areas and flag real NBC shortfalls, the Vastu checks and the plan-on-mandala
drawing, the take-off lands inside Indian thumb-rule bands, the priced stages add up to the
estimate's construction cost (one cost formula), a foundation sized for a future floor costs
more, and the house health card.
"""

import os
import sys
import unittest
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from floorplan_generator.layout_engine import LayoutEngine  # noqa: E402
from floorplan_generator.family_analyzer import FamilyProfile  # noqa: E402
from floorplan_generator import house_report as hr  # noqa: E402
from floorplan_generator.takeoff import material_takeoff  # noqa: E402
from model_folder.cost_engine import CostEngine  # noqa: E402

FAM = FamilyProfile(adults=2, children=2, needs_pooja_room=True)
HOUSE = "Independent House / Villa"
_PLANS = {}


def _plan(w, h, bhk, face, floors=1, comps=None):
    key = (w, h, bhk, face, floors, repr(comps))
    if key not in _PLANS:
        _PLANS[key] = LayoutEngine(w, h, bhk, True, face, family_profile=FAM, num_floors=floors,
                                   building_type=HOUSE, additional_components=comps or {}).generate()
    return _PLANS[key]


CASES = [(9.55, 12.71, "2BHK", "N", {"bike_parking": {"enabled": True, "slots": 2}}),
         (12.33, 16.40, "3BHK", "E", {"car_parking": {"enabled": True, "slots": 1}, "water_tank": {"enabled": True}}),
         (11.03, 14.67, "2BHK", "S", None),
         (10.0, 15.0, "3BHK", "W", None)]          # last one: compact fallback layout


class TestHouseReport(unittest.TestCase):

    def test_01_vastu_ledger_reproduces_the_score(self):
        for w, h, bhk, face, comps in CASES:
            p = _plan(w, h, bhk, face, 1, comps)
            led = hr.vastu_ledger(p)
            self.assertAlmostEqual(led["score"], p.vastu_score, delta=0.15, msg=f"{bhk} {face}")
            self.assertTrue(all(i["lost"] >= 0 for i in led["items"]))
            self.assertEqual(led["items"], sorted(led["items"], key=lambda i: -i["lost"]))

    def test_02_room_schedule_inside_sizes_and_nbc(self):
        p = _plan(9.55, 12.71, "2BHK", "N", 1, CASES[0][4])
        rs = hr.room_schedule(p)
        for r in rs["rows"]:
            if r["group"] != "site":
                self.assertAlmostEqual(r["w"] * r["h"], r["area"], delta=0.06, msg=r["name"])
            self.assertIn(r["nbc"]["status"], ("ok", "low", "na"))
        names = {r["name"]: r for r in rs["rows"]}
        cb = names["Common Bathroom"]
        self.assertEqual(cb["nbc"]["status"], "low", "2.69 m² is under NBC's 2.8 m² for bath + WC")
        self.assertIn("2.8", cb["nbc"]["text"])
        self.assertEqual(names["Master Bedroom"]["nbc"]["status"], "ok")
        self.assertEqual(hr.ft_in(3.45), "11'4\"")
        self.assertGreater(rs["totals"]["efficiency"], 80)
        self.assertTrue(rs["groups"])

    def test_03_vastu_checks_and_drawing(self):
        for w, h, bhk, face, comps in CASES:
            p = _plan(w, h, bhk, face, 1, comps)
            vc = hr.vastu_checks(p)
            rules = [c["rule"] for c in vc["checks"]]
            self.assertTrue(any("Main door" in r for r in rules))
            self.assertTrue(any("Brahmasthan" in r for r in rules))
            self.assertTrue(all(c["status"] in ("pass", "warn", "fail") for c in vc["checks"]))
            self.assertTrue(vc["site_only"])
            ET.fromstring(hr.vastu_plan_svg(p))          # well-formed SVG
        bad = hr.vastu_checks(_plan(10.0, 15.0, "3BHK", "W"))
        self.assertGreaterEqual(bad["counts"]["fail"], 1, "toilets in the NE must be flagged")

    def test_04_takeoff_in_thumb_rule_bands(self):
        for w, h, bhk in ((9.55, 12.71, "2BHK"), (12.33, 16.40, "3BHK")):
            p = _plan(w, h, bhk, "N")
            t = material_takeoff(p)
            sqft = t["basis"]["built_up_sqft"]
            T = t["totals"]
            self.assertTrue(0.35 <= T["cement_bags"] / sqft <= 0.55, (bhk, T["cement_bags"] / sqft))
            self.assertTrue(2.8 <= T["steel_kg"] / sqft <= 4.8, (bhk, T["steel_kg"] / sqft))
            self.assertTrue(1.2 <= T["sand_cft"] / sqft <= 2.0, (bhk, T["sand_cft"] / sqft))
            self.assertTrue(all(l["how"] for l in t["lines"]))
        p = _plan(9.55, 12.71, "2BHK", "N")
        now = sum(l["qty"] for l in material_takeoff(p)["lines"] if l["item"] == "RCC footings (M20)")
        future = sum(l["qty"] for l in material_takeoff(p, design_floors=2)["lines"] if l["item"] == "RCC footings (M20)")
        self.assertGreater(future, now * 1.2, "footings sized for G+1 are bigger")

    def test_05_boq_stages_add_up_to_the_estimate(self):
        p = _plan(9.55, 12.71, "2BHK", "N", 1, CASES[0][4])
        for q in ("Economy", "Standard", "Luxury"):
            est = CostEngine.estimate_for_plan(p, quality_level=q, country="India")
            boq = CostEngine.boq_for_plan(p, est, q)
            self.assertTrue(boq["available"])
            self.assertAlmostEqual(sum(s["amount"] for s in boq["stages"]), est["construction_cost"], delta=5)
            self.assertLess(boq["materials_total"], boq["building_cost"])
            self.assertTrue(0.40 <= boq["materials_total"] / boq["building_cost"] <= 0.75, q)
            self.assertEqual([s["key"] for s in boq["stages"][:3]], ["foundation", "plinth", "frame"])
        us = CostEngine.estimate_for_plan(p, country="United States", region="National Average")
        self.assertFalse(CostEngine.boq_for_plan(p, us)["available"])

    def test_06_health_card(self):
        for w, h, bhk, face, comps in CASES:
            p = _plan(w, h, bhk, face, 1, comps)
            hc = hr.health_card(p)
            self.assertEqual(len(hc["dims"]), 6)
            self.assertTrue(0 <= hc["overall"] <= 100)
            self.assertTrue(all(d["evidence"] for d in hc["dims"]))
        good = hr.health_card(_plan(12.33, 16.40, "3BHK", "E", 1, CASES[1][4]))
        poor = hr.health_card(_plan(10.0, 15.0, "3BHK", "W"))
        self.assertGreater(good["overall"], poor["overall"])


if __name__ == "__main__":
    unittest.main()
