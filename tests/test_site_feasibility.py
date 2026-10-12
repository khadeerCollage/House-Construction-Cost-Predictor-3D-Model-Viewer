"""
Site feasibility (floorplan_generator/site_feasibility.py) — the Design form locks choices the
plot cannot take, based on the plot / BHK / floors chosen before them.

Covers: a 3-cent 2BHK gets two-wheeler parking but no car bay (with a reason and what would
unlock it), the generator honours the gated components on every facing, bigger plots unlock
the car, a multi-storey house parks the car on a stilt floor instead of squeezing the rooms,
BHK options follow the plot, the garden slider stops at what can really be laid out, and
apply_fit switches off / clamps what does not fit.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from floorplan_generator.layout_engine import LayoutEngine  # noqa: E402
from floorplan_generator.family_analyzer import FamilyProfile  # noqa: E402
from floorplan_generator.land_units import LandUnitsEngine  # noqa: E402
from floorplan_generator.site_feasibility import check_site, bhk_options, apply_fit  # noqa: E402

FAM = FamilyProfile(adults=2, children=2, needs_pooja_room=True)
HOUSE = "Independent House / Villa"


def _plot(cents, ratio=1.33):
    p = LandUnitsEngine.solve_parcel(cents=cents, shape="Rectangle", aspect_ratio=ratio)
    return p.width_m, p.length_m


class TestSiteFeasibility(unittest.TestCase):

    def test_01_three_cent_2bhk_bike_not_car(self):
        w, h = _plot(3.0)
        comps = {"car_parking": {"enabled": True, "slots": 1}, "bike_parking": {"enabled": True, "slots": 2}}
        for face in "NESW":
            fit = check_site(w, h, face, "2BHK", FAM, HOUSE, 1, "Standard", True, components=comps)
            car, bike = fit.amenities["car_parking"], fit.amenities["bike_parking"]
            self.assertEqual(car.status, "no", face)
            self.assertTrue(car.reason and car.unlock, face)
            self.assertIn("cents", car.unlock)
            self.assertEqual(bike.status, "ok", face)
            self.assertEqual(bike.max_value, 2, face)
            gated = apply_fit(comps, fit)
            self.assertFalse(gated["car_parking"]["enabled"])
            self.assertTrue(gated["bike_parking"]["enabled"])
            plan = LayoutEngine(w, h, "2BHK", True, face, family_profile=FAM, building_type=HOUSE,
                                additional_components=gated).generate()
            cats = {r.category for r in plan.rooms}
            self.assertIn("BIKE_PARKING", cats, f"{face}: bikes must fit a 3-cent front yard")
            self.assertNotIn("PARKING", cats)

    def test_02_bigger_plot_unlocks_the_car(self):
        w, h = _plot(6.0)
        fit = check_site(w, h, "N", "2BHK", FAM, HOUSE, 1, "Standard", True,
                         components={"car_parking": {"enabled": True, "slots": 1}})
        car = fit.amenities["car_parking"]
        self.assertEqual(car.status, "ok")
        self.assertGreaterEqual(car.max_value, 1)
        plan = LayoutEngine(w, h, "2BHK", True, "N", family_profile=FAM, building_type=HOUSE,
                            additional_components=apply_fit({"car_parking": {"enabled": True, "slots": 1}},
                                                            fit)).generate()
        self.assertIn("PARKING", {r.category for r in plan.rooms})

    def test_03_multi_storey_parks_on_stilt(self):
        w, h = _plot(3.0)
        # 1 car: an open bay would fit but squeeze the rooms → the checker asks for a stilt floor
        # 4 cars: no open bay at all → the planner chooses the stilt floor itself; slots are capped
        for slots in (1, 4):
            comps = {"car_parking": {"enabled": True, "slots": slots}}
            fit = check_site(w, h, "N", "2BHK", FAM, HOUSE, 2, "Standard", True, components=comps)
            car = fit.amenities["car_parking"]
            self.assertEqual(car.status, "ok")
            self.assertIn("stilt", car.reason)
            self.assertLessEqual(car.max_value, 3, "a ~72 m² stilt floor holds 2-3 cars")
            if slots == 1:
                self.assertEqual(car.hint, {"mode": "stilt"})
            gated = apply_fit(comps, fit)
            self.assertLessEqual(gated["car_parking"]["slots"], car.max_value)
            plan = LayoutEngine(w, h, "2BHK", True, "N", family_profile=FAM, num_floors=2, building_type=HOUSE,
                                additional_components=gated).generate()
            self.assertNotIn("PARKING", {r.category for r in plan.rooms}, slots)
            self.assertTrue(any("stilt" in n for n in plan.design_notes), slots)

    def test_04_bhk_options_follow_the_plot(self):
        w, h = _plot(3.0)
        opts = bhk_options(w, h, "N", FAM, HOUSE, 1, True)
        self.assertEqual(opts["1BHK"].status, "ok")
        self.assertEqual(opts["2BHK"].status, "ok")
        self.assertEqual(opts["4BHK"].status, "no")
        self.assertIn("cents", opts["4BHK"].unlock)
        w2, h2 = _plot(8.0)
        self.assertTrue(all(f.status == "ok" for f in bhk_options(w2, h2, "N", FAM, HOUSE, 1, True).values()))

    def test_05_garden_slider_stops_at_real_lawn(self):
        for cents in (3.0, 5.0, 8.0):
            w, h = _plot(cents)
            fit = check_site(w, h, "N", "2BHK", FAM, HOUSE, 1, "Standard", True,
                             components={"garden": {"enabled": True, "pct": 25}})
            g = fit.amenities["garden"]
            self.assertNotEqual(g.status, "no")
            if g.max_value > 5:     # every offered % must really be laid out (≥ 85 %)
                self.assertGreaterEqual(g.amount_m2, 0.85 * w * h * g.max_value / 100.0 - 1e-6, cents)

    def test_06_apply_fit_clamps_and_switches_off(self):
        w, h = _plot(3.0)
        comps = {"car_parking": {"enabled": True, "slots": 2}, "bike_parking": {"enabled": True, "slots": 10},
                 "swimming_pool": {"enabled": True, "size": "Large (5m × 10m)"},
                 "water_tank": {"enabled": True}}
        fit = check_site(w, h, "N", "2BHK", FAM, HOUSE, 1, "Standard", True, components=comps)
        out = apply_fit(comps, fit)
        self.assertEqual(out["car_parking"], {"enabled": False, "slots": 0})
        self.assertEqual(out["swimming_pool"]["enabled"], False)
        self.assertEqual(out["bike_parking"]["slots"], fit.amenities["bike_parking"].max_value)
        self.assertTrue(out["water_tank"]["enabled"])
        self.assertIn("do not fit", fit.summary())


if __name__ == "__main__":
    unittest.main()
