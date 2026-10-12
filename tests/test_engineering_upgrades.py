"""
Engineering & Vastu Upgrade Test Suite
=======================================
Locks in the behaviour of the upgraded generative engine:
  1. BHK honoured in Vastu mode (all bedrooms placed)
  2. Road-facing ↔ compass geometry and 32-pada positions
  3. Non-saturating Vastu score
  4. Vastu anchors, access, tiling for every road facing
  5. Vastu site rule (N/E setbacks ≥ S/W) and footprint inside the plot
  6. Lifestyle spaces become real rooms
  7. Site amenities in sensible Vastu zones, clear of the house
  8. Vertical circulation engineering (NBC risers/treads, terrace stair)
  9. Cost engine label normalisation and site-works costing
 10. Door openings cut into walls
 11. SVG drawn with the road side at the top
"""

import html
import os
import re
import sys
import unittest

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from floorplan_generator.layout_engine import LayoutEngine
from floorplan_generator.family_analyzer import FamilyAnalyzer, FamilyProfile
from floorplan_generator.land_units import LandUnitsEngine
from floorplan_generator.vastu_engine import (
    VastuEngine, plan_sides_for_facing, north_edge_for_facing, pada_for_point,
)
from floorplan_generator.space_program import design_staircase
from floorplan_renderer.svg_renderer import SVGRenderer
from model_folder.cost_engine import CostEngine

SITE = {"PARKING", "BIKE_PARKING", "GARDEN", "POOL", "WATER_TANK", "GENERATOR", "SECURITY"}


def house_rooms(plan):
    return [r for r in plan.rooms if r.category not in SITE and "External" not in r.name]


def zone_of(plan, r, scope="building"):
    return VastuEngine.for_plan(plan, scope).get_zone_for_position(r.cx, r.cy).value


class TestEngineeringUpgrades(unittest.TestCase):

    def test_01_bhk_honoured_in_vastu_mode(self):
        for bhk, (w, h) in {"1BHK": (9, 11), "2BHK": (11, 14), "3BHK": (14, 17), "4BHK": (16, 19)}.items():
            plan = LayoutEngine(w, h, bhk, vastu_enabled=True).generate()
            beds = [r for r in plan.rooms if r.category == "BEDROOM"]
            self.assertEqual(len(beds), int(bhk[0]), f"{bhk}: {[r.name for r in beds]}")

    def test_02_facing_geometry(self):
        self.assertEqual(plan_sides_for_facing("E"), {"top": "E", "bottom": "W", "left": "N", "right": "S"})
        self.assertEqual(north_edge_for_facing("N"), "N")
        self.assertEqual(north_edge_for_facing("E"), "W")
        self.assertEqual(north_edge_for_facing("S"), "S")
        self.assertEqual(north_edge_for_facing("W"), "E")
        # East-facing: the top-middle cell is the East zone, the top-left corner is NE
        eng = VastuEngine.for_facing(9, 9, "E")
        self.assertEqual(eng.get_zone_for_position(4.5, 8.0).value, "E")
        self.assertEqual(eng.get_zone_for_position(0.5, 8.5).value, "NE")
        # 32 padas: North side counted from the NW corner
        rect = (0, 0, 8, 8)
        self.assertEqual(pada_for_point(0.5, 8, rect, "N").pada_id, "N1")
        self.assertEqual(pada_for_point(3.5, 8, rect, "N").pada_id, "N4")
        self.assertEqual(pada_for_point(7.5, 8, rect, "N").pada_id, "N8")
        # East-facing road side (top) is the East facade, counted from NE (left)
        self.assertEqual(pada_for_point(2.5, 8, rect, "E").pada_id, "E3")

    def test_03_vastu_score_not_saturated(self):
        eng = VastuEngine(12, 15, "N")
        good = eng.compute_plan_score([("MASTER_BEDROOM", "SW"), ("KITCHEN", "SE"), ("POOJA", "NE")])
        bad = eng.compute_plan_score([("MASTER_BEDROOM", "NE"), ("KITCHEN", "SW"), ("POOJA", "SE")])
        meh = eng.compute_plan_score([("MASTER_BEDROOM", "SE"), ("KITCHEN", "SW"), ("POOJA", "W")])
        self.assertGreaterEqual(good, 99.0)
        self.assertLess(bad, 40.0)
        self.assertLess(meh, 80.0)

    def test_04_every_facing_is_vastu_sound_and_accessible(self):
        parcel = LandUnitsEngine.solve_parcel(cents=3.0, shape="Rectangle", aspect_ratio=1.33)
        fam = FamilyProfile(adults=2, children=2, needs_pooja_room=True)
        for facing in "NESW":
            plan = LayoutEngine(parcel.width_m, parcel.length_m, "2BHK", True, facing,
                                family_profile=fam).generate()
            rooms = house_rooms(plan)
            master = next(r for r in rooms if r.role == "MASTER_BEDROOM")
            kitchen = next(r for r in rooms if r.category == "KITCHEN")
            self.assertIn(zone_of(plan, master), ("SW", "S", "W"), facing)
            self.assertNotIn(zone_of(plan, kitchen), ("NE", "CENTER", "SW"), facing)
            self.assertGreaterEqual(plan.vastu_score, 80.0, facing)
            taboos = [row for row in plan.vastu_breakdown["rooms"] if row["taboo"]]
            self.assertEqual(taboos, [], facing)
            # every room has a door (main door counts for the living room)
            with_doors = {d.room_name for d in plan.doors}
            for r in rooms:
                if r.category in ("CORRIDOR", "DINING", "STAIRS") or r.name in ("Foyer", "Passage"):
                    continue
                self.assertIn(r.name, with_doors, f"{facing}: {r.name} has no door")
            # rooms tile the footprint without overlaps
            fx, fy, fw, fd = plan.building_footprint
            self.assertAlmostEqual(sum(r.w * r.h for r in rooms), fw * fd, delta=0.05)
            for i, a in enumerate(rooms):
                for b in rooms[i + 1:]:
                    self.assertFalse(a.overlaps(b), f"{facing}: {a.name} overlaps {b.name}")
            self.assertEqual(sum(1 for d in plan.doors if d.is_main_entrance), 1)

    def test_05_vastu_site_rule_and_footprint(self):
        plan = LayoutEngine(15.0, 20.0, "3BHK", True, "S").generate()
        sb = plan.setbacks
        self.assertGreaterEqual(sb["N"] + 1e-6, sb["S"] - 2.5)  # front (S) may carry bylaw depth
        self.assertGreaterEqual(sb["N"] + sb["E"], sb["W"])
        fx, fy, fw, fd = plan.building_footprint
        self.assertGreaterEqual(fx, 0)
        self.assertGreaterEqual(fy, 0)
        self.assertLessEqual(fx + fw, plan.plot_width + 1e-6)
        self.assertLessEqual(fy + fd, plan.plot_height + 1e-6)
        self.assertLess(plan.total_carpet_area, plan.total_built_up_area)
        self.assertLessEqual(plan.total_built_up_area, plan.plot_width * plan.plot_height)

    def test_06_lifestyle_spaces_are_real_rooms(self):
        fam = FamilyProfile(adults=2, children=1, needs_home_office=True, needs_pooja_room=True,
                            needs_store_room=True, needs_servant_quarter=True)
        plan = LayoutEngine(15.0, 20.0, "3BHK", True, "N", family_profile=fam).generate()
        cats = {r.category for r in plan.rooms}
        for c in ("STUDY", "POOJA", "STORE", "SERVANT"):
            self.assertIn(c, cats)
        # FamilyAnalyzer lists pooja even for small homes now
        res = FamilyAnalyzer().analyze(FamilyProfile(adults=2, needs_pooja_room=True), plot_area=60)
        self.assertIn("Pooja Room", res.additional_rooms)

    def test_07_site_amenities(self):
        comps = {
            'car_parking': {'enabled': True, 'slots': 1}, 'garden': {'enabled': True, 'pct': 10},
            'water_tank': {'enabled': True}, 'generator_room': {'enabled': True},
            'security_cabin': {'enabled': True}, 'bike_parking': {'enabled': True, 'slots': 2},
        }
        for facing in "NESW":
            plan = LayoutEngine(16.0, 22.0, "3BHK", True, facing, additional_components=comps).generate()
            fx, fy, fw, fd = plan.building_footprint
            site = [r for r in plan.rooms if r.category in SITE]
            self.assertTrue(any(r.category == "WATER_TANK" for r in site), facing)
            for r in site:
                # clear of the house envelope
                self.assertFalse(r.x < fx + fw and r.right > fx and r.y < fy + fd and r.top > fy,
                                 f"{facing}: {r.name} overlaps the house")
                self.assertGreaterEqual(r.x, -1e-6)
                self.assertLessEqual(r.right, plan.plot_width + 1e-6)
            for i, a in enumerate(site):
                for b in site[i + 1:]:
                    self.assertFalse(a.overlaps(b), f"{facing}: {a.name} overlaps {b.name}")
            sump = next(r for r in site if r.category == "WATER_TANK")
            self.assertIn(zone_of(plan, sump, "plot"), ("NE", "N", "E"), facing)
            gen = [r for r in site if r.category == "GENERATOR"]
            if gen:
                self.assertNotIn(zone_of(plan, gen[0], "plot"), ("NE", "N", "CENTER"), facing)

    def test_08_vertical_circulation(self):
        s = design_staircase("dog_leg", "Independent House")
        self.assertLessEqual(s.riser_height, 0.19)
        self.assertGreaterEqual(s.tread, 0.25)
        self.assertTrue(0.55 <= s.blondel <= 0.70)
        apt = design_staircase("dog_leg", "Apartment / Multi-Storey Building")
        self.assertGreaterEqual(apt.flight_width, 1.25)
        single = LayoutEngine(12, 16, "2BHK", True, "N", num_floors=1).generate()
        self.assertFalse(any(r.category == "STAIRS" and "External" not in r.name for r in single.rooms))
        g1 = LayoutEngine(12, 16, "2BHK", True, "N", num_floors=2).generate()
        stairs = [s for s in g1.staircases if "External" not in s.name]
        self.assertEqual(len(stairs), 1)
        self.assertLessEqual(stairs[0].riser_height_m, 0.19)

    def test_09_cost_engine_labels_and_site_works(self):
        base = CostEngine.estimate_cost(100, city_tier="Tier-2 (Urban)", quality_level="Standard (Medium)")
        prem = CostEngine.estimate_cost(100, city_tier="Tier-1 Metro", quality_level="Premium")
        lux = CostEngine.estimate_cost(100, city_tier="Tier-1 Metro", quality_level="Luxury")
        eco = CostEngine.estimate_cost(100, city_tier="Rural", quality_level="Economy")
        self.assertGreater(prem["total_cost"], 1.6 * base["total_cost"])
        self.assertGreater(lux["total_cost"], prem["total_cost"])
        self.assertLess(eco["total_cost"], base["total_cost"])
        comps = {'garden': {'enabled': True, 'pct': 10}, 'water_tank': {'enabled': True}}
        plan = LayoutEngine(16.0, 22.0, "3BHK", True, "N", additional_components=comps).generate()
        est = CostEngine.estimate_for_plan(plan, "Tier-2 Urban", "Standard", 1)
        self.assertIn("Site Works & Amenities", est["cost_breakdown"])
        self.assertGreater(est["total_cost"], est["building_cost"])

    def test_10_door_openings_cut_into_walls(self):
        plan = LayoutEngine(12.0, 15.0, "2BHK", True, "N").generate()
        main = next(d for d in plan.doors if d.is_main_entrance)
        mid = main.x + main.width / 2
        for w in plan.walls:
            if w.is_horizontal and abs(w.y1 - main.y) < 0.02:
                lo, hi = sorted((w.x1, w.x2))
                self.assertFalse(lo + 0.01 < mid < hi - 0.01, "a wall still blocks the main door")

    def test_11_svg_road_side_on_top(self):
        plan = LayoutEngine(12.0, 15.0, "2BHK", True, "N").generate()
        svg = SVGRenderer().render_to_string(plan, mode="clean")
        living = next(r for r in plan.rooms if r.category == "LIVING")
        master = next(r for r in plan.rooms if r.role == "MASTER_BEDROOM")
        self.assertGreater(living.cy, master.cy)  # plan: living towards the road (N, top)

        def label_y(name):
            m = re.search(r'<text[^>]*\by="([-\d.]+)"[^>]*>' + re.escape(html.escape(name.upper(), quote=False)) + r'</text>', svg)
            self.assertIsNotNone(m, name)
            return float(m.group(1))
        # SVG is y-down: the living room label must be drawn ABOVE the master bedroom
        self.assertLess(label_y(living.name), label_y(master.name))

    def test_12_cost_markets_and_regions(self):
        # every market/region prices a house, in its own currency, inside 2026 benchmark bands
        bands = {  # Standard spec, all-in project cost per market unit for a ~1,600 sqft home
            "India": (1800, 4500), "United States": (180, 450), "United Kingdom": (2200, 4500),
            "United Arab Emirates": (350, 900), "Canada": (280, 750), "Australia": (2600, 5500)}
        for country in CostEngine.markets():
            for region in CostEngine.regions(country):
                est = CostEngine.estimate_cost(150, bathrooms=2, country=country, region=region,
                                               quality_level="Standard")
                self.assertEqual(est["country"], country)
                self.assertEqual(est["region"], region)
                lo, hi = bands[country]
                self.assertTrue(lo <= est["cost_per_unit"] <= hi, (country, region, est["cost_per_unit"]))
                self.assertAlmostEqual(sum(est["cost_breakdown"].values()), est["total_cost"], delta=5)
        # location index ordering and legacy tier labels
        mum = CostEngine.estimate_cost(100, country="India", region="Mumbai / MMR (Metro)")
        rural = CostEngine.estimate_cost(100, country="India", region="Rural / Village")
        legacy = CostEngine.estimate_cost(100, city_tier="Tier-2 Urban")
        self.assertGreater(mum["total_cost"], 1.7 * rural["total_cost"])
        self.assertEqual(legacy["region"], "Other Tier-2 City (average)")
        self.assertTrue(legacy["display"]["total_compact"].endswith("Lakhs"))
        nyc = CostEngine.estimate_cost(100, country="US", region="New York City (NY)")
        hou = CostEngine.estimate_cost(100, country="US", region="Houston (TX)")
        self.assertGreater(nyc["total_cost"], hou["total_cost"])
        self.assertTrue(nyc["display"]["total"].startswith("$"))
        # plan pricing: site works + matrix covers every region, current region's Standard matches
        plan = LayoutEngine(16.0, 22.0, "3BHK", True, "N",
                            additional_components={'garden': {'enabled': True, 'pct': 10}}).generate()
        est = CostEngine.estimate_for_plan(plan, quality_level="Standard", country="United Kingdom",
                                           region="Greater London")
        self.assertIn("Site Works & Amenities", est["cost_breakdown"])
        matrix = CostEngine.region_matrix_for_plan(plan, "United Kingdom")
        self.assertEqual(len(matrix), len(CostEngine.regions("United Kingdom")))
        london = next(r for r in matrix if r["region"] == "Greater London")
        self.assertAlmostEqual(london["Standard"], est["total_cost"], delta=1)


if __name__ == "__main__":
    unittest.main()
