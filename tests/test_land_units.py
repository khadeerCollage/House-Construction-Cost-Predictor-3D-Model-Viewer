"""
Test Suite for Indian Land Units & Shape Constraint Engine
===========================================================
Verifies:
  1. Cent conversion accuracy (Sq Ft, Sq M, Gaj, Guntha, Ground, Ankanam)
  2. Square plot geometry constraint solving
  3. Rectangular plot geometry constraint solving (aspect ratios & frontage)
  4. BHK configuration auto-recommendation from Cents
  5. End-to-end floor plan generation and CAD rendering with Cents
"""

import os
import sys
import unittest
import math

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from floorplan_generator.land_units import (
    LandUnitsEngine, LandParcel,
    SQFT_PER_CENT, SQM_PER_CENT, GAJ_PER_CENT, GUNTHA_PER_CENT, GROUND_PER_CENT
)
from floorplan_generator.layout_engine import LayoutEngine
from floorplan_renderer.svg_renderer import SVGRenderer


class TestLandUnits(unittest.TestCase):

    def test_01_unit_conversions(self):
        """Verify mathematical precision of Indian land unit conversions."""
        # 1 Cent = 435.60 sq ft
        self.assertAlmostEqual(LandUnitsEngine.cents_to_sqft(1.0), 435.60, places=2)
        # 1 Cent = 40.4686 sq m
        self.assertAlmostEqual(LandUnitsEngine.cents_to_sqm(1.0), 40.4686, places=3)
        # 3 Cents = 1306.8 sq ft
        self.assertAlmostEqual(LandUnitsEngine.cents_to_sqft(3.0), 1306.8, places=1)
        # Round trip
        self.assertAlmostEqual(LandUnitsEngine.sqft_to_cents(1306.8), 3.0, places=2)
        self.assertAlmostEqual(LandUnitsEngine.sqm_to_cents(LandUnitsEngine.cents_to_sqm(5.0)), 5.0, places=2)

    def test_02_square_plot_solver(self):
        """Verify Square (Chaturasra) plot constraint solver."""
        # 3 Cents = 121.4057 m² -> Side = sqrt(121.4057) = 11.018m
        parcel = LandUnitsEngine.solve_parcel(cents=3.0, shape="Square")
        self.assertEqual(parcel.shape, "Square")
        self.assertAlmostEqual(parcel.width_m, parcel.length_m, places=2)
        self.assertAlmostEqual(parcel.width_m, 11.02, delta=0.05)
        self.assertAlmostEqual(parcel.width_ft, parcel.length_ft, places=1)
        self.assertIn("Chaturasra", parcel.vastu_geometry_grade)
        # Total area check
        self.assertAlmostEqual(parcel.area_sqft, 3.0 * 435.6, delta=1.0)
        self.assertAlmostEqual(parcel.area_gaj, 3.0 * 48.4, delta=0.5)

    def test_03_rectangular_plot_ratio_solver(self):
        """Verify Rectangle (Ayatasra) plot constraint solver with aspect ratio."""
        # 3 Cents with 1:1.33 aspect ratio
        parcel = LandUnitsEngine.solve_parcel(cents=3.0, shape="Rectangle", aspect_ratio=1.33)
        self.assertEqual(parcel.shape, "Rectangle")
        self.assertGreater(parcel.length_m, parcel.width_m)
        self.assertAlmostEqual(parcel.aspect_ratio, 1.33, places=2)
        # Check area matches 3 cents
        computed_area_m2 = parcel.width_m * parcel.length_m
        expected_area_m2 = 3.0 * SQM_PER_CENT
        self.assertAlmostEqual(computed_area_m2, expected_area_m2, delta=1.0)

    def test_04_rectangular_plot_frontage_solver(self):
        """Verify Rectangle plot constraint solver with fixed road frontage."""
        # 3 Cents with 30 feet road frontage
        parcel = LandUnitsEngine.solve_parcel(cents=3.0, shape="Rectangle", frontage_ft=30.0)
        self.assertEqual(parcel.shape, "Rectangle")
        self.assertAlmostEqual(parcel.width_ft, 30.0, places=1)
        # Depth should be 1306.8 / 30 = 43.56 ft
        self.assertAlmostEqual(parcel.length_ft, 43.56, delta=0.2)
        # Check area matches 3 cents
        self.assertAlmostEqual(parcel.width_ft * parcel.length_ft, 1306.8, delta=1.0)

    def test_05_bhk_recommendation_from_cents(self):
        """Verify BHK recommendations scaled by land Cents."""
        self.assertEqual(LandUnitsEngine.recommend_bhk_for_cents(1.2), "1BHK")
        self.assertEqual(LandUnitsEngine.recommend_bhk_for_cents(2.5), "2BHK")
        self.assertEqual(LandUnitsEngine.recommend_bhk_for_cents(4.0), "3BHK")
        self.assertEqual(LandUnitsEngine.recommend_bhk_for_cents(6.0), "4BHK")

    def test_06_end_to_end_cents_generation(self):
        """Verify end-to-end plan generation from Cents constraint."""
        # Generate plan for 3 cents rectangular plot (30 ft frontage)
        parcel = LandUnitsEngine.solve_parcel(cents=3.0, shape="Rectangle", frontage_ft=30.0)
        
        engine = LayoutEngine(
            plot_width=parcel.width_m,
            plot_height=parcel.length_m,
            bhk_config=parcel.recommended_bhk,
            vastu_enabled=True,
            orientation='N',
            quality_level='Standard',
            project_name=f"{parcel.recommended_bhk} ({parcel.cents} Cents) Villa"
        )
        plan = engine.generate()

        self.assertIsNotNone(plan)
        self.assertGreater(len(plan.rooms), 0)
        self.assertGreater(len(plan.walls), 0)
        self.assertGreater(plan.total_carpet_area, 0)

        # SVG export with Cents in title block
        svg_content = SVGRenderer().render_to_string(plan)
        self.assertIn("Cents", svg_content)
        self.assertIn(f"{parcel.cents} Cents", svg_content)


if __name__ == '__main__':
    unittest.main()
