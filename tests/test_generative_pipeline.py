"""
Comprehensive Verification Suite: Generative Floor Plan System (Phase 1)
========================================================================
Tests:
1. NBC 2016 Room Specs & Indian Architecture Standards
2. Vastu Shastra Engine: 3x3 Mandala, Scoring, Taboos
3. Family Intent Analyzer: Household needs to BHK
4. V-HSP Layout Engine: 1BHK, 2BHK, 3BHK, 4BHK generation
5. All 4 Renderers: DXF, SVG, PDF, PNG
6. Cost Engine Integration with Material BOQ
7. Flask REST API: /generate-plan endpoint
"""

import os
import sys
import unittest
import numpy as np

# Ensure root directory is on sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from floorplan_generator.room_specs import (
    NBC_ROOM_SPECS, BHKConfig, get_bhk_config, recommend_bhk_for_plot
)
from floorplan_generator.vastu_engine import VastuEngine, VastuZone
from floorplan_generator.family_analyzer import FamilyAnalyzer, FamilyProfile
from floorplan_generator.layout_engine import LayoutEngine, GeneratedFloorPlan
from floorplan_renderer.dxf_renderer import DXFRenderer
from floorplan_renderer.svg_renderer import SVGRenderer
from floorplan_renderer.pdf_renderer import PDFRenderer
from floorplan_renderer.png_renderer import PNGRenderer
from model_folder.cost_engine import CostEngine
from model_folder.app import app


class TestGenerativePipeline(unittest.TestCase):
    """Full system verification for Phase 1 generative floor plan product."""

    def test_01_room_specs_nbc_compliance(self):
        """Verify NBC 2016 minimum standards are defined correctly."""
        self.assertIn("living_room", NBC_ROOM_SPECS)
        self.assertIn("kitchen", NBC_ROOM_SPECS)
        self.assertIn("master_bedroom", NBC_ROOM_SPECS)
        self.assertIn("bathroom", NBC_ROOM_SPECS)

        # NBC minimum habitable area >= 9.5 sqm
        living = NBC_ROOM_SPECS["living_room"]
        self.assertGreaterEqual(living.min_area, 9.5)
        self.assertGreaterEqual(living.min_width, 2.4)

        # Kitchen minimum area >= 5.0 sqm, width >= 1.8m
        kitchen = NBC_ROOM_SPECS["kitchen"]
        self.assertGreaterEqual(kitchen.min_area, 5.0)
        self.assertGreaterEqual(kitchen.min_width, 1.8)

        # Bathroom minimum area >= 1.8 sqm, width >= 1.2m
        bath = NBC_ROOM_SPECS["bathroom"]
        self.assertGreaterEqual(bath.min_area, 1.8)
        self.assertGreaterEqual(bath.min_width, 1.2)

    def test_02_bhk_configurations(self):
        """Verify all standard BHK configurations (1BHK-4BHK) load properly."""
        for bhk in ["1BHK", "2BHK", "3BHK", "4BHK"]:
            cfg = get_bhk_config(bhk)
            self.assertEqual(cfg.name, bhk)
            self.assertGreater(len(cfg.rooms), 3)
            self.assertGreater(cfg.min_plot_area, 0)
            self.assertGreater(cfg.typical_carpet_area, 0)

        # Plot area recommendations
        self.assertEqual(recommend_bhk_for_plot(50), "1BHK")
        self.assertEqual(recommend_bhk_for_plot(80), "2BHK")
        self.assertEqual(recommend_bhk_for_plot(130), "3BHK")
        self.assertEqual(recommend_bhk_for_plot(200), "4BHK")

    def test_03_vastu_engine(self):
        """Verify Vastu Purusha Mandala 3x3 grid, scoring, and taboos."""
        vastu = VastuEngine(plot_width=12.0, plot_height=15.0, north_direction="N")
        zones = vastu.get_all_zones()
        self.assertEqual(len(zones), 9)

        # Master bedroom optimal in SW
        self.assertEqual(vastu.get_optimal_zone("BEDROOM"), VastuZone.SW)
        # Kitchen optimal in SE (Agni)
        self.assertEqual(vastu.get_optimal_zone("KITCHEN"), VastuZone.SE)
        # Pooja optimal in NE (Ishan)
        self.assertEqual(vastu.get_optimal_zone("POOJA"), VastuZone.NE)

        # Strict taboo checks
        self.assertTrue(vastu.is_taboo("BATHROOM", "NE"))
        self.assertTrue(vastu.is_taboo("KITCHEN", "NE"))
        self.assertTrue(vastu.is_taboo("BEDROOM", "CENTER"))

        # Scoring
        good_placements = [
            ("BEDROOM", "SW"),
            ("KITCHEN", "SE"),
            ("POOJA", "NE"),
            ("BATHROOM", "NW"),
            ("LIVING", "N")
        ]
        score = vastu.compute_plan_score(good_placements)
        self.assertGreaterEqual(score, 90.0)

    def test_04_family_analyzer(self):
        """Verify family composition translates to appropriate BHK and reasoning."""
        analyzer = FamilyAnalyzer()

        # Couple with 2 kids
        p1 = FamilyProfile(adults=2, children=2, needs_pooja_room=True)
        res1 = analyzer.analyze(p1, plot_area=120)
        self.assertEqual(res1.recommended_bhk, "2BHK")
        self.assertIn("Pooja Room", res1.additional_rooms)

        # Large joint family: 4 adults, 2 kids, 2 elderly
        p2 = FamilyProfile(adults=4, children=2, elderly=2, needs_servant_quarter=True)
        res2 = analyzer.analyze(p2, plot_area=250)
        self.assertEqual(res2.recommended_bhk, "4BHK")
        self.assertGreaterEqual(res2.bedrooms_needed, 4)

    def test_05_layout_engine_all_bhks(self):
        """Verify layout engine generates valid geometry for 1BHK, 2BHK, 3BHK, 4BHK."""
        for bhk in ["1BHK", "2BHK", "3BHK", "4BHK"]:
            w, h = (10.0, 12.0) if bhk in ("1BHK", "2BHK") else (15.0, 18.0)
            engine = LayoutEngine(plot_width=w, plot_height=h, bhk_config=bhk, vastu_enabled=True)
            plan = engine.generate()

            self.assertGreater(len(plan.rooms), 3)
            self.assertGreater(len(plan.walls), 4)
            self.assertGreater(len(plan.doors), 0)
            self.assertGreater(len(plan.windows), 0)
            self.assertGreater(len(plan.dimensions), 0)
            self.assertGreater(plan.total_carpet_area, 0)
            self.assertGreater(plan.total_built_up_area, plan.total_carpet_area)
            self.assertGreater(plan.vastu_score, 0)

            # Ensure all rooms have valid non-negative dimensions and fit in plot
            for r in plan.rooms:
                self.assertGreater(r.w, 0)
                self.assertGreater(r.h, 0)
                self.assertGreaterEqual(r.x, 0)
                self.assertGreaterEqual(r.y, 0)
                self.assertLessEqual(r.right, w + 0.1)
                self.assertLessEqual(r.top, h + 0.1)

    def test_06_renderers(self):
        """Verify all 4 CAD renderers output valid files."""
        engine = LayoutEngine(12.0, 15.0, "2BHK", vastu_enabled=True)
        plan = engine.generate()

        test_dir = os.path.join(root_dir, "test_output", "unit_test")
        os.makedirs(test_dir, exist_ok=True)

        # SVG
        svg_r = SVGRenderer()
        svg_path = os.path.join(test_dir, "plan.svg")
        svg_r.render(plan, svg_path, mode="clean")
        self.assertTrue(os.path.exists(svg_path))
        self.assertGreater(os.path.getsize(svg_path), 500)

        # PNG & Mask
        png_r = PNGRenderer()
        png_path = os.path.join(test_dir, "plan.png")
        png_r.render(plan, png_path)
        self.assertTrue(os.path.exists(png_path))
        self.assertGreater(os.path.getsize(png_path), 1000)

        mask = png_r.render_wall_mask(plan, ppm=50)
        self.assertIsInstance(mask, np.ndarray)
        self.assertEqual(len(mask.shape), 2)

        # PDF
        pdf_r = PDFRenderer()
        pdf_path = os.path.join(test_dir, "plan.pdf")
        pdf_r.render(plan, pdf_path)
        self.assertTrue(os.path.exists(pdf_path))
        self.assertGreater(os.path.getsize(pdf_path), 1000)

        # DXF
        dxf_r = DXFRenderer()
        dxf_path = os.path.join(test_dir, "plan.dxf")
        dxf_r.render(plan, dxf_path)
        self.assertTrue(os.path.exists(dxf_path))
        self.assertGreater(os.path.getsize(dxf_path), 5000)

    def test_07_cost_engine_integration(self):
        """Verify CostEngine calculates estimates and material BOQ for generated plan."""
        engine = LayoutEngine(12.0, 15.0, "2BHK", vastu_enabled=True)
        plan = engine.generate()

        cost = CostEngine.estimate_cost(
            area_sqm=plan.total_carpet_area,
            city_tier="Tier-2 Urban",
            quality_level="Standard",
            num_floors=1
        )
        self.assertIn("total_cost", cost)
        self.assertGreater(cost["total_cost"], 500000)
        self.assertIn("materials", cost)
        self.assertIn("Cement", cost["materials"])
        self.assertIn("Steel", cost["materials"])
        self.assertIn("Bricks", cost["materials"])

    def test_08_flask_api_endpoint(self):
        """Verify Flask /generate-plan and /download-plan endpoints."""
        client = app.test_client()

        payload = {
            "plot_width": 12.0,
            "plot_height": 15.0,
            "adults": 2,
            "children": 2,
            "elderly": 0,
            "vastu_enabled": True,
            "orientation": "N",
            "quality_level": "Standard",
            "city_tier": "Tier-2 Urban",
            "num_floors": 1,
            "project_name": "API Test Villa"
        }

        response = client.post("/generate-plan", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertTrue(data.get("success"))
        self.assertIn("plan_id", data)
        self.assertIn("bhk_config", data)
        self.assertIn("vastu_score", data)
        self.assertIn("rooms", data)
        self.assertIn("cost_estimation", data)
        self.assertIn("download_urls", data)

        plan_id = data["plan_id"]
        # Test download endpoint for each format
        for fmt in ["dxf", "svg", "pdf", "png"]:
            dl_res = client.get(f"/download-plan/{fmt}/{plan_id}")
            self.assertEqual(dl_res.status_code, 200)
            self.assertGreater(len(dl_res.data), 100)


if __name__ == "__main__":
    unittest.main()
