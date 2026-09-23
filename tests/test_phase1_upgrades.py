"""
Phase 1 Upgrades Comprehensive Test Suite
==========================================
Verifies:
  1. Architectural SVG rendering (door swing arcs, windows, dimensions, entrance steps)
  2. Architectural PNG rendering and wall mask generation
  3. Architectural PDF generation
  4. AutoCAD DXF generation with dimensions and steps
  5. Ultra-Premium Vastu Shastra audit & analysis engine
  6. Interactive Floor Plan Editor JSON conversion and parsing
"""

import os
import sys
import tempfile
import unittest

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from floorplan_generator.layout_engine import LayoutEngine, GeneratedFloorPlan
from floorplan_renderer.svg_renderer import SVGRenderer
from floorplan_renderer.png_renderer import PNGRenderer
from floorplan_renderer.pdf_renderer import PDFRenderer
from floorplan_renderer.dxf_renderer import DXFRenderer
from floorplan_generator.vastu_analyzer import VastuAnalyzer
from model_folder.floorplan_editor import FloorPlanEditor


class TestPhase1Upgrades(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Generate a standard 2BHK plan
        engine = LayoutEngine(
            plot_width=12.0,
            plot_height=15.0,
            bhk_config='2BHK',
            vastu_enabled=True,
            orientation='N',
            quality_level='Standard',
            project_name='Test Villa'
        )
        cls.plan = engine.generate()

    def test_01_svg_architectural_features(self):
        """Verify SVG contains proper architectural symbols."""
        renderer = SVGRenderer()
        svg_clean = renderer.render_to_string(self.plan, mode='clean')
        svg_blueprint = renderer.render_to_string(self.plan, mode='blueprint')

        self.assertIn("<svg", svg_clean)
        self.assertIn("</svg>", svg_clean)

        # 1. Door swing arc command (A rx ry x-axis-rotation large-arc sweep x y)
        self.assertTrue(" A" in svg_clean or " a" in svg_clean, "Door swing arc path command missing in SVG")

        # 2. Main entrance landing / steps
        self.assertTrue("LANDING" in svg_clean or "UP" in svg_clean, "Main entrance landing/steps missing in SVG")

        # 3. Room dimensions (e.g. ' x ' and ' m2')
        self.assertIn(" x ", svg_clean, "Room width x length dimensions missing in SVG")
        self.assertIn(" m2", svg_clean, "Room area label missing in SVG")

        # 4. Blueprint mode styles
        self.assertIn("#0A2342", svg_blueprint)
        self.assertIn("#00FFFF", svg_blueprint)

    def test_02_png_rendering(self):
        """Verify PNG rendering and wall mask generation."""
        renderer = PNGRenderer()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            png_path = tf.name

        try:
            renderer.render(self.plan, png_path)
            self.assertTrue(os.path.exists(png_path))
            self.assertGreater(os.path.getsize(png_path), 5000)

            # Test wall mask for 3D pipeline
            mask = renderer.render_wall_mask(self.plan, ppm=50)
            self.assertIsNotNone(mask)
            self.assertEqual(len(mask.shape), 2)
            self.assertGreater(mask.shape[0], 0)
            self.assertGreater(mask.shape[1], 0)
        finally:
            if os.path.exists(png_path):
                os.remove(png_path)

    def test_03_pdf_rendering(self):
        """Verify PDF generation produces valid A3 blueprint."""
        renderer = PDFRenderer()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
            pdf_path = tf.name

        try:
            renderer.render(self.plan, pdf_path)
            self.assertTrue(os.path.exists(pdf_path))
            self.assertGreater(os.path.getsize(pdf_path), 2000)
        finally:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)

    def test_04_dxf_rendering(self):
        """Verify DXF generation with layers and room text."""
        renderer = DXFRenderer()
        with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as tf:
            dxf_path = tf.name

        try:
            renderer.render(self.plan, dxf_path)
            self.assertTrue(os.path.exists(dxf_path))
            self.assertGreater(os.path.getsize(dxf_path), 5000)

            with open(dxf_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            self.assertIn("A-WALL-FULL", content)
            self.assertIn("A-DOOR-FULL", content)
            self.assertIn("A-GLAZ-FULL", content)
        finally:
            if os.path.exists(dxf_path):
                os.remove(dxf_path)

    def test_05_vastu_analyzer_engine(self):
        """Verify the comprehensive Vastu analyzer outputs."""
        analyzer = VastuAnalyzer(self.plan)

        # Zone-by-zone audit
        zones = analyzer.get_zone_analysis()
        self.assertEqual(len(zones), 9)
        zone_keys = [z["zone_key"] for z in zones]
        for expected in ["NE", "E", "SE", "S", "SW", "W", "NW", "N", "CENTER"]:
            self.assertIn(expected, zone_keys)

        # 32-Pada Entrance gate audit
        entrance = analyzer.audit_entrance()
        self.assertIn("pada_id", entrance)
        self.assertIn("deity", entrance)
        self.assertIn("rating", entrance)

        # Elemental balance
        balance = analyzer.get_elemental_balance()
        self.assertEqual(len(balance), 5)

        # Defects & remedies
        defects = analyzer.get_defects_and_remedies()
        self.assertIsInstance(defects, list)

        # House recommendation
        rec = analyzer.get_house_recommendation()
        self.assertIn("house_archetype", rec)
        self.assertIn("vastu_rating_grade", rec)
        self.assertIn("room_swap_recommendations", rec)

        # Formal Certificate
        cert = analyzer.generate_certificate_text()
        self.assertIn("VASTU SHASTRA COMPLIANCE AUDIT CERTIFICATE", cert)
        self.assertIn("ENTRANCE GATE AUDIT", cert)

        # Mandala SVG
        mandala_svg = analyzer.render_mandala_svg()
        self.assertIn("<svg", mandala_svg)
        self.assertIn("VASTU PURUSHA MANDALA", mandala_svg)

    def test_06_floorplan_editor_json(self):
        """Verify Fabric.js JSON conversion and room extraction in editor."""
        fabric_data = FloorPlanEditor.plan_to_fabric_json(self.plan, 800, 600)
        self.assertIn("objects", fabric_data)
        self.assertGreater(len(fabric_data["objects"]), 0)

        # Check blueprint CAD elements presence
        types = [obj.get("type") for obj in fabric_data["objects"]]
        self.assertIn("rect", types)
        self.assertIn("line", types)
        self.assertIn("text", types)
        self.assertIn("circle", types)

        # Check room label text includes dimensions and area
        texts = [obj.get("text", "") for obj in fabric_data["objects"] if obj.get("type") == "text"]
        has_room_label = any("m²" in t and "×" in t for t in texts)
        self.assertTrue(has_room_label, "Room label text must contain meters dimensions and m² area")

        # Check scale and compass indicators
        has_scale = any("SCALE: 1:50" in t for t in texts)
        self.assertTrue(has_scale, "Scale indicator must be present on blueprint canvas")
        has_compass = any("N" in t for t in texts)
        self.assertTrue(has_compass, "Compass rosette must be present on blueprint canvas")

        # Extract edited rooms back
        extracted = FloorPlanEditor.extract_edited_rooms(fabric_data, 600)
        self.assertGreater(len(extracted), 0)
        first_room = extracted[0]
        self.assertIn("name", first_room)
        self.assertIn("w", first_room)
        self.assertIn("h", first_room)
        self.assertIn("area", first_room)
        self.assertGreater(first_room["area"], 0)


if __name__ == '__main__':
    unittest.main()
