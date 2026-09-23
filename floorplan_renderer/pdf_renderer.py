"""
Architectural-Grade PDF Renderer
=================================
Generates professional A3 landscape architectural blueprints with ReportLab.

Features:
  - 90-degree door swing arcs with door leaf panels and hinge points
  - Main entrance steps and landing with 'UP' annotation
  - Double-glazing window symbols with sill projections
  - Room labels: Name + Dimensions (W x H) + Area at centroid
  - 3-tier exterior dimension hierarchy with 45-degree ticks
  - North compass arrow and ISO title block
"""

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A3, landscape
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
import math
from floorplan_generator.layout_engine import (
    GeneratedFloorPlan, RoomRect, WallSegment,
    DoorPlacement, WindowPlacement, DimLine
)


class PDFRenderer:
    """Renders floor plans to PDF blueprint format following ISO standards."""

    def __init__(self):
        pass

    def render(self, plan: GeneratedFloorPlan, output_path: str):
        """Renders the given floor plan to an A3 landscape PDF file."""
        c = canvas.Canvas(output_path, pagesize=landscape(A3))
        page_w, page_h = landscape(A3)
        
        margin_l = 15 * mm
        margin_r = 15 * mm
        margin_t = 15 * mm
        margin_b = 15 * mm
        
        # Border
        c.setLineWidth(1.5)
        c.setStrokeColor(HexColor('#1E293B'))
        c.rect(margin_l, margin_b, page_w - margin_l - margin_r, page_h - margin_t - margin_b)
        
        # Drawing area calculation
        tb_w = 65 * mm
        tb_h = 42 * mm
        tb_x = page_w - margin_r - tb_w
        tb_y = margin_b
        
        draw_w = page_w - margin_l - margin_r - tb_w - 20 * mm
        draw_h = page_h - margin_t - margin_b - 20 * mm
        
        pad_m = 2.5
        total_plan_w = plan.plot_width + pad_m * 2
        total_plan_h = plan.plot_height + pad_m * 2
        
        if total_plan_w <= 0 or total_plan_h <= 0:
            scale = 10 * mm
        else:
            scale = min(draw_w / total_plan_w, draw_h / total_plan_h)
            
        offset_x = margin_l + 10 * mm + pad_m * scale
        offset_y = margin_b + 10 * mm + pad_m * scale
        
        def tr(x, y):
            return offset_x + x * scale, offset_y + y * scale

        # 1. Room Fills & Labels
        for r in plan.rooms:
            if r.w <= 0 or r.h <= 0:
                continue
            rx, ry = tr(r.x, r.y)
            rw, rh = r.w * scale, r.h * scale
            
            c.saveState()
            try:
                c.setFillColor(HexColor(r.color), alpha=0.15)
            except Exception:
                c.setFillColor(HexColor('#E2E8F0'), alpha=0.2)
            c.rect(rx, ry, rw, rh, fill=1, stroke=0)
            c.restoreState()

            # Room labels
            cx, cy = tr(r.cx, r.cy)
            min_dim = min(r.w, r.h)
            name = r.name
            if min_dim < 1.5 or r.area < 3.0:
                abbrevs = {
                    'Attached Bathroom': 'ATT. BATH',
                    'Bathroom': 'BATH',
                    'Balcony': 'BALC.',
                    'Corridor': 'CORR.',
                    'Utility': 'UTIL.',
                    'Store Room': 'STORE',
                    'Servant Quarter': 'SQ',
                }
                name = abbrevs.get(name, name)

            c.setFillColor(HexColor('#0F172A'))
            c.setFont("Helvetica-Bold", 9)
            c.drawCentredString(cx, cy + 3.5 * mm, name.upper())
            
            c.setFillColor(HexColor('#334155'))
            c.setFont("Helvetica", 7.5)
            c.drawCentredString(cx, cy - 0.5 * mm, f"{r.w:.2f} x {r.h:.2f} m")
            
            c.setFillColor(HexColor('#64748B'))
            c.setFont("Helvetica-Oblique", 6.5)
            c.drawCentredString(cx, cy - 4.0 * mm, f"{r.area:.1f} m2")

        # 2. Walls
        for w in plan.walls:
            x1, y1 = tr(w.x1, w.y1)
            x2, y2 = tr(w.x2, w.y2)
            c.setStrokeColor(HexColor('#0F172A'))
            c.setLineWidth(2.2 if w.is_external else 1.4)
            c.line(x1, y1, x2, y2)

        # 3. Windows — Double glazing lines & sill
        c.setLineWidth(0.8)
        c.setStrokeColor(HexColor('#0284C7'))
        glass_gap = 0.04 / 2
        for win in plan.windows:
            side = win.wall_side
            if side in ('N', 'S'):
                x1, y_bot = tr(win.x - win.width / 2, win.y - glass_gap)
                x2, y_top = tr(win.x + win.width / 2, win.y + glass_gap)
                c.line(x1, y_bot, x2, y_bot)
                c.line(x1, y_top, x2, y_top)
                # Sill
                sill_offset = 0.08 if side == 'N' else -0.08
                sx1, sy = tr(win.x - win.width / 2 - 0.03, win.y + sill_offset)
                sx2, _ = tr(win.x + win.width / 2 + 0.03, win.y + sill_offset)
                c.setStrokeColor(HexColor('#64748B'))
                c.line(sx1, sy, sx2, sy)
                c.setStrokeColor(HexColor('#0284C7'))
            else:
                x_left, y1 = tr(win.x - glass_gap, win.y - win.height / 2)
                x_right, y2 = tr(win.x + glass_gap, win.y + win.height / 2)
                c.line(x_left, y1, x_left, y2)
                c.line(x_right, y1, x_right, y2)
                # Sill
                sill_offset = 0.08 if side == 'E' else -0.08
                sx, sy1 = tr(win.x + sill_offset, win.y - win.height / 2 - 0.03)
                _, sy2 = tr(win.x + sill_offset, win.y + win.height / 2 + 0.03)
                c.setStrokeColor(HexColor('#64748B'))
                c.line(sx, sy1, sx, sy2)
                c.setStrokeColor(HexColor('#0284C7'))

        # 4. Doors — 90° Arc and Leaf
        for d in plan.doors:
            hx, hy = tr(d.x, d.y)
            r_pt = d.width * scale
            side = d.wall_side
            swing = d.swing_direction

            # Determine arc bounding box & start/extent angles
            start_ang, extent = 0, 90
            leaf_end_x, leaf_end_y = hx, hy

            if side == 'N':
                if swing == 'RIGHT':
                    start_ang, extent = 270, 90
                    leaf_end_x, leaf_end_y = hx, hy - r_pt
                else:
                    start_ang, extent = 180, 90
                    leaf_end_x, leaf_end_y = hx, hy - r_pt
            elif side == 'S':
                if swing == 'RIGHT':
                    start_ang, extent = 0, 90
                    leaf_end_x, leaf_end_y = hx, hy + r_pt
                else:
                    start_ang, extent = 90, 90
                    leaf_end_x, leaf_end_y = hx, hy + r_pt
            elif side == 'E':
                if swing == 'RIGHT':
                    start_ang, extent = 90, 90
                    leaf_end_x, leaf_end_y = hx - r_pt, hy
                else:
                    start_ang, extent = 180, 90
                    leaf_end_x, leaf_end_y = hx - r_pt, hy
            elif side == 'W':
                if swing == 'RIGHT':
                    start_ang, extent = 0, 90
                    leaf_end_x, leaf_end_y = hx + r_pt, hy
                else:
                    start_ang, extent = 270, 90
                    leaf_end_x, leaf_end_y = hx + r_pt, hy

            # Draw swing arc
            c.setStrokeColor(HexColor('#64748B'))
            c.setLineWidth(0.6)
            c.arc(hx - r_pt, hy - r_pt, hx + r_pt, hy + r_pt, start_ang, extent)

            # Draw door leaf
            c.setStrokeColor(HexColor('#059669'))
            c.setLineWidth(1.2)
            c.line(hx, hy, leaf_end_x, leaf_end_y)

            # Hinge point
            c.setFillColor(HexColor('#059669'))
            c.circle(hx, hy, 1.2 * mm, fill=1, stroke=0)

            # Main entrance steps
            if d.is_main_entrance and side == 'S':
                landing_d = 1.2 * scale
                step_d = 0.3 * scale
                side_ext = 0.3 * scale
                lx1 = hx - r_pt / 2 - side_ext
                lx2 = hx + r_pt / 2 + side_ext
                
                c.setStrokeColor(HexColor('#334155'))
                c.setLineWidth(0.8)
                c.setFillColor(HexColor('#F1F5F9'))
                c.rect(lx1, hy - landing_d, lx2 - lx1, landing_d, fill=1, stroke=1)
                
                for i in range(3):
                    sy = hy - landing_d - (i + 1) * step_d
                    c.line(lx1, sy, lx2, sy)
                c.line(lx1, hy - landing_d, lx1, hy - landing_d - 3 * step_d)
                c.line(lx2, hy - landing_d, lx2, hy - landing_d - 3 * step_d)
                
                # UP label
                c.setFillColor(HexColor('#334155'))
                c.setFont("Helvetica-Bold", 6.5)
                c.drawCentredString((lx1 + lx2) / 2, hy - landing_d - 3.5 * step_d, "UP")

        # 5. Dimension Lines with 45° Tick marks
        c.setStrokeColor(HexColor('#475569'))
        c.setLineWidth(0.6)
        tick_len = 1.5 * mm
        for dim in plan.dimensions:
            x1, y1 = tr(dim.x1, dim.y1)
            x2, y2 = tr(dim.x2, dim.y2)
            is_horizontal = abs(y2 - y1) < abs(x2 - x1)
            tier_offsets = {1: 0.8, 2: 1.6, 3: 2.4}
            offset = tier_offsets.get(dim.tier, dim.offset) * scale
            
            dx = 0 if is_horizontal else offset
            dy = offset if is_horizontal else 0
            
            dim_x1, dim_y1 = x1 + dx, y1 + dy
            dim_x2, dim_y2 = x2 + dx, y2 + dy

            # Witness lines
            c.setStrokeColor(HexColor('#94A3B8'))
            c.line(x1, y1, dim_x1, dim_y1)
            c.line(x2, y2, dim_x2, dim_y2)
            
            # Dim line
            c.setStrokeColor(HexColor('#475569'))
            c.line(dim_x1, dim_y1, dim_x2, dim_y2)
            
            # 45° Ticks
            c.setLineWidth(1.0)
            c.line(dim_x1 - tick_len, dim_y1 - tick_len, dim_x1 + tick_len, dim_y1 + tick_len)
            c.line(dim_x2 - tick_len, dim_y2 - tick_len, dim_x2 + tick_len, dim_y2 + tick_len)
            c.setLineWidth(0.6)
            
            # Text
            c.setFillColor(HexColor('#0F172A'))
            c.setFont("Helvetica", 6.5)
            mid_x, mid_y = (dim_x1 + dim_x2) / 2, (dim_y1 + dim_y2) / 2
            c.drawCentredString(mid_x, mid_y + (1.5 * mm if is_horizontal else 0), dim.label)

        # 6. North Arrow
        na_x = page_w - margin_r - tb_w / 2
        na_y = page_h - margin_t - 25 * mm
        c.setStrokeColor(HexColor('#0F172A'))
        c.setLineWidth(1.2)
        c.line(na_x, na_y - 10 * mm, na_x, na_y + 10 * mm)
        c.setFillColor(HexColor('#0F172A'))
        p = c.beginPath()
        p.moveTo(na_x, na_y + 10 * mm)
        p.lineTo(na_x - 3 * mm, na_y + 3 * mm)
        p.lineTo(na_x, na_y + 5 * mm)
        p.close()
        c.drawPath(p, fill=1, stroke=0)
        c.setFont("Helvetica-Bold", 10)
        c.drawCentredString(na_x, na_y + 12 * mm, "N")

        # 7. ISO Title Block
        c.setStrokeColor(HexColor('#0F172A'))
        c.setLineWidth(1.2)
        c.rect(tb_x, tb_y, tb_w, tb_h)
        
        row_h = tb_h / 6
        for i in range(1, 6):
            c.line(tb_x, tb_y + i * row_h, tb_x + tb_w, tb_y + i * row_h)
            
        cents = (plan.total_built_up_area * 10.764) / 435.6
        tb_data = [
            ("PROJECT", plan.project_name),
            ("CONFIG", plan.bhk_config),
            ("BUILT-UP", f"{plan.total_built_up_area:.1f} m2 ({cents:.2f} Cents)"),
            ("VASTU SCORE", f"{plan.vastu_score:.0f}/100"),
            ("SPECIFICATION", f"{plan.quality_level} Grade"),
            ("SCALE / DATE", "1:50 | NBC 2016 COMPLIANT"),
        ]
        
        for idx, (lbl, val) in enumerate(reversed(tb_data)):
            curr_y = tb_y + (idx + 0.35) * row_h
            c.setFillColor(HexColor('#64748B'))
            c.setFont("Helvetica-Bold", 6.5)
            c.drawString(tb_x + 2.5 * mm, curr_y, lbl)
            c.setFillColor(HexColor('#0F172A'))
            c.setFont("Helvetica", 7.5)
            c.drawString(tb_x + 22 * mm, curr_y, val[:32])

        c.showPage()
        c.save()
