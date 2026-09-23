"""
Architectural-Grade SVG Renderer
=================================
Production-quality SVG renderer following ISO 7519 / IS 962 drafting standards.

Features:
  - 90-degree door swing arcs with door leaf rectangles and hinge points
  - Main entrance landing with 3 flat steps and "UP" directional arrow
  - Double-glazing window symbols with jamb end-caps and sill projection
  - Room dimension labels: Name + Width x Height + Carpet Area
  - 3-tier exterior dimension hierarchy with 45-degree architectural ticks
  - Enhanced north arrow compass rosette and title block
  - Blueprint (dark) and Clean (white) rendering modes

Reference Standards: ISO 128 (line weights), ISO 7519 (floor plans),
                     IS 962 (Indian drafting), AIA CAD Layer Standards
"""

import drawsvg as draw
import math
from floorplan_generator.layout_engine import (
    GeneratedFloorPlan, RoomRect, WallSegment,
    DoorPlacement, WindowPlacement, DimLine
)


# =============================================================================
# Constants
# =============================================================================
PPM = 50  # Pixels per meter (rendering scale)
WALL_STROKE_HEAVY = 2.5    # External / cut walls
WALL_STROKE_MEDIUM = 1.8   # Internal walls
DOOR_LEAF_STROKE = 1.5     # Door leaf rectangle
DOOR_ARC_STROKE = 0.8      # Door swing arc (thin)
WINDOW_FRAME_STROKE = 1.2  # Window jamb and sill
WINDOW_GLASS_STROKE = 1.0  # Glazing lines
DIM_LINE_STROKE = 0.8      # Dimension lines
DIM_TICK_STROKE = 1.5      # 45-degree tick marks
DOOR_LEAF_THICKNESS_M = 0.04   # 40mm door panel thickness
WINDOW_GLASS_GAP_M = 0.04     # 40mm gap between double glazing lines
ENTRANCE_LANDING_DEPTH_M = 1.2  # Landing depth
ENTRANCE_STEP_DEPTH_M = 0.3    # Each tread depth
ENTRANCE_NUM_STEPS = 3


class SVGRenderer:
    """Renders floor plans to architectural-grade vector SVG format."""

    def __init__(self):
        pass

    def render_to_string(self, plan: GeneratedFloorPlan, mode: str = 'clean') -> str:
        """Returns the rendered SVG as a string."""
        d = self._build_svg(plan, mode)
        return d.as_svg()

    def render(self, plan: GeneratedFloorPlan, output_path: str, mode: str = 'clean'):
        """Renders the floor plan and saves it to a file."""
        d = self._build_svg(plan, mode)
        d.save_svg(output_path)

    def _build_svg(self, plan: GeneratedFloorPlan, mode: str):
        pad = 3.0  # Extra padding for dimensions and title block
        width_m = plan.plot_width + pad * 2 + 5.0  # Extra for title block
        height_m = plan.plot_height + pad * 2

        width_px = width_m * PPM
        height_px = height_m * PPM

        d = draw.Drawing(width_px, height_px, origin=(-pad * PPM, -pad * PPM))

        # Color scheme based on mode
        if mode == 'blueprint':
            colors = {
                'bg': '#0A2342', 'wall_fill': '#1B3A5C', 'wall_stroke': '#00FFFF',
                'text': '#FFFFFF', 'text_dim': '#B0C4DE', 'door': '#00FF88',
                'door_arc': '#4A9E7F', 'window_glass': '#00BFFF', 'window_frame': '#00DDFF',
                'dim_line': '#6B8DAF', 'dim_text': '#8EB8E5', 'dim_tick': '#6B8DAF',
                'room_fill_alpha': 0.08, 'hatch': '#1E4D6E',
                'step_fill': '#0D2B45', 'step_stroke': '#00DDDD',
                'sill': '#00AACC', 'title_stroke': '#00CCCC', 'title_text': '#E0F0FF',
                'north_fill': '#00FFCC', 'north_stroke': '#00DDAA',
                'door_leaf_fill': '#0A2342',
            }
        else:
            colors = {
                'bg': '#FFFFFF', 'wall_fill': '#D1D5DB', 'wall_stroke': '#1F2937',
                'text': '#111827', 'text_dim': '#4B5563', 'door': '#059669',
                'door_arc': '#6B7280', 'window_glass': '#0284C7', 'window_frame': '#334155',
                'dim_line': '#6B7280', 'dim_text': '#374151', 'dim_tick': '#374151',
                'room_fill_alpha': 0.15, 'hatch': '#9CA3AF',
                'step_fill': '#F1F5F9', 'step_stroke': '#334155',
                'sill': '#64748B', 'title_stroke': '#1F2937', 'title_text': '#111827',
                'north_fill': '#1F2937', 'north_stroke': '#374151',
                'door_leaf_fill': '#FFFFFF',
            }

        # Background
        d.append(draw.Rectangle(-pad * PPM, -pad * PPM, width_px, height_px, fill=colors['bg']))

        # Wall hatch pattern
        hatch = draw.Pattern(width=10, height=10, patternUnits="userSpaceOnUse")
        hatch.append(draw.Line(0, 10, 10, 0, stroke=colors['hatch'], stroke_width=0.5))

        # Render layers in order (back to front)
        self._draw_room_fills(d, plan.rooms, colors)
        self._draw_walls(d, plan.walls, colors, hatch)
        self._draw_windows(d, plan.windows, plan.walls, colors)
        self._draw_doors(d, plan.doors, colors)
        self._draw_room_labels(d, plan.rooms, colors)
        self._draw_dimensions(d, plan.dimensions, colors)
        self._draw_north_arrow(d, plan, colors)
        self._draw_title_block(d, plan, colors)

        return d

    # =========================================================================
    # Room Fills
    # =========================================================================
    def _draw_room_fills(self, d, rooms, colors):
        """Draw room floor area fills with subtle color."""
        for r in rooms:
            if r.w <= 0 or r.h <= 0:
                continue
            d.append(draw.Rectangle(
                r.x * PPM, r.y * PPM, r.w * PPM, r.h * PPM,
                fill=r.color, fill_opacity=colors['room_fill_alpha'],
                stroke='none'
            ))

    # =========================================================================
    # Walls
    # =========================================================================
    def _draw_walls(self, d, walls, colors, hatch):
        """Draw walls with thickness, hatching, and proper line weights."""
        for w in walls:
            dx = w.x2 - w.x1
            dy = w.y2 - w.y1
            L = math.hypot(dx, dy)
            if L < 1e-5:
                continue

            nx = -dy / L * w.thickness / 2
            ny = dx / L * w.thickness / 2

            p1 = ((w.x1 + nx) * PPM, (w.y1 + ny) * PPM)
            p2 = ((w.x2 + nx) * PPM, (w.y2 + ny) * PPM)
            p3 = ((w.x2 - nx) * PPM, (w.y2 - ny) * PPM)
            p4 = ((w.x1 - nx) * PPM, (w.y1 - ny) * PPM)

            stroke_w = WALL_STROKE_HEAVY if w.is_external else WALL_STROKE_MEDIUM
            poly = draw.Lines(
                *p1, *p2, *p3, *p4, close=True,
                fill=hatch, stroke=colors['wall_stroke'], stroke_width=stroke_w
            )
            d.append(poly)

    # =========================================================================
    # Doors — 90-degree Swing Arc + Leaf Rectangle + Hinge Point
    # =========================================================================
    def _draw_doors(self, d, doors, colors):
        """Draw architectural door symbols with swing arcs and leaf panels."""
        for door in doors:
            hx = door.x * PPM
            hy = door.y * PPM
            r_px = door.width * PPM
            leaf_t = DOOR_LEAF_THICKNESS_M * PPM  # Leaf thickness in pixels

            # Compute closed edge, open edge, leaf rect, and arc sweep flag
            # based on wall_side and swing_direction
            closed_x, closed_y = hx, hy
            open_x, open_y = hx, hy
            leaf_rect = None  # (x, y, w, h)
            sweep = 1

            side = door.wall_side
            swing = door.swing_direction

            if side == 'N':
                # Door on north wall — swings inward (toward +Y in our coordinate system)
                if swing == 'RIGHT':
                    closed_x, closed_y = hx + r_px, hy
                    open_x, open_y = hx, hy + r_px
                    sweep = 1
                    # Leaf rectangle: vertical along +Y from hinge
                    leaf_rect = (hx - leaf_t / 2, hy, leaf_t, r_px)
                else:  # LEFT
                    closed_x, closed_y = hx - r_px, hy
                    open_x, open_y = hx, hy + r_px
                    sweep = 0
                    leaf_rect = (hx - leaf_t / 2, hy, leaf_t, r_px)

            elif side == 'S':
                # Door on south wall — swings inward (toward -Y)
                if swing == 'RIGHT':
                    closed_x, closed_y = hx + r_px, hy
                    open_x, open_y = hx, hy - r_px
                    sweep = 0
                    leaf_rect = (hx - leaf_t / 2, hy - r_px, leaf_t, r_px)
                else:
                    closed_x, closed_y = hx - r_px, hy
                    open_x, open_y = hx, hy - r_px
                    sweep = 1
                    leaf_rect = (hx - leaf_t / 2, hy - r_px, leaf_t, r_px)

            elif side == 'E':
                # Door on east wall — swings inward (toward -X)
                if swing == 'RIGHT':
                    closed_x, closed_y = hx, hy + r_px
                    open_x, open_y = hx - r_px, hy
                    sweep = 1
                    leaf_rect = (hx - r_px, hy - leaf_t / 2, r_px, leaf_t)
                else:
                    closed_x, closed_y = hx, hy - r_px
                    open_x, open_y = hx - r_px, hy
                    sweep = 0
                    leaf_rect = (hx - r_px, hy - leaf_t / 2, r_px, leaf_t)

            elif side == 'W':
                # Door on west wall — swings inward (toward +X)
                if swing == 'RIGHT':
                    closed_x, closed_y = hx, hy - r_px
                    open_x, open_y = hx + r_px, hy
                    sweep = 0
                    leaf_rect = (hx, hy - leaf_t / 2, r_px, leaf_t)
                else:
                    closed_x, closed_y = hx, hy + r_px
                    open_x, open_y = hx + r_px, hy
                    sweep = 1
                    leaf_rect = (hx, hy - leaf_t / 2, r_px, leaf_t)

            # 1. Draw swing arc (thin, quarter circle)
            path = draw.Path(stroke=colors['door_arc'], stroke_width=DOOR_ARC_STROKE, fill='none')
            path.M(closed_x, closed_y)
            path.A(r_px, r_px, 0, 0, sweep, open_x, open_y)
            d.append(path)

            # 2. Draw door leaf rectangle (the panel in open position)
            if leaf_rect:
                lx, ly, lw, lh = leaf_rect
                d.append(draw.Rectangle(
                    lx, ly, lw, lh,
                    fill=colors['door_leaf_fill'], stroke=colors['door'],
                    stroke_width=DOOR_LEAF_STROKE
                ))

            # 3. Draw hinge point (small filled circle)
            d.append(draw.Circle(hx, hy, 2.5, fill=colors['door']))

            # 4. Draw line from hinge to closed position (threshold line)
            d.append(draw.Line(
                hx, hy, closed_x, closed_y,
                stroke=colors['door'], stroke_width=DOOR_LEAF_STROKE
            ))

            # 5. Draw entrance steps for main entrance
            if door.is_main_entrance:
                self._draw_entrance_steps(d, door, colors)

    def _draw_entrance_steps(self, d, door, colors):
        """Draw landing + 3 flat steps at the main entrance door."""
        hx = door.x * PPM
        hy = door.y * PPM
        door_w = door.width * PPM
        landing_depth = ENTRANCE_LANDING_DEPTH_M * PPM
        step_depth = ENTRANCE_STEP_DEPTH_M * PPM
        side_extend = 0.3 * PPM  # 300mm overhang on each side

        side = door.wall_side

        if side == 'S':
            # Steps go downward (toward -Y) from the south wall
            x_left = hx - door_w / 2 - side_extend
            x_right = hx + door_w / 2 + side_extend
            total_w = x_right - x_left

            # Landing rectangle
            d.append(draw.Rectangle(
                x_left, hy - landing_depth, total_w, landing_depth,
                fill=colors['step_fill'], stroke=colors['step_stroke'],
                stroke_width=1.5
            ))
            # Landing label
            d.append(draw.Text(
                "LANDING", 8, (x_left + x_right) / 2, hy - landing_depth / 2,
                fill=colors['text_dim'], text_anchor='middle',
                font_weight='normal', font_style='italic'
            ))

            # 3 tread lines
            for i in range(ENTRANCE_NUM_STEPS):
                step_y = hy - landing_depth - (i + 1) * step_depth
                d.append(draw.Line(
                    x_left, step_y, x_right, step_y,
                    stroke=colors['step_stroke'], stroke_width=1.5
                ))
            # Side boundary lines
            total_steps_h = ENTRANCE_NUM_STEPS * step_depth
            d.append(draw.Line(
                x_left, hy - landing_depth, x_left, hy - landing_depth - total_steps_h,
                stroke=colors['step_stroke'], stroke_width=1.5
            ))
            d.append(draw.Line(
                x_right, hy - landing_depth, x_right, hy - landing_depth - total_steps_h,
                stroke=colors['step_stroke'], stroke_width=1.5
            ))

            # UP arrow (pointing from grade toward entrance)
            arrow_x = (x_left + x_right) / 2
            arrow_start_y = hy - landing_depth - total_steps_h - 10
            arrow_end_y = hy - landing_depth + 5
            d.append(draw.Line(
                arrow_x, arrow_start_y, arrow_x, arrow_end_y,
                stroke=colors['step_stroke'], stroke_width=1.0
            ))
            # Arrowhead
            d.append(draw.Lines(
                arrow_x - 4, arrow_end_y - 8,
                arrow_x, arrow_end_y,
                arrow_x + 4, arrow_end_y - 8,
                close=False, fill='none',
                stroke=colors['step_stroke'], stroke_width=1.2
            ))
            d.append(draw.Text(
                "UP", 9, arrow_x + 8, (arrow_start_y + arrow_end_y) / 2,
                fill=colors['text_dim'], font_weight='bold'
            ))

        elif side == 'N':
            x_left = hx - door_w / 2 - side_extend
            x_right = hx + door_w / 2 + side_extend
            total_w = x_right - x_left

            d.append(draw.Rectangle(
                x_left, hy, total_w, landing_depth,
                fill=colors['step_fill'], stroke=colors['step_stroke'],
                stroke_width=1.5
            ))
            d.append(draw.Text(
                "LANDING", 8, (x_left + x_right) / 2, hy + landing_depth / 2,
                fill=colors['text_dim'], text_anchor='middle',
                font_weight='normal', font_style='italic'
            ))

            for i in range(ENTRANCE_NUM_STEPS):
                step_y = hy + landing_depth + (i + 1) * step_depth
                d.append(draw.Line(
                    x_left, step_y, x_right, step_y,
                    stroke=colors['step_stroke'], stroke_width=1.5
                ))

            total_steps_h = ENTRANCE_NUM_STEPS * step_depth
            d.append(draw.Line(
                x_left, hy + landing_depth, x_left, hy + landing_depth + total_steps_h,
                stroke=colors['step_stroke'], stroke_width=1.5
            ))
            d.append(draw.Line(
                x_right, hy + landing_depth, x_right, hy + landing_depth + total_steps_h,
                stroke=colors['step_stroke'], stroke_width=1.5
            ))

            arrow_x = (x_left + x_right) / 2
            arrow_start_y = hy + landing_depth + total_steps_h + 10
            arrow_end_y = hy + landing_depth - 5
            d.append(draw.Line(
                arrow_x, arrow_start_y, arrow_x, arrow_end_y,
                stroke=colors['step_stroke'], stroke_width=1.0
            ))
            d.append(draw.Lines(
                arrow_x - 4, arrow_end_y + 8,
                arrow_x, arrow_end_y,
                arrow_x + 4, arrow_end_y + 8,
                close=False, fill='none',
                stroke=colors['step_stroke'], stroke_width=1.2
            ))
            d.append(draw.Text(
                "UP", 9, arrow_x + 8, (arrow_start_y + arrow_end_y) / 2,
                fill=colors['text_dim'], font_weight='bold'
            ))

        elif side == 'W':
            y_top = hy - door_w / 2 - side_extend
            y_bottom = hy + door_w / 2 + side_extend
            total_h = y_bottom - y_top

            d.append(draw.Rectangle(
                hx - landing_depth, y_top, landing_depth, total_h,
                fill=colors['step_fill'], stroke=colors['step_stroke'],
                stroke_width=1.5
            ))

            for i in range(ENTRANCE_NUM_STEPS):
                step_x = hx - landing_depth - (i + 1) * step_depth
                d.append(draw.Line(
                    step_x, y_top, step_x, y_bottom,
                    stroke=colors['step_stroke'], stroke_width=1.5
                ))

            total_steps_w = ENTRANCE_NUM_STEPS * step_depth
            d.append(draw.Line(
                hx - landing_depth, y_top, hx - landing_depth - total_steps_w, y_top,
                stroke=colors['step_stroke'], stroke_width=1.5
            ))
            d.append(draw.Line(
                hx - landing_depth, y_bottom, hx - landing_depth - total_steps_w, y_bottom,
                stroke=colors['step_stroke'], stroke_width=1.5
            ))

        elif side == 'E':
            y_top = hy - door_w / 2 - side_extend
            y_bottom = hy + door_w / 2 + side_extend
            total_h = y_bottom - y_top

            d.append(draw.Rectangle(
                hx, y_top, landing_depth, total_h,
                fill=colors['step_fill'], stroke=colors['step_stroke'],
                stroke_width=1.5
            ))

            for i in range(ENTRANCE_NUM_STEPS):
                step_x = hx + landing_depth + (i + 1) * step_depth
                d.append(draw.Line(
                    step_x, y_top, step_x, y_bottom,
                    stroke=colors['step_stroke'], stroke_width=1.5
                ))

            total_steps_w = ENTRANCE_NUM_STEPS * step_depth
            d.append(draw.Line(
                hx + landing_depth, y_top, hx + landing_depth + total_steps_w, y_top,
                stroke=colors['step_stroke'], stroke_width=1.5
            ))
            d.append(draw.Line(
                hx + landing_depth, y_bottom, hx + landing_depth + total_steps_w, y_bottom,
                stroke=colors['step_stroke'], stroke_width=1.5
            ))

    # =========================================================================
    # Windows — Double-Glazing with Jamb End-Caps and Sill
    # =========================================================================
    def _draw_windows(self, d, windows, walls, colors):
        """Draw architectural window symbols with double glazing and sill."""
        for win in windows:
            wx = win.x * PPM
            wy = win.y * PPM
            w_half = (win.width / 2) * PPM
            h_half = (win.height / 2) * PPM

            # Find wall thickness for this window (approximate from nearby walls)
            wall_t = 0.23 * PPM  # Default 230mm

            side = win.wall_side
            glass_gap = WINDOW_GLASS_GAP_M * PPM / 2  # Half gap from center

            if side in ('N', 'S'):
                # Horizontal window in N or S wall
                x1 = wx - w_half
                x2 = wx + w_half

                # Jamb end-caps (small filled rectangles at each end)
                jamb_w = 0.06 * PPM  # 60mm jamb profile
                jamb_h = wall_t * 0.4
                d.append(draw.Rectangle(
                    x1 - jamb_w / 2, wy - jamb_h / 2, jamb_w, jamb_h,
                    fill=colors['wall_stroke'], stroke='none'
                ))
                d.append(draw.Rectangle(
                    x2 - jamb_w / 2, wy - jamb_h / 2, jamb_w, jamb_h,
                    fill=colors['wall_stroke'], stroke='none'
                ))

                # Double glazing lines (two parallel horizontal lines)
                d.append(draw.Line(
                    x1, wy - glass_gap, x2, wy - glass_gap,
                    stroke=colors['window_glass'], stroke_width=WINDOW_GLASS_STROKE
                ))
                d.append(draw.Line(
                    x1, wy + glass_gap, x2, wy + glass_gap,
                    stroke=colors['window_glass'], stroke_width=WINDOW_GLASS_STROKE
                ))

                # Exterior sill projection line (slightly outside wall)
                sill_offset = wall_t * 0.3
                if side == 'N':
                    d.append(draw.Line(
                        x1 - 0.03 * PPM, wy + sill_offset,
                        x2 + 0.03 * PPM, wy + sill_offset,
                        stroke=colors['sill'], stroke_width=WINDOW_FRAME_STROKE
                    ))
                else:
                    d.append(draw.Line(
                        x1 - 0.03 * PPM, wy - sill_offset,
                        x2 + 0.03 * PPM, wy - sill_offset,
                        stroke=colors['sill'], stroke_width=WINDOW_FRAME_STROKE
                    ))

                # Interior sill line (on opposite side)
                if side == 'N':
                    d.append(draw.Line(
                        x1, wy - sill_offset * 0.6,
                        x2, wy - sill_offset * 0.6,
                        stroke=colors['sill'], stroke_width=0.8
                    ))
                else:
                    d.append(draw.Line(
                        x1, wy + sill_offset * 0.6,
                        x2, wy + sill_offset * 0.6,
                        stroke=colors['sill'], stroke_width=0.8
                    ))

            else:
                # Vertical window in E or W wall
                y1 = wy - h_half
                y2 = wy + h_half

                # Jamb end-caps
                jamb_w = wall_t * 0.4
                jamb_h = 0.06 * PPM
                d.append(draw.Rectangle(
                    wx - jamb_w / 2, y1 - jamb_h / 2, jamb_w, jamb_h,
                    fill=colors['wall_stroke'], stroke='none'
                ))
                d.append(draw.Rectangle(
                    wx - jamb_w / 2, y2 - jamb_h / 2, jamb_w, jamb_h,
                    fill=colors['wall_stroke'], stroke='none'
                ))

                # Double glazing lines (two parallel vertical lines)
                d.append(draw.Line(
                    wx - glass_gap, y1, wx - glass_gap, y2,
                    stroke=colors['window_glass'], stroke_width=WINDOW_GLASS_STROKE
                ))
                d.append(draw.Line(
                    wx + glass_gap, y1, wx + glass_gap, y2,
                    stroke=colors['window_glass'], stroke_width=WINDOW_GLASS_STROKE
                ))

                # Exterior sill projection
                sill_offset = wall_t * 0.3
                if side == 'E':
                    d.append(draw.Line(
                        wx + sill_offset, y1 - 0.03 * PPM,
                        wx + sill_offset, y2 + 0.03 * PPM,
                        stroke=colors['sill'], stroke_width=WINDOW_FRAME_STROKE
                    ))
                else:
                    d.append(draw.Line(
                        wx - sill_offset, y1 - 0.03 * PPM,
                        wx - sill_offset, y2 + 0.03 * PPM,
                        stroke=colors['sill'], stroke_width=WINDOW_FRAME_STROKE
                    ))

                # Interior sill line
                if side == 'E':
                    d.append(draw.Line(
                        wx - sill_offset * 0.6, y1,
                        wx - sill_offset * 0.6, y2,
                        stroke=colors['sill'], stroke_width=0.8
                    ))
                else:
                    d.append(draw.Line(
                        wx + sill_offset * 0.6, y1,
                        wx + sill_offset * 0.6, y2,
                        stroke=colors['sill'], stroke_width=0.8
                    ))

    # =========================================================================
    # Room Labels — Name + Width x Height + Area
    # =========================================================================
    def _draw_room_labels(self, d, rooms, colors):
        """Draw room name, dimensions, and area at centroid with dynamic font sizing."""
        for r in rooms:
            if r.w <= 0 or r.h <= 0:
                continue

            cx = r.cx * PPM
            cy = r.cy * PPM
            area = r.area

            # Dynamic font sizing based on room area
            min_dim = min(r.w, r.h)
            font_title = max(10, min(16, 8 + 1.2 * math.sqrt(area)))
            font_dim = max(8, font_title - 3)
            font_area = max(7, font_title - 4)

            # For very small rooms, abbreviate
            name = r.name
            if min_dim < 1.5 or area < 3.0:
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

            line_spacing = font_title + 4

            # Line 1: Room name (bold, uppercase)
            d.append(draw.Text(
                name.upper(), font_title,
                cx, cy - line_spacing * 0.8,
                fill=colors['text'], text_anchor='middle',
                font_weight='bold',
                font_family='Arial, Helvetica, sans-serif'
            ))

            # Line 2: Width x Height dimensions
            dim_text = f"{r.w:.2f} x {r.h:.2f} m"
            d.append(draw.Text(
                dim_text, font_dim,
                cx, cy + 2,
                fill=colors['text_dim'], text_anchor='middle',
                font_family='Arial, Helvetica, sans-serif'
            ))

            # Line 3: Carpet area
            area_text = f"{area:.1f} m2"
            d.append(draw.Text(
                area_text, font_area,
                cx, cy + line_spacing * 0.7,
                fill=colors['text_dim'], text_anchor='middle',
                font_style='italic',
                font_family='Arial, Helvetica, sans-serif'
            ))

    # =========================================================================
    # Dimension Lines — 3-Tier Hierarchy with 45-degree Ticks
    # =========================================================================
    def _draw_dimensions(self, d, dimensions, colors):
        """Draw architectural dimension lines with 45-degree tick marks."""
        for dim in dimensions:
            x1 = dim.x1 * PPM
            y1 = dim.y1 * PPM
            x2 = dim.x2 * PPM
            y2 = dim.y2 * PPM

            is_horizontal = abs(y2 - y1) < abs(x2 - x1)

            # Offset from wall based on tier
            tier_offsets = {1: 0.8, 2: 1.6, 3: 2.4}
            offset_m = tier_offsets.get(dim.tier, dim.offset)
            offset_px = offset_m * PPM

            if is_horizontal:
                dy = offset_px
                dx = 0
            else:
                dx = offset_px
                dy = 0

            dim_x1 = x1 + dx
            dim_y1 = y1 + dy
            dim_x2 = x2 + dx
            dim_y2 = y2 + dy

            # Extension (witness) lines from measured point to dimension line
            # Start 2mm (1px) from wall, overshoot 3mm (1.5px) past dim line
            d.append(draw.Line(
                x1, y1 + (1 if is_horizontal else 0),
                dim_x1, dim_y1 + (1.5 if is_horizontal else 0),
                stroke=colors['dim_line'], stroke_width=0.5
            ))
            d.append(draw.Line(
                x2, y2 + (1 if is_horizontal else 0),
                dim_x2, dim_y2 + (1.5 if is_horizontal else 0),
                stroke=colors['dim_line'], stroke_width=0.5
            ))

            # Dimension line
            d.append(draw.Line(
                dim_x1, dim_y1, dim_x2, dim_y2,
                stroke=colors['dim_line'], stroke_width=DIM_LINE_STROKE
            ))

            # 45-degree architectural tick marks at each end
            tick_len = 4  # pixels
            for tx, ty in [(dim_x1, dim_y1), (dim_x2, dim_y2)]:
                d.append(draw.Line(
                    tx - tick_len, ty + tick_len,
                    tx + tick_len, ty - tick_len,
                    stroke=colors['dim_tick'], stroke_width=DIM_TICK_STROKE
                ))

            # Dimension text (centered above the line)
            mid_x = (dim_x1 + dim_x2) / 2
            mid_y = (dim_y1 + dim_y2) / 2

            # Font size based on tier
            font_size = 10 if dim.tier == 3 else 9

            text_offset_y = -6 if is_horizontal else 0
            text_offset_x = -6 if not is_horizontal else 0

            d.append(draw.Text(
                dim.label, font_size,
                mid_x + text_offset_x, mid_y + text_offset_y,
                fill=colors['dim_text'], text_anchor='middle',
                font_weight='bold' if dim.tier == 3 else 'normal',
                font_family='Arial, Helvetica, sans-serif'
            ))

    # =========================================================================
    # North Arrow — Compass Rosette
    # =========================================================================
    def _draw_north_arrow(self, d, plan, colors):
        """Draw a compass north arrow with N/E/S/W labels."""
        # Position in top-right area
        cx = (plan.plot_width + 1.5) * PPM
        cy = (plan.plot_height - 1.5) * PPM
        r = 20  # Arrow radius in pixels

        # Circle outline
        d.append(draw.Circle(cx, cy, r + 8,
                             stroke=colors['north_stroke'], fill='none', stroke_width=1.5))

        # Solid north half (filled triangle)
        d.append(draw.Lines(
            cx, cy - r,        # North tip
            cx - r * 0.35, cy, # Left base
            cx, cy,            # Center
            close=True,
            fill=colors['north_fill'], stroke=colors['north_stroke'], stroke_width=1
        ))
        # Outline east half
        d.append(draw.Lines(
            cx, cy - r,
            cx + r * 0.35, cy,
            cx, cy,
            close=True,
            fill='none', stroke=colors['north_stroke'], stroke_width=1
        ))

        # N label
        d.append(draw.Text(
            "N", 12, cx, cy - r - 12,
            fill=colors['north_fill'], text_anchor='middle',
            font_weight='bold', font_family='Arial'
        ))
        # E, S, W labels (smaller)
        d.append(draw.Text("E", 8, cx + r + 12, cy + 3,
                           fill=colors['text_dim'], text_anchor='middle', font_family='Arial'))
        d.append(draw.Text("S", 8, cx, cy + r + 14,
                           fill=colors['text_dim'], text_anchor='middle', font_family='Arial'))
        d.append(draw.Text("W", 8, cx - r - 12, cy + 3,
                           fill=colors['text_dim'], text_anchor='middle', font_family='Arial'))

    # =========================================================================
    # Title Block
    # =========================================================================
    def _draw_title_block(self, d, plan, colors):
        """Draw ISO 7200 style title block."""
        tb_x = (plan.plot_width + 0.8) * PPM
        tb_y = 0
        tb_w = 4.0 * PPM  # 200px
        tb_h = 3.0 * PPM  # 150px

        # Border
        d.append(draw.Rectangle(
            tb_x, tb_y, tb_w, tb_h,
            stroke=colors['title_stroke'], fill='none', stroke_width=2
        ))

        # Horizontal dividers
        row_h = tb_h / 6
        for i in range(1, 6):
            d.append(draw.Line(
                tb_x, tb_y + i * row_h,
                tb_x + tb_w, tb_y + i * row_h,
                stroke=colors['title_stroke'], stroke_width=0.8
            ))

        # Title block content
        cents = (plan.total_built_up_area * 10.764) / 435.6
        labels = [
            ("PROJECT", plan.project_name),
            ("TYPE", plan.bhk_config),
            ("AREA", f"{plan.total_built_up_area:.1f} m2 ({cents:.2f} Cents)"),
            ("QUALITY", plan.quality_level),
            ("VASTU", f"{plan.vastu_score:.0f}/100"),
            ("SCALE", "1:50 | NBC 2016"),
        ]

        for i, (label, value) in enumerate(labels):
            y_pos = tb_y + i * row_h + row_h * 0.65
            d.append(draw.Text(
                f"{label}: {value}", 9,
                tb_x + 8, y_pos,
                fill=colors['title_text'],
                font_family='Arial, Helvetica, sans-serif'
            ))
