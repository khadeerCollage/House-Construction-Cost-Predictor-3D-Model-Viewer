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
import copy
import dataclasses
import math
from types import SimpleNamespace
from floorplan_generator.layout_engine import (
    GeneratedFloorPlan, RoomRect, WallSegment,
    DoorPlacement, WindowPlacement, DimLine
)
from floorplan_renderer._common import north_angle, dim_offset, fit_label
from floorplan_renderer.drafting import build_drafting


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

    @staticmethod
    def _to_screen(plan: GeneratedFloorPlan) -> GeneratedFloorPlan:
        """
        Mirror the y-up plan into SVG's y-down screen space so the road/entrance
        side (plan top) is drawn at the top of the sheet. N/S labels of windows
        are swapped so sills stay on the outside; E-wall door swings are swapped
        so hinges stay at the same physical end.
        """
        H = plan.plot_height
        rep = dataclasses.replace
        p = copy.copy(plan)
        p.rooms = [rep(r, y=round(H - r.y - r.h, 4)) for r in plan.rooms]
        p.walls = [rep(w, y1=round(H - w.y1, 4), y2=round(H - w.y2, 4)) for w in plan.walls]
        swap_lr = {"LEFT": "RIGHT", "RIGHT": "LEFT"}
        p.doors = [rep(d, y=round(H - d.y, 4),
                       swing_direction=swap_lr.get(d.swing_direction, d.swing_direction)
                       if d.wall_side == "E" else d.swing_direction)
                   for d in plan.doors]
        swap_ns = {"N": "S", "S": "N"}
        p.windows = [rep(w, y=round(H - w.y, 4), wall_side=swap_ns.get(w.wall_side, w.wall_side))
                     for w in plan.windows]
        p.dimensions = [rep(dm, y1=round(H - dm.y1, 4), y2=round(H - dm.y2, 4),
                            offset=-dm.offset if abs(dm.y2 - dm.y1) < abs(dm.x2 - dm.x1) else dm.offset)
                        for dm in plan.dimensions]
        p.staircases = [rep(s, y=round(H - s.y - s.h, 4)) for s in getattr(plan, "staircases", [])]
        return p

    def _build_svg(self, plan: GeneratedFloorPlan, mode: str):
        orig = plan
        plan = self._to_screen(plan)
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
                'furn': '#6FA8DC', 'furn_fill': '#0D2A4A', 'furn_detail': '#4F86B8',
                'fixture': '#BFE9FF', 'fixture_fill': '#0F3157', 'tile': '#173F66',
                'grid': '#3E6E9C', 'column': '#00E5FF', 'plate': '#0A2342',
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
                'furn': '#6B7280', 'furn_fill': '#F9FAFB', 'furn_detail': '#9CA3AF',
                'fixture': '#334155', 'fixture_fill': '#FFFFFF', 'tile': '#E5E7EB',
                'grid': '#9CA3AF', 'column': '#111827', 'plate': '#FFFFFF',
            }

        # Background
        d.append(draw.Rectangle(-pad * PPM, -pad * PPM, width_px, height_px, fill=colors['bg']))

        # Wall hatch pattern
        hatch = draw.Pattern(width=10, height=10, patternUnits="userSpaceOnUse")
        hatch.append(draw.Line(0, 10, 10, 0, stroke=colors['hatch'], stroke_width=0.5))

        # Drafting layer (furniture, poché, grid, schedule) — computed in plan coordinates
        try:
            layer = build_drafting(orig, label_size=self._label_box)
        except Exception:
            layer = None
        H = orig.plot_height
        S = lambda x, y: (x * PPM, (H - y) * PPM)  # noqa: E731 — plan (y-up) → screen
        poche = bool(layer and layer.walls)

        # Render layers in order (back to front)
        self._draw_room_fills(d, plan.rooms, colors)
        if layer:
            self._draw_road(d, orig, colors)
            self._draw_grid_lines(d, layer, S, colors)
            self._draw_prims(d, layer.tiles, S, colors)
        self._draw_amenities(d, plan.rooms, colors)
        if layer:
            self._draw_prims(d, layer.furniture, S, colors)
            self._draw_brahmasthan(d, layer, S, colors)
        if poche:
            self._draw_poche(d, layer, S, colors, hatch)
        else:
            self._draw_walls(d, plan.walls, colors, hatch)
        self._draw_staircases(d, plan, colors)
        if poche:
            self._draw_cut_windows(d, layer, S, colors)
        else:
            self._draw_windows(d, plan.windows, plan.walls, colors)
        self._draw_doors(d, plan.doors, colors)
        if layer and layer.gates:
            self._draw_prims(d, layer.gates, S, colors)
        anchors = None
        if layer:
            self._draw_rwp(d, layer, S, colors)
            anchors = {n: (*S(x, y), crowded) for n, (x, y, crowded) in layer.labels.items()}
        self._draw_room_labels(d, plan.rooms, colors, anchors)
        if layer:
            self._draw_tags(d, layer, S, colors)
            self._draw_levels(d, layer, S, colors)
        self._draw_dimensions(d, plan.dimensions, colors)
        if layer:
            self._draw_grid_bubbles(d, layer, S, orig, colors)
        self._draw_north_arrow(d, plan, colors, beside_title=bool(layer))
        self._draw_title_block(d, plan, colors)
        if layer:
            self._draw_side_panel(d, orig, layer, colors)

        return d

    # =========================================================================
    # Drafting layer — furniture, poché walls, columns, grid, tags, schedule
    # =========================================================================
    def _prim_styles(self, colors):
        return {
            'furn': dict(stroke=colors['furn'], stroke_width=0.9, fill=colors['furn_fill']),
            'furn_d': dict(stroke=colors['furn_detail'], stroke_width=0.6, fill='none'),
            'fix': dict(stroke=colors['fixture'], stroke_width=0.9, fill=colors['fixture_fill']),
            'fix_d': dict(stroke=colors['fixture'], stroke_width=0.55, fill='none'),
            'dash': dict(stroke=colors['furn_detail'], stroke_width=0.6, fill='none', stroke_dasharray='3,2'),
            'tile': dict(stroke=colors['tile'], stroke_width=0.5, fill='none'),
            'gate': dict(stroke=colors['door'], stroke_width=1.4, fill=colors['door'], fill_opacity=0.25),
            'gate_d': dict(stroke=colors['door'], stroke_width=0.7, fill='none', stroke_dasharray='4,3'),
        }

    def _draw_prims(self, d, prims, S, colors):
        styles = self._prim_styles(colors)
        for p in prims:
            kind = p[0]
            if kind == 'poly':
                st = dict(styles.get(p[3], styles['furn_d']))
                if not p[2]:
                    st['fill'] = 'none'
                flat = [c for q in p[1] for c in S(*q)]
                d.append(draw.Lines(*flat, close=p[2], **st))
            elif kind == 'line':
                st = dict(styles.get(p[3], styles['furn_d']))
                st.pop('fill', None)
                d.append(draw.Line(*S(*p[1]), *S(*p[2]), **st))
            elif kind == 'circle':
                cx, cy = S(*p[1])
                d.append(draw.Circle(cx, cy, p[2] * PPM, **styles.get(p[3], styles['furn_d'])))
            elif kind == 'text':
                cx, cy = S(*p[1])
                size = p[3] * PPM
                d.append(draw.Text(p[2], size, cx, cy + size * 0.35, fill=colors['furn_detail'],
                                   text_anchor='middle', font_family='Arial, Helvetica, sans-serif'))

    @staticmethod
    def _ring_path(rings, S):
        parts = []
        for ring in rings:
            pts = [S(x, y) for x, y in ring]
            parts.append("M " + " L ".join(f"{x:.2f} {y:.2f}" for x, y in pts) + " Z")
        return " ".join(parts)

    def _draw_poche(self, d, layer, S, colors, hatch):
        """Walls as merged polygons: one clean outline, hatched poché, no seams at joints."""
        for ext, holes in layer.compound:
            d.append(draw.Path(self._ring_path([ext] + holes, S), fill=hatch, fill_rule='evenodd',
                               stroke=colors['wall_stroke'], stroke_width=1.3))
        for ext, holes in layer.walls:
            d.append(draw.Path(self._ring_path([ext] + holes, S), fill=hatch, fill_rule='evenodd',
                               stroke=colors['wall_stroke'], stroke_width=WALL_STROKE_HEAVY - 0.5))
        for x0, y0, x1, y1 in layer.grid.columns if layer.grid else []:
            sx0, sy0 = S(x0, y1)
            d.append(draw.Rectangle(sx0, sy0, (x1 - x0) * PPM, (y1 - y0) * PPM,
                                    fill=colors['column'], stroke=colors['wall_stroke'], stroke_width=0.6))

    def _draw_cut_windows(self, d, layer, S, colors):
        """Windows in cut wall openings: frame faces, sliding sashes, external sill."""
        for w in layer.windows:
            a, b, c, t = w['a'], w['b'], w['coord'], w['t']
            ox, oy = w['out']
            sgn = ox + oy
            mid = (a + b) / 2
            vent = w['category'] in ('BATHROOM', 'STORE') or w['length'] <= 0.6

            def P(u, v):  # u along the wall, v across (+ = outside)
                return S(u, c + sgn * v) if w['side'] in ('N', 'S') else S(c + sgn * v, u)

            for v in (t / 2, -t / 2):  # frame at both wall faces
                d.append(draw.Line(*P(a, v), *P(b, v), stroke=colors['window_frame'], stroke_width=0.8))
            if vent:
                d.append(draw.Line(*P(a, 0), *P(b, 0), stroke=colors['window_glass'], stroke_width=1.2))
                n = max(2, int((b - a) / 0.12))
                for i in range(1, n):
                    u = a + i * (b - a) / n
                    d.append(draw.Line(*P(u - 0.03, -0.04), *P(u + 0.03, 0.04),
                                       stroke=colors['window_glass'], stroke_width=0.6))
            else:  # two overlapping sliding sashes
                d.append(draw.Line(*P(a, 0.025), *P(mid + 0.06, 0.025), stroke=colors['window_glass'], stroke_width=1.3))
                d.append(draw.Line(*P(mid - 0.06, -0.025), *P(b, -0.025), stroke=colors['window_glass'], stroke_width=1.3))
            d.append(draw.Line(*P(a - 0.05, t / 2 + 0.05), *P(b + 0.05, t / 2 + 0.05),
                               stroke=colors['sill'], stroke_width=WINDOW_FRAME_STROKE))

    def _draw_grid_lines(self, d, layer, S, colors):
        g = layer.grid
        if not g or not g.xs:
            return
        fx, fy, fw, fd = g.footprint
        for x in g.xs:  # up to the bubbles above the plot
            d.append(draw.Line(*S(x, fy - 0.35), x * PPM, -0.55 * PPM, stroke=colors['grid'],
                               stroke_width=0.6, stroke_dasharray='10,3,2,3', stroke_opacity=0.6))
        for y in g.ys:  # continued to the right-hand bubbles in _draw_grid_bubbles
            d.append(draw.Line(*S(fx - 0.35, y), *S(fx + fw + 0.35, y), stroke=colors['grid'],
                               stroke_width=0.6, stroke_dasharray='10,3,2,3', stroke_opacity=0.6))

    def _draw_grid_bubbles(self, d, layer, S, plan, colors):
        g = layer.grid
        if not g or not g.xs:
            return
        fx, fy, fw, fd = g.footprint
        r = 9
        bx = (plan.plot_width + 0.45) * PPM
        for y, lab in zip(g.ys, g.y_labels):
            sy = S(0, y)[1]
            d.append(draw.Line(*S(fx + fw + 0.35, y), bx - r, sy, stroke=colors['grid'],
                               stroke_width=0.6, stroke_dasharray='10,3,2,3', stroke_opacity=0.6))
            d.append(draw.Circle(bx, sy, r, fill=colors['bg'], stroke=colors['grid'], stroke_width=1.1))
            d.append(draw.Text(lab, 9, bx, sy + 3.2, fill=colors['dim_text'], text_anchor='middle',
                               font_weight='bold', font_family='Arial, Helvetica, sans-serif'))
        by = -0.55 * PPM - r
        for x, lab in zip(g.xs, g.x_labels):
            sx = x * PPM
            d.append(draw.Circle(sx, by, r, fill=colors['bg'], stroke=colors['grid'], stroke_width=1.1))
            d.append(draw.Text(lab, 9, sx, by + 3.2, fill=colors['dim_text'], text_anchor='middle',
                               font_weight='bold', font_family='Arial, Helvetica, sans-serif'))

    def _draw_road(self, d, plan, colors):
        """Road strip beyond the front (entrance) boundary."""
        names = {"N": "NORTH", "E": "EAST", "S": "SOUTH", "W": "WEST"}
        face = names.get((plan.orientation or "N").upper()[:1], "NORTH")
        road = (getattr(plan, "engineering", None) or {}).get("road_m")
        width = f"{int(round(road / 0.3048 / 5) * 5)} FT ({road:g} M)  ·  " if road else ""
        x0, x1 = -0.6 * PPM, (plan.plot_width + 0.6) * PPM
        y_near, y_far = -1.25 * PPM, -2.75 * PPM
        for y in (y_near, y_far):
            d.append(draw.Line(x0, y, x1, y, stroke=colors['dim_line'], stroke_width=1.0))
        d.append(draw.Line(x0, (y_near + y_far) / 2, x1, (y_near + y_far) / 2, stroke=colors['dim_line'],
                           stroke_width=0.8, stroke_dasharray='14,8', stroke_opacity=0.7))
        d.append(draw.Text(f"ROAD  ·  {width}{face} FACING  ·  MAIN ENTRANCE", 10, (x0 + x1) / 2, y_near - 0.3 * PPM,
                           fill=colors['text_dim'], text_anchor='middle', font_weight='bold',
                           font_family='Arial, Helvetica, sans-serif', letter_spacing='1'))

    def _draw_brahmasthan(self, d, layer, S, colors):
        if not layer.brahmasthan:
            return
        x0, y0, x1, y1 = layer.brahmasthan
        sx, sy = S(x0, y1)
        d.append(draw.Rectangle(sx, sy, (x1 - x0) * PPM, (y1 - y0) * PPM, fill='none', stroke=colors['grid'],
                                stroke_width=0.8, stroke_dasharray='4,4', stroke_opacity=0.55))

    def _draw_rwp(self, d, layer, S, colors):
        for x, y in layer.rwp:
            sx, sy = S(x, y)
            d.append(draw.Circle(sx, sy, 3.5, fill=colors['bg'], stroke=colors['window_glass'], stroke_width=0.9))
            d.append(draw.Text("RWP", 5.5, sx, sy - 5.5, fill=colors['text_dim'], text_anchor='middle',
                               font_family='Arial, Helvetica, sans-serif'))

    def _draw_tags(self, d, layer, S, colors):
        for x, y, text, kind in layer.tags:
            sx, sy = S(x, y)
            col = colors['door'] if kind in ('door', 'gate') else colors['window_glass']
            w = 6 + 5.2 * len(text)
            d.append(draw.Rectangle(sx - w / 2, sy - 5.5, w, 11, rx=2.5, fill=colors['plate'], fill_opacity=0.9,
                                    stroke=col, stroke_width=0.6))
            d.append(draw.Text(text, 7, sx, sy + 2.5, fill=col, text_anchor='middle', font_weight='bold',
                               font_family='Arial, Helvetica, sans-serif'))

    def _draw_levels(self, d, layer, S, colors):
        for x, y, text in layer.levels:
            sx, sy = S(x, y)
            r = 4
            cx = sx - 2.6 * len(text) - 7
            d.append(draw.Circle(cx, sy, r, fill='none', stroke=colors['dim_text'], stroke_width=0.8))
            d.append(draw.Path(f"M {cx} {sy - r} A {r} {r} 0 0 1 {cx + r} {sy} L {cx} {sy} Z",
                               fill=colors['dim_text']))
            d.append(draw.Path(f"M {cx} {sy + r} A {r} {r} 0 0 1 {cx - r} {sy} L {cx} {sy} Z",
                               fill=colors['dim_text']))
            d.append(draw.Text(text, 7.5, cx + r + 3, sy + 2.6, fill=colors['dim_text'],
                               font_weight='bold', font_family='Arial, Helvetica, sans-serif'))

    def _draw_side_panel(self, d, plan, layer, colors):
        """Schedule of openings, legend, general notes and a graphic scale bar."""
        x0 = (plan.plot_width + 0.8) * PPM
        width = 6.4 * PPM
        y = 3.0 * PPM + 14
        font = 'Arial, Helvetica, sans-serif'
        tc, dc = colors['title_text'], colors['text_dim']

        def header(text, y):
            d.append(draw.Rectangle(x0, y, width, 16, fill=colors['title_stroke'], fill_opacity=0.18,
                                    stroke=colors['title_stroke'], stroke_width=1.0))
            d.append(draw.Text(text, 8.5, x0 + 6, y + 11, fill=tc, font_weight='bold', font_family=font,
                               letter_spacing='0.8'))
            return y + 16

        # --- schedule of openings
        y = header("SCHEDULE OF DOORS, WINDOWS & GATES" if any(r['kind'] == 'gate' for r in layer.schedule)
                   else "SCHEDULE OF DOORS & WINDOWS", y)
        cols = [(6, "TAG"), (36, "SIZE (mm)"), (104, "SILL"), (136, "NOS"), (164, "SPECIFICATION")]
        top = y
        for cx, lab in cols:
            d.append(draw.Text(lab, 6.5, x0 + cx, y + 10, fill=dc, font_weight='bold', font_family=font))
        y += 14
        d.append(draw.Line(x0, y - 1, x0 + width, y - 1, stroke=colors['title_stroke'], stroke_width=0.5))
        for row in layer.schedule:
            col = colors['door'] if row['kind'] in ('door', 'gate') else colors['window_glass']
            vals = [row['tag'], row['size'], row['sill'], str(row['nos']), row['spec']]
            for (cx, _), v in zip(cols, vals):
                d.append(draw.Text(v, 7, x0 + cx, y + 9, fill=col if cx == 6 else tc,
                                   font_weight='bold' if cx == 6 else 'normal', font_family=font))
            y += 13
        d.append(draw.Rectangle(x0, top, width, y - top + 3, fill='none', stroke=colors['title_stroke'],
                                stroke_width=1.0))
        y += 14

        # --- legend
        y = header("LEGEND", y)
        y += 6
        items = [("col", "RCC column 230 × 300"), ("grid", "Structural grid line"),
                 ("wall", "Brick wall (hatched)"), ("rwp", "RWP - rain-water pipe"),
                 ("tile", "Anti-skid tiles (wet areas)"), ("brahma", "Brahmasthan - keep open")]
        for i, (kind, text) in enumerate(items):
            cx = x0 + 8 + (i % 2) * (width / 2)
            cy = y + (i // 2) * 16
            if kind == "col":
                d.append(draw.Rectangle(cx, cy, 8, 10, fill=colors['column']))
            elif kind == "grid":
                d.append(draw.Line(cx - 2, cy + 5, cx + 18, cy + 5, stroke=colors['grid'], stroke_width=1.0,
                                   stroke_dasharray='6,2,1,2'))
            elif kind == "wall":
                d.append(draw.Rectangle(cx, cy + 1, 18, 8, fill=colors['hatch'], stroke=colors['wall_stroke'],
                                        stroke_width=1.0))
            elif kind == "rwp":
                d.append(draw.Circle(cx + 5, cy + 5, 3.5, fill='none', stroke=colors['window_glass'], stroke_width=0.9))
            elif kind == "tile":
                for k in range(4):
                    d.append(draw.Line(cx + k * 5, cy, cx + k * 5, cy + 10, stroke=colors['tile'], stroke_width=0.6))
                    d.append(draw.Line(cx, cy + k * 3.3, cx + 15, cy + k * 3.3, stroke=colors['tile'], stroke_width=0.6))
            else:
                d.append(draw.Rectangle(cx, cy, 18, 10, fill='none', stroke=colors['grid'], stroke_width=0.8,
                                        stroke_dasharray='3,3'))
            d.append(draw.Text(text, 7, cx + 26, cy + 8, fill=tc, font_family=font))
        y += ((len(items) + 1) // 2) * 16 + 10

        # --- general notes
        y = header("GENERAL NOTES", y)
        y += 4
        for i, note in enumerate(layer.notes, 1):
            d.append(draw.Text(f"{i}. {note}", 7, x0 + 6, y + 9, fill=tc, font_family=font))
            y += 12
        y += 14

        # --- graphic scale bar (5 m)
        d.append(draw.Text("GRAPHIC SCALE (m)", 7, x0, y, fill=dc, font_weight='bold', font_family=font))
        y += 6
        for i in range(5):
            d.append(draw.Rectangle(x0 + i * PPM, y, PPM, 6, stroke=colors['title_stroke'], stroke_width=0.8,
                                    fill=colors['title_stroke'] if i % 2 == 0 else 'none'))
        for i in range(6):
            d.append(draw.Text(str(i), 7, x0 + i * PPM, y + 16, fill=dc, text_anchor='middle', font_family=font))

    def _label_metrics(self, r):
        area = r.area
        # drafting-scale lettering: rooms now carry furniture, so names stay compact
        font_title = max(9, min(14, 7 + 1.0 * math.sqrt(max(area, 0))))
        font_dim = max(7, font_title - 2.5)
        font_area = max(6.5, font_title - 3.5)
        avail_px = r.w * PPM - 8
        name, font_title = fit_label(r.name, avail_px, font_title)
        dim_text = f"{r.w:.2f} x {r.h:.2f} m"
        if len(dim_text) * font_dim * 0.55 > avail_px:
            font_dim = max(6.0, avail_px / (len(dim_text) * 0.55))
        font_area = min(font_area, font_dim)
        return name, font_title, font_dim, font_area, dim_text, font_title + 4

    def _label_box(self, r):
        """Label footprint in metres (used by the drafting layer to keep labels clear of furniture)."""
        name, ft, fd, fa, dim_text, ls = self._label_metrics(r)
        w = max(len(name) * ft * 0.62, len(dim_text) * fd * 0.55) + 6
        h = 1.5 * ls + 0.75 * ft + 0.2 * fa + 4
        return w / PPM, h / PPM


    # =========================================================================
    # Staircases & Lift Shafts
    # =========================================================================
    def _draw_staircases(self, d, plan, colors):
        """Draw staircase tread lines and lift shaft X-pattern for all stair/lift rooms."""
        stair_map = {}
        if hasattr(plan, 'staircases') and plan.staircases:
            for sc in plan.staircases:
                stair_map[sc.name] = getattr(sc, 'stair_type', 'dog_leg')

        default_stair = getattr(plan, 'stair_type', 'dog_leg')

        for room in plan.rooms:
            if hasattr(room, 'category'):
                if room.category == 'LIFT':
                    self._draw_lift_room(d, room, colors)
                elif room.category == 'STAIRS':
                    s_type = stair_map.get(room.name, default_stair)
                    self._draw_stair_room(d, room, colors, stair_type=s_type)

    def _draw_stair_room(self, d, room, colors, stair_type="dog_leg"):
        """Draw staircase treads, risers, landings, and arrows according to stair_type."""
        if room.w <= 0 or room.h <= 0:
            return

        rx = room.x * PPM
        ry = room.y * PPM
        rw = room.w * PPM
        rh = room.h * PPM
        cx = room.cx * PPM
        cy = room.cy * PPM

        tread_stroke = colors.get('dim_line', '#6B8DAF')
        wall_stroke = colors.get('wall_stroke', '#00FFFF')
        text_fill = colors.get('text', '#FFFFFF')
        arrow_fill = colors.get('text_dim', '#8EB8E5')

        stype = str(stair_type).lower()

        if stype == "straight":
            # Single continuous flight of treads with directional arrow
            num_treads = max(6, int(rh / (0.25 * PPM)))
            for i in range(1, num_treads):
                y_t = ry + i * (rh / num_treads)
                d.append(draw.Line(rx, y_t, rx + rw, y_t, stroke=tread_stroke, stroke_width=0.8))
            # Directional arrow
            d.append(draw.Line(cx, ry + 12, cx, ry + rh - 12, stroke=arrow_fill, stroke_width=1.2))
            d.append(draw.Lines(cx - 4, ry + rh - 20, cx, ry + rh - 12, cx + 4, ry + rh - 20,
                                close=False, fill='none', stroke=arrow_fill, stroke_width=1.5))
            d.append(draw.Text('UP ▲', 8.5, cx + 6, cy, fill=text_fill, font_weight='bold'))

        elif stype == "l_shaped":
            # Two perpendicular flights meeting at a corner landing
            land_w = rw * 0.45
            land_h = rh * 0.40
            # Corner Landing (top-left)
            d.append(draw.Rectangle(rx, ry + rh - land_h, land_w, land_h,
                                    fill='none', stroke=tread_stroke, stroke_width=1.0, stroke_dasharray='3,2'))
            d.append(draw.Text('LANDING', 7, rx + land_w / 2, ry + rh - land_h / 2 + 2,
                               fill=arrow_fill, text_anchor='middle'))

            # Flight 1 (Vertical, bottom-left)
            num_v = max(3, int((rh - land_h) / (0.25 * PPM)))
            for i in range(1, num_v):
                y_t = ry + i * ((rh - land_h) / num_v)
                d.append(draw.Line(rx, y_t, rx + land_w, y_t, stroke=tread_stroke, stroke_width=0.8))

            # Flight 2 (Horizontal, top-right)
            num_h = max(3, int((rw - land_w) / (0.25 * PPM)))
            for i in range(1, num_h):
                x_t = rx + land_w + i * ((rw - land_w) / num_h)
                d.append(draw.Line(x_t, ry + rh - land_h, x_t, ry + rh, stroke=tread_stroke, stroke_width=0.8))

            # L-shaped walk line
            mid_v_x = rx + land_w / 2
            mid_h_y = ry + rh - land_h / 2
            d.append(draw.Line(mid_v_x, ry + 10, mid_v_x, mid_h_y, stroke=arrow_fill, stroke_width=1.0))
            d.append(draw.Line(mid_v_x, mid_h_y, rx + rw - 10, mid_h_y, stroke=arrow_fill, stroke_width=1.0))
            d.append(draw.Lines(rx + rw - 16, mid_h_y - 4, rx + rw - 10, mid_h_y, rx + rw - 16, mid_h_y + 4,
                                close=False, fill='none', stroke=arrow_fill, stroke_width=1.2))
            d.append(draw.Text('UP ►', 8, rx + rw - 22, mid_h_y - 6, fill=text_fill, text_anchor='middle', font_weight='bold'))

        elif stype == "open_well":
            # Three-sided or bifurcated staircase with clear open center well
            well_w = rw * 0.32
            well_h = rh * 0.35
            # Center well opening
            d.append(draw.Rectangle(cx - well_w / 2, cy - well_h / 2, well_w, well_h,
                                    fill='none', stroke=wall_stroke, stroke_width=1.2, stroke_dasharray='4,2'))
            d.append(draw.Text('OPEN\nWELL', 7.5, cx, cy - 2, fill=arrow_fill, text_anchor='middle'))

            # Flight 1 (Left): Treads from bottom to top
            num_t = max(4, int(rh / (0.28 * PPM)))
            for i in range(1, num_t):
                y_t = ry + i * (rh / num_t)
                d.append(draw.Line(rx, y_t, cx - well_w / 2, y_t, stroke=tread_stroke, stroke_width=0.8))

            # Flight 2 (Top): Landing / top flight
            num_top = max(2, int(well_w / (0.25 * PPM)))
            for i in range(1, num_top):
                x_t = (cx - well_w / 2) + i * (well_w / num_top)
                d.append(draw.Line(x_t, cy + well_h / 2, x_t, ry + rh, stroke=tread_stroke, stroke_width=0.8))

            # Flight 3 (Right): Treads
            for i in range(1, num_t):
                y_t = ry + i * (rh / num_t)
                d.append(draw.Line(cx + well_w / 2, y_t, rx + rw, y_t, stroke=tread_stroke, stroke_width=0.8))

            # Walking directional arrow around the well
            d.append(draw.Text('▲ UP', 8, rx + (cx - well_w / 2 - rx) / 2, ry + 15, fill=text_fill, text_anchor='middle', font_weight='bold'))

        elif stype == "spiral":
            # Circular outer ring with radial tread spokes radiating from a central pillar
            r_pillar = max(6, min(rw, rh) * 0.12)
            r_outer = min(rw, rh) * 0.44

            d.append(draw.Circle(cx, cy, r_outer, fill='none', stroke=tread_stroke, stroke_width=1.5, stroke_dasharray='4,2'))
            d.append(draw.Circle(cx, cy, r_pillar, fill=tread_stroke, stroke=wall_stroke, stroke_width=1.2))

            num_spokes = 14
            for s in range(num_spokes):
                theta = s * (2 * math.pi / num_spokes)
                x1 = cx + r_pillar * math.cos(theta)
                y1 = cy + r_pillar * math.sin(theta)
                x2 = cx + r_outer * math.cos(theta)
                y2 = cy + r_outer * math.sin(theta)
                d.append(draw.Line(x1, y1, x2, y2, stroke=tread_stroke, stroke_width=0.8))

            d.append(draw.Text('UP ↺', 8.5, cx, cy - r_pillar - 6, fill=text_fill, text_anchor='middle', font_weight='bold'))

        elif stype == "zigzag":
            # Alternating split-flight treads with crossing markers
            num_t = max(5, int(rh / (0.25 * PPM)))
            mid_x = rx + rw / 2
            d.append(draw.Line(mid_x, ry, mid_x, ry + rh, stroke=wall_stroke, stroke_width=1.2))

            for i in range(1, num_t):
                y_left = ry + (i - 0.5) * (rh / num_t)
                y_right = ry + i * (rh / num_t)
                d.append(draw.Line(rx, y_left, mid_x, y_left, stroke=tread_stroke, stroke_width=0.8))
                d.append(draw.Line(mid_x, y_right, rx + rw, y_right, stroke=tread_stroke, stroke_width=0.8))

            # Crossing zigzag line connecting flights
            d.append(draw.Line(rx + rw * 0.25, ry + rh * 0.5, rx + rw * 0.75, ry + rh * 0.55, stroke=arrow_fill, stroke_width=1.5, stroke_dasharray='3,2'))
            d.append(draw.Text('UP ⮵', 8, cx, cy, fill=text_fill, text_anchor='middle', font_weight='bold'))

        else:
            # Default dog_leg: Two parallel flights with landing and central dividing well line
            landing_h = rh * 0.25
            mid_x = rx + rw / 2

            # Central dividing line
            d.append(draw.Line(mid_x, ry, mid_x, ry + rh - landing_h, stroke=wall_stroke, stroke_width=1.2))
            # Landing dividing line
            d.append(draw.Line(rx, ry + rh - landing_h, rx + rw, ry + rh - landing_h, stroke=tread_stroke, stroke_width=1.2))
            d.append(draw.Text('LANDING', 7.5, cx, ry + rh - landing_h / 2 + 2, fill=arrow_fill, text_anchor='middle'))

            # Flight 1 treads (Left half, going up)
            flight_h = rh - landing_h
            num_treads = max(4, int(flight_h / (0.25 * PPM)))
            for i in range(1, num_treads):
                y_t = ry + i * (flight_h / num_treads)
                d.append(draw.Line(rx, y_t, mid_x - 1, y_t, stroke=tread_stroke, stroke_width=0.8))

            # Flight 2 treads (Right half)
            for i in range(1, num_treads):
                y_t = ry + i * (flight_h / num_treads)
                d.append(draw.Line(mid_x + 1, y_t, rx + rw, y_t, stroke=tread_stroke, stroke_width=0.8))

            # Up arrow on left flight
            f1_cx = rx + rw * 0.25
            d.append(draw.Line(f1_cx, ry + 10, f1_cx, ry + flight_h - 10, stroke=arrow_fill, stroke_width=1.0))
            d.append(draw.Lines(f1_cx - 3, ry + flight_h - 16, f1_cx, ry + flight_h - 10, f1_cx + 3, ry + flight_h - 16,
                                close=False, fill='none', stroke=arrow_fill, stroke_width=1.2))
            d.append(draw.Text('UP', 8, f1_cx, ry + flight_h * 0.5, fill=text_fill, text_anchor='middle', font_weight='bold'))

    def _draw_lift_room(self, d, room, colors):
        """Draw diagonal X-pattern and LIFT label inside a lift shaft room."""
        if room.w <= 0 or room.h <= 0:
            return
        x1 = room.x * PPM
        y1 = room.y * PPM
        x2 = room.right * PPM
        y2 = room.top * PPM
        lift_color = colors.get('window_glass', '#00BFFF')
        # Diagonal X
        d.append(draw.Line(x1, y1, x2, y2, stroke=lift_color, stroke_width=1.0))
        d.append(draw.Line(x2, y1, x1, y2, stroke=lift_color, stroke_width=1.0))
        # LIFT label
        d.append(draw.Text(
            'LIFT', 9, room.cx * PPM, room.cy * PPM,
            fill=lift_color,
            text_anchor='middle',
            font_weight='bold'
        ))

    # =========================================================================
    # Amenity / Site Components (Parking, Garden, Pool, Sump, Genset, Security)
    # =========================================================================
    @staticmethod
    def _car_symbol(d, cx, cy, stall_w, stall_h, along_x):
        """Plan-view car (1.8 × 4.5 m hatchback/sedan) centred in a stall, nose towards +x / +y."""
        L = min(4.5 * PPM, (stall_w if along_x else stall_h) * 0.86)
        W = min(1.8 * PPM, (stall_h if along_x else stall_w) * 0.70)
        bw, bh = (L, W) if along_x else (W, L)
        x0, y0 = cx - bw / 2, cy - bh / 2
        stroke = '#94A3B8'
        d.append(draw.Rectangle(x0, y0, bw, bh, rx=W * 0.22, fill='none', stroke=stroke, stroke_width=1.0))

        def band(t0, t1):
            # windscreen / rear-glass band across the car at fractions t0..t1 of its length
            if along_x:
                return draw.Rectangle(x0 + L * t0, y0 + W * 0.12, L * (t1 - t0), W * 0.76, rx=2,
                                      fill='none', stroke=stroke, stroke_width=0.7)
            return draw.Rectangle(x0 + W * 0.12, y0 + L * t0, W * 0.76, L * (t1 - t0), rx=2,
                                  fill='none', stroke=stroke, stroke_width=0.7)

        d.append(band(0.30, 0.74))          # cabin
        d.append(band(0.66, 0.74))          # windscreen (nose side)
        d.append(band(0.30, 0.36))          # rear glass
        # wheels
        for t in (0.17, 0.83):
            for s in (-1, 1):
                if along_x:
                    wx, wy = x0 + L * t, cy + s * (W / 2)
                    d.append(draw.Line(wx - 6, wy, wx + 6, wy, stroke=stroke, stroke_width=2.2))
                else:
                    wx, wy = cx + s * (W / 2), y0 + L * t
                    d.append(draw.Line(wx, wy - 6, wx, wy + 6, stroke=stroke, stroke_width=2.2))

    def _draw_amenities(self, d, rooms, colors):
        """
        Render exterior site amenities and utilities:
        - PARKING & BIKE_PARKING: Stalls, divider lines, bay numbers, 🅿️ / 🏍️ labels.
        - GARDEN: Textured lawn fill, dashed border, decorative tree symbols at corners.
        - POOL: Double outline pool coping, cyan water fill with wave lines and depth notation.
        - WATER_TANK: Inset circle symbol with SUMP / 💧 notation.
        - GENERATOR: Equipment hatched cross with ⚡ GENSET label.
        - SECURITY: Compact cabin booth with door swing arc and 🛡️ SECURITY label.
        """
        for r in rooms:
            cat = getattr(r, 'category', '')
            if cat not in ('PARKING', 'BIKE_PARKING', 'GARDEN', 'POOL', 'WATER_TANK', 'GENERATOR', 'SECURITY'):
                continue
            if r.w <= 0 or r.h <= 0:
                continue

            rx = r.x * PPM
            ry = r.y * PPM
            rw = r.w * PPM
            rh = r.h * PPM
            cx = r.cx * PPM
            cy = r.cy * PPM

            if cat == 'PARKING':
                slots = 1
                if '(' in r.name and 'Bay' in r.name:
                    try:
                        slots = max(1, int(r.name.split('(')[1].split()[0]))
                    except Exception:
                        slots = 1

                d.append(draw.Rectangle(rx, ry, rw, rh, fill='#1E293B', fill_opacity=0.60,
                                        stroke='#38BDF8', stroke_width=1.8, stroke_dasharray='6,4'))
                # a 2.6 m bay is the short side: stalls sit side by side across the long side
                if rw >= rh:
                    stall_w = rw / slots
                    for i in range(1, slots):
                        bx = rx + i * stall_w
                        d.append(draw.Line(bx, ry, bx, ry + rh, stroke='#38BDF8', stroke_width=1.2, stroke_dasharray='4,4'))
                    for i in range(slots):
                        bx_mid = rx + (i + 0.5) * stall_w
                        self._car_symbol(d, bx_mid, ry + rh / 2, stall_w, rh, along_x=stall_w > rh)
                        d.append(draw.Text(f"SLOT {i+1}", 7, bx_mid, ry + 10, fill='#38BDF8', text_anchor='middle', font_weight='bold'))
                else:
                    stall_h = rh / slots
                    for i in range(1, slots):
                        by = ry + i * stall_h
                        d.append(draw.Line(rx, by, rx + rw, by, stroke='#38BDF8', stroke_width=1.2, stroke_dasharray='4,4'))
                    for i in range(slots):
                        by_mid = ry + (i + 0.5) * stall_h
                        self._car_symbol(d, rx + rw / 2, by_mid, rw, stall_h, along_x=rw > stall_h)
                        d.append(draw.Text(f"SLOT {i+1}", 7, rx + rw / 2, ry + i * stall_h + 10, fill='#38BDF8',
                                           text_anchor='middle', font_weight='bold'))

                # title on a backing plate so it reads over the car outline
                d.append(draw.Rectangle(cx - 38, cy - 13, 76, 30, fill='#0F172A', fill_opacity=0.80, rx=2))
                d.append(draw.Text("CAR PARKING", 9, cx, cy - 1, fill='#FFFFFF', text_anchor='middle', font_weight='bold'))
                d.append(draw.Text(f"{r.w:.1f} × {r.h:.1f} m", 7.5, cx, cy + 11, fill='#94A3B8', text_anchor='middle'))

            elif cat == 'BIKE_PARKING':
                slots = 2
                if '(' in r.name and 'Bike' in r.name:
                    try:
                        slots = max(2, int(r.name.split('(')[1].split()[0]))
                    except Exception:
                        slots = 2

                d.append(draw.Rectangle(rx, ry, rw, rh, fill='#334155', fill_opacity=0.55,
                                        stroke='#94A3B8', stroke_width=1.5, stroke_dasharray='5,3'))
                num_divs = min(slots, 6)
                if num_divs > 1:
                    # bays run along the side whose share per bike is nearest a bike's width
                    # (perpendicular stalls 0.9 m; parallel-parked scooters 0.75 m)
                    split_x = abs(r.w / num_divs - 0.8) <= abs(r.h / num_divs - 0.8)
                    for i in range(1, num_divs):
                        if split_x:
                            bx = rx + i * rw / num_divs
                            d.append(draw.Line(bx, ry, bx, ry + rh, stroke='#94A3B8', stroke_width=0.8, stroke_dasharray='3,3'))
                        else:
                            by = ry + i * rh / num_divs
                            d.append(draw.Line(rx, by, rx + rw, by, stroke='#94A3B8', stroke_width=0.8, stroke_dasharray='3,3'))

                d.append(draw.Text("🏍️ BIKE PARKING", 9, cx, cy - 4, fill='#F1F5F9', text_anchor='middle', font_weight='bold'))
                d.append(draw.Text(f"{slots} Bays ({r.w:.1f}×{r.h:.1f}m)", 7.5, cx, cy + 10, fill='#94A3B8', text_anchor='middle'))

            elif cat == 'GARDEN':
                d.append(draw.Rectangle(rx, ry, rw, rh, fill='#14532D', fill_opacity=0.35,
                                        stroke='#22C55E', stroke_width=1.8, stroke_dasharray='8,4'))
                coff = 14
                tree_pts = [
                    (rx + coff, ry + coff),
                    (rx + rw - coff, ry + coff),
                    (rx + coff, ry + rh - coff),
                    (rx + rw - coff, ry + rh - coff),
                ]
                for tx, ty in tree_pts:
                    if tx < rx + rw and ty < ry + rh:
                        d.append(draw.Text("🌳", 12, tx, ty + 4, text_anchor='middle'))

                d.append(draw.Text("🌿 LANDSCAPED GARDEN", 10, cx, cy - 4, fill='#86EFAC', text_anchor='middle', font_weight='bold'))
                d.append(draw.Text(f"{r.area:.1f} m²", 8.5, cx, cy + 11, fill='#BBF7D0', text_anchor='middle'))

            elif cat == 'POOL':
                d.append(draw.Rectangle(rx, ry, rw, rh, fill='none', stroke='#94A3B8', stroke_width=3.0))
                coping = 5.0
                d.append(draw.Rectangle(rx + coping, ry + coping, rw - 2 * coping, rh - 2 * coping,
                                        fill='#0284C7', fill_opacity=0.70, stroke='#38BDF8', stroke_width=1.5))

                wave_y1 = ry + rh * 0.32
                wave_y2 = ry + rh * 0.68
                seg_w = 20
                num_segs = max(2, int((rw - 20) / seg_w))
                w_path1 = f"M {rx + 10} {wave_y1}"
                w_path2 = f"M {rx + 10} {wave_y2}"
                for s in range(num_segs):
                    x_start = rx + 10 + s * seg_w
                    x_mid = x_start + seg_w / 2
                    x_end = x_start + seg_w
                    w_path1 += f" Q {x_mid} {wave_y1 - 4} {x_end} {wave_y1}"
                    w_path2 += f" Q {x_mid} {wave_y2 + 4} {x_end} {wave_y2}"
                d.append(draw.Path(w_path1, fill='none', stroke='#E0F2FE', stroke_width=1.0, stroke_opacity=0.7))
                d.append(draw.Path(w_path2, fill='none', stroke='#E0F2FE', stroke_width=1.0, stroke_opacity=0.7))

                d.append(draw.Text("🏊 SWIMMING POOL", 9.5, cx, cy - 4, fill='#FFFFFF', text_anchor='middle', font_weight='bold'))
                d.append(draw.Text("Depth: 1.2m – 1.8m", 8.0, cx, cy + 11, fill='#BAE6FD', text_anchor='middle', font_weight='bold'))

            elif cat == 'WATER_TANK':
                d.append(draw.Rectangle(rx, ry, rw, rh, fill='#0284C7', fill_opacity=0.30,
                                        stroke='#0284C7', stroke_width=1.8))
                r_c = max(10, min(rw, rh) * 0.35)
                d.append(draw.Circle(cx, cy, r_c, fill='none', stroke='#38BDF8', stroke_width=1.5, stroke_dasharray='4,3'))
                cap = r.name.split('(')[1].rstrip(')') if '(' in r.name else "Ishan (NE)"
                d.append(draw.Text("SUMP / 💧", 8.5, cx, cy - 3, fill='#38BDF8', text_anchor='middle', font_weight='bold'))
                d.append(draw.Text(cap, 7.5, cx, cy + 10, fill='#E0F2FE', text_anchor='middle'))

            elif cat == 'GENERATOR':
                d.append(draw.Rectangle(rx, ry, rw, rh, fill='#B45309', fill_opacity=0.25,
                                        stroke='#F59E0B', stroke_width=1.8))
                d.append(draw.Line(rx + 4, ry + 4, rx + rw - 4, ry + rh - 4, stroke='#F59E0B', stroke_width=1.0))
                d.append(draw.Line(rx + rw - 4, ry + 4, rx + 4, ry + rh - 4, stroke='#F59E0B', stroke_width=1.0))
                box_w = min(rw * 0.5, 30)
                box_h = min(rh * 0.4, 22)
                d.append(draw.Rectangle(cx - box_w / 2, cy - box_h / 2, box_w, box_h,
                                        fill='#78350F', stroke='#F59E0B', stroke_width=1.0))
                d.append(draw.Text("⚡ GENSET", 8.5, cx, cy + 3, fill='#FDE68A', text_anchor='middle', font_weight='bold'))
                if '(' in r.name:
                    d.append(draw.Text(r.name.split('(')[1].rstrip(')'), 7, cx, cy + 14,
                                       fill='#FDE68A', text_anchor='middle'))

            elif cat == 'SECURITY':
                d.append(draw.Rectangle(rx, ry, rw, rh, fill='#6D28D9', fill_opacity=0.30,
                                        stroke='#8B5CF6', stroke_width=1.8))
                d.append(draw.Rectangle(rx + 4, ry + 4, rw * 0.40, rh * 0.35, fill='#4C1D95', stroke='#A78BFA', stroke_width=0.8))
                arc_r = min(rw, rh) * 0.30
                d.append(draw.Line(rx + rw - 4, ry + 4, rx + rw - 4, ry + 4 + arc_r, stroke='#A78BFA', stroke_width=1.0))
                d_path = f"M {rx + rw - 4} {ry + 4 + arc_r} A {arc_r} {arc_r} 0 0 1 {rx + rw - 4 - arc_r} {ry + 4}"
                d.append(draw.Path(d_path, fill='none', stroke='#A78BFA', stroke_width=0.8, stroke_dasharray='2,2'))
                d.append(draw.Text("🛡️ SECURITY", 8.5, cx, cy + 6, fill='#DDD6FE', text_anchor='middle', font_weight='bold'))

    # =========================================================================
    # Room Fills
    # =========================================================================
    def _draw_room_fills(self, d, rooms, colors):
        """Draw room floor area fills with subtle color."""
        amenity_cats = {'PARKING', 'BIKE_PARKING', 'GARDEN', 'POOL', 'WATER_TANK', 'GENERATOR', 'SECURITY'}
        for r in rooms:
            if r.w <= 0 or r.h <= 0 or getattr(r, 'category', '') in amenity_cats:
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
            d.append(draw.Circle(hx, hy, 1.8, fill=colors['door']))

            # 4. Threshold across the cut opening (thin; heavy only for the main door sill)
            d.append(draw.Line(
                hx, hy, closed_x, closed_y,
                stroke=colors['door'],
                stroke_width=DOOR_LEAF_STROKE if door.is_main_entrance else 0.6,
                stroke_dasharray='none' if door.is_main_entrance else '3,2'
            ))

            # 5. Draw entrance steps for main entrance (outward, centred on the opening)
            if door.is_main_entrance:
                w = door.width
                if side in ('N', 'S'):
                    c = door.x + w / 2 if swing != 'LEFT' else door.x - w / 2
                    shim = SimpleNamespace(x=c, y=door.y, width=w,
                                           wall_side={'N': 'S', 'S': 'N'}[side])
                else:
                    down = (side == 'E' and swing == 'RIGHT') or (side == 'W' and swing == 'LEFT')
                    c = door.y + w / 2 if down else door.y - w / 2
                    shim = SimpleNamespace(x=door.x, y=c, width=w, wall_side=side)
                self._draw_entrance_steps(d, shim, colors)

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
    def _draw_room_labels(self, d, rooms, colors, anchors=None):
        """Room name, dimensions and area — at the drafting layer's free spot, else the centroid."""
        amenity_cats = {'PARKING', 'BIKE_PARKING', 'GARDEN', 'POOL', 'WATER_TANK', 'GENERATOR', 'SECURITY'}
        for r in rooms:
            if r.w <= 0 or r.h <= 0 or getattr(r, 'category', '') in amenity_cats:
                continue

            name, font_title, font_dim, font_area, dim_text, line_spacing = self._label_metrics(r)
            area = r.area
            cx, cy = r.cx * PPM, r.cy * PPM
            if anchors and r.name in anchors:
                ax, ay, crowded = anchors[r.name]
                # text block's visual centre sits ≈(0.75·title − 0.1·spacing)/2 above the middle line
                cx, cy = ax, ay + (0.1 * line_spacing + 0.75 * font_title - 0.2 * font_area) / 2
                if crowded:
                    bw, bh = self._label_box(r)
                    d.append(draw.Rectangle(ax - bw * PPM / 2, ay - bh * PPM / 2, bw * PPM, bh * PPM, rx=3,
                                            fill=colors['plate'], fill_opacity=0.78, stroke='none'))

            # Line 1: Room name (bold, uppercase)
            d.append(draw.Text(
                name, font_title,
                cx, cy - line_spacing * 0.8,
                fill=colors['text'], text_anchor='middle',
                font_weight='bold',
                font_family='Arial, Helvetica, sans-serif'
            ))

            # Line 2: Width x Height dimensions
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
            offset_m = dim_offset(dim)
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
                x1, y1, dim_x1, dim_y1,
                stroke=colors['dim_line'], stroke_width=0.5
            ))
            d.append(draw.Line(
                x2, y2, dim_x2, dim_y2,
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
    def _draw_north_arrow(self, d, plan, colors, beside_title=False):
        """Draw a compass north arrow with N/E/S/W labels."""
        if beside_title:  # right of the title block (the side panel fills the column below it)
            cx = (plan.plot_width + 6.0) * PPM
            cy = 1.4 * PPM
        else:             # beside the plot, near the bottom
            cx = (plan.plot_width + 1.5) * PPM
            cy = (plan.plot_height - 1.5) * PPM
        r = 20  # Arrow radius in pixels
        ang = north_angle(plan)  # clockwise from 'up' on the sheet

        # Circle outline
        d.append(draw.Circle(cx, cy, r + 8,
                             stroke=colors['north_stroke'], fill='none', stroke_width=1.5))

        g = draw.Group(transform=f"rotate({ang:.1f},{cx:.2f},{cy:.2f})")
        # Solid north half (filled triangle)
        g.append(draw.Lines(
            cx, cy - r,        # North tip
            cx - r * 0.35, cy, # Left base
            cx, cy,            # Center
            close=True,
            fill=colors['north_fill'], stroke=colors['north_stroke'], stroke_width=1
        ))
        # Outline east half
        g.append(draw.Lines(
            cx, cy - r,
            cx + r * 0.35, cy,
            cx, cy,
            close=True,
            fill='none', stroke=colors['north_stroke'], stroke_width=1
        ))
        d.append(g)

        # Upright compass letters at rotated positions
        for label, base, size, rad in (("N", 0, 12, r + 13), ("E", 90, 8, r + 13),
                                       ("S", 180, 8, r + 13), ("W", 270, 8, r + 13)):
            a = math.radians(base + ang)
            tx = cx + rad * math.sin(a)
            ty = cy - rad * math.cos(a) + (4 if size == 12 else 3)
            d.append(draw.Text(
                label, size, tx, ty,
                fill=colors['north_fill'] if label == "N" else colors['text_dim'],
                text_anchor='middle', font_weight='bold' if label == "N" else 'normal',
                font_family='Arial'
            ))

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
