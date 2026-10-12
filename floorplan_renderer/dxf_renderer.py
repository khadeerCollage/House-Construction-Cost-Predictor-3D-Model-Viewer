import math
import ezdxf
from floorplan_renderer._common import north_angle, dim_offset, entrance_steps, door_arc, rotate_point
from floorplan_generator.layout_engine import (
    GeneratedFloorPlan, RoomRect, WallSegment,
    DoorPlacement, WindowPlacement, DimLine
)

class DXFRenderer:
    """Renders floor plans to AutoCAD DXF format following AIA CAD layer standards."""

    def __init__(self):
        pass

    def render(self, plan: GeneratedFloorPlan, output_path: str):
        """Renders the given floor plan to a DXF file."""
        doc = ezdxf.new('R2010', setup=True)
        msp = doc.modelspace()
        
        # Setup layers
        self._setup_layers(doc)
        
        # Draw elements
        self._draw_walls(msp, plan.walls)
        self._draw_rooms(msp, plan.rooms)
        self._draw_doors(msp, plan.doors)
        self._draw_windows(msp, plan.windows)
        self._draw_dimensions(doc, msp, plan.dimensions)
        self._draw_title_block(msp, plan)
        self._draw_north_arrow(msp, plan)
        
        doc.saveas(output_path)
        
    def _setup_layers(self, doc):
        layers = [
            ("A-WALL-FULL", 2),   # Yellow
            ("A-DOOR-FULL", 3),   # Green
            ("A-GLAZ-FULL", 4),   # Cyan
            ("A-ANNO-DIMS", 1),   # Red
            ("A-ANNO-TEXT", 7),   # White
            ("A-TTLB-BORD", 7),   # White
        ]
        for name, color in layers:
            if name not in doc.layers:
                doc.layers.add(name, color=color)
            
    def _draw_walls(self, msp, walls):
        for w in walls:
            dx = w.x2 - w.x1
            dy = w.y2 - w.y1
            L = math.hypot(dx, dy)
            if L < 1e-5: continue
            nx = -dy / L * w.thickness / 2
            ny = dx / L * w.thickness / 2
            
            p1 = (w.x1 + nx, w.y1 + ny)
            p2 = (w.x2 + nx, w.y2 + ny)
            p3 = (w.x2 - nx, w.y2 - ny)
            p4 = (w.x1 - nx, w.y1 - ny)
            
            # outline
            msp.add_lwpolyline([p1, p2, p3, p4], close=True, dxfattribs={'layer': 'A-WALL-FULL'})
            # simple hatch ANSI31
            hatch = msp.add_hatch(color=2, dxfattribs={'layer': 'A-WALL-FULL'})
            hatch.set_pattern_fill('ANSI31', scale=0.05)
            hatch.paths.add_polyline_path([p1, p2, p3, p4], is_closed=True)
            
    def _draw_rooms(self, msp, rooms):
        for r in rooms:
            if r.w <= 0 or r.h <= 0: continue
            text = f"{r.name.upper()}\\P{r.w:.2f} x {r.h:.2f} m\\P{r.area:.1f} sq.m"
            msp.add_mtext(text, dxfattribs={
                'layer': 'A-ANNO-TEXT',
                'char_height': 0.18,
                'insert': (r.x + r.w/2, r.y + r.h/2),
                'attachment_point': 5 # middle center
            })

    def _draw_doors(self, msp, doors):
        for d in doors:
            # Swing arc centred on the hinge; the leaf opens into the owning room
            start_angle, end_angle, leaf = door_arc(d)
            msp.add_arc((d.x, d.y), radius=d.width, start_angle=start_angle, end_angle=end_angle, dxfattribs={'layer': 'A-DOOR-FULL'})
            msp.add_line((d.x, d.y), leaf, dxfattribs={'layer': 'A-DOOR-FULL'})

            # Entrance landing + steps outside the main door (any wall side)
            if d.is_main_entrance:
                land, lines, _, _ = entrance_steps(d)
                msp.add_lwpolyline(land, close=True, dxfattribs={'layer': 'A-DOOR-FULL'})
                for (p1, p2) in lines:
                    msp.add_line(p1, p2, dxfattribs={'layer': 'A-DOOR-FULL'})

    def _draw_windows(self, msp, windows):
        for w in windows:
            dx = w.width / 2 if w.wall_side in ('N', 'S') else 0
            dy = w.height / 2 if w.wall_side in ('E', 'W') else 0
            
            if w.wall_side in ('N', 'S'):
                msp.add_line((w.x - dx, w.y - 0.03), (w.x + dx, w.y - 0.03), dxfattribs={'layer': 'A-GLAZ-FULL'})
                msp.add_line((w.x - dx, w.y + 0.03), (w.x + dx, w.y + 0.03), dxfattribs={'layer': 'A-GLAZ-FULL'})
                # Sill line
                sill_offset = 0.08 if w.wall_side == 'N' else -0.08
                msp.add_line((w.x - dx - 0.05, w.y + sill_offset), (w.x + dx + 0.05, w.y + sill_offset), dxfattribs={'layer': 'A-GLAZ-FULL'})
            else:
                msp.add_line((w.x - 0.03, w.y - dy), (w.x - 0.03, w.y + dy), dxfattribs={'layer': 'A-GLAZ-FULL'})
                msp.add_line((w.x + 0.03, w.y - dy), (w.x + 0.03, w.y + dy), dxfattribs={'layer': 'A-GLAZ-FULL'})
                # Sill line
                sill_offset = 0.08 if w.wall_side == 'E' else -0.08
                msp.add_line((w.x + sill_offset, w.y - dy - 0.05), (w.x + sill_offset, w.y + dy + 0.05), dxfattribs={'layer': 'A-GLAZ-FULL'})

    def _draw_dimensions(self, doc, msp, dimensions):
        if 'ARCH_DIM' not in doc.dimstyles:
            dimstyle = doc.dimstyles.new('ARCH_DIM')
            dimstyle.dxf.dimasz = 0.15 
            dimstyle.dxf.dimblk = 'ARCHTICK' 
            dimstyle.dxf.dimtxt = 0.2
            
        for dim in dimensions:
            vertical = abs(dim.x2 - dim.x1) < abs(dim.y2 - dim.y1)
            off = dim_offset(dim)
            mx, my = (dim.x1 + dim.x2) / 2, (dim.y1 + dim.y2) / 2
            base = (mx + off, my) if vertical else (mx, my + off)
            msp.add_linear_dim(
                base=base,
                p1=(dim.x1, dim.y1),
                p2=(dim.x2, dim.y2),
                angle=90 if vertical else 0,
                dimstyle='ARCH_DIM',
                text=dim.label,
                dxfattribs={'layer': 'A-ANNO-DIMS'}
            ).render()

    def _draw_title_block(self, msp, plan):
        x, y = plan.plot_width + 1, 0
        msp.add_lwpolyline([(x, y), (x+8, y), (x+8, y+5), (x, y+5)], close=True, dxfattribs={'layer': 'A-TTLB-BORD'})
        msp.add_text(f"Project: {plan.project_name}", dxfattribs={'layer': 'A-ANNO-TEXT', 'height': 0.2}).set_placement((x+0.2, y+4))
        msp.add_text(f"Config: {plan.bhk_config}", dxfattribs={'layer': 'A-ANNO-TEXT', 'height': 0.2}).set_placement((x+0.2, y+3))
        msp.add_text(f"Quality: {plan.quality_level}", dxfattribs={'layer': 'A-ANNO-TEXT', 'height': 0.2}).set_placement((x+0.2, y+2))
        cents = (plan.total_built_up_area * 10.764) / 435.6
        msp.add_text(f"Built Area: {plan.total_built_up_area:.1f} sqm ({cents:.2f} Cents)", dxfattribs={'layer': 'A-ANNO-TEXT', 'height': 0.2}).set_placement((x+0.2, y+1))

    def _draw_north_arrow(self, msp, plan):
        cx, cy = plan.plot_width + 1, plan.plot_height - 0.5
        ang = north_angle(plan)
        rp = lambda px, py: rotate_point(px, py, cx, cy, ang)
        tip, left, right, notch = rp(cx, cy + 0.5), rp(cx - 0.5, cy - 0.5), rp(cx + 0.5, cy - 0.5), rp(cx, cy - 0.3)
        msp.add_lwpolyline([tip, left, notch], close=True, dxfattribs={'layer': 'A-ANNO-TEXT'})
        msp.add_lwpolyline([tip, right, notch], close=True, dxfattribs={'layer': 'A-ANNO-TEXT'})
        msp.add_hatch(color=7, dxfattribs={'layer': 'A-ANNO-TEXT'}).paths.add_polyline_path([tip, right, notch], is_closed=True)
        lx, ly = rp(cx, cy + 0.8)
        msp.add_text("N", dxfattribs={'layer': 'A-ANNO-TEXT', 'height': 0.3}).set_placement((lx - 0.1, ly))
