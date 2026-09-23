import math
import ezdxf
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
            if d.wall_side == 'N':
                start_angle = 0 if d.swing_direction == 'RIGHT' else 90
                end_angle = 90 if d.swing_direction == 'RIGHT' else 180
            elif d.wall_side == 'S':
                start_angle = 180 if d.swing_direction == 'RIGHT' else 270
                end_angle = 270 if d.swing_direction == 'RIGHT' else 360
            elif d.wall_side == 'E':
                start_angle = 270 if d.swing_direction == 'RIGHT' else 0
                end_angle = 360 if d.swing_direction == 'RIGHT' else 90
            else: # 'W'
                start_angle = 90 if d.swing_direction == 'RIGHT' else 180
                end_angle = 180 if d.swing_direction == 'RIGHT' else 270
            
            msp.add_arc((d.x, d.y), radius=d.width, start_angle=start_angle, end_angle=end_angle, dxfattribs={'layer': 'A-DOOR-FULL'})
            msp.add_line((d.x, d.y), (d.x + math.cos(math.radians(end_angle))*d.width, d.y + math.sin(math.radians(end_angle))*d.width), dxfattribs={'layer': 'A-DOOR-FULL'})

            # Entrance steps
            if d.is_main_entrance and d.wall_side == 'S':
                landing_d = 1.2
                step_d = 0.3
                lx1 = d.x - d.width/2 - 0.3
                lx2 = d.x + d.width/2 + 0.3
                msp.add_lwpolyline([(lx1, d.y), (lx2, d.y), (lx2, d.y - landing_d), (lx1, d.y - landing_d)], close=True, dxfattribs={'layer': 'A-DOOR-FULL'})
                for i in range(3):
                    sy = d.y - landing_d - (i + 1) * step_d
                    msp.add_line((lx1, sy), (lx2, sy), dxfattribs={'layer': 'A-DOOR-FULL'})
                msp.add_line((lx1, d.y - landing_d), (lx1, d.y - landing_d - 0.9), dxfattribs={'layer': 'A-DOOR-FULL'})
                msp.add_line((lx2, d.y - landing_d), (lx2, d.y - landing_d - 0.9), dxfattribs={'layer': 'A-DOOR-FULL'})

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
            msp.add_linear_dim(
                base=(dim.x1 + (dim.x2-dim.x1)/2, dim.y1 + (dim.y2-dim.y1)/2 + dim.offset),
                p1=(dim.x1, dim.y1),
                p2=(dim.x2, dim.y2),
                dimstyle='ARCH_DIM',
                text=dim.label,
                override={'layer': 'A-ANNO-DIMS'}
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
        x, y = plan.plot_width + 1, plan.plot_height
        msp.add_lwpolyline([(x, y), (x-0.5, y-1), (x, y-0.8)], close=True, dxfattribs={'layer': 'A-ANNO-TEXT'})
        msp.add_lwpolyline([(x, y), (x+0.5, y-1), (x, y-0.8)], close=True, dxfattribs={'layer': 'A-ANNO-TEXT'})
        msp.add_hatch(color=7, dxfattribs={'layer': 'A-ANNO-TEXT'}).paths.add_polyline_path([(x, y), (x+0.5, y-1), (x, y-0.8)], is_closed=True)
        msp.add_text("N", dxfattribs={'layer': 'A-ANNO-TEXT', 'height': 0.3}).set_placement((x-0.1, y+0.2))
