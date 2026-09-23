"""
Interactive Floor Plan Editor Component & Live CAD Studio
==========================================================
Architectural-grade 2D CAD blueprint editor and customizer for residential floor plans.
Enables architects, civil engineers, builders, and homeowners to:
  - View and interact with the EXACT CAD blueprint aesthetic (navy background, cyan double walls, 
    entrance flat steps, 90-degree door swing arcs, room labels with width x length and area)
  - Dynamically add components: Doors (entrance with flat steps, internal, balcony), 
    Entrance Steps (custom flat treads & landings), Windows (double-glazed), and Custom Rooms
  - Dynamically delete components: Delete any Room, Delete any Door, Remove Steps, Delete Windows
  - Dynamically increase or decrease room dimensions with instant meter-level live sync
  - Freehand sketch and drag/resize rooms directly on the canvas
  - Live Property HUD: real-time carpet area (m2, sq ft, Cents), built-up area, 
    live 9-zone Vastu Purusha Mandala score, and dynamic construction cost estimation
  - Apply custom revisions to instantly regenerate CAD DXF, vector SVG, PDF blueprint, and PNG exports
"""

import copy
import math
from typing import Dict, List, Optional, Tuple, Any
import streamlit as st

try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except Exception:
    st_canvas = None
    CANVAS_AVAILABLE = False

from floorplan_generator.layout_engine import (
    GeneratedFloorPlan, RoomRect, WallSegment,
    DoorPlacement, WindowPlacement, DimLine, LayoutEngine
)
from floorplan_generator.vastu_engine import VastuEngine, VastuZone
from model_folder.cost_engine import CostEngine
from floorplan_renderer.svg_renderer import SVGRenderer
from floorplan_renderer.png_renderer import PNGRenderer
from floorplan_renderer.dxf_renderer import DXFRenderer
from floorplan_renderer.pdf_renderer import PDFRenderer


PPM_EDITOR = 50  # 50 pixels per meter — MATCHES SVGRenderer PPM exactly
PAD_M = 3.0      # 3.0 meters margin — MATCHES SVGRenderer pad exactly


class FloorPlanEditor:
    """Manages interactive CAD blueprint drawing, component placement, and live-synchronization."""

    @staticmethod
    def plan_to_fabric_json(plan: GeneratedFloorPlan, canvas_w: int, canvas_h: int) -> Dict[str, Any]:
        """
        Converts a GeneratedFloorPlan into architectural Fabric.js objects format
        matching the exact blueprint visual standard (cyan double walls, door swing arcs,
        entrance landing & steps, room dimensions & area typography, scale, and compass).
        """
        objects = []

        def to_cx(xm: float) -> float:
            return round((xm + PAD_M) * PPM_EDITOR, 1)

        def to_cy(ym: float) -> float:
            return round(canvas_h - (ym + PAD_M) * PPM_EDITOR, 1)

        # -------------------------------------------------------------
        # 1. Outer Plot Boundary & Framing (Double Cyan CAD Lines)
        # -------------------------------------------------------------
        plot_left = to_cx(0)
        plot_top = to_cy(plan.plot_height)
        plot_w_px = plan.plot_width * PPM_EDITOR
        plot_h_px = plan.plot_height * PPM_EDITOR

        objects.append({
            "type": "rect",
            "version": "4.4.0",
            "originX": "left",
            "originY": "top",
            "left": plot_left - 8,
            "top": plot_top - 8,
            "width": plot_w_px + 16,
            "height": plot_h_px + 16,
            "fill": "transparent",
            "stroke": "#00FFFF",
            "strokeWidth": 2.5,
            "selectable": False,
            "evented": False,
        })
        objects.append({
            "type": "rect",
            "version": "4.4.0",
            "originX": "left",
            "originY": "top",
            "left": plot_left - 14,
            "top": plot_top - 14,
            "width": plot_w_px + 28,
            "height": plot_h_px + 28,
            "fill": "transparent",
            "stroke": "#1E3A8A",
            "strokeWidth": 1.2,
            "strokeDashArray": [8, 5],
            "selectable": False,
            "evented": False,
        })

        # -------------------------------------------------------------
        # 2. Architectural Wall Segments (Heavy & Medium Cyan)
        # -------------------------------------------------------------
        for w in plan.walls:
            w_stroke = "#00FFFF" if w.is_external else "#38BDF8"
            w_thickness = 3.0 if w.is_external else 2.0
            objects.append({
                "type": "line",
                "version": "4.4.0",
                "x1": to_cx(w.x1),
                "y1": to_cy(w.y1),
                "x2": to_cx(w.x2),
                "y2": to_cy(w.y2),
                "stroke": w_stroke,
                "strokeWidth": w_thickness,
                "selectable": False,
                "evented": False,
            })

        # -------------------------------------------------------------
        # 3. Interactive CAD Room Rectangles (Translucent Cyan Fill)
        # -------------------------------------------------------------
        for r in plan.rooms:
            if r.w <= 0 or r.h <= 0:
                continue

            rx = to_cx(r.x)
            ry = to_cy(r.y + r.h)
            rw = r.w * PPM_EDITOR
            rh = r.h * PPM_EDITOR

            objects.append({
                "type": "rect",
                "version": "4.4.0",
                "originX": "left",
                "originY": "top",
                "left": rx,
                "top": ry,
                "width": rw,
                "height": rh,
                "fill": "rgba(14, 116, 144, 0.22)",
                "stroke": "#00E5FF",
                "strokeWidth": 2.0,
                "strokeUniform": True,
                "scaleX": 1.0,
                "scaleY": 1.0,
                "angle": 0,
                "opacity": 0.85,
                "selectable": True,
                "hasControls": True,
                "room_name": r.name,
                "category": r.category,
                "color": r.color,
                "original_area": r.area,
            })

        # -------------------------------------------------------------
        # 4. Architectural Room Typography (Centered Name, Size, Area)
        # -------------------------------------------------------------
        abbrevs = {
            'Attached Bathroom': 'ATT. BATH',
            'Bathroom': 'BATH',
            'Balcony': 'BALC.',
            'Utility / Balcony': 'UTIL / BALC',
            'Utility': 'UTIL.',
            'Corridor': 'CORR.',
            'Store Room': 'STORE',
            'Pooja Room': 'POOJA',
            'Servant Quarter': 'SQ',
            'Living & Dining': 'LIVING & DINING',
            'Master Bedroom': 'MASTER BED',
            'Bedroom': 'BEDROOM',
        }

        for r in plan.rooms:
            if r.w <= 0 or r.h <= 0:
                continue

            cx = to_cx(r.cx)
            cy = to_cy(r.cy)

            # Compact abbreviation and dynamic font sizing for narrow rooms
            is_narrow = (r.w < 2.0 or r.h < 2.0 or r.area < 5.0)
            if is_narrow:
                display_name = abbrevs.get(r.name, r.name)
                f_size = 8.5
                text_content = f"{display_name.upper()}\n{r.w:.1f}×{r.h:.1f}m\n{r.area:.1f}m²"
            else:
                display_name = r.name
                f_size = 11 if (r.w >= 2.6 and r.h >= 2.6) else 9.5
                text_content = f"{display_name.upper()}\n{r.w:.2f} × {r.h:.2f} m\n{r.area:.1f} m²"

            objects.append({
                "type": "text",
                "version": "4.4.0",
                "originX": "center",
                "originY": "center",
                "left": cx,
                "top": cy,
                "text": text_content,
                "fontSize": f_size,
                "fontWeight": "bold",
                "fontFamily": "Segoe UI, Arial, sans-serif",
                "fill": "#FFFFFF",
                "textAlign": "center",
                "lineHeight": 1.2,
                "selectable": False,
                "evented": False,
                "shadow": {
                    "color": "rgba(0, 0, 0, 0.9)",
                    "blur": 4,
                    "offsetX": 1,
                    "offsetY": 1
                }
            })

        # -------------------------------------------------------------
        # 5. Architectural Windows (Double-Glazing & Sills)
        # -------------------------------------------------------------
        for win in plan.windows:
            wx_px = to_cx(win.x)
            wy_px = to_cy(win.y)
            w_len_px = win.width * PPM_EDITOR

            if win.wall_side in ('N', 'S'):
                # Horizontal window
                objects.append({
                    "type": "line",
                    "x1": wx_px - w_len_px / 2, "y1": wy_px - 2,
                    "x2": wx_px + w_len_px / 2, "y2": wy_px - 2,
                    "stroke": "#00BFFF", "strokeWidth": 1.5, "selectable": False, "evented": False
                })
                objects.append({
                    "type": "line",
                    "x1": wx_px - w_len_px / 2, "y1": wy_px + 2,
                    "x2": wx_px + w_len_px / 2, "y2": wy_px + 2,
                    "stroke": "#00BFFF", "strokeWidth": 1.5, "selectable": False, "evented": False
                })
                objects.append({
                    "type": "line",
                    "x1": wx_px - w_len_px / 2 - 4, "y1": wy_px + 4,
                    "x2": wx_px + w_len_px / 2 + 4, "y2": wy_px + 4,
                    "stroke": "#00AACC", "strokeWidth": 1.2, "selectable": False, "evented": False
                })
            else:
                # Vertical window (E / W)
                objects.append({
                    "type": "line",
                    "x1": wx_px - 2, "y1": wy_px - w_len_px / 2,
                    "x2": wx_px - 2, "y2": wy_px + w_len_px / 2,
                    "stroke": "#00BFFF", "strokeWidth": 1.5, "selectable": False, "evented": False
                })
                objects.append({
                    "type": "line",
                    "x1": wx_px + 2, "y1": wy_px - w_len_px / 2,
                    "x2": wx_px + 2, "y2": wy_px + w_len_px / 2,
                    "stroke": "#00BFFF", "strokeWidth": 1.5, "selectable": False, "evented": False
                })
                objects.append({
                    "type": "line",
                    "x1": wx_px + 4, "y1": wy_px - w_len_px / 2 - 4,
                    "x2": wx_px + 4, "y2": wy_px + w_len_px / 2 + 4,
                    "stroke": "#00AACC", "strokeWidth": 1.2, "selectable": False, "evented": False
                })

        # -------------------------------------------------------------
        # 6. Architectural Doors (90° Swing Arc, Leaf, & Threshold)
        # -------------------------------------------------------------
        for d in plan.doors:
            hx = to_cx(d.x)
            hy = to_cy(d.y)
            r_px = d.width * PPM_EDITOR
            side = d.wall_side
            swing = d.swing_direction

            # Draw threshold opening
            if side in ('N', 'S'):
                thresh_x2 = hx + r_px if swing == 'RIGHT' else hx - r_px
                objects.append({
                    "type": "line",
                    "x1": hx, "y1": hy, "x2": thresh_x2, "y2": hy,
                    "stroke": "#00FF88", "strokeWidth": 3.0, "selectable": False, "evented": False
                })
                leaf_y = hy + r_px if side == 'N' else hy - r_px
                objects.append({
                    "type": "line",
                    "x1": hx, "y1": hy, "x2": hx, "y2": leaf_y,
                    "stroke": "#00FF88", "strokeWidth": 2.0, "selectable": False, "evented": False
                })
                sweep_flag = "1" if (side == 'N' and swing == 'RIGHT') or (side == 'S' and swing == 'LEFT') else "0"
                objects.append({
                    "type": "path",
                    "path": f"M {thresh_x2} {hy} A {r_px} {r_px} 0 0 {sweep_flag} {hx} {leaf_y}",
                    "stroke": "#4A9E7F", "strokeWidth": 1.2, "strokeDashArray": [3, 3],
                    "fill": "transparent", "selectable": False, "evented": False
                })
            else:
                thresh_y2 = hy + r_px if swing == 'RIGHT' else hy - r_px
                objects.append({
                    "type": "line",
                    "x1": hx, "y1": hy, "x2": hx, "y2": thresh_y2,
                    "stroke": "#00FF88", "strokeWidth": 3.0, "selectable": False, "evented": False
                })
                leaf_x = hx - r_px if side == 'E' else hx + r_px
                objects.append({
                    "type": "line",
                    "x1": hx, "y1": hy, "x2": leaf_x, "y2": hy,
                    "stroke": "#00FF88", "strokeWidth": 2.0, "selectable": False, "evented": False
                })
                sweep_flag = "1" if (side == 'E' and swing == 'RIGHT') or (side == 'W' and swing == 'LEFT') else "0"
                objects.append({
                    "type": "path",
                    "path": f"M {hx} {thresh_y2} A {r_px} {r_px} 0 0 {sweep_flag} {leaf_x} {hy}",
                    "stroke": "#4A9E7F", "strokeWidth": 1.2, "strokeDashArray": [3, 3],
                    "fill": "transparent", "selectable": False, "evented": False
                })

            # Door interactive handle circle
            objects.append({
                "type": "circle",
                "originX": "center",
                "originY": "center",
                "left": hx,
                "top": hy,
                "radius": 7,
                "fill": "#00FF88",
                "stroke": "#064E3B",
                "strokeWidth": 2.0,
                "opacity": 0.95,
                "selectable": True,
                "door_for": d.room_name,
                "is_main_entrance": d.is_main_entrance,
                "wall_side": d.wall_side,
            })

            # -------------------------------------------------------------
            # 7. Entrance Flat Steps & Landing (for Main Entrance Door)
            # -------------------------------------------------------------
            if d.is_main_entrance:
                landing_depth_px = 1.2 * PPM_EDITOR
                step_depth_px = 0.3 * PPM_EDITOR
                side_ext_px = 0.3 * PPM_EDITOR
                num_steps = 3

                if side == 'E':
                    land_l = hx
                    land_t = hy - (r_px / 2 + side_ext_px)
                    land_w = landing_depth_px
                    land_h = r_px + side_ext_px * 2

                    objects.append({
                        "type": "rect",
                        "left": land_l, "top": land_t, "width": land_w, "height": land_h,
                        "fill": "rgba(13, 43, 69, 0.85)", "stroke": "#00DDDD", "strokeWidth": 2.0,
                        "selectable": False, "evented": False
                    })
                    objects.append({
                        "type": "text",
                        "originX": "center", "originY": "center",
                        "left": land_l + land_w / 2, "top": land_t + land_h / 2,
                        "text": "LANDING", "fontSize": 8.5, "fill": "#8EB8E5",
                        "selectable": False, "evented": False
                    })

                    for i in range(1, num_steps + 1):
                        sx = land_l + land_w + i * step_depth_px
                        objects.append({
                            "type": "line",
                            "x1": sx, "y1": land_t, "x2": sx, "y2": land_t + land_h,
                            "stroke": "#00DDDD", "strokeWidth": 2.0, "selectable": False, "evented": False
                        })

                    tot_step_w = num_steps * step_depth_px
                    objects.append({
                        "type": "line",
                        "x1": land_l + land_w, "y1": land_t, "x2": land_l + land_w + tot_step_w, "y2": land_t,
                        "stroke": "#00DDDD", "strokeWidth": 2.0, "selectable": False, "evented": False
                    })
                    objects.append({
                        "type": "line",
                        "x1": land_l + land_w, "y1": land_t + land_h, "x2": land_l + land_w + tot_step_w, "y2": land_t + land_h,
                        "stroke": "#00DDDD", "strokeWidth": 2.0, "selectable": False, "evented": False
                    })
                    objects.append({
                        "type": "line",
                        "x1": land_l + land_w + tot_step_w, "y1": land_t, "x2": land_l + land_w + tot_step_w, "y2": land_t + land_h,
                        "stroke": "#00DDDD", "strokeWidth": 2.0, "selectable": False, "evented": False
                    })
                    objects.append({
                        "type": "text",
                        "originX": "center", "originY": "center",
                        "left": land_l + land_w + tot_step_w / 2, "top": land_t - 14,
                        "text": "◄ UP", "fontSize": 10, "fontWeight": "bold", "fill": "#00FFCC",
                        "selectable": False, "evented": False
                    })

                elif side == 'S':
                    land_l = hx - (r_px / 2 + side_ext_px)
                    land_t = hy
                    land_w = r_px + side_ext_px * 2
                    land_h = landing_depth_px

                    objects.append({
                        "type": "rect",
                        "left": land_l, "top": land_t, "width": land_w, "height": land_h,
                        "fill": "rgba(13, 43, 69, 0.85)", "stroke": "#00DDDD", "strokeWidth": 2.0,
                        "selectable": False, "evented": False
                    })
                    for i in range(1, num_steps + 1):
                        sy = land_t + land_h + i * step_depth_px
                        objects.append({
                            "type": "line",
                            "x1": land_l, "y1": sy, "x2": land_l + land_w, "y2": sy,
                            "stroke": "#00DDDD", "strokeWidth": 2.0, "selectable": False, "evented": False
                        })
                    tot_step_h = num_steps * step_depth_px
                    objects.append({
                        "type": "line",
                        "x1": land_l, "y1": land_t + land_h, "x2": land_l, "y2": land_t + land_h + tot_step_h,
                        "stroke": "#00DDDD", "strokeWidth": 2.0, "selectable": False, "evented": False
                    })
                    objects.append({
                        "type": "line",
                        "x1": land_l + land_w, "y1": land_t + land_h, "x2": land_l + land_w, "y2": land_t + land_h + tot_step_h,
                        "stroke": "#00DDDD", "strokeWidth": 2.0, "selectable": False, "evented": False
                    })
                    objects.append({
                        "type": "line",
                        "x1": land_l, "y1": land_t + land_h + tot_step_h, "x2": land_l + land_w, "y2": land_t + land_h + tot_step_h,
                        "stroke": "#00DDDD", "strokeWidth": 2.0, "selectable": False, "evented": False
                    })
                    objects.append({
                        "type": "text",
                        "originX": "center", "originY": "center",
                        "left": land_l + land_w + 16, "top": land_t + land_h + tot_step_h / 2,
                        "text": "▲ UP", "fontSize": 10, "fontWeight": "bold", "fill": "#00FFCC",
                        "selectable": False, "evented": False
                    })

                elif side == 'W':
                    land_w = landing_depth_px
                    land_h = r_px + side_ext_px * 2
                    land_l = hx - land_w
                    land_t = hy - land_h / 2

                    objects.append({
                        "type": "rect",
                        "left": land_l, "top": land_t, "width": land_w, "height": land_h,
                        "fill": "rgba(13, 43, 69, 0.85)", "stroke": "#00DDDD", "strokeWidth": 2.0,
                        "selectable": False, "evented": False
                    })
                    for i in range(1, num_steps + 1):
                        sx = land_l - i * step_depth_px
                        objects.append({
                            "type": "line",
                            "x1": sx, "y1": land_t, "x2": sx, "y2": land_t + land_h,
                            "stroke": "#00DDDD", "strokeWidth": 2.0, "selectable": False, "evented": False
                        })
                    tot_step_w = num_steps * step_depth_px
                    objects.append({
                        "type": "line",
                        "x1": land_l, "y1": land_t, "x2": land_l - tot_step_w, "y2": land_t,
                        "stroke": "#00DDDD", "strokeWidth": 2.0, "selectable": False, "evented": False
                    })
                    objects.append({
                        "type": "line",
                        "x1": land_l, "y1": land_t + land_h, "x2": land_l - tot_step_w, "y2": land_t + land_h,
                        "stroke": "#00DDDD", "strokeWidth": 2.0, "selectable": False, "evented": False
                    })
                    objects.append({
                        "type": "line",
                        "x1": land_l - tot_step_w, "y1": land_t, "x2": land_l - tot_step_w, "y2": land_t + land_h,
                        "stroke": "#00DDDD", "strokeWidth": 2.0, "selectable": False, "evented": False
                    })
                    objects.append({
                        "type": "text",
                        "originX": "center", "originY": "center",
                        "left": land_l - tot_step_w / 2, "top": land_t - 14,
                        "text": "UP ►", "fontSize": 10, "fontWeight": "bold", "fill": "#00FFCC",
                        "selectable": False, "evented": False
                    })

                elif side == 'N':
                    land_l = hx - (r_px / 2 + side_ext_px)
                    land_w = r_px + side_ext_px * 2
                    land_h = landing_depth_px
                    land_t = hy - land_h

                    objects.append({
                        "type": "rect",
                        "left": land_l, "top": land_t, "width": land_w, "height": land_h,
                        "fill": "rgba(13, 43, 69, 0.85)", "stroke": "#00DDDD", "strokeWidth": 2.0,
                        "selectable": False, "evented": False
                    })
                    for i in range(1, num_steps + 1):
                        sy = land_t - i * step_depth_px
                        objects.append({
                            "type": "line",
                            "x1": land_l, "y1": sy, "x2": land_l + land_w, "y2": sy,
                            "stroke": "#00DDDD", "strokeWidth": 2.0, "selectable": False, "evented": False
                        })
                    tot_step_h = num_steps * step_depth_px
                    objects.append({
                        "type": "line",
                        "x1": land_l, "y1": land_t, "x2": land_l, "y2": land_t - tot_step_h,
                        "stroke": "#00DDDD", "strokeWidth": 2.0, "selectable": False, "evented": False
                    })
                    objects.append({
                        "type": "line",
                        "x1": land_l + land_w, "y1": land_t, "x2": land_l + land_w, "y2": land_t - tot_step_h,
                        "stroke": "#00DDDD", "strokeWidth": 2.0, "selectable": False, "evented": False
                    })
                    objects.append({
                        "type": "line",
                        "x1": land_l, "y1": land_t - tot_step_h, "x2": land_l + land_w, "y2": land_t - tot_step_h,
                        "stroke": "#00DDDD", "strokeWidth": 2.0, "selectable": False, "evented": False
                    })
                    objects.append({
                        "type": "text",
                        "originX": "center", "originY": "center",
                        "left": land_l + land_w + 16, "top": land_t - tot_step_h / 2,
                        "text": "▼ UP", "fontSize": 10, "fontWeight": "bold", "fill": "#00FFCC",
                        "selectable": False, "evented": False
                    })

        # -------------------------------------------------------------
        # 8. Title Block (Right side — matching SVGRenderer layout)
        # -------------------------------------------------------------
        # SVGRenderer places title block at x = plot_width*PPM + pad*PPM + 5px gap
        title_x = to_cx(plan.plot_width) + 20   # right of plot + 20px gap
        title_y = plot_top + 10
        title_w = 5.0 * PPM_EDITOR - 30         # ~5m wide title area
        title_h = min(plot_h_px - 20, 250)

        # Outer title block border
        objects.append({
            "type": "rect",
            "left": title_x, "top": title_y,
            "width": title_w, "height": title_h,
            "fill": "rgba(10, 35, 66, 0.90)", "stroke": "#00CCCC", "strokeWidth": 1.5,
            "selectable": False, "evented": False
        })

        # 5 rows inside title block
        row_h = title_h / 5
        cents = (plan.total_built_up_area * 10.764) / 435.6
        tb_labels = [
            f"PROJECT: {plan.project_name}",
            f"TYPE: {plan.bhk_config}",
            f"AREA: {plan.total_built_up_area:.1f}m² ({cents:.2f}C)",
            f"QUALITY: {getattr(plan, 'quality_level', 'Standard')}",
            f"VASTU: {plan.vastu_score:.0f}/100 | SCALE: 1:50",
        ]
        for idx, label_text in enumerate(reversed(tb_labels)):
            row_y = title_y + (idx + 0.05) * row_h
            # Row separator line
            if idx > 0:
                objects.append({
                    "type": "line",
                    "x1": title_x, "y1": row_y, "x2": title_x + title_w, "y2": row_y,
                    "stroke": "#00CCCC", "strokeWidth": 0.8, "selectable": False, "evented": False
                })
            objects.append({
                "type": "text",
                "originX": "left", "originY": "top",
                "left": title_x + 6, "top": row_y + 6,
                "text": label_text,
                "fontSize": 9, "fontWeight": "bold" if idx == 4 else "normal",
                "fontFamily": "Segoe UI, Arial, sans-serif",
                "fill": "#E0F0FF",
                "selectable": False, "evented": False
            })

        # North compass rosette — positioned at top of title block area
        comp_x = title_x + title_w / 2
        comp_y = title_y + title_h + 50
        if comp_y + 40 < canvas_h:
            objects.append({
                "type": "circle",
                "originX": "center", "originY": "center",
                "left": comp_x, "top": comp_y, "radius": 24,
                "fill": "rgba(10, 35, 66, 0.75)", "stroke": "#00DDAA", "strokeWidth": 1.5,
                "selectable": False, "evented": False
            })
            objects.append({
                "type": "text",
                "originX": "center", "originY": "center",
                "left": comp_x, "top": comp_y - 4,
                "text": "▲\nN",
                "fontSize": 12, "fontWeight": "bold", "fill": "#00FFCC",
                "textAlign": "center", "lineHeight": 0.9,
                "selectable": False, "evented": False
            })
            for label, dx, dy in [("W", -32, 0), ("E", 32, 0), ("S", 0, 32)]:
                objects.append({
                    "type": "text", "originX": "center", "originY": "center",
                    "left": comp_x + dx, "top": comp_y + dy, "text": label,
                    "fontSize": 9, "fill": "#8EB8E5",
                    "selectable": False, "evented": False
                })

        # -------------------------------------------------------------
        # 9. Exterior Dimension Lines & Tick Annotations
        # -------------------------------------------------------------
        # Top dimension line (plot width in mm)
        dim_y = plot_top - 28
        objects.append({
            "type": "line",
            "x1": plot_left, "y1": dim_y, "x2": plot_left + plot_w_px, "y2": dim_y,
            "stroke": "#6B8DAF", "strokeWidth": 1.0, "selectable": False, "evented": False
        })
        # Tick marks (45°)
        for tick_x in [plot_left, plot_left + plot_w_px]:
            objects.append({
                "type": "line",
                "x1": tick_x - 5, "y1": dim_y + 5, "x2": tick_x + 5, "y2": dim_y - 5,
                "stroke": "#6B8DAF", "strokeWidth": 1.5, "selectable": False, "evented": False
            })
        objects.append({
            "type": "text",
            "originX": "center", "originY": "center",
            "left": plot_left + plot_w_px / 2, "top": dim_y - 12,
            "text": f"{int(round(plan.plot_width * 1000))}",
            "fontSize": 9, "fill": "#8EB8E5", "selectable": False, "evented": False
        })

        # Left dimension line (plot height in mm)
        dim_x = plot_left - 32
        objects.append({
            "type": "line",
            "x1": dim_x, "y1": plot_top, "x2": dim_x, "y2": plot_top + plot_h_px,
            "stroke": "#6B8DAF", "strokeWidth": 1.0, "selectable": False, "evented": False
        })
        for tick_y in [plot_top, plot_top + plot_h_px]:
            objects.append({
                "type": "line",
                "x1": dim_x - 5, "y1": tick_y + 5, "x2": dim_x + 5, "y2": tick_y - 5,
                "stroke": "#6B8DAF", "strokeWidth": 1.5, "selectable": False, "evented": False
            })
        objects.append({
            "type": "text",
            "originX": "center", "originY": "center",
            "left": dim_x - 16, "top": plot_top + plot_h_px / 2,
            "text": f"{int(round(plan.plot_height * 1000))}",
            "fontSize": 9, "fill": "#8EB8E5", "angle": -90, "selectable": False, "evented": False
        })

        return {
            "version": "4.4.0",
            "objects": objects
        }

    @staticmethod
    def extract_edited_rooms(json_data: Dict[str, Any], canvas_h: int) -> List[Dict[str, Any]]:
        """
        Parses Fabric.js json_data from the canvas and extracts updated room coordinates & areas.
        """
        if not json_data or "objects" not in json_data:
            return []

        updated_rooms = []
        for obj in json_data["objects"]:
            if obj.get("type") == "rect" and "room_name" in obj:
                scale_x = obj.get("scaleX", 1.0)
                scale_y = obj.get("scaleY", 1.0)
                raw_w = obj.get("width", 0) * scale_x
                raw_h = obj.get("height", 0) * scale_y
                raw_left = obj.get("left", 0)
                raw_top = obj.get("top", 0)

                # Convert px back to meters
                w_m = max(1.0, round(raw_w / PPM_EDITOR, 2))
                h_m = max(1.0, round(raw_h / PPM_EDITOR, 2))
                x_m = max(0.0, round((raw_left / PPM_EDITOR) - PAD_M, 2))
                y_m = max(0.0, round(((canvas_h - (raw_top + raw_h)) / PPM_EDITOR) - PAD_M, 2))
                area_m2 = round(w_m * h_m, 2)

                updated_rooms.append({
                    "name": obj.get("room_name", "Room"),
                    "category": obj.get("category", "BEDROOM"),
                    "x": x_m,
                    "y": y_m,
                    "w": w_m,
                    "h": h_m,
                    "area": area_m2,
                    "color": obj.get("color", obj.get("fill", "#94A3B8")),
                })

        return updated_rooms

    @staticmethod
    def render_editor_section(current_plan: GeneratedFloorPlan, cost_params: Dict[str, Any]):
        """
        Renders the complete interactive CAD blueprint editor with:
          - Identical blueprint aesthetic matching Screenshot 2
          - Component Manager: Add Components (Doors, Steps, Windows, Rooms, Columns)
          - Component Manager: Delete Components (Delete Room, Delete Door, Remove Steps, Delete Window)
          - Live Room Dimension Stepper (increasing/decreasing)
          - Live Property HUD & dynamic side meters calculations
        """
        if not CANVAS_AVAILABLE:
            st.warning("⚠️ `streamlit-drawable-canvas` is not available. Please install it to use the interactive editor.")
            return

        # Initialize working copy of plan in session state
        if 'editor_working_plan' not in st.session_state or st.session_state.get('editor_base_plan_id') != id(current_plan):
            st.session_state['editor_working_plan'] = copy.deepcopy(current_plan)
            st.session_state['editor_base_plan_id'] = id(current_plan)
            st.session_state.pop('editor_initial_json', None)

        working_plan: GeneratedFloorPlan = st.session_state['editor_working_plan']

        pad_m = PAD_M
        # SVGRenderer uses: width_m = plot_width + pad*2 + 5.0 (title block area)
        # We match those dimensions exactly so the editor looks identical to the static blueprint.
        canvas_w = int((working_plan.plot_width + pad_m * 2 + 5.0) * PPM_EDITOR)
        canvas_h = int((working_plan.plot_height + pad_m * 2) * PPM_EDITOR)
        canvas_w = max(700, min(1400, canvas_w))
        canvas_h = max(500, min(1000, canvas_h))

        st.markdown("### ✏️ Interactive Floor Plan Studio & Live Customizer")
        st.caption("Live CAD blueprint editing playground: Add components, delete components, increase/decrease meters, and watch all metrics sync in real-time.")

        # -------------------------------------------------------------
        # Interactive Controls Bar: Add, Delete & Adjust Dimensions
        # -------------------------------------------------------------
        ctrl_tab_add, ctrl_tab_del, ctrl_tab_dim = st.tabs([
            "➕ Add Components",
            "🗑️ Delete Components",
            "📐 Increase / Decrease Dimensions"
        ])

        # -------------------------------------------------------------
        # TAB 1: ADD COMPONENTS
        # -------------------------------------------------------------
        with ctrl_tab_add:
            comp_type = st.radio(
                "Choose Component to Place:",
                ["🚪 Door (with Swing / Steps)", "🪜 Entrance Steps", "🪟 Window", "🏠 New Room", "🏛️ Column / Pillar"],
                horizontal=True,
                key="add_comp_radio"
            )
            room_names = [r.name for r in working_plan.rooms]

            if comp_type == "🚪 Door (with Swing / Steps)":
                d_c1, d_c2, d_c3, d_c4 = st.columns(4)
                with d_c1:
                    d_room = st.selectbox("Assign to Room", room_names if room_names else ["Default"], key="door_room_sel")
                with d_c2:
                    d_wall = st.selectbox("Wall Placement", ["E (East)", "N (North)", "W (West)", "S (South)"], key="door_wall_sel")
                    d_side = d_wall[0]
                with d_c3:
                    d_width = st.selectbox("Door Aperture", [1.0, 1.2, 0.9, 0.8], format_func=lambda w: f"{w:.1f} m ({w*3.28:.1f} ft)", key="door_w_sel")
                with d_c4:
                    is_main = st.checkbox("Main Entrance (+ Flat Steps)", value=False, key="door_is_main_sel")
                    d_swing = st.selectbox("Swing Direction", ["RIGHT", "LEFT"], key="door_swing_sel")

                if st.button("🚪 Place Door on Blueprint", type="secondary", use_container_width=True):
                    if working_plan.rooms:
                        target_r = next((r for r in working_plan.rooms if r.name == d_room), working_plan.rooms[0])
                        if d_side == 'E':
                            hx, hy = target_r.right, target_r.y + target_r.h / 2
                        elif d_side == 'W':
                            hx, hy = target_r.x, target_r.y + target_r.h / 2
                        elif d_side == 'N':
                            hx, hy = target_r.x + target_r.w / 2, target_r.top
                        else:
                            hx, hy = target_r.x + target_r.w / 2, target_r.y

                        if is_main:
                            for existing_d in working_plan.doors:
                                existing_d.is_main_entrance = False

                        new_door = DoorPlacement(
                            x=round(hx, 2),
                            y=round(hy, 2),
                            width=d_width,
                            wall_side=d_side,
                            swing_direction=d_swing,
                            room_name=d_room,
                            is_main_entrance=is_main
                        )
                        working_plan.doors.append(new_door)
                        st.session_state.pop('editor_initial_json', None)
                        st.success(f"✅ Added {d_width}m door on {d_side} wall of {d_room}!")
                        st.rerun()

            elif comp_type == "🪜 Entrance Steps":
                s_c1, s_c2, s_c3 = st.columns(3)
                with s_c1:
                    s_wall = st.selectbox("Entrance Wall Side", ["E (East)", "N (North)", "W (West)", "S (South)"], key="steps_wall_sel")
                    step_side = s_wall[0]
                with s_c2:
                    s_treads = st.selectbox("Number of Flat Treads", [3, 4, 5, 6], index=0, key="steps_treads_sel")
                with s_c3:
                    s_landing = st.selectbox("Landing Depth", [1.2, 1.5, 1.8], format_func=lambda d: f"{d:.1f} m", key="steps_landing_sel")

                if st.button("🪜 Add / Update Entrance Steps", type="secondary", use_container_width=True):
                    found = False
                    for d in working_plan.doors:
                        if d.wall_side == step_side:
                            d.is_main_entrance = True
                            found = True
                            break
                    if not found and working_plan.rooms:
                        first_r = working_plan.rooms[0]
                        if step_side == 'E':
                            hx, hy = first_r.right, first_r.y + first_r.h / 2
                        elif step_side == 'W':
                            hx, hy = first_r.x, first_r.y + first_r.h / 2
                        elif step_side == 'N':
                            hx, hy = first_r.x + first_r.w / 2, first_r.top
                        else:
                            hx, hy = first_r.x + first_r.w / 2, first_r.y

                        working_plan.doors.append(DoorPlacement(
                            x=round(hx, 2), y=round(hy, 2), width=1.0,
                            wall_side=step_side, swing_direction="RIGHT",
                            room_name=first_r.name, is_main_entrance=True
                        ))
                    st.session_state.pop('editor_initial_json', None)
                    st.success(f"✅ Entrance steps updated on {step_side} wall with {s_treads} flat treads!")
                    st.rerun()

            elif comp_type == "🪟 Window":
                w_c1, w_c2, w_c3 = st.columns(3)
                with w_c1:
                    w_room = st.selectbox("Assign Window to Room", room_names if room_names else ["Default"], key="win_room_sel")
                with w_c2:
                    w_wall = st.selectbox("Wall Side", ["N (North)", "E (East)", "S (South)", "W (West)"], key="win_wall_sel")
                    win_side = w_wall[0]
                with w_c3:
                    win_w = st.selectbox("Window Width", [1.2, 1.5, 1.8, 1.0, 2.0], format_func=lambda w: f"{w:.1f} m ({w*3.28:.1f} ft)", key="win_w_sel")

                if st.button("🪟 Place Window on Blueprint", type="secondary", use_container_width=True):
                    if working_plan.rooms:
                        target_r = next((r for r in working_plan.rooms if r.name == w_room), working_plan.rooms[0])
                        if win_side == 'N':
                            wx, wy = target_r.x + target_r.w / 2, target_r.top
                        elif win_side == 'S':
                            wx, wy = target_r.x + target_r.w / 2, target_r.y
                        elif win_side == 'E':
                            wx, wy = target_r.right, target_r.y + target_r.h / 2
                        else:
                            wx, wy = target_r.x, target_r.y + target_r.h / 2

                        working_plan.windows.append(WindowPlacement(
                            x=round(wx, 2), y=round(wy, 2), width=win_w, height=1.2,
                            wall_side=win_side, room_name=w_room
                        ))
                        st.session_state.pop('editor_initial_json', None)
                        st.success(f"✅ Added {win_w}m window on {win_side} wall of {w_room}!")
                        st.rerun()

            elif comp_type == "🏠 New Room":
                r_c1, r_c2, r_c3, r_c4 = st.columns(4)
                with r_c1:
                    new_r_name = st.text_input("Room Title", value="Pooja Room", key="new_r_name_input")
                with r_c2:
                    new_r_cat = st.selectbox("Category", ["POOJA", "BEDROOM", "LIVING", "KITCHEN", "BATHROOM", "BALCONY", "DINING", "STORE"], key="new_r_cat_sel")
                with r_c3:
                    new_r_w = st.number_input("Width (m)", min_value=1.2, max_value=8.0, value=2.4, step=0.1, key="new_r_w_input")
                with r_c4:
                    new_r_h = st.number_input("Length (m)", min_value=1.2, max_value=8.0, value=2.4, step=0.1, key="new_r_h_input")

                if st.button("🏠 Place Room on Blueprint", type="secondary", use_container_width=True):
                    new_rx = round(working_plan.plot_width * 0.1, 2)
                    new_ry = round(working_plan.plot_height * 0.1, 2)
                    working_plan.rooms.append(RoomRect(
                        name=new_r_name,
                        category=new_r_cat,
                        x=new_rx, y=new_ry,
                        w=round(new_r_w, 2), h=round(new_r_h, 2),
                        color="#0284C7",
                        area=round(new_r_w * new_r_h, 2)
                    ))
                    # Rebuild walls
                    try:
                        eng = LayoutEngine(
                            plot_width=working_plan.plot_width, plot_height=working_plan.plot_height,
                            bhk_config=working_plan.bhk_config or "auto", vastu_enabled=working_plan.vastu_enabled,
                            orientation=working_plan.orientation
                        )
                        working_plan.walls = eng._generate_walls(working_plan.rooms)
                        working_plan.dimensions = eng._generate_dimensions(working_plan.rooms)
                    except Exception:
                        pass
                    st.session_state.pop('editor_initial_json', None)
                    st.success(f"✅ Added room '{new_r_name}' ({new_r_w:.2f} × {new_r_h:.2f} m)!")
                    st.rerun()

            elif comp_type == "🏛️ Column / Pillar":
                col_c1, col_c2, col_c3 = st.columns(3)
                with col_c1:
                    col_name = st.text_input("Column Tag", value="RCC Column C1", key="col_name_input")
                with col_c2:
                    col_size = st.selectbox("Column Size", [0.3, 0.4, 0.23], format_func=lambda s: f"{s*1000:.0f} mm ({s*39.37:.0f} in)", key="col_size_sel")
                with col_c3:
                    st.write("")
                    st.write("")
                    if st.button("🏛️ Place Column on Blueprint", type="secondary", use_container_width=True):
                        working_plan.rooms.append(RoomRect(
                            name=col_name,
                            category="STRUCTURE",
                            x=round(working_plan.plot_width * 0.5, 2),
                            y=round(working_plan.plot_height * 0.5, 2),
                            w=col_size, h=col_size,
                            color="#38BDF8", area=round(col_size * col_size, 2)
                        ))
                        st.session_state.pop('editor_initial_json', None)
                        st.success(f"✅ Added structural column '{col_name}'!")
                        st.rerun()

        # -------------------------------------------------------------
        # TAB 2: DELETE COMPONENTS
        # -------------------------------------------------------------
        with ctrl_tab_del:
            del_category = st.radio(
                "Choose Component to Delete:",
                ["🏠 Delete Room", "🚪 Delete Door", "🪜 Remove Entrance Steps", "🪟 Delete Window"],
                horizontal=True,
                key="del_comp_radio"
            )

            if del_category == "🏠 Delete Room":
                if working_plan.rooms:
                    del_room_options = [f"{r.name} ({r.w:.2f}×{r.h:.2f}m - {r.area:.1f} m²)" for r in working_plan.rooms]
                    del_sel = st.selectbox("Select Room to Remove", del_room_options, key="del_room_dropdown")
                    target_del_name = working_plan.rooms[del_room_options.index(del_sel)].name

                    if st.button(f"🗑️ Delete '{target_del_name}'", type="primary", use_container_width=True):
                        working_plan.rooms = [r for r in working_plan.rooms if r.name != target_del_name]
                        working_plan.doors = [d for d in working_plan.doors if d.room_name != target_del_name]
                        working_plan.windows = [w for w in working_plan.windows if w.room_name != target_del_name]
                        try:
                            eng = LayoutEngine(
                                plot_width=working_plan.plot_width,
                                plot_height=working_plan.plot_height,
                                bhk_config=working_plan.bhk_config or "auto",
                                vastu_enabled=working_plan.vastu_enabled,
                                orientation=working_plan.orientation
                            )
                            working_plan.walls = eng._generate_walls(working_plan.rooms)
                            working_plan.dimensions = eng._generate_dimensions(working_plan.rooms)
                        except Exception:
                            pass
                        st.session_state.pop('editor_initial_json', None)
                        st.success(f"🗑️ Successfully deleted '{target_del_name}' and updated all CAD walls!")
                        st.rerun()
                else:
                    st.info("No rooms to delete.")

            elif del_category == "🚪 Delete Door":
                if working_plan.doors:
                    del_door_options = [
                        f"Door {i+1}: {d.room_name} ({d.wall_side} Wall, {d.width:.1f}m{' [MAIN ENTRANCE + STEPS]' if d.is_main_entrance else ''})"
                        for i, d in enumerate(working_plan.doors)
                    ]
                    del_d_sel = st.selectbox("Select Door to Remove", del_door_options, key="del_door_dropdown")
                    del_d_idx = del_door_options.index(del_d_sel)

                    if st.button("🗑️ Delete Selected Door", type="primary", use_container_width=True):
                        removed_door = working_plan.doors.pop(del_d_idx)
                        st.session_state.pop('editor_initial_json', None)
                        st.success(f"🗑️ Removed door on {removed_door.wall_side} wall of {removed_door.room_name}!")
                        st.rerun()
                else:
                    st.info("No doors on the floor plan.")

            elif del_category == "🪜 Remove Entrance Steps":
                has_entrance = any(d.is_main_entrance for d in working_plan.doors)
                if has_entrance:
                    st.warning("Clicking below will remove the entrance flat steps and landing from the main entrance.")
                    if st.button("🗑️ Remove Entrance Steps", type="primary", use_container_width=True):
                        for d in working_plan.doors:
                            d.is_main_entrance = False
                        st.session_state.pop('editor_initial_json', None)
                        st.success("🗑️ Entrance steps removed!")
                        st.rerun()
                else:
                    st.info("No entrance steps currently active on the blueprint.")

            elif del_category == "🪟 Delete Window":
                if working_plan.windows:
                    del_win_options = [
                        f"Window {i+1}: {w.room_name} ({w.wall_side} Wall, {w.width:.1f}m)"
                        for i, w in enumerate(working_plan.windows)
                    ]
                    del_w_sel = st.selectbox("Select Window to Remove", del_win_options, key="del_win_dropdown")
                    del_w_idx = del_win_options.index(del_w_sel)

                    if st.button("🗑️ Delete Selected Window", type="primary", use_container_width=True):
                        removed_win = working_plan.windows.pop(del_w_idx)
                        st.session_state.pop('editor_initial_json', None)
                        st.success(f"🗑️ Removed window on {removed_win.wall_side} wall of {removed_win.room_name}!")
                        st.rerun()
                else:
                    st.info("No windows on the floor plan.")

        # -------------------------------------------------------------
        # TAB 3: ADJUST ROOM DIMENSIONS
        # -------------------------------------------------------------
        with ctrl_tab_dim:
            st.markdown("##### Adjust Room Dimensions Dynamically:")
            room_names = [r.name for r in working_plan.rooms]
            
            if room_names:
                sc1, sc2, sc3 = st.columns([1.5, 1.2, 1.2])
                with sc1:
                    sel_room_name = st.selectbox("Select Room to Resize", room_names, key="resize_sel_room")
                
                target_room = next((r for r in working_plan.rooms if r.name == sel_room_name), working_plan.rooms[0])
                
                with sc2:
                    st.markdown(f"**Width:** `{target_room.w:.2f} m` ({target_room.w * 3.28084:.1f} ft)")
                    w_cols = st.columns(4)
                    if w_cols[0].button("−0.5m", key="w_m5"):
                        target_room.w = max(1.2, round(target_room.w - 0.5, 2))
                        target_room.area = round(target_room.w * target_room.h, 2)
                        st.session_state.pop('editor_initial_json', None)
                        st.rerun()
                    if w_cols[1].button("−0.1m", key="w_m1"):
                        target_room.w = max(1.2, round(target_room.w - 0.1, 2))
                        target_room.area = round(target_room.w * target_room.h, 2)
                        st.session_state.pop('editor_initial_json', None)
                        st.rerun()
                    if w_cols[2].button("+0.1m", key="w_p1"):
                        target_room.w = round(target_room.w + 0.1, 2)
                        target_room.area = round(target_room.w * target_room.h, 2)
                        st.session_state.pop('editor_initial_json', None)
                        st.rerun()
                    if w_cols[3].button("+0.5m", key="w_p5"):
                        target_room.w = round(target_room.w + 0.5, 2)
                        target_room.area = round(target_room.w * target_room.h, 2)
                        st.session_state.pop('editor_initial_json', None)
                        st.rerun()

                with sc3:
                    st.markdown(f"**Length / Height:** `{target_room.h:.2f} m` ({target_room.h * 3.28084:.1f} ft)")
                    h_cols = st.columns(4)
                    if h_cols[0].button("−0.5m", key="h_m5"):
                        target_room.h = max(1.2, round(target_room.h - 0.5, 2))
                        target_room.area = round(target_room.w * target_room.h, 2)
                        st.session_state.pop('editor_initial_json', None)
                        st.rerun()
                    if h_cols[1].button("−0.1m", key="h_m1"):
                        target_room.h = max(1.2, round(target_room.h - 0.1, 2))
                        target_room.area = round(target_room.w * target_room.h, 2)
                        st.session_state.pop('editor_initial_json', None)
                        st.rerun()
                    if h_cols[2].button("+0.1m", key="h_p1"):
                        target_room.h = round(target_room.h + 0.1, 2)
                        target_room.area = round(target_room.w * target_room.h, 2)
                        st.session_state.pop('editor_initial_json', None)
                        st.rerun()
                    if h_cols[3].button("+0.5m", key="h_p5"):
                        target_room.h = round(target_room.h + 0.5, 2)
                        target_room.area = round(target_room.w * target_room.h, 2)
                        st.session_state.pop('editor_initial_json', None)
                        st.rerun()

        st.write("")

        # -------------------------------------------------------------
        # Canvas Toolbar & Freehand Tools
        # -------------------------------------------------------------
        tool_col1, tool_col2, tool_col3, tool_col4 = st.columns([1.2, 1, 1, 1.2])
        with tool_col1:
            draw_mode = st.selectbox(
                "Tool Mode",
                ["Freedraw (Pencil Notes)", "Draw Line", "Draw Rect"],
                index=0,
                help="Select tool to sketch annotations. To move/resize existing rooms, use the ✏️/↖ toggle inside the canvas toolbar (top-left of the canvas)."
            )
        with tool_col2:
            stroke_width = st.slider("Pen Width", 1, 8, 3)
        with tool_col3:
            stroke_color = st.color_picker("Pen Color", "#00FFFF")
        with tool_col4:
            grid_snap = st.checkbox("Snap to 0.1m Module", value=True)

        mode_map = {
            "Freedraw (Pencil Notes)": "freedraw",
            "Draw Line": "line",
            "Draw Rect": "rect",
        }
        active_mode = mode_map.get(draw_mode, "freedraw")

        # Initialize canvas JSON if not already stored
        # Cache key includes canvas dimensions so any PPM/PAD change auto-busts the cache
        cache_key = f"{id(working_plan)}_{canvas_w}_{canvas_h}"
        if 'editor_initial_json' not in st.session_state or st.session_state.get('editor_plan_id') != cache_key:
            st.session_state['editor_initial_json'] = FloorPlanEditor.plan_to_fabric_json(
                working_plan, canvas_w, canvas_h
            )
            st.session_state['editor_plan_id'] = cache_key

        # -------------------------------------------------------------
        # Editor Canvas Layout: Left Canvas, Right Live-Sync HUD
        # -------------------------------------------------------------
        col_canvas, col_hud = st.columns([2.6, 1.4], gap="medium")

        with col_canvas:
            st.markdown(
                """
                <div style="border: 2px solid #00E5FF; border-radius: 8px; overflow: hidden; background: #0A2342; text-align: center; box-shadow: 0 4px 12px rgba(0, 229, 255, 0.15);">
                """,
                unsafe_allow_html=True
            )
            # Pure Fabric.js rendering on dark navy background.
            # plan_to_fabric_json uses PPM_EDITOR=50 and PAD_M=3.0, exactly matching
            # the SVGRenderer constants, so rooms appear in the same positions as the
            # static blueprint displayed above.
            canvas_result = st_canvas(
                fill_color="rgba(0, 229, 255, 0.15)",
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_color="#0A2342",
                initial_drawing=st.session_state['editor_initial_json'],
                update_streamlit=True,
                height=canvas_h,
                width=canvas_w,
                drawing_mode=active_mode,
                key="interactive_floorplan_canvas",
            )
            st.markdown("</div>", unsafe_allow_html=True)

        # Process Live Coordinates from Canvas Interaction
        edited_rooms = []
        if canvas_result and canvas_result.json_data:
            edited_rooms = FloorPlanEditor.extract_edited_rooms(canvas_result.json_data, canvas_h)
            # Synchronize canvas deletions (if user clicked trash icon in canvas toolbar)
            if edited_rooms and len(edited_rooms) < len(working_plan.rooms):
                surviving_names = {er["name"] for er in edited_rooms}
                working_plan.rooms = [r for r in working_plan.rooms if r.name in surviving_names]
                working_plan.doors = [d for d in working_plan.doors if d.room_name in surviving_names]
                working_plan.windows = [w for w in working_plan.windows if w.room_name in surviving_names]

        # Fallback to working_plan rooms if canvas extraction hasn't completed
        active_room_list = edited_rooms if edited_rooms else [
            {"name": r.name, "category": r.category, "x": r.x, "y": r.y, "w": r.w, "h": r.h, "area": r.area, "color": r.color}
            for r in working_plan.rooms
        ]

        # -------------------------------------------------------------
        # Live-Sync HUD & Dynamic Side Meters Calculations
        # -------------------------------------------------------------
        with col_hud:
            st.markdown("#### ⚡ Dynamic Side Meters & Property HUD")

            tot_carpet = sum(r["area"] for r in active_room_list)
            tot_built = round(tot_carpet * 1.22, 1)
            tot_sqft = tot_carpet * 10.764
            tot_cents = tot_sqft / 435.6

            # Live Vastu Score computation
            v_engine = VastuEngine(
                plot_width=working_plan.plot_width,
                plot_height=working_plan.plot_height,
                north_direction=working_plan.orientation
            )
            placements = []
            for r in active_room_list:
                cx = r["x"] + r["w"] / 2
                cy = r["y"] + r["h"] / 2
                z = v_engine.get_zone_for_position(cx, cy)
                placements.append((r["category"], z.value))
            
            live_vastu = v_engine.compute_plan_score(placements)

            # Live Cost Computation
            live_cost = CostEngine.estimate_cost(
                area_sqm=tot_carpet,
                city_tier=cost_params.get("city_tier", "Tier-2 (Pune, Hyderabad, Ahmedabad)"),
                quality_level=cost_params.get("quality", "Standard"),
                num_floors=cost_params.get("floors", 1)
            )

            # Metrics Display
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.metric("Live Carpet Area", f"{tot_carpet:.1f} m²", f"{tot_sqft:.0f} sqft ({tot_cents:.2f} C)")
                st.metric("Live Vastu Score", f"{live_vastu:.0f}/100", f"{live_vastu - current_plan.vastu_score:+.0f} pts")
            with m_col2:
                st.metric("Live Built-up Area", f"{tot_built:.1f} m²", f"{tot_built * 10.764:.0f} sq ft")
                st.metric("Live Est. Cost", f"₹ {live_cost['total_cost']/100000:.2f} L", f"₹ {live_cost['cost_per_sqft']}/sqft")

            # Dynamic Room Schedule in Meters and Feet
            st.markdown("##### 📐 Dynamic Room Schedule (Meters & Feet):")
            schedule_md = [
                "| Room Name | Size (Meters) | Size (Feet) | Area | Vastu |",
                "| :--- | :---: | :---: | :---: | :---: |"
            ]
            for r in active_room_list:
                w_ft = r["w"] * 3.28084
                h_ft = r["h"] * 3.28084
                cx = r["x"] + r["w"] / 2
                cy = r["y"] + r["h"] / 2
                z = v_engine.get_zone_for_position(cx, cy)
                zone_name = z.name.replace("_", " ").title()
                schedule_md.append(f"| **{r['name']}** | `{r['w']:.2f}×{r['h']:.2f}m` | `{w_ft:.1f}×{h_ft:.1f}'` | `{r['area']:.1f}m²` | `{zone_name}` |")

            st.markdown("\n".join(schedule_md))

            # Component Counts
            num_doors = len(working_plan.doors)
            num_entrance = sum(1 for d in working_plan.doors if d.is_main_entrance)
            num_wins = len(working_plan.windows)
            st.caption(f"🚪 Doors: **{num_doors}** (Main Entrance: **{num_entrance}** with flat steps) | 🪟 Windows: **{num_wins}**")

            st.write("")

            # Apply Button
            if st.button("💾 Apply Edits & Re-Generate Blueprints", type="primary", use_container_width=True):
                with st.spinner("Applying custom revisions and updating all CAD exports..."):
                    new_plan = copy.deepcopy(working_plan)
                    
                    new_room_objs = []
                    for er in active_room_list:
                        new_room_objs.append(RoomRect(
                            name=er["name"],
                            category=er["category"],
                            x=er["x"],
                            y=er["y"],
                            w=er["w"],
                            h=er["h"],
                            color=er["color"],
                            area=er["area"]
                        ))
                    new_plan.rooms = new_room_objs
                    new_plan.total_carpet_area = tot_carpet
                    new_plan.total_built_up_area = tot_built
                    new_plan.vastu_score = live_vastu

                    try:
                        eng = LayoutEngine(
                            plot_width=new_plan.plot_width,
                            plot_height=new_plan.plot_height,
                            bhk_config=new_plan.bhk_config or "auto",
                            vastu_enabled=new_plan.vastu_enabled,
                            orientation=new_plan.orientation
                        )
                        new_plan.walls = eng._generate_walls(new_room_objs)
                        new_plan.dimensions = eng._generate_dimensions(new_room_objs)
                    except Exception:
                        pass

                    svg_r = SVGRenderer()
                    svg_clean = svg_r.render_to_string(new_plan, mode='clean')
                    svg_blueprint = svg_r.render_to_string(new_plan, mode='blueprint')

                    st.session_state['gen_plan'] = new_plan
                    st.session_state['gen_cost'] = live_cost
                    st.session_state['gen_svg_clean'] = svg_clean
                    st.session_state['gen_svg_blueprint'] = svg_blueprint

                    import tempfile, os
                    temp_d = tempfile.gettempdir()
                    dxf_p = os.path.join(temp_d, "blueprint_edit.dxf")
                    pdf_p = os.path.join(temp_d, "blueprint_edit.pdf")
                    png_p = os.path.join(temp_d, "blueprint_edit.png")

                    DXFRenderer().render(new_plan, dxf_p)
                    PDFRenderer().render(new_plan, pdf_p)
                    PNGRenderer().render(new_plan, png_p)

                    with open(dxf_p, "rb") as f:
                        st.session_state['gen_dxf_bytes'] = f.read()
                    with open(pdf_p, "rb") as f:
                        st.session_state['gen_pdf_bytes'] = f.read()
                    with open(png_p, "rb") as f:
                        st.session_state['gen_png_bytes'] = f.read()
                    st.session_state['gen_svg_bytes'] = svg_clean.encode('utf-8')

                    st.session_state.pop('editor_initial_json', None)
                    st.session_state['editor_working_plan'] = copy.deepcopy(new_plan)

                    st.success("✅ Floor plan updated! All CAD downloads and metrics are fully synced.")
                    st.rerun()

            if st.button("🔄 Reset to Original AI Plan", use_container_width=True):
                st.session_state['editor_working_plan'] = copy.deepcopy(current_plan)
                st.session_state.pop('editor_initial_json', None)
                st.rerun()
