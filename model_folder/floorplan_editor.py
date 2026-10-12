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

LOCKED_PROPS = {
    "selectable": False,
    "evented": False,
    "lockMovementX": True,
    "lockMovementY": True,
    "lockRotation": True,
    "lockScalingX": True,
    "lockScalingY": True,
    "hasControls": False,
    "hasBorders": False,
    "hoverCursor": "default",
}


class FloorPlanEditor:
    """Manages interactive CAD blueprint drawing, component placement, and live-synchronization."""

    @staticmethod
    def compute_wall_point(room: RoomRect, wall_side: str, position_mode: str = "CENTER", custom_offset: float = 0.0) -> Tuple[float, float]:
        """Computes exact (x, y) coordinates on a room's wall for placing a door or window block."""
        margin = 0.6  # 600mm corner clearance
        if wall_side == 'N':
            y = room.top
            if position_mode == "CENTER":
                x = room.x + room.w / 2
            elif position_mode == "LEFT_OR_BOTTOM":
                x = room.x + min(margin, room.w * 0.25)
            elif position_mode == "RIGHT_OR_TOP":
                x = room.right - min(margin, room.w * 0.25)
            else:
                x = max(room.x + 0.3, min(room.right - 0.3, room.x + custom_offset))
            return round(x, 2), round(y, 2)
        elif wall_side == 'S':
            y = room.y
            if position_mode == "CENTER":
                x = room.x + room.w / 2
            elif position_mode == "LEFT_OR_BOTTOM":
                x = room.x + min(margin, room.w * 0.25)
            elif position_mode == "RIGHT_OR_TOP":
                x = room.right - min(margin, room.w * 0.25)
            else:
                x = max(room.x + 0.3, min(room.right - 0.3, room.x + custom_offset))
            return round(x, 2), round(y, 2)
        elif wall_side == 'E':
            x = room.right
            if position_mode == "CENTER":
                y = room.y + room.h / 2
            elif position_mode == "LEFT_OR_BOTTOM":
                y = room.y + min(margin, room.h * 0.25)
            elif position_mode == "RIGHT_OR_TOP":
                y = room.top - min(margin, room.h * 0.25)
            else:
                y = max(room.y + 0.3, min(room.top - 0.3, room.y + custom_offset))
            return round(x, 2), round(y, 2)
        else:  # 'W'
            x = room.x
            if position_mode == "CENTER":
                y = room.y + room.h / 2
            elif position_mode == "LEFT_OR_BOTTOM":
                y = room.y + min(margin, room.h * 0.25)
            elif position_mode == "RIGHT_OR_TOP":
                y = room.top - min(margin, room.h * 0.25)
            else:
                y = max(room.y + 0.3, min(room.top - 0.3, room.y + custom_offset))
            return round(x, 2), round(y, 2)

    @staticmethod
    def _rebuild_walls_and_dims(plan: GeneratedFloorPlan):
        """Regenerates CAD wall segments and dimension annotations after any component revision."""
        try:
            eng = LayoutEngine(
                plot_width=plan.plot_width,
                plot_height=plan.plot_height,
                bhk_config=plan.bhk_config or "auto",
                vastu_enabled=plan.vastu_enabled,
                orientation=plan.orientation
            )
            plan.walls = eng._generate_walls(plan.rooms)
            plan.dimensions = eng._generate_dimensions(plan.rooms)
        except Exception:
            pass

    @staticmethod
    def snap_point_to_nearest_wall(xm: float, ym: float, rooms: List[RoomRect]) -> Tuple[str, str, float, float]:
        """Snaps an arbitrary (xm, ym) point in meters to the nearest room wall with 300mm corner clearance."""
        if not rooms:
            return ("Default", "N", round(xm, 2), round(ym, 2))
        best_dist = float('inf')
        best_room = rooms[0].name
        best_side = 'N'
        best_pt = (round(xm, 2), round(ym, 2))
        for r in rooms:
            min_x = r.x + min(0.3, r.w * 0.2)
            max_x = r.right - min(0.3, r.w * 0.2)
            min_y = r.y + min(0.3, r.h * 0.2)
            max_y = r.top - min(0.3, r.h * 0.2)

            clamped_x = max(min_x, min(max_x, xm))
            d_n = math.hypot(xm - clamped_x, ym - r.top)
            if d_n < best_dist:
                best_dist = d_n; best_room = r.name; best_side = 'N'; best_pt = (round(clamped_x, 2), round(r.top, 2))

            d_s = math.hypot(xm - clamped_x, ym - r.y)
            if d_s < best_dist:
                best_dist = d_s; best_room = r.name; best_side = 'S'; best_pt = (round(clamped_x, 2), round(r.y, 2))

            clamped_y = max(min_y, min(max_y, ym))
            d_e = math.hypot(xm - r.right, ym - clamped_y)
            if d_e < best_dist:
                best_dist = d_e; best_room = r.name; best_side = 'E'; best_pt = (round(r.right, 2), round(clamped_y, 2))

            d_w = math.hypot(xm - r.x, ym - clamped_y)
            if d_w < best_dist:
                best_dist = d_w; best_room = r.name; best_side = 'W'; best_pt = (round(r.x, 2), round(clamped_y, 2))

        return (best_room, best_side, best_pt[0], best_pt[1])

    @staticmethod
    def plan_to_fabric_json(plan: GeneratedFloorPlan, canvas_w: int, canvas_h: int) -> Dict[str, Any]:
        """
        Converts a GeneratedFloorPlan into architectural Fabric.js objects format.
        Components (Rooms, Doors with swing arcs & entrance steps, Windows with double sills,
        Staircases with treads, and Lifts) are unified into atomic Fabric.js Groups with
        subTargetCheck=False, allowing full free-hand drag & drop component-by-component.
        Static CAD blueprint graphics (walls, title block, north compass, outer boundary,
        and millimeter dimensions) remain strictly locked.
        """
        objects = []

        def to_cx(xm: float) -> float:
            return round((xm + PAD_M) * PPM_EDITOR, 1)

        def to_cy(ym: float) -> float:
            return round(canvas_h - (ym + PAD_M) * PPM_EDITOR, 1)

        # -------------------------------------------------------------
        # 1. Outer Plot Boundary & Framing (LOCKED)
        # -------------------------------------------------------------
        plot_left = to_cx(0)
        plot_top = to_cy(plan.plot_height)
        plot_w_px = plan.plot_width * PPM_EDITOR
        plot_h_px = plan.plot_height * PPM_EDITOR

        objects.append({
            "type": "rect",
            "originX": "left",
            "originY": "top",
            "left": plot_left - 8,
            "top": plot_top - 8,
            "width": plot_w_px + 16,
            "height": plot_h_px + 16,
            "fill": "transparent",
            "stroke": "#00FFFF",
            "strokeWidth": 2.5,
            **LOCKED_PROPS,
        })
        objects.append({
            "type": "rect",
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
            **LOCKED_PROPS,
        })

        # -------------------------------------------------------------
        # 2. Architectural Wall Segments (LOCKED)
        # -------------------------------------------------------------
        for w in plan.walls:
            w_stroke = "#00FFFF" if w.is_external else "#38BDF8"
            w_thickness = 3.0 if w.is_external else 2.0
            objects.append({
                "type": "line",
                "x1": to_cx(w.x1),
                "y1": to_cy(w.y1),
                "x2": to_cx(w.x2),
                "y2": to_cy(w.y2),
                "stroke": w_stroke,
                "strokeWidth": w_thickness,
                **LOCKED_PROPS,
            })

        # -------------------------------------------------------------
        # 3. Interactive CAD Room Blocks (Unified Groups with Labels & Fittings)
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
            'Staircase': 'STAIRCASE',
            'Lift / Elevator': 'LIFT',
        }

        for idx, r in enumerate(plan.rooms):
            if r.w <= 0 or r.h <= 0:
                continue

            rw = r.w * PPM_EDITOR
            rh = r.h * PPM_EDITOR
            cx = to_cx(r.cx)
            cy = to_cy(r.cy)

            is_narrow = (r.w < 2.0 or r.h < 2.0 or r.area < 5.0)
            if is_narrow:
                display_name = abbrevs.get(r.name, r.name)
                f_size = 8.5
                text_content = f"{display_name.upper()}\n{r.w:.1f}×{r.h:.1f}m\n{r.area:.1f}m²"
            else:
                display_name = r.name
                f_size = 11 if (r.w >= 2.6 and r.h >= 2.6) else 9.5
                text_content = f"{display_name.upper()}\n{r.w:.2f} × {r.h:.2f} m\n{r.area:.1f} m²"

            room_children = []

            # Check if this room is an exterior amenity / site component
            category = getattr(r, 'category', '')
            is_amenity = category in ('PARKING', 'BIKE_PARKING', 'GARDEN', 'POOL', 'WATER_TANK', 'GENERATOR', 'SECURITY')

            if category == 'PARKING':
                slots = 1
                if '(' in r.name and 'Bay' in r.name:
                    try:
                        slots = max(1, int(r.name.split('(')[1].split()[0]))
                    except Exception:
                        slots = 1

                room_children.append({
                    "type": "rect",
                    "originX": "center", "originY": "center",
                    "left": 0, "top": 0,
                    "width": rw, "height": rh,
                    "fill": "rgba(51, 65, 85, 0.45)",
                    "stroke": "#38BDF8", "strokeWidth": 2.0, "strokeDashArray": [6, 4],
                    "selectable": False, "evented": False,
                })
                if rw >= rh:
                    stall_w = rw / slots
                    for i in range(1, slots):
                        bx = -rw / 2 + i * stall_w
                        room_children.append({
                            "type": "line",
                            "x1": bx, "y1": -rh / 2, "x2": bx, "y2": rh / 2,
                            "stroke": "#38BDF8", "strokeWidth": 1.5, "strokeDashArray": [4, 4],
                            "selectable": False, "evented": False
                        })
                    for i in range(slots):
                        bx_mid = -rw / 2 + (i + 0.5) * stall_w
                        room_children.append({
                            "type": "text", "originX": "center", "originY": "center",
                            "left": bx_mid, "top": -rh / 2 + 14,
                            "text": f"SLOT {i+1}", "fontSize": 8.5, "fontWeight": "bold", "fill": "#38BDF8",
                            "selectable": False, "evented": False
                        })
                        room_children.append({
                            "type": "line",
                            "x1": bx_mid - min(stall_w * 0.35, 18), "y1": rh / 2 - 12,
                            "x2": bx_mid + min(stall_w * 0.35, 18), "y2": rh / 2 - 12,
                            "stroke": "#E2E8F0", "strokeWidth": 2.5,
                            "selectable": False, "evented": False
                        })
                room_children.append({
                    "type": "text", "originX": "center", "originY": "center",
                    "left": 0, "top": 0,
                    "text": f"🅿️ CAR PARKING\n{slots} BAYS\n{r.w:.1f} × {r.h:.1f} m",
                    "fontSize": 10.5, "fontWeight": "bold", "fill": "#FFFFFF", "textAlign": "center",
                    "selectable": False, "evented": False,
                    "shadow": {"color": "rgba(0,0,0,0.9)", "blur": 4, "offsetX": 1, "offsetY": 1}
                })

            elif category == 'BIKE_PARKING':
                slots = 2
                if '(' in r.name and 'Bike' in r.name:
                    try:
                        slots = max(2, int(r.name.split('(')[1].split()[0]))
                    except Exception:
                        slots = 2

                room_children.append({
                    "type": "rect",
                    "originX": "center", "originY": "center",
                    "left": 0, "top": 0,
                    "width": rw, "height": rh,
                    "fill": "rgba(71, 85, 105, 0.45)",
                    "stroke": "#94A3B8", "strokeWidth": 1.8, "strokeDashArray": [5, 3],
                    "selectable": False, "evented": False,
                })
                num_divs = min(slots, 6)
                if num_divs > 1:
                    sw = rw / num_divs
                    for i in range(1, num_divs):
                        bx = -rw / 2 + i * sw
                        room_children.append({
                            "type": "line",
                            "x1": bx, "y1": -rh / 2, "x2": bx, "y2": rh / 2,
                            "stroke": "#94A3B8", "strokeWidth": 1.0, "strokeDashArray": [3, 3],
                            "selectable": False, "evented": False
                        })
                room_children.append({
                    "type": "text", "originX": "center", "originY": "center",
                    "left": 0, "top": 0,
                    "text": f"🏍️ BIKE PARKING\n{slots} BIKES\n{r.w:.1f} × {r.h:.1f} m",
                    "fontSize": 9.5, "fontWeight": "bold", "fill": "#F1F5F9", "textAlign": "center",
                    "selectable": False, "evented": False,
                    "shadow": {"color": "rgba(0,0,0,0.9)", "blur": 4, "offsetX": 1, "offsetY": 1}
                })

            elif category == 'GARDEN':
                room_children.append({
                    "type": "rect",
                    "originX": "center", "originY": "center",
                    "left": 0, "top": 0,
                    "width": rw, "height": rh,
                    "fill": "rgba(20, 83, 45, 0.35)",
                    "stroke": "#22C55E", "strokeWidth": 2.0, "strokeDashArray": [8, 4],
                    "selectable": False, "evented": False,
                })
                coff = 16
                for tx, ty in [(-rw/2 + coff, -rh/2 + coff), (rw/2 - coff, -rh/2 + coff),
                               (-rw/2 + coff, rh/2 - coff), (rw/2 - coff, rh/2 - coff)]:
                    room_children.append({
                        "type": "text", "originX": "center", "originY": "center",
                        "left": tx, "top": ty, "text": "🌳", "fontSize": 13,
                        "selectable": False, "evented": False
                    })
                room_children.append({
                    "type": "text", "originX": "center", "originY": "center",
                    "left": 0, "top": 0,
                    "text": f"🌿 LANDSCAPED GARDEN\n{r.area:.1f} m² Lawn",
                    "fontSize": 10.5, "fontWeight": "bold", "fill": "#86EFAC", "textAlign": "center",
                    "selectable": False, "evented": False,
                    "shadow": {"color": "rgba(0,0,0,0.9)", "blur": 4, "offsetX": 1, "offsetY": 1}
                })

            elif category == 'POOL':
                # Coping
                room_children.append({
                    "type": "rect",
                    "originX": "center", "originY": "center",
                    "left": 0, "top": 0,
                    "width": rw, "height": rh,
                    "fill": "rgba(148, 163, 184, 0.25)",
                    "stroke": "#94A3B8", "strokeWidth": 3.0,
                    "selectable": False, "evented": False,
                })
                # Water
                room_children.append({
                    "type": "rect",
                    "originX": "center", "originY": "center",
                    "left": 0, "top": 0,
                    "width": max(10, rw - 12), "height": max(10, rh - 12),
                    "fill": "rgba(2, 132, 199, 0.55)",
                    "stroke": "#38BDF8", "strokeWidth": 1.5,
                    "selectable": False, "evented": False,
                })
                room_children.append({
                    "type": "text", "originX": "center", "originY": "center",
                    "left": 0, "top": -rh / 3,
                    "text": "~ ~ ~ ~ ~ ~ ~", "fontSize": 11, "fill": "#BAE6FD",
                    "selectable": False, "evented": False
                })
                room_children.append({
                    "type": "text", "originX": "center", "originY": "center",
                    "left": 0, "top": 2,
                    "text": f"🏊 SWIMMING POOL\n{r.w:.1f} × {r.h:.1f} m\nDepth: 1.2m – 1.8m",
                    "fontSize": 10, "fontWeight": "bold", "fill": "#FFFFFF", "textAlign": "center",
                    "selectable": False, "evented": False,
                    "shadow": {"color": "rgba(0,0,0,0.9)", "blur": 4, "offsetX": 1, "offsetY": 1}
                })

            elif category == 'WATER_TANK':
                room_children.append({
                    "type": "rect",
                    "originX": "center", "originY": "center",
                    "left": 0, "top": 0,
                    "width": rw, "height": rh,
                    "fill": "rgba(2, 132, 199, 0.30)",
                    "stroke": "#0284C7", "strokeWidth": 2.0,
                    "selectable": False, "evented": False,
                })
                rc = max(8, min(rw, rh) * 0.32)
                room_children.append({
                    "type": "circle",
                    "originX": "center", "originY": "center",
                    "left": 0, "top": 0, "radius": rc,
                    "fill": "transparent", "stroke": "#38BDF8", "strokeWidth": 1.8, "strokeDashArray": [4, 3],
                    "selectable": False, "evented": False,
                })
                room_children.append({
                    "type": "text", "originX": "center", "originY": "center",
                    "left": 0, "top": 0,
                    "text": "💧 SUMP (NE)\n2.0 × 2.0 m\nUNDERGROUND",
                    "fontSize": 9.5, "fontWeight": "bold", "fill": "#38BDF8", "textAlign": "center",
                    "selectable": False, "evented": False,
                    "shadow": {"color": "rgba(0,0,0,0.9)", "blur": 4, "offsetX": 1, "offsetY": 1}
                })

            elif category == 'GENERATOR':
                room_children.append({
                    "type": "rect",
                    "originX": "center", "originY": "center",
                    "left": 0, "top": 0,
                    "width": rw, "height": rh,
                    "fill": "rgba(180, 83, 9, 0.30)",
                    "stroke": "#F59E0B", "strokeWidth": 2.0,
                    "selectable": False, "evented": False,
                })
                room_children.append({
                    "type": "line",
                    "x1": -rw / 2 + 4, "y1": -rh / 2 + 4, "x2": rw / 2 - 4, "y2": rh / 2 - 4,
                    "stroke": "#F59E0B", "strokeWidth": 1.0,
                    "selectable": False, "evented": False,
                })
                room_children.append({
                    "type": "line",
                    "x1": rw / 2 - 4, "y1": -rh / 2 + 4, "x2": -rw / 2 + 4, "y2": rh / 2 - 4,
                    "stroke": "#F59E0B", "strokeWidth": 1.0,
                    "selectable": False, "evented": False,
                })
                room_children.append({
                    "type": "rect",
                    "originX": "center", "originY": "center",
                    "left": 0, "top": 0, "width": 30, "height": 22,
                    "fill": "#78350F", "stroke": "#F59E0B", "strokeWidth": 1.0,
                    "selectable": False, "evented": False,
                })
                room_children.append({
                    "type": "text", "originX": "center", "originY": "center",
                    "left": 0, "top": 0,
                    "text": "⚡ GENSET\n2.5 × 2.5 m",
                    "fontSize": 9.0, "fontWeight": "bold", "fill": "#FDE68A", "textAlign": "center",
                    "selectable": False, "evented": False,
                    "shadow": {"color": "rgba(0,0,0,0.9)", "blur": 4, "offsetX": 1, "offsetY": 1}
                })

            elif category == 'SECURITY':
                room_children.append({
                    "type": "rect",
                    "originX": "center", "originY": "center",
                    "left": 0, "top": 0,
                    "width": rw, "height": rh,
                    "fill": "rgba(109, 40, 217, 0.30)",
                    "stroke": "#8B5CF6", "strokeWidth": 2.0,
                    "selectable": False, "evented": False,
                })
                room_children.append({
                    "type": "rect",
                    "originX": "center", "originY": "center",
                    "left": -rw * 0.15, "top": -rh * 0.15,
                    "width": rw * 0.40, "height": rh * 0.35,
                    "fill": "#4C1D95", "stroke": "#A78BFA", "strokeWidth": 1.0,
                    "selectable": False, "evented": False,
                })
                room_children.append({
                    "type": "text", "originX": "center", "originY": "center",
                    "left": 0, "top": rh * 0.15,
                    "text": "🛡️ SECURITY\n2.0 × 2.0 m",
                    "fontSize": 9.0, "fontWeight": "bold", "fill": "#DDD6FE", "textAlign": "center",
                    "selectable": False, "evented": False,
                    "shadow": {"color": "rgba(0,0,0,0.9)", "blur": 4, "offsetX": 1, "offsetY": 1}
                })

            elif category == 'STAIRS':
                room_children.append({
                    "type": "rect",
                    "originX": "center", "originY": "center",
                    "left": 0, "top": 0,
                    "width": rw, "height": rh,
                    "fill": "rgba(14, 116, 144, 0.22)",
                    "stroke": "#00E5FF", "strokeWidth": 2.0, "strokeUniform": True,
                    "selectable": False, "evented": False,
                })
                num_treads = max(3, int(r.h / 0.25))
                for t_i in range(1, num_treads):
                    rel_ty = -rh / 2 + t_i * (rh / num_treads)
                    room_children.append({
                        'type': 'line',
                        'x1': -rw / 2, 'y1': rel_ty,
                        'x2': rw / 2, 'y2': rel_ty,
                        'stroke': '#6B8DAF', 'strokeWidth': 1.0,
                        'selectable': False, 'evented': False
                    })
                room_children.append({
                    'type': 'text',
                    'originX': 'center', 'originY': 'center',
                    'left': 0, 'top': 0,
                    'text': '▲ UP',
                    'fontSize': 10, 'fontWeight': 'bold', 'fill': '#00FFCC',
                    'selectable': False, 'evented': False
                })
            elif category == 'LIFT':
                room_children.append({
                    "type": "rect",
                    "originX": "center", "originY": "center",
                    "left": 0, "top": 0,
                    "width": rw, "height": rh,
                    "fill": "rgba(14, 116, 144, 0.22)",
                    "stroke": "#00E5FF", "strokeWidth": 2.0, "strokeUniform": True,
                    "selectable": False, "evented": False,
                })
                room_children.append({
                    'type': 'line',
                    'x1': -rw / 2, 'y1': -rh / 2,
                    'x2': rw / 2, 'y2': rh / 2,
                    'stroke': '#38BDF8', 'strokeWidth': 1.0,
                    'selectable': False, 'evented': False
                })
                room_children.append({
                    'type': 'line',
                    'x1': rw / 2, 'y1': -rh / 2,
                    'x2': -rw / 2, 'y2': rh / 2,
                    'stroke': '#38BDF8', 'strokeWidth': 1.0,
                    'selectable': False, 'evented': False
                })
                room_children.append({
                    'type': 'text',
                    'originX': 'center', 'originY': 'center',
                    'left': 0, 'top': 0,
                    'text': 'LIFT\n🛗',
                    'fontSize': 10, 'fontWeight': 'bold', 'fill': '#38BDF8',
                    'textAlign': 'center',
                    'selectable': False, 'evented': False
                })
            else:
                room_children.append({
                    "type": "rect",
                    "originX": "center", "originY": "center",
                    "left": 0, "top": 0,
                    "width": rw, "height": rh,
                    "fill": "rgba(14, 116, 144, 0.22)",
                    "stroke": "#00E5FF", "strokeWidth": 2.0, "strokeUniform": True,
                    "selectable": False, "evented": False,
                })
                room_children.append({
                    "type": "text",
                    "originX": "center",
                    "originY": "center",
                    "left": 0,
                    "top": 0,
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

            dash_tag = 4000 + idx if is_amenity else 1000 + idx

            objects.append({
                "type": "Group",
                "originX": "center",
                "originY": "center",
                "left": cx,
                "top": cy,
                "width": rw,
                "height": rh,
                "selectable": True,
                "hasControls": True,
                "hasBorders": True,
                "subTargetCheck": False,
                "interactive": False,
                "lockRotation": True,
                "strokeDashArray": [dash_tag, 0],
                "objects": room_children,
            })

        # -------------------------------------------------------------
        # 4. Interactive Doors (Unified Groups with Swings, Leaves & Entrance Steps)
        # -------------------------------------------------------------
        for idx, d in enumerate(plan.doors):
            hx = to_cx(d.x)
            hy = to_cy(d.y)
            r_px = d.width * PPM_EDITOR
            side = d.wall_side
            swing = d.swing_direction

            door_children = []

            # Transparent hit-box rect
            door_children.append({
                "type": "rect",
                "originX": "center",
                "originY": "center",
                "left": 0,
                "top": 0,
                "width": max(r_px * 2.2, 40),
                "height": max(r_px * 2.2, 40),
                "fill": "transparent",
                "stroke": "transparent",
                "selectable": False,
                "evented": False,
            })

            # Door threshold, leaf, and swing arc
            if side in ('N', 'S'):
                thresh_x2 = r_px if swing == 'RIGHT' else -r_px
                door_children.append({
                    "type": "line",
                    "x1": 0, "y1": 0, "x2": thresh_x2, "y2": 0,
                    "stroke": "#00FF88", "strokeWidth": 3.0,
                    "selectable": False, "evented": False
                })
                leaf_y = r_px if side == 'N' else -r_px
                door_children.append({
                    "type": "line",
                    "x1": 0, "y1": 0, "x2": 0, "y2": leaf_y,
                    "stroke": "#00FF88", "strokeWidth": 2.0,
                    "selectable": False, "evented": False
                })
                sweep_flag = "1" if (side == 'N' and swing == 'RIGHT') or (side == 'S' and swing == 'LEFT') else "0"
                door_children.append({
                    "type": "path",
                    "path": f"M {thresh_x2} 0 A {r_px} {r_px} 0 0 {sweep_flag} 0 {leaf_y}",
                    "stroke": "#4A9E7F", "strokeWidth": 1.2, "strokeDashArray": [3, 3],
                    "fill": "transparent", "selectable": False, "evented": False
                })
            else:
                thresh_y2 = r_px if swing == 'RIGHT' else -r_px
                door_children.append({
                    "type": "line",
                    "x1": 0, "y1": 0, "x2": 0, "y2": thresh_y2,
                    "stroke": "#00FF88", "strokeWidth": 3.0,
                    "selectable": False, "evented": False
                })
                leaf_x = -r_px if side == 'E' else r_px
                door_children.append({
                    "type": "line",
                    "x1": 0, "y1": 0, "x2": leaf_x, "y2": 0,
                    "stroke": "#00FF88", "strokeWidth": 2.0,
                    "selectable": False, "evented": False
                })
                sweep_flag = "1" if (side == 'E' and swing == 'RIGHT') or (side == 'W' and swing == 'LEFT') else "0"
                door_children.append({
                    "type": "path",
                    "path": f"M 0 {thresh_y2} A {r_px} {r_px} 0 0 {sweep_flag} {leaf_x} 0",
                    "stroke": "#4A9E7F", "strokeWidth": 1.2, "strokeDashArray": [3, 3],
                    "fill": "transparent", "selectable": False, "evented": False
                })

            # Door interactive handle circle
            door_children.append({
                "type": "circle",
                "originX": "center",
                "originY": "center",
                "left": 0,
                "top": 0,
                "radius": 7,
                "fill": "#00FF88",
                "stroke": "#064E3B",
                "strokeWidth": 2.0,
                "selectable": False,
                "evented": False
            })

            # Entrance Flat Steps & Landing (for Main Entrance Door)
            if d.is_main_entrance:
                landing_depth_px = 1.2 * PPM_EDITOR
                step_depth_px = 0.3 * PPM_EDITOR
                side_ext_px = 0.3 * PPM_EDITOR
                num_steps = 3

                if side == 'E':
                    rel_l = 0
                    rel_t = -(r_px / 2 + side_ext_px)
                    land_w = landing_depth_px
                    land_h = r_px + side_ext_px * 2

                    door_children.append({
                        "type": "rect",
                        "originX": "left", "originY": "top",
                        "left": rel_l, "top": rel_t, "width": land_w, "height": land_h,
                        "fill": "rgba(13, 43, 69, 0.85)", "stroke": "#00DDDD", "strokeWidth": 2.0,
                        "selectable": False, "evented": False
                    })
                    door_children.append({
                        "type": "text",
                        "originX": "center", "originY": "center",
                        "left": rel_l + land_w / 2, "top": rel_t + land_h / 2,
                        "text": "LANDING", "fontSize": 8.5, "fill": "#8EB8E5",
                        "selectable": False, "evented": False
                    })
                    for i in range(1, num_steps + 1):
                        sx = rel_l + land_w + i * step_depth_px
                        door_children.append({
                            "type": "line",
                            "x1": sx, "y1": rel_t, "x2": sx, "y2": rel_t + land_h,
                            "stroke": "#00DDDD", "strokeWidth": 2.0, "selectable": False, "evented": False
                        })
                    tot_step_w = num_steps * step_depth_px
                    door_children.append({
                        "type": "line",
                        "x1": rel_l + land_w, "y1": rel_t, "x2": rel_l + land_w + tot_step_w, "y2": rel_t,
                        "stroke": "#00DDDD", "strokeWidth": 2.0, "selectable": False, "evented": False
                    })
                    door_children.append({
                        "type": "line",
                        "x1": rel_l + land_w, "y1": rel_t + land_h, "x2": rel_l + land_w + tot_step_w, "y2": rel_t + land_h,
                        "stroke": "#00DDDD", "strokeWidth": 2.0, "selectable": False, "evented": False
                    })
                    door_children.append({
                        "type": "line",
                        "x1": rel_l + land_w + tot_step_w, "y1": rel_t, "x2": rel_l + land_w + tot_step_w, "y2": rel_t + land_h,
                        "stroke": "#00DDDD", "strokeWidth": 2.0, "selectable": False, "evented": False
                    })
                    door_children.append({
                        "type": "text",
                        "originX": "center", "originY": "center",
                        "left": rel_l + land_w + tot_step_w / 2, "top": rel_t - 14,
                        "text": "◄ UP", "fontSize": 10, "fontWeight": "bold", "fill": "#00FFCC",
                        "selectable": False, "evented": False
                    })

                elif side == 'S':
                    rel_l = -(r_px / 2 + side_ext_px)
                    rel_t = 0
                    land_w = r_px + side_ext_px * 2
                    land_h = landing_depth_px

                    door_children.append({
                        "type": "rect",
                        "originX": "left", "originY": "top",
                        "left": rel_l, "top": rel_t, "width": land_w, "height": land_h,
                        "fill": "rgba(13, 43, 69, 0.85)", "stroke": "#00DDDD", "strokeWidth": 2.0,
                        "selectable": False, "evented": False
                    })
                    for i in range(1, num_steps + 1):
                        sy = rel_t + land_h + i * step_depth_px
                        door_children.append({
                            "type": "line",
                            "x1": rel_l, "y1": sy, "x2": rel_l + land_w, "y2": sy,
                            "stroke": "#00DDDD", "strokeWidth": 2.0, "selectable": False, "evented": False
                        })
                    tot_step_h = num_steps * step_depth_px
                    door_children.append({
                        "type": "line",
                        "x1": rel_l, "y1": rel_t + land_h, "x2": rel_l, "y2": rel_t + land_h + tot_step_h,
                        "stroke": "#00DDDD", "strokeWidth": 2.0, "selectable": False, "evented": False
                    })
                    door_children.append({
                        "type": "line",
                        "x1": rel_l + land_w, "y1": rel_t + land_h, "x2": rel_l + land_w, "y2": rel_t + land_h + tot_step_h,
                        "stroke": "#00DDDD", "strokeWidth": 2.0, "selectable": False, "evented": False
                    })
                    door_children.append({
                        "type": "line",
                        "x1": rel_l, "y1": rel_t + land_h + tot_step_h, "x2": rel_l + land_w, "y2": rel_t + land_h + tot_step_h,
                        "stroke": "#00DDDD", "strokeWidth": 2.0, "selectable": False, "evented": False
                    })
                    door_children.append({
                        "type": "text",
                        "originX": "center", "originY": "center",
                        "left": rel_l + land_w + 16, "top": rel_t + land_h + tot_step_h / 2,
                        "text": "▲ UP", "fontSize": 10, "fontWeight": "bold", "fill": "#00FFCC",
                        "selectable": False, "evented": False
                    })

                elif side == 'W':
                    land_w = landing_depth_px
                    land_h = r_px + side_ext_px * 2
                    rel_l = -land_w
                    rel_t = -land_h / 2

                    door_children.append({
                        "type": "rect",
                        "originX": "left", "originY": "top",
                        "left": rel_l, "top": rel_t, "width": land_w, "height": land_h,
                        "fill": "rgba(13, 43, 69, 0.85)", "stroke": "#00DDDD", "strokeWidth": 2.0,
                        "selectable": False, "evented": False
                    })
                    for i in range(1, num_steps + 1):
                        sx = rel_l - i * step_depth_px
                        door_children.append({
                            "type": "line",
                            "x1": sx, "y1": rel_t, "x2": sx, "y2": rel_t + land_h,
                            "stroke": "#00DDDD", "strokeWidth": 2.0, "selectable": False, "evented": False
                        })
                    tot_step_w = num_steps * step_depth_px
                    door_children.append({
                        "type": "line",
                        "x1": rel_l, "y1": rel_t, "x2": rel_l - tot_step_w, "y2": rel_t,
                        "stroke": "#00DDDD", "strokeWidth": 2.0, "selectable": False, "evented": False
                    })
                    door_children.append({
                        "type": "line",
                        "x1": rel_l, "y1": rel_t + land_h, "x2": rel_l - tot_step_w, "y2": rel_t + land_h,
                        "stroke": "#00DDDD", "strokeWidth": 2.0, "selectable": False, "evented": False
                    })
                    door_children.append({
                        "type": "line",
                        "x1": rel_l - tot_step_w, "y1": rel_t, "x2": rel_l - tot_step_w, "y2": rel_t + land_h,
                        "stroke": "#00DDDD", "strokeWidth": 2.0, "selectable": False, "evented": False
                    })
                    door_children.append({
                        "type": "text",
                        "originX": "center", "originY": "center",
                        "left": rel_l - tot_step_w / 2, "top": rel_t - 14,
                        "text": "UP ►", "fontSize": 10, "fontWeight": "bold", "fill": "#00FFCC",
                        "selectable": False, "evented": False
                    })

                elif side == 'N':
                    rel_l = -(r_px / 2 + side_ext_px)
                    land_w = r_px + side_ext_px * 2
                    land_h = landing_depth_px
                    rel_t = -land_h

                    door_children.append({
                        "type": "rect",
                        "originX": "left", "originY": "top",
                        "left": rel_l, "top": rel_t, "width": land_w, "height": land_h,
                        "fill": "rgba(13, 43, 69, 0.85)", "stroke": "#00DDDD", "strokeWidth": 2.0,
                        "selectable": False, "evented": False
                    })
                    for i in range(1, num_steps + 1):
                        sy = rel_t - i * step_depth_px
                        door_children.append({
                            "type": "line",
                            "x1": rel_l, "y1": sy, "x2": rel_l + land_w, "y2": sy,
                            "stroke": "#00DDDD", "strokeWidth": 2.0, "selectable": False, "evented": False
                        })
                    tot_step_h = num_steps * step_depth_px
                    door_children.append({
                        "type": "line",
                        "x1": rel_l, "y1": rel_t - tot_step_h, "x2": rel_l, "y2": rel_t,
                        "stroke": "#00DDDD", "strokeWidth": 2.0, "selectable": False, "evented": False
                    })
                    door_children.append({
                        "type": "line",
                        "x1": rel_l + land_w, "y1": rel_t - tot_step_h, "x2": rel_l + land_w, "y2": rel_t,
                        "stroke": "#00DDDD", "strokeWidth": 2.0, "selectable": False, "evented": False
                    })
                    door_children.append({
                        "type": "line",
                        "x1": rel_l, "y1": rel_t - tot_step_h, "x2": rel_l + land_w, "y2": rel_t - tot_step_h,
                        "stroke": "#00DDDD", "strokeWidth": 2.0, "selectable": False, "evented": False
                    })
                    door_children.append({
                        "type": "text",
                        "originX": "center", "originY": "center",
                        "left": rel_l + land_w + 16, "top": rel_t - tot_step_h / 2,
                        "text": "▼ UP", "fontSize": 10, "fontWeight": "bold", "fill": "#00FFCC",
                        "selectable": False, "evented": False
                    })

            objects.append({
                "type": "Group",
                "originX": "center",
                "originY": "center",
                "left": hx,
                "top": hy,
                "selectable": True,
                "hasControls": True,
                "hasBorders": True,
                "subTargetCheck": False,
                "interactive": False,
                "strokeDashArray": [2000 + idx, 0],
                "objects": door_children,
            })

        # -------------------------------------------------------------
        # 5. Interactive Windows (Unified Groups with Double Glazing & Sills)
        # -------------------------------------------------------------
        for idx, win in enumerate(plan.windows):
            wx_px = to_cx(win.x)
            wy_px = to_cy(win.y)
            w_len_px = win.width * PPM_EDITOR
            win_children = []

            if win.wall_side in ('N', 'S'):
                win_children.append({
                    "type": "rect",
                    "originX": "center", "originY": "center",
                    "left": 0,
                    "top": 0,
                    "width": w_len_px + 8,
                    "height": 24,
                    "fill": "transparent",
                    "stroke": "transparent",
                    "selectable": False,
                    "evented": False
                })
                win_children.append({
                    "type": "line",
                    "x1": -w_len_px / 2, "y1": -2,
                    "x2": w_len_px / 2, "y2": -2,
                    "stroke": "#00BFFF", "strokeWidth": 1.5,
                    "selectable": False, "evented": False
                })
                win_children.append({
                    "type": "line",
                    "x1": -w_len_px / 2, "y1": 2,
                    "x2": w_len_px / 2, "y2": 2,
                    "stroke": "#00BFFF", "strokeWidth": 1.5,
                    "selectable": False, "evented": False
                })
                win_children.append({
                    "type": "line",
                    "x1": -w_len_px / 2 - 4, "y1": 4,
                    "x2": w_len_px / 2 + 4, "y2": 4,
                    "stroke": "#00AACC", "strokeWidth": 1.2,
                    "selectable": False, "evented": False
                })
            else:
                win_children.append({
                    "type": "rect",
                    "originX": "center", "originY": "center",
                    "left": 0,
                    "top": 0,
                    "width": 24,
                    "height": w_len_px + 8,
                    "fill": "transparent",
                    "stroke": "transparent",
                    "selectable": False,
                    "evented": False
                })
                win_children.append({
                    "type": "line",
                    "x1": -2, "y1": -w_len_px / 2,
                    "x2": -2, "y2": w_len_px / 2,
                    "stroke": "#00BFFF", "strokeWidth": 1.5,
                    "selectable": False, "evented": False
                })
                win_children.append({
                    "type": "line",
                    "x1": 2, "y1": -w_len_px / 2,
                    "x2": 2, "y2": w_len_px / 2,
                    "stroke": "#00BFFF", "strokeWidth": 1.5,
                    "selectable": False, "evented": False
                })
                win_children.append({
                    "type": "line",
                    "x1": 4, "y1": -w_len_px / 2 - 4,
                    "x2": 4, "y2": w_len_px / 2 + 4,
                    "stroke": "#00AACC", "strokeWidth": 1.2,
                    "selectable": False, "evented": False
                })

            objects.append({
                "type": "Group",
                "originX": "center",
                "originY": "center",
                "left": wx_px,
                "top": wy_px,
                "selectable": True,
                "hasControls": True,
                "hasBorders": True,
                "subTargetCheck": False,
                "interactive": False,
                "strokeDashArray": [3000 + idx, 0],
                "objects": win_children,
            })

        # -------------------------------------------------------------
        # 6. Title Block (LOCKED)
        # -------------------------------------------------------------
        title_x = to_cx(plan.plot_width) + 20
        title_y = plot_top + 10
        title_w = 5.0 * PPM_EDITOR - 30
        title_h = min(plot_h_px - 20, 250)

        objects.append({
            "type": "rect",
            "originX": "left",
            "originY": "top",
            "left": title_x, "top": title_y,
            "width": title_w, "height": title_h,
            "fill": "rgba(10, 35, 66, 0.90)", "stroke": "#00CCCC", "strokeWidth": 1.5,
            **LOCKED_PROPS,
        })

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
            if idx > 0:
                objects.append({
                    "type": "line",
                    "x1": title_x, "y1": row_y, "x2": title_x + title_w, "y2": row_y,
                    "stroke": "#00CCCC", "strokeWidth": 0.8,
                    **LOCKED_PROPS,
                })
            objects.append({
                "type": "text",
                "originX": "left", "originY": "top",
                "left": title_x + 6, "top": row_y + 6,
                "text": label_text,
                "fontSize": 9, "fontWeight": "bold" if idx == 4 else "normal",
                "fontFamily": "Segoe UI, Arial, sans-serif",
                "fill": "#E0F0FF",
                **LOCKED_PROPS,
            })

        # -------------------------------------------------------------
        # 7. North Compass (LOCKED)
        # -------------------------------------------------------------
        comp_x = title_x + title_w / 2
        comp_y = title_y + title_h + 50
        if comp_y + 40 < canvas_h:
            objects.append({
                "type": "circle",
                "originX": "center", "originY": "center",
                "left": comp_x, "top": comp_y, "radius": 24,
                "fill": "rgba(10, 35, 66, 0.75)", "stroke": "#00DDAA", "strokeWidth": 1.5,
                **LOCKED_PROPS,
            })
            objects.append({
                "type": "text",
                "originX": "center", "originY": "center",
                "left": comp_x, "top": comp_y - 4,
                "text": "▲\nN",
                "fontSize": 12, "fontWeight": "bold", "fill": "#00FFCC",
                "textAlign": "center", "lineHeight": 0.9,
                **LOCKED_PROPS,
            })
            for label, dx, dy in [("W", -32, 0), ("E", 32, 0), ("S", 0, 32)]:
                objects.append({
                    "type": "text", "originX": "center", "originY": "center",
                    "left": comp_x + dx, "top": comp_y + dy, "text": label,
                    "fontSize": 9, "fill": "#8EB8E5",
                    **LOCKED_PROPS,
                })

        # -------------------------------------------------------------
        # 8. Exterior Dimension Lines & Tick Annotations (LOCKED)
        # -------------------------------------------------------------
        dim_y = plot_top - 28
        objects.append({
            "type": "line",
            "x1": plot_left, "y1": dim_y, "x2": plot_left + plot_w_px, "y2": dim_y,
            "stroke": "#6B8DAF", "strokeWidth": 1.0,
            **LOCKED_PROPS,
        })
        for tick_x in [plot_left, plot_left + plot_w_px]:
            objects.append({
                "type": "line",
                "x1": tick_x - 5, "y1": dim_y + 5, "x2": tick_x + 5, "y2": dim_y - 5,
                "stroke": "#6B8DAF", "strokeWidth": 1.5,
                **LOCKED_PROPS,
            })
        objects.append({
            "type": "text",
            "originX": "center", "originY": "center",
            "left": plot_left + plot_w_px / 2, "top": dim_y - 12,
            "text": f"{int(round(plan.plot_width * 1000))}",
            "fontSize": 9, "fill": "#8EB8E5",
            **LOCKED_PROPS,
        })

        dim_x = plot_left - 32
        objects.append({
            "type": "line",
            "x1": dim_x, "y1": plot_top, "x2": dim_x, "y2": plot_top + plot_h_px,
            "stroke": "#6B8DAF", "strokeWidth": 1.0,
            **LOCKED_PROPS,
        })
        for tick_y in [plot_top, plot_top + plot_h_px]:
            objects.append({
                "type": "line",
                "x1": dim_x - 5, "y1": tick_y + 5, "x2": dim_x + 5, "y2": tick_y - 5,
                "stroke": "#6B8DAF", "strokeWidth": 1.5,
                **LOCKED_PROPS,
            })
        objects.append({
            "type": "text",
            "originX": "center", "originY": "center",
            "left": dim_x - 16, "top": plot_top + plot_h_px / 2,
            "text": f"{int(round(plan.plot_height * 1000))}",
            "fontSize": 9, "fill": "#8EB8E5", "angle": -90,
            **LOCKED_PROPS,
        })

        return {
            "version": "4.4.0",
            "objects": objects
        }

    @staticmethod
    def sync_canvas_drag_edits(json_data: Dict[str, Any], working_plan: GeneratedFloorPlan, canvas_w: int, canvas_h: int) -> bool:
        """
        Detects free-hand dragged or resized component blocks (rooms, doors, windows, staircases)
        from Fabric canvas json_data, snaps doors and windows to walls, and updates working_plan.
        Returns True if any component was modified or moved.
        """
        if not json_data or "objects" not in json_data:
            return False

        def to_cx(xm: float) -> float:
            return round((xm + PAD_M) * PPM_EDITOR, 1)

        def to_cy(ym: float) -> float:
            return round(canvas_h - (ym + PAD_M) * PPM_EDITOR, 1)

        def from_cx(cx: float) -> float:
            return round((cx / PPM_EDITOR) - PAD_M, 2)

        def from_cy(cy: float) -> float:
            return round(((canvas_h - cy) / PPM_EDITOR) - PAD_M, 2)

        plan_changed = False
        surviving_room_indices = set()
        surviving_door_indices = set()
        surviving_win_indices = set()

        for obj in json_data.get("objects", []):
            o_type = obj.get("type", "").lower()
            if o_type not in ("group",):
                continue

            dash = obj.get("strokeDashArray")
            tag_code = dash[0] if (dash and isinstance(dash, (list, tuple)) and len(dash) > 0) else None
            if tag_code is None:
                continue

            curr_left = float(obj.get("left", 0))
            curr_top = float(obj.get("top", 0))
            scale_x = float(obj.get("scaleX", 1.0))
            scale_y = float(obj.get("scaleY", 1.0))

            # 1. Room Group (Tag 1000 - 1999) or Amenity Group (Tag 4000 - 4999)
            if (1000 <= tag_code < 2000) or (4000 <= tag_code < 5000):
                r_idx = tag_code - 1000 if tag_code < 2000 else tag_code - 4000
                if 0 <= r_idx < len(working_plan.rooms):
                    surviving_room_indices.add(r_idx)
                    r = working_plan.rooms[r_idx]
                    orig_left = to_cx(r.cx)
                    orig_top = to_cy(r.cy)
                    dx = abs(curr_left - orig_left)
                    dy = abs(curr_top - orig_top)
                    d_scale = abs(scale_x - 1.0) > 0.05 or abs(scale_y - 1.0) > 0.05

                    if dx > 4.0 or dy > 4.0 or d_scale:
                        new_cx = from_cx(curr_left)
                        new_cy = from_cy(curr_top)
                        new_w = max(1.2, round(r.w * scale_x, 2))
                        new_h = max(1.2, round(r.h * scale_y, 2))
                        new_x = max(0.0, min(round(working_plan.plot_width - new_w, 2), round(new_cx - new_w / 2, 2)))
                        new_y = max(0.0, min(round(working_plan.plot_height - new_h, 2), round(new_cy - new_h / 2, 2)))

                        shift_x = round(new_x - r.x, 2)
                        shift_y = round(new_y - r.y, 2)
                        r.x = new_x
                        r.y = new_y
                        r.w = new_w
                        r.h = new_h
                        r.area = round(new_w * new_h, 2)

                        if abs(shift_x) > 0.05 or abs(shift_y) > 0.05:
                            for d in working_plan.doors:
                                if d.room_name == r.name:
                                    d.x = round(d.x + shift_x, 2)
                                    d.y = round(d.y + shift_y, 2)
                            for w in working_plan.windows:
                                if w.room_name == r.name:
                                    w.x = round(w.x + shift_x, 2)
                                    w.y = round(w.y + shift_y, 2)

                        plan_changed = True

            # 2. Door Group (Tag 2000 - 2999)
            elif 2000 <= tag_code < 3000:
                d_idx = tag_code - 2000
                if 0 <= d_idx < len(working_plan.doors):
                    surviving_door_indices.add(d_idx)
                    d = working_plan.doors[d_idx]
                    orig_left = to_cx(d.x)
                    orig_top = to_cy(d.y)
                    dx = abs(curr_left - orig_left)
                    dy = abs(curr_top - orig_top)

                    if dx > 4.0 or dy > 4.0:
                        drag_xm = from_cx(curr_left)
                        drag_ym = from_cy(curr_top)
                        best_room, best_side, sx, sy = FloorPlanEditor.snap_point_to_nearest_wall(
                            drag_xm, drag_ym, working_plan.rooms
                        )
                        d.room_name = best_room
                        d.wall_side = best_side
                        d.x = sx
                        d.y = sy
                        plan_changed = True

            # 3. Window Group (Tag 3000 - 3999)
            elif 3000 <= tag_code < 4000:
                w_idx = tag_code - 3000
                if 0 <= w_idx < len(working_plan.windows):
                    surviving_win_indices.add(w_idx)
                    win = working_plan.windows[w_idx]
                    orig_left = to_cx(win.x)
                    orig_top = to_cy(win.y)
                    dx = abs(curr_left - orig_left)
                    dy = abs(curr_top - orig_top)

                    if dx > 4.0 or dy > 4.0:
                        drag_xm = from_cx(curr_left)
                        drag_ym = from_cy(curr_top)
                        best_room, best_side, sx, sy = FloorPlanEditor.snap_point_to_nearest_wall(
                            drag_xm, drag_ym, working_plan.rooms
                        )
                        win.room_name = best_room
                        win.wall_side = best_side
                        win.x = sx
                        win.y = sy
                        plan_changed = True

        # Check if any component was removed on canvas
        if surviving_room_indices and len(surviving_room_indices) < len(working_plan.rooms):
            working_plan.rooms = [r for i, r in enumerate(working_plan.rooms) if i in surviving_room_indices]
            valid_names = {r.name for r in working_plan.rooms}
            working_plan.doors = [d for d in working_plan.doors if d.room_name in valid_names]
            working_plan.windows = [w for w in working_plan.windows if w.room_name in valid_names]
            plan_changed = True

        if surviving_door_indices and len(surviving_door_indices) < len(working_plan.doors):
            working_plan.doors = [d for i, d in enumerate(working_plan.doors) if i in surviving_door_indices]
            plan_changed = True

        if surviving_win_indices and len(surviving_win_indices) < len(working_plan.windows):
            working_plan.windows = [w for i, w in enumerate(working_plan.windows) if i in surviving_win_indices]
            plan_changed = True

        if plan_changed:
            FloorPlanEditor._rebuild_walls_and_dims(working_plan)

        return plan_changed

    @staticmethod
    def extract_edited_rooms(json_data: Dict[str, Any], canvas_h: int) -> List[Dict[str, Any]]:
        """
        Parses Fabric.js json_data from the canvas and extracts updated room coordinates & areas.
        """
        if not json_data or "objects" not in json_data:
            return []

        updated_rooms = []
        for obj in json_data["objects"]:
            o_type = obj.get("type", "").lower()
            if o_type in ("group",) and "strokeDashArray" in obj:
                dash = obj.get("strokeDashArray")
                tag_code = dash[0] if (dash and isinstance(dash, (list, tuple)) and len(dash) > 0) else None
                if tag_code and ((1000 <= tag_code < 2000) or (4000 <= tag_code < 5000)):
                    scale_x = obj.get("scaleX", 1.0)
                    scale_y = obj.get("scaleY", 1.0)
                    raw_w = obj.get("width", 0) * scale_x
                    raw_h = obj.get("height", 0) * scale_y
                    raw_left = obj.get("left", 0)
                    raw_top = obj.get("top", 0)

                    w_m = max(1.0, round(raw_w / PPM_EDITOR, 2))
                    h_m = max(1.0, round(raw_h / PPM_EDITOR, 2))
                    x_m = max(0.0, round((raw_left / PPM_EDITOR) - PAD_M - w_m / 2, 2))
                    y_m = max(0.0, round(((canvas_h - raw_top) / PPM_EDITOR) - PAD_M - h_m / 2, 2))
                    area_m2 = round(w_m * h_m, 2)

                    r_idx = tag_code - 1000 if tag_code < 2000 else tag_code - 4000
                    updated_rooms.append({
                        "name": f"Room {r_idx}",
                        "x": x_m, "y": y_m, "w": w_m, "h": h_m, "area": area_m2
                    })
            elif obj.get("type") == "rect" and "room_name" in obj:
                scale_x = obj.get("scaleX", 1.0)
                scale_y = obj.get("scaleY", 1.0)
                raw_w = obj.get("width", 0) * scale_x
                raw_h = obj.get("height", 0) * scale_y
                raw_left = obj.get("left", 0)
                raw_top = obj.get("top", 0)
                w_m = max(1.0, round(raw_w / PPM_EDITOR, 2))
                h_m = max(1.0, round(raw_h / PPM_EDITOR, 2))
                x_m = max(0.0, round((raw_left / PPM_EDITOR) - PAD_M, 2))
                y_m = max(0.0, round(((canvas_h - (raw_top + raw_h)) / PPM_EDITOR) - PAD_M, 2))
                area_m2 = round(w_m * h_m, 2)
                updated_rooms.append({
                    "name": obj.get("room_name", "Room"),
                    "category": obj.get("category", "BEDROOM"),
                    "x": x_m, "y": y_m, "w": w_m, "h": h_m, "area": area_m2,
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
        # Interactive Controls Bar: Move, Add, Delete & Adjust Dimensions
        # -------------------------------------------------------------
        ctrl_tab_move, ctrl_tab_add, ctrl_tab_del, ctrl_tab_dim = st.tabs([
            "🔄 Move / Re-place Block",
            "➕ Add Component",
            "🗑️ Delete Component",
            "📐 Room Dimensions & Resizing"
        ])

        # -------------------------------------------------------------
        # TAB 1: MOVE / RE-PLACE COMPONENT BLOCK
        # -------------------------------------------------------------
        with ctrl_tab_move:
            move_type = st.radio(
                "Select Component Block to Move / Re-place:",
                ["🚪 Move Door Block", "🪟 Move Window Block", "🪜 Move Entrance Steps", "🏠 Move / Swap Room Block", "🪜 Move Staircase / Lift"],
                horizontal=True,
                key="move_comp_radio"
            )

            if move_type == "🚪 Move Door Block":
                if not working_plan.doors:
                    st.info("No doors currently placed on the floor plan.")
                else:
                    door_labels = []
                    for i, d in enumerate(working_plan.doors):
                        ent_tag = " [★ MAIN ENTRANCE + STEPS]" if d.is_main_entrance else ""
                        door_labels.append(f"Door #{i+1}: {d.room_name} ({d.wall_side} Wall, {d.width:.1f}m, Swing {d.swing_direction}){ent_tag}")

                    selected_door_label = st.selectbox("Select Door Block to Move:", door_labels, key="sel_move_door")
                    door_idx = door_labels.index(selected_door_label)
                    target_d = working_plan.doors[door_idx]

                    st.markdown(
                        f"""
                        <div style="background: rgba(0, 229, 255, 0.08); border-left: 4px solid #00E5FF; padding: 10px 14px; border-radius: 4px; margin-bottom: 12px;">
                            <b>Selected Block:</b> Door #{door_idx+1} for <b>{target_d.room_name}</b> | Current Wall: <b>{target_d.wall_side}</b> | Width: <b>{target_d.width:.1f} m</b> | Position: <code>({target_d.x:.2f}, {target_d.y:.2f})</code>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    d_col1, d_col2, d_col3, d_col4 = st.columns(4)
                    with d_col1:
                        room_names = [r.name for r in working_plan.rooms]
                        curr_room_idx = room_names.index(target_d.room_name) if target_d.room_name in room_names else 0
                        new_d_room = st.selectbox("Destination Room", room_names, index=curr_room_idx, key="move_d_room")
                    with d_col2:
                        wall_options = ["N (North Wall - Top)", "E (East Wall - Right)", "S (South Wall - Bottom)", "W (West Wall - Left)"]
                        wall_map = {"N": 0, "E": 1, "S": 2, "W": 3}
                        curr_wall_idx = wall_map.get(target_d.wall_side, 0)
                        new_d_wall = st.selectbox("Destination Wall", wall_options, index=curr_wall_idx, key="move_d_wall")
                        new_d_side = new_d_wall[0]
                    with d_col3:
                        pos_options = ["Center of Wall", "Left / Bottom Corner", "Right / Top Corner", "Custom Position (Slider)"]
                        new_d_pos = st.selectbox("Position along Wall", pos_options, index=0, key="move_d_pos")
                    with d_col4:
                        new_d_w = st.selectbox(
                            "Door Aperture",
                            [0.8, 0.9, 1.0, 1.2],
                            index=[0.8, 0.9, 1.0, 1.2].index(target_d.width) if target_d.width in [0.8, 0.9, 1.0, 1.2] else 2,
                            format_func=lambda w: f"{w:.1f} m ({w*3.28:.1f} ft)",
                            key="move_d_w"
                        )

                    d_extra1, d_extra2 = st.columns(2)
                    with d_extra1:
                        new_d_swing = st.selectbox(
                            "Swing Direction",
                            ["RIGHT", "LEFT"],
                            index=0 if target_d.swing_direction == "RIGHT" else 1,
                            key="move_d_swing"
                        )
                    with d_extra2:
                        new_is_main = st.checkbox(
                            "★ Set as Main Entrance (+ 3 Flat Steps & Landing)",
                            value=target_d.is_main_entrance,
                            key="move_d_ismain"
                        )

                    custom_offset = 0.0
                    dest_room_obj = next((r for r in working_plan.rooms if r.name == new_d_room), working_plan.rooms[0])
                    wall_len = dest_room_obj.w if new_d_side in ('N', 'S') else dest_room_obj.h
                    if new_d_pos == "Custom Position (Slider)":
                        custom_offset = st.slider(
                            f"Offset along {new_d_side} wall (0 to {wall_len:.2f}m):",
                            min_value=0.3,
                            max_value=max(0.4, float(round(wall_len - 0.3, 2))),
                            value=float(round(wall_len / 2, 2)),
                            step=0.1,
                            key="move_d_custom_slider"
                        )

                    if st.button("🚀 Move Door Block to New Location", type="primary", use_container_width=True, key="btn_move_door_exec"):
                        pos_mode_code = "CENTER" if "Center" in new_d_pos else "LEFT_OR_BOTTOM" if "Left" in new_d_pos else "RIGHT_OR_TOP" if "Right" in new_d_pos else "CUSTOM"
                        hx, hy = FloorPlanEditor.compute_wall_point(dest_room_obj, new_d_side, pos_mode_code, custom_offset)

                        if new_is_main:
                            for d in working_plan.doors:
                                d.is_main_entrance = False

                        target_d.room_name = new_d_room
                        target_d.wall_side = new_d_side
                        target_d.x = hx
                        target_d.y = hy
                        target_d.width = new_d_w
                        target_d.swing_direction = new_d_swing
                        target_d.is_main_entrance = new_is_main

                        FloorPlanEditor._rebuild_walls_and_dims(working_plan)
                        st.session_state.pop('editor_initial_json', None)
                        st.success(f"✅ Successfully moved Door #{door_idx+1} to {new_d_side} wall of {new_d_room}! Whole door assembly (leaf, arc, threshold & steps) relocated.")
                        st.rerun()

            elif move_type == "🪟 Move Window Block":
                if not working_plan.windows:
                    st.info("No windows currently placed on the floor plan.")
                else:
                    win_labels = [
                        f"Window #{i+1}: {w.room_name} ({w.wall_side} Wall, {w.width:.1f}m)"
                        for i, w in enumerate(working_plan.windows)
                    ]
                    selected_win_label = st.selectbox("Select Window Block to Move:", win_labels, key="sel_move_win")
                    win_idx = win_labels.index(selected_win_label)
                    target_w = working_plan.windows[win_idx]

                    st.markdown(
                        f"""
                        <div style="background: rgba(0, 229, 255, 0.08); border-left: 4px solid #00E5FF; padding: 10px 14px; border-radius: 4px; margin-bottom: 12px;">
                            <b>Selected Block:</b> Window #{win_idx+1} for <b>{target_w.room_name}</b> | Current Wall: <b>{target_w.wall_side}</b> | Width: <b>{target_w.width:.1f} m</b> | Position: <code>({target_w.x:.2f}, {target_w.y:.2f})</code>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    w_col1, w_col2, w_col3, w_col4 = st.columns(4)
                    with w_col1:
                        room_names = [r.name for r in working_plan.rooms]
                        curr_room_idx = room_names.index(target_w.room_name) if target_w.room_name in room_names else 0
                        new_w_room = st.selectbox("Destination Room", room_names, index=curr_room_idx, key="move_w_room")
                    with w_col2:
                        wall_options = ["N (North Wall - Top)", "E (East Wall - Right)", "S (South Wall - Bottom)", "W (West Wall - Left)"]
                        wall_map = {"N": 0, "E": 1, "S": 2, "W": 3}
                        curr_wall_idx = wall_map.get(target_w.wall_side, 0)
                        new_w_wall = st.selectbox("Destination Wall", wall_options, index=curr_wall_idx, key="move_w_wall")
                        new_w_side = new_w_wall[0]
                    with w_col3:
                        pos_options = ["Center of Wall", "Left / Bottom Corner", "Right / Top Corner", "Custom Position (Slider)"]
                        new_w_pos = st.selectbox("Position along Wall", pos_options, index=0, key="move_w_pos")
                    with w_col4:
                        new_w_w = st.selectbox(
                            "Window Width",
                            [1.0, 1.2, 1.5, 1.8, 2.0],
                            index=[1.0, 1.2, 1.5, 1.8, 2.0].index(target_w.width) if target_w.width in [1.0, 1.2, 1.5, 1.8, 2.0] else 2,
                            format_func=lambda w: f"{w:.1f} m ({w*3.28:.1f} ft)",
                            key="move_w_w"
                        )

                    custom_w_offset = 0.0
                    dest_room_obj = next((r for r in working_plan.rooms if r.name == new_w_room), working_plan.rooms[0])
                    wall_len = dest_room_obj.w if new_w_side in ('N', 'S') else dest_room_obj.h
                    if new_w_pos == "Custom Position (Slider)":
                        custom_w_offset = st.slider(
                            f"Offset along {new_w_side} wall (0 to {wall_len:.2f}m):",
                            min_value=0.3,
                            max_value=max(0.4, float(round(wall_len - 0.3, 2))),
                            value=float(round(wall_len / 2, 2)),
                            step=0.1,
                            key="move_w_custom_slider"
                        )

                    if st.button("🚀 Move Window Block to New Wall", type="primary", use_container_width=True, key="btn_move_win_exec"):
                        pos_mode_code = "CENTER" if "Center" in new_w_pos else "LEFT_OR_BOTTOM" if "Left" in new_w_pos else "RIGHT_OR_TOP" if "Right" in new_w_pos else "CUSTOM"
                        wx, wy = FloorPlanEditor.compute_wall_point(dest_room_obj, new_w_side, pos_mode_code, custom_w_offset)

                        target_w.room_name = new_w_room
                        target_w.wall_side = new_w_side
                        target_w.x = wx
                        target_w.y = wy
                        target_w.width = new_w_w

                        FloorPlanEditor._rebuild_walls_and_dims(working_plan)
                        st.session_state.pop('editor_initial_json', None)
                        st.success(f"✅ Successfully moved Window #{win_idx+1} to {new_w_side} wall of {new_w_room}! Entire double glazing & sill relocated.")
                        st.rerun()

            elif move_type == "🪜 Move Entrance Steps":
                st.markdown("##### Relocate Entrance Steps Block (Landing + Flat Treads + UP Arrow)")
                current_main = next((d for d in working_plan.doors if d.is_main_entrance), None)
                curr_side = current_main.wall_side if current_main else "E"
                
                st.info(f"📍 Currently entrance steps are positioned on: **{curr_side} Wall** ({current_main.room_name if current_main else 'None'})")

                st_col1, st_col2, st_col3 = st.columns(3)
                with st_col1:
                    step_wall = st.selectbox(
                        "Move Entrance Steps to Wall:",
                        ["N (North Wall - Top)", "E (East Wall - Right)", "S (South Wall - Bottom)", "W (West Wall - Left)"],
                        index={"N": 0, "E": 1, "S": 2, "W": 3}.get(curr_side, 1),
                        key="move_step_wall_sel"
                    )
                    new_step_side = step_wall[0]
                with st_col2:
                    step_treads = st.selectbox("Number of Flat Treads", [3, 4, 5, 6], index=0, key="move_step_treads_sel")
                with st_col3:
                    step_landing = st.selectbox("Landing Depth", [1.2, 1.5, 1.8], format_func=lambda d: f"{d:.1f} m", key="move_step_landing_sel")

                if st.button("🚀 Relocate Entrance Steps Block", type="primary", use_container_width=True, key="btn_move_steps_exec"):
                    for d in working_plan.doors:
                        d.is_main_entrance = False
                    
                    target_door = next((d for d in working_plan.doors if d.wall_side == new_step_side), None)
                    if target_door:
                        target_door.is_main_entrance = True
                        st.success(f"✅ Attached entrance steps to door on {new_step_side} wall of {target_door.room_name}!")
                    else:
                        candidate_rooms = working_plan.rooms
                        if new_step_side == 'N':
                            r = max(candidate_rooms, key=lambda rm: rm.top)
                        elif new_step_side == 'S':
                            r = min(candidate_rooms, key=lambda rm: rm.y)
                        elif new_step_side == 'E':
                            r = max(candidate_rooms, key=lambda rm: rm.right)
                        else:
                            r = min(candidate_rooms, key=lambda rm: rm.x)
                        
                        hx, hy = FloorPlanEditor.compute_wall_point(r, new_step_side, "CENTER")
                        working_plan.doors.append(DoorPlacement(
                            x=hx, y=hy, width=1.2, wall_side=new_step_side,
                            swing_direction="RIGHT", room_name=r.name, is_main_entrance=True
                        ))
                        st.success(f"✅ Relocated entrance steps with new entrance door on {new_step_side} wall of {r.name}!")
                    
                    FloorPlanEditor._rebuild_walls_and_dims(working_plan)
                    st.session_state.pop('editor_initial_json', None)
                    st.rerun()

            elif move_type == "🏠 Move / Swap Room Block":
                st.markdown("##### Relocate or Swap Complete Room Blocks (Walls, Doors, Labels & Fittings stay unified)")
                room_names = [r.name for r in working_plan.rooms]
                if not room_names:
                    st.info("No rooms on the floor plan.")
                else:
                    rm_action = st.radio(
                        "Room Action:",
                        ["🔄 Swap Two Rooms (Instant Vastu Flip)", "📍 Shift / Nudge Room by Meters", "🎯 Set Exact Coordinates (X, Y)"],
                        horizontal=True,
                        key="rm_action_radio"
                    )

                    if rm_action == "🔄 Swap Two Rooms (Instant Vastu Flip)":
                        sw_c1, sw_c2 = st.columns(2)
                        with sw_c1:
                            r1_name = st.selectbox("Select Room A to Swap", room_names, index=0, key="swap_r1")
                        with sw_c2:
                            r2_candidates = [n for n in room_names if n != r1_name]
                            r2_name = st.selectbox("Select Room B to Swap With", r2_candidates if r2_candidates else room_names, index=0, key="swap_r2")

                        r1 = next(r for r in working_plan.rooms if r.name == r1_name)
                        r2 = next(r for r in working_plan.rooms if r.name == r2_name)

                        st.caption(f"🔄 Swapping `{r1.name}` at `({r1.x:.2f}, {r1.y:.2f})` with `{r2.name}` at `({r2.x:.2f}, {r2.y:.2f})`")

                        if st.button(f"🔄 Swap '{r1.name}' ↔ '{r2.name}'", type="primary", use_container_width=True, key="btn_swap_rooms"):
                            dx1, dy1 = r2.x - r1.x, r2.y - r1.y
                            dx2, dy2 = r1.x - r2.x, r1.y - r2.y

                            r1.x, r2.x = r2.x, r1.x
                            r1.y, r2.y = r2.y, r1.y

                            for d in working_plan.doors:
                                if d.room_name == r1.name:
                                    d.x = round(d.x + dx1, 2); d.y = round(d.y + dy1, 2)
                                elif d.room_name == r2.name:
                                    d.x = round(d.x + dx2, 2); d.y = round(d.y + dy2, 2)

                            for w in working_plan.windows:
                                if w.room_name == r1.name:
                                    w.x = round(w.x + dx1, 2); w.y = round(w.y + dy1, 2)
                                elif w.room_name == r2.name:
                                    w.x = round(w.x + dx2, 2); w.y = round(w.y + dy2, 2)

                            FloorPlanEditor._rebuild_walls_and_dims(working_plan)
                            st.session_state.pop('editor_initial_json', None)
                            st.success(f"✅ Successfully swapped '{r1.name}' and '{r2.name}'!")
                            st.rerun()

                    elif rm_action == "📍 Shift / Nudge Room by Meters":
                        target_rm_name = st.selectbox("Select Room Block to Nudge", room_names, key="nudge_rm_sel")
                        target_rm = next(r for r in working_plan.rooms if r.name == target_rm_name)
                        
                        st.markdown(f"**Current Position:** `X = {target_rm.x:.2f} m`, `Y = {target_rm.y:.2f} m` (Size: `{target_rm.w:.2f} × {target_rm.h:.2f} m`)")
                        
                        nudge_step = st.select_slider("Nudge Step Distance (Meters)", options=[0.2, 0.5, 1.0, 1.5, 2.0], value=0.5, key="nudge_dist_slider")
                        
                        n_c1, n_c2, n_c3, n_c4 = st.columns(4)
                        if n_c1.button(f"⬆️ North (+{nudge_step}m)", use_container_width=True, key="btn_nudge_n"):
                            target_rm.y = round(target_rm.y + nudge_step, 2)
                            for d in working_plan.doors:
                                if d.room_name == target_rm.name: d.y = round(d.y + nudge_step, 2)
                            for w in working_plan.windows:
                                if w.room_name == target_rm.name: w.y = round(w.y + nudge_step, 2)
                            FloorPlanEditor._rebuild_walls_and_dims(working_plan)
                            st.session_state.pop('editor_initial_json', None)
                            st.rerun()

                        if n_c2.button(f"⬇️ South (-{nudge_step}m)", use_container_width=True, key="btn_nudge_s"):
                            target_rm.y = max(0.0, round(target_rm.y - nudge_step, 2))
                            for d in working_plan.doors:
                                if d.room_name == target_rm.name: d.y = max(0.0, round(d.y - nudge_step, 2))
                            for w in working_plan.windows:
                                if w.room_name == target_rm.name: w.y = max(0.0, round(w.y - nudge_step, 2))
                            FloorPlanEditor._rebuild_walls_and_dims(working_plan)
                            st.session_state.pop('editor_initial_json', None)
                            st.rerun()

                        if n_c3.button(f"⬅️ West (-{nudge_step}m)", use_container_width=True, key="btn_nudge_w"):
                            target_rm.x = max(0.0, round(target_rm.x - nudge_step, 2))
                            for d in working_plan.doors:
                                if d.room_name == target_rm.name: d.x = max(0.0, round(d.x - nudge_step, 2))
                            for w in working_plan.windows:
                                if w.room_name == target_rm.name: w.x = max(0.0, round(w.x - nudge_step, 2))
                            FloorPlanEditor._rebuild_walls_and_dims(working_plan)
                            st.session_state.pop('editor_initial_json', None)
                            st.rerun()

                        if n_c4.button(f"➡️ East (+{nudge_step}m)", use_container_width=True, key="btn_nudge_e"):
                            target_rm.x = round(target_rm.x + nudge_step, 2)
                            for d in working_plan.doors:
                                if d.room_name == target_rm.name: d.x = round(d.x + nudge_step, 2)
                            for w in working_plan.windows:
                                if w.room_name == target_rm.name: w.x = round(w.x + nudge_step, 2)
                            FloorPlanEditor._rebuild_walls_and_dims(working_plan)
                            st.session_state.pop('editor_initial_json', None)
                            st.rerun()

                    else: # 🎯 Set Exact Coordinates (X, Y)
                        target_rm_name = st.selectbox("Select Room Block", room_names, key="coord_rm_sel")
                        target_rm = next(r for r in working_plan.rooms if r.name == target_rm_name)

                        co_c1, co_c2 = st.columns(2)
                        with co_c1:
                            new_x = st.number_input("X Coordinate (Meters from West)", min_value=0.0, max_value=float(working_plan.plot_width), value=float(target_rm.x), step=0.1, key="input_rm_x")
                        with co_c2:
                            new_y = st.number_input("Y Coordinate (Meters from South)", min_value=0.0, max_value=float(working_plan.plot_height), value=float(target_rm.y), step=0.1, key="input_rm_y")

                        if st.button("🚀 Set Exact Room Coordinates", type="primary", use_container_width=True, key="btn_set_rm_coords"):
                            dx = round(new_x - target_rm.x, 2)
                            dy = round(new_y - target_rm.y, 2)
                            target_rm.x = round(new_x, 2)
                            target_rm.y = round(new_y, 2)
                            for d in working_plan.doors:
                                if d.room_name == target_rm.name:
                                    d.x = round(d.x + dx, 2); d.y = round(d.y + dy, 2)
                            for w in working_plan.windows:
                                if w.room_name == target_rm.name:
                                    w.x = round(w.x + dx, 2); w.y = round(w.y + dy, 2)
                            FloorPlanEditor._rebuild_walls_and_dims(working_plan)
                            st.session_state.pop('editor_initial_json', None)
                            st.success(f"✅ Positioned '{target_rm.name}' at `({new_x:.2f}, {new_y:.2f}) m`!")
                            st.rerun()

            elif move_type == "🪜 Move Staircase / Lift":
                st.markdown("##### Relocate Staircase Core or Lift Shaft Block")
                stair_rooms = [r for r in working_plan.rooms if getattr(r, 'category', '') in ('STAIRS', 'LIFT')]
                
                if not stair_rooms:
                    st.info("No dedicated staircase or lift room currently on the floor plan.")
                    if st.button("➕ Place Dog-Leg Staircase in South-West (Vastu Recommended)", type="secondary", key="btn_place_stair_init"):
                        working_plan.rooms.append(RoomRect(
                            name="Staircase", category="STAIRS",
                            x=0.0, y=0.0, w=1.5, h=3.0,
                            color="#64748B", area=4.5
                        ))
                        FloorPlanEditor._rebuild_walls_and_dims(working_plan)
                        st.session_state.pop('editor_initial_json', None)
                        st.rerun()
                else:
                    stair_names = [f"{r.name} ({r.w:.2f}×{r.h:.2f}m)" for r in stair_rooms]
                    sel_stair_label = st.selectbox("Select Stair / Lift to Relocate", stair_names, key="sel_stair_reloc")
                    target_stair = stair_rooms[stair_names.index(sel_stair_label)]

                    st_c1, st_c2 = st.columns(2)
                    with st_c1:
                        zone_choice = st.selectbox(
                            "Target Vastu Zone:",
                            ["South-West (SW — Heavy Zone, Ideal for Stairs)", "South Core (S)", "West Wall (W)", "North-West (NW)"],
                            key="stair_zone_choice"
                        )
                    with st_c2:
                        stair_w = st.number_input("Width (m)", min_value=1.0, max_value=4.0, value=float(target_stair.w), step=0.1, key="stair_w_input")
                        stair_h = st.number_input("Length (m)", min_value=1.5, max_value=6.0, value=float(target_stair.h), step=0.1, key="stair_h_input")

                    if st.button("🚀 Relocate Staircase / Lift Block", type="primary", use_container_width=True, key="btn_reloc_stair_exec"):
                        if "South-West" in zone_choice:
                            new_sx = 0.0
                            new_sy = 0.0
                        elif "South" in zone_choice:
                            new_sx = round(working_plan.plot_width / 2 - stair_w / 2, 2)
                            new_sy = 0.0
                        elif "West" in zone_choice:
                            new_sx = 0.0
                            new_sy = round(working_plan.plot_height / 2 - stair_h / 2, 2)
                        else: # NW
                            new_sx = 0.0
                            new_sy = round(working_plan.plot_height - stair_h, 2)

                        target_stair.x = new_sx
                        target_stair.y = new_sy
                        target_stair.w = stair_w
                        target_stair.h = stair_h
                        target_stair.area = round(stair_w * stair_h, 2)

                        FloorPlanEditor._rebuild_walls_and_dims(working_plan)
                        st.session_state.pop('editor_initial_json', None)
                        st.success(f"✅ Relocated '{target_stair.name}' to {zone_choice.split('(')[0]}!")
                        st.rerun()

        # -------------------------------------------------------------
        # TAB 2: ADD COMPONENTS
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
                        hx, hy = FloorPlanEditor.compute_wall_point(target_r, d_side, "CENTER")

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
                        FloorPlanEditor._rebuild_walls_and_dims(working_plan)
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
                        hx, hy = FloorPlanEditor.compute_wall_point(first_r, step_side, "CENTER")

                        working_plan.doors.append(DoorPlacement(
                            x=round(hx, 2), y=round(hy, 2), width=1.0,
                            wall_side=step_side, swing_direction="RIGHT",
                            room_name=first_r.name, is_main_entrance=True
                        ))
                    FloorPlanEditor._rebuild_walls_and_dims(working_plan)
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
                        wx, wy = FloorPlanEditor.compute_wall_point(target_r, win_side, "CENTER")

                        working_plan.windows.append(WindowPlacement(
                            x=round(wx, 2), y=round(wy, 2), width=win_w, height=1.2,
                            wall_side=win_side, room_name=w_room
                        ))
                        FloorPlanEditor._rebuild_walls_and_dims(working_plan)
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
                    FloorPlanEditor._rebuild_walls_and_dims(working_plan)
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
        # TAB 3: DELETE COMPONENTS
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
                        FloorPlanEditor._rebuild_walls_and_dims(working_plan)
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
                        FloorPlanEditor._rebuild_walls_and_dims(working_plan)
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
                        FloorPlanEditor._rebuild_walls_and_dims(working_plan)
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
                        FloorPlanEditor._rebuild_walls_and_dims(working_plan)
                        st.session_state.pop('editor_initial_json', None)
                        st.success(f"🗑️ Removed window on {removed_win.wall_side} wall of {removed_win.room_name}!")
                        st.rerun()
                else:
                    st.info("No windows on the floor plan.")

        # -------------------------------------------------------------
        # TAB 4: ADJUST ROOM DIMENSIONS
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
                        FloorPlanEditor._rebuild_walls_and_dims(working_plan)
                        st.session_state.pop('editor_initial_json', None)
                        st.rerun()
                    if w_cols[1].button("−0.1m", key="w_m1"):
                        target_room.w = max(1.2, round(target_room.w - 0.1, 2))
                        target_room.area = round(target_room.w * target_room.h, 2)
                        FloorPlanEditor._rebuild_walls_and_dims(working_plan)
                        st.session_state.pop('editor_initial_json', None)
                        st.rerun()
                    if w_cols[2].button("+0.1m", key="w_p1"):
                        target_room.w = round(target_room.w + 0.1, 2)
                        target_room.area = round(target_room.w * target_room.h, 2)
                        FloorPlanEditor._rebuild_walls_and_dims(working_plan)
                        st.session_state.pop('editor_initial_json', None)
                        st.rerun()
                    if w_cols[3].button("+0.5m", key="w_p5"):
                        target_room.w = round(target_room.w + 0.5, 2)
                        target_room.area = round(target_room.w * target_room.h, 2)
                        FloorPlanEditor._rebuild_walls_and_dims(working_plan)
                        st.session_state.pop('editor_initial_json', None)
                        st.rerun()

                with sc3:
                    st.markdown(f"**Length / Height:** `{target_room.h:.2f} m` ({target_room.h * 3.28084:.1f} ft)")
                    h_cols = st.columns(4)
                    if h_cols[0].button("−0.5m", key="h_m5"):
                        target_room.h = max(1.2, round(target_room.h - 0.5, 2))
                        target_room.area = round(target_room.w * target_room.h, 2)
                        FloorPlanEditor._rebuild_walls_and_dims(working_plan)
                        st.session_state.pop('editor_initial_json', None)
                        st.rerun()
                    if h_cols[1].button("−0.1m", key="h_m1"):
                        target_room.h = max(1.2, round(target_room.h - 0.1, 2))
                        target_room.area = round(target_room.w * target_room.h, 2)
                        FloorPlanEditor._rebuild_walls_and_dims(working_plan)
                        st.session_state.pop('editor_initial_json', None)
                        st.rerun()
                    if h_cols[2].button("+0.1m", key="h_p1"):
                        target_room.h = round(target_room.h + 0.1, 2)
                        target_room.area = round(target_room.w * target_room.h, 2)
                        FloorPlanEditor._rebuild_walls_and_dims(working_plan)
                        st.session_state.pop('editor_initial_json', None)
                        st.rerun()
                    if h_cols[3].button("+0.5m", key="h_p5"):
                        target_room.h = round(target_room.h + 0.5, 2)
                        target_room.area = round(target_room.w * target_room.h, 2)
                        FloorPlanEditor._rebuild_walls_and_dims(working_plan)
                        st.session_state.pop('editor_initial_json', None)
                        st.rerun()

        st.write("")

        # -------------------------------------------------------------
        # Canvas Toolbar & Freehand Tools
        # -------------------------------------------------------------
        tool_col1, tool_col2, tool_col3, tool_col4 = st.columns([1.5, 1, 1, 1])
        with tool_col1:
            draw_mode = st.selectbox(
                "Drawing / Annotation Tool",
                ["Freedraw (Pencil Notes)", "Draw Line", "Draw Rect"],
                index=0,
                help="Sketch freehand annotations or markup directly on the blueprint. To move components free-hand, toggle the Edit pencil tool (✏️) on the top canvas toolbar!"
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

        # Free-hand Component Interaction Guidance
        st.info(
            "🖱️ **Free-Hand Component Editing Playground:** Click the **Edit tool (✏️ pencil icon)** on the "
            "canvas toolbar to select and drag components directly! "
            "🚪 **Click any Door** → Full door block with swing arc, handle & entrance steps is selected and dragged together (auto-snaps to nearest wall). "
            "🪟 **Click any Window** → Complete window block is selected and dragged together. "
            "🏠 **Click any Room** → Complete room block with dimensions & label is selected together. "
            "🧱 **Walls & CAD dimensions** remain strictly locked and recalculate automatically!"
        )

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
        if canvas_result and canvas_result.json_data:
            if FloorPlanEditor.sync_canvas_drag_edits(canvas_result.json_data, working_plan, canvas_w, canvas_h):
                st.session_state['editor_initial_json'] = FloorPlanEditor.plan_to_fabric_json(
                    working_plan, canvas_w, canvas_h
                )
                st.rerun()

        # Build active_room_list directly from working_plan
        active_room_list = [
            {
                "name": r.name,
                "category": getattr(r, 'category', 'BEDROOM'),
                "x": r.x,
                "y": r.y,
                "w": r.w,
                "h": r.h,
                "area": r.area,
                "color": getattr(r, 'color', '#94A3B8')
            }
            for r in working_plan.rooms
        ]

        # -------------------------------------------------------------
        # Live-Sync HUD & Dynamic Side Meters Calculations
        # -------------------------------------------------------------
        with col_hud:
            st.markdown("#### ⚡ Dynamic Side Meters & Property HUD")

            site_cats = ('PARKING', 'BIKE_PARKING', 'GARDEN', 'POOL', 'WATER_TANK', 'GENERATOR', 'SECURITY')
            house_rooms = [r for r in active_room_list
                           if r["category"] not in site_cats and "External" not in r["name"]]
            tot_carpet = sum(r["area"] for r in house_rooms)
            # Keep the generator's plinth-to-carpet ratio (envelope walls) for the live built-up figure
            bu_ratio = (current_plan.total_built_up_area / current_plan.total_carpet_area
                        if current_plan.total_carpet_area else 1.22)
            tot_built = round(tot_carpet * bu_ratio, 1)
            tot_sqft = tot_carpet * 10.764
            tot_cents = tot_sqft / 435.6

            # Live Vastu Score — same audit as the generator (building mandala, pada, Brahmasthan)
            from floorplan_generator.layout_engine import audit_vastu
            live_rooms = [RoomRect(name=r["name"], category=r["category"], x=r["x"], y=r["y"],
                                   w=r["w"], h=r["h"], color=r["color"], area=r["area"],
                                   role=r.get("role", "")) for r in active_room_list]
            live_vastu, _ = audit_vastu(
                live_rooms, working_plan.doors, working_plan.plot_width, working_plan.plot_height,
                working_plan.orientation, None, getattr(working_plan, "setbacks", {}) or {},
                working_plan.vastu_enabled)
            v_engine = VastuEngine.for_plan(working_plan, scope="building")

            # Live Cost Computation (UI labels are normalised inside CostEngine)
            live_cost = CostEngine.estimate_cost(
                area_sqm=tot_built,
                city_tier=cost_params.get("city_tier", "Tier-2 (Urban)"),
                quality_level=cost_params.get("quality", "Standard"),
                num_floors=cost_params.get("floors", 1),
                country=cost_params.get("country"),
            )

            # Metrics Display
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.metric("Live Carpet Area", f"{tot_carpet:.1f} m²", f"{tot_sqft:.0f} sqft ({tot_cents:.2f} C)")
                st.metric("Live Vastu Score", f"{live_vastu:.0f}/100", f"{live_vastu - current_plan.vastu_score:+.0f} pts")
            with m_col2:
                st.metric("Live Built-up Area", f"{tot_built:.1f} m²", f"{tot_built * 10.764:.0f} sq ft")
                st.metric("Live Est. Cost", live_cost['display']['total_compact'], live_cost['display']['rate'])

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
