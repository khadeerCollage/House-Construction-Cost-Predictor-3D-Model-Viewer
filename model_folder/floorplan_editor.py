"""
Floor Plan Editor (Design tab editing section)
==============================================
The editing section is the Smart Re-Plan studio (model_folder/intent_studio.py): owners
change the generated house by stating wishes — tap a room and an action, pick a Vastu
zone, or type in their own words — and the whole layout is re-solved into complete,
valid options with Vastu and cost impact.

This module keeps the Fabric.js export of a plan (`plan_to_fabric_json`, the blueprint
drawn as canvas objects with identity tags) and `extract_edited_rooms`, which other tools
and tests use. The literal Move / Add / Delete / Resize tabs and canvas drag-and-drop were
retired on 2026-10-12.
"""

from typing import Dict, List, Any

from floorplan_generator.layout_engine import GeneratedFloorPlan


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
        The editing section of the Design tab: the Smart Re-Plan studio. Owners change the
        house by stating wishes (tap a room + an action, a Vastu zone grid, or plain words);
        the whole layout is re-solved and complete options are offered. The older literal
        Move / Add / Delete / Resize tabs and canvas drag-and-drop were retired (2026-10-12):
        they broke walls and doors and were hard to use for many owners.
        """
        import importlib     # import_module (not `from … import`) so an edited studio is re-imported on rerun
        importlib.import_module("model_folder.intent_studio").render_intent_studio(current_plan, cost_params)
