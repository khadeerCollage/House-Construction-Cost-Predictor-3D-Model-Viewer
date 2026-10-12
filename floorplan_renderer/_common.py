"""
Shared drawing conventions for all renderers.

Plan frame: metres, +x right, +y up, road/entrance side at the top (y = plot_height).
Door convention (DoorPlacement): hinge at (x, y); swing 'RIGHT' opens along +x
(N/S walls) or +y (E/W walls); 'LEFT' the opposite way; the leaf swings into
the room that owns the door.
"""

import math
from typing import List, Tuple

from floorplan_generator.vastu_engine import north_edge_for_facing

TIER_OFFSETS = {1: 0.8, 2: 1.6, 3: 2.4}


def north_angle(plan) -> float:
    """Clockwise rotation (degrees) of true North from the drawing's 'up' direction."""
    edge = getattr(plan, "north_edge", "") or north_edge_for_facing(getattr(plan, "orientation", "N"))
    return {"N": 0.0, "E": 90.0, "S": 180.0, "W": 270.0}.get(edge, 0.0)


def dim_offset(dim) -> float:
    """Signed offset (plan metres) for a dimension line: tier spacing, generator's side."""
    mag = TIER_OFFSETS.get(dim.tier, abs(dim.offset) or 0.8)
    return -mag if dim.offset < 0 else mag


def door_span(d) -> Tuple[float, float]:
    """(start, end) of the opening along its wall in plan coordinates."""
    if d.wall_side in ("N", "S"):
        a = d.x if d.swing_direction != "LEFT" else d.x - d.width
    else:
        a = d.y if d.swing_direction != "LEFT" else d.y - d.width
    return a, a + d.width


def door_arc(d) -> Tuple[float, float, Tuple[float, float]]:
    """(start_deg, end_deg CCW, leaf_end_xy) of the swing arc centred on the hinge."""
    right = d.swing_direction != "LEFT"
    side = d.wall_side
    if side == "N":
        start, end, open_ang = (270, 360, 270) if right else (180, 270, 270)
    elif side == "S":
        start, end, open_ang = (0, 90, 90) if right else (90, 180, 90)
    elif side == "E":
        start, end, open_ang = (90, 180, 180) if right else (180, 270, 180)
    else:  # W
        start, end, open_ang = (0, 90, 0) if right else (270, 360, 0)
    lx = d.x + math.cos(math.radians(open_ang)) * d.width
    ly = d.y + math.sin(math.radians(open_ang)) * d.width
    return start, end, (lx, ly)


def entrance_steps(d, landing: float = 1.2, step: float = 0.3, n_steps: int = 3,
                   side_ext: float = 0.3):
    """
    Landing + flat steps outside a main door (plan coordinates).

    Returns (landing_polygon, step_lines, arrow_from, arrow_to).
    """
    a, b = door_span(d)
    out = {"N": (0, 1), "S": (0, -1), "E": (1, 0), "W": (-1, 0)}[d.wall_side]
    if d.wall_side in ("N", "S"):
        x0, x1 = a - side_ext, b + side_ext
        y0 = d.y
        sgn = out[1]
        land = [(x0, y0), (x1, y0), (x1, y0 + sgn * landing), (x0, y0 + sgn * landing)]
        lines = [((x0, y0 + sgn * (landing + i * step)), (x1, y0 + sgn * (landing + i * step)))
                 for i in range(1, n_steps + 1)]
        lines += [((x0, y0 + sgn * landing), (x0, y0 + sgn * (landing + n_steps * step))),
                  ((x1, y0 + sgn * landing), (x1, y0 + sgn * (landing + n_steps * step)))]
        mx = (x0 + x1) / 2
        arrow = ((mx, y0 + sgn * (landing + n_steps * step + 0.2)), (mx, y0 + sgn * landing * 0.5))
    else:
        y0, y1 = a - side_ext, b + side_ext
        x0 = d.x
        sgn = out[0]
        land = [(x0, y0), (x0, y1), (x0 + sgn * landing, y1), (x0 + sgn * landing, y0)]
        lines = [((x0 + sgn * (landing + i * step), y0), (x0 + sgn * (landing + i * step), y1))
                 for i in range(1, n_steps + 1)]
        lines += [((x0 + sgn * landing, y0), (x0 + sgn * (landing + n_steps * step), y0)),
                  ((x0 + sgn * landing, y1), (x0 + sgn * (landing + n_steps * step), y1))]
        my = (y0 + y1) / 2
        arrow = ((x0 + sgn * (landing + n_steps * step + 0.2), my), (x0 + sgn * landing * 0.5, my))
    return land, lines, arrow[0], arrow[1]


LABEL_ABBREVIATIONS = {
    "Attached Bathroom": "ATT. BATH", "Common Bathroom": "C. BATH", "Bathroom": "BATH",
    "Dress / Wardrobe": "DRESS", "Wardrobe": "WARDROBE", "Children's Bedroom": "CHILD BED",
    "Elders' Bedroom": "ELDERS' BED", "Guest Bedroom": "GUEST BED", "Master Bedroom": "MASTER BED",
    "Utility / Wash": "UTILITY", "Utility / Balcony": "UTIL / BALC", "Utility": "UTIL.",
    "Terrace Staircase (External)": "TERRACE STAIR", "Study / Home Office": "STUDY",
    "Lift Machine / Duct": "DUCT", "Servant Quarter": "SQ", "Servant Toilet": "S. WC",
    "Dining & Family Lounge": "DINING & LOUNGE", "Living & Dining": "LIVING & DINING",
    "Lift / Elevator": "LIFT", "Store Room": "STORE", "Pooja Room": "POOJA", "Corridor": "CORR.",
    "Balcony": "BALC.", "Passage / Foyer": "PASSAGE",
}


def fit_label(name: str, avail: float, font: float, char_w: float = 0.62,
              min_font: float = 6.0) -> Tuple[str, float]:
    """
    Pick a room label and font size that fit `avail` (same units as font).
    Uses a standard abbreviation when the full name is too wide, then shrinks the font.
    """
    import re
    text = name.upper()
    if len(text) * font * char_w > avail:
        base = re.sub(r"\s+\d+$", "", name)
        suffix = name[len(base):].strip()
        short = LABEL_ABBREVIATIONS.get(base)
        if short:
            text = short + (f" {suffix}" if suffix else "")
    if len(text) * font * char_w > avail and len(text) > 0:
        font = max(min_font, avail / (len(text) * char_w))
    return text, font


def rotate_point(px: float, py: float, cx: float, cy: float, deg_cw: float, y_up: bool = True):
    """Rotate (px, py) around (cx, cy) clockwise by deg_cw (in a y-up or y-down frame)."""
    a = math.radians(deg_cw)
    dx, dy = px - cx, py - cy
    if y_up:
        return cx + dx * math.cos(a) + dy * math.sin(a), cy - dx * math.sin(a) + dy * math.cos(a)
    return cx + dx * math.cos(a) - dy * math.sin(a), cy + dx * math.sin(a) + dy * math.cos(a)
