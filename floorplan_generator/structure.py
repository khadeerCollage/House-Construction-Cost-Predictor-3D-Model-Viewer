"""
Structural grid for a generated plan
====================================
Places an RCC column grid on the room layout the way a structural engineer
sets out a residential frame:

  * grid lines run on wall centre-lines (columns hide inside walls)
  * the footprint edges are always grid lines
  * intermediate lines are chosen so bays stay between MIN_BAY and MAX_BAY,
    preferring the lines that carry the longest walls (beam support)
  * a column sits at every grid intersection that lies on a wall

Shared by the engineering report (column count) and the renderers (drawing).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

MIN_BAY = 2.4     # m — closer columns are not worth the formwork
MAX_BAY = 4.5     # m — economical RCC beam span for houses (230 × 375–450 beams)
COL_SHORT = 0.23  # m — column width (matches the 230 mm envelope wall)
COL_LONG = 0.30   # m — column depth

SITE_CATEGORIES = {"PARKING", "BIKE_PARKING", "GARDEN", "POOL", "WATER_TANK", "GENERATOR", "SECURITY"}


@dataclass
class StructuralGrid:
    xs: List[float] = field(default_factory=list)          # vertical grid lines (left → right)
    ys: List[float] = field(default_factory=list)          # horizontal grid lines (bottom → top)
    columns: List[Tuple[float, float, float, float]] = field(default_factory=list)  # x0, y0, x1, y1
    footprint: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)            # x, y, w, h

    @property
    def x_labels(self) -> List[str]:
        return [chr(ord("A") + i) if i < 26 else f"A{i - 25}" for i in range(len(self.xs))]

    @property
    def y_labels(self) -> List[str]:
        # numbered from the road side (plan top) towards the rear
        n = len(self.ys)
        return [str(n - i) for i in range(n)]

    @property
    def max_bay(self) -> float:
        bays = [b - a for a, b in zip(self.xs, self.xs[1:])] + [b - a for a, b in zip(self.ys, self.ys[1:])]
        return max(bays) if bays else 0.0


def _building(rooms):
    return [r for r in rooms if getattr(r, "category", "") not in SITE_CATEGORIES and r.w > 0 and r.h > 0]


def building_footprint(rooms, footprint: Optional[Tuple[float, float, float, float]] = None):
    b = _building(rooms)
    if not b:
        return footprint or (0.0, 0.0, 0.0, 0.0)
    x0, y0 = min(r.x for r in b), min(r.y for r in b)
    x1, y1 = max(r.right for r in b), max(r.top for r in b)
    return (x0, y0, x1 - x0, y1 - y0)


def _cluster(vals, tol=0.12):
    out: List[float] = []
    for v in sorted(vals):
        if out and v - out[-1] < tol:
            continue
        out.append(round(v, 3))
    return out


def _pick_lines(cands, lo, hi, weight):
    chosen = [lo]
    inner = [v for v in cands if lo + 0.05 < v < hi - 0.05]
    while hi - chosen[-1] > MAX_BAY + 1e-6:
        last = chosen[-1]
        window = [v for v in inner if last + MIN_BAY <= v <= last + MAX_BAY and hi - v >= MIN_BAY]
        if not window:
            nxt = [v for v in inner if v >= last + MIN_BAY and hi - v >= MIN_BAY]
            if not nxt:
                break
            window = [nxt[0]]
        chosen.append(max(window, key=lambda v: (round(weight(v), 1), v)))
    chosen.append(hi)
    return chosen


def structural_grid(rooms, footprint=None, tol: float = 0.03, x_lines=None) -> StructuralGrid:
    """
    x_lines: column lines across the frontage set out for parking under a stilt building (apartment.py,
    core at the rear) — between the car bays, never in a car's lane. They replace the wall-picked lines;
    every grid crossing then gets a column (a frame over open parking), except inside the stair / lift well.
    """
    if footprint and len(footprint) == 4 and footprint[2] > 0 and footprint[3] > 0:
        fx, fy, fw, fd = footprint
    else:
        fx, fy, fw, fd = building_footprint(rooms)
    # only rooms inside the house envelope (an external terrace stair gets no frame columns)
    b = [r for r in _building(rooms)
         if fx - tol <= r.x and r.right <= fx + fw + tol and fy - tol <= r.y and r.top <= fy + fd + tol]
    if not b or fw <= 0 or fd <= 0:
        return StructuralGrid(footprint=(fx, fy, fw, fd))
    x_lo, x_hi, y_lo, y_hi = fx, fx + fw, fy, fy + fd

    def v_len(x):   # wall length on a vertical line
        return sum(r.h for r in b if abs(r.x - x) < tol or abs(r.right - x) < tol) / 2

    def h_len(y):
        return sum(r.w for r in b if abs(r.y - y) < tol or abs(r.top - y) < tol) / 2

    forced = bool(x_lines)
    xs = sorted(round(v, 3) for v in x_lines) if forced else \
        _pick_lines(_cluster([r.x for r in b] + [r.right for r in b]), x_lo, x_hi, v_len)
    ys = _pick_lines(_cluster([r.y for r in b] + [r.top for r in b]), y_lo, y_hi, h_len)
    wells = [r for r in b if getattr(r, "category", "") in ("STAIRS", "LIFT")]

    def on_vertical_wall(x, y):
        return any((abs(r.x - x) < tol or abs(r.right - x) < tol) and r.y - tol <= y <= r.top + tol for r in b)

    def on_horizontal_wall(x, y):
        return any((abs(r.y - y) < tol or abs(r.top - y) < tol) and r.x - tol <= x <= r.right + tol for r in b)

    def in_well(x, y):
        return any(r.x + 0.2 < x < r.right - 0.2 and r.y + 0.2 < y < r.top - 0.2 for r in wells)

    cols = []
    for x in xs:
        for y in ys:
            v, h = on_vertical_wall(x, y), on_horizontal_wall(x, y)
            if forced:
                if in_well(x, y):
                    continue
            elif not (v or h):
                continue
            # long side along the wall that carries more length through the node; over parking (forced lines)
            # the thin side faces the cars unless only a cross wall runs through
            along_y = (v or not h) if forced else (v and (not h or v_len(x) >= h_len(y)))
            hw, hh = (COL_SHORT / 2, COL_LONG / 2) if along_y else (COL_LONG / 2, COL_SHORT / 2)
            cols.append((round(x - hw, 3), round(y - hh, 3), round(x + hw, 3), round(y + hh, 3)))
    return StructuralGrid(xs=xs, ys=ys, columns=cols, footprint=(fx, fy, fw, fd))
