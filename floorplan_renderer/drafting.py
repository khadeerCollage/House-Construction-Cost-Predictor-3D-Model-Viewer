"""
Architectural Drafting Layer
============================
Turns a GeneratedFloorPlan into the content a professional working drawing
carries, computed once in plan coordinates (metres, +y up) so every renderer
can draw it with its own transform:

  * wall poché      — walls merged into clean polygons (no seams at joints),
                      cut at doors, arches and windows
  * structural grid — RCC column grid A, B, C… / 1, 2, 3… (floorplan_generator.structure)
  * furniture       — clearance-checked, Vastu-aware layouts per room:
                      beds head to S/W (children E), heavy wardrobes on S/W walls,
                      sofa on S/W walls facing N/E, cook faces E, pooja on the E wall,
                      WC seat facing N/S, desk facing E/N; nothing inside a door swing,
                      tall units never across a window, walkways kept in front
  * wet-area tiles, level marks, door/window tags + schedule, rain-water pipes,
    Brahmasthan marker and general notes

Primitives are plain tuples:
  ("poly", [(x, y), ...], closed, style)    ("line", (x1, y1), (x2, y2), style)
  ("circle", (x, y), r, style)              ("text", (x, y), text, size_m, style)
Styles: furn, furn_d (detail), fix (sanitary), fix_d, dash, tile, text.

Drafting.items records every furniture / fixture symbol semantically (kind, wall frame,
size, room) so the 3D walk-through builds the same furniture in the same places.
"""

import math
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from floorplan_generator.structure import SITE_CATEGORIES, StructuralGrid, building_footprint, structural_grid
from floorplan_generator.vastu_engine import plan_sides_for_facing

try:  # shapely gives seamless wall poché; renderers fall back to per-segment walls without it
    from shapely.geometry import box as _box
    from shapely.ops import unary_union
    HAS_SHAPELY = True
except Exception:  # pragma: no cover
    HAS_SHAPELY = False

Rect = Tuple[float, float, float, float]  # x0, y0, x1, y1
EPS = 1e-6
INSET = 0.12          # room centre-line → finished wall face (half wall + plaster)
SIDE_TO_PLAN = {"N": "top", "S": "bottom", "E": "right", "W": "left"}
OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}
OUTWARD = {"N": (0, 1), "S": (0, -1), "E": (1, 0), "W": (-1, 0)}


@dataclass
class Drafting:
    walls: List = field(default_factory=list)       # [(exterior_pts, [hole_pts, ...]), ...]
    compound: List = field(default_factory=list)
    windows: List[Dict] = field(default_factory=list)
    grid: Optional[StructuralGrid] = None
    furniture: List = field(default_factory=list)
    tiles: List = field(default_factory=list)
    tags: List = field(default_factory=list)        # (x, y, text, kind)
    schedule: List[Dict] = field(default_factory=list)
    levels: List = field(default_factory=list)      # (x, y, text)
    labels: Dict[str, Tuple[float, float, bool]] = field(default_factory=dict)
    brahmasthan: Optional[Rect] = None
    rwp: List = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    # furniture as objects: {kind, room, o, u, v (wall frame, plan metres) | x, y, W, D, …}
    items: List[Dict] = field(default_factory=list)
    gates: List = field(default_factory=list)       # gate leaves, swings, sliding tracks (engineering['gates'])


def _overlap(a: Rect, b: Rect) -> bool:
    return a[0] < b[2] - EPS and b[0] < a[2] - EPS and a[1] < b[3] - EPS and b[1] < a[3] - EPS


# ---- symbol recorder: build_drafting collects what each sym_* draws (thread-local, so
# concurrent sessions never mix); outside build_drafting it records nothing
_REC = threading.local()


def _record(kind: str, T=None, **params) -> None:
    items = getattr(_REC, "items", None)
    if items is None:
        return
    if T is not None:    # local frame → origin + unit axes (u along the wall, v away from it)
        o, pu, pv = T(0.0, 0.0), T(1.0, 0.0), T(0.0, 1.0)
        params.update(o=(round(o[0], 4), round(o[1], 4)),
                      u=(round(pu[0] - o[0], 6), round(pu[1] - o[1], 6)),
                      v=(round(pv[0] - o[0], 6), round(pv[1] - o[1], 6)))
    params["kind"] = kind
    items.append(params)


def _rec_mark() -> int:
    items = getattr(_REC, "items", None)
    return len(items) if items is not None else 0


def _rec_reset(mark: int) -> None:
    items = getattr(_REC, "items", None)
    if items is not None:
        del items[mark:]


# =============================================================================
# Local-frame drawing helpers (u along the wall, v away from it)
# =============================================================================
def _shift(T, du=0.0, dv=0.0):
    return lambda u, v: T(u + du, v + dv)


def _rect(T, u0, v0, u1, v1, style):
    return ("poly", [T(u0, v0), T(u1, v0), T(u1, v1), T(u0, v1)], True, style)


def _line(T, a, b, style):
    return ("line", T(*a), T(*b), style)


def _circle(T, u, v, r, style):
    return ("circle", T(u, v), r, style)


def _ellipse(T, cu, cv, ru, rv, style, n=24):
    return ("poly", [T(cu + ru * math.cos(2 * math.pi * i / n), cv + rv * math.sin(2 * math.pi * i / n))
                     for i in range(n)], True, style)


def _rrect(T, u0, v0, u1, v1, rad, style, n=4):
    rad = max(0.0, min(rad, (u1 - u0) / 2, (v1 - v0) / 2))
    pts = []
    for cu, cv, a0 in ((u1 - rad, v0 + rad, -90), (u1 - rad, v1 - rad, 0),
                       (u0 + rad, v1 - rad, 90), (u0 + rad, v0 + rad, 180)):
        for k in range(n + 1):
            a = math.radians(a0 + 90 * k / n)
            pts.append(T(cu + rad * math.cos(a), cv + rad * math.sin(a)))
    return ("poly", pts, True, style)


# ---- furniture & fixture symbols --------------------------------------------
def sym_bed(T, W, D, double=True):
    _record("bed", T, W=W, D=D, double=double)
    out = [_rect(T, 0, 0, W, D, "furn"), _rect(T, 0, 0, W, 0.07, "furn_d")]
    if double:
        out += [_rrect(T, 0.10, 0.13, W / 2 - 0.05, 0.43, 0.06, "furn_d"),
                _rrect(T, W / 2 + 0.05, 0.13, W - 0.10, 0.43, 0.06, "furn_d")]
    else:
        out.append(_rrect(T, 0.12, 0.13, W - 0.12, 0.41, 0.06, "furn_d"))
    vf = 0.58
    out += [_line(T, (0, vf), (W, vf), "furn_d"), _line(T, (W * 0.55, vf), (W, min(D, vf + 0.42)), "furn_d")]
    return out


def sym_side_table(T, s=0.45):
    _record("side_table", T, W=s, D=s)
    return [_rect(T, 0, 0, s, s, "furn"), _circle(T, s / 2, s / 2, s * 0.22, "furn_d")]


def sym_wardrobe(T, W, D=0.6):
    _record("wardrobe", T, W=W, D=D)
    out = [_rect(T, 0, 0, W, D, "furn"), _line(T, (0.04, D / 2), (W - 0.04, D / 2), "dash")]
    u = 0.10
    while u < W - 0.08:
        out.append(_line(T, (u - 0.05, D / 2 - 0.14), (u + 0.05, D / 2 + 0.14), "furn_d"))
        u += 0.14
    return out


def sym_sofa(T, W, D=0.85, seats=3):
    _record("sofa", T, W=W, D=D, seats=seats)
    arm = 0.18
    out = [_rect(T, 0, 0, W, D, "furn"), _rect(T, 0, 0, W, 0.2, "furn_d"),
           _rect(T, 0, 0.2, arm, D, "furn_d"), _rect(T, W - arm, 0.2, W, D, "furn_d")]
    for i in range(1, seats):
        u = arm + i * (W - 2 * arm) / seats
        out.append(_line(T, (u, 0.2), (u, D - 0.04), "furn_d"))
    return out


def sym_coffee_table(T, W, D):
    _record("coffee_table", T, W=W, D=D)
    return [_rect(T, 0, 0, W, D, "furn"), _rect(T, 0.06, 0.06, W - 0.06, D - 0.06, "furn_d")]


def sym_tv_unit(T, W, D=0.4):
    _record("tv_unit", T, W=W, D=D)
    return [_rect(T, 0, 0, W, D, "furn"), _rect(T, W * 0.18, 0.05, W * 0.82, 0.11, "furn_d"),
            ("text", T(W / 2, D * 0.62), "TV", 0.14, "text")]


def sym_fridge(T, s=0.7):
    _record("fridge", T, W=s, D=s)
    return [_rect(T, 0, 0, s, s, "furn"), _line(T, (0, s - 0.06), (s, s - 0.06), "furn_d"),
            ("text", T(s / 2, s / 2), "REF", 0.13, "text")]


def sym_sink(T, a, double=True):
    _record("sink", T, a=a, double=double)
    out = []
    if double:
        out += [_rrect(T, a, 0.09, a + 0.37, 0.5, 0.05, "fix"), _rrect(T, a + 0.43, 0.09, a + 0.8, 0.5, 0.05, "fix"),
                _circle(T, a + 0.185, 0.3, 0.025, "fix_d"), _circle(T, a + 0.615, 0.3, 0.025, "fix_d"),
                _circle(T, a + 0.4, 0.05, 0.022, "fix_d")]
    else:
        out += [_rrect(T, a, 0.09, a + 0.45, 0.5, 0.05, "fix"), _circle(T, a + 0.225, 0.3, 0.025, "fix_d"),
                _circle(T, a + 0.225, 0.05, 0.022, "fix_d")]
    return out


def sym_hob(T, a):
    _record("hob", T, a=a)
    out = [_rect(T, a, 0.06, a + 0.6, 0.54, "fix_d")]
    for cu, cv in ((a + 0.15, 0.18), (a + 0.45, 0.18), (a + 0.15, 0.42), (a + 0.45, 0.42)):
        out += [_circle(T, cu, cv, 0.08, "fix_d"), _circle(T, cu, cv, 0.035, "fix_d")]
    return out


def sym_wc(T, W=0.42, D=0.72):
    _record("wc", T, W=W, D=D)
    return [_rrect(T, 0.02, 0, W - 0.02, 0.2, 0.03, "fix"), _ellipse(T, W / 2, 0.46, 0.19, 0.26, "fix"),
            _ellipse(T, W / 2, 0.47, 0.12, 0.17, "fix_d")]


def sym_basin(T, W=0.55, D=0.42):
    _record("basin", T, W=W, D=D)
    return [_rrect(T, 0, 0, W, D, 0.04, "fix"), _ellipse(T, W / 2, 0.23, 0.19, 0.13, "fix_d"),
            _circle(T, W / 2, 0.055, 0.02, "fix_d")]


def sym_shower(T, W=0.9, D=0.9):
    _record("shower", T, W=W, D=D)
    return [_rect(T, 0, 0, W, D, "fix"), _line(T, (0, 0), (W, D), "fix_d"), _line(T, (W, 0), (0, D), "fix_d"),
            _circle(T, W / 2, D / 2, 0.045, "fix_d"), _circle(T, W / 2, 0.08, 0.06, "fix_d")]


def sym_floor_trap(x, y):
    _record("floor_trap", x=x, y=y)
    return [("circle", (x, y), 0.07, "fix_d"), ("line", (x - 0.05, y), (x + 0.05, y), "fix_d"),
            ("line", (x, y - 0.05), (x, y + 0.05), "fix_d")]


def sym_washer(T, s=0.6):
    _record("washer", T, W=s, D=s)
    return [_rect(T, 0, 0, s, s, "fix"), _rect(T, 0, 0, s, 0.09, "fix_d"),
            _circle(T, s / 2, s / 2 + 0.03, 0.2, "fix_d"), _circle(T, s / 2, s / 2 + 0.03, 0.12, "fix_d")]


def sym_utility_sink(T, W=0.6, D=0.5):
    _record("utility_sink", T, W=W, D=D)
    return [_rect(T, 0, 0, W, D, "fix"), _rrect(T, 0.06, 0.08, W - 0.06, D - 0.05, 0.05, "fix_d"),
            _circle(T, W / 2, D / 2 + 0.02, 0.025, "fix_d")]


def sym_desk(T, W, D=0.6):
    _record("desk", T, W=W, D=D)
    return [_rect(T, 0, 0, W, D, "furn"), _rect(T, 0.08, 0.05, W * 0.45, 0.3, "furn_d"),
            _rrect(T, W / 2 - 0.22, D + 0.08, W / 2 + 0.22, D + 0.5, 0.06, "furn"),
            _line(T, (W / 2 - 0.2, D + 0.44), (W / 2 + 0.2, D + 0.44), "furn_d")]


def sym_bookshelf(T, W, D=0.35):
    _record("bookshelf", T, W=W, D=D)
    out = [_rect(T, 0, 0, W, D, "furn")]
    n = max(2, int(round(W / 0.3)))
    for i in range(1, n):
        out.append(_line(T, (i * W / n, 0), (i * W / n, D), "furn_d"))
    return out


def sym_shelves(T, W, D=0.45):
    _record("shelves", T, W=W, D=D)
    out = [_rect(T, 0, 0, W, D, "furn")]
    n = max(1, int(math.ceil(W / 0.9)))
    for i in range(n):
        a, b = i * W / n, (i + 1) * W / n
        out += [_line(T, (a, 0), (b, D), "furn_d"), _line(T, (b, 0), (a, D), "furn_d")]
    return out


def sym_pooja(T, W, D=0.45):
    _record("pooja", T, W=W, D=D)
    c = W / 2
    return [_rect(T, 0, 0, W, D, "furn"), _rect(T, 0.06, 0.04, W - 0.06, D * 0.55, "furn_d"),
            ("poly", [T(c, 0.08), T(c + 0.07, D * 0.3), T(c, D * 0.5), T(c - 0.07, D * 0.3)], True, "furn_d"),
            _circle(T, c, D * 0.78, 0.04, "furn_d")]


def sym_shoe_rack(T, W, D=0.35):
    _record("shoe_rack", T, W=W, D=D)
    return [_rect(T, 0, 0, W, D, "furn"), _line(T, (0.03, D / 2), (W - 0.03, D / 2), "furn_d")]


def sym_planter(x, y, r=0.2):
    _record("planter", x=x, y=y, r=r)
    out = [("circle", (x, y), r, "furn"), ("circle", (x, y), r * 0.35, "furn_d")]
    for k in range(6):
        a = math.radians(30 + 60 * k)
        out.append(("line", (x + r * 0.35 * math.cos(a), y + r * 0.35 * math.sin(a)),
                    (x + r * 0.9 * math.cos(a), y + r * 0.9 * math.sin(a)), "furn_d"))
    return out


def sym_dining(x0, y0, tl, tw, seats, horizontal, record=True):
    """Table centred in the footprint (x0, y0) with chairs on the long sides (plan frame)."""
    ch, gap = 0.42, 0.1
    fw, fd = (tl, tw + 2 * (ch + gap)) if horizontal else (tw + 2 * (ch + gap), tl)
    if horizontal:
        T = lambda u, v: (x0 + u, y0 + ch + gap + v)  # noqa: E731
    else:
        T = lambda u, v: (x0 + ch + gap + v, y0 + u)  # noqa: E731
    if record:
        _record("dining", T, W=tl, D=tw, seats=seats)
    out = [_rect(T, 0, 0, tl, tw, "furn"), _rect(T, 0.08, 0.08, tl - 0.08, tw - 0.08, "furn_d")]
    n = max(1, seats // 2)
    for i in range(n):
        uc = (i + 0.5) * tl / n
        for v0, v1, back in ((-gap - ch, -gap, -gap - ch), (tw + gap, tw + gap + ch, tw + gap + ch)):
            out.append(_rrect(T, uc - ch / 2, v0, uc + ch / 2, v1, 0.06, "furn"))
            bv = back + (0.06 if back < 0 else -0.06)
            out.append(_line(T, (uc - ch / 2 + 0.04, bv), (uc + ch / 2 - 0.04, bv), "furn_d"))
    return out, (fw, fd)


# =============================================================================
# Room space: clearance-checked placement against walls
# =============================================================================
class _Space:
    def __init__(self, room, inset=INSET):
        self.room = room
        self.x0, self.y0 = room.x + inset, room.y + inset
        self.x1, self.y1 = room.right - inset, room.top - inset
        self.hard: List[Rect] = []     # furniture
        self.clear: List[Rect] = []    # door swings / walkway at openings
        self.windows: Dict[str, List[Tuple[float, float]]] = {s: [] for s in "NSEW"}
        self.door_mids: List[Tuple[float, float]] = []

    @property
    def usable(self):
        return self.x1 - self.x0 > 0.5 and self.y1 - self.y0 > 0.5

    def length(self, side):
        return (self.x1 - self.x0) if side in "NS" else (self.y1 - self.y0)

    def depth(self, side):
        return (self.y1 - self.y0) if side in "NS" else (self.x1 - self.x0)

    def rect(self, side, u0, W, D) -> Rect:
        if side == "S":
            return (self.x0 + u0, self.y0, self.x0 + u0 + W, self.y0 + D)
        if side == "N":
            return (self.x0 + u0, self.y1 - D, self.x0 + u0 + W, self.y1)
        if side == "W":
            return (self.x0, self.y0 + u0, self.x0 + D, self.y0 + u0 + W)
        return (self.x1 - D, self.y0 + u0, self.x1, self.y0 + u0 + W)

    def to_plan(self, side, u0=0.0):
        if side == "S":
            return lambda u, v: (self.x0 + u0 + u, self.y0 + v)
        if side == "N":
            return lambda u, v: (self.x0 + u0 + u, self.y1 - v)
        if side == "W":
            return lambda u, v: (self.x0 + v, self.y0 + u0 + u)
        return lambda u, v: (self.x1 - v, self.y0 + u0 + u)

    def inside(self, r: Rect) -> bool:
        return (r[0] >= self.x0 - EPS and r[1] >= self.y0 - EPS
                and r[2] <= self.x1 + EPS and r[3] <= self.y1 + EPS)

    def free(self, r: Rect, use_clear=True) -> bool:
        if not self.inside(r) or any(_overlap(r, o) for o in self.hard):
            return False
        return not (use_clear and any(_overlap(r, o) for o in self.clear))

    def window_hit(self, side, u0, W) -> bool:
        a = (self.x0 if side in "NS" else self.y0) + u0
        return any(a < hi + 0.05 and lo - 0.05 < a + W for lo, hi in self.windows[side])

    def window_center(self, side) -> Optional[float]:
        if not self.windows[side]:
            return None
        lo, hi = max(self.windows[side], key=lambda iv: iv[1] - iv[0])
        return (lo + hi) / 2 - (self.x0 if side in "NS" else self.y0)

    def door_dist(self, r: Rect) -> float:
        if not self.door_mids:
            return 3.0
        cx, cy = (r[0] + r[2]) / 2, (r[1] + r[3]) / 2
        return min(math.hypot(cx - x, cy - y) for x, y in self.door_mids)

    def place(self, W, D, side_score=None, align="center", tall=False, front=0.0,
              sides="NSEW", extra: Optional[Callable] = None, step=0.05):
        best = None
        for side in sides:
            L = self.length(side)
            if W > L + EPS or D + front > self.depth(side) + EPS:
                continue
            n = int(math.floor((L - W) / step + EPS))
            for u0 in sorted({round(i * step, 4) for i in range(n + 1)} | {round(L - W, 4)}):
                r = self.rect(side, u0, W, D)
                if not self.free(r) or (tall and self.window_hit(side, u0, W)):
                    continue
                if front > 0:
                    fr = self.rect(side, u0, W, D + front)
                    if not self.inside(fr) or any(_overlap(fr, o) for o in self.hard):
                        continue
                s = 10.0 * (side_score or {}).get(side, 0)
                if align == "center":
                    s -= abs(u0 + W / 2 - L / 2)
                elif align == "corner":
                    s -= min(u0, L - W - u0)
                elif align == "far_door":
                    s += 0.8 * self.door_dist(r) - 0.2 * min(u0, L - W - u0)
                elif align == "near_door":
                    s -= 0.8 * self.door_dist(r)
                if extra:
                    s += extra(side, u0, r)
                if best is None or s > best[0] + 1e-9:
                    best = (s, side, u0, r)
        if best is None:
            return None
        _, side, u0, r = best
        self.hard.append(r)
        return side, u0, r

    def place_free(self, W, D, target=None, step=0.1):
        tx, ty = target or ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)
        best = None
        nx = int((self.x1 - self.x0 - W) / step + EPS)
        ny = int((self.y1 - self.y0 - D) / step + EPS)
        for i in range(nx + 1):
            for j in range(ny + 1):
                r = (self.x0 + i * step, self.y0 + j * step, self.x0 + i * step + W, self.y0 + j * step + D)
                if not self.free(r):
                    continue
                s = -math.hypot((r[0] + r[2]) / 2 - tx, (r[1] + r[3]) / 2 - ty)
                if best is None or s > best[0]:
                    best = (s, r)
        if best is None:
            return None
        self.hard.append(best[1])
        return best[1]

    def runs(self, side, D, min_len):
        """Free stretches along a wall for a band of depth D (counters, shelves)."""
        L = self.length(side)
        band = self.rect(side, 0, L, D)
        blocked = []
        for o in self.hard + self.clear:
            if _overlap(band, o):
                a, b = (o[0] - self.x0, o[2] - self.x0) if side in "NS" else (o[1] - self.y0, o[3] - self.y0)
                blocked.append((max(0.0, a), min(L, b)))
        blocked.sort()
        out, cur = [], 0.0
        for a, b in blocked:
            if a - cur >= min_len:
                out.append((cur, a))
            cur = max(cur, b)
        if L - cur >= min_len:
            out.append((cur, L))
        return out


# =============================================================================
# Room recipes
# =============================================================================
def _try_place_block(sp, widths_tables, D, prefs, front):
    for W, tables in widths_tables:
        res = sp.place(W, D, side_score=prefs, align="center", front=front)
        if res:
            return res, W, tables
    return None, 0, 0


def _furnish_bedroom(sp, room, P, has_dress):
    iw = min(sp.length("S"), sp.length("W"))
    child = room.role == "CHILD_BEDROOM"
    prefs = P({"E": 3, "S": 2, "W": 1, "N": -6}) if child else P({"S": 3, "W": 2, "E": 1, "N": -6})
    out = []
    bw = (1.35 if iw >= 2.9 else 0.95) if child else (1.8 if iw >= 3.6 else 1.5)
    options = [(bw + 0.9, 2), (bw + 0.45, 1), (bw, 0)]
    if bw > 1.0:
        options += [(0.95 + 0.45, 1), (0.95, 0)]
    res, W, tables = _try_place_block(sp, options, 2.0, prefs, 0.6)
    if res:
        side, u0, _ = res
        T = sp.to_plan(side, u0)
        bed_w = W - 0.45 * tables
        start = 0.45 if tables >= 1 else 0.0
        out += sym_bed(_shift(T, start), bed_w, 2.0, double=bed_w > 1.1)
        if tables >= 1:
            out += sym_side_table(T)
        if tables == 2:
            out += sym_side_table(_shift(T, start + bed_w))
    if not has_dress:
        for wl in (1.8, 1.5, 1.2, 0.9):
            r = sp.place(wl, 0.6, side_score=P({"S": 2, "W": 2}), align="corner", tall=True, front=0.6)
            if r:
                out += sym_wardrobe(sp.to_plan(r[0], r[1]), wl)
                break
    if child:
        for dw in (1.0, 0.8):
            r = sp.place(dw, 0.5, side_score=P({"E": 3, "N": 2}), align="corner", front=0.6)
            if r:
                out += sym_desk(sp.to_plan(r[0], r[1]), dw, 0.5)
                sp.hard.append(sp.rect(r[0], r[1] + dw / 2 - 0.25, 0.5, 1.05))
                break
    return out


def _dining_set(sp, target=None, seats_options=(6, 4, 2)):
    long_x = sp.length("S") >= sp.length("W")
    for seats in seats_options:
        tl, tw = {6: (1.6, 0.9), 4: (1.2, 0.8), 2: (0.8, 0.75)}[seats]
        for horizontal in ((True, False) if long_x else (False, True)):
            _, (fw, fd) = sym_dining(0, 0, tl, tw, seats, horizontal, record=False)
            r = sp.place_free(fw + 0.2, fd + 0.2, target)
            if r:
                prims, _ = sym_dining(r[0] + 0.1, r[1] + 0.1, tl, tw, seats, horizontal)
                return prims
    return []


def _furnish_living(sp, room, P, combined):
    out = []
    sofa_side = None
    for sw, seats in ((2.1, 3), (1.8, 3), (1.5, 2)):
        res = sp.place(sw, 0.85, side_score=P({"S": 3, "W": 3}), align="center", front=1.25)
        if res:
            side, u0, r = res
            sofa_side = side
            T = sp.to_plan(side, u0)
            out += sym_sofa(T, sw, 0.85, seats)
            ct_w = min(1.0, sw * 0.55)
            out += sym_coffee_table(_shift(T, (sw - ct_w) / 2, 0.85 + 0.35), ct_w, 0.5)
            sp.hard.append(sp.rect(side, u0 + (sw - ct_w) / 2, ct_w, 0.85 + 0.85))
            sofa_c = u0 + sw / 2
            tv = sp.place(min(1.6, sw), 0.4, align="center", tall=True, sides=OPPOSITE[side],
                          extra=lambda s, u, rr, c=sofa_c, w=min(1.6, sw): -abs(u + w / 2 - c))
            if tv:
                out += sym_tv_unit(sp.to_plan(tv[0], tv[1]), min(1.6, sw))
            perp = "EW" if side in "NS" else "NS"
            ac = sp.place(0.75, 0.75, align="near_door", front=0.3, sides=perp,
                          extra=lambda s, u, rr: -2.0 * math.hypot((rr[0] + rr[2]) / 2 - (r[0] + r[2]) / 2,
                                                                   (rr[1] + rr[3]) / 2 - (r[1] + r[3]) / 2))
            if ac:
                out += sym_sofa(sp.to_plan(ac[0], ac[1]), 0.75, 0.75, 1)
            break
    if combined:
        far = None
        if sofa_side:
            far = {"N": ((sp.x0 + sp.x1) / 2, sp.y0), "S": ((sp.x0 + sp.x1) / 2, sp.y1),
                   "E": (sp.x0, (sp.y0 + sp.y1) / 2), "W": (sp.x1, (sp.y0 + sp.y1) / 2)}[sofa_side]
        out += _dining_set(sp, far, (6, 4, 2))
    pl = sp.place(0.4, 0.4, align="corner")
    if pl:
        r = pl[2]
        out += sym_planter((r[0] + r[2]) / 2, (r[1] + r[3]) / 2, 0.18)
    return out


def _furnish_dining(sp, room, P):
    out = _dining_set(sp, None, (6, 4, 2))
    r = sp.place(1.2, 0.45, side_score=P({"S": 2, "W": 2}), align="center", tall=True, front=0.6)
    if r:
        out += sym_shelves(sp.to_plan(r[0], r[1]), 1.2, 0.45)
    return out


def _furnish_kitchen(sp, room, P):
    out = []
    east = P({"E": 2, "S": 1})
    best = None
    for side in "NSEW":
        runs = sp.runs(side, 0.6, 1.2)
        if not runs:
            continue
        a, b = max(runs, key=lambda iv: iv[1] - iv[0])
        score = (b - a) + (2.0 if sp.windows[side] else 0) + 0.8 * east[side]
        if best is None or score > best[0]:
            best = (score, side, a, b)
    if best is None:
        return out
    _, side, a, b = best
    L = sp.length(side)
    sp.hard.append(sp.rect(side, a, b - a, 0.6))
    T = sp.to_plan(side, 0)
    out.append(_rect(T, a, 0, b, 0.6, "furn"))
    _record("counter", T, a=a, b=b, D=0.6)
    counters = [(side, a, b, T)]
    # L-shaped return along the adjacent wall at a corner the run reaches
    s2 = None
    if a <= 0.05:
        s2 = "W" if side in "NS" else "S"      # corner at the low end of the main run
    elif b >= L - 0.05:
        s2 = "E" if side in "NS" else "N"      # corner at the high end
    if s2:
        L2 = sp.length(s2)
        corner_low = side in "SW"              # where that corner sits along the return wall
        for a2, b2 in sp.runs(s2, 0.6, 0.9):
            if corner_low and a2 <= 0.65:
                a2, b2 = a2, min(b2, a2 + 1.8)
            elif not corner_low and b2 >= L2 - 0.65:
                a2, b2 = max(a2, b2 - 1.8), b2
            else:
                continue
            if b2 - a2 >= 0.9:
                sp.hard.append(sp.rect(s2, a2, b2 - a2, 0.6))
                T2 = sp.to_plan(s2, 0)
                out.append(_rect(T2, a2, 0, b2, 0.6, "furn"))
                _record("counter", T2, a=a2, b=b2, D=0.6)
                counters.append((s2, a2, b2, T2))
            break
    # sink under the window (else a third along the main run); hob ≥ 0.75 m away, prefer the E wall
    wc = sp.window_center(side)
    dbl = (b - a) >= 2.2
    sw = 0.8 if dbl else 0.45
    s0 = (wc - sw / 2) if wc is not None else a + 0.3 * (b - a) - sw / 2
    s0 = max(a + 0.05, min(b - sw - 0.05, s0))
    out += sym_sink(T, s0, dbl)
    sink_mid = s0 + sw / 2
    hob = None
    for cs, ca, cb, CT in counters:
        u = ca + 0.1
        while u + 0.6 <= cb - 0.05 + EPS:
            if cs == side and abs(u + 0.3 - sink_mid) < 0.75 + sw / 2:
                u += 0.05
                continue
            sc = 3 * east[cs] - 0.1 * abs(u + 0.3 - (ca + cb) / 2)
            if hob is None or sc > hob[0]:
                hob = (sc, CT, u)
            u += 0.05
    if hob:
        out += sym_hob(hob[1], hob[2])
    fr = sp.place(0.7, 0.7, align="near_door", tall=True, front=0.6)
    if fr:
        out += sym_fridge(sp.to_plan(fr[0], fr[1]))
    return out


def _furnish_utility(sp, room, P):
    out = []
    r = sp.place(0.6, 0.6, align="corner", front=0.45)
    if r:
        out += sym_washer(sp.to_plan(r[0], r[1]))
    r = sp.place(0.6, 0.5, align="corner", front=0.45)
    if r:
        out += sym_utility_sink(sp.to_plan(r[0], r[1]))
    return out


def _furnish_bath(sp, room, P):
    wc_pref = P({"N": 1, "S": 1})
    base = len(sp.hard)
    mark = _rec_mark()
    for order in (("shower", "wc", "basin"), ("wc", "basin", "shower")):
        del sp.hard[base:]
        _rec_reset(mark)
        out, ok, has_shower = [], True, False
        for item in order:
            if item == "shower":
                r = sp.place(0.9, 0.9, align="far_door") or sp.place(0.8, 0.8, align="far_door")
                if r:
                    s = r[2][2] - r[2][0] if r[0] in "NS" else r[2][3] - r[2][1]
                    out += sym_shower(sp.to_plan(r[0], r[1]), s, s)
                    has_shower = True
            elif item == "wc":
                r = (sp.place(0.42, 0.72, side_score=wc_pref, align="far_door", front=0.45)
                     or sp.place(0.42, 0.72, side_score=wc_pref, align="far_door", front=0.15))
                if not r:
                    ok = False
                    break
                out += sym_wc(sp.to_plan(r[0], r[1]))
            else:
                r = (sp.place(0.55, 0.42, align="near_door", front=0.4)
                     or sp.place(0.45, 0.38, align="near_door", front=0.1))
                if r:
                    out += sym_basin(sp.to_plan(r[0], r[1]), r[2][2] - r[2][0] if r[0] in "NS" else r[2][3] - r[2][1],
                                     r[2][3] - r[2][1] if r[0] in "NS" else r[2][2] - r[2][0])
        if ok:
            if not has_shower:
                spot = sp.place_free(0.2, 0.2)
                if spot:
                    out += sym_floor_trap((spot[0] + spot[2]) / 2, (spot[1] + spot[3]) / 2)
            return out
    _rec_reset(mark)
    return []


def _furnish_pooja(sp, room, P):
    w = max(0.6, min(0.9, max(sp.length("S"), sp.length("W")) - 0.1))
    prefs = P({"E": 3, "N": 2, "W": 1, "S": -4})
    r = (sp.place(w, 0.45, side_score=prefs, align="center", tall=True, front=0.5)
         or sp.place(w, 0.45, side_score=prefs, align="center", front=0.3)
         or sp.place(0.6, 0.4, side_score=prefs, align="center"))
    if not r:
        return []
    width = r[2][2] - r[2][0] if r[0] in "NS" else r[2][3] - r[2][1]
    return sym_pooja(sp.to_plan(r[0], r[1]), width, 0.45 if width > 0.6 else 0.4)


def _furnish_study(sp, room, P):
    out = []
    for dw in (1.2, 1.0, 0.8):
        r = sp.place(dw, 0.6, side_score=P({"E": 3, "N": 2}), align="center", front=0.7)
        if r:
            out += sym_desk(sp.to_plan(r[0], r[1]), dw, 0.6)
            sp.hard.append(sp.rect(r[0], r[1] + dw / 2 - 0.25, 0.5, 1.15))
            break
    r = sp.place(0.9, 0.35, align="corner", tall=True, front=0.5)
    if r:
        out += sym_bookshelf(sp.to_plan(r[0], r[1]), 0.9, 0.35)
    return out


def _furnish_store(sp, room, P):
    out = []
    sides = sorted("NSEW", key=lambda s: -sp.length(s))
    budget = 2 if min(sp.length("S"), sp.length("W")) < 1.4 else 3
    for side in sides[:budget]:
        for a, b in sp.runs(side, 0.45, 0.6):
            if sp.depth(side) - 0.45 < 0.6:
                continue
            sp.hard.append(sp.rect(side, a, b - a, 0.45))
            out += sym_shelves(sp.to_plan(side, a), b - a, 0.45)
    return out


def _furnish_servant(sp, room, P):
    out = []
    r = sp.place(0.9, 1.9, side_score=P({"S": 2, "W": 1, "N": -6}), align="corner", front=0.5)
    if r:
        out += sym_bed(sp.to_plan(r[0], r[1]), 0.9, 1.9, double=False)
    r = sp.place(0.6, 0.45, align="corner", tall=True, front=0.5)
    if r:
        out += sym_wardrobe(sp.to_plan(r[0], r[1]), 0.6, 0.45)
    return out


def _furnish_corridor(sp, room, P):
    out = []
    name = room.name.lower()
    if "dress" in name or "wardrobe" in name:
        across = min(sp.length("S"), sp.length("W"))
        long_sides = ("E", "W") if sp.length("W") >= sp.length("S") else ("N", "S")
        for side in long_sides[: (2 if across >= 1.8 else 1)]:
            for a, b in sp.runs(side, 0.6, 0.6):
                if sp.depth(side) - 0.6 < 0.6:
                    continue
                sp.hard.append(sp.rect(side, a, b - a, 0.6))
                out += sym_wardrobe(sp.to_plan(side, a), b - a)
    elif "foyer" in name:
        r = sp.place(0.9, 0.35, align="center", front=0.6)
        if r:
            out += sym_shoe_rack(sp.to_plan(r[0], r[1]), 0.9)
    return out


# =============================================================================
# Walls, openings, schedule
# =============================================================================
def _polys(geom):
    if geom is None or geom.is_empty:
        return []
    geoms = getattr(geom, "geoms", [geom])
    out = []
    for g in geoms:
        if g.geom_type != "Polygon" or g.area < 1e-5:
            continue
        out.append((list(g.exterior.coords), [list(i.coords) for i in g.interiors]))
    return out


def _wall_poche(plan, windows):
    bld, cmp_ = [], []
    segs = []
    for w in plan.walls:
        t = w.thickness
        if abs(w.y1 - w.y2) < 1e-6:
            x0, x1 = sorted((w.x1, w.x2))
            g = _box(x0, w.y1 - t / 2, x1, w.y1 + t / 2)
        elif abs(w.x1 - w.x2) < 1e-6:
            y0, y1 = sorted((w.y1, w.y2))
            g = _box(w.x1 - t / 2, y0, w.x1 + t / 2, y1)
        else:
            continue
        if getattr(w, "wall_type", "") == "compound":
            cmp_.append(g)
        else:
            bld.append(g)
            segs.append(w)
    # fill L / T junction corners (centre-line segments stop at the node)
    for w in segs:
        horiz = abs(w.y1 - w.y2) < 1e-6
        for (px, py) in ((w.x1, w.y1), (w.x2, w.y2)):
            for o in segs:
                if o is w or (abs(o.y1 - o.y2) < 1e-6) == horiz:
                    continue
                if horiz:  # o vertical
                    lo, hi = sorted((o.y1, o.y2))
                    hit = abs(o.x1 - px) < 0.02 and lo - 0.02 <= py <= hi + 0.02
                else:
                    lo, hi = sorted((o.x1, o.x2))
                    hit = abs(o.y1 - py) < 0.02 and lo - 0.02 <= px <= hi + 0.02
                if hit:
                    s = max(w.thickness, o.thickness) / 2
                    bld.append(_box(px - s, py - s, px + s, py + s))
                    break
    geom = unary_union(bld)
    cuts = [_box(*wd["box"]) for wd in windows]
    if cuts:
        geom = geom.difference(unary_union(cuts))
    return _polys(geom), _polys(unary_union(cmp_) if cmp_ else None)


def _window_records(plan, rooms_by_name):
    recs = []
    for w in plan.windows:
        side = w.wall_side
        horiz = side in ("N", "S")
        length = w.width if horiz else (w.height or w.width)
        c = w.x if horiz else w.y
        coord = w.y if horiz else w.x
        t = 0.23
        for wall in plan.walls:
            if getattr(wall, "wall_type", "") == "compound":
                continue
            if horiz and abs(wall.y1 - wall.y2) < 1e-6 and abs(wall.y1 - coord) < 0.03:
                t = wall.thickness
                break
            if not horiz and abs(wall.x1 - wall.x2) < 1e-6 and abs(wall.x1 - coord) < 0.03:
                t = wall.thickness
                break
        a, b = c - length / 2, c + length / 2
        bx = (a, coord - t / 2 - 0.005, b, coord + t / 2 + 0.005) if horiz else (coord - t / 2 - 0.005, a, coord + t / 2 + 0.005, b)
        room = rooms_by_name.get(w.room_name)
        recs.append({"side": side, "a": a, "b": b, "coord": coord, "t": t, "box": bx,
                     "out": OUTWARD[side], "room": w.room_name,
                     "category": getattr(room, "category", ""), "length": length})
    return recs


def _schedule(plan, rooms_by_name, windows, footprint):
    fx, fy, fw, fd = footprint
    tags, rows = [], []
    # ---- doors
    groups: Dict[Tuple, List] = {}
    for d in plan.doors:
        room = rooms_by_name.get(d.room_name)
        cat = getattr(room, "category", "")
        horiz = d.wall_side in ("N", "S")
        exterior = (abs(d.y - fy) < 0.03 or abs(d.y - (fy + fd)) < 0.03) if horiz else \
                   (abs(d.x - fx) < 0.03 or abs(d.x - (fx + fw)) < 0.03)
        if d.is_main_entrance:
            key = ("MD", 0, "Teak panelled + MS safety grille")
        elif cat == "BATHROOM":
            key = ("D", round(d.width * 1000), "WPC flush, waterproof (wet areas)")
        elif cat == "POOJA":
            key = ("D", round(d.width * 1000), "Teak panelled with jaali")
        elif exterior:
            key = ("D", round(d.width * 1000), "Steel / FRP service door")
        else:
            key = ("D", round(d.width * 1000), "Flush door, laminated both faces")
        groups.setdefault(key, []).append(d)
    codes = {}
    n = 0
    for key in sorted(groups, key=lambda k: (k[0] != "MD", -k[1], k[2])):
        if key[0] == "MD":
            codes[key] = "MD"
        else:
            n += 1
            codes[key] = f"D{n}"
        w_mm = round(groups[key][0].width * 1000)
        rows.append({"kind": "door", "tag": codes[key], "size": f"{w_mm} × 2100", "sill": "-",
                     "nos": len(groups[key]), "spec": key[2]})
    for key, ds in groups.items():
        for d in ds:
            horiz = d.wall_side in ("N", "S")
            mx, my = (d.x + d.width / 2, d.y) if horiz else (d.x, d.y + d.width / 2)
            nx, ny = {"N": (0, -1), "S": (0, 1), "E": (-1, 0), "W": (1, 0)}[d.wall_side]  # into room_name
            off = 0.34
            tags.append((mx - nx * off, my - ny * off, codes[key], "door"))
    # ---- windows
    wgroups: Dict[Tuple, List] = {}
    for w in windows:
        vent = w["category"] in ("BATHROOM", "STORE") or w["length"] <= 0.6
        if vent:
            key = ("V", int(round(w["length"] * 100)) * 10, 450, 1800, "Alu. louvred ventilator, frosted")
        else:
            h = {"KITCHEN": 1000, "UTILITY": 900, "POOJA": 900, "SERVANT": 1000}.get(w["category"], 1200)
            sill = 1050 if w["category"] == "KITCHEN" else 900
            key = ("W", int(round(w["length"] * 100)) * 10, h, sill, "UPVC sliding, toughened + MS grille")
        wgroups.setdefault(key, []).append(w)
    counters = {"W": 0, "V": 0}
    for key in sorted(wgroups, key=lambda k: (k[0] == "V", -k[1], -k[2])):
        counters[key[0]] += 1
        code = f"{key[0]}{counters[key[0]]}"
        rows.append({"kind": "window", "tag": code, "size": f"{key[1]} × {key[2]}", "sill": str(key[3]),
                     "nos": len(wgroups[key]), "spec": key[4]})
        for w in wgroups[key]:
            c = (w["a"] + w["b"]) / 2
            ox, oy = w["out"]
            off = w["t"] / 2 + 0.38
            x, y = (c, w["coord"]) if w["side"] in ("N", "S") else (w["coord"], c)
            tags.append((x + ox * off, y + oy * off, code, "window"))
    return tags, rows


def _gate_symbols(plan):
    """Gates in the compound wall (floorplan_generator/gates.py): leaves drawn open with their swing, a
    sliding leaf on its track with the wall it runs behind, bi-fold panels; tags G1… and schedule rows."""
    gates = ((getattr(plan, "engineering", None) or {}).get("gates") or [])
    H = plan.plot_height
    prims, tags, rows = [], [], []

    def arc(cx, cy, r, a0, a1, n=12):
        return [(cx + r * math.cos(a0 + (a1 - a0) * k / n), cy + r * math.sin(a0 + (a1 - a0) * k / n)) for k in range(n + 1)]

    for g in gates:
        for lf in g.get("leaves") or []:
            t = lf.get("type")
            if t in ("swing", "wicket"):
                hx, L, s = lf["hinge"], lf["len"], lf.get("dir", 1)
                prims.append(("line", (hx, H), (hx, H - L), "gate"))                       # the leaf, open
                a0, a1 = (0.0, -math.pi / 2) if s > 0 else (math.pi, 3 * math.pi / 2)
                prims.append(("poly", arc(hx, H, L, a0, a1), False, "gate_d"))            # its swing (inward)
            elif t == "slide":
                x0, x1, y = lf["x0"], lf["x1"], H - 0.16
                prims.append(("poly", [(x0, y - 0.03), (x1, y - 0.03), (x1, y + 0.03), (x0, y + 0.03)], True, "gate"))
            elif t == "fold":
                hx, L, s = lf["hinge"], lf["len"], lf.get("dir", 1)
                prims.append(("poly", [(hx, H), (hx + s * 0.25 * L, H - L * 0.97), (hx + s * 0.5 * L, H - 0.05)],
                              False, "gate"))
                prims.append(("poly", arc(hx, H, L, 0.0 if s > 0 else math.pi, -math.pi / 2 if s > 0 else 3 * math.pi / 2),
                              False, "gate_d"))
        rb = g.get("runback")
        if rb:                                                   # the wall the sliding leaf runs behind
            y = H - 0.16
            prims.append(("poly", [(rb[0], y - 0.05), (rb[1], y - 0.05), (rb[1], y + 0.05), (rb[0], y + 0.05)], True, "gate_d"))
            to = 1 if rb[0] >= g["x1"] - 0.05 else -1
            ax0 = (g["x0"] + g["x1"]) / 2
            ax1 = rb[0] + 0.3 if to > 0 else rb[1] - 0.3
            prims.append(("line", (ax0, y - 0.32), (ax1, y - 0.32), "gate_d"))
            prims.append(("poly", [(ax1 - to * 0.18, y - 0.22), (ax1, y - 0.32), (ax1 - to * 0.18, y - 0.42)], False, "gate"))
        cx = (g["x0"] + g["x1"]) / 2
        tags.append((cx, H + 0.55, g.get("tag", "G"), "gate"))
        what = {"walk": "people", "two_wheeler": "people + bikes", "vehicle": "car", "vehicle_wicket": "car + wicket"}.get(
            g.get("kind"), "gate")
        op = {"swing": "swing", "sliding": "sliding", "telescopic": "telescopic", "bifold": "bi-fold"}.get(g.get("op"), "")
        rows.append({"kind": "gate", "tag": g.get("tag", "G"),
                     "size": f"{round(g['width'] * 1000)} × {round(g.get('height', 1.5) * 1000)}", "sill": "-", "nos": 1,
                     "spec": f"MS {op} gate · {what} · {g.get('feet', '')}"})
    return prims, tags, rows


# =============================================================================
# Public entry point
# =============================================================================
def build_drafting(plan, label_size: Optional[Callable] = None) -> Drafting:
    """Compute the full drafting layer for a plan (plan coordinates)."""
    out = Drafting()
    _REC.items = out.items
    try:
        return _build_drafting(plan, label_size, out)
    finally:
        _REC.items = None


def _build_drafting(plan, label_size: Optional[Callable], out: Drafting) -> Drafting:
    rooms = list(plan.rooms)
    building = [r for r in rooms if getattr(r, "category", "") not in SITE_CATEGORIES and r.w > 0 and r.h > 0]
    fp = getattr(plan, "building_footprint", None)
    footprint = tuple(fp) if fp and len(fp) == 4 and fp[2] > 0 else building_footprint(rooms)
    rooms_by_name = {r.name: r for r in rooms}

    windows = _window_records(plan, rooms_by_name)
    out.windows = windows
    if HAS_SHAPELY:
        try:
            out.walls, out.compound = _wall_poche(plan, windows)
        except Exception:
            out.walls, out.compound = [], []

    out.grid = structural_grid(rooms, footprint)

    sides = plan_sides_for_facing(getattr(plan, "orientation", "N"))
    cmap = {s: sides[SIDE_TO_PLAN[s]] for s in "NSEW"}

    def P(compass_scores):
        return {s: compass_scores.get(cmap[s], 0) for s in "NSEW"}

    has_dress = any("Dress" in r.name or "Wardrobe" in r.name for r in building)
    openings = list(getattr(plan, "openings", []) or [])

    for room in building:
        cat = getattr(room, "category", "")
        if cat in ("STAIRS", "LIFT"):
            continue
        sp = _Space(room)
        if not sp.usable:
            continue
        # door swings and walkway in front of every opening on this room's edges
        for d in plan.doors:
            horiz = d.wall_side in ("N", "S")
            a, b = (d.x, d.x + d.width) if horiz else (d.y, d.y + d.width)
            line = d.y if horiz else d.x
            edge = None
            if horiz and room.x - 0.02 < b and a < room.right + 0.02:
                edge = "N" if abs(room.top - line) < 0.03 else "S" if abs(room.y - line) < 0.03 else None
            elif not horiz and room.y - 0.02 < b and a < room.top + 0.02:
                edge = "E" if abs(room.right - line) < 0.03 else "W" if abs(room.x - line) < 0.03 else None
            if edge is None:
                continue
            depth = d.width + 0.08 if d.room_name == room.name else 0.75
            lo = a - 0.05 - (sp.x0 if edge in "NS" else sp.y0)
            sp.clear.append(sp.rect(edge, lo, b - a + 0.1, depth))
            sp.door_mids.append(((a + b) / 2, line) if horiz else (line, (a + b) / 2))
        for (x1, y1, x2, y2) in openings:
            horiz = abs(y1 - y2) < 1e-6
            a, b = sorted((x1, x2)) if horiz else sorted((y1, y2))
            line = y1 if horiz else x1
            edge = None
            if horiz and room.x - 0.02 < b and a < room.right + 0.02:
                edge = "N" if abs(room.top - line) < 0.03 else "S" if abs(room.y - line) < 0.03 else None
            elif not horiz and room.y - 0.02 < b and a < room.top + 0.02:
                edge = "E" if abs(room.right - line) < 0.03 else "W" if abs(room.x - line) < 0.03 else None
            if edge is None:
                continue
            lo = a - (sp.x0 if edge in "NS" else sp.y0)
            sp.clear.append(sp.rect(edge, lo, b - a, 0.8))
            sp.door_mids.append(((a + b) / 2, line) if horiz else (line, (a + b) / 2))
        for w in windows:
            if w["room"] == room.name:
                sp.windows[w["side"]].append((w["a"], w["b"]))

        mark = _rec_mark()
        try:
            if cat == "BEDROOM":
                prims = _furnish_bedroom(sp, room, P, has_dress and room.role == "MASTER_BEDROOM")
            elif cat == "LIVING":
                prims = _furnish_living(sp, room, P, "dining" in room.name.lower())
            elif cat == "DINING":
                prims = _furnish_dining(sp, room, P)
            elif cat == "KITCHEN":
                prims = _furnish_kitchen(sp, room, P)
            elif cat == "UTILITY":
                prims = _furnish_utility(sp, room, P)
            elif cat == "BATHROOM":
                prims = _furnish_bath(sp, room, P)
            elif cat == "POOJA":
                prims = _furnish_pooja(sp, room, P)
            elif cat == "STUDY":
                prims = _furnish_study(sp, room, P)
            elif cat == "STORE":
                prims = _furnish_store(sp, room, P)
            elif cat == "SERVANT":
                prims = _furnish_servant(sp, room, P)
            elif cat == "CORRIDOR":
                prims = _furnish_corridor(sp, room, P)
            else:
                prims = []
        except Exception:
            prims = []
            _rec_reset(mark)
        for it in out.items[mark:]:
            it["room"] = room.name
        out.furniture.extend(prims)

        # wet-area anti-skid tiles (300 mm grid)
        if cat in ("BATHROOM", "UTILITY"):
            step = 0.3
            x = sp.x0 + step
            while x < sp.x1 - 0.02:
                out.tiles.append(("line", (x, sp.y0), (x, sp.y1), "tile"))
                x += step
            y = sp.y0 + step
            while y < sp.y1 - 0.02:
                out.tiles.append(("line", (sp.x0, y), (sp.x1, y), "tile"))
                y += step

        # label in the freest spot near the centre (falls back to centre with a mask)
        if label_size:
            lw, lh = label_size(room)
            cx, cy = room.cx, room.cy
            spot, crowded = None, False
            if sp.free((cx - lw / 2, cy - lh / 2, cx + lw / 2, cy + lh / 2), use_clear=False):
                spot = (cx, cy)
            else:
                best = None
                stepx = max(0.05, (sp.x1 - sp.x0) / 24)
                stepy = max(0.05, (sp.y1 - sp.y0) / 24)
                x = sp.x0 + lw / 2
                while x <= sp.x1 - lw / 2 + EPS:
                    y = sp.y0 + lh / 2
                    while y <= sp.y1 - lh / 2 + EPS:
                        if sp.free((x - lw / 2, y - lh / 2, x + lw / 2, y + lh / 2), use_clear=False):
                            dd = math.hypot(x - cx, y - cy)
                            if best is None or dd < best[0]:
                                best = (dd, (x, y))
                        y += stepy
                    x += stepx
                if best:
                    spot = best[1]
                else:
                    spot, crowded = (cx, cy), True
            out.labels[room.name] = (spot[0], spot[1], crowded)
            sp.hard.append((spot[0] - lw / 2, spot[1] - lh / 2, spot[0] + lw / 2, spot[1] + lh / 2))

        if cat == "LIVING":
            main = next((d for d in plan.doors if d.is_main_entrance and d.room_name == room.name), None)
            target = (main.x + main.width + 0.8, main.y - 0.9) if main and main.wall_side == "N" else None
            spot = sp.place_free(1.15, 0.24, target, step=0.05)
            if spot:
                out.levels.append(((spot[0] + spot[2]) / 2, (spot[1] + spot[3]) / 2, "FFL +0.450"))

    # outside the main door: ground level beyond the entrance steps
    main = next((d for d in plan.doors if d.is_main_entrance), None)
    if main is not None:
        horiz = main.wall_side in ("N", "S")
        ox, oy = OUTWARD.get(main.wall_side, (0, 1))
        if horiz:
            out.levels.append((main.x + main.width + 1.6, main.y + oy * 1.9, "GL ±0.000"))
        else:
            out.levels.append((main.x + ox * 1.9, main.y + main.width + 1.6, "GL ±0.000"))

    out.tags, out.schedule = _schedule(plan, rooms_by_name, windows, footprint)
    g_prims, g_tags, g_rows = _gate_symbols(plan)
    out.gates, out.tags, out.schedule = g_prims, out.tags + g_tags, out.schedule + g_rows

    fx, fy, fw, fd = footprint
    if fw > 0 and fd > 0:
        k = 0.115 + 0.12
        out.rwp = [(fx - k, fy - k), (fx + fw + k, fy - k), (fx - k, fy + fd + k), (fx + fw + k, fy + fd + k)]
        if getattr(plan, "vastu_enabled", False):
            out.brahmasthan = (fx + fw / 3, fy + fd / 3, fx + 2 * fw / 3, fy + 2 * fd / 3)

    eng = getattr(plan, "engineering", {}) or {}
    g = out.grid
    out.notes = [
        "All dimensions in mm. Room sizes are centre-line; areas are carpet (RERA).",
        "Walls: 230 external / 115 internal brick, 12 mm plaster.",
        f"RCC frame M20 / Fe500: {len(g.columns)} columns 230 × 300 on grid "
        f"{len(g.xs)} × {len(g.ys)} (max bay {g.max_bay:.2f} m).",
    ]
    if eng.get("slab_mm"):
        out.notes.append(f"Slab {eng['slab_mm']} thk two-way; beams {eng['beam_mm'][0]} × {eng['beam_mm'][1]}.")
    out.notes += ["Plinth / FFL +450 above GL; toilets sunk 30 mm with floor traps.",
                  "Lintel at 2100; window sills 900 (kitchen 1050, ventilators 1800)."]
    return out
