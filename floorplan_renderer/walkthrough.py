"""
3D Walk-through scene — the plan as a house you can walk inside before it is built
==================================================================================
build_scene(plan, latitude=None) turns a GeneratedFloorPlan into the JSON scene that
model_folder/walkthrough3d/walk.js draws with Three.js (vendored — works fully offline).
Nothing is re-designed here: every wall, opening, stair, piece of furniture and amenity
comes from the plan and from the drafting layer the 2D blueprint is drawn with.

Heights (Indian residential practice, as the drawing's general notes):
  plinth / FFL +0.45 above ground · floor-to-floor = stair risers × riser (else 3.0 m) ·
  slab from the engineering report · door heads and lintels 2.1 m · window sills 0.9 m
  (kitchen 1.05, ventilators 1.8, heights as the drawing's opening schedule) · arches 2.4 m ·
  compound wall 1.5 m · terrace parapet 1.0 m · stilt floor when the car parks under the house

Walls are plan.walls (already cut at doors, arches and gates) split around every window into a
sill piece and a head piece, with lintels over doors/arches and corner posts at junctions (as
the drafting poché). Door leaves keep the plan's hinge and swing (floorplan_renderer._common).

Walking: two 0.1 m occupancy grids (walls, furniture, water, cars … grown by the walker's radius) —
the house / ground level and the level the stairs reach (the terrace of a one-storey house, else the
first-floor slab) — go to the browser for collision and tap-to-walk; every stair is climbable and
switches level half-way up. An outside terrace stair starts on the ground, so it is built to climb the
plinth too, and arrives over a bridge through a gap in the parapet. The guided tour is planned here —
road → gate → main door → every room, nearest one next — each stop with a plain-words note:
inside size in metres and feet, Vastu zone, daylight and what furniture fits.

Pure Python + numpy (no Streamlit).
"""

import base64
import math
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from floorplan_generator.vastu_engine import VastuEngine, plan_sides_for_facing
from floorplan_generator.house_report import room_schedule, ZONE_WORDS, good_zones
from floorplan_generator.structure import building_footprint
from floorplan_renderer._common import door_span, north_angle
from floorplan_renderer.drafting import build_drafting

SITE_CATS = {"PARKING", "BIKE_PARKING", "GARDEN", "POOL", "WATER_TANK", "GENERATOR", "SECURITY"}
SQFT = 10.7639
M2_PER_CENT = 40.4686

FFL = 0.45            # plinth: finished floor level above ground
DOOR_HEAD = 2.10
ARCH_HEAD = 2.40
COMPOUND_H = 1.50
PARAPET_H = 1.00
ROAD_W = 9.0           # a 30 ft residential road in front of the plot
VERGE = 0.6

CELL = 0.10            # walking grid (m)
BODY_R = 0.20          # walker radius (m)
EYE = 1.60

# approximate centre latitude of each state (for the sun path); the town would refine it by < 2°
STATE_LAT = {
    "Andhra Pradesh": 16.1, "Telangana": 17.6, "Tamil Nadu": 11.1, "Kerala": 10.3, "Karnataka": 14.5,
    "Maharashtra": 19.4, "Goa": 15.4, "Gujarat": 22.6, "Rajasthan": 26.6, "Madhya Pradesh": 23.5,
    "Chhattisgarh": 21.3, "Uttar Pradesh": 26.8, "Bihar": 25.6, "Jharkhand": 23.6, "West Bengal": 23.0,
    "Odisha": 20.5, "Assam": 26.2, "Punjab": 30.9, "Haryana": 29.2, "Himachal Pradesh": 31.9,
    "Uttarakhand": 30.1, "Jammu & Kashmir": 33.9, "Delhi": 28.6, "Chandigarh": 30.7, "Puducherry": 11.9,
    "Sikkim": 27.5, "Tripura": 23.8, "Meghalaya": 25.5, "Manipur": 24.7, "Mizoram": 23.2, "Nagaland": 26.1,
    "Arunachal Pradesh": 28.0,
}
DEFAULT_LAT = 17.0

OUT = {"N": (0, 1), "S": (0, -1), "E": (1, 0), "W": (-1, 0)}
INTO = {"N": (0, -1), "S": (0, 1), "E": (-1, 0), "W": (1, 0)}     # into the room that owns a door
NOT_STOPS = {"STAIRS", "LIFT"}


def _r(v: float, n: int = 3) -> float:
    return round(float(v), n)


def _rect(x0, y0, x1, y1) -> List[float]:
    return [_r(min(x0, x1)), _r(min(y0, y1)), _r(max(x0, x1)), _r(max(y0, y1))]


def _ft_in(m: float) -> str:
    inches = round(m / 0.0254)
    return f"{inches // 12}'{inches % 12}\""


def _is_site(r) -> bool:
    return getattr(r, "category", "") in SITE_CATS


# =============================================================================
# Heights
# =============================================================================

def levels(plan) -> Dict[str, float]:
    """Storey heights for the 3D model, from the plan's engineering report."""
    eng = getattr(plan, "engineering", None) or {}
    st = eng.get("stair") or {}
    H = 3.0
    if st.get("risers") and st.get("riser_mm"):
        H = st["risers"] * st["riser_mm"] / 1000.0
    H = min(3.6, max(2.9, H))
    slab = (eng.get("slab_mm") or 125) / 1000.0
    stilt = eng.get("car_parking") == "stilt"
    base = (0.15 + H) if stilt else FFL
    ceiling = H - slab
    return {"gl": 0.0, "ffl": _r(base), "plinth": FFL, "storey": _r(H), "slab": _r(slab), "ceiling": _r(ceiling),
            "door": DOOR_HEAD, "arch": _r(min(ARCH_HEAD, ceiling - 0.15)), "compound": COMPOUND_H,
            "parapet": PARAPET_H, "stilt": stilt, "eye": EYE}


def _window_heights(cat: str, length: float) -> Tuple[float, float, bool]:
    """(sill, head, ventilator) — the drawing's opening schedule (drafting._schedule)."""
    if cat in ("BATHROOM", "STORE") or length <= 0.6:
        return 1.80, 2.25, True
    h = {"KITCHEN": 1.0, "UTILITY": 0.9, "POOJA": 0.9, "SERVANT": 1.0}.get(cat, 1.2)
    sill = 1.05 if cat == "KITCHEN" else 0.9
    return sill, sill + h, False


# =============================================================================
# Walls, openings, slab
# =============================================================================

def _horiz(w) -> bool:
    return abs(w.y1 - w.y2) < 1e-6


def _span(w) -> Tuple[float, float, float]:
    """(lo, hi, line) of an axis-aligned wall centre-line."""
    if _horiz(w):
        lo, hi = sorted((w.x1, w.x2))
        return lo, hi, w.y1
    lo, hi = sorted((w.y1, w.y2))
    return lo, hi, w.x1


def _box(horiz: bool, a: float, b: float, line: float, t: float, z0: float, z1: float) -> List[float]:
    if horiz:
        return [_r(a), _r(line - t / 2), _r(b), _r(line + t / 2), _r(z0), _r(z1)]
    return [_r(line - t / 2), _r(a), _r(line + t / 2), _r(b), _r(z0), _r(z1)]


def _wall_t(walls, horiz: bool, line: float, a: float, b: float, default: float) -> float:
    """Thickness of the wall a door / arch sits in (the pieces either side of the cut)."""
    for w in walls:
        if getattr(w, "wall_type", "") == "compound" or _horiz(w) != horiz:
            continue
        lo, hi, ln = _span(w)
        if abs(ln - line) < 0.03 and (abs(hi - a) < 0.06 or abs(lo - b) < 0.06 or (lo < a and hi > b)):
            return w.thickness
    return default


def _gates(plan) -> List[Tuple[float, float]]:
    """Gaps in the compound wall along the road edge (plan top)."""
    W, H = plan.plot_width, plan.plot_height
    cov = []
    has = False
    for w in plan.walls:
        if getattr(w, "wall_type", "") != "compound":
            continue
        has = True
        if _horiz(w) and abs(w.y1 - H) < 0.03:
            lo, hi, _ = _span(w)
            cov.append((lo, hi))
    if not has:
        return []
    cov.sort()
    gaps, pos = [], 0.0
    for lo, hi in cov:
        if lo - pos > 0.6:
            gaps.append((pos, lo))
        pos = max(pos, hi)
    if W - pos > 0.6:
        gaps.append((pos, W))
    return gaps


def _walls(plan, lv, windows, footprint) -> Tuple[Dict[str, List], List[Dict], List[Dict], List[Dict]]:
    """Wall boxes by kind, window openings, door leaves and arches."""
    z0, z1 = lv["ffl"], lv["ffl"] + lv["ceiling"]
    fx, fy, fw, fd = footprint
    rooms = [r for r in plan.rooms if not _is_site(r) and r.w > 0 and r.h > 0]
    by_name = {r.name: r for r in plan.rooms}
    out: Dict[str, List] = {"ext": [], "int": [], "cmp": [], "post": [], "pillar": []}

    # windows → cuts on their wall line
    wins: List[Dict] = []
    cuts: Dict[Tuple[bool, float], List[Tuple[float, float, float, float]]] = {}
    for w in windows:
        horiz = w["side"] in ("N", "S")
        sill, head, vent = _window_heights(w["category"], w["length"])
        cuts.setdefault((horiz, round(w["coord"], 2)), []).append((w["a"], w["b"], sill, head))
        a, b, c = w["a"], w["b"], w["coord"]
        pa, pb = ((a, c), (b, c)) if horiz else ((c, a), (c, b))
        wins.append({"a": [_r(pa[0]), _r(pa[1])], "b": [_r(pb[0]), _r(pb[1])], "t": _r(w["t"]),
                     "z0": _r(z0 + sill), "z1": _r(z0 + head), "vent": vent, "room": w["room"],
                     "out": list(w["out"]), "side": w["side"]})

    segs = []
    for w in plan.walls:
        kind = getattr(w, "wall_type", "") or ("external" if w.is_external else "internal")
        horiz = _horiz(w)
        if not horiz and abs(w.x1 - w.x2) > 1e-6:
            continue
        lo, hi, line = _span(w)
        t = w.thickness
        if kind == "compound":
            out["cmp"].append(_box(horiz, lo, hi, line, t, 0.0, COMPOUND_H))
            continue
        segs.append(w)
        key = "ext" if kind == "external" else "int"
        mine = [c for c in cuts.get((horiz, round(line, 2)), []) + cuts.get((horiz, round(line + 0.01, 2)), [])
                + cuts.get((horiz, round(line - 0.01, 2)), []) if c[1] > lo + 0.01 and c[0] < hi - 0.01]
        pos = lo
        for a, b, sill, head in sorted(set(mine)):
            a, b = max(a, lo), min(b, hi)
            if b - a < 0.01:
                continue
            if a > pos + 0.005:
                out[key].append(_box(horiz, pos, a, line, t, z0, z1))
            out[key].append(_box(horiz, a, b, line, t, z0, z0 + sill))
            out[key].append(_box(horiz, a, b, line, t, z0 + head, z1))
            pos = max(pos, b)
        if hi > pos + 0.005:
            out[key].append(_box(horiz, pos, hi, line, t, z0, z1))

    # junction posts (centre-line segments stop at the node — same rule as the drafting poché)
    seen = set()
    for w in segs:
        horiz = _horiz(w)
        for (px, py) in ((w.x1, w.y1), (w.x2, w.y2)):
            for o in segs:
                if o is w or _horiz(o) == horiz:
                    continue
                lo, hi, ln = _span(o)
                hit = (abs(ln - px) < 0.02 and lo - 0.02 <= py <= hi + 0.02) if horiz else \
                      (abs(ln - py) < 0.02 and lo - 0.02 <= px <= hi + 0.02)
                if hit:
                    s = max(w.thickness, o.thickness) / 2
                    k = (round(px, 2), round(py, 2))
                    if k not in seen:
                        seen.add(k)
                        out["post"].append([_r(px - s), _r(py - s), _r(px + s), _r(py + s), _r(z0), _r(z1)])
                    break

    # compound corners and gate pillars
    W, H = plan.plot_width, plan.plot_height
    cmp_segs = [w for w in plan.walls if getattr(w, "wall_type", "") == "compound"]
    if cmp_segs:
        for (px, py) in ((0, 0), (W, 0), (W, H), (0, H)):
            out["pillar"].append([_r(px - 0.15), _r(py - 0.15), _r(px + 0.15), _r(py + 0.15), 0.0, _r(COMPOUND_H + 0.1)])
        for g0, g1 in _gates(plan):
            for gx in (g0, g1):
                if 0.2 < gx < W - 0.2:
                    out["pillar"].append([_r(gx - 0.18), _r(H - 0.18), _r(gx + 0.18), _r(H + 0.18), 0.0, 1.85])

    # doors: lintel over each, and a leaf that keeps the plan's hinge and swing
    doors = []
    for i, d in enumerate(plan.doors):
        horiz = d.wall_side in ("N", "S")
        a, b = door_span(d)
        line = d.y if horiz else d.x
        on_edge = (abs(line - fy) < 0.03 or abs(line - (fy + fd)) < 0.03) if horiz else \
                  (abs(line - fx) < 0.03 or abs(line - (fx + fw)) < 0.03)
        t = _wall_t(plan.walls, horiz, line, a, b, 0.23 if on_edge else 0.115)
        key = "ext" if on_edge else "int"
        out[key].append(_box(horiz, a, b, line, t, z0 + DOOR_HEAD, z1))
        mid = ((a + b) / 2, line) if horiz else (line, (a + b) / 2)
        nx, ny = INTO[d.wall_side]
        probe = (mid[0] - nx * (t / 2 + 0.25), mid[1] - ny * (t / 2 + 0.25))
        other = next((r.name for r in rooms if r.x < probe[0] < r.right and r.y < probe[1] < r.top), "Outside")
        room = by_name.get(d.room_name)
        cat = getattr(room, "category", "")
        kind = ("main" if d.is_main_entrance else "lift" if cat == "LIFT" else "bath" if cat == "BATHROOM"
                else "pooja" if cat == "POOJA" else "service" if on_edge else "internal")
        right = d.swing_direction != "LEFT"
        closed = (0.0 if right else 180.0) if horiz else (90.0 if right else 270.0)
        opened = {"N": 270.0, "S": 90.0, "E": 180.0, "W": 0.0}[d.wall_side]
        doors.append({"id": i, "hinge": [_r(d.x), _r(d.y)], "w": _r(d.width), "t": _r(t), "h": DOOR_HEAD,
                      "closed": closed, "open": opened, "kind": kind, "room": d.room_name, "other": other,
                      "mid": [_r(mid[0]), _r(mid[1])], "n": [nx, ny], "horiz": horiz})

    arches = []
    for (x1, y1, x2, y2) in (getattr(plan, "openings", None) or []):
        horiz = abs(y1 - y2) < 1e-6
        a, b = sorted((x1, x2)) if horiz else sorted((y1, y2))
        line = y1 if horiz else x1
        t = _wall_t(plan.walls, horiz, line, a, b, 0.115)
        out["int"].append(_box(horiz, a, b, line, t, z0 + lv["arch"], z1))
        arches.append({"a": [_r(x1), _r(y1)], "b": [_r(x2), _r(y2)], "t": _r(t), "h": lv["arch"]})
    return out, wins, doors, arches


def _ring(o, t) -> List[List[float]]:
    """Four strips of width t on the inside of rectangle o."""
    return [[o[0], o[1], o[2], o[1] + t], [o[0], o[3] - t, o[2], o[3]],
            [o[0], o[1] + t, o[0] + t, o[3] - t], [o[2] - t, o[1] + t, o[2], o[3] - t]]


def _minus(rect, holes) -> List[List[float]]:
    """Axis-aligned rectangle minus holes → list of rectangles (for the slab's stair openings)."""
    pieces = [list(rect)]
    for hx0, hy0, hx1, hy1 in holes:
        nxt = []
        for x0, y0, x1, y1 in pieces:
            if hx1 <= x0 or hx0 >= x1 or hy1 <= y0 or hy0 >= y1:
                nxt.append([x0, y0, x1, y1])
                continue
            ix0, iy0, ix1, iy1 = max(x0, hx0), max(y0, hy0), min(x1, hx1), min(y1, hy1)
            if iy0 > y0:
                nxt.append([x0, y0, x1, iy0])
            if iy1 < y1:
                nxt.append([x0, iy1, x1, y1])
            if ix0 > x0:
                nxt.append([x0, iy0, ix0, iy1])
            if ix1 < x1:
                nxt.append([ix1, iy0, x1, iy1])
        pieces = nxt
    return [[_r(v) for v in p] for p in pieces if p[2] - p[0] > 0.02 and p[3] - p[1] > 0.02]


# =============================================================================
# Stairs
# =============================================================================

def _stair(sc, base: float, storey: float, external: bool, top: Optional[float] = None,
           entries: Sequence[Tuple[float, float]] = (), outer: Optional[Sequence[float]] = None) -> Dict:
    """
    Treads, landings and rails of a staircase (boxes in plan metres, absolute heights) and how it is
    walked: rectangles closed on the lower level (block0) or the upper one (block1), rectangles open
    upstairs (free1: arrival, exit, bridge), barriers (rails, the well) and the balustrade round the
    opening upstairs (ring). `top` = the level the stair must reach; an outside stair starts on the
    ground, so it also climbs the plinth and arrives over a bridge onto the roof next to it.
    """
    along_y = sc.h >= sc.w
    L, Wd = (sc.h, sc.w) if along_y else (sc.w, sc.h)
    n = max(4, int(sc.num_steps or 15))
    r = sc.riser_height_m or storey / n
    t = sc.tread_depth_m or 0.25
    if top is not None and abs(base + n * r - top) > 0.02:
        n = max(4, math.ceil((top - base) / 0.19 - 1e-9))
        r = (top - base) / n
    z_top = base + n * r
    x0, y0, x1, y1 = sc.x, sc.y, sc.x + sc.w, sc.y + sc.h

    def frame(fs, fq):     # local (s along the flight, q across) → plan, optionally mirrored
        def P(s_, q_):
            s_, q_ = (L - s_ if fs else s_), (Wd - q_ if fq else q_)
            return (sc.x + q_, sc.y + s_) if along_y else (sc.x + s_, sc.y + q_)
        return P

    def rect_of(P, s0, s1, q0, q1):
        (ax, ay), (bx, by) = P(s0, q0), P(s1, q1)
        return _rect(ax, ay, bx, by)

    out = {"boxes": [], "rails": [], "block0": [], "block1": [], "free1": [], "barrier": [], "ring": [],
           "exit": None, "bridge": None}

    def rail(pts):
        if len(pts) > 1:
            out["rails"].append([[_r(a), _r(b), _r(c)] for a, b, c in pts])

    stype = (sc.stair_type or "dog_leg").lower()
    if stype == "straight" or external or Wd < 1.6:
        flights = 2 if (getattr(sc, "has_landing", False) and n > 6) else 1
        n1 = (n + 1) // 2 if flights == 2 else n
        mid_land = (sc.landing_depth_m or Wd) if flights == 2 else 0.0
        treads = n - flights                     # each flight's last riser lands on a landing / the floor
        tt = min(t, (L - mid_land) / max(1, treads))
        s0 = max(0.0, L - mid_land - treads * tt)
        P, side = frame(False, False), None
        if external and outer is not None:       # rise towards the end that lies alongside the house
            best = None
            for fs in (False, True):
                a = rect_of(frame(fs, False), L - 0.9, L, 0, Wd)
                lo, hi = (a[1], a[3]) if along_y else (a[0], a[2])
                olo, ohi = (outer[1], outer[3]) if along_y else (outer[0], outer[2])
                ov = min(hi, ohi - 0.2) - max(lo, olo + 0.2)
                if best is None or ov > best[0] + 1e-6:
                    best = (ov, fs)
            P = frame(best[1], False)
            gaps = ({"+": outer[0] - x1, "-": x0 - outer[2]} if along_y else {"+": outer[1] - y1, "-": y0 - outer[3]})
            near = sorted((g, k) for k, g in gaps.items() if g > -0.05)
            if near and near[0][0] <= 0.8 and best[0] >= 0.6:
                side = near[0][1]
        steps, sp = [], s0
        for k in range(1, n1):
            steps.append((sp, sp + tt, base + k * r))
            sp += tt
        if flights == 2:
            steps.append((sp, sp + mid_land, base + n1 * r))
            sp += mid_land
            for k in range(n1 + 1, n):
                steps.append((sp, sp + tt, base + k * r))
                sp += tt
        for a, b, h in steps:
            out["boxes"].append(rect_of(P, a, b, 0, Wd) + [_r(base), _r(h)])
        house_q = None if side is None else (Wd if side == "+" else 0.0)
        open_qs = [0.0, Wd] if house_q is None else [Wd - house_q]
        first = next((a for a, b, h in steps if h - base > 0.45), s0)
        for q in open_qs:                         # rails where a fall is possible, and across the top end
            qq = 0.05 if q == 0.0 else Wd - 0.05
            rail([P((a + b) / 2, qq) + (h + 0.9,) for a, b, h in steps if h - base > 0.45]
                 + [P(L - 0.05, qq) + (z_top - r + 0.9,)])
            out["barrier"].append(rect_of(P, first, L, max(0.0, qq - 0.03), min(Wd, qq + 0.03)))
        rail([P(L - 0.05, 0.05) + (z_top - r + 0.9,), P(L - 0.05, Wd - 0.05) + (z_top - r + 0.9,)])
        out["barrier"].append(rect_of(P, L - 0.06, L, 0, Wd))
        if house_q is not None:                  # bridge from the top steps over the gap and the parapet
            span0, reach = L - 0.9, 0.15 + 0.4
            a = rect_of(P, span0, L, 0, Wd)
            if along_y:
                bx = (x1, outer[0] + reach) if side == "+" else (outer[2] - reach, x0)
                bridge = _rect(bx[0], a[1], bx[1], a[3])
            else:
                by = (y1, outer[1] + reach) if side == "+" else (outer[3] - reach, y0)
                bridge = _rect(a[0], by[0], a[2], by[1])
            out["boxes"].append(bridge + [_r(z_top - 0.15), _r(z_top)])
            out["bridge"] = bridge
            out["free1"].append(bridge)
            # the top steps reach over the house wall (it ends under the roof slab) onto the bridge
            out["free1"].append(rect_of(P, span0, L, Wd - 0.35, Wd) if house_q == Wd else rect_of(P, span0, L, 0, 0.35))
            hq = Wd - 0.05 if house_q == Wd else 0.05
            out["barrier"].append(rect_of(P, first, span0, max(0.0, hq - 0.03), min(Wd, hq + 0.03)))
        if s0 > 0.05:
            out["block1"].append(rect_of(P, 0, s0, 0, Wd))
        out.update(top=_r(z_top), n=n, riser=r, tread=tt)
        return out

    gap = 0.1
    fw = (Wd - gap) / 2
    n1 = (n + 1) // 2
    n2 = n - n1
    landing = max(sc.landing_depth_m or 1.0, min(fw, 1.2))
    tt = t
    if L - landing - (n1 - 1) * tt < 0:
        tt = max(0.2, (L - landing) / max(1, n1 - 1))
        landing = max(0.8, L - (n1 - 1) * tt)
    s_land = L - landing
    s_start = max(0.0, s_land - (n1 - 1) * tt)
    s_end = max(0.0, s_land - (n2 - 1) * tt)

    # orientation: the first flight starts next to the way in, and upstairs you step out onto the floor
    opts = ([(True, False), (False, False), (True, True), (False, True)] if along_y
            else [(False, False), (True, False), (False, True), (True, True)])

    def score(o):
        Pf = frame(*o)
        fx, fy = Pf(max(0.15, s_start / 2), fw / 2)
        d = min((math.hypot(fx - ex, fy - ey) for ex, ey in entries), default=0.0)
        ex_ = rect_of(Pf, -0.9, 0.0, Wd - fw, Wd)
        inside = outer is None or (ex_[0] >= outer[0] - 0.01 and ex_[1] >= outer[1] - 0.01
                                   and ex_[2] <= outer[2] + 0.01 and ex_[3] <= outer[3] + 0.01)
        return (0 if inside else 1, round(d, 1), opts.index(o))
    P = frame(*min(opts, key=score))

    def box(s0, s1, q0, q1, za, zb):
        return rect_of(P, s0, s1, q0, q1) + [_r(za), _r(zb)]

    # flight 1 (solid, from the floor), landing, flight 2 (stepped waist slab back over the start)
    for k in range(1, n1):
        s_a = s_land - (n1 - k) * tt
        out["boxes"].append(box(max(0, s_a), s_land - (n1 - k - 1) * tt, 0, fw, base, base + k * r))
    zl = base + n1 * r
    out["boxes"].append(box(s_land, L, 0, Wd, zl - 0.15, zl))
    for j in range(1, n2):
        top_j = zl + j * r
        out["boxes"].append(box(max(0, s_land - j * tt), s_land - (j - 1) * tt, Wd - fw, Wd, top_j - r - 0.15, top_j))
    if s_end > 0.05:   # arrival floor of the level above, inside the stair well
        out["boxes"].append(box(0, s_end, Wd - fw, Wd, z_top - 0.15, z_top))
    # rails along the well (inner edges) and a balustrade across the landing
    r1 = [P(s_land - (n1 - k - 0.5) * tt, fw) + (base + k * r + 0.9,) for k in range(1, n1)]
    r1.append(P(s_land, fw) + (zl + 0.9,))
    r2 = [P(s_land, Wd - fw) + (zl + 0.9,)]
    r2 += [P(s_land - (j - 0.5) * tt, Wd - fw) + (zl + j * r + 0.9,) for j in range(1, n2)]
    rail(r1)
    rail(r2)
    out["barrier"].append(rect_of(P, 0, s_land, fw, Wd - fw))           # the well between the flights
    if s_end > 0.05:
        out["block0"].append(rect_of(P, 0, s_end, Wd - fw, Wd))           # the arrival floor is upstairs only
        out["free1"].append(rect_of(P, 0, s_end, Wd - fw, Wd))
    if s_start > 0.05:
        out["block1"].append(rect_of(P, 0, s_start, 0, fw))               # the foot is downstairs only
    exit_ = rect_of(P, -0.9, 0.0, Wd - fw, Wd)
    out["exit"] = exit_
    out["free1"].append(exit_)
    # balustrade round the opening upstairs, open where you step out
    ring = [[x0 - 0.12, y0 - 0.12, x1 + 0.12, y0], [x0 - 0.12, y1, x1 + 0.12, y1 + 0.12],
            [x0 - 0.12, y0, x0, y1], [x1, y0, x1 + 0.12, y1]]
    for rc in ring:
        for piece in _minus(rc, [exit_]):
            out["ring"].append(piece)
            horiz = piece[2] - piece[0] >= piece[3] - piece[1]
            c = (piece[1] + piece[3]) / 2 if horiz else (piece[0] + piece[2]) / 2
            a, b = ((piece[0], c), (piece[2], c)) if horiz else ((c, piece[1]), (c, piece[3]))
            rail([a + (z_top + 0.9,), b + (z_top + 0.9,)])
    out.update(top=_r(z_top), n=n, riser=r, tread=tt)
    return out


# =============================================================================
# Site
# =============================================================================

def _slots(name: str, word: str, default: int) -> int:
    try:
        if "(" in name and word in name:
            return max(1, int(name.split("(")[1].split()[0]))
    except Exception:
        pass
    return default


def _cars(r, base: Optional[Tuple[float, float, float, float]] = None) -> List[Dict]:
    """Cars in their stalls — the same stall layout as the SVG drawing."""
    x, y, w, h = base or (r.x, r.y, r.w, r.h)
    slots = _slots(r.name, "Bay", 1) if base is None else 1
    cars = []
    if w >= h:
        sw = w / slots
        for i in range(slots):
            along_x = sw > h
            L = min(4.5, (sw if along_x else h) * 0.86)
            Wc = min(1.8, (h if along_x else sw) * 0.70)
            cars.append({"c": [_r(x + (i + 0.5) * sw), _r(y + h / 2)], "along": "x" if along_x else "y",
                         "L": _r(L), "W": _r(Wc), "i": i})
    else:
        sh = h / slots
        for i in range(slots):
            along_x = w > sh
            L = min(4.5, (w if along_x else sh) * 0.86)
            Wc = min(1.8, (sh if along_x else w) * 0.70)
            cars.append({"c": [_r(x + w / 2), _r(y + (i + 0.5) * sh)], "along": "x" if along_x else "y",
                         "L": _r(L), "W": _r(Wc), "i": i})
    return cars


def _bikes(r) -> List[Dict]:
    n = _slots(r.name, "Bike", 2)
    split_x = abs(r.w / n - 0.8) <= abs(r.h / n - 0.8)
    length = r.w if split_x else r.h
    depth = r.h if split_x else r.w
    per_row = max(1, int(length / 0.62))
    rows = max(1, min(2, math.ceil(n / per_row)))
    out = []
    for k in range(n):
        row, i = divmod(k, per_row)
        if row >= rows:
            break
        cnt = min(per_row, n - row * per_row)
        u = (i + 0.5) * length / cnt
        v = (row + 0.5) * depth / rows
        c = (r.x + u, r.y + v) if split_x else (r.x + v, r.y + u)
        out.append({"c": [_r(c[0]), _r(c[1])], "along": "y" if split_x else "x",
                    "L": _r(min(1.8, depth / rows * 0.92)), "i": k})
    return out


def _trees(r) -> List[List[float]]:
    if r.w < 1.4 or r.h < 1.4:
        return [[_r(r.cx), _r(r.cy), 0.8]]
    inset = min(0.8, r.w / 4, r.h / 4)
    pts = [(r.x + inset, r.y + inset), (r.right - inset, r.y + inset),
           (r.x + inset, r.top - inset), (r.right - inset, r.top - inset)]
    size = min(1.6, max(0.8, min(r.w, r.h) / 2.5))
    out = [[_r(x), _r(y), _r(size)] for x, y in pts]
    if r.w * r.h > 24:
        out.append([_r(r.cx), _r(r.cy), _r(size * 1.2)])
    return out


# =============================================================================
# Walking grid
# =============================================================================

class _Grid:
    def __init__(self, x0, y0, x1, y1):
        self.x0, self.y0 = x0, y0
        self.nx = int(math.ceil((x1 - x0) / CELL))
        self.ny = int(math.ceil((y1 - y0) / CELL))
        self.free = np.ones((self.ny, self.nx), dtype=bool)

    def like(self, free: bool = True) -> "_Grid":
        """A grid with exactly the same cells."""
        g = _Grid.__new__(_Grid)
        g.x0, g.y0, g.nx, g.ny = self.x0, self.y0, self.nx, self.ny
        g.free = self.free.copy() if free else np.zeros_like(self.free)
        return g

    def block(self, x0, y0, x1, y1, pad=0.0):
        self.set((x0, y0, x1, y1), False, pad)

    def span(self, rc, pad=0.0, clip=None):
        """Index slices of the cells whose centres lie inside rc (grown by pad, cut to clip)."""
        x0, y0, x1, y1 = rc[0] - pad, rc[1] - pad, rc[2] + pad, rc[3] + pad
        if clip is not None:
            x0, y0, x1, y1 = max(x0, clip[0]), max(y0, clip[1]), min(x1, clip[2]), min(y1, clip[3])
        xs = self.x0 + (np.arange(self.nx) + 0.5) * CELL
        ys = self.y0 + (np.arange(self.ny) + 0.5) * CELL
        ix = np.nonzero((xs > x0) & (xs < x1))[0]
        iy = np.nonzero((ys > y0) & (ys < y1))[0]
        if not len(ix) or not len(iy):
            return None
        return slice(iy[0], iy[-1] + 1), slice(ix[0], ix[-1] + 1)

    def set(self, rc, value, pad=0.0, clip=None):
        sl = self.span(rc, pad, clip)
        if sl:
            self.free[sl] = value

    def keep_only(self, x0, y0, x1, y1):
        mask = np.zeros_like(self.free)
        xs = self.x0 + (np.arange(self.nx) + 0.5) * CELL
        ys = self.y0 + (np.arange(self.ny) + 0.5) * CELL
        ix = np.nonzero((xs > x0) & (xs < x1))[0]
        iy = np.nonzero((ys > y0) & (ys < y1))[0]
        if len(ix) and len(iy):
            mask[iy[0]:iy[-1] + 1, ix[0]:ix[-1] + 1] = True
        self.free &= mask

    def cell(self, x, y) -> Tuple[int, int]:
        return int((x - self.x0) / CELL), int((y - self.y0) / CELL)

    def centre(self, i, j) -> Tuple[float, float]:
        return self.x0 + (i + 0.5) * CELL, self.y0 + (j + 0.5) * CELL

    def ok(self, i, j) -> bool:
        return 0 <= i < self.nx and 0 <= j < self.ny and bool(self.free[j, i])

    def nearest_free(self, x, y, within=1.5) -> Optional[Tuple[int, int]]:
        ci, cj = self.cell(x, y)
        if self.ok(ci, cj):
            return ci, cj
        k = int(within / CELL)
        best = None
        for dj in range(-k, k + 1):
            for di in range(-k, k + 1):
                i, j = ci + di, cj + dj
                if self.ok(i, j):
                    d = di * di + dj * dj
                    if best is None or d < best[0]:
                        best = (d, (i, j))
        return best[1] if best else None

    def bfs(self, start: Tuple[int, int], targets: Optional[set] = None):
        """Uniform-cost 8-neighbour BFS (no corner cutting). Stops once every target is reached."""
        nx, ny, free = self.nx, self.ny, self.free
        parent = {start: None}
        dq = deque([start])
        left = set(targets) if targets else None
        while dq:
            c = dq.popleft()
            if left is not None:
                left.discard(c)
                if not left:
                    break
            i, j = c
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                a, b = i + di, j + dj
                if not (0 <= a < nx and 0 <= b < ny) or (a, b) in parent or not free[b, a]:
                    continue
                if di and dj and not (free[j, a] and free[b, i]):
                    continue
                parent[(a, b)] = c
                dq.append((a, b))
        return parent

    def path(self, parent, goal) -> List[Tuple[int, int]]:
        out, c = [], goal
        while c is not None:
            out.append(c)
            c = parent.get(c)
        return out[::-1]

    def los(self, a, b) -> bool:
        (i0, j0), (i1, j1) = a, b
        n = int(max(abs(i1 - i0), abs(j1 - j0)) * 2) + 1
        for k in range(n + 1):
            f = k / n
            i, j = int(round(i0 + (i1 - i0) * f)), int(round(j0 + (j1 - j0) * f))
            if not self.ok(i, j):
                return False
        return True

    def smooth(self, cells: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if len(cells) < 3:
            return cells
        out = [cells[0]]
        k = 0
        while k < len(cells) - 1:
            nxt = k + 1
            for m in range(len(cells) - 1, k + 1, -1):
                if self.los(cells[k], cells[m]):
                    nxt = m
                    break
            out.append(cells[nxt])
            k = nxt
        return out

    def clearance(self) -> np.ndarray:
        """Cells to the nearest blocked cell (4-neighbour BFS from every blocked cell)."""
        d = np.full(self.free.shape, -1, dtype=np.int32)
        dq = deque()
        for j, i in zip(*np.nonzero(~self.free)):
            d[j, i] = 0
            dq.append((i, j))
        while dq:
            i, j = dq.popleft()
            for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if 0 <= a < self.nx and 0 <= b < self.ny and d[b, a] < 0:
                    d[b, a] = d[j, i] + 1
                    dq.append((a, b))
        d[d < 0] = 99
        return d

    def encode(self) -> str:
        return base64.b64encode(np.packbits(self.free.astype(np.uint8).ravel()).tobytes()).decode("ascii")


def _furniture_rect(it) -> Optional[Tuple[float, float, float, float, bool]]:
    """Plan footprint of a recorded furniture item → (x0, y0, x1, y1, tall)."""
    k = it["kind"]
    if k in ("floor_trap", "shower", "sink", "hob"):
        return None
    if k == "planter":
        rr = it.get("r", 0.2)
        return it["x"] - rr, it["y"] - rr, it["x"] + rr, it["y"] + rr, False
    o, u, v = it["o"], it["u"], it["v"]
    if k == "counter":
        u0, u1, d = it["a"], it["b"], it.get("D", 0.6)
        v0, v1 = 0.0, d
    elif k == "dining":
        u0, u1, v0, v1 = 0.0, it["W"], -0.52, it["D"] + 0.52
    elif k == "desk":
        u0, u1, v0, v1 = 0.0, it["W"], 0.0, it["D"] + 0.45
    else:
        u0, u1, v0, v1 = 0.0, it.get("W", 0.5), 0.0, it.get("D", 0.5)
    pts = [(o[0] + u[0] * a + v[0] * b, o[1] + u[1] * a + v[1] * b) for a in (u0, u1) for b in (v0, v1)]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    tall = k in ("wardrobe", "fridge", "bookshelf", "shelves", "tv_unit")
    return min(xs), min(ys), max(xs), max(ys), tall


# =============================================================================
# Words for the guided tour
# =============================================================================

def _verdict(score: Optional[float]) -> Tuple[str, str]:
    if score is None:
        return "", ""
    if score >= 8:
        return "best", "the best place for it in Vastu"
    if score >= 3:
        return "good", "a good place for it in Vastu"
    if score >= -10:
        return "ok", "acceptable in Vastu"
    return "poor", "not ideal in Vastu"


def _bed_word(w: float) -> str:
    return "king-size" if w >= 1.75 else "queen-size" if w >= 1.45 else "double" if w >= 1.15 else "single"


def _fits(items: List[Dict]) -> List[str]:
    out, kinds = [], {}
    for it in items:
        kinds.setdefault(it["kind"], []).append(it)
    for it in kinds.get("bed", []):
        out.append(f"a {_bed_word(it['W'])} bed ({it['W']:.1f} × {it['D']:.1f} m)")
    if kinds.get("side_table"):
        n = len(kinds["side_table"])
        out.append("two side tables" if n == 2 else "a side table" if n == 1 else f"{n} side tables")
    for it in kinds.get("wardrobe", []):
        out.append(f"a {it['W']:.1f} m wardrobe")
    for it in kinds.get("sofa", []):
        out.append(f"a {it['seats']}-seat sofa" if it.get("seats", 1) > 1 else "an easy chair")
    for it in kinds.get("dining", []):
        out.append(f"a {it['seats']}-seat dining table")
    if kinds.get("counter"):
        run = sum(it["b"] - it["a"] for it in kinds["counter"])
        out.append(f"{run:.1f} m of kitchen counter with sink and stove")
    words = {"tv_unit": "a TV unit", "coffee_table": "a coffee table", "fridge": "a fridge",
             "washer": "a washing machine", "utility_sink": "a wash sink", "desk": "a study desk with chair",
             "bookshelf": "a bookshelf", "shelves": "storage shelves", "pooja": "a pooja unit",
             "shoe_rack": "a shoe rack"}
    for k, w in words.items():
        if kinds.get(k):
            out.append(w)
    bath = [w for k, w in (("wc", "a WC"), ("basin", "a wash basin"), ("shower", "a shower area")) if kinds.get(k)]
    if bath:
        out.append(", ".join(bath))
    return out


def _join(words: Sequence[str]) -> str:
    words = [w for w in words if w]
    if len(words) <= 1:
        return "".join(words)
    return ", ".join(words[:-1]) + " and " + words[-1]


def _room_story(r, row, vrow, wins_c, items, stair_info=None) -> Tuple[str, str]:
    """(title, sentence) the guide says in a room."""
    zone = ZONE_WORDS.get(row["zone"], row["zone"]) if row else ""
    title = r.name
    parts = []
    if row:
        parts.append(f"{r.name}. Inside it is {row['w_ft']} by {row['h_ft']} — {row['w']:.1f} by {row['h']:.1f} metres, "
                     f"{row['sqft']} square feet.")
    if vrow is not None:
        _, txt = _verdict(vrow.get("score"))
        sent = f"It sits in the {ZONE_WORDS.get(vrow['zone'], vrow['zone'])} — {txt}"
        if vrow.get("score", 10) < -10:
            best = [ZONE_WORDS.get(z, z) for z in good_zones(vrow["key"])[:2]]
            if best:
                sent += f"; {' or '.join(best)} would be better"
        parts.append(sent + ".")
    elif zone:
        parts.append(f"It is in the {zone} of the house.")
    habitable = r.category in ("LIVING", "BEDROOM", "STUDY", "DINING", "KITCHEN")
    if wins_c:
        uniq = sorted(set(wins_c), key=wins_c.index)
        sides = _join([ZONE_WORDS.get(s, s) for s in uniq])
        if r.category in ("BATHROOM", "STORE"):
            parts.append(f"A ventilator high on the {sides} wall lets air out.")
        else:
            sun = " — morning sun comes in" if "E" in uniq else (" — evening sun, it can get warm" if uniq == ["W"] else "")
            parts.append(f"{'One window faces' if len(wins_c) == 1 else f'{len(wins_c)} windows face'} {sides}{sun}.")
    elif habitable:
        parts.append("There is no outside window here, so it needs a light and fan.")
    elif r.category == "BATHROOM":
        parts.append("It has no outside wall, so it needs an exhaust fan.")
    fits = _fits(items)
    if fits:
        parts.append(f"It fits {_join(fits)}.")
    return title, " ".join(parts)


# =============================================================================
# Scene
# =============================================================================

def build_scene(plan, latitude: Optional[float] = None, state: Optional[str] = None) -> Dict:
    """The JSON scene for the 3D walk-through (see the module docstring)."""
    lv = levels(plan)
    W, H = float(plan.plot_width), float(plan.plot_height)
    rooms_all = list(plan.rooms)
    building = [r for r in rooms_all if not _is_site(r) and r.w > 0 and r.h > 0]
    fp = getattr(plan, "building_footprint", None)
    footprint = tuple(fp) if fp and len(fp) == 4 and fp[2] > 0 else building_footprint(rooms_all)
    fx, fy, fw, fd = footprint
    df = build_drafting(plan)
    walls, wins, doors, arches = _walls(plan, lv, df.windows, footprint)
    sides = plan_sides_for_facing(getattr(plan, "orientation", "N"))
    compass = {s: sides[p] for s, p in (("N", "top"), ("S", "bottom"), ("E", "right"), ("W", "left"))}

    # rooms
    sched = {row["name"]: row for row in room_schedule(plan)["rows"]}
    vrows = {row["room"]: row for row in (getattr(plan, "vastu_breakdown", None) or {}).get("rooms", [])}
    wins_by_room: Dict[str, List[str]] = {}
    for w in wins:
        wins_by_room.setdefault(w["room"], []).append(compass.get(w["side"], w["side"]))
    items_by_room: Dict[str, List[Dict]] = {}
    for it in df.items:
        items_by_room.setdefault(it.get("room", ""), []).append(it)
    rooms = []
    for r in building:
        row = sched.get(r.name)
        vrow = vrows.get(r.name)
        title, say = _room_story(r, row, vrow, wins_by_room.get(r.name, []), items_by_room.get(r.name, []))
        rating, _ = _verdict(vrow.get("score") if vrow else None)
        rooms.append({
            "name": r.name, "cat": r.category, "role": getattr(r, "role", "") or "", "key": getattr(r, "key", "") or "",
            "r": _rect(r.x, r.y, r.right, r.top), "area": _r(row["area"] if row else r.area, 2),
            "sqft": row["sqft"] if row else round(r.area * SQFT),
            "size": f"{row['w']:.1f} × {row['h']:.1f} m" if row else f"{r.w:.1f} × {r.h:.1f} m",
            "ft": f"{row['w_ft']} × {row['h_ft']}" if row else f"{_ft_in(r.w)} × {_ft_in(r.h)}",
            "zone": (vrow or {}).get("zone") or (row or {}).get("zone", ""),
            "zone_word": ZONE_WORDS.get((vrow or {}).get("zone") or (row or {}).get("zone", ""), ""),
            "vastu": rating, "windows": len(wins_by_room.get(r.name, [])), "say": say})

    # stairs (each raised to the level it serves) and lift
    e = 0.115
    outer = [fx - e, fy - e, fx + fw + e, fy + fd + e]
    n_floors = max(1, int(getattr(plan, "num_floors", 1) or 1))
    upper_z = lv["ffl"] + lv["storey"]          # terrace of a one-storey house, else the first-floor slab
    entries = [tuple(d["mid"]) for d in doors] + [((a["a"][0] + a["b"][0]) / 2, (a["a"][1] + a["b"][1]) / 2) for a in arches]
    stairs = []
    for sc in getattr(plan, "staircases", None) or []:
        external = "External" in sc.name
        base = 0.0 if external else lv["ffl"]
        g = _stair(sc, base, lv["storey"], external, top=None if (external and lv["stilt"]) else upper_z,
                   entries=entries, outer=outer)
        xs = [b[0] for b in g["boxes"]] + [b[2] for b in g["boxes"]]
        ys = [b[1] for b in g["boxes"]] + [b[3] for b in g["boxes"]]
        g.update({"name": sc.name, "r": _rect(sc.x, sc.y, sc.x + sc.w, sc.y + sc.h), "external": external,
                  "steps": int(g.pop("n")), "riser_mm": round(g.pop("riser") * 1000), "tread_mm": round(g.pop("tread") * 1000),
                  "low": _r(base), "mid": _r((base + g["top"]) / 2),
                  "wr": [_r(min(xs)), _r(min(ys)), _r(max(xs)), _r(max(ys))]})
        stairs.append(g)
    lifts = [_rect(r.x, r.y, r.right, r.top) for r in building if r.category == "LIFT"]
    internal_stairs = [s["r"] for s in stairs if not s["external"]]

    # slab, parapet (open where an outside stair arrives), plinth, mumty
    roof_z = lv["ffl"] + lv["storey"] * n_floors
    bridges = [s["bridge"] for s in stairs if s.get("bridge")] if n_floors == 1 else []
    parapet = [pc + [_r(roof_z), _r(roof_z + PARAPET_H)] for b in _ring(outer, 0.15) for pc in _minus(b, bridges)]
    shell = {"outer": [_r(v) for v in outer], "slab": _minus(outer, internal_stairs),
             "slab_z": [_r(lv["ffl"] + lv["ceiling"]), _r(lv["ffl"] + lv["storey"])],
             "roof_z": _r(roof_z), "floors": n_floors, "parapet": parapet,
             "mumty": internal_stairs[0] if internal_stairs else None}
    inner = next((s for s in stairs if not s["external"]), None)
    if inner is not None:
        m, tw = inner["r"], 0.115
        walls_m = _ring([m[0] - tw / 2, m[1] - tw / 2, m[2] + tw / 2, m[3] + tw / 2], tw)
        cut = [inner["exit"]] if inner.get("exit") else []
        shell["mumty_walls"] = [pc + [_r(roof_z), _r(roof_z + 2.4)] for w in walls_m for pc in _minus(w, cut)]
        shell["mumty_roof"] = [_r(m[0] - 0.2), _r(m[1] - 0.2), _r(m[2] + 0.2), _r(m[3] + 0.2), _r(roof_z + 2.4), _r(roof_z + 2.52)]

    # floors and ceilings by room
    floors = [{"r": _rect(r.x, r.y, r.right, r.top), "cat": r.category, "name": r.name} for r in building]

    # site
    site, cars, bikes, trees = [], [], [], []
    for r in rooms_all:
        if not _is_site(r) or r.w <= 0 or r.h <= 0:
            continue
        site.append({"cat": r.category, "name": r.name, "r": _rect(r.x, r.y, r.right, r.top)})
        if r.category == "PARKING":
            cars += _cars(r)
        elif r.category == "BIKE_PARKING":
            bikes += _bikes(r)
        elif r.category == "GARDEN":
            trees += _trees(r)
    stilt = None
    if lv["stilt"]:
        cols = [list(map(_r, c)) for c in (df.grid.columns if df.grid else [])]
        slots = int((getattr(plan, "additional_components", {}) or {}).get("car_parking", {}).get("slots", 1) or 1)
        bay = (fx + 0.4, fy + fd - min(fd - 0.8, 5.2), fx + fw - 0.4, fy + fd - 0.4)
        stilt_cars = []
        width = (bay[2] - bay[0]) / max(1, slots)
        for i in range(slots):
            stilt_cars.append({"c": [_r(bay[0] + (i + 0.5) * width), _r((bay[1] + bay[3]) / 2)], "along": "y",
                               "L": _r(min(4.5, bay[3] - bay[1] - 0.3)), "W": _r(min(1.8, width * 0.7)), "i": i,
                               "stilt": True})
        cars += stilt_cars
        stilt = {"h": _r(lv["ffl"] - 0.15), "columns": cols}

    gates = [[_r(a), _r(b)] for a, b in _gates(plan)]
    main = next((d for d in plan.doors if d.is_main_entrance), None)
    steps = []
    landing_foot = None
    if main is not None and not lv["stilt"]:
        a, b = door_span(main)
        ox, oy = OUT[main.wall_side]
        line = main.y if main.wall_side in ("N", "S") else main.x
        horiz = main.wall_side in ("N", "S")
        lo, hi = a - 0.3, b + 0.3
        rises = [lv["ffl"]] + [lv["ffl"] * (1 - k / 4) for k in (1, 2, 3)]
        depth = [(0.0, 1.2)] + [(1.2 + (k - 1) * 0.3, 1.2 + k * 0.3) for k in (1, 2, 3)]
        for z, (d0, d1) in zip(rises, depth):
            s0, s1 = (line + d0, line + d1) if (ox + oy) > 0 else (line - d1, line - d0)
            steps.append((_rect(lo, s0, hi, s1) if horiz else _rect(s0, lo, s1, hi)) + [0.0, _r(z)])
        far = 1.2 + 3 * 0.3 + 0.6
        mid = (a + b) / 2
        landing_foot = (mid, line + oy * far) if horiz else (line + ox * far, mid)

    # walking grid
    grid = _Grid(-1.0, -1.0, W + 1.0, H + VERGE + 4.0)
    for kind in ("ext", "int", "post", "cmp", "pillar"):
        for bx in walls[kind]:
            if bx[4] < 1.2 + (lv["ffl"] if kind in ("ext", "int", "post") else 0):   # body-height walls only
                grid.block(bx[0], bx[1], bx[2], bx[3], BODY_R)
    for it in df.items:
        fr = _furniture_rect(it)
        if fr:
            grid.block(fr[0], fr[1], fr[2], fr[3], 0.1 if fr[4] else 0.05)   # near plane 0.05 m: no clipping
    for x0, y0, x1, y1 in lifts:
        grid.block(x0, y0, x1, y1, 0.05)
    for r in building:
        if "Duct" in r.name:
            grid.block(r.x, r.y, r.right, r.top, 0.05)
    for s in site:
        x0, y0, x1, y1 = s["r"]
        if s["cat"] == "POOL":
            grid.block(x0, y0, x1, y1, 0.15)
        elif s["cat"] in ("GENERATOR", "SECURITY"):
            grid.block(x0, y0, x1, y1, BODY_R)
    for c in cars:
        if c.get("stilt"):          # parked on the floor below the house
            continue
        hl, hw = (c["L"] / 2, c["W"] / 2) if c["along"] == "x" else (c["W"] / 2, c["L"] / 2)
        grid.block(c["c"][0] - hl, c["c"][1] - hw, c["c"][0] + hl, c["c"][1] + hw, 0.15)
    for bk in bikes:
        hl, hw = (bk["L"] / 2, 0.35) if bk["along"] == "x" else (0.35, bk["L"] / 2)
        grid.block(bk["c"][0] - hl, bk["c"][1] - hw, bk["c"][0] + hl, bk["c"][1] + hw, 0.05)
    for tx, ty, _ in trees:
        grid.block(tx - 0.15, ty - 0.15, tx + 0.15, ty + 0.15, 0.1)
    # stay on the plot (the road strip in front stays walkable)
    grid.block(-5, -5, 0.0, H + 50, BODY_R)
    grid.block(W, -5, W + 5, H + 50, BODY_R)
    grid.block(-5, -5, W + 5, 0.0, BODY_R)
    if lv["stilt"]:
        grid.keep_only(fx, fy, fx + fw, fy + fd)

    # walking grids for the browser: the house / ground level with climbable stairs, and the level above
    walk0 = grid.like()
    for s in stairs:
        for rc in s["barrier"]:
            walk0.set(rc, False, 0.12, clip=s["r"])
        for rc in s["block0"]:
            walk0.set(rc, False, clip=s["r"])
    walk1 = grid.like(free=False)
    keep = (0.15 if n_floors == 1 else 0.23) + BODY_R                 # parapet / upper-floor outline
    walk1.set([outer[0] + keep, outer[1] + keep, outer[2] - keep, outer[3] - keep], True)
    if n_floors == 1:
        for b in shell.get("mumty_walls", []):
            walk1.set(b, False, BODY_R)
    for s in stairs:
        sl = walk1.span(s["r"], 0.01)            # +1 cm: cells centred exactly on an edge belong to both sides
        if sl:
            walk1.free[sl] = walk0.free[sl]
        for rc in s["block1"]:
            walk1.set(rc, False, clip=s["r"])
        for rc in s["free1"]:
            walk1.set(rc, True, 0.01)
        for rc in s["barrier"]:
            walk1.set(rc, False, 0.12, clip=s["r"])
        for rc in s["ring"]:
            walk1.set(rc, False, 0.12)
    # the tour stays on the ground floor: stairs are only looked at
    for s in stairs:
        x0, y0, x1, y1 = s["r"]
        grid.block(x0, y0, x1, y1, 0.05)
    for s in stairs:
        for k in ("block0", "block1", "free1", "barrier", "ring"):
            s.pop(k, None)

    # ground heights the walker stands on (max of the rectangles under the feet)
    ground = [[_r(fx - 0.12), _r(fy - 0.12), _r(fx + fw + 0.12), _r(fy + fd + 0.12), lv["ffl"]]]
    ground += [s[:4] + [s[5]] for s in steps]

    tour = _tour(plan, grid, lv, rooms, building, doors, gates, landing_foot, stairs, footprint, compass, main)

    # Vastu mandala over the building (3×3) for the overlay
    eng = VastuEngine.for_plan(plan, scope="building")
    mandala = []
    for i in range(3):
        for j in range(3):
            x0, x1 = fx + fw * i / 3, fx + fw * (i + 1) / 3
            y0, y1 = fy + fd * j / 3, fy + fd * (j + 1) / 3
            z = eng.get_zone_for_position((x0 + x1) / 2, (y0 + y1) / 2).value
            mandala.append({"r": _rect(x0, y0, x1, y1), "zone": z, "word": ZONE_WORDS.get(z, z)})

    a = math.radians(north_angle(plan))
    if latitude is None:
        latitude = STATE_LAT.get(state or "", DEFAULT_LAT)
    area = W * H
    return {
        "v": 1,
        "meta": {"title": f"{plan.bhk_config} · {W:.1f} × {H:.1f} m plot · {getattr(plan, 'orientation', 'N')}-facing",
                 "bhk": plan.bhk_config, "floors": shell["floors"], "quality": getattr(plan, "quality_level", ""),
                 "cents": _r(area / M2_PER_CENT, 2), "carpet": _r(getattr(plan, "total_carpet_area", 0), 1),
                 "vastu": _r(getattr(plan, "vastu_score", 0), 1), "facing": getattr(plan, "orientation", "N")},
        "plot": {"w": _r(W), "h": _r(H)}, "levels": lv,
        "footprint": [_r(fx), _r(fy), _r(fw), _r(fd)],
        "compass": compass, "north": [_r(math.sin(a), 4), _r(math.cos(a), 4)], "lat": _r(latitude, 2),
        "rooms": rooms, "floors": floors, "walls": walls, "windows": wins, "doors": doors, "arches": arches,
        "stairs": stairs, "lifts": lifts, "shell": shell, "furniture": [_jsonable(it) for it in df.items],
        "site": site, "cars": cars, "bikes": bikes, "trees": trees, "gates": gates, "steps": steps,
        "stilt": stilt, "ground": ground, "mandala": mandala,
        "brahmasthan": [_r(v) for v in df.brahmasthan] if df.brahmasthan else None,
        "road": {"y0": _r(H + VERGE), "y1": _r(H + VERGE + ROAD_W)},
        "nav": {"x0": walk0.x0, "y0": walk0.y0, "nx": walk0.nx, "ny": walk0.ny, "cell": CELL, "r": BODY_R,
                "bits": walk0.encode()},
        "nav1": {"x0": walk1.x0, "y0": walk1.y0, "nx": walk1.nx, "ny": walk1.ny, "cell": CELL, "r": BODY_R,
                 "bits": walk1.encode()},
        "upper": {"z": _r(upper_z), "kind": "terrace" if n_floors == 1 else "floor"},
        "tour": tour,
    }


def _jsonable(it: Dict) -> Dict:
    out = {}
    for k, v in it.items():
        if isinstance(v, float):
            out[k] = _r(v, 4)
        elif isinstance(v, tuple):
            out[k] = [_r(x, 4) for x in v]
        else:
            out[k] = v
    return out


# =============================================================================
# Guided tour
# =============================================================================

def _tour(plan, grid: _Grid, lv, rooms, building, doors, gates, landing_foot, stairs, footprint, compass, main) -> List[Dict]:
    W, H = plan.plot_width, plan.plot_height
    fx, fy, fw, fd = footprint
    clear = grid.clearance()
    rect_of = {r.name: r for r in building}
    info = {rm["name"]: rm for rm in rooms}

    def pt(c):
        x, y = grid.centre(*c)
        return [_r(x, 2), _r(y, 2)]

    # start: on the road in front of the gate (else at the main door)
    start = None
    facing_word = ZONE_WORDS.get(getattr(plan, "orientation", "N"), "")
    if gates and not lv["stilt"]:
        g0, g1 = max(gates, key=lambda g: g[1] - g[0])
        start = grid.nearest_free((g0 + g1) / 2, H + VERGE + 2.8, 1.0)
    if start is None and landing_foot and not lv["stilt"]:
        start = grid.nearest_free(*landing_foot, 1.5)
    if start is None and main is not None:
        m = next((d for d in doors if d["kind"] == "main"), None)
        if m:
            nx, ny = m["n"]
            start = grid.nearest_free(m["mid"][0] + nx * 0.8, m["mid"][1] + ny * 0.8, 1.5)
    if start is None:
        return []
    reach = grid.bfs(start)

    # a viewpoint in every room: roomy free spot near the centre, else peek in from its door
    stops = []
    for r in building:
        if r.category in NOT_STOPS or "Duct" in r.name or r.w * r.h < 1.2:
            continue
        best = None
        i0, j0 = grid.cell(r.x + 0.15, r.y + 0.15)
        i1, j1 = grid.cell(r.right - 0.15, r.top - 0.15)
        for j in range(max(0, j0), min(grid.ny, j1 + 1)):
            for i in range(max(0, i0), min(grid.nx, i1 + 1)):
                if (i, j) not in reach:
                    continue
                x, y = grid.centre(i, j)
                s = min(int(clear[j, i]), 6) * 0.1 - 0.35 * math.hypot(x - r.cx, y - r.cy)
                if best is None or s > best[0]:
                    best = (s, (i, j))
        inside = best is not None
        if not inside:
            for d in doors:
                if r.name not in (d["room"], d["other"]):
                    continue
                sgn = 1 if d["room"] == r.name else -1
                nx, ny = d["n"]
                px, py = d["mid"][0] - sgn * nx * (d["t"] / 2 + 0.45), d["mid"][1] - sgn * ny * (d["t"] / 2 + 0.45)
                c = grid.nearest_free(px, py, 0.6)
                if c and c in reach:
                    best = (0, c)
                    break
        if best is None:
            continue
        x, y = grid.centre(*best[1])
        if inside and math.hypot(x - r.cx, y - r.cy) < 1.0:
            corners = [(r.x + 0.2, r.y + 0.2), (r.right - 0.2, r.y + 0.2), (r.x + 0.2, r.top - 0.2), (r.right - 0.2, r.top - 0.2)]
            look = max(corners, key=lambda p: math.hypot(p[0] - x, p[1] - y))
        else:
            look = (r.cx, r.cy)
        stops.append({"room": r.name, "cell": best[1], "look": [_r(look[0], 2), _r(look[1], 2)],
                      "pan": 300 if r.w * r.h >= 9 and inside else 100})

    tour: List[Dict] = []
    cents = W * H / M2_PER_CENT
    car = any(r.category == "PARKING" for r in plan.rooms)
    intro = (f"Welcome to your new home. You are standing on the road in front of your {cents:.1f} cent plot, "
             f"{W:.1f} by {H:.1f} metres. The house faces {facing_word}.")
    if car:
        intro += " There is parking for your car inside the gate."
    if lv["stilt"]:
        intro = (f"Welcome to your new home on a {cents:.1f} cent plot. The car parks on the stilt floor below; "
                 "the house starts one floor up. You are standing inside, at the main door.")
    target_main = None
    m = next((d for d in doors if d["kind"] == "main"), None)
    if m is not None:
        mx, my = m["mid"]
        look_main = [_r(mx, 2), _r(my, 2)]
    else:
        look_main = [_r(fx + fw / 2, 2), _r(fy + fd / 2, 2)]
    tour.append({"kind": "start", "title": "Your plot", "say": intro, "path": [pt(start)], "look": look_main, "pan": 0})

    cur = start
    if landing_foot and not lv["stilt"]:
        target_main = grid.nearest_free(*landing_foot, 1.2)
    if target_main and target_main in reach and target_main != cur:
        par = grid.bfs(cur, {target_main})
        if target_main in par:
            bd = (getattr(plan, "vastu_breakdown", None) or {})
            _, vtxt = _verdict(bd.get("entrance_score"))
            say = f"This is the main door, on the {facing_word} side of the house"
            if bd.get("entrance_pada"):
                say += f", in pada {bd['entrance_pada']} — {vtxt}" if vtxt else f", in pada {bd['entrance_pada']}"
            say += f". Three steps take you up {int(lv['plinth'] * 100)} centimetres to the floor level."
            tour.append({"kind": "door", "title": "Main door", "say": say,
                         "path": [pt(c) for c in grid.smooth(grid.path(par, target_main))], "look": look_main, "pan": 0})
            cur = target_main

    left = {s["cell"]: s for s in stops}
    while left:
        par = grid.bfs(cur, set(left))
        got = [c for c in left if c in par]
        if not got:
            break
        # nearest next (by walking distance ≈ path length in cells)
        nxt = min(got, key=lambda c: len(grid.path(par, c)))
        s = left.pop(nxt)
        rm = info[s["room"]]
        tour.append({"kind": "room", "title": rm["name"], "room": rm["name"], "say": rm["say"],
                     "path": [pt(c) for c in grid.smooth(grid.path(par, nxt))], "look": s["look"], "pan": s["pan"]})
        cur = nxt

    inner = [s for s in stairs if not s["external"]]
    if inner:
        st = inner[0]
        x0, y0, x1, y1 = st["r"]
        cands = []
        for d in doors:
            if d["room"] in rect_of and rect_of[d["room"]].category == "STAIRS" or \
                    d["other"] in rect_of and rect_of[d["other"]].category == "STAIRS":
                cands.append(d["mid"])
        ring = cands or [((x0 + x1) / 2, y0 - 0.6), ((x0 + x1) / 2, y1 + 0.6), (x0 - 0.6, (y0 + y1) / 2), (x1 + 0.6, (y0 + y1) / 2)]
        par = grid.bfs(cur)
        best = None
        for px, py in ring:
            c = grid.nearest_free(px, py, 1.2)
            if c and c in par:
                n = len(grid.path(par, c))
                if best is None or n < best[0]:
                    best = (n, c)
        if best:
            up = "the first floor" if shell_floors(plan) > 1 else "the terrace"
            say = (f"The staircase goes up to {up}: {st['steps']} steps of {st['riser_mm']} millimetres "
                   f"with {st['tread_mm']} millimetre treads — easy on the knees "
                   f"(the building code allows risers up to 190).")
            tour.append({"kind": "stair", "title": "Staircase", "say": say,
                         "path": [pt(c) for c in grid.smooth(grid.path(par, best[1]))],
                         "look": [_r((x0 + x1) / 2, 2), _r((y0 + y1) / 2, 2)], "pan": 0})
    tour.append({"kind": "end", "title": "That's your house",
                 "say": "That was the whole house. Now see it from above — every room at once.",
                 "path": [], "look": None, "pan": 0})
    return tour


def shell_floors(plan) -> int:
    return max(1, int(getattr(plan, "num_floors", 1) or 1))
