"""
Gate engineering (2026-10-15)
=============================
Every gate in the compound wall is sized for what has to pass through it, and gets the kind of leaf
that can really open on this plot.

  what passes              clear width                          why
  -----------------------  -----------------------------------  ------------------------------------------
  people only              1.05 m (3½ ft) house,                a wheelchair needs 0.9 m clear (NBC 2016
                           1.2 m (4 ft) block of homes          Part 3 accessibility); many homes → prams,
                                                                two people passing
  two-wheelers, no car     1.5 m (5 ft); 2.45 m (8 ft) when     a scooter is ~0.75 m across the mirrors;
                           6 or more share it                   two passing need 2.4 m
  cars                     the design car's own swept path      simulated (below); a narrow road needs a
                           from the road + 0.3 m each side,     wider gate
                           at least 3.05 m (10 ft)
  6 or more cars           ≥ 5.2 m (17 ft)                      a car going out can pass one coming in
  building 15 m or taller  ≥ 4.5 m                              NBC 2016 Part 4: fire-tender access

Widths are rounded up to whole feet (gates are made and sold by the foot in India).

Swept path. The design car (from the quality / building type: hatchback, compact SUV or 7-seat MUV,
with its brochure length, width, wheelbase and minimum turning radius) drives along the road, turns
in on a circle (rear-axle "bicycle" model: the rear axle follows the circle, the body sweeps around
it) and runs straight into its bay. Every start lane on the road and every turning radius from the
tightest up is tried; the gate must cover the body wherever it crosses the wall line, for a car
coming from either end of the road. The yard behind the gate (the throat) may be used to finish the
turn; deeper than that the car must already be in its own lane. If no single sweep works (a lane
too narrow) the yard is used more freely, and failing that the owner is told the car needs a
2-point turn and the gate is as wide as the frontage allows.

Leaves never swing over the public road. Up to 1.25 m one leaf, up to 4.25 m two leaves — when their
swing arcs miss the parked car, the porch and the house; otherwise a sliding gate if the wall beside
it can take the leaf (run-back), else a telescopic one (half the run-back), else a bi-fold.

Stilt parking. Cars under the building park only where a car can drive straight in from the front:
never in the stair, lift or duct, never on the entrance lobby, never on a column (the structural
grid of the floors above), in 2.5 × 5.0 m bays (NBC); a car that only fits behind another is a
tandem bay. The car gate is then sized for those bays.

Vastu: the pada of every gate on the plot boundary (32 padas); a gate with slack is slid to the best
one, and leaves open inward.
"""

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np

FT = 0.3048
CLEAR = 0.30        # each side of the swept body: mirrors + margin to the gate pillar
PIER = 0.45         # brick pier left between two gates
EDGE = 0.15         # a gate's opening may start at the corner pillar's face (its own pillar merges with it)
DRAIN = 0.50        # at each road edge: an open drain / kerb / a parked scooter
STOREY = 3.0
FIRE_H = 15.0       # NBC 2016 Part 4: 15 m and taller
FIRE_W = 4.5
TWO_WAY = 6         # cars sharing one gate before two must pass in it
SWING_MAX = 4.25    # wider swing leaves sag on their hinges
TURN_IN = 1.5       # the car may still be turning this far behind the gate …
SWING_OUT = 0.6     # … swinging this far beyond its own bay
WALK_W = {"house": (1.05, "3½"), "block": (1.2, "4"), "bike": (1.5, "5"), "bikes": (2.45, "8")}
STALL_W, STALL_D = 2.5, 5.0

CARS = {
    "hatchback": dict(label="hatchback (Swift / i20 class)", L=3.85, W=1.74, wb=2.45, fo=0.75, R=4.8),
    "suv": dict(label="compact SUV / sedan (Creta / City class)", L=4.40, W=1.79, wb=2.61, fo=0.88, R=5.3),
    "muv": dict(label="7-seat MUV (Innova / XUV700 class)", L=4.75, W=1.85, wb=2.75, fo=0.95, R=5.4),
}
ROADS = [(4.5, "15 ft lane"), (6.0, "20 ft road"), (7.5, "25 ft road"), (9.0, "30 ft road"),
         (12.0, "40 ft road"), (18.0, "60 ft road")]


def road_label(w: float) -> str:
    return f"{int(round(w / FT / 5.0) * 5)} ft road ({w:g} m)"


def _snap(v: float, g: float = 0.05) -> float:
    return round(round(v / g) * g, 3)


def feet_up(m: float) -> Tuple[float, int]:
    """Round a width up to whole feet → (metres on the 50 mm grid, feet)."""
    ft = int(math.ceil(m / FT - 1e-6))
    return _snap(ft * FT), ft


def ft_words(m: float) -> str:
    half = round(m / FT * 2) / 2
    return f"{int(half)}½ ft" if half % 1 else f"{int(half)} ft"


def design_car(quality: str = "Standard", building_type: str = "") -> str:
    """The biggest car the gate should take: many families → MUV; premium / luxury → MUV; economy → hatchback."""
    bt, q = (building_type or "").lower(), (quality or "").lower()
    if "apartment" in bt or "premium" in q or "luxury" in q:
        return "muv"
    if "economy" in q:
        return "hatchback"
    return "suv"


# ───────────────────────────── swept path (frame: wall line y = 0, road y > 0, plot y < 0, car axis x = 0)
def _rear_radius(car: Dict) -> float:
    """Brochure turning radius = path of the outer front wheel; the rear axle centre turns tighter."""
    track = car["W"] - 0.2
    return math.sqrt(max(car["R"] ** 2 - car["wb"] ** 2, 1.0)) - track / 2


def _body(car: Dict) -> List[Tuple[float, float]]:
    ro, h = car["L"] - car["wb"] - car["fo"], car["W"] / 2
    return [(-ro, -h), (car["wb"] + car["fo"], -h), (car["wb"] + car["fo"], h), (-ro, h)]


def _corners(body, x, y, th):
    c, s = math.cos(th), math.sin(th)
    return [(x + u * c - v * s, y + u * s + v * c) for u, v in body]


def _cross(pts, line: float) -> Optional[Tuple[float, float]]:
    xs = []
    n = len(pts)
    for i in range(n):
        (x1, y1), (x2, y2) = pts[i], pts[(i + 1) % n]
        if (y1 - line) * (y2 - line) <= 0 and abs(y2 - y1) > 1e-9:
            xs.append(x1 + (line - y1) / (y2 - y1) * (x2 - x1))
    return (min(xs), max(xs)) if xs else None


def _arc(y0: float, R: float, step_deg: float = 2.0):
    """Rear-axle poses of the turn: heading +x at (−R, y0), right turn on radius R, heading −y at (0, y0 − R)."""
    cx, cy = -R, y0 - R
    k, out = 0, []
    while k * step_deg <= 90.0 + 1e-9:
        phi = math.radians(k * step_deg)
        out.append((cx + R * math.sin(phi), cy + R * math.cos(phi), -phi))
        k += 1
    return out


def _cross_np(X, Y, line):
    """Per candidate: the x-range where any body edge of any pose meets y = line (±inf where none)."""
    X2, Y2 = np.roll(X, -1, axis=2), np.roll(Y, -1, axis=2)
    dy = Y2 - Y
    hit = ((Y - line) * (Y2 - line) <= 0) & (np.abs(dy) > 1e-9)
    t = np.where(hit, (line - Y) / np.where(hit, dy, 1.0), 0.0)
    xc = X + t * (X2 - X)
    return np.where(hit, xc, np.inf).min(axis=(1, 2)), np.where(hit, xc, -np.inf).max(axis=(1, 2))


@lru_cache(maxsize=2048)
def swept(car_key: str, road: float, throat: float, lane_lo: float, lane_hi: float,
          wall_lo: float = -99.0, wall_hi: float = 99.0, s_min: float = -99.0):
    """
    Narrowest wall crossing of the design car coming from the plan's left (driving +x, turning right
    into the plot) → (lo, hi, y0, R) relative to the car's final axis, or None if it cannot get in
    in one sweep. "Narrowest" = the smallest opening centred on the car's axis (max(hi, −lo)), so a
    wider road never asks for a wider gate. road = usable road width; throat = yard depth behind the
    wall where the car may still be turning; lane = x-range (relative) the body must keep deeper than
    the throat; wall_lo / wall_hi = how far the opening can reach (the plot corners — near a corner the
    car takes a lopsided sweep); s_min = the turn is finished (rear axle straight) before this depth,
    so the car is straight when it stops in its bay. Every start lane (0.25 m) × turning radius
    (0.4 m, from the tightest) is tried at once (numpy), each turn checked every 2°.
    """
    car = CARS[car_key]
    h = car["W"] / 2
    if lane_lo > -h - 0.02 or lane_hi < h + 0.02 or -h < wall_lo or h > wall_hi:
        return None
    Rr = _rear_radius(car)
    y0s = np.arange(h + 0.15, road - h + 1e-9, 0.25)
    Rs = np.arange(Rr, Rr + 6.0 + 1e-9, 0.4)
    if not len(y0s):
        return None
    Y0, RR = (a.ravel() for a in np.meshgrid(y0s, Rs, indexing="ij"))
    keep = RR <= Y0 - s_min + 1e-9
    Y0, RR = Y0[keep], RR[keep]
    if not len(Y0):
        return None
    phi = np.radians(np.arange(0.0, 90.0 + 1e-9, 2.0))
    px = (-RR)[:, None] + RR[:, None] * np.sin(phi)[None, :]
    py = (Y0 - RR)[:, None] + RR[:, None] * np.cos(phi)[None, :]
    c, sn = np.cos(-phi)[None, :, None], np.sin(-phi)[None, :, None]
    body = np.array(_body(car))
    u, v = body[:, 0][None, None, :], body[:, 1][None, None, :]
    X = px[:, :, None] + u * c - v * sn
    Y = py[:, :, None] + u * sn + v * c
    ok = Y.max(axis=(1, 2)) <= road + 1e-6
    deep = Y < -throat
    ok &= ~(deep & ((X < lane_lo - 1e-6) | (X > lane_hi + 1e-6))).any(axis=(1, 2))
    tlo, thi = _cross_np(X, Y, -throat)
    ok &= ~(np.isfinite(tlo) & ((tlo < lane_lo - 1e-6) | (thi > lane_hi + 1e-6)))
    lo, hi = _cross_np(X, Y, 0.0)
    lo, hi = np.minimum(lo, -h), np.maximum(hi, h)        # the straight run-in crosses at the car's own width
    ok &= (lo >= wall_lo - 1e-9) & (hi <= wall_hi + 1e-9)
    if not ok.any():
        return None
    score = np.where(ok, np.maximum(hi, -lo), np.inf)
    i = int(np.argmin(score))
    return round(float(lo[i]), 3), round(float(hi[i]), 3), round(float(Y0[i]), 3), round(float(RR[i]), 3)


def _need(car_key, road, throat, lane_lo, lane_hi, wall_lo=-99.0, wall_hi=99.0, s_min=-99.0):
    """Both approach directions → {'L': (lo, hi, y0, R) | None, 'R': (lo, hi, y0, R) | None} (relative)."""
    r = lambda v: round(v, 1)                                          # noqa: E731 — cache-friendly
    wl, wh = math.ceil(wall_lo * 20) / 20, math.floor(wall_hi * 20) / 20
    sm = math.ceil(s_min * 10) / 10
    left = swept(car_key, r(road), r(throat), r(lane_lo), r(lane_hi), wl, wh, sm)
    m = swept(car_key, r(road), r(throat), r(-lane_hi), r(-lane_lo), -wh, -wl, sm)
    right = (-m[1], -m[0], m[2], m[3]) if m else None                  # mirror x → −x
    return {"L": left, "R": right}


def drive_path(car_key: str, axis: float, H: float, sweep, side: str, park_y: float,
               start_back: float = 9.0, step: float = 0.25) -> List[List[float]]:
    """Centre poses [x, y, heading] of the design car from the road, through the gate, to its bay (plan frame)."""
    car = CARS[car_key]
    _, _, y0, R = sweep
    s = 1.0 if side == "L" else -1.0
    uc = (car["wb"] + car["fo"] - (car["L"] - car["wb"] - car["fo"])) / 2       # rear axle → body centre
    poses = []
    x = -R - start_back
    while x < -R - 1e-9:
        poses.append((x, y0, 0.0))
        x += step
    poses += _arc(y0, R, 3.0)
    y = y0 - R - step
    end = park_y - H + uc                                    # rear axle when the body centre is at park_y (heading -y)
    while y > end:
        poses.append((0.0, y, -math.pi / 2))
        y -= step
    if end < y0 - R:
        poses.append((0.0, end, -math.pi / 2))
    out = []
    for px, py, th in poses:
        X, Y = axis + s * px, H + py
        hd = th if s > 0 else math.pi - th
        out.append([round(X + uc * math.cos(hd), 3), round(Y + uc * math.sin(hd), 3), round(hd, 4)])
    return out


# ───────────────────────────── stilt bays
def stilt_bays(footprint, shafts, no_park, columns, n: int, car_key: str = "suv") -> Dict:
    """
    Parking bays under the building: 2.5 × 5.0 m (NBC), cars nose-in from the front edge.
    shafts (stair / lift / duct) and columns block both bays and the drive; no_park (the entrance lobby)
    may be driven across but not parked on. Returns {'bays': [[x0, y0, x1, y1, tandem, gap0, gap1]] (gap = the
    opening between columns the car passes at the front edge), 'mouth': [x0, x1], 'wanted': n, 'fitted': k}.
    """
    fx, fy, fw, fd = footprint
    front = fy + fd
    car = CARS[car_key]
    body_w, body_l = car["W"] + 0.2, car["L"] + 0.2

    def hit(r, rects, pad=0.0):
        return any(r[0] < q[2] - 1e-6 + pad and q[0] < r[2] - 1e-6 + pad and r[1] < q[3] - 1e-6 + pad
                   and q[1] < r[3] - 1e-6 + pad for q in rects)

    def hits(r, rects):
        return [q for q in rects if r[0] < q[2] - 1e-6 and q[0] < r[2] - 1e-6 and r[1] < q[3] - 1e-6
                and q[1] < r[3] - 1e-6]

    def car_rect(x0, y_top):                 # body inside a stall whose road-side edge is y_top
        cx = x0 + STALL_W / 2
        return (cx - body_w / 2, y_top - 0.15 - body_l, cx + body_w / 2, y_top - 0.15)

    taken: List[Tuple] = []                  # stalls already given (x0, y0, x1, y1)
    drives: List[Tuple] = []                 # their straight drives from the front edge
    bays = []
    half = body_w / 2 + 0.1                  # half the lane a car needs between columns
    lo_x, hi_x = fx + 0.15, fx + fw - 0.15 - STALL_W

    def xs():
        """Stall positions: a coarse sweep + every position that just clears an obstacle edge."""
        out = set()
        x = lo_x
        while x <= hi_x + 1e-9:
            out.add(round(x, 3))
            x += 0.25
        for o in list(shafts) + list(columns) + list(no_park) + taken:
            for v in (o[2] + half - STALL_W / 2 + 0.01, o[0] - half - STALL_W / 2 - 0.01,
                      o[2] + 0.01, o[0] - STALL_W - 0.01):
                if lo_x - 1e-9 <= v <= hi_x + 1e-9:
                    out.add(round(v, 3))
        out.add(round(hi_x, 3))
        return sorted(out)

    for _ in range(max(0, n)):
        best = None
        for x in xs():
            cx = x + STALL_W / 2
            lane = (cx - half, cx + half)
            for tandem in (False, True):
                y = front
                ok = None
                while y - STALL_D >= fy - 1e-9:
                    stall = (x, y - STALL_D, x + STALL_W, y)
                    body = car_rect(x, y)
                    run = (lane[0], body[3], lane[1], front)
                    if hit(run, shafts) or hit(run, columns) or (not tandem and hit(run, taken)):
                        break
                    in_body = hits(body, list(shafts) + list(columns) + drives)
                    in_stall = hits(stall, list(no_park) + taken)
                    if not (in_body or in_stall):
                        ok = y
                        break
                    # jump straight past what is in the way (the stall's road edge below its far side)
                    y = round(min([o[1] + 0.15 for o in in_body] + [o[1] for o in in_stall] + [y - 0.02]), 3)
                if ok is not None:
                    key = (1 if tandem else 0, round(front - ok, 1), x)
                    if best is None or key < best[0]:
                        best = (key, x, ok, tandem)
                    break
        if best is None:
            break
        _, x, y, tandem = best
        stall = (round(x, 3), round(y - STALL_D, 3), round(x + STALL_W, 3), round(y, 3))
        taken.append(stall)
        cx = x + STALL_W / 2
        if not tandem:
            drives.append((cx - half, y, cx + half, front))
        # the gap the car passes through at the front edge (between columns / shafts)
        g0, g1 = fx - 0.115, fx + fw + 0.115
        for o in list(shafts) + list(columns):
            if o[1] < front + 0.3 and o[3] > front - 0.3:
                if o[2] <= cx:
                    g0 = max(g0, o[2])
                elif o[0] >= cx:
                    g1 = min(g1, o[0])
        bays.append([*stall, bool(tandem), round(g0, 3), round(g1, 3)])
    mouth = None
    open_bays = [b for b in bays if not b[4]] or bays
    if open_bays:
        mouth = [min(b[5] for b in open_bays), max(b[6] for b in open_bays)]
    return {"bays": bays, "mouth": mouth, "wanted": n, "fitted": len(bays), "front": front}


HALLS = ("Stair Lobby", "Entrance Foyer", "Upper Lobby", "Lift & Stair Lobby", "Common Gallery", "Common Corridor")


def front_gap(footprint, shafts, columns) -> float:
    """The widest clear opening at the stilt's front edge between columns and shafts (m)."""
    fx, fy, fw, fd = footprint
    front = fy + fd
    obs = sorted((o[0], o[2]) for o in list(shafts) + list(columns) if o[1] < front + 0.3 and o[3] > front - 0.3)
    best, pos = 0.0, fx - 0.115
    for a, b in obs:
        best = max(best, a - pos)
        pos = max(pos, b)
    return round(max(best, fx + fw + 0.115 - pos), 2)


def stilt_context(rooms, footprint, is_site) -> Tuple[List, List, List]:
    """(shafts, no-park lobbies, columns) of the floors above, for stilt_bays."""
    from floorplan_generator.structure import structural_grid
    building = [r for r in rooms if not is_site(r) and r.w > 0 and r.h > 0]
    shafts, lobby = [], []
    for r in building:
        rect = (r.x, r.y, r.x + r.w, r.y + r.h)
        if r.category in ("STAIRS", "LIFT") or "Duct" in r.name:
            shafts.append(rect)
        elif getattr(r, "role", "") == "COMMON" or r.name in HALLS:     # the entrance lobby / hall (not a flat's foyer)
            lobby.append(rect)
    cols = [tuple(c) for c in structural_grid(building, footprint).columns]
    return shafts, lobby, cols


def stilt_plan(rooms, footprint, components: Dict, quality: str, building_type: str, is_site) -> Dict:
    """Stilt bays for the cars asked for, under the floors above (see stilt_bays) + a note if fewer fit."""
    cfg = (components or {}).get("car_parking") or {}
    n = max(1, int(cfg.get("slots", 1) or 1))
    car = design_car(quality, building_type)
    shafts, lobby, cols = stilt_context(rooms, footprint, is_site)
    info = stilt_bays(footprint, shafts, lobby, cols, n, car)
    info["car"] = car
    tandem = sum(1 for b in info["bays"] if b[4])
    car_d = CARS[car]
    info["front_gap"] = front_gap(footprint, shafts, cols)
    info["needs"] = round(car_d["W"] + 0.4, 2)
    if info["fitted"] == 0:
        info["note"] = (f"No car fits under the building on this plan: the widest opening between the columns at the front "
                        f"is {info['front_gap']:.2f} m and a car needs about {info['needs']:.2f} m to drive in (a 2.5 m bay). "
                        "Two-wheelers can still park on the stilt floor.")
    elif info["fitted"] < n:
        info["note"] = (f"Only {info['fitted']} of the {n} cars fit under the building: a car must drive straight in "
                        "clear of the stair, the lift, the entrance lobby and the columns.")
    elif tandem:
        info["note"] = (f"{tandem} of the {n} stilt bays {'is' if tandem == 1 else 'are'} tandem (behind another car) — "
                        "good for one family's two cars.")
    return info


# ───────────────────────────── gates
@dataclass
class Gate:
    kind: str                                    # 'walk' | 'two_wheeler' | 'vehicle' | 'vehicle_wicket'
    x0: float
    x1: float
    height: float = 1.5
    op: str = "swing"                            # swing | sliding | telescopic | bifold
    leaves: List[Dict] = field(default_factory=list)
    runback: Optional[List[float]] = None        # wall the sliding leaf runs behind [x0, x1]
    wicket: Optional[List[float]] = None         # the people's door inside a merged gate
    feet: str = ""
    why: List[str] = field(default_factory=list)
    pada: Optional[Dict] = None
    tag: str = ""
    car: Optional[str] = None
    approach: str = ""
    vehicle: Optional[List[float]] = None        # the car's part of a merged gate [x0, x1]

    @property
    def width(self) -> float:
        return round(self.x1 - self.x0, 3)

    def as_dict(self) -> Dict:
        return {"kind": self.kind, "x0": self.x0, "x1": self.x1, "width": self.width, "height": self.height,
                "op": self.op, "leaves": self.leaves, "runback": self.runback, "wicket": self.wicket,
                "feet": self.feet, "why": list(self.why), "pada": self.pada, "tag": self.tag,
                "car": self.car, "approach": self.approach, "vehicle": self.vehicle}


def _overlap(a, b) -> bool:
    """a, b as (x0, y0, x1, y1)."""
    return a[0] < b[2] - 1e-6 and b[0] < a[2] - 1e-6 and a[1] < b[3] - 1e-6 and b[1] < a[3] - 1e-6


class GateDesigner:
    """Sizes and places the gates of one plot (plan frame: road along the top edge y = H)."""

    def __init__(self, W: float, H: float, facing: str = "N", vastu: bool = True, road: float = 9.0,
                 quality: str = "Standard", building_type: str = "", floors: int = 1, stilt: bool = False,
                 cars: int = 0, bikes: int = 0):
        self.W, self.H = W, H
        self.facing = (facing or "N").upper()[0]
        self.vastu = vastu
        self.road = float(road or 9.0)
        self.usable = max(2.6, self.road - 2 * DRAIN)
        self.block = "apartment" in (building_type or "").lower()
        self.floors = max(1, int(floors or 1))
        self.stilt = stilt
        self.cars = max(0, int(cars or 0))
        self.bikes = max(0, int(bikes or 0))
        self.car_key = design_car(quality, building_type)
        self.height_m = round((0.15 + STOREY if stilt else 0.45) + self.floors * STOREY, 2)
        self.notes: List[str] = []

    # -- Vastu pada of a gate centred at x on the road edge (plot mandala)
    def pada(self, x: float) -> Optional[Dict]:
        try:
            from floorplan_generator.vastu_engine import pada_for_point
            p = pada_for_point(x, self.H, (0.0, 0.0, self.W, self.H), self.facing)
        except Exception:
            return None
        return {"id": p.pada_id, "deity": p.deity, "rating": p.rating, "score": p.score,
                "effect": p.effect, "remedy": p.remedy}

    def _pscore(self, x0: float, w: float) -> float:
        if not self.vastu:
            return 0.0
        p = self.pada(x0 + w / 2)
        return float(p["score"]) if p else 0.0

    # -- the people's gate (and the two-wheelers' when no car comes in)
    def walk_gate(self, door_x0: float, door_x1: float, vehicle: bool) -> Gate:
        if vehicle or not self.bikes:
            key = "block" if self.block else "house"
            kind = "walk"
            why = ("people only — a wheelchair needs 0.9 m clear" if not self.block else
                   "people of many homes — two can pass, with a pram or a wheelchair")
            if vehicle and self.bikes:
                why += "; two-wheelers use the car gate"
        else:
            key = "bikes" if self.bikes >= TWO_WAY else "bike"
            kind = "two_wheeler"
            why = (f"{self.bikes} two-wheelers and people — a scooter is 0.75 m across the mirrors"
                   + (", two can pass in the gate" if key == "bikes" else ", ridden in with room each side"))
        w, ft = WALK_W[key]
        mid = (door_x0 + door_x1) / 2
        slack = max(0.0, (door_x1 - door_x0 + 0.9 - w) / 2 - 0.05)
        best = None
        k = -slack
        while k <= slack + 1e-9:
            x0 = _snap(min(max(mid - w / 2 + k, EDGE), self.W - EDGE - w))
            sc = self._pscore(x0, w) - 0.5 * abs(k)
            if best is None or sc > best[0] + 1e-9:
                best = (sc, x0)
            k += 0.05
        x0 = best[1]
        g = Gate(kind=kind, x0=x0, x1=_snap(x0 + w), height=1.5, feet=f"{ft} ft",
                 why=[f"{w:.2f} m ({ft} ft): {why}."])
        self._swing(g, [], force=True)
        g.pada = self.pada((g.x0 + g.x1) / 2) if self.vastu else None
        return g

    # -- the car gate
    def vehicle_gate(self, lanes: List[Dict], walk: Gate, blocked: List[Tuple], parked: List[Tuple],
                     house: Tuple, mode: str) -> Tuple[Optional[Gate], Optional[Dict], List[Tuple]]:
        """
        lanes: [{'axis', 'throat', 'lo', 'hi', 'park_y', 'stop_y', 'exact'?}] in absolute plan x (lo/hi = the
        lane the car must keep deeper than the throat; 'exact' = no turning beyond it, e.g. between stilt
        columns). mode: 'front' | 'parallel' | 'side' | 'stilt' | 'fire'. blocked / parked / house as
        (x0, y0, x1, y1). Returns (gate, drive, reservations as (x, y, w, h)).
        """
        car = CARS[self.car_key]
        sweeps, how = [], "sweep"
        for ln in (lanes if mode in ("front", "side", "stilt") else []):
            exact = bool(ln.get("exact"))
            t_add, s_add = (0.0, 0.0) if exact else (TURN_IN, SWING_OUT)
            wall = (EDGE + CLEAR - ln["axis"], self.W - EDGE - CLEAR - ln["axis"])     # the plot corners
            uc = (car["wb"] + car["fo"] - (car["L"] - car["wb"] - car["fo"])) / 2
            s_min = ln["park_y"] - self.H + uc            # straight by the time it stops in its bay
            res = _need(self.car_key, self.usable, ln["throat"] + t_add, ln["lo"] - ln["axis"] - s_add,
                        ln["hi"] - ln["axis"] + s_add, *wall, s_min)
            if not (res["L"] or res["R"]) and not exact:      # a narrow road: finish the turn inside the yard
                res = _need(self.car_key, self.road - 0.2, ln["throat"] + 2.5,
                            ln["lo"] - ln["axis"] - 1.0, ln["hi"] - ln["axis"] + 1.0, *wall, s_min)
                if res["L"] or res["R"]:
                    how = "yard"
            ivs = {k: (ln["axis"] + v[0] - CLEAR, ln["axis"] + v[1] + CLEAR) for k, v in res.items() if v}
            if not ivs:
                how = "manoeuvre"
            sweeps.append((ln, res, ivs))

        def hull(dirs):
            lo = hi = None
            for ln, _, ivs in sweeps:
                got = [ivs[d] for d in dirs if d in ivs]
                if ivs and not got:
                    return None                       # this bay cannot be reached from that end
                if not got:                           # no sweep at all: straight in after a reverse
                    got = [(ln["axis"] - car["W"] / 2 - CLEAR, ln["axis"] + car["W"] / 2 + CLEAR)]
                for a, b in got:
                    lo, hi = (a, b) if lo is None else (min(lo, a), max(hi, b))
            return (lo, hi) if lo is not None else None

        def fits(iv):
            return iv is not None and iv[0] >= EDGE - 1e-6 and iv[1] <= self.W - EDGE + 1e-6

        approach = "LR"
        need = None
        if sweeps:
            for dirs in ("LR", "L", "R"):
                iv = hull(dirs)
                if fits(iv):
                    need, approach = iv, dirs
                    break
            if need is None:
                need = hull("LR") or hull("L") or hull("R")
                need = (max(need[0], EDGE), min(need[1], self.W - EDGE))
                how = "manoeuvre"
        elif mode == "parallel" and lanes:
            need = (lanes[0]["lo"] + 0.15, lanes[0]["hi"] - 0.15)
        elif mode == "fire":
            need = (self.W / 2 - 0.5, self.W / 2 + 0.5)
        if need is None:
            return None, None, []
        swept_w = need[1] - need[0]
        if mode == "parallel":
            wants = [(swept_w, "the car parks parallel behind the wall, so the gate spans its bay")]
        elif mode == "fire":
            wants = []
        else:
            wants = [(swept_w, f"the {car['label']} turning in from a {road_label(self.road)} sweeps "
                               f"{swept_w - 2 * CLEAR:.2f} m at the wall, + {CLEAR:.1f} m each side")]
        if mode != "fire":
            wants.append((10 * FT, "no car gate narrower than 10 ft"))
        if self.cars >= TWO_WAY:
            wants.append((2 * car["W"] + 3 * PIER, f"{self.cars} cars share it — one going out passes one coming in"))
        if self.height_m >= FIRE_H:
            wants.append((FIRE_W, f"the building is {self.height_m:.1f} m tall — a fire engine must get in "
                                   "(NBC 2016 Part 4: 4.5 m)"))
        if not wants:
            return None, None, []
        width, reason = max(wants, key=lambda t: t[0])
        w, ft = feet_up(width)
        w = min(w, _snap(self.W - 2 * EDGE))
        # where: covering the swept need, off the walking gate, on the best pada
        lo_x, hi_x = need
        if mode == "fire":
            lo_x, hi_x = self.W - EDGE, EDGE                 # anywhere along the frontage
        cands = []
        x0 = _snap(max(hi_x - w, EDGE))
        while x0 <= min(lo_x, self.W - EDGE - w) + 1e-9:
            cands.append(x0)
            x0 = round(x0 + 0.05, 3)
        if not cands:
            cands = [_snap(min(max((lo_x + hi_x - w) / 2, EDGE), self.W - EDGE - w))]
        stop_y = min((ln["stop_y"] for ln in lanes), default=self.H - 1.0)
        free = [c for c in cands if not any(_overlap((c, stop_y, c + w, self.H), b) for b in blocked)]
        cands = free or cands
        apart = [c for c in cands if c + w <= walk.x0 - PIER + 1e-9 or c >= walk.x1 + PIER - 1e-9]
        merged = not apart
        pool = apart or cands
        mid = (lo_x + hi_x) / 2 if mode != "fire" else self.W / 2
        x0 = max(pool, key=lambda c: self._pscore(c, w) - 0.3 * abs(c + w / 2 - mid))
        g = Gate(kind="vehicle", x0=x0, x1=_snap(x0 + w), height=1.8, feet=f"{ft} ft", car=self.car_key)
        g.why = [f"{w:.2f} m ({ft} ft): {reason}."]
        for wv, text in wants:
            if text != reason and not text.startswith("no car gate"):
                g.why.append(f"also: {text} ({wv:.2f} m).")
        if mode == "fire":
            g.why.append("No car parks inside, but the gate is kept wide for the fire engine and for deliveries.")
        if how == "yard":
            g.why.append("The road is narrow: the car finishes its turn inside the gate — keep the yard behind it free.")
        elif how == "manoeuvre":
            g.why.append(f"From a {road_label(self.road)} the car cannot turn straight in: it needs one reverse "
                         "(a 2-point turn) to line up with its bay.")
        if approach in ("L", "R"):
            from floorplan_generator.vastu_engine import plan_sides_for_facing
            end = plan_sides_for_facing(self.facing)["left" if approach == "L" else "right"]
            g.approach = end
            g.why.append(f"Turn in coming from the {_COMPASS.get(end, end)} end of the road — from the other end "
                         "the car would need the wall beyond the plot corner.")
        if merged:
            g.kind, g.wicket, g.vehicle = "vehicle_wicket", [walk.x0, walk.x1], [g.x0, g.x1]
            inside = walk.x0 >= g.x0 - 1e-6 and walk.x1 <= g.x1 + 1e-6
            g.x0, g.x1 = _snap(min(g.x0, walk.x0)), _snap(max(g.x1, walk.x1))
            g.feet = ft_words(g.x1 - g.x0)
            g.why.append(f"The frontage is too short for a separate walking gate: people use a "
                         f"{walk.width:.2f} m wicket door " + ("built into the car gate's leaf."
                                                               if inside else "beside the car gate."))
        # drive: the bay nearest the gate's centre, from the end of the road the car can come from
        drive = None
        gc = (g.x0 + g.x1) / 2
        if sweeps:
            ln, res, _ = min(sweeps, key=lambda t: abs(t[0]["axis"] - gc))
            side = next((d for d in approach if res.get(d)), None)
            if side:
                drive = {"car": self.car_key, "label": car["label"],
                         "dims": {k: car[k] for k in ("L", "W", "wb", "fo")},
                         "path": drive_path(self.car_key, ln["axis"], self.H, res[side], side, ln["park_y"]),
                         "park": [round(ln["axis"], 3), round(ln["park_y"], 3)]}
        reserve = [(g.x0, stop_y, g.width, self.H - stop_y)] if mode != "fire" else []
        reserve += self._operate(g, walk, parked, house, blocked)
        g.pada = self.pada((g.x0 + g.x1) / 2) if self.vastu else None
        return g, drive, reserve

    # -- leaves
    def _swing(self, g: Gate, obstacles, force=False, span=None) -> Optional[List[Tuple]]:
        """One or two inward swing leaves over [span]; returns the arcs' boxes, or None when they hit something."""
        a, b = span or (g.x0, g.x1)
        w = b - a
        leaves = [{"type": "swing", "hinge": a, "len": round(w, 3), "dir": 1}] if w <= 1.25 + 1e-9 else \
            [{"type": "swing", "hinge": a, "len": round(w / 2, 3), "dir": 1},
             {"type": "swing", "hinge": b, "len": round(w / 2, 3), "dir": -1}]
        boxes = []
        for lf in leaves:
            L = lf["len"]
            x0 = lf["hinge"] if lf["dir"] > 0 else lf["hinge"] - L
            boxes.append((x0, self.H - L - 0.05, x0 + L, self.H))
        if not force and any(_overlap(bx, o) for bx in boxes for o in obstacles):
            return None
        g.leaves = [lf for lf in g.leaves if lf.get("wicket")] + leaves
        g.op = "swing"
        return [(bx[0], bx[1], bx[2] - bx[0], bx[3] - bx[1]) for bx in boxes]

    def _operate(self, g: Gate, walk: Gate, parked, house, blocked) -> List[Tuple]:
        """Pick the leaf type the plot allows; returns the space it needs as (x, y, w, h)."""
        span = (g.x0, g.x1)
        others = [(walk.x0, walk.x1)]                # openings a sliding leaf may not run behind
        if g.wicket:
            v0, v1 = g.vehicle
            inside = g.wicket[0] >= v0 - 1e-6 and g.wicket[1] <= v1 + 1e-6
            span = (v0, v1) if inside else ((g.x0, g.wicket[0]) if g.wicket[0] > g.x0 + 0.5 else (g.wicket[1], g.x1))
            g.leaves = [{"type": "wicket" if inside else "swing", "hinge": g.wicket[0],
                         "len": round(g.wicket[1] - g.wicket[0], 3), "dir": 1, "wicket": True}]
            others = [] if inside else [tuple(g.wicket)]
        w = span[1] - span[0]
        obstacles = [tuple(p) for p in parked] + [house] + [tuple(b) for b in blocked]
        if not g.wicket:
            obstacles.append((walk.x0 - 0.45, self.H - 3.0, walk.x1 + 0.45, self.H))
        if w <= SWING_MAX + 1e-9:
            boxes = self._swing(g, obstacles, span=span)
            if boxes is not None:
                g.why.append("Two leaves swing inward (never over the road), clear of the parked car and the porch."
                             if len(g.leaves) - (1 if g.wicket else 0) > 1 else "One leaf swings inward.")
                return boxes
        # sliding: the leaf runs back behind the wall on one side
        for kind, travel in (("sliding", w + 0.3), ("telescopic", w / 2 + 0.3)):
            opts = []
            for to in (1, -1):
                r0, r1 = (span[1], span[1] + travel) if to > 0 else (span[0] - travel, span[0])
                if r0 < 0.2 or r1 > self.W - 0.2:
                    continue
                if any(o[0] < r1 and r0 < o[1] for o in others):
                    continue
                strip = (r0, self.H - 0.4, r1, self.H)
                if any(_overlap(strip, tuple(b)) for b in blocked):
                    continue
                opts.append((abs((r0 + r1) / 2 - self.W / 2), to, r0, r1))
            if opts:
                _, to, r0, r1 = min(opts)
                if kind == "sliding":
                    g.leaves = [lf for lf in g.leaves if lf.get("wicket")] + [
                        {"type": "slide", "x0": span[0], "x1": span[1], "to": to, "travel": round(w, 3)}]
                    g.why.append(f"Sliding gate: the leaf rolls {w:.1f} m back behind the wall "
                                 f"({'right' if to > 0 else 'left'}) — an inward swing would hit "
                                 "the parked car or the porch, or the leaves would be too heavy.")
                else:
                    half = w / 2
                    tl, tr = (round(w, 3), round(half, 3)) if to > 0 else (round(half, 3), round(w, 3))
                    g.leaves = [lf for lf in g.leaves if lf.get("wicket")] + [
                        {"type": "slide", "x0": round(span[0], 3), "x1": round(span[0] + half, 3), "to": to, "travel": tl},
                        {"type": "slide", "x0": round(span[0] + half, 3), "x1": round(span[1], 3), "to": to, "travel": tr}]
                    g.why.append(f"Telescopic sliding gate (two leaves, {half + 0.3:.1f} m of run-back): the wall "
                                 "beside it is too short for one sliding leaf.")
                g.op = kind
                g.runback = [round(r0, 3), round(r1, 3)]
                return [(r0, self.H - 0.4, r1 - r0, 0.4)]
        # bi-fold: four leaves folding in pairs to the pillars
        q = w / 4
        g.leaves = [lf for lf in g.leaves if lf.get("wicket")] + [
            {"type": "fold", "hinge": span[0], "len": round(q, 3), "dir": 1},
            {"type": "fold", "hinge": span[1], "len": round(q, 3), "dir": -1}]
        g.op = "bifold"
        g.why.append(f"Bi-fold gate: the leaves fold in pairs ({q:.2f} m deep) — no room to swing or slide; "
                     f"keep {q + 0.2:.1f} m behind each pillar free.")
        return [(span[0], self.H - q - 0.05, q, q + 0.05), (span[1] - q, self.H - q - 0.05, q, q + 0.05)]


_COMPASS = {"N": "North", "E": "East", "S": "South", "W": "West"}


def summary(gates: List[Dict], road: float) -> str:
    """One line for the design notes."""
    parts = []
    for g in gates:
        what = {"walk": "walking gate", "two_wheeler": "walking + two-wheeler gate", "vehicle": "car gate",
                "vehicle_wicket": "car gate with a wicket door"}[g["kind"]]
        op = {"swing": "", "sliding": " sliding", "telescopic": " telescopic", "bifold": " bi-fold"}[g["op"]]
        parts.append(f"{g['tag']} {what} {g['width']:.2f} m ({g['feet']}){op}")
    return f"Gates on the {road_label(road)}: " + "; ".join(parts) + "."
