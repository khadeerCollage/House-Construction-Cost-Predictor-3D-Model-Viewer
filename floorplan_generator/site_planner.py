"""
Site Planner
============
Plot-level engineering that happens *around* the house:

  1. Setbacks from indicative Indian municipal bylaws (plot area & storeys)
  2. Vastu site rule: keep more open space on North & East than South & West,
     push the building mass towards the South-West (Nairutya)
  3. Ground-coverage cap and footprint sizing from the space program
     (a 30-cent plot does not get a 30-cent house — the rest becomes open site)
  4. Reservations for site amenities (car porch depth, garden, pool, external
     terrace stair) — enlarging the right setback only while the house still fits
  5. Collision-free placement of amenities in the open strips, keeping a clear
     entrance path from the gate to the main door
  6. Gates sized for what passes through them (gates.py): people, two-wheelers, or the
     design car's swept path from the road; leaves that can open on this plot; stilt bays

Plan frame: +x right, +y up, road/entrance side = top edge (y = plot_height).
All dimensions in metres.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import re

from floorplan_generator.vastu_engine import plan_sides_for_facing, compass_vector

Rect = Tuple[float, float, float, float]   # x, y, w, h

GRID = 0.05
MARGIN_WALL = 0.30      # clear gap between an amenity and the building face
MARGIN_PLOT = 0.20      # clear gap between an amenity and the compound wall


def _snap(v: float, g: float = GRID) -> float:
    return round(round(v / g) * g, 3)


def _overlap(a: Rect, b: Rect, tol: float = 1e-6) -> bool:
    return (a[0] < b[0] + b[2] - tol and a[0] + a[2] > b[0] + tol and
            a[1] < b[1] + b[3] - tol and a[1] + a[3] > b[1] + tol)


def _inside(a: Rect, b: Rect, tol: float = 1e-6) -> bool:
    return (a[0] >= b[0] - tol and a[1] >= b[1] - tol and
            a[0] + a[2] <= b[0] + b[2] + tol and a[1] + a[3] <= b[1] + b[3] + tol)


def bylaw_setbacks(plot_area: float, floors: int) -> Tuple[float, float, float]:
    """Indicative (front, rear, side) setbacks in metres for Indian residential plots."""
    if plot_area <= 60:
        f, r, s = 1.0, 0.5, 0.0
    elif plot_area <= 100:
        f, r, s = 1.5, 0.9, 0.6
    elif plot_area <= 200:
        f, r, s = 2.0, 1.0, 0.9
    elif plot_area <= 300:
        f, r, s = 3.0, 1.5, 1.0
    elif plot_area <= 500:
        f, r, s = 3.0, 2.0, 1.5
    else:
        f, r, s = 4.5, 3.0, 2.0
    if floors > 3:   # buildings above ~12 m need fire-tender clearances
        extra = 0.5 * (floors - 3)
        r = min(r + extra, 6.0)
        s = min(s + extra, 6.0)
    return f, r, s


def coverage_cap(plot_area: float) -> float:
    if plot_area <= 100:
        return 0.75
    if plot_area <= 200:
        return 0.65
    if plot_area <= 500:
        return 0.60
    return 0.50


@dataclass
class SiteItem:
    key: str
    name: str
    category: str
    rect: Rect
    color: str


@dataclass
class SitePlan:
    plot_w: float
    plot_h: float
    facing: str
    setbacks: Dict[str, float]               # plan sides: top/bottom/left/right
    footprint: Rect                          # house footprint (wall centre-lines)
    car_mode: Optional[str] = None           # 'front' | 'left' | 'right' | 'stilt' | None
    ext_stair_side: Optional[str] = None     # plan side holding the terrace stair
    ext_stair_want: Optional[str] = None     # the side the owner asked for (plan side), None = automatic
    ext_stair_refused: Optional[str] = None  # why the owner's side could not take it (plain words)
    notes: List[str] = field(default_factory=list)
    items: List[SiteItem] = field(default_factory=list)
    gates: List[Tuple[float, float]] = field(default_factory=list)  # x-ranges on the top edge
    coverage: float = 0.0
    # x-range of the road facade the main door must keep clear of (a corner car bay sits in front of it)
    door_exclusion: Optional[Tuple[float, float]] = None
    # a lift tower outside the house: 'top' (beside the main door) | 'left' | 'right' (against a side wall)
    lift_side: Optional[str] = None
    reserved: List[Rect] = field(default_factory=list)   # placed before the amenities (lift tower + landing)
    gate_specs: List[Dict] = field(default_factory=list)  # gates.Gate.as_dict() for each gate, left → right
    drive: Optional[Dict] = None                           # the design car's path from the road to its bay
    road: float = 9.0                                      # road width in front (m)


class SitePlanner:
    """Computes setbacks/footprint first, then places amenities once the door is known."""

    def __init__(self, plot_w: float, plot_h: float, facing: str,
                 vastu_enabled: bool = True, building_type: str = "Independent House",
                 num_floors: int = 1, components: Optional[Dict] = None,
                 members: int = 4):
        self.W = plot_w
        self.H = plot_h
        self.facing = (facing or "N").upper()[0]
        self.vastu = vastu_enabled
        self.building_type = building_type or ""
        self.floors = max(1, int(num_floors or 1))
        self.components = components or {}
        self.members = max(1, members)
        self.sides = plan_sides_for_facing(self.facing)          # plan side -> compass
        self.side_of = {c: s for s, c in self.sides.items()}     # compass -> plan side
        self.multi_level = (self.floors > 1 or "apartment" in self.building_type.lower()
                            or "duplex" in self.building_type.lower())
        self.road = 9.0                 # set by the engine: road width in front of the plot (m)
        self.quality = "Standard"       # set by the engine: picks the design car for the car gate
        self.stilt_info = None          # set by the engine: stilt bays clear of the core and the columns

    # ------------------------------------------------------------------ utils
    def _enabled(self, key: str) -> Dict:
        cfg = self.components.get(key) or {}
        return cfg if (cfg.get("enabled") or cfg.get("slots", 0) or cfg.get("pct", 0)) else {}

    def _available(self, sb: Dict[str, float]) -> Rect:
        x = sb["left"]
        y = sb["bottom"]
        return (x, y, self.W - sb["left"] - sb["right"], self.H - sb["bottom"] - sb["top"])

    @staticmethod
    def _fits(rect: Rect, min_w: float, min_d: float, min_area: float) -> bool:
        return rect[2] >= min_w and rect[3] >= min_d and rect[2] * rect[3] >= min_area

    def corner_point(self, compass_corner: str) -> Tuple[float, float]:
        """Plan coordinates of the plot corner nearest a compass corner like 'NE'."""
        vx = vy = 0.0
        for c in compass_corner:
            dx, dy = compass_vector(c, self.facing)
            vx += dx
            vy += dy
        x = self.W if vx > 0 else 0.0
        y = self.H if vy > 0 else 0.0
        return x, y

    # ----------------------------------------------------------- stage 1: plan
    @staticmethod
    def lift_front_need(lift_out: Dict) -> float:
        """Front setback for a lift tower beside the main door: landing in front of the lift door + 3 steps."""
        sw, dw = lift_out["w"], lift_out.get("door", 0.9)
        landing = max(1.2, 0.23 + (sw + dw) / 2 + 0.15)
        return round(landing + 0.9 + 0.3, 2)

    @staticmethod
    def lift_side_need(lift_out: Dict) -> float:
        """Side setback for a lift tower against the side wall (tower + its walls + clearance to the compound)."""
        return round(0.23 + lift_out["w"] + 0.115 + MARGIN_PLOT + 0.05, 2)

    # an outside stair on the road side: the stair band along the house face + a walk past it to the gate
    EXT_STAIR_BAND = 1.15                 # 0.9 m flight + 0.15 m off the wall + the wall's outer half
    EXT_STAIR_FRONT = 2.25                # front yard: the band + 1.1 m to walk past it to the stair foot
    EXT_STAIR_DOOR = 2.6                  # facade length the main door + its steps need beside a front stair

    def ext_stair_need(self, side: str) -> float:
        """Setback a side needs for the outside terrace stair hugging the house."""
        return self.EXT_STAIR_FRONT if side == "top" else self.EXT_STAIR_BAND

    def plan_footprint(self, target_area: float, min_area: float,
                       min_w: float, min_d: float, ext_stair_len: float = 0.0,
                       lift_out: Optional[Dict] = None, ext_stair_want: Optional[str] = None) -> SitePlan:
        """
        Setbacks, footprint and the reservations of the site. A side asked for the outside terrace stair is
        taken only when it costs nothing else the owner asked for: the plot is planned both ways and, if the
        stair's side would push out the car parking or the swimming pool, the stair keeps its automatic side
        and the reason is kept (site.ext_stair_refused) — a wish never takes away another wish.
        """
        want = ext_stair_want if ext_stair_want in ("top", "bottom", "left", "right") else None
        if not want or ext_stair_len <= 0:
            return self._plan_footprint(target_area, min_area, min_w, min_d, ext_stair_len, lift_out, want)
        import copy
        comps0 = copy.deepcopy(self.components)
        base = self._plan_footprint(target_area, min_area, min_w, min_d, ext_stair_len, lift_out, None)
        comps_base = self.components
        self.components = copy.deepcopy(comps0)
        site = self._plan_footprint(target_area, min_area, min_w, min_d, ext_stair_len, lift_out, want)
        if site.ext_stair_side != want or base.ext_stair_side == want:
            return site
        lost = []
        if base.car_mode and not site.car_mode:
            lost.append("car parking")
        pool = lambda c: bool((c.get("swimming_pool") or {}).get("enabled"))       # noqa: E731
        if pool(comps_base) and not pool(self.components):
            lost.append("swimming pool")
        if not lost:
            return site
        self.components = comps_base
        cname = {"N": "North", "E": "East", "S": "South", "W": "West"}
        why = f"the {' and the '.join(lost)} would no longer fit beside it"
        base.notes.append(f"Terrace stair: the {cname[self.sides[want]]} side cannot take it — {why}. "
                          f"It stays on the {cname[self.sides[base.ext_stair_side]]} side."
                          if base.ext_stair_side else f"Terrace stair: {why}.")
        base.ext_stair_want, base.ext_stair_refused = want, why
        return base

    def _plan_footprint(self, target_area: float, min_area: float, min_w: float, min_d: float,
                        ext_stair_len: float = 0.0, lift_out: Optional[Dict] = None,
                        ext_stair_want: Optional[str] = None) -> SitePlan:
        plot_area = self.W * self.H
        f, r, s = bylaw_setbacks(plot_area, self.floors)
        if "apartment" in self.building_type.lower() and plot_area > 200:
            s = max(s, 1.5)
        if self.H >= 10.0:
            f = max(f, 2.25)      # room for the entrance landing + 3 steps
        sb = {"top": f, "bottom": r, "left": s, "right": s}
        notes: List[str] = []

        def feasible(cand: Dict[str, float]) -> bool:
            return self._fits(self._available(cand), min_w, min_d, min_area)

        # Relax bylaw minimums progressively on very small plots (row-house style)
        relax_steps = [
            {"left": 0.0, "right": 0.0},
            {"bottom": 0.5},
            {"top": 1.0},
            {"bottom": 0.0, "top": 0.6},
        ]
        relaxed = False
        for step in relax_steps:
            if feasible(sb):
                break
            sb.update(step)
            relaxed = True
        if relaxed:
            notes.append("Plot below the bylaw threshold for full setbacks — side setbacks relaxed "
                         "(row-house / small-plot relaxation; confirm with the local authority).")

        # Vastu: open space on N ≥ S and E ≥ W — only when the house keeps its
        # full target area (a cramped house would lose the SW master suite, which
        # costs more Vastu than the extra open sky gains).
        if self.vastu:
            changed = False
            for a, b in (("N", "S"), ("E", "W")):
                sa, sb_side = self.side_of[a], self.side_of[b]
                if sb[sa] + 1e-6 < sb[sb_side]:
                    cand = dict(sb)
                    cand[sa] = sb[sb_side]
                    av = self._available(cand)
                    if feasible(cand) and av[2] * av[3] >= target_area and av[2] >= min_w + 1.0:
                        sb = cand
                        changed = True
            if changed:
                notes.append("Vastu site rule applied: North and East setbacks kept ≥ South and West "
                             "(more open sky and morning light on the Ishan side).")

        # External terrace stair strip (single storey): automatically on a S or W side, never the road
        # side; the owner may ask for any side (Smart Re-Plan) — that side is tried first and, when it
        # cannot take the stair, the reason is kept and the automatic side is used
        ext_side = None
        want = ext_stair_want if ext_stair_want in ("top", "bottom", "left", "right") else None
        refused = None
        if ext_stair_len > 0:
            candidates = [self.side_of[c] for c in ("S", "W", "E", "N") if self.side_of[c] != "top"]
            if want:
                candidates = [want] + [c for c in candidates if c != want]
            for side in candidates:
                cand = dict(sb)
                cand[side] = max(cand[side], self.ext_stair_need(side) if side == want else 1.15)
                avail = self._available(cand)
                run = avail[3] if side in ("left", "right") else avail[2]
                need_run = ext_stair_len + 0.3 + (self.EXT_STAIR_DOOR if side == "top" else 0.0)
                length_ok = run >= need_run
                if feasible(cand) and length_ok:
                    sb = cand
                    ext_side = side
                    break
                if side == want:
                    refused = self._ext_stair_refusal(side, cand, run, need_run, ext_stair_len, min_area,
                                                      min_w, min_d)

        if ext_stair_len > 0 and ext_side is None:
            notes.append("No setback can take an external terrace stair — provide an internal "
                         "ladder/hatch or a compact internal stair to the terrace.")

        # Lift tower outside the house: beside the main door (deeper front yard) or against a side wall
        lift_side = None
        if lift_out and lift_out.get("position") in ("entrance", "side"):
            front = self.lift_front_need(lift_out)
            if lift_out["position"] == "entrance":
                cand = dict(sb)
                cand["top"] = max(cand["top"], front)
                if feasible(cand):
                    sb, lift_side = cand, "top"
            else:
                from floorplan_generator.vastu_engine import VastuEngine
                plot_eng = VastuEngine.for_facing(self.W, self.H, self.facing)
                need = self.lift_side_need(lift_out)
                best = None
                for side in ("left", "right"):
                    cand = dict(sb)
                    cand[side] = max(cand[side], need)
                    cand["top"] = max(cand["top"], front)       # the landing may join the entrance instead
                    if not feasible(cand):
                        continue
                    av = self._available(cand)
                    cx = need / 2 if side == "left" else self.W - need / 2
                    zone = plot_eng.get_zone_for_position(cx, av[1] + av[3] * 0.6).value
                    vs = plot_eng.get_room_score("LIFT", zone) if self.vastu else 0
                    key = vs + 0.4 * av[2] * av[3]
                    if best is None or key > best[0]:
                        best = (key, side, cand)
                if best:
                    sb, lift_side = best[2], best[1]
                else:                                    # no side yard is wide enough: beside the main door
                    cand = dict(sb)
                    cand["top"] = max(cand["top"], front)
                    if feasible(cand):
                        sb, lift_side = cand, "top"
                        notes.append("No side yard is wide enough for the lift tower — it stands beside the "
                                     "main door instead.")
            if lift_side is None:
                notes.append("The lift could not go outside the house on this plot (the "
                             + ("front yard" if lift_out["position"] == "entrance" else "side yards")
                             + " would leave too little house) — it stays inside, next to the stair.")

        # Car parking reservation
        car_mode = None
        door_exclusion = None
        car_cfg = self._enabled("car_parking")
        if car_cfg and car_cfg.get("mode") == "stilt" and self.multi_level:
            # site_feasibility found an open bay would squeeze the rooms below a comfortable size
            car_mode = "stilt"
            n_cars = max(1, int(car_cfg.get("slots", 1) or 1))
            notes.append(f"Car parking ({n_cars} bay{'s' if n_cars > 1 else ''}) on a stilt floor below the "
                         "building: an open bay would make every room smaller than comfortable.")
            car_cfg = {}
        if car_cfg:
            slots = max(1, int(car_cfg.get("slots", 1) or 1))
            path_w = 1.9                      # entrance path to the main door
            options = []
            # front open parking, cars perpendicular to the road (5.0 m bay + clearances)
            stair_f = self.EXT_STAIR_BAND if ext_side == "top" else 0.0   # a front stair hugs the house face
            if slots * 2.6 + path_w <= self.W - 0.4:
                cand = dict(sb)
                cand["top"] = max(cand["top"], 5.8 + stair_f)
                options.append(("front", cand))
            # front parallel parking along the compound wall (2.6 m bay + clearances)
            if slots * 5.5 + path_w <= self.W - 0.4:
                cand = dict(sb)
                cand["top"] = max(cand["top"], 3.3 + stair_f)
                options.append(("front_parallel", cand))
            # side parking (tandem along the depth)
            for side in ("left", "right"):
                if side == lift_side:
                    continue
                cand = dict(sb)
                # the outside terrace stair hugs the house in this yard too: the car lane goes beside it
                cand[side] = max(cand[side], 3.25 + (1.15 if side == ext_side else 0.0))
                if slots * 5.2 <= self.H - cand["top"] - 0.4:
                    options.append((side, cand))
            def front_free(cand) -> float:
                """Longest free run along the front strip outside the likely entrance zone
                (auspicious padas sit at 25–62.5 % of the facade from the left)."""
                aw = self.W - cand["left"] - cand["right"]
                left_free = cand["left"] + 0.25 * aw - 0.45 - 0.2
                right_free = self.W - (cand["left"] + 0.625 * aw) - 0.45 - 0.2
                return max(left_free, right_free)

            chosen = None
            for mode, cand in options:
                if mode == "front" and slots * 2.6 > front_free(cand):
                    continue
                if mode == "front_parallel" and slots * 5.5 > front_free(cand):
                    continue
                if feasible(cand):
                    area = self._available(cand)
                    score = area[2] * area[3]
                    if mode == "front":
                        score *= 1.04   # gate-side parking is the convenient default
                    if chosen is None or score > chosen[2]:
                        chosen = (mode, cand, score)
            if not chosen and slots * 5.5 + path_w <= self.W - 0.4:
                # Narrow plot: park parallel to the road at the vehicle-friendly front corner
                # (NW / SE) and keep the main door on the rest of the facade, instead of dropping
                # the car the owner asked for.
                cand = dict(sb)
                cand["top"] = max(cand["top"], 3.3 + stair_f)
                if feasible(cand):
                    chosen = ("front_parallel", cand, 0.0)
                    bay = slots * 5.5
                    if self._car_corner_side(bay, self._available(cand)) == "left":
                        door_exclusion = (0.0, MARGIN_PLOT + bay + 0.45)
                    else:
                        door_exclusion = (self.W - MARGIN_PLOT - bay - 0.45, self.W)
                    notes.append(f"Narrow frontage: car parked parallel to the road in the front yard "
                                 f"({slots * 5.5:.1f} × 2.6 m bay); the main door is kept clear of it.")
            if chosen:
                car_mode, sb = chosen[0], chosen[1]
            elif self.multi_level:
                car_mode = "stilt"
                notes.append(f"Car parking ({slots} bays) moved to a stilt floor below the building: "
                             "the plot cannot hold open parking and the house together.")
            else:
                notes.append("Car parking not feasible on this plot without making the house smaller "
                             "than NBC minimums — use on-street parking or reduce rooms.")

        # Pool / garden reservations: grow N or E setback while the house still fits
        pool_cfg = self._enabled("swimming_pool")
        if pool_cfg:
            pw, pl = self._pool_dims(pool_cfg)
            need = pw + 1.6
            placed = False
            for c in ("N", "E"):
                side = self.side_of[c]
                if side == "top" and car_mode in ("front", "front_parallel"):
                    continue
                cand = dict(sb)
                cand[side] = max(cand[side], need)
                strip_len = self.W if side in ("top", "bottom") else self.H - cand["top"] - cand["bottom"]
                if feasible(cand) and strip_len >= pl + 1.0:
                    sb = cand
                    placed = True
                    break
            if not placed:
                notes.append("Swimming pool skipped: no North/East open strip can take it without "
                             "shrinking the house below the brief.")
                self.components = dict(self.components)
                self.components["swimming_pool"] = {"enabled": False}

        garden_cfg = self._enabled("garden")
        if garden_cfg:
            target = self.W * self.H * max(5, int(garden_cfg.get("pct", 10) or 10)) / 100.0
            for _ in range(12):
                open_ne = self._open_area_ne(sb)
                if open_ne >= target:
                    break
                grown = False
                for c in ("N", "E"):
                    side = self.side_of[c]
                    if side == "top" and car_mode == "front":
                        continue
                    cand = dict(sb)
                    cand[side] = cand[side] + 0.5
                    if feasible(cand):
                        sb = cand
                        grown = True
                        break
                if not grown:
                    break

        # Footprint sizing inside the available rectangle
        avail = self._available(sb)
        cap = coverage_cap(plot_area)
        wanted = max(min_area, target_area)
        max_by_cov = cap * plot_area
        if wanted > max_by_cov and min_area <= max_by_cov:
            wanted = max_by_cov
        fw, fd = avail[2], avail[3]
        if fw * fd > wanted * 1.06:
            ratio = fd / max(fw, 1e-6)
            fw_new = math.sqrt(wanted / ratio)
            fd_new = wanted / fw_new
            fw = max(min_w, min(fw, fw_new))
            fd = max(min_d, min(fd, wanted / fw if fw > 0 else fd_new))
        fw = round(math.floor(min(fw, avail[2]) / GRID + 1e-6) * GRID, 3)
        fd = round(math.floor(min(fd, avail[3]) / GRID + 1e-6) * GRID, 3)

        # Anchor: Vastu → towards the SW corner of the available rect; else road side, centred
        if self.vastu:
            sw_x, sw_y = self.corner_point("SW")
            fx = avail[0] if sw_x == 0.0 else avail[0] + avail[2] - fw
            fy = avail[1] if sw_y == 0.0 else avail[1] + avail[3] - fd
        else:
            fx = avail[0] + (avail[2] - fw) / 2.0
            fy = avail[1] + avail[3] - fd
        fx, fy = _snap(fx), _snap(fy)
        # Report the real (final) setbacks
        final_sb = {
            "left": round(fx, 2), "bottom": round(fy, 2),
            "right": round(self.W - fx - fw, 2), "top": round(self.H - fy - fd, 2),
        }
        coverage = (fw + 0.23) * (fd + 0.23) / plot_area
        notes.append(
            "Setbacks: " + ", ".join(f"{self.sides[k]} {v:.2f} m" for k, v in
                                      sorted(final_sb.items(), key=lambda kv: "NESW".index(self.sides[kv[0]]))) +
            f" · ground coverage {coverage * 100:.0f}% (cap {cap * 100:.0f}%).")
        if coverage > cap + 0.01:
            notes.append("Ground coverage exceeds the indicative cap — consider an additional floor instead.")

        if ext_side == "top":
            # a stair along the road face takes one end of it: the main door stays on the rest
            door_exclusion = self._front_stair_exclusion(ext_stair_len, (fx, fy, fw, fd), door_exclusion)
        cname = {"N": "North", "E": "East", "S": "South", "W": "West"}
        if want and refused:
            notes.append(f"Terrace stair: the {cname[self.sides[want]]} side cannot take it — {refused}."
                         + (f" It stays on the {cname[self.sides[ext_side]]} side." if ext_side else ""))
        elif want and ext_side == want:
            notes.append(f"Terrace stair on the {cname[self.sides[want]]} side, as you asked.")
        return SitePlan(plot_w=self.W, plot_h=self.H, facing=self.facing, setbacks=final_sb,
                        footprint=(fx, fy, fw, fd), car_mode=car_mode, ext_stair_side=ext_side,
                        notes=notes, coverage=coverage, door_exclusion=door_exclusion, lift_side=lift_side,
                        ext_stair_want=want, ext_stair_refused=refused)

    def ext_stair_climb(self, rect: Rect, footprint: Rect) -> str:
        """
        Which way the outside stair climbs (plan N/S/E/W = the end it arrives at). Its top must lie
        alongside the house (a short bridge crosses onto the roof — walkthrough._stair measures the
        same overlap); of the ends that do, Vastu wants the climb clockwise, rising towards the South
        or the West. Without Vastu (or when only one end reaches the house) the old rule stands.
        """
        fx, fy, fw, fd = footprint
        x, y, w, h = rect
        along_y = h >= w
        lo_o, hi_o = ((fy - 0.115, fy + fd + 0.115) if along_y else (fx - 0.115, fx + fw + 0.115))
        lo_r, hi_r = (y, y + h) if along_y else (x, x + w)
        ends = {}
        for d, (a, b) in ((("N", (hi_r - 0.9, hi_r)), ("S", (lo_r, lo_r + 0.9))) if along_y else
                          (("E", (hi_r - 0.9, hi_r)), ("W", (lo_r, lo_r + 0.9)))):
            ends[d] = min(b, hi_o - 0.2) - max(a, lo_o + 0.2)
        default = "N" if along_y else "E"
        reach = [d for d, ov in ends.items() if ov >= 0.6 - 1e-6]
        if not reach:
            return max(ends, key=lambda d: (ends[d] - (1e-6 if d != default else 0.0)))
        if self.vastu:
            plan_side = {"N": "top", "S": "bottom", "E": "right", "W": "left"}
            good = [d for d in reach if self.sides[plan_side[d]] in ("S", "W")]
            if good:
                return good[0]
        return default if default in reach else reach[0]

    CLIMB_BONUS = 3.0     # a clockwise climb (rising towards S / W) against the zone score (-50 … +10)

    def ext_stair_vastu(self, rect: Rect, footprint: Rect) -> float:
        """Vastu worth of an outside stair here: its zone on the plot mandala + a clockwise climb."""
        from floorplan_generator.vastu_engine import VastuEngine
        eng = VastuEngine.for_facing(self.W, self.H, self.facing)
        zone = eng.get_zone_for_position(rect[0] + rect[2] / 2, rect[1] + rect[3] / 2).value
        climb = self.ext_stair_climb(rect, footprint)
        plan_side = {"N": "top", "S": "bottom", "E": "right", "W": "left"}
        ok = self.sides[plan_side[climb]] in ("S", "W")
        return eng.get_room_score("STAIRS", zone) + (self.CLIMB_BONUS if ok else 0.0)

    def _ext_stair_refusal(self, side: str, cand: Dict[str, float], run: float, need_run: float,
                           stair_len: float, min_area: float, min_w: float, min_d: float) -> str:
        """Why `side` cannot take the outside stair, in plain words (measured)."""
        avail = self._available(cand)
        if run < need_run:
            fmt = ".2f" if f"{run:.1f}" == f"{need_run:.1f}" else ".1f"
            if side == "top":
                return (f"the front of the house is only {run:{fmt}} m wide; the stair ({stair_len:.1f} m long) "
                        f"and the main door with its steps need {need_run:{fmt}} m")
            return f"that side is only {run:{fmt}} m long and the stair needs {need_run:{fmt}} m"
        if avail[2] < min_w:
            return (f"with a {cand[side]:.2f} m strip for it the house could be only {avail[2]:.1f} m wide — "
                    f"the rooms need {min_w:.1f} m")
        if avail[3] < min_d:
            return (f"with a {cand[side]:.2f} m strip for it the house could be only {avail[3]:.1f} m deep — "
                    f"the rooms need {min_d:.1f} m")
        if avail[2] * avail[3] < min_area:
            return (f"a {cand[side]:.2f} m strip there would leave a house of {avail[2] * avail[3]:.0f} m², "
                    f"less than the {min_area:.0f} m² the rooms need")
        return "the plot has no room for it there"

    def _front_stair_exclusion(self, stair_len: float, footprint: Rect,
                               door_exclusion: Optional[Tuple[float, float]]) -> Tuple[float, float]:
        """
        x-range of the road face a front stair takes (the main door is kept off it). The stair goes
        to the end a front car bay already blocks (one blocked end, not two), else the Vastu end
        (towards the South / West, where a heavy stair belongs).
        """
        fx, fy, fw, fd = footprint
        span = stair_len + 0.45
        if door_exclusion is not None:
            left = door_exclusion[0] <= 0.05
        else:
            sx, _ = self.corner_point("SW")
            left = sx == 0.0
        if left:
            hi = fx - 0.115 + span
            return (0.0, max(hi, door_exclusion[1]) if door_exclusion else hi)
        lo = fx + fw + 0.115 - span
        return (min(lo, door_exclusion[0]) if door_exclusion else lo, self.W)

    def _car_corner_side(self, bay: float, facade: Rect) -> str:
        """
        Front corner (plan 'left' / 'right') for a parallel car bay. The bay blocks that end of
        the facade for the main door, so with Vastu on the side is chosen to keep the most
        auspicious entrance pada free (entrance weight 3) while preferring a vehicle zone
        (NW/SE, weight 0.75); without Vastu, the NW/SE corner.
        """
        from floorplan_generator.vastu_engine import VastuEngine, pada_for_point
        corner = "NW" if "NW" in self._front_corners() else "SE"
        default = "left" if corner == self._front_corners()[0] else "right"
        if not self.vastu:
            return default
        fx, fy, fw, fd = facade
        plot_eng = VastuEngine.for_facing(self.W, self.H, self.facing)
        best = None
        for side in ("left", "right"):
            ex = (0.0, MARGIN_PLOT + bay + 0.45) if side == "left" else (self.W - MARGIN_PLOT - bay - 0.45, self.W)
            door = None
            x = fx + 0.45
            while x <= fx + fw - 1.45 + 1e-6:
                if x >= ex[1] or x + 1.0 <= ex[0]:
                    sc = pada_for_point(x + 0.5, fy + fd, facade, self.facing).score
                    door = sc if door is None else max(door, sc)
                x += 0.1
            if door is None:
                continue
            cx = MARGIN_PLOT + bay / 2 if side == "left" else self.W - MARGIN_PLOT - bay / 2
            car = plot_eng.get_room_score("PARKING", plot_eng.get_zone_for_position(cx, self.H - 1.5).value)
            val = 3.0 * door + 0.75 * car + (0.1 if side == default else 0.0)
            if best is None or val > best[0]:
                best = (val, side)
        return best[1] if best else default

    def _open_area_ne(self, sb: Dict[str, float]) -> float:
        area = 0.0
        for c in ("N", "E"):
            side = self.side_of[c]
            if side in ("top", "bottom"):
                area += sb[side] * self.W
            else:
                area += sb[side] * (self.H - sb["top"] - sb["bottom"])
        return area

    @staticmethod
    def _pool_dims(cfg: Dict) -> Tuple[float, float]:
        size = str(cfg.get("size", "Small")).lower()
        if "large" in size:
            return 5.0, 10.0
        if "medium" in size:
            return 4.0, 8.0
        return 3.0, 6.0

    # -------------------------------------------------------- stage 2: place
    def strips(self, site: SitePlan) -> Dict[str, Rect]:
        fx, fy, fw, fd = site.footprint
        face = 0.115 + MARGIN_WALL          # outer wall face + clearance
        top_y = fy + fd + face
        strips = {
            "top": (MARGIN_PLOT, top_y, self.W - 2 * MARGIN_PLOT, self.H - top_y - MARGIN_PLOT),
            "bottom": (MARGIN_PLOT, MARGIN_PLOT, self.W - 2 * MARGIN_PLOT, fy - face - MARGIN_PLOT),
            "left": (MARGIN_PLOT, fy - 0.115, fx - face - MARGIN_PLOT, fd + 0.23),
            "right": (fx + fw + face, fy - 0.115, self.W - fx - fw - face - MARGIN_PLOT, fd + 0.23),
        }
        return {k: v for k, v in strips.items() if v[2] > 0.3 and v[3] > 0.3}

    def _free_positions(self, w: float, h: float, strip: Rect, toward: Tuple[float, float],
                        occupied: List[Rect], limit: int = 80) -> List[Rect]:
        """Free w×h positions inside `strip`, nearest to `toward` first."""
        sx, sy, sw, sh = strip
        if w > sw + 1e-6 or h > sh + 1e-6:
            return []
        step = 0.25
        nx = int((sw - w) / step) + 1
        ny = int((sh - h) / step) + 1
        cands = []
        for i in range(nx + 1):
            for j in range(ny + 1):
                x = min(sx + i * step, sx + sw - w)
                y = min(sy + j * step, sy + sh - h)
                cands.append((abs(x + w / 2 - toward[0]) + abs(y + h / 2 - toward[1]), x, y))
        cands.sort()
        out = []
        seen = set()
        for _, x, y in cands:
            rect = (_snap(x), _snap(y), w, h)
            if rect in seen or not _inside(rect, strip, tol=0.06):
                continue
            seen.add(rect)
            if any(_overlap(rect, o) for o in occupied):
                continue
            out.append(rect)
            if len(out) >= limit:
                break
        return out

    def _slide_place(self, w: float, h: float, strip: Rect, toward: Tuple[float, float],
                     occupied: List[Rect]) -> Optional[Rect]:
        """Put a w×h rect in `strip`, as close as possible to point `toward`, avoiding `occupied`."""
        sx, sy, sw, sh = strip
        if w > sw + 1e-6 or h > sh + 1e-6:
            return None
        tx = min(max(toward[0] - w / 2, sx), sx + sw - w)
        ty = min(max(toward[1] - h / 2, sy), sy + sh - h)
        candidates = []
        step = 0.25
        nx = int((sw - w) / step) + 1
        ny = int((sh - h) / step) + 1
        for i in range(nx + 1):
            for j in range(ny + 1):
                x = min(sx + i * step, sx + sw - w)
                y = min(sy + j * step, sy + sh - h)
                candidates.append((abs(x - tx) + abs(y - ty), x, y))
        candidates.sort()
        for _, x, y in candidates:
            rect = (_snap(x), _snap(y), w, h)
            if not _inside(rect, strip, tol=0.06):
                continue
            if any(_overlap(rect, o) for o in occupied):
                continue
            return rect
        return None

    def place_amenities(self, site: SitePlan, door_x0: float, door_x1: float,
                        ext_stair=None) -> SitePlan:
        """Place all requested site amenities; records gates and notes on `site`."""
        strips = self.strips(site)
        fx, fy, fw, fd = site.footprint
        occupied: List[Rect] = []
        items: List[SiteItem] = []
        notes = site.notes

        # Gates (gates.py): the walking gate's width follows who uses it, so it is sized first
        import importlib
        _gates = importlib.import_module("floorplan_generator.gates")
        car_cfg0, bike_cfg0 = self._enabled("car_parking"), self._enabled("bike_parking")
        vehicle = bool(car_cfg0) and site.car_mode in ("front", "front_parallel", "left", "right", "stilt")
        n_cars = max(1, int(car_cfg0.get("slots", 1) or 1)) if vehicle else 0
        n_bikes = max(2, int(bike_cfg0.get("slots", 2) or 2)) if bike_cfg0 else 0
        designer = _gates.GateDesigner(self.W, self.H, self.facing, self.vastu, self.road, self.quality,
                                       self.building_type, self.floors, stilt=site.car_mode == "stilt",
                                       cars=n_cars, bikes=n_bikes)
        walk = designer.walk_gate(door_x0, door_x1, vehicle)

        # Clear entrance path: gate → landing → main door
        px0, px1 = min(door_x0 - 0.45, walk.x0), max(door_x1 + 0.45, walk.x1)
        path = (px0, fy + fd, px1 - px0, self.H - (fy + fd))
        occupied.append(path)
        occupied.extend(getattr(site, "reserved", None) or [])      # an outside lift tower and its landing
        gates = []

        from floorplan_generator.vastu_engine import VastuEngine
        plot_eng = VastuEngine.for_facing(self.W, self.H, self.facing)

        def put(key, name, cat, w, h, strip_order, corner, color, rotate=True,
                min_score: int = -15) -> Optional[Rect]:
            """Best free position: Vastu zone score of the item first, then nearness to `corner`."""
            target = self.corner_point(corner)
            best = None
            for rank, side in enumerate(strip_order):
                strip = strips.get(side)
                if not strip:
                    continue
                for (ww, hh) in ([(w, h), (h, w)] if rotate and abs(w - h) > 1e-6 else [(w, h)]):
                    for rect in self._free_positions(ww, hh, strip, target, occupied):
                        cx, cy = rect[0] + rect[2] / 2, rect[1] + rect[3] / 2
                        zone = plot_eng.get_zone_for_position(cx, cy).value
                        sc = plot_eng.get_room_score(cat, zone) if self.vastu else 0
                        dist = abs(cx - target[0]) + abs(cy - target[1])
                        key_ = sc - 0.25 * dist - 0.5 * rank
                        if best is None or key_ > best[0]:
                            best = (key_, sc, rect)
            if best is None or (self.vastu and best[1] < min_score):
                return None
            rect = best[2]
            occupied.append(rect)
            items.append(SiteItem(key, name, cat, rect, color))
            return rect

        # Compass corner → candidate strips touching it
        def strips_near(corner: str) -> List[str]:
            sides = [self.side_of[c] for c in corner if c in self.side_of]
            rest = [s for s in ("top", "bottom", "left", "right") if s not in sides]
            return sides + rest

        front_corner = "NW" if "NW" in self._front_corners() else "SE"

        # 1. External terrace staircase
        if ext_stair is not None and site.ext_stair_side:
            side = site.ext_stair_side
            strip = strips.get(side)
            if strip:
                L, Wd = ext_stair.footprint_l, ext_stair.flight_width
                along_x = side in ("top", "bottom")
                w, h = (L, Wd) if along_x else (Wd, L)
                # hug the building face
                # Two runs along a front / back face: the whole strip (the foot may pass the house corner)
                # and the house face itself (both ends alongside the house, so either can be the top —
                # its bridge crosses onto the roof). The better for Vastu is kept: the zone the stair
                # stands in, and a climb rising towards the South / West (clockwise).
                runs = [(strip[0], strip[2])]
                if side in ("top", "bottom") and fw + 0.23 >= w - 1e-6:
                    # the house face, kept inside the compound wall when the house stands on the boundary
                    f0, f1 = max(fx - 0.115, MARGIN_PLOT / 2), min(fx + fw + 0.115, self.W - MARGIN_PLOT / 2)
                    face = [(f0, f1 - f0)] if f1 - f0 >= w - 1e-6 else []
                    runs = face + ([] if side == "top" else runs)
                toward = self.corner_point("SW")
                if side == "top" and site.door_exclusion:
                    toward = (0.0 if site.door_exclusion[0] <= 0.05 else self.W, self.H)
                best = None
                for x0, span in runs:
                    # 0.15 m off the house face, but never over the compound wall (a tight side yard
                    # gives up part of that gap)
                    lo_x, hi_x = MARGIN_PLOT / 2, self.W - MARGIN_PLOT / 2
                    lo_y = MARGIN_PLOT / 2
                    if side == "bottom":
                        strip_rect = (x0, max(lo_y, fy - 0.115 - 0.15 - h), span, h)
                    elif side == "top":        # alongside the road face, at the end kept off the main door
                        strip_rect = (x0, fy + fd + 0.115 + 0.15, span, h)
                    elif side == "left":
                        strip_rect = (max(lo_x, fx - 0.115 - 0.15 - w), strip[1], w, strip[3])
                    else:
                        strip_rect = (min(hi_x - w, fx + fw + 0.115 + 0.15), strip[1], w, strip[3])
                    cand = self._slide_place(w, h, strip_rect, toward, occupied)
                    if cand is None:
                        continue
                    key = self.ext_stair_vastu(cand, site.footprint) if self.vastu else 0.0
                    if best is None or key > best[0] + 1e-6:
                        best = (key, cand)
                rect = best[1] if best else None
                if rect:
                    occupied.append(rect)
                    items.append(SiteItem("ext_stair", "Terrace Staircase (External)", "STAIRS", rect, "#94A3B8"))
                else:
                    notes.append("External terrace stair could not be fitted; provide a ladder/hatch access.")

        # 2. Car parking
        car_cfg = self._enabled("car_parking")
        car_rect = None
        if car_cfg and site.car_mode in ("front", "front_parallel", "left", "right"):
            slots = max(1, int(car_cfg.get("slots", 1) or 1))
            if site.car_mode in ("front", "front_parallel"):
                car_rect = None
                for w, h in ((slots * 2.6, 5.0), (slots * 5.5, 2.6)):
                    car_rect = put("car", f"Car Parking ({slots} Bays)", "PARKING", w, h, ["top"],
                                   front_corner, "#334155", rotate=False, min_score=-25)
                    if car_rect:
                        break
            else:
                w, h = 2.6, slots * 5.2
                car_rect = put("car", f"Car Parking ({slots} Bays)", "PARKING", w, h, [site.car_mode],
                               front_corner, "#334155", rotate=False, min_score=-25)
            if car_rect:
                zone = self._zone_of(car_rect)
                notes.append(f"Car parking: {slots} bay(s) of 2.6 × 5.0 m in the {zone} of the site "
                             "(NBC parking bay; Vastu prefers NW/SE for vehicles).")
            else:
                notes.append("Car parking could not be placed clear of the entrance path.")

        # 2b. The car gate: sized for the car's swept path from the road, clear of the walking gate
        car_gate, drive = None, None
        lanes, mode, parked = [], None, []
        door_mid = (door_x0 + door_x1) / 2
        porch = (door_mid - 1.2, fy + fd, door_mid + 1.2, fy + fd + 0.115 + 2.2)
        if car_rect:
            x, y, w, h = car_rect
            slots = max(1, int(car_cfg.get("slots", 1) or 1))
            throat = max(0.0, self.H - (y + h))
            if site.car_mode in ("front", "front_parallel") and h >= 4.5:
                per = w / slots
                for i in range(slots):
                    ax = x + (i + 0.5) * per
                    lanes.append({"axis": ax, "throat": throat, "lo": ax - per / 2, "hi": ax + per / 2,
                                  "park_y": y + h / 2, "stop_y": y + h})
                mode = "front"
            elif site.car_mode in ("front", "front_parallel"):
                lanes.append({"axis": x + w / 2, "throat": throat, "lo": x, "hi": x + w,
                              "park_y": y + h / 2, "stop_y": y + h})
                mode = "parallel"
            else:
                lanes.append({"axis": x + w / 2, "throat": throat, "lo": x, "hi": x + w,
                              "park_y": y + h - 2.6, "stop_y": y + h})
                mode = "side"
            parked = [(x, y, x + w, y + h)]
        elif vehicle and site.car_mode == "stilt":
            info = self.stilt_info or _gates.stilt_bays(site.footprint, [], [], [], n_cars, designer.car_key)
            self.stilt_info = info
            front = info["front"]
            for b in (b for b in info["bays"] if not b[4]):
                ax = (b[0] + b[2]) / 2
                # straight between the columns at the front edge: no turning under the building
                lanes.append({"axis": ax, "throat": max(0.0, self.H - front), "lo": b[5], "hi": b[6],
                              "park_y": (b[1] + b[3]) / 2, "stop_y": front, "exact": True})
            if lanes:
                mode = "stilt"
                parked = [(fx - 0.115, fy - 0.115, fx + fw + 0.115, fy + fd + 0.115)]
        if not lanes and designer.height_m >= _gates.FIRE_H:
            mode = "fire"                             # a tall building: the fire engine's gate all the same
            lanes = [{"axis": self.W / 2, "throat": 0.0, "lo": 0.0, "hi": self.W, "park_y": fy + fd,
                      "stop_y": fy + fd}]
        if lanes:
            blocked = [(r[0], r[1], r[0] + r[2], r[1] + r[3]) for r in (getattr(site, "reserved", None) or [])]
            blocked += [(it.rect[0], it.rect[1], it.rect[0] + it.rect[2], it.rect[1] + it.rect[3])
                        for it in items if it.key == "ext_stair"]
            house = (fx - 0.115, fy - 0.115, fx + fw + 0.115, fy + fd + 0.115)
            car_gate, drive, reserve = designer.vehicle_gate(lanes, walk, blocked, parked + [porch], house, mode)
            occupied.extend(reserve)

        # 3. Security cabin beside the vehicle gate (NW / SE)
        if self._enabled("security_cabin"):
            if not put("security", "Security Cabin", "SECURITY", 1.8, 1.8, ["top", "left", "right"],
                       front_corner, "#6D28D9", min_score=-25):
                notes.append("Security cabin skipped: front setback too shallow (needs ≥ 1.8 m).")

        # 4. Two-wheeler parking
        bike_cfg = self._enabled("bike_parking")
        if bike_cfg:
            n = max(2, int(bike_cfg.get("slots", 2) or 2))
            per_row = max(1, int(min(n, 8)))
            rows = math.ceil(n / per_row)
            w, h = per_row * 0.9, rows * 2.0
            if not put("bike", f"Bike Parking ({n} Bikes)", "BIKE_PARKING", w, h,
                       ["top", "left", "right", "bottom"], front_corner, "#475569", min_score=-25):
                per_row = max(1, math.ceil(n / 2))
                if not put("bike", f"Bike Parking ({n} Bikes)", "BIKE_PARKING", per_row * 0.9, 4.0,
                           ["top", "left", "right", "bottom"], front_corner, "#475569", min_score=-25):
                    # Shallow front yard (small plots): scooters parked parallel to the wall,
                    # two abreast (0.75 m each), 2.0 m long
                    rows = 2 if n > 1 else 1
                    if not put("bike", f"Bike Parking ({n} Bikes, parallel)", "BIKE_PARKING",
                               math.ceil(n / rows) * 2.0, rows * 0.75,
                               ["top", "left", "right", "bottom"], front_corner, "#475569", min_score=-25):
                        notes.append("Two-wheeler parking skipped: no open strip ≥ 1.5 m deep.")

        # 5. Underground sump in Ishan (NE)
        if self._enabled("water_tank"):
            litres = self.members * 135 * 1.5 * (self.floors if self.multi_level else 1)
            vol = litres / 1000.0
            side_len = max(1.5, _snap(math.sqrt(vol / 1.5) + 0.3))
            rect = put("sump", f"Water Sump ({vol:.1f} kL)", "WATER_TANK", side_len, side_len,
                       strips_near("NE"), "NE", "#0284C7")
            if rect:
                notes.append(f"Underground sump {vol:.1f} kL = {self.members} persons × 135 lpcd × 1.5 days "
                             f"(CPHEEO); {side_len:.2f} m square × 1.5 m water depth, placed in the "
                             f"{self._zone_of(rect)} corner.")
            else:
                notes.append("Underground sump could not be placed in the open area.")

        # 6. Generator in Agni (SE)
        if self._enabled("generator_room"):
            kva = self._genset_kva(fw * fd)
            gw, gh = (1.8, 1.4) if kva <= 10 else (2.4, 1.5) if kva <= 25 else (3.0, 1.8)
            rect = put("genset", f"Generator ({kva:g} kVA)", "GENERATOR", gw, gh, strips_near("SE"), "SE", "#B45309")
            if rect:
                zone = self._zone_of(rect)
                notes.append(f"Backup genset {kva:g} kVA (≈40 W/m² connected load × floors, 0.6 demand factor"
                             f"{', + 5.5 kW lift motor' if self.components.get('_has_lift') else ''}); "
                             f"acoustic enclosure in the {zone}{' (Agni)' if zone == 'SE' else ''} "
                             "part of the site, ≥ 1 m from openings.")
            else:
                notes.append("Generator not placed: no open strip in an acceptable Vastu zone (SE/NW/S) "
                             "can take a 1.5 m enclosure — use a terrace-mounted silent genset.")

        # 7. Swimming pool in N / E / NE
        pool_cfg = self._enabled("swimming_pool")
        if pool_cfg:
            pw, pl = self._pool_dims(pool_cfg)
            order = [self.side_of["N"], self.side_of["E"], self.side_of["W"], self.side_of["S"]]
            rect = put("pool", "Swimming Pool", "POOL", pl, pw, order, "NE", "#0369A1")
            if rect:
                notes.append(f"Swimming pool {pw:g} × {pl:g} m in the {self._zone_of(rect)} open area "
                             "(water bodies belong to Ishan/North/East).")
            else:
                notes.append("Swimming pool skipped: no open strip large enough.")

        # 8. Garden: largest free rectangles in N and E strips
        garden_cfg = self._enabled("garden")
        if garden_cfg:
            target = self.W * self.H * max(5, int(garden_cfg.get("pct", 10) or 10)) / 100.0
            got = 0.0
            for idx, c in enumerate(("N", "E", "W", "S")):
                if got >= target * 0.95:
                    break
                side = self.side_of[c]
                strip = strips.get(side)
                if not strip or min(strip[2], strip[3]) < 1.0:
                    continue
                rect = self._largest_free(strip, occupied, min_side=1.2)
                if not rect:
                    continue
                need = target - got
                x, y, w, h = rect
                if w * h > need:
                    if w >= h:
                        w = max(1.2, _snap(need / h))
                        # keep the garden towards the NE end of the strip
                        nx, _ = self.corner_point("NE")
                        if nx > 0:
                            x = rect[0] + rect[2] - w
                    else:
                        h = max(1.2, _snap(need / w))
                        _, ny = self.corner_point("NE")
                        if ny > 0:
                            y = rect[1] + rect[3] - h
                rect = (_snap(x), _snap(y), _snap(w), _snap(h))
                occupied.append(rect)
                label = "Landscaped Garden" if not got else f"Landscaped Garden ({c})"
                items.append(SiteItem(f"garden_{c}", label, "GARDEN", rect, "#14532D"))
                got += rect[2] * rect[3]
            if got > 0:
                notes.append(f"Landscaped garden {got:.0f} m² ({got / (self.W * self.H) * 100:.0f}% of plot) "
                             "laid out in the North/East open strips; keep tall trees to the S/W.")
            else:
                notes.append("Garden skipped: no open strip ≥ 1.2 m wide is left.")

        # the gates, left to right: G1, G2 …
        specs = [g for g in (walk, car_gate) if g is not None]
        if car_gate is not None and car_gate.kind == "vehicle_wicket":
            specs = [car_gate]                       # people use the wicket door in the car gate
        specs.sort(key=lambda g: g.x0)
        for i, g in enumerate(specs, 1):
            g.tag = f"G{i}"
        gates += [(g.x0, g.x1) for g in specs]
        site.gate_specs = [g.as_dict() for g in specs]
        site.drive = drive
        site.road = designer.road
        notes.append(_gates.summary(site.gate_specs, designer.road))

        site.items = items
        site.gates = gates
        return site

    def _front_corners(self) -> List[str]:
        top = self.sides["top"]
        left, right = self.sides["left"], self.sides["right"]
        from floorplan_generator.vastu_engine import compose_zone
        return [compose_zone(top, left), compose_zone(top, right)]

    def _zone_of(self, rect: Rect) -> str:
        from floorplan_generator.vastu_engine import VastuEngine
        eng = VastuEngine.for_facing(self.W, self.H, self.facing)
        return eng.get_zone_for_position(rect[0] + rect[2] / 2, rect[1] + rect[3] / 2).value

    def _genset_kva(self, footprint_area: float) -> float:
        load_kw = footprint_area * 0.040 * self.floors
        if self.components.get("_has_lift"):
            load_kw += 5.5
        kva = load_kw * 0.6 / 0.8
        for std in (5, 7.5, 10, 15, 20, 25, 30, 40, 62.5, 82.5, 100):
            if kva <= std:
                return std
        return 125

    @staticmethod
    def _largest_free(strip: Rect, occupied: List[Rect], min_side: float) -> Optional[Rect]:
        """Largest free full-depth sub-rectangle of a strip between occupied intervals."""
        sx, sy, sw, sh = strip
        along_x = sw >= sh
        lo, hi = (sx, sx + sw) if along_x else (sy, sy + sh)
        blocks = []
        for o in occupied:
            if _overlap(o, strip):
                blocks.append((o[0], o[0] + o[2]) if along_x else (o[1], o[1] + o[3]))
        blocks.sort()
        best = None
        cur = lo
        for b0, b1 in blocks + [(hi, hi)]:
            if b0 - cur >= min_side:
                seg = (cur, b0)
                length = seg[1] - seg[0]
                if best is None or length > best[1] - best[0]:
                    best = seg
            cur = max(cur, b1)
        if not best:
            return None
        gap = 0.1
        if along_x:
            return (best[0] + gap, sy, best[1] - best[0] - 2 * gap, sh)
        return (sx, best[0] + gap, sw, best[1] - best[0] - 2 * gap)
