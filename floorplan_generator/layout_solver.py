"""
Band-Spine Layout Solver (BSL)
==============================
Places the space program inside the building footprint.

Topology (plan frame, road side at the top):

      ┌───────────── front band ─────────────┐   living + front suites (main door here)
      ├──── private band (deep plots only) ──┤   bedrooms + a passage linking living ↔ spine
      ├─[cell]──── circulation spine ──[cell]┤   dining & family lounge / passage (+ small rooms
      │                                      │   or the stair in the end cells)
      └───────────── rear band ──────────────┘   suites, kitchen, services

Every room in the front, private and rear bands touches the spine (or living),
so every room gets a door to circulation. Bands run full-width, so walls line up
into a clean structural grid.

The arrangement (which group goes to which band, left→right order, which side a
bath/utility sub-column sits on, which small rooms go to the spine cells) is
optimised by deterministic local search against a weighted objective:

   Vastu Purusha Mandala score (building mandala, role-aware, entrance pada,
   Brahmasthan)  +  functional score (target areas, proportions)
   −  penalties (no access, NBC minimum area/width, no daylight for habitable
      rooms, kitchen not beside dining, toilet sharing a wall with kitchen or
      pooja)  +  plumbing-stack bonus (wet rooms sharing walls).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import copy
import math
import random

from floorplan_generator.space_program import SpaceProgram, RoomGroup, ProgramRoom
from floorplan_generator.vastu_engine import (
    VastuEngine, ROOM_IMPORTANCE, plan_sides_for_facing, compose_zone, pada_for_point,
)

GRID = 0.05
TOL = 0.02


def _snap(v: float) -> float:
    return round(round(v / GRID) * GRID, 3)


# =============================================================================
# Placed geometry
# =============================================================================

@dataclass
class Placed:
    key: str
    name: str
    category: str
    role: str
    x: float
    y: float
    w: float
    h: float
    color: str
    parent: Optional[str] = None
    circulation: bool = False
    habitable: bool = False
    wet: bool = False
    target_area: float = 0.0
    min_area: float = 0.0
    min_width: float = 0.0
    group: str = ""
    band: str = ""

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def top(self) -> float:
        return self.y + self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    @property
    def area(self) -> float:
        return self.w * self.h


def shared_edge(a: Placed, b: Placed) -> Optional[Tuple[str, float, float, float]]:
    """(side of a, start, end, wall coordinate) for the wall shared by a and b."""
    if abs(a.top - b.y) < TOL:
        lo, hi = max(a.x, b.x), min(a.right, b.right)
        if hi - lo > TOL:
            return ("N", lo, hi, a.top)
    if abs(a.y - b.top) < TOL:
        lo, hi = max(a.x, b.x), min(a.right, b.right)
        if hi - lo > TOL:
            return ("S", lo, hi, a.y)
    if abs(a.right - b.x) < TOL:
        lo, hi = max(a.y, b.y), min(a.top, b.top)
        if hi - lo > TOL:
            return ("E", lo, hi, a.right)
    if abs(a.x - b.right) < TOL:
        lo, hi = max(a.y, b.y), min(a.top, b.top)
        if hi - lo > TOL:
            return ("W", lo, hi, a.x)
    return None


DOOR_WIDTH = {
    "BEDROOM": 0.9, "KITCHEN": 0.9, "BATHROOM": 0.75, "POOJA": 0.8, "STORE": 0.75,
    "STUDY": 0.9, "SERVANT": 0.8, "STAIRS": 1.0, "LIFT": 0.9, "UTILITY": 0.75,
    "CORRIDOR": 0.75, "DINING": 1.0, "LIVING": 1.0,
}


def door_width_for(p: Placed) -> float:
    if p.role == "ELDER_BEDROOM":
        return 1.0
    return DOOR_WIDTH.get(p.category, 0.8)


# =============================================================================
# Solver state
# =============================================================================

@dataclass
class State:
    mode: str                                  # 'std' | 'deep' | 'corridor'
    bands: Dict[str, List[str]]                # band -> ordered group keys
    cells: Dict[str, List[str]]                # 'L' / 'R' -> group keys
    flip: Dict[str, bool] = field(default_factory=dict)

    def clone(self) -> "State":
        return State(self.mode, {k: list(v) for k, v in self.bands.items()},
                     {k: list(v) for k, v in self.cells.items()}, dict(self.flip))


@dataclass
class LayoutResult:
    rooms: List[Placed]
    state: State
    score: float
    vastu_score: float
    door: Tuple[float, float]                  # main door x-range on the top facade
    band_edges: List[float]                    # y coordinates of band boundaries
    penalties: Dict[str, float]
    mode: str


SMALL_KINDS = ("single",)
CELL_MAX_ROOMS = 2


class BandSpineSolver:
    def __init__(self, program: SpaceProgram, footprint: Tuple[float, float, float, float],
                 facing: str, vastu_enabled: bool = True, seed: int = 7):
        self.p = program
        self.fx, self.fy, self.fw, self.fd = footprint
        self.facing = facing
        self.vastu_enabled = vastu_enabled
        self.groups: Dict[str, RoomGroup] = {g.key: g for g in program.groups}
        self.living_group = RoomGroup(key="living", kind="single", rooms=[program.living],
                                      bands=("front",), growth=3.0, min_depth=3.3)
        self.passage_group = RoomGroup(
            key="passage", kind="passage",
            rooms=[ProgramRoom(key="passage", name="Passage / Foyer", category="CORRIDOR",
                               target_area=3.5, min_area=1.6, min_width=1.2, color="#E8E8E8",
                               circulation=True)],
            bands=("private",), growth=0.1, min_depth=1.0)
        self.engine = VastuEngine.for_facing(self.fw, self.fd, facing, self.fx, self.fy)
        self.sides = plan_sides_for_facing(facing)
        self.rng = random.Random(seed)
        self._cache: Dict[str, Tuple[float, Optional[LayoutResult]]] = {}
        self._overflow = 0.0
        self._bw_cache: Dict = {}
        self._gw_cache: Dict = {}
        self._door_cache: Dict = {}

    # ------------------------------------------------------------- modes
    def candidate_modes(self) -> List[str]:
        modes = []
        D = self.fd
        if D >= 3.3 + 2.7 + 3.0 - 0.05:
            modes.append("std")
        if D >= 12.0 and D / max(self.fw, 1.0) >= 1.05 and len(self.p.groups) >= 4:
            modes.append("deep")
        if D >= 3.0 + 1.05 + 2.7 - 0.05:
            modes.append("corridor")
        return modes or ["corridor"]

    # ------------------------------------------------------- construction
    def _vastu_role(self, r: ProgramRoom) -> str:
        if r.role in ("MASTER_BEDROOM", "CHILD_BEDROOM", "GUEST_BEDROOM", "ELDER_BEDROOM", "STAIRS", "LIFT"):
            return r.role
        return r.category

    def _nominal_depths(self, mode: str) -> Dict[str, float]:
        if mode == "corridor":
            d = (self.fd - 1.1) / 2
            return {"front": d, "rear": d}
        if mode == "deep":
            d = (self.fd - 2.7) / 3
            return {"front": d, "private": d, "rear": d}
        d = (self.fd - 2.7) / 2
        return {"front": d, "rear": d}

    def _slot_zone(self, band: str, lat: int) -> str:
        vert = {"front": self.sides["top"], "rear": self.sides["bottom"]}.get(band)
        horiz = {0: self.sides["left"], 2: self.sides["right"]}.get(lat)
        return compose_zone(vert, horiz)

    def construct(self, mode: str, temperature: float = 0.0) -> State:
        """
        Randomised greedy construction. Groups are placed in order of Vastu
        importance; each picks the (band, lateral slot) with the best zone score
        among slots that still have width capacity. temperature=0 is pure greedy.
        """
        depths = self._nominal_depths(mode)
        non_spine = list(depths.keys())
        bands: Dict[str, List[str]] = {b: [] for b in non_spine}
        used: Dict[str, float] = {b: 0.0 for b in non_spine}
        lateral: Dict[str, float] = {"living": 1.0}
        cells: Dict[str, List[str]] = {"L": [], "R": []}
        cell_used = 0.0
        flip: Dict[str, bool] = {}

        def minw(g: RoomGroup, b: str) -> float:
            if self._is_small(g):
                return max(1.2, g.primary.min_width)
            gw = self._group_width(g, max(2.4, depths[b]))
            return gw[0] if gw else 1e9

        bands["front"].append("living")
        used["front"] += minw(self.living_group, "front")
        if mode == "deep":
            bands["private"].append("passage")
            used["private"] += 1.2
            lateral["passage"] = 1.0

        order = sorted(self.p.groups, key=lambda g: -self._importance(g) - self.rng.random() * temperature)
        for g in order:
            role = self._vastu_role(g.primary)
            options = []
            for b in non_spine:
                if b not in g.bands:
                    continue
                need = minw(g, b)
                if used[b] + need <= self.fw + 1e-6:
                    for lat in (0, 1, 2):
                        sc = self.engine.get_room_score(role, self._slot_zone(b, lat))
                        options.append((sc, b, lat, need))
                # A small room can stack onto another lone small room in the band at no extra width
                if self._is_small(g):
                    for k in bands[b]:
                        gk = self._group(k)
                        partners = [x for x in bands[b] if abs(lateral.get(x, 1.0) - lateral.get(k, 1.0)) < 0.05]
                        if self._is_small(gk) and len(partners) == 1:
                            clash = {gk.primary.category, g.primary.category} == {"BATHROOM", "POOJA"}
                            lat_k = lateral.get(k, 1.0)
                            sc = self.engine.get_room_score(role, self._slot_zone(b, int(round(lat_k)))) - (8 if clash else 0)
                            options.append((sc, b, ("stack", k), 0.0))
            can_cell = ("cell" in g.bands and mode != "corridor" and not (g.kind == "core" and g.lift))
            if can_cell:
                w_cell = (g.stair.footprint_l if g.kind == "core" else
                          max(g.primary.min_width, min(2.4, g.primary.target_area / 2.7)))
                for c, lat in (("L", 0), ("R", 2)):
                    has_core = any(self._group(x).kind == "core" for x in cells[c])
                    limit = 1 if (g.kind == "core" or has_core) else CELL_MAX_ROOMS
                    if len(cells[c]) >= limit or (g.kind == "core" and cells[c]):
                        continue
                    extra = w_cell if not cells[c] else 0.0
                    if cell_used + extra > self.fw - 2.4 - 1e-6:
                        continue
                    sc = self.engine.get_room_score(role, self._slot_zone("spine", lat))
                    options.append((sc + 1.0, "cell" + c, lat, extra))
            if not options:
                # no capacity anywhere: overflow into the allowed band with most spare width
                allowed = [b for b in non_spine if b in g.bands] or ["rear"]
                b = min(allowed, key=lambda b: used[b])
                bands[b].append(g.key)
                used[b] += minw(g, b)
                lateral[g.key] = 1.0
                continue
            if temperature > 0:
                options = [(sc + self.rng.gauss(0, 6.0 * temperature), b, lat, need)
                           for sc, b, lat, need in options]
            sc, b, lat, need = max(options, key=lambda o: o[0])
            if b.startswith("cell"):
                c = b[-1]
                cells[c].append(g.key)
                cell_used += need
            elif isinstance(lat, tuple):
                partner = lat[1]
                bands[b].append(g.key)
                lateral[g.key] = lateral.get(partner, 1.0) + 0.001
            else:
                bands[b].append(g.key)
                used[b] += need
                lateral[g.key] = lat + self.rng.random() * 0.2
            # put a bath / utility sub-column on the side facing W/NW (wet zones)
            west_left = self.sides["left"] in ("W", "N")
            flip[g.key] = (not west_left) if temperature == 0 else self.rng.random() < 0.5
        for b in bands:
            bands[b].sort(key=lambda k: lateral.get(k, 1.0))
        return State(mode=mode, bands=bands, cells=cells, flip=flip)

    def initial_state(self, mode: str, variant: int = 0) -> State:
        return self.construct(mode, 0.0 if variant == 0 else 0.5 + 0.25 * variant)

    def _importance(self, g: RoomGroup) -> float:
        r = g.primary
        return ROOM_IMPORTANCE.get(r.role or r.category, ROOM_IMPORTANCE.get(r.category, 1.0))

    @staticmethod
    def _is_small(g: RoomGroup) -> bool:
        return g.kind == "single" and g.key not in ("living", "passage") and \
            g.primary.category in ("POOJA", "BATHROOM", "STORE")

    def _columns(self, keys: List[str]) -> List[List[str]]:
        """Consecutive small rooms (pooja / bath / store) share one stacked column."""
        cols: List[List[str]] = []
        for k in keys:
            g = self._group(k)
            if self._is_small(g) and cols and len(cols[-1]) == 1 and self._is_small(self._group(cols[-1][0])):
                cols[-1].append(k)
            else:
                cols.append([k])
        return cols

    def _small_col_width(self, col: List[str], d: float) -> Tuple[float, float]:
        rs = [self._group(k).primary for k in col]
        mn = max(max(r.min_width for r in rs), 1.2, sum(r.min_area for r in rs) / max(d, 0.1))
        tw = max(mn, min(2.4, max(1.5, sum(r.target_area for r in rs) / max(1.2, min(d, 2.2 * len(rs))))))
        return mn, tw

    def _place_small_col(self, col: List[str], x: float, y: float, w: float, d: float,
                         spine_above: bool, band: str) -> List[Placed]:
        """Stack small rooms at the outer end of the column; a Foyer/Passage fills the spine end."""
        prio = {"POOJA": 0, "BATHROOM": 1, "STORE": 2}
        rs = sorted([self._group(k).primary for k in col], key=lambda r: prio.get(r.category, 3))
        depths = [min(d, max(1.2, r.min_area / w, r.target_area / w)) for r in rs]
        if sum(depths) > d:
            f = d / sum(depths)
            depths = [dd * f for dd in depths]
        filler = d - sum(depths)
        if filler < 1.0:
            depths[-1] += filler
            filler = 0.0
        depths = [_snap(dd) for dd in depths]
        out: List[Placed] = []
        gkey = self._group(col[0]).key
        if spine_above:          # rear band: outer end is the bottom
            cur = y
            for r, dd in zip(rs, depths):
                out.append(self._mk_small(r, x, cur, w, dd, gkey, band))
                cur = round(cur + dd, 3)
            inner_top = y + d
            if filler > 0:
                out.append(self._mk_filler(col[0], x, cur, w, inner_top - cur, band))
            else:
                out[-1].h = round(inner_top - out[-1].y, 3)
        else:                    # front/private band: outer end is the top
            cur = y + d
            for r, dd in zip(rs, depths):
                cur = round(cur - dd, 3)
                out.append(self._mk_small(r, x, cur, w, dd, gkey, band))
            if filler > 0:
                out.append(self._mk_filler(col[0], x, y, w, cur - y, band))
            else:
                last = out[-1]
                last.h = round(last.top - y, 3)
                last.y = round(y, 3)
        return out

    def _band_min_width(self, cols: List[List[str]], d: float) -> float:
        key = (tuple(tuple(c) for c in cols), round(d, 3))
        hit = self._bw_cache.get(key)
        if hit is not None:
            return hit
        tot = self._band_min_width_raw(cols, d)
        self._bw_cache[key] = tot
        return tot

    def _band_min_width_raw(self, cols: List[List[str]], d: float) -> float:
        tot = 0.0
        for col in cols:
            g = self._group(col[0])
            if self._is_small(g):
                tot += self._small_col_width(col, d)[0]
            else:
                gw = self._group_width(g, d)
                tot += gw[0] if gw else 1e3
        return tot

    @staticmethod
    def _mk_small(r: ProgramRoom, x, y, w, h, gkey, band) -> Placed:
        return Placed(key=r.key, name=r.name, category=r.category, role=r.role,
                      x=round(x, 3), y=round(y, 3), w=round(w, 3), h=round(h, 3), color=r.color,
                      wet=r.wet, target_area=r.target_area, min_area=r.min_area,
                      min_width=r.min_width, group=gkey, band=band)

    @staticmethod
    def _mk_filler(k, x, y, w, h, band) -> Placed:
        return Placed(key=f"filler_{k}", name="Foyer" if band == "front" else "Passage",
                      category="CORRIDOR", role="CORRIDOR", x=round(x, 3), y=round(y, 3),
                      w=round(w, 3), h=round(h, 3), color="#E8E8E8", circulation=True,
                      min_width=0.9, group=k, band=band)

    def _group(self, key: str) -> RoomGroup:
        if key == "living":
            return self.living_group
        if key == "passage":
            return self.passage_group
        return self.groups[key]

    # ----------------------------------------------------------- geometry
    def _spine_depth(self, state: State) -> Optional[float]:
        if state.mode == "corridor":
            return 1.2 if self.p.elder_friendly else 1.05
        dining_w = max(2.4, self.fw * 0.55)
        ds = min(3.6, max(2.7, self.p.dining.target_area / dining_w))
        for side in ("L", "R"):
            for k in state.cells[side]:
                g = self._group(k)
                if g.kind == "core":
                    ds = max(ds, g.stair.footprint_w)
        return _snap(ds) if ds <= 3.9 else None

    def _group_width(self, g: RoomGroup, d: float) -> Optional[Tuple[float, float]]:
        """(min width, target width) of a group placed in a band of depth d (memoised)."""
        key = (g.key, len(g.rooms), g.kind, round(d, 3))
        if key in self._gw_cache:
            return self._gw_cache[key]
        val = self._group_width_raw(g, d)
        self._gw_cache[key] = val
        return val

    def _group_width_raw(self, g: RoomGroup, d: float) -> Optional[Tuple[float, float]]:
        r0 = g.primary
        if g.kind in ("single", "passage"):
            mn = max(r0.min_width, r0.min_area / d)
            return mn, max(mn, r0.target_area / d)
        if g.kind in ("suite", "servant"):
            sub = g.rooms[1]
            subw = 1.5 if r0.role in ("ELDER_BEDROOM", "MASTER_BEDROOM") else 1.4
            if g.kind == "servant":
                subw = 1.1
            subw = max(subw, sub.min_width)
            mn = max(r0.min_width, r0.min_area / d) + subw
            return mn, max(mn, r0.target_area / d + subw)
        if g.kind == "service":
            k, u = g.rooms[0], g.rooms[1]
            if d >= 3.6:
                ud = 1.35 if u.target_area < 3.2 else 1.5
                kd = d - ud
                mn = max(k.min_width, k.min_area / kd, u.min_width)
                return mn, max(mn, k.target_area / kd, u.target_area / ud)
            subw = 1.2
            mn = max(k.min_width, k.min_area / d) + subw
            return mn, max(mn, k.target_area / d + subw)
        if g.kind == "core":
            s = g.stair
            if d >= s.footprint_l - 0.05:
                ws = s.footprint_w
            elif d >= s.footprint_w - 0.05:
                ws = s.footprint_l
            else:
                return None
            if g.lift:
                if d < g.lift.shaft_d - 0.05:
                    return None
                ws += g.lift.shaft_w
            return ws, ws
        return None

    def _allocate(self, widths: List[Tuple[float, float, float]], total: float) -> Optional[List[float]]:
        """widths: (min, target, growth) → snapped widths summing exactly to total."""
        mins = [w[0] for w in widths]
        if sum(mins) > total + 1e-6:
            self._overflow += sum(mins) - total
            return None
        surplus = total - sum(mins)
        need = [max(0.0, w[1] - w[0]) for w in widths]
        out = list(mins)
        if sum(need) >= surplus and sum(need) > 0:
            out = [m + n * surplus / sum(need) for m, n in zip(mins, need)]
        else:
            out = [m + n for m, n in zip(mins, need)]
            rest = surplus - sum(need)
            gw = [w[2] for w in widths]
            if sum(gw) <= 0:
                gw = [1.0] * len(widths)
            out = [o + rest * g / sum(gw) for o, g in zip(out, gw)]
        # snap cumulative edges for exact tiling
        edges = [0.0]
        acc = 0.0
        for o in out[:-1]:
            acc += o
            edges.append(_snap(acc))
        edges.append(total)
        res = [edges[i + 1] - edges[i] for i in range(len(out))]
        if any(r < m - 0.051 for r, m in zip(res, mins)):
            return None
        return res

    def _place_group(self, g: RoomGroup, x: float, y: float, w: float, d: float,
                     spine_above: bool, flip: bool, band: str) -> List[Placed]:
        out: List[Placed] = []

        def mk(r: ProgramRoom, rx, ry, rw, rh, circ=None) -> Placed:
            return Placed(key=r.key, name=r.name, category=r.category, role=r.role,
                          x=round(rx, 3), y=round(ry, 3), w=round(rw, 3), h=round(rh, 3),
                          color=r.color, parent=r.parent,
                          circulation=r.circulation if circ is None else circ,
                          habitable=r.habitable, wet=r.wet, target_area=r.target_area,
                          min_area=r.min_area, min_width=r.min_width, group=g.key, band=band)

        if g.kind in ("single", "passage"):
            out.append(mk(g.primary, x, y, w, d))
            return out

        if g.kind in ("suite", "servant"):
            bed, bath = g.rooms[0], g.rooms[1]
            dress = g.rooms[2] if len(g.rooms) > 2 else None
            subw = 1.5 if bed.role in ("ELDER_BEDROOM", "MASTER_BEDROOM") else 1.4
            if g.kind == "servant":
                subw = 1.1
            subw = _snap(max(bath.min_width, min(subw, w - max(bed.min_width, 2.1))))
            bw_ = w - subw
            sx = x + bw_ if flip else x
            bx = x if flip else x + subw
            out.append(mk(bed, bx, y, bw_, d))
            bd = _snap(min(d, max(1.8, bath.target_area / subw)))
            if dress is not None and d - bd >= 1.2:
                # bath at the outer end (away from the spine), dress towards the spine
                if spine_above:
                    out.append(mk(bath, sx, y, subw, bd))
                    out.append(mk(dress, sx, y + bd, subw, d - bd))
                else:
                    out.append(mk(dress, sx, y, subw, d - bd))
                    out.append(mk(bath, sx, y + d - bd, subw, bd))
            else:
                out.append(mk(bath, sx, y, subw, d))
            return out

        if g.kind == "service":
            k, u = g.rooms[0], g.rooms[1]
            if d >= 3.6:
                ud = 1.35 if u.target_area < 3.2 else 1.5
                if spine_above:
                    out.append(mk(u, x, y, w, ud))
                    out.append(mk(k, x, y + ud, w, d - ud))
                else:
                    out.append(mk(k, x, y, w, d - ud))
                    out.append(mk(u, x, y + d - ud, w, ud))
            else:
                subw = 1.2
                ux = x + w - subw if flip else x
                kx = x if flip else x + subw
                out.append(mk(k, kx, y, w - subw, d))
                out.append(mk(u, ux, y, subw, d))
            return out

        if g.kind == "core":
            s = g.stair
            stair_r = g.rooms[0]
            lift_r = g.rooms[1] if len(g.rooms) > 1 else None
            lw = g.lift.shaft_w if (g.lift and lift_r) else 0.0
            ws = w - lw
            stx = x + lw if flip else x
            lx = x if flip else x + ws
            out.append(mk(stair_r, stx, y, ws, d))
            if lift_r is not None:
                ld = g.lift.shaft_d
                if d - ld >= 0.9:
                    duct = ProgramRoom(key=f"{g.key}_duct", name="Lift Machine / Duct", category="STORE",
                                       target_area=0, min_area=0, min_width=0, color="#CBD5E1",
                                       parent=None)
                    if spine_above:
                        out.append(mk(duct, lx, y, lw, d - ld))
                        out.append(mk(lift_r, lx, y + d - ld, lw, ld))
                    else:
                        out.append(mk(lift_r, lx, y, lw, ld))
                        out.append(mk(duct, lx, y + ld, lw, d - ld))
                else:
                    out.append(mk(lift_r, lx, y, lw, d))
            return out
        return out

    def build(self, state: State) -> Optional[Tuple[List[Placed], List[float]]]:
        ds = self._spine_depth(state)
        if ds is None:
            return None
        non_spine = ["rear", "front"] if state.mode != "deep" else ["rear", "private", "front"]
        D = self.fd - ds
        if D <= 0:
            return None

        # Band depths
        mins, targets = {}, {}
        for b in non_spine:
            keys = state.bands.get(b, [])
            if not keys:
                return None
            gs = [self._group(k) for k in keys]
            mn = max(g.min_depth for g in gs)
            for g in gs:
                if g.kind == "core":
                    mn = max(mn, min(g.stair.footprint_w, g.stair.footprint_l))
            area = sum(g.target_area for g in gs)
            if state.mode == "corridor" and b == "front":
                area += self.p.dining.target_area
            mins[b] = mn
            targets[b] = max(mn, area / self.fw)
        # Feasibility-aware minimum depth: deepen a band until its columns fit the width
        # (e.g. a kitchen stacks its utility only when the band is ≥ 3.6 m deep).
        for b in non_spine:
            cols = self._columns(state.bands[b])
            lo = mins[b]
            if self._band_min_width(cols, lo) > self.fw + 1e-6:
                if self._band_min_width(cols, D) > self.fw + 1e-6:
                    self._overflow += self._band_min_width(cols, D) - self.fw
                    return None
                hi = D
                while hi - lo > 0.05:          # min width never grows with depth
                    mid = _snap((lo + hi) / 2)
                    if mid in (lo, hi):
                        break
                    if self._band_min_width(cols, mid) > self.fw + 1e-6:
                        lo = mid
                    else:
                        hi = mid
                lo = hi
            mins[b] = lo
            targets[b] = max(targets[b], lo)
        if sum(mins.values()) > D + 1e-6:
            self._overflow += sum(mins.values()) - D
            return None
        scale = D / sum(targets.values())
        depth = {b: targets[b] * scale for b in non_spine}
        for _ in range(4):
            low = [b for b in non_spine if depth[b] < mins[b]]
            if not low:
                break
            fixed = sum(mins[b] for b in low)
            free = [b for b in non_spine if b not in low]
            rest = D - fixed
            tot = sum(targets[b] for b in free) or 1.0
            for b in low:
                depth[b] = mins[b]
            for b in free:
                depth[b] = targets[b] * rest / tot
        # Band stacking bottom→top: rear, spine, (private), front
        order = ["rear", "spine"] + (["private"] if state.mode == "deep" else []) + ["front"]
        y = self.fy
        band_y: Dict[str, Tuple[float, float]] = {}
        for b in order:
            d = ds if b == "spine" else depth[b]
            y0 = y
            y = _snap(y + d) if b != order[-1] else self.fy + self.fd
            band_y[b] = (round(y0, 3), round(y - y0, 3))
        rooms: List[Placed] = []

        # Non-spine bands
        for b in non_spine:
            by, bd = band_y[b]
            cols = self._columns(state.bands[b])
            specs = []
            for col in cols:
                g = self._group(col[0])
                if self._is_small(g):
                    mn, tw = self._small_col_width(col, bd)
                    specs.append((mn, tw, 0.05))
                    continue
                gw = self._group_width(g, bd)
                if gw is None:
                    return None
                specs.append((gw[0], gw[1], g.growth))
            widths = self._allocate(specs, self.fw)
            if widths is None:
                return None
            x = self.fx
            spine_above = (b == "rear")
            for col, w in zip(cols, widths):
                g = self._group(col[0])
                if self._is_small(g):
                    rooms.extend(self._place_small_col(col, x, by, w, bd, spine_above, b))
                    x += w
                    continue
                k = col[0]
                placed = self._place_group(g, x, by, w, bd, spine_above, state.flip.get(k, False), b)
                if state.mode == "corridor" and k == "living":
                    placed[0].name = "Living & Dining"
                    placed[0].target_area = self.p.living.target_area + self.p.dining.target_area
                rooms.extend(placed)
                x += w

        # Spine
        sy, sd = band_y["spine"]
        cell_rooms: List[Placed] = []
        cell_w = {"L": 0.0, "R": 0.0}
        if state.mode != "corridor":
            for side in ("L", "R"):
                keys = state.cells.get(side, [])
                if not keys:
                    continue
                gs = [self._group(k) for k in keys]
                if any(g.kind == "core" for g in gs):
                    if len(gs) > 1:
                        return None
                    g = gs[0]
                    if g.lift:
                        return None
                    w = g.stair.footprint_l if sd < g.stair.footprint_l else g.stair.footprint_w
                    w = _snap(w)
                    x = self.fx if side == "L" else self.fx + self.fw - w
                    r = g.rooms[0]
                    cell_rooms.append(Placed(key=r.key, name=r.name, category=r.category, role=r.role,
                                             x=round(x, 3), y=sy, w=w, h=sd, color=r.color,
                                             target_area=r.target_area, min_area=r.min_area,
                                             min_width=r.min_width, group=g.key, band="spine"))
                    cell_w[side] = w
                    continue
                if len(gs) > CELL_MAX_ROOMS:
                    return None
                tot = sum(g.target_area for g in gs)
                ys = sy
                wmax = 0.0
                parts = []
                for i, g in enumerate(gs):
                    h = sd if len(gs) == 1 else (_snap(sd * g.target_area / tot) if i == 0 else round(sy + sd - ys, 3))
                    if h < 1.2:
                        return None
                    r = g.primary
                    wneed = max(r.min_width, r.min_area / h, min(2.4, r.target_area / h))
                    wmax = max(wmax, wneed)
                    parts.append((g, r, ys, h))
                    ys = round(ys + h, 3)
                w = _snap(min(2.6, wmax))
                x = self.fx if side == "L" else self.fx + self.fw - w
                for g, r, yy, h in parts:
                    cell_rooms.append(Placed(key=r.key, name=r.name, category=r.category, role=r.role,
                                             x=round(x, 3), y=yy, w=w, h=h, color=r.color,
                                             wet=r.wet, target_area=r.target_area, min_area=r.min_area,
                                             min_width=r.min_width, group=g.key, band="spine"))
                cell_w[side] = w
        cw = self.fw - cell_w["L"] - cell_w["R"]
        if state.mode != "corridor" and cw < 2.4:
            self._overflow += 2.4 - cw
            return None
        if state.mode == "corridor":
            spine = Placed(key="dining", name="Passage", category="CORRIDOR", role="CORRIDOR",
                           x=self.fx, y=sy, w=self.fw, h=sd, color="#E8E8E8", circulation=True,
                           target_area=0, min_area=0, min_width=1.0, group="spine", band="spine")
        else:
            d = self.p.dining
            spine = Placed(key="dining", name=d.name, category="DINING", role="DINING",
                           x=round(self.fx + cell_w["L"], 3), y=sy, w=round(cw, 3), h=sd,
                           color="#FEF3C7", circulation=True, target_area=d.target_area,
                           min_area=d.min_area, min_width=d.min_width, group="spine", band="spine")
        rooms.extend(cell_rooms)
        rooms.append(spine)
        edges = sorted({v[0] for v in band_y.values()} | {self.fy + self.fd})
        return rooms, edges

    # -------------------------------------------------------------- scoring
    def _vastu_key(self, p: Placed) -> Optional[str]:
        if p.role in ("MASTER_BEDROOM", "CHILD_BEDROOM", "GUEST_BEDROOM", "ELDER_BEDROOM"):
            return p.role
        if "Dress" in p.name or "Wardrobe" in p.name or "Duct" in p.name:
            return None
        return p.category

    @staticmethod
    def _adjacency(rooms: List[Placed]) -> Dict[Tuple[int, int], float]:
        """Shared wall length for every touching pair, via coordinate buckets (≈ O(n))."""
        tops: Dict[float, List[int]] = {}
        bots: Dict[float, List[int]] = {}
        rights: Dict[float, List[int]] = {}
        lefts: Dict[float, List[int]] = {}
        for i, r in enumerate(rooms):
            tops.setdefault(round(r.top, 2), []).append(i)
            bots.setdefault(round(r.y, 2), []).append(i)
            rights.setdefault(round(r.right, 2), []).append(i)
            lefts.setdefault(round(r.x, 2), []).append(i)
        adj: Dict[Tuple[int, int], float] = {}
        for y, ia in tops.items():
            for i in ia:
                a = rooms[i]
                for j in bots.get(y, ()):
                    b = rooms[j]
                    L = min(a.right, b.right) - max(a.x, b.x)
                    if L > TOL:
                        adj[(i, j) if i < j else (j, i)] = L
        for x, ia in rights.items():
            for i in ia:
                a = rooms[i]
                for j in lefts.get(x, ()):
                    b = rooms[j]
                    L = min(a.top, b.top) - max(a.y, b.y)
                    if L > TOL:
                        adj[(i, j) if i < j else (j, i)] = L
        return adj

    def best_door(self, living: Placed) -> Tuple[float, float, int]:
        key = (round(living.x, 3), round(living.right, 3), round(living.top, 3))
        hit = self._door_cache.get(key)
        if hit is None:
            hit = self._best_door_raw(living)
            self._door_cache[key] = hit
        return hit

    def _best_door_raw(self, living: Placed) -> Tuple[float, float, int]:
        """Best 1.0 m main door position on the living room's front facade."""
        dw = 1.0
        lo = living.x + 0.45
        hi = living.right - 0.45 - dw
        if hi < lo:
            c = living.cx - dw / 2
            pada = pada_for_point(c + dw / 2, living.top, (self.fx, self.fy, self.fw, self.fd), self.facing)
            return c, c + dw, pada.score
        best = None
        x = lo
        while x <= hi + 1e-6:
            pada = pada_for_point(x + dw / 2, living.top, (self.fx, self.fy, self.fw, self.fd), self.facing)
            key = (pada.score if self.vastu_enabled else 0) - 0.05 * abs(x + dw / 2 - living.cx)
            if best is None or key > best[0]:
                best = (key, x, pada.score)
            x += 0.1
        return _snap(best[1]), _snap(best[1]) + dw, best[2]

    def score(self, rooms: List[Placed]) -> Tuple[float, float, Tuple[float, float], Dict[str, float]]:
        pen: Dict[str, float] = {}

        def add(k, v):
            pen[k] = pen.get(k, 0.0) + v

        adj = self._adjacency(rooms)

        def shared(i, j):
            return adj.get((i, j) if i < j else (j, i), 0.0)

        idx = {r.key: i for i, r in enumerate(rooms)}

        # Circulation connectivity
        circ = [i for i, r in enumerate(rooms) if r.circulation]
        start = idx.get("living")
        reach = set()
        if start is not None:
            stack = [start]
            while stack:
                c = stack.pop()
                if c in reach:
                    continue
                reach.add(c)
                for j in circ:
                    if j not in reach and shared(c, j) >= 1.0:
                        stack.append(j)
        for j in circ:
            if j not in reach:
                add("circulation", 30)

        # Access for every other room
        for i, r in enumerate(rooms):
            if r.circulation:
                continue
            dw = door_width_for(r)
            ok = False
            if r.parent and r.parent in idx:
                ok = shared(i, idx[r.parent]) >= dw + 0.1
                if not ok and r.parent:
                    # bath reached through the dress area of its bedroom
                    for j, o in enumerate(rooms):
                        if o.parent == r.parent and o.key != r.key and shared(i, j) >= dw + 0.1 and \
                                shared(j, idx[r.parent]) >= 0.85:
                            ok = True
                            break
            else:
                ok = any(shared(i, j) >= dw + 0.15 for j in reach)
                if not ok and r.category == "STORE":
                    k = idx.get("kitchen")
                    ok = k is not None and shared(i, k) >= dw + 0.1
            if not ok:
                add("access", 40)

        fp = (self.fx, self.fy, self.fw, self.fd)
        exterior = {}
        for i, r in enumerate(rooms):
            e = 0.0
            if abs(r.x - self.fx) < TOL:
                e += r.h
            if abs(r.right - (self.fx + self.fw)) < TOL:
                e += r.h
            if abs(r.y - self.fy) < TOL:
                e += r.w
            if abs(r.top - (self.fy + self.fd)) < TOL:
                e += r.w
            exterior[i] = e

        func_acc, func_n = 0.0, 0
        for i, r in enumerate(rooms):
            a = r.area
            short, long_ = min(r.w, r.h), max(r.w, r.h)
            if r.min_area and a < r.min_area - 0.05:
                add("nbc_area", 12 + 30 * (1 - a / r.min_area))
            if r.min_width and short < r.min_width - 0.05:
                add("nbc_width", 12 + 20 * (1 - short / r.min_width))
            ar = long_ / max(short, 0.1)
            limit = {"BEDROOM": 1.7, "LIVING": 2.1, "KITCHEN": 2.3, "STUDY": 1.9, "DINING": 3.2,
                     "POOJA": 2.4, "BATHROOM": 2.5}.get(r.category)
            if limit and ar > limit:
                add("proportion", 6 * (ar - limit) * (2.0 if r.category in ("BEDROOM", "LIVING") else 1.0))
            if r.habitable and exterior[i] < 0.9:
                add("daylight", 18 if r.category != "KITCHEN" else 6)
            if r.target_area > 0 and a > 1.6 * r.target_area + 1.0:
                add("oversize", 0.6 * (a - 1.6 * r.target_area))
            if r.category == "CORRIDOR" and r.circulation and r.name.startswith("Passage / Foyer") and a > 6.0:
                add("oversize", 2.0 * (a - 6.0))
            if r.target_area > 0:
                ratio = a / r.target_area
                sat = ratio if ratio <= 1 else max(0.0, 1 - (ratio - 1) * 0.35)
                func_acc += sat
                func_n += 1
        functional = 100.0 * func_acc / max(1, func_n)

        # Kitchen should open into dining (or living-dining)
        k = idx.get("kitchen")
        if k is not None:
            d = idx.get("dining")
            lv = idx.get("living")
            ok = (d is not None and rooms[d].category == "DINING" and shared(k, d) >= 1.0) or \
                 (lv is not None and rooms[lv].name == "Living & Dining" and shared(k, lv) >= 1.0)
            if not ok:
                add("kitchen_dining", 8)

        # Wet-wall rules and plumbing stacks
        bonus = 0.0
        for (i, j), L in adj.items():
            a, b = rooms[i], rooms[j]
            cats = {a.category, b.category}
            if L < 0.6:
                continue
            if "BATHROOM" in cats and "KITCHEN" in cats:
                add("toilet_kitchen", 3)
            if "BATHROOM" in cats and "POOJA" in cats:
                add("toilet_pooja", 8)
            if "STAIRS" in cats and "POOJA" in cats:
                add("stair_pooja", 2)
            if a.wet and b.wet:
                bonus += 1.5
        bonus = min(bonus, 6.0)

        # Vastu
        living = rooms[idx["living"]] if "living" in idx else None
        door = (0.0, 0.0)
        pada_score = 0
        if living is not None:
            d0, d1, pada_score = self.best_door(living)
            door = (d0, d1)
        placements, weights = [], []
        for r in rooms:
            key = self._vastu_key(r)
            if not key:
                continue
            z = self.engine.get_zone_for_position(r.cx, r.cy).value
            placements.append((key, z))
            weights.append(ROOM_IMPORTANCE.get(key, 1.0))
        # Brahmasthan: the central ninth must not host wet/heavy rooms
        bcx, bcy = self.fx + self.fw / 2, self.fy + self.fd / 2
        brahma_hit = 0
        for r in rooms:
            if r.x < bcx < r.right and r.y < bcy < r.top and r.category in (
                    "BATHROOM", "KITCHEN", "STAIRS", "LIFT", "STORE", "UTILITY"):
                brahma_hit += 1
        if self.vastu_enabled:
            # Classical anchors: master → SW, kitchen → SE (Agni), pooja → NE (Ishan)
            for (key, z) in placements:
                if self.engine.is_taboo(key, z):
                    add("taboo", 12)
                if key == "MASTER_BEDROOM":
                    add("anchor", 0 if z == "SW" else (5 if z in ("S", "W") else 14))
                elif key == "KITCHEN":
                    add("anchor", 0 if z == "SE" else (3 if z in ("NW", "E") else 9))
                elif key == "POOJA":
                    add("anchor", 0 if z == "NE" else (3 if z in ("E", "N") else 9))
                elif key in ("ELDER_BEDROOM", "CHILD_BEDROOM", "GUEST_BEDROOM") and z in ("NE", "SE"):
                    add("anchor", 6)
                elif key == "STAIRS" and z in ("NE", "N", "E", "CENTER"):
                    add("anchor", 6)
                elif key == "BATHROOM" and z == "SW":
                    add("anchor", 4)
        vastu = self.engine.compute_plan_score(placements, weights, extra_scores=[(pada_score, 3.0)])
        vastu = max(0.0, vastu - 8 * brahma_hit)

        total_pen = sum(pen.values())
        if self.vastu_enabled:
            score = 0.6 * vastu + 0.4 * functional - total_pen + bonus
        else:
            score = 0.15 * vastu + 0.85 * functional - total_pen + bonus
        return score, vastu, door, pen

    # --------------------------------------------------------------- search
    def evaluate(self, state: State) -> Tuple[float, Optional[LayoutResult]]:
        key = repr((state.mode, sorted(state.bands.items()), sorted(state.cells.items()),
                    sorted(state.flip.items())))
        if key in self._cache:
            return self._cache[key]
        self._overflow = 0.0
        built = self.build(state)
        if built is None:
            res = (-1000.0 - 50.0 * self._overflow - (200.0 if self._overflow == 0 else 0.0), None)
        else:
            rooms, edges = built
            s, v, door, pen = self.score(rooms)
            res = (s, LayoutResult(rooms=rooms, state=state.clone(), score=s, vastu_score=v,
                                   door=door, band_edges=edges, penalties=pen, mode=state.mode))
        self._cache[key] = res
        return res

    def _neighbours(self, st: State) -> List[State]:
        out = []
        bands = list(st.bands.keys())
        # swaps within a band (plain, and with both sub-columns flipped)
        for b in bands:
            lst = st.bands[b]
            for i in range(len(lst)):
                for j in range(i + 1, len(lst)):
                    s2 = st.clone()
                    s2.bands[b][i], s2.bands[b][j] = s2.bands[b][j], s2.bands[b][i]
                    out.append(s2)
                    s3 = s2.clone()
                    for k in (lst[i], lst[j]):
                        if k in self.groups:
                            s3.flip[k] = not s3.flip.get(k, False)
                    out.append(s3)
        # move a group to another band / position
        for b in bands:
            for k in st.bands[b]:
                if k in ("living",):
                    continue
                g = self._group(k)
                for b2 in bands:
                    if b2 == b or b2 not in g.bands:
                        continue
                    for pos in range(len(st.bands[b2]) + 1):
                        s2 = st.clone()
                        s2.bands[b].remove(k)
                        s2.bands[b2].insert(pos, k)
                        out.append(s2)
                # into a spine cell
                if "cell" in g.bands and st.mode != "corridor" and not (g.kind == "core" and g.lift):
                    for c in ("L", "R"):
                        if len(st.cells[c]) < (1 if g.kind == "core" else CELL_MAX_ROOMS) and \
                                not any(self._group(x).kind == "core" for x in st.cells[c]):
                            s2 = st.clone()
                            s2.bands[b].remove(k)
                            s2.cells[c].append(k)
                            out.append(s2)
        # out of / between cells
        for c in ("L", "R"):
            for k in st.cells[c]:
                g = self._group(k)
                other = "R" if c == "L" else "L"
                s2 = st.clone()
                s2.cells[c].remove(k)
                s2.cells[other].append(k)
                out.append(s2)
                for b in bands:
                    if b in g.bands:
                        for pos in range(len(st.bands[b]) + 1):
                            s2 = st.clone()
                            s2.cells[c].remove(k)
                            s2.bands[b].insert(pos, k)
                            out.append(s2)
        # swap two groups between different bands
        for bi, b1 in enumerate(bands):
            for b2 in bands[bi + 1:]:
                for i, k1 in enumerate(st.bands[b1]):
                    if k1 in ("living", "passage") or b2 not in self._group(k1).bands:
                        continue
                    for j, k2 in enumerate(st.bands[b2]):
                        if k2 in ("living", "passage") or b1 not in self._group(k2).bands:
                            continue
                        s2 = st.clone()
                        s2.bands[b1][i], s2.bands[b2][j] = k2, k1
                        out.append(s2)
        # swap a band group with a cell group
        if st.mode != "corridor":
            for c in ("L", "R"):
                for ci, kc in enumerate(st.cells[c]):
                    gc = self._group(kc)
                    for b in bands:
                        if b not in gc.bands:
                            continue
                        for j, kb in enumerate(st.bands[b]):
                            gb = self._group(kb)
                            if kb in ("living", "passage") or "cell" not in gb.bands or (gb.kind == "core" and gb.lift):
                                continue
                            s2 = st.clone()
                            s2.cells[c][ci] = kb
                            s2.bands[b][j] = kc
                            out.append(s2)
        # side flips
        for k, g in self.groups.items():
            if g.kind in ("suite", "service", "servant", "core"):
                s2 = st.clone()
                s2.flip[k] = not st.flip.get(k, False)
                out.append(s2)
        return out

    def _climb(self, st: State, budget: int) -> Tuple[State, float, Optional[LayoutResult], int]:
        cur_s, cur = self.evaluate(st)
        used = 1
        improved = True
        while improved and used < budget:
            improved = False
            neigh = self._neighbours(st)
            self.rng.shuffle(neigh)
            for s2 in neigh:
                sc, res = self.evaluate(s2)
                used += 1
                if sc > cur_s + 1e-6:
                    st, cur_s, cur = s2, sc, res
                    improved = True
                    break
                if used >= budget:
                    break
        return st, cur_s, cur, used

    def solve(self, max_evals: int = 4200, samples: int = 120) -> Optional[LayoutResult]:
        """
        For every feasible band mode: sample randomised constructions, then run
        iterated local search from the best few. Returns the best layout found.
        """
        best: Optional[LayoutResult] = None
        modes = self.candidate_modes()
        per_mode = max_evals // max(1, len(modes))
        for mode in modes:
            pool = []
            for i in range(samples):
                st = self.construct(mode, 0.0 if i == 0 else 0.3 + (i % 5) * 0.3)
                sc, res = self.evaluate(st)
                pool.append((sc, i, st))
                if res is not None and (best is None or res.score > best.score):
                    best = res
            pool.sort(key=lambda t: -t[0])
            used = samples
            mode_best_s, mode_best_state = pool[0][0], pool[0][2]
            for sc0, _, st in pool[:4]:
                if used >= per_mode:
                    break
                st, sc, res, u = self._climb(st, min(450, per_mode - used))
                used += u
                if sc > mode_best_s:
                    mode_best_s, mode_best_state = sc, st
                if res is not None and (best is None or res.score > best.score):
                    best = res
            kicks = 0
            while used < per_mode and kicks < 6:
                kicks += 1
                st = mode_best_state.clone()
                for _ in range(2):
                    neigh = self._neighbours(st)
                    if not neigh:
                        break
                    st = self.rng.choice(neigh)
                st, sc, res, u = self._climb(st, min(400, per_mode - used))
                used += u
                if sc > mode_best_s:
                    mode_best_s, mode_best_state = sc, st
                if res is not None and (best is None or res.score > best.score):
                    best = res
        return best
