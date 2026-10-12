"""
Intent Engine — user-intent driven re-planning
==============================================
Lets a homeowner edit a generated plan by *what they want*, not by dragging lines:

    "kitchen to the north-west"      "swap the study and the children's bedroom"
    "attached bath on the other side" "pooja next to the living room"
    "make the living room bigger"     "keep the master bedroom where it is"

Each wish becomes a small, JSON-friendly intent dict. The plan is then
*re-solved*, not patched: the band-spine solver searches arrangements again with
an extended objective

    objective = base score (Vastu + function − NBC/access penalties)
              − Σ intent penalties        (user wishes dominate; graded so the
                                            search is pulled towards them)
              − λ · Σ disruption           (every room the user did not mention
                                            stays as close as possible to where
                                            it was — minimal-change editing)

and returns a few *distinct* options (balanced / best Vastu / least change), each
with a plain-language report: which wishes were met, what else had to move,
and any Vastu taboo or NBC trade-off the wish caused. The user picks one; walls,
doors, windows, amenities, dimensions, Vastu audit and cost are all rebuilt by the
normal pipeline, so an edited plan is as complete and valid as a generated one.

Intent dicts (room fields are space-program keys: 'bed1', 'kitchen', 'bed1_bath', …):
    {"type": "zone",     "room": k, "zones": ["NW"]}       put a room in a compass zone / side
    {"type": "flip",     "room": k}                         attached room to its parent's other side
    {"type": "swap",     "a": k1, "b": k2}                  exchange two rooms
    {"type": "adjacent", "a": k1, "b": k2}                  share a wall
    {"type": "lock",     "room": k}                         keep a room where it is
    {"type": "size",     "room": k, "factor": 1.2}          bigger / smaller (same house envelope)
    {"type": "mirror"}                                      mirror the plan left ↔ right
"""

import math
from typing import Dict, List, Optional, Set, Tuple

from floorplan_generator.layout_solver import BandSpineSolver, LayoutResult, Placed, State, shared_edge
from floorplan_generator.vastu_engine import ROOM_IMPORTANCE, VastuEngine, compose_zone, plan_sides_for_facing

ZONE_NAMES = {"N": "North", "NE": "North-East", "E": "East", "SE": "South-East", "S": "South",
              "SW": "South-West", "W": "West", "NW": "North-West", "CENTER": "Centre (Brahmasthan)"}

W_INTENT = 40.0        # a fully unmet wish costs about as much as a room without a door
LAMBDA = 8.0           # disruption weight per room (× Vastu importance × normalised shift)
ZONE_CHANGE = 1.5      # extra disruption when an unmentioned room changes zone
LOCK_MULT = 4.0        # locked rooms resist moving this much harder
COLLATERAL_TABOO = 20.0  # a strict taboo in a room the user never mentioned
SIZE_LIMITS = (0.6, 1.6)

INTENT_TYPES = ("zone", "flip", "swap", "adjacent", "lock", "size", "mirror")

_TABOO_REASON = {
    ("KITCHEN", "NE"): "fire in the sacred Ishan corner",
    ("BATHROOM", "NE"): "a toilet in the sacred Ishan corner",
    ("STAIRS", "NE"): "a heavy stair blocking the Ishan inflow",
    ("SERVANT", "NE"): "a servant room in the Ishan corner",
}

_PENALTY_TEXT = {
    "daylight": "a habitable room lost its outside wall (no direct window)",
    "nbc_area": "a room fell below its NBC 2016 minimum area",
    "nbc_width": "a room fell below its NBC 2016 minimum width",
    "kitchen_dining": "the kitchen no longer opens into the dining area",
    "toilet_kitchen": "a toilet now shares a wall with the kitchen",
    "toilet_pooja": "a toilet now shares a wall with the pooja room",
    "proportion": "a room became long and narrow",
    "access": "a room could not get a door to the circulation",
}


# =============================================================================
# Compass ↔ plan-cell helpers (plan frame: road / entrance side at the top)
# =============================================================================

def cell_zone(col: int, row: int, facing: str) -> str:
    """Compass zone of a plan cell; col −1/0/1 = left/centre/right, row 1/0/−1 = road side/middle/rear."""
    s = plan_sides_for_facing(facing)
    vert = s["top"] if row > 0 else s["bottom"] if row < 0 else None
    horiz = s["left"] if col < 0 else s["right"] if col > 0 else None
    return compose_zone(vert, horiz)


def zone_cell(zone: str, facing: str) -> Tuple[int, int]:
    for col in (-1, 0, 1):
        for row in (1, 0, -1):
            if cell_zone(col, row, facing) == zone:
                return col, row
    return 0, 0


def plan_grid(facing: str) -> List[List[str]]:
    """3×3 zone grid laid out like the drawing (row 0 = road side)."""
    return [[cell_zone(c, r, facing) for c in (-1, 0, 1)] for r in (1, 0, -1)]


def zone_point(zone: str, footprint, facing: str) -> Tuple[float, float]:
    fx, fy, fw, fd = footprint
    col, row = zone_cell(zone, facing)
    return fx + fw * (0.5 + col / 3.0), fy + fd * (0.5 + row / 3.0)


def _cell_of(cx: float, cy: float, footprint) -> Tuple[int, int]:
    fx, fy, fw, fd = footprint
    col = -1 if cx < fx + fw / 3 else (1 if cx > fx + 2 * fw / 3 else 0)
    row = -1 if cy < fy + fd / 3 else (1 if cy > fy + 2 * fd / 3 else 0)
    return col, row


# =============================================================================
# Plan-level helpers used by the editor UI
# =============================================================================

def _vastu_key(info: Dict) -> str:
    role = info.get("role") or ""
    if role in ("MASTER_BEDROOM", "CHILD_BEDROOM", "GUEST_BEDROOM", "ELDER_BEDROOM"):
        return role
    return info.get("category", "")


def _importance(info: Dict) -> float:
    k = _vastu_key(info)
    return ROOM_IMPORTANCE.get(k, ROOM_IMPORTANCE.get(info.get("category", ""), 1.0))


def is_editable(info: Dict) -> bool:
    """Whole rooms the user can move (not filler foyers, the dining spine, dress areas or ducts)."""
    if info.get("kind") in ("filler", "spine", "passage"):
        return False
    name = info.get("name", "")
    return not any(t in name for t in ("Dress", "Wardrobe", "Duct"))


def editable_rooms(layout_state: Dict) -> List[Tuple[str, Dict]]:
    rooms = (layout_state or {}).get("rooms", {})
    out = [(k, v) for k, v in rooms.items() if is_editable(v)]
    out.sort(key=lambda kv: (kv[1].get("parent") is not None, -_importance(kv[1]), kv[1].get("name", "")))
    return out


def resizable_rooms(layout_state: Dict) -> List[Tuple[str, Dict]]:
    """Editable rooms plus the dining area (size is the one thing the spine can change)."""
    rooms = (layout_state or {}).get("rooms", {})
    out = editable_rooms(layout_state)
    for k, v in rooms.items():
        if v.get("kind") == "spine" and v.get("category") == "DINING":
            out.append((k, v))
    return out


def feasible_cells(layout_state: Dict, key: str) -> Set[Tuple[int, int]]:
    """Plan cells a room can reach in the band-spine topology (what the solver can honour)."""
    rooms = layout_state.get("rooms", {})
    info = rooms.get(key)
    if not info or not is_editable(info):
        return set()
    mode = layout_state.get("mode", "std")
    if info.get("kind") == "living":
        rows = {1}
        bands: List[str] = []
    else:
        bands = info.get("bands") or []
        rows = set()
        if "front" in bands:
            rows.add(1)
        if "rear" in bands:
            rows.add(-1)
        if "private" in bands and mode == "deep":
            rows.add(0)
    cells = {(c, r) for r in rows for c in (-1, 0, 1)}
    if "cell" in bands and mode != "corridor":
        cells |= {(-1, 0), (1, 0)}
    return cells


def feasible_zones(layout_state: Dict, key: str) -> List[str]:
    facing = layout_state.get("facing", "N")
    return [cell_zone(c, r, facing) for (c, r) in sorted(feasible_cells(layout_state, key))]


def zone_fit(layout_state: Dict, key: str, zone: str) -> Tuple[int, bool]:
    """(Vastu score −50…+10, strict taboo?) of putting the room in a zone."""
    info = layout_state.get("rooms", {}).get(key, {})
    fx, fy, fw, fd = layout_state.get("footprint", (0, 0, 10, 10))
    eng = VastuEngine.for_facing(fw, fd, layout_state.get("facing", "N"), fx, fy)
    vk = _vastu_key(info)
    return eng.get_room_score(vk, zone), eng.is_taboo(vk, zone)


def current_zone(layout_state: Dict, key: str) -> str:
    info = layout_state.get("rooms", {}).get(key)
    if not info:
        return "CENTER"
    fx, fy, fw, fd = layout_state.get("footprint", (0, 0, 10, 10))
    eng = VastuEngine.for_facing(fw, fd, layout_state.get("facing", "N"), fx, fy)
    return eng.get_zone_for_position(info["cx"], info["cy"]).value


def room_at_point(layout_state: Dict, x: float, y: float, exclude: Optional[str] = None) -> Optional[str]:
    """Editable room whose rectangle contains (x, y)."""
    for k, v in editable_rooms(layout_state):
        if k == exclude:
            continue
        if abs(x - v["cx"]) <= v["w"] / 2 and abs(y - v["cy"]) <= v["h"] / 2:
            return k
    return None


def zone_at_point(layout_state: Dict, x: float, y: float) -> str:
    fx, fy, fw, fd = layout_state.get("footprint", (0, 0, 10, 10))
    eng = VastuEngine.for_facing(fw, fd, layout_state.get("facing", "N"), fx, fy)
    return eng.get_zone_for_position(x, y).value


def key_for_room(plan, room) -> Optional[str]:
    """Space-program key of a plan room (RoomRect.key, else matched by name)."""
    k = getattr(room, "key", "") or ""
    rooms = (getattr(plan, "layout_state", None) or {}).get("rooms", {})
    if k and k in rooms:
        return k
    for kk, v in rooms.items():
        if v.get("name") == room.name:
            return kk
    return None


def _group_mates(layout_state: Dict, key: str) -> Set[str]:
    rooms = layout_state.get("rooms", {})
    g = rooms.get(key, {}).get("group")
    if not g:
        return {key}
    return {k for k, v in rooms.items() if v.get("group") == g} | {key}


def opposite_intent(layout_state: Dict, key: str) -> Optional[Dict]:
    """
    'Put it on the other side': an attached room (bath / utility) flips to the other
    side of its parent; a whole room mirrors across the house to the nearest
    reachable opposite cell (left ↔ right first, then road side ↔ rear).
    """
    rooms = layout_state.get("rooms", {})
    info = rooms.get(key)
    if not info:
        return None
    if info.get("parent") and info.get("parent") in rooms and \
            info.get("kind") in ("suite", "service", "servant", "core"):
        return {"type": "flip", "room": key}
    fp = layout_state.get("footprint")
    facing = layout_state.get("facing", "N")
    col, row = _cell_of(info["cx"], info["cy"], fp)
    ok = feasible_cells(layout_state, key)
    for cand in ((-col, row) if col else None, (col, -row) if row else None, (-col, -row) if col and row else None):
        if cand and cand != (col, row) and cand in ok:
            return {"type": "zone", "room": key, "zones": [cell_zone(cand[0], cand[1], facing)], "via": "opposite"}
    return None


def normalise_intent(it: Dict) -> Optional[Dict]:
    if not isinstance(it, dict) or it.get("type") not in INTENT_TYPES:
        return None
    out = dict(it)
    if out["type"] == "zone":
        zs = [z for z in out.get("zones", []) if z in ZONE_NAMES]
        if not zs or not out.get("room"):
            return None
        out["zones"] = zs
    if out["type"] == "size":
        f = float(out.get("factor", 1.2))
        out["factor"] = max(SIZE_LIMITS[0], min(SIZE_LIMITS[1], f))
    if out["type"] in ("swap", "adjacent") and (not out.get("a") or not out.get("b") or out["a"] == out["b"]):
        return None
    if out["type"] in ("flip", "lock", "size") and not out.get("room"):
        return None
    return out


def describe_intent(it: Dict, layout_state: Dict) -> str:
    rooms = (layout_state or {}).get("rooms", {})

    def nm(k):
        return rooms.get(k, {}).get("name", k)

    t = it.get("type")
    if t == "zone":
        if it.get("avoid"):
            return f"Move {nm(it['room'])} out of the " + " / ".join(ZONE_NAMES.get(z, z) for z in it["avoid"])
        zs = " / ".join(ZONE_NAMES.get(z, z) for z in it["zones"])
        pre = "opposite side → " if it.get("via") == "opposite" else ""
        return f"Move {nm(it['room'])} → {pre}{zs}"
    if t == "flip":
        par = rooms.get(it["room"], {}).get("parent")
        return f"{nm(it['room'])} → other side of {nm(par) if par else 'its room'}"
    if t == "swap":
        return f"Swap {nm(it['a'])} ↔ {nm(it['b'])}"
    if t == "adjacent":
        return f"{nm(it['a'])} next to {nm(it['b'])}"
    if t == "lock":
        return f"Keep {nm(it['room'])} where it is"
    if t == "size":
        f = it.get("factor", 1.2)
        return f"{nm(it['room'])} {'bigger' if f > 1 else 'smaller'} ({(f - 1) * 100:+.0f}% area)"
    if t == "mirror":
        return "Mirror the whole plan (left ↔ right)"
    return str(it)


def validate_intent(it: Dict, layout_state: Dict) -> Optional[str]:
    """Plain-language reason an intent cannot be honoured in this topology, else None."""
    rooms = layout_state.get("rooms", {})
    t = it.get("type")
    keys = [it.get(k) for k in ("room", "a", "b") if it.get(k)]
    for k in keys:
        if k not in rooms:
            return f"'{k}' is not a room in this plan."
    if t == "zone":
        ok = set(feasible_zones(layout_state, it["room"]))
        if not ok:
            return f"{rooms[it['room']]['name']} is fixed in this layout (it can only be resized)."
        if not set(it["zones"]) & ok:
            names = ", ".join(ZONE_NAMES[z] for z in sorted(ok))
            nm = rooms[it["room"]]["name"]
            if rooms[it["room"]].get("kind") == "living":
                return f"{nm} must stay on the entrance (road) side — choose from: {names}."
            return f"{nm} cannot reach {' / '.join(ZONE_NAMES[z] for z in it['zones'])} here — reachable: {names}."
    if t in ("swap", "adjacent"):
        ga, gb = rooms[it["a"]].get("group"), rooms[it["b"]].get("group")
        if t == "swap" and ga == gb:
            return "Both rooms belong to the same suite — use 'other side' instead."
        if t == "swap" and not (is_editable(rooms[it["a"]]) and is_editable(rooms[it["b"]])):
            return "Only whole rooms can be swapped."
    if t == "flip" and not rooms[it["room"]].get("parent"):
        return f"{rooms[it['room']]['name']} is not an attached room."
    return None


def apply_size_intents(program, intents: List[Dict], base_layout: Optional[Dict] = None) -> List[str]:
    """
    Scale target areas of rooms the user wants bigger / smaller (the envelope stays fixed).
    The new target is taken from the room's *current* size in the plan being edited
    (program targets can be far from what the band allocation actually gave it).
    """
    notes: List[str] = []
    base_rooms = (base_layout or {}).get("rooms", {})
    rooms = {program.living.key: (program.living, None), program.dining.key: (program.dining, None)}
    for g in program.groups:
        for r in g.rooms:
            rooms[r.key] = (r, g)
    for it in intents or []:
        it = normalise_intent(it)
        if not it or it["type"] != "size" or it["room"] not in rooms:
            continue
        r, g = rooms[it["room"]]
        f = it["factor"]
        now = base_rooms.get(it["room"], {}).get("area", 0.0)
        if f > 1:
            r.target_area = round(max(r.target_area * f, now * f * 1.1), 2)
        else:
            r.target_area = round(min(r.target_area * f, now * f) if now else r.target_area * f, 2)
        if g is not None and r is g.primary:
            g.growth = round(g.growth * (f * 1.8 if f > 1 else f * 0.5), 3)
        notes.append(f"{r.name}: target area {'increased' if f > 1 else 'reduced'} to "
                     f"{r.target_area:.1f} m² at your request (other rooms re-balanced in the same envelope).")
    return notes


# =============================================================================
# The intent-aware solver
# =============================================================================

class IntentSolver(BandSpineSolver):
    """BandSpineSolver whose objective adds the user's wishes and a minimal-change term."""

    def __init__(self, program, footprint, facing, vastu_enabled=True, seed=7,
                 intents: Optional[List[Dict]] = None, base: Optional[Dict] = None):
        super().__init__(program, footprint, facing, vastu_enabled, seed)
        self.base = base or {}
        self.intents = [i for i in (normalise_intent(x) for x in intents or []) if i]
        self.diag = math.hypot(self.fw, self.fd) or 1.0
        fp = (self.fx, self.fy, self.fw, self.fd)
        self.fp = fp
        brooms = self.base.get("rooms", {})
        bfp = self.base.get("footprint") or fp

        def remap(x, y):
            bx, by, bw, bd = bfp
            if bw <= 0 or bd <= 0:
                return x, y
            return self.fx + (x - bx) / bw * self.fw, self.fy + (y - by) / bd * self.fd

        self.base_info = brooms
        self.countable = {k for k, v in brooms.items() if is_editable(v)}
        # O(1) zone lookup: same half-open 3×3 cells as VastuEngine.get_zone_for_position
        eng = self.engine
        self._w3, self._h3 = eng.plot_width / 3.0, eng.plot_height / 3.0
        self._ox, self._oy = eng.origin_x, eng.origin_y
        self._cells = {}
        for zone, b in eng.get_all_zones().items():
            c = int(round((b.x - self._ox) / self._w3)) if self._w3 > 0 else 1
            rr = int(round((b.y - self._oy) / self._h3)) if self._h3 > 0 else 1
            self._cells[(c, rr)] = zone.value
        self.base_pt: Dict[str, Tuple[float, float]] = {k: remap(v["cx"], v["cy"]) for k, v in brooms.items()}
        self.base_zone: Dict[str, str] = {k: self._zone_xy(*p) for k, p in self.base_pt.items()}
        self.mirror = any(i["type"] == "mirror" for i in self.intents)
        self.locked = {i["room"] for i in self.intents if i["type"] == "lock"}
        involved: Set[str] = set()
        ls = {"rooms": brooms}
        for i in self.intents:
            for k in (i.get("room"), i.get("a"), i.get("b")):
                if not k or i["type"] in ("lock", "size"):   # a resized room should stay where it is
                    continue
                if i["type"] == "flip":
                    involved.add(k)          # the suite itself should stay put; only its bath swaps sides
                else:
                    involved |= _group_mates(ls, k) if k in brooms else {k}
        self.involved = involved
        cx_mid = self.fx + self.fw / 2
        self.mirror_pt = {k: (2 * cx_mid - p[0], p[1]) for k, p in self.base_pt.items()}
        self.base_result: Optional[LayoutResult] = None

    # ------------------------------------------------------------- helpers
    def _zone_xy(self, x: float, y: float) -> str:
        c = min(2, max(0, int((x - self._ox) // self._w3))) if self._w3 > 0 else 1
        r = min(2, max(0, int((y - self._oy) // self._h3))) if self._h3 > 0 else 1
        return self._cells.get((c, r), "CENTER")

    def _zone(self, p: Placed) -> str:
        return self._zone_xy(p.cx, p.cy)

    def _dist(self, p: Placed, pt: Tuple[float, float]) -> float:
        return math.hypot(p.cx - pt[0], p.cy - pt[1]) / self.diag

    @staticmethod
    def _graded(d: float) -> float:
        """Unmet-wish penalty fraction: ≥ 0.5, rising with distance so search is pulled in."""
        return 0.5 + 0.5 * min(1.0, d / 0.5)

    def _countable(self, key: str) -> bool:
        return key in self.countable

    # ------------------------------------------------------------ intents
    def _intent_eval(self, idx: Dict[str, Placed], explain: bool = False):
        pen = 0.0
        details = []
        names = {k: v.get("name", k) for k, v in self.base_info.items()}
        for it in self.intents:
            t = it["type"]
            ok, p, msg = True, 0.0, ""
            if t == "zone":
                r = idx.get(it["room"])
                if r is None:
                    ok, p, msg = False, W_INTENT, "room is not in the re-planned program"
                else:
                    z = self._zone(r)
                    ok = z in it["zones"]
                    if not ok:
                        d = min(self._dist(r, zone_point(zz, self.fp, self.facing)) for zz in it["zones"])
                        p = W_INTENT * self._graded(d)
                    msg = f"now in {ZONE_NAMES.get(z, z)}"
            elif t == "flip":
                r = idx.get(it["room"])
                par_key = self.base_info.get(it["room"], {}).get("parent")
                par = idx.get(par_key) if par_key else None
                if r is None or par is None or it["room"] not in self.base_pt or par_key not in self.base_pt:
                    ok, p, msg = False, W_INTENT, "attached room not found"
                else:
                    bdx = self.base_pt[it["room"]][0] - self.base_pt[par_key][0]
                    bdy = self.base_pt[it["room"]][1] - self.base_pt[par_key][1]
                    if abs(bdx) >= abs(bdy):
                        now, before = r.cx - par.cx, bdx
                    else:
                        now, before = r.cy - par.cy, bdy
                    ok = now * before < 0 and abs(now) > 0.1
                    p = 0.0 if ok else W_INTENT
                    msg = f"now on the {ZONE_NAMES.get(self._zone(r), '')} side"
            elif t == "swap":
                ra, rb = idx.get(it["a"]), idx.get(it["b"])
                pa, pb = self.base_pt.get(it["a"]), self.base_pt.get(it["b"])
                if ra is None or rb is None or pa is None or pb is None:
                    ok, p, msg = False, W_INTENT, "room not found"
                else:
                    oks = []
                    for r, target_key, tp, own in ((ra, it["b"], pb, pa), (rb, it["a"], pa, pb)):
                        d = self._dist(r, tp)
                        # really exchanged: nearer the partner's old spot than its own, and in its zone
                        good = d + 0.02 < self._dist(r, own) and \
                            (self._zone(r) == self.base_zone.get(target_key) or d < 0.12)
                        oks.append(good)
                        if not good:
                            p += 0.5 * W_INTENT * self._graded(d)
                    ok = all(oks)
                    msg = f"{names.get(it['a'])} in {ZONE_NAMES.get(self._zone(ra))}, " \
                          f"{names.get(it['b'])} in {ZONE_NAMES.get(self._zone(rb))}"
            elif t == "adjacent":
                ra, rb = idx.get(it["a"]), idx.get(it["b"])
                if ra is None or rb is None:
                    ok, p, msg = False, W_INTENT, "room not found"
                else:
                    e = shared_edge(ra, rb)
                    L = (e[2] - e[1]) if e else 0.0
                    ok = L >= 0.9
                    if not ok:
                        p = W_INTENT * self._graded(self._dist(ra, (rb.cx, rb.cy)) * 0.6)
                    msg = f"shared wall {L:.2f} m" if L > 0 else "not touching"
            elif t == "lock":
                r = idx.get(it["room"])
                if r is None:
                    ok, p, msg = False, W_INTENT, "room not found"
                else:
                    z = self._zone(r)
                    ok = z == self.base_zone.get(it["room"], z)
                    p = 0.0 if ok else W_INTENT
                    msg = f"stays in {ZONE_NAMES.get(z, z)}" if ok else f"moved to {ZONE_NAMES.get(z, z)}"
            elif t == "size":
                r = idx.get(it["room"])
                b = self.base_info.get(it["room"], {})
                if r is None or not b:
                    ok, p, msg = False, W_INTENT, "room not found"
                else:
                    ratio = r.area / max(0.1, b.get("w", 0) * b.get("h", 0))
                    f = it["factor"]
                    goal = 1 + 0.5 * (f - 1)          # at least half the requested change
                    short = (goal - ratio) / max(0.02, goal - 1) if f > 1 else (ratio - goal) / max(0.02, 1 - goal)
                    ok = short <= 0
                    p = 0.0 if ok else W_INTENT * (0.5 + 0.5 * min(1.0, short))
                    # don't overshoot: "+20 %" should not become "+50 %" at the neighbours' expense
                    over = (ratio - (1 + 1.5 * (f - 1))) if f > 1 else ((1 - 1.5 * (1 - f)) - ratio)
                    if over > 0:
                        p += 0.4 * W_INTENT * min(1.0, over / 0.3)
                    msg = f"{b.get('w', 0) * b.get('h', 0):.1f} → {r.area:.1f} m² ({(ratio - 1) * 100:+.0f}%)"
            elif t == "mirror":
                keys = [k for k in self.mirror_pt if self._countable(k) and k in idx]
                miss, dsum = 0, 0.0
                for k in keys:
                    tp = self.mirror_pt[k]
                    if self._zone(idx[k]) != self._zone_xy(*tp):
                        miss += 1
                    dsum += self._dist(idx[k], tp)
                n = max(1, len(keys))
                ok = miss <= max(1, int(0.2 * n))
                # a mirror is a whole-plan wish: it must outweigh the Vastu anchors it breaks
                p = 3 * W_INTENT * (miss / n) + 0.5 * W_INTENT * (dsum / n)
                msg = f"{n - miss}/{n} rooms in their mirrored zone"
            pen += p
            if explain:
                details.append({"text": describe_intent(it, {"rooms": self.base_info}), "satisfied": bool(ok),
                                "detail": msg, "type": t})
        return pen, details

    def _disruption(self, idx: Dict[str, Placed]) -> float:
        if self.mirror or not self.base_pt:
            return 0.0
        pen = 0.0
        for k, pt in self.base_pt.items():
            if k in self.involved or not self._countable(k):
                continue
            r = idx.get(k)
            if r is None:
                continue
            w = _importance(self.base_info[k])
            m = LOCK_MULT if k in self.locked else 1.0
            d = self._dist(r, pt)
            pen += m * (LAMBDA * w * min(d, 0.5) + (ZONE_CHANGE * w if self._zone(r) != self.base_zone[k] else 0.0))
        return pen

    def _collateral_taboos(self, idx: Dict[str, Placed]) -> float:
        """Strict Vastu taboos created in rooms the user did not mention (an edit must not cause them)."""
        if not self.vastu_enabled:
            return 0.0
        pen = 0.0
        for k, r in idx.items():
            if k in self.involved or not self._countable(k) or k not in self.base_zone:
                continue
            vk = self._vastu_key(r)
            if vk and self.engine.is_taboo(vk, self._zone(r)) and not self.engine.is_taboo(vk, self.base_zone[k]):
                pen += COLLATERAL_TABOO
        return pen

    def _anchor_refund(self, idx: Dict[str, Placed]) -> float:
        """
        The user overrides the classical anchor (master → SW, kitchen → SE …) for the rooms
        they explicitly moved; the Vastu score itself still reflects the choice.
        """
        if not self.vastu_enabled:
            return 0.0
        refund = 0.0
        for k in self.involved:
            r = idx.get(k)
            if r is None or k not in self.base_zone:
                continue
            vk = self._vastu_key(r)
            if not vk:
                continue
            # only the anchor penalty the move itself adds — staying put earns nothing
            now = self.anchor_penalty(vk, self._zone(r)) or 0.0
            before = self.anchor_penalty(vk, self.base_zone[k]) or 0.0
            refund += max(0.0, now - before)
        return refund

    def score(self, rooms: List[Placed]):
        s, v, door, pen = super().score(rooms)
        idx = {r.key: r for r in rooms}
        ip, _ = self._intent_eval(idx)
        dp = self._disruption(idx)
        cp = self._collateral_taboos(idx)
        s += self._anchor_refund(idx)
        pen = dict(pen)
        if ip:
            pen["intent"] = round(ip, 2)
        if dp:
            pen["change"] = round(dp, 2)
        if cp:
            pen["collateral_taboo"] = cp
        return s - ip - dp - cp, v, door, pen

    # -------------------------------------------------------------- search
    def _base_state(self) -> Optional[State]:
        b = self.base
        if not b.get("bands") or b.get("mode") not in self.candidate_modes():
            return None
        valid = set(self.groups) | {"living"} | ({"passage"} if b["mode"] == "deep" else set())
        bands = {k: [g for g in v if g in valid] for k, v in b["bands"].items()}
        cells = {c: [g for g in b.get("cells", {}).get(c, []) if g in self.groups] for c in ("L", "R")}
        placed = {g for v in bands.values() for g in v} | {g for v in cells.values() for g in v}
        for gk, g in self.groups.items():
            if gk in placed:
                continue
            allowed = [x for x in bands if x in g.bands] or list(bands)
            tgt = min(allowed, key=lambda x: len(bands[x]))
            bands[tgt].append(gk)
        if "front" in bands and "living" not in bands["front"]:
            for v in bands.values():
                if "living" in v:
                    v.remove("living")
            bands["front"].insert(0, "living")
        flip = {k: bool(v) for k, v in (b.get("flip") or {}).items() if k in self.groups}
        return State(mode=b["mode"], bands=bands, cells=cells, flip=flip)

    def _locate(self, st: State, gk: str):
        for b, lst in st.bands.items():
            if gk in lst:
                return ("band", b, lst.index(gk))
        for c, lst in st.cells.items():
            if gk in lst:
                return ("cell", c, lst.index(gk))
        return None

    def _intent_seeds(self, base_st: State) -> List[State]:
        seeds: List[State] = []
        info = self.base_info
        for it in self.intents:
            t = it["type"]
            if t == "swap":
                ga, gb = info.get(it["a"], {}).get("group"), info.get(it["b"], {}).get("group")
                la, lb = self._locate(base_st, ga), self._locate(base_st, gb)
                if la and lb:
                    s2 = base_st.clone()
                    ca = s2.bands[la[1]] if la[0] == "band" else s2.cells[la[1]]
                    cb = s2.bands[lb[1]] if lb[0] == "band" else s2.cells[lb[1]]
                    ca[la[2]], cb[lb[2]] = gb, ga
                    seeds.append(s2)
                    s3 = s2.clone()
                    for g in (ga, gb):
                        if g in self.groups:
                            s3.flip[g] = not s3.flip.get(g, False)
                    seeds.append(s3)
            elif t == "flip":
                g = info.get(it["room"], {}).get("group")
                if g in self.groups:
                    s2 = base_st.clone()
                    s2.flip[g] = not s2.flip.get(g, False)
                    seeds.append(s2)
            elif t == "mirror":
                s2 = base_st.clone()
                for b in s2.bands:
                    s2.bands[b] = list(reversed(s2.bands[b]))
                s2.cells = {"L": list(base_st.cells.get("R", [])), "R": list(base_st.cells.get("L", []))}
                s2.flip = {k: not base_st.flip.get(k, False) for k in self.groups}
                seeds.append(s2)
        return seeds

    def solve(self, max_evals: int = 6000, samples: int = 80) -> Optional[LayoutResult]:
        """
        Seeds = the plan being edited, direct edits of it (swap / flip / mirror applied),
        their 1-move neighbourhoods, and fresh randomised constructions (so a wish that
        needs a different zoning can still find it). Then iterated local search from
        several elites, with random kicks until the budget is spent.
        """
        seeds: List[State] = []
        base_st = self._base_state()
        if base_st is not None:
            direct = self._intent_seeds(base_st)
            seeds.append(base_st)
            seeds.extend(direct)
            seeds.extend(self._neighbours(base_st))
            for s in direct:
                seeds.extend(self._neighbours(s)[:400])
        for mode in self.candidate_modes():
            for i in range(samples):
                seeds.append(self.construct(mode, 0.0 if i == 0 else 0.3 + (i % 5) * 0.3))
        scored = []
        for i, s in enumerate(seeds):
            sc, res = self.evaluate(s)
            scored.append((sc, i, s))
        if base_st is not None:
            self.base_result = self.evaluate(base_st)[1]
        used = len(seeds)
        scored.sort(key=lambda t: -t[0])
        elites: List[Tuple[float, State]] = []
        for sc0, _, st in scored[:8]:
            if used >= max_evals:
                break
            st, sc, _, u = self._climb(st, min(500, max_evals - used))
            used += u
            elites.append((sc, st))
        elites.sort(key=lambda t: -t[0])
        kicks = 0
        while elites and used < max_evals and kicks < 40:
            src = elites[kicks % min(3, len(elites))][1]
            kicks += 1
            st = src.clone()
            for _ in range(2 + kicks % 2):
                neigh = self._neighbours(st)
                if not neigh:
                    break
                st = self.rng.choice(neigh)
            st, sc, _, u = self._climb(st, min(400, max_evals - used))
            used += u
            if sc > elites[-1][0]:
                elites.append((sc, st))
                elites.sort(key=lambda t: -t[0])
                elites = elites[:5]
        return self._best()

    def _pool(self) -> List[LayoutResult]:
        pool = [res for _, res in self._cache.values() if res is not None]
        clean = [r for r in pool if not r.penalties.get("access") and not r.penalties.get("circulation")]
        return clean or pool

    def _best(self) -> Optional[LayoutResult]:
        pool = self._pool()
        return max(pool, key=lambda r: r.score) if pool else None

    # ------------------------------------------------------- alternatives
    def _sat(self, res: LayoutResult) -> int:
        idx = {r.key: r for r in res.rooms}
        _, det = self._intent_eval(idx, explain=True)
        return sum(1 for d in det if d["satisfied"])

    def _changes(self, res: LayoutResult) -> int:
        n = 0
        for r in res.rooms:
            if r.key in self.involved or not self._countable(r.key) or r.key not in self.base_zone:
                continue
            if self._zone(r) != self.base_zone[r.key]:
                n += 1
        return n

    def _regression(self, res: LayoutResult) -> float:
        """Functional penalties this layout adds over the plan being edited."""
        base = self.base_result.penalties if self.base_result is not None else {}
        return sum(max(0.0, res.penalties.get(k, 0) - base.get(k, 0)) for k in _PENALTY_TEXT)

    def _signature(self, res: LayoutResult):
        return tuple(sorted((r.key, self._zone(r)) for r in res.rooms if self._countable(r.key)))

    def alternatives(self, k: int = 3) -> List[Tuple[LayoutResult, str]]:
        """Up to k distinct layouts: balanced (best objective), best Vastu, least change."""
        pool = sorted(self._pool(), key=lambda r: -r.score)
        if not pool:
            return []
        sat = {id(r): self._sat(r) for r in pool[:300]}
        pool = pool[:300]
        top = pool[0]
        top_sat = sat[id(top)]
        # recommend the cleanest layout (fewest new functional problems / taboos) within a small
        # margin of the best objective — a few points of score are not worth a dark bedroom
        near = [r for r in pool if sat[id(r)] >= top_sat and r.score >= top.score - 6.0]
        best = min(near, key=lambda r: (self._regression(r) + r.penalties.get("collateral_taboo", 0), -r.score))
        chosen: List[Tuple[LayoutResult, str]] = [(best, "Balanced — recommended")]
        sigs = {self._signature(best)}
        peers = [r for r in pool if sat[id(r)] >= top_sat and r.score > best.score - 45]
        # prefer options that add no new functional problems beyond the recommended one
        reg_best = self._regression(best)
        clean_peers = [r for r in peers if self._regression(r) <= reg_best + 3.0
                       and r.penalties.get("collateral_taboo", 0) <= best.penalties.get("collateral_taboo", 0)]
        peers = clean_peers + [r for r in peers if r not in clean_peers]

        def take(r, label):
            chosen.append((r, label))
            sigs.add(self._signature(r))

        pick_from = clean_peers or peers
        vastu_c = [r for r in pick_from if self._signature(r) not in sigs and r.vastu_score > best.vastu_score + 0.5]
        if vastu_c and len(chosen) < k:
            take(max(vastu_c, key=lambda r: (r.vastu_score, r.score)), "Best Vastu")
        best_chg = self._changes(best)
        calm_c = [r for r in pick_from if self._signature(r) not in sigs and self._changes(r) < best_chg]
        if calm_c and len(chosen) < k:
            take(min(calm_c, key=lambda r: (self._changes(r), -r.score)), "Least change")
        for r in peers + pool:
            if len(chosen) >= k:
                break
            if self._signature(r) not in sigs:
                take(r, "Alternative")
        return chosen[:k]

    # ------------------------------------------------------------- report
    def _blocker(self, res: LayoutResult, i: int) -> str:
        """Why wish i is unmet: what the best layout that *does* meet it would cost."""
        best = None
        for _, cand in self._cache.values():
            if cand is None or (best is not None and cand.score <= best.score):
                continue
            _, det = self._intent_eval({r.key: r for r in cand.rooms}, explain=True)
            if det[i]["satisfied"]:
                best = cand
        if best is None:
            return "the house footprint has no room to arrange it that way"
        why = []
        for key, txt in list(_PENALTY_TEXT.items()) + [("circulation", "the circulation would break"),
                                                     ("taboo", "it creates a strict Vastu taboo"),
                                                     ("collateral_taboo", "another room would land in a Vastu taboo")]:
            extra = best.penalties.get(key, 0) - res.penalties.get(key, 0)
            if extra > 0.5:
                why.append((extra, txt))
        why.sort(key=lambda t: -t[0])
        if res.vastu_score - best.vastu_score > 4:
            why.append((0, f"Vastu would drop {res.vastu_score - best.vastu_score:.0f} points"))
        if not why:
            return "it would disturb many other rooms"
        return "; ".join(t for _, t in why[:2])

    def report(self, res: LayoutResult) -> Dict:
        idx = {r.key: r for r in res.rooms}
        _, details = self._intent_eval(idx, explain=True)
        for i, d in enumerate(details):
            if not d["satisfied"]:
                d["blocked_by"] = self._blocker(res, i)
        changes, warnings = [], []
        for r in res.rooms:
            if not self._countable(r.key) or r.key not in self.base_zone:
                continue
            z0, z1 = self.base_zone[r.key], self._zone(r)
            if z0 != z1:
                changes.append({"room": r.name, "from": z0, "to": z1, "requested": r.key in self.involved})
            vk = self._vastu_key(r)
            if self.vastu_enabled and vk and (z0 != z1 or r.key in self.involved):
                if self.engine.is_taboo(vk, z1):
                    why = _TABOO_REASON.get((r.category, z1), "a strict Vastu taboo")
                    warnings.append(f"{r.name} in the {ZONE_NAMES.get(z1, z1)} is {why}.")
                elif z0 != z1 and self.engine.get_room_score(vk, z1) <= -10 < self.engine.get_room_score(vk, z0):
                    warnings.append(f"{r.name} in the {ZONE_NAMES.get(z1, z1)} is a weak Vastu zone for it "
                                    f"(score {self.engine.get_room_score(vk, z1)}).")
        if self.base_result is not None:
            for key, txt in _PENALTY_TEXT.items():
                if res.penalties.get(key, 0) > self.base_result.penalties.get(key, 0) + 0.5:
                    warnings.append(f"Trade-off: {txt}.")
        changes.sort(key=lambda c: (not c["requested"], c["room"]))
        return {"intents": details, "satisfied": sum(1 for d in details if d["satisfied"]),
                "total": len(details), "changes": changes, "warnings": warnings,
                "solver_vastu": round(res.vastu_score, 1), "base_vastu": self.base.get("vastu"),
                "mode": res.mode}


# =============================================================================
# Natural-language wishes → intents (delegates to wish_parser: Indian English, misspellings,
# Hindi / Malayalam / Tamil words, complaints, negation — offline, rule-based)
# =============================================================================

def check_intent(it: Dict, layout_state: Dict) -> Tuple[Optional[Dict], Optional[str]]:
    """Normalise + validate one intent; a zone wish keeps only its reachable zones."""
    n = normalise_intent(it)
    if not n:
        return None, "That wish could not be used."
    why = validate_intent(n, layout_state)
    if n["type"] == "zone" and why:
        keep = [z for z in n["zones"] if z in set(feasible_zones(layout_state, n["room"]))]
        if keep:
            n["zones"], why = keep, None
    return (None, why) if why else (n, None)


def parse_intents(text: str, layout_state: Dict) -> Tuple[List[Dict], List[str]]:
    """
    Turn free text into intents. Returns (intents, messages): confident wishes become
    intents; unclear ones come back as "did you mean …" messages (see wish_parser.understand
    for the structured form the UI uses).
    """
    if not (layout_state or {}).get("rooms"):
        return [], ["This plan has no editable AI layout (compact-plot fallback) — regenerate on a larger plot."]
    from floorplan_generator.wish_parser import understand
    good: List[Dict] = []
    msgs: List[str] = []
    for r in understand(text, layout_state):
        for it in r["intents"]:
            n, why = check_intent(it, layout_state)
            if n:
                good.append(n)
            else:
                msgs.append(why)
        if not r["intents"]:
            sugg = [describe_intent(s, layout_state) for s in r["suggestions"] if normalise_intent(s)]
            if sugg:
                msgs.append(f"“{r['clause']}” — did you mean: " + " / ".join(sugg) + "?")
            elif r["note"]:
                msgs.append(r["note"])
            else:
                msgs.append(f"Not sure what to do with: “{r['clause']}” — try 'move kitchen to north-west'.")
    return good, msgs
