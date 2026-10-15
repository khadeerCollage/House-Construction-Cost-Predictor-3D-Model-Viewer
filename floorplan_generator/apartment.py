"""
Apartment floor plate — flats around a common stair + lift core
================================================================
For building_type 'Apartment', LayoutEngine plans a *typical floor* of flats instead of one
house with its stair inside it (residents must never walk through a neighbour's home):

  1 flat / floor    a front band holds the stair, the lift and a lobby; the flat sits behind it
                    and its main door opens off the lobby (the solver keeps it clear of the
                    stair and lift)
  2 flats / floor   flat A | stair + lift | flat B behind an open gallery that runs along the
                    front and reaches both main doors; behind the core an open-to-sky duct gives
                    light and air to toilets and carries the service pipes
  4 flats / floor   two rows of two with a central corridor: the stair sits in the middle of the
                    rear row, the lift opposite it in the front row (ducts behind both); the front
                    flats are planned for their own facing (their doors face the corridor, away
                    from the road) and turned 180° into place, so every flat's Vastu is judged
                    for the direction its own door faces

The same plate serves two kinds of house (LayoutEngine._stack_mode):

  'house'   2+ floors of an independent house — one home on every floor. A front band holds the
            stair (turned to run along the front), the lift when it is inside, and a stair lobby;
            every floor's main door opens off that lobby, so nobody walks through another floor's
            home and each floor arrives at the same lobby. A lift tower outside (beside the
            entrance / at the side) lands on the lobby instead.
  'duplex'  one home on two floors (floorplan_generator/duplex.py): the band is the home's foyer
            with the stair rising from it (and the lift, inside the home); the ground floor holds
            living, dining, kitchen and a guest / elders' room, the first floor the bedrooms and a
            family lounge — each floor planned by the engine behind the same band.

Every flat is planned by the same engine (space program → band-spine solver → doors, windows)
inside its own rectangle, so a flat is exactly a house of that BHK. The floor is then assembled:
walls regenerated for the whole floor with 230 mm party walls between flats and the core, the
open gallery's front as a 1 m parapet, windows kept only where they face open air (outside,
the gallery, a duct), the stair open to the lobby, the lift door on the lobby, and the
building's entrance on the ground floor. Smart Re-Plan edits the flat type; every flat of that
type follows.
"""

import copy
import importlib
import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

GALLERY = 1.5            # open gallery along the front (2 flats / floor)
CORRIDOR4 = 1.8          # central corridor that is also the lift lobby (4 flats / floor)
LOBBY_MIN = 1.5          # lift / stair lobby width (1 flat / floor)
FLOOR_LANDING = 1.25     # an upright stair's floor landing inside its room (narrow plates, 1 flat / floor)
UNIT_MIN_W = {1: 5.6, 2: 6.6, 3: 7.6, 4: 8.4}
UNIT_MIN_D = 6.8
OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}
GRID = 0.05
COMMON_COLORS = {"gallery": "#E3ECF5", "lobby": "#E3ECF5", "duct": "#CFD8DC"}

Rect = Tuple[float, float, float, float]          # x, y, w, h


def _snap(v: float) -> float:
    return round(round(v / GRID) * GRID, 3)


def _le():
    return importlib.import_module("floorplan_generator.layout_engine")


def is_apartment(building_type: str) -> bool:
    return "apartment" in (building_type or "").lower()


# =============================================================================
# Core and floor layout
# =============================================================================

@dataclass
class Slot:
    label: str               # 'A', 'B', … ('' when the floor has one flat)
    rect: Rect               # wall centre-lines, plan frame
    turned: bool             # planned for the opposite facing and turned 180°


@dataclass
class FloorLayout:
    n: int
    footprint: Rect
    slots: List[Slot]
    common: List[Tuple[str, str, Rect, str]]       # (name, category, rect, colour)
    stair: Rect
    lift: Optional[Rect]
    lobby: str                                     # the room the stair, lift and flat doors open to
    lift_side: str                                 # wall of the lift room that faces the lobby
    stair_side: str                                # wall of the stair room open to the lobby
    entrance: Tuple[str, float, float]             # (plan side of the lobby room, a, b) building entrance
    open_edges: List[Tuple[float, float, float, float]] = field(default_factory=list)   # parapet lines
    door_exclusion: Optional[Tuple[float, float]] = None                              # unit-local x
    notes: List[str] = field(default_factory=list)


def core_parts(stair_type: str, has_lift: bool, lift_capacity: str, mode: str = "apartment"):
    """Stair (NBC apartment flights 1.25 m, a home's 1.0 m) and lift sizes, as wall centre-line rectangles."""
    sp = importlib.import_module("floorplan_generator.space_program")
    stype = stair_type if stair_type in ("dog_leg", "open_well") else "dog_leg"
    stair = sp.design_staircase(stype, "Apartment" if mode == "apartment" else "Independent House")
    lift = sp.design_lift(lift_capacity) if has_lift else None
    sw, sl = _snap(stair.footprint_w), _snap(stair.footprint_l)
    lw, ld = ((_snap(lift.shaft_w + 0.2), _snap(lift.shaft_d + 0.2)) if lift else (0.0, 0.0))
    return stair, lift, sw, sl, lw, ld


def floor_layout(n: int, footprint: Rect, facing: str, vastu: bool, sw: float, sl: float,
                 lw: float, ld: float, mode: str = "apartment", stair_end: Optional[str] = None) -> Optional[FloorLayout]:
    """Split the building footprint into flats, core and common spaces (None if it cannot)."""
    vm = importlib.import_module("floorplan_generator.vastu_engine")
    fx, fy, fw, fd = footprint
    eng = vm.VastuEngine(fw, fd, vm.north_edge_for_facing(facing), fx, fy)

    def stair_score(r: Rect) -> float:
        if not vastu:
            return 0.0
        z = eng.get_zone_for_position(r[0] + r[2] / 2, r[1] + r[3] / 2).value
        return eng.get_room_score("STAIRS", z) - (30 if eng.is_taboo("STAIRS", z) else 0)

    if n == 1 and mode in ("house", "duplex"):
        # front band [stair | lobby / foyer | lift]: the dog-leg turned to run along the front keeps the band
        # ~2.1 m deep (a hall, not a hall-sized room); a narrow plate keeps it upright instead
        for rotated in (True, False):
            run = _snap(sl if rotated else sw)                 # stair length along the front
            D = _snap(max(sw if rotated else sl, ld, 1.8))
            hall_w = _snap(fw - run - lw)
            if hall_w >= LOBBY_MIN and fd - D >= UNIT_MIN_D - 0.6:
                break
        else:
            return None
        by = _snap(fy + fd - D)
        best = None
        for end in ([stair_end] if stair_end in ("left", "right") else ["left", "right"]):
            st = (fx if end == "left" else _snap(fx + fw - run), by, run, D)
            sc = stair_score(st)
            if best is None or sc > best[0] + 1e-6:
                best = (sc, end, st)
        _, end, stair = best
        left = end == "left"
        hall_name = "Stair Lobby" if mode == "house" else "Entrance Foyer"
        hall = (_snap(fx + run) if left else _snap(fx + lw), by, hall_w, D)
        common = [(hall_name, "CORRIDOR" if mode == "house" else "FOYER", hall, COMMON_COLORS["lobby"])]
        lift = None
        excl = [((stair[0] - fx) - 0.1, (stair[0] + run - fx) + 0.1)]
        if lw > 0:
            lx = _snap(fx + fw - lw) if left else fx
            lift = (lx, by, lw, D)
            excl.append(((lx - fx) - 0.1, (lx + lw - fx) + 0.1))
        slot = Slot("", (fx, fy, fw, _snap(fd - D)), False)
        mid = hall[0] + hall_w / 2
        half = 0.6 if mode == "house" else 0.5
        if mode == "house":
            note = (f"A home on every floor: the stair{', the lift' if lw else ''} and each floor's main door open "
                    f"onto a {hall_w:.2f} × {D:.2f} m stair lobby at the front — every floor is reached the same way, "
                    "and nobody walks through another floor's home. The lobby's front is an open sit-out with a 1 m "
                    "parapet, so the home's front rooms keep their light and air.")
        else:
            note = (f"Duplex: the stair rises from a {hall_w:.2f} × {D:.2f} m foyer at the front"
                    f"{' beside the lift' if lw else ''}; upstairs it lands in the upper lobby at the same spot.")
        return FloorLayout(
            n=1, footprint=footprint, slots=[slot], common=common, stair=stair, lift=lift,
            lobby=hall_name, lift_side="W" if left else "E", stair_side="E" if left else "W",
            entrance=("N", _snap(mid - half), _snap(mid + half)),
            open_edges=[(hall[0], fy + fd, hall[0] + hall_w, fy + fd)] if mode == "house" else [],
            door_exclusion=[(round(a, 3), round(b, 3)) for a, b in excl], notes=[note])

    if n == 1:
        # front band: [stair | lobby | lift] (or mirrored) — both open onto the lobby, the flat's door too.
        # The dog-leg runs ALONG the front, so the end where a flight starts and the one below arrives faces
        # the lobby: on every floor you step off the stair straight onto the lobby (the lobby is the floor
        # landing). A plate too narrow for that keeps the stair upright with a floor landing at its rear
        # end inside the stair room, open sideways to the lobby.
        for rotated in (True, False):
            run = _snap(sl if rotated else sw)                  # stair room along the front
            sd = _snap(sw if rotated else sl + FLOOR_LANDING)   # and its depth
            cd = _snap(max(sd, ld))
            lobby_w = _snap(fw - run - lw)
            if lobby_w >= LOBBY_MIN and fd - cd >= UNIT_MIN_D - 0.6:
                break
        else:
            return None
        by = _snap(fy + fd - cd)
        best = None
        for left in (True, False):
            sx = fx if left else _snap(fx + fw - run)
            stair = (sx, by, run, cd)
            sc = stair_score(stair)
            if best is None or sc > best[0] + 1e-6:
                best = (sc, left, stair)
        _, left, stair = best
        sw = run
        lobby_x = _snap(fx + sw) if left else _snap(fx + lw)
        lobby = (lobby_x, by, lobby_w, cd)
        common = [("Lift & Stair Lobby", "CORRIDOR", lobby, COMMON_COLORS["lobby"])]
        lift = None
        excl = [((stair[0] - fx) - 0.1, (stair[0] + sw - fx) + 0.1)]
        if lw > 0:
            lx = _snap(fx + fw - lw) if left else fx
            lift = (lx, by, lw, _snap(min(ld, cd)))
            if cd - ld > 0.3:
                common.append(("Duct / Meter Room", "CORRIDOR", (lx, _snap(by + ld), lw, _snap(cd - ld)),
                               COMMON_COLORS["duct"]))
            excl.append(((lx - fx) - 0.1, (lx + lw - fx) + 0.1))
        slot = Slot("", (fx, fy, fw, _snap(fd - cd)), False)
        mid = lobby[0] + lobby[2] / 2
        return FloorLayout(
            n=1, footprint=footprint, slots=[slot], common=common, stair=stair, lift=lift,
            lobby="Lift & Stair Lobby", lift_side="W" if left else "E", stair_side="E" if left else "W",
            entrance=("N", _snap(mid - 0.6), _snap(mid + 0.6)),
            door_exclusion=[(round(a, 3), round(b, 3)) for a, b in excl],
            notes=[f"Typical floor: one flat behind a front core — the stair, "
                   f"{'the lift and ' if lw else ''}the flat's main door all open onto a {lobby_w:.2f} m lobby."])

    if n == 2:
        C = GALLERY
        cw = _snap(sw + lw)
        cd = _snap(max(sl, ld))
        uw = _snap((fw - cw) / 2)
        if fd - C - cd < 0.3 or uw <= 0:
            return None
        cx = _snap(fx + uw)
        top = _snap(fy + fd - C)
        stair_left = stair_score((cx, top - sl, sw, sl)) >= stair_score((cx + lw, top - sl, sw, sl))
        sx, lx = (cx, cx + sw) if stair_left else (cx + lw, cx)
        stair = (sx, _snap(top - sl), sw, sl)
        common = [("Common Gallery", "CORRIDOR", (fx, top, fw, C), COMMON_COLORS["gallery"])]
        lift = None
        if lw > 0:
            lift = (lx, _snap(top - ld), lw, ld)
            if cd - ld > 0.3:
                common.append(("Duct / OTS", "CORRIDOR", (lx, _snap(top - cd), lw, _snap(cd - ld)), COMMON_COLORS["duct"]))
        if cd - sl > 0.3:
            common.append(("Duct / OTS", "CORRIDOR", (sx, _snap(top - cd), sw, _snap(cd - sl)), COMMON_COLORS["duct"]))
        if top - cd - fy > 0.3:
            common.append(("Duct / OTS (light well)", "CORRIDOR", (cx, fy, cw, _snap(top - cd - fy)), COMMON_COLORS["duct"]))
        right_w = _snap(fx + fw - (cx + cw))
        w_unit = min(uw, right_w)
        slots = [Slot("A", (fx, fy, w_unit, _snap(fd - C)), False),
                 Slot("B", (_snap(cx + cw), fy, w_unit, _snap(fd - C)), False)]
        mid = cx + cw / 2
        return FloorLayout(
            n=2, footprint=footprint, slots=slots, common=common, stair=stair, lift=lift,
            lobby="Common Gallery", lift_side="N", stair_side="N",
            entrance=("N", _snap(mid - 0.7), _snap(mid + 0.7)),
            open_edges=[(fx, fy + fd, fx + fw, fy + fd), (fx, top, fx, fy + fd), (fx + fw, top, fx + fw, fy + fd)],
            notes=[f"Typical floor: two flats with a stair{' and lift' if lw else ''} core between them; "
                   f"a {C:.1f} m open gallery along the front leads to both main doors; the duct behind the "
                   "core lights and ventilates the toilets."])

    if n == 4:
        C = CORRIDOR4
        cw = _snap(max(sw, lw))
        du = _snap((fd - C) / 2)
        uw = _snap((fw - cw) / 2)
        if du - sl < -1e-6 or uw <= 0:
            return None
        cx = _snap(fx + uw)
        y_c0, y_c1 = _snap(fy + du), _snap(fy + du + C)
        stair = (cx, _snap(y_c0 - sl), cw, sl)
        common = [("Common Corridor", "CORRIDOR", (fx, y_c0, fw, C), COMMON_COLORS["lobby"])]
        if du - sl > 0.3:
            common.append(("Duct / OTS", "CORRIDOR", (cx, fy, cw, _snap(du - sl)), COMMON_COLORS["duct"]))
        lift = None
        front_d = _snap(fy + fd - y_c1)
        if lw > 0:
            lift = (cx, y_c1, cw, ld)
            if front_d - ld > 0.3:
                common.append(("Duct / OTS", "CORRIDOR", (cx, _snap(y_c1 + ld), cw, _snap(front_d - ld)), COMMON_COLORS["duct"]))
        else:
            common.append(("Duct / OTS", "CORRIDOR", (cx, y_c1, cw, front_d), COMMON_COLORS["duct"]))
        right_w = _snap(fx + fw - (cx + cw))
        w_unit = min(uw, right_w)
        d_unit = min(du, front_d)
        slots = [Slot("A", (fx, fy, w_unit, du), False), Slot("B", (_snap(cx + cw), fy, w_unit, du), False),
                 Slot("C", (fx, y_c1, w_unit, d_unit), True), Slot("D", (_snap(cx + cw), y_c1, w_unit, d_unit), True)]
        return FloorLayout(
            n=4, footprint=footprint, slots=slots, common=common, stair=stair, lift=lift,
            lobby="Common Corridor", lift_side="S", stair_side="N",
            entrance=("W", _snap(y_c0 + 0.15), _snap(y_c1 - 0.15)),
            notes=[f"Typical floor: four flats on a {C:.1f} m central corridor; the stair and "
                   f"{'lift' if lw else 'duct'} face each other across it; front flats are planned for their own "
                   "facing and turned to open onto the corridor."])
    return None


def _plate_needs(n: int, program, nb: int, sw, sl, lw, ld, mode: str = "apartment") -> Tuple[float, float, float, float]:
    """(target area, min area, min width, min depth) of the building footprint for n flats."""
    mw, md = UNIT_MIN_W.get(nb, 6.6), UNIT_MIN_D
    unit_t, unit_min = program.target_area * 1.12, program.min_area
    if n == 1 and mode != "apartment":
        D = max(sw, ld, 1.8)
        width = max(mw, sl + lw + LOBBY_MIN, math.sqrt(unit_t * 1.15))     # the band runs the full width
        core = width * D
        return unit_t + core, unit_min + core, max(mw, sl + lw + LOBBY_MIN), md + D
    if n == 1:                       # the stair turned along the front (floor_layout)
        core = (sl + lw + LOBBY_MIN) * max(sw, ld)
        return unit_t + core, unit_min + core, max(mw, sl + lw + LOBBY_MIN), md + max(sw, ld)
    if n == 2:
        cw, cd = sw + lw, max(sl, ld)
        width = 2 * mw + cw
        return 2 * unit_t + (2 * unit_t / width) * cw + GALLERY * width, \
            2 * unit_min + cw * md + GALLERY * width, width, max(md, cd + 0.4) + GALLERY
    cw = max(sw, lw)
    width = 2 * mw + cw
    return 4 * unit_t + cw * (2 * unit_t / width) * 2 / 2 + CORRIDOR4 * width, \
        4 * unit_min + CORRIDOR4 * width, width, 2 * md + CORRIDOR4


def _unit_fits(slot: Slot, program, nb: int, comfy: bool) -> bool:
    w, h = slot.rect[2], slot.rect[3]
    mw = UNIT_MIN_W.get(nb, 6.6)
    area = w * h
    need = program.target_area * (0.92 if comfy else 0.0) if comfy else program.min_area
    return w >= mw - 0.5 and h >= UNIT_MIN_D - 0.6 and area >= max(program.min_area, need)


def plan_floor(plot_w: float, plot_h: float, facing: str, vastu: bool, bhk: str, family=None,
               building_type: str = "Apartment", num_floors: int = 1, quality: str = "Standard",
               has_lift: bool = False, lift_capacity: str = "4 Person", stair_type: str = "dog_leg",
               components: Optional[Dict] = None, members: int = 4, wanted="auto", mode: str = "apartment",
               lift_out: Optional[Dict] = None, program=None, boost: float = 1.0):
    """
    Choose flats per floor and lay out the footprint. Returns (layout, site, planner, program, stair,
    lift, report) — report = {n: {'fits', 'comfortable', 'unit': (w, h), 'area'}} for every option.
    """
    sp = importlib.import_module("floorplan_generator.space_program")
    spl = importlib.import_module("floorplan_generator.site_planner")
    if program is None:
        program = sp.build_program(bhk, family, "Independent House", 1, quality, False, lift_capacity, stair_type)
    program.external_stair = None
    inside = has_lift and not lift_out                 # an outside tower is not part of the band
    stair, lift, sw, sl, lw, ld = core_parts(stair_type, inside, lift_capacity, mode)
    nb = int(program.bhk[0])
    comps = dict(components or {})
    comps["_has_lift"] = bool(has_lift)
    report, found = {}, {}
    for n in ((1, 2, 4) if mode == "apartment" else (1,)):
        planner = spl.SitePlanner(plot_w, plot_h, facing, vastu, building_type, num_floors, copy.deepcopy(comps), members)
        target, min_area, mw, md = _plate_needs(n, program, nb, sw, sl, lw, ld, mode)
        site = planner.plan_footprint(target * boost, min_area, mw, md, 0.0, lift_out=lift_out)
        end = None
        if getattr(site, "lift_side", None) in ("left", "right"):     # the lobby reaches the tower's side wall
            end = "right" if site.lift_side == "left" else "left"
        lay = floor_layout(n, site.footprint, facing, vastu, sw, sl, lw, ld, mode, end)
        fits = bool(lay) and all(_unit_fits(s, program, nb, False) for s in lay.slots)
        comfy = bool(lay) and all(_unit_fits(s, program, nb, True) for s in lay.slots)
        u = lay.slots[0].rect if lay else None
        report[n] = {"fits": fits, "comfortable": comfy, "unit": (u[2], u[3]) if u else None,
                     "area": round(u[2] * u[3], 1) if u else 0.0}
        found[n] = (lay, site, planner)
    if mode != "apartment":
        n = 1
    elif wanted in (1, 2, 4, "1", "2", "4"):
        n = int(wanted)
        if not found[n][0]:
            n = next((m for m in (1, 2, 4) if found[m][0]), 1)
    else:
        n = next((m for m in (4, 2) if report[m]["comfortable"]), 1)
    lay, site, planner = found[n]
    return lay, site, planner, program, stair, lift, report


def flats_options(plot_w, plot_h, facing="N", bhk="2BHK", family=None, num_floors=1, vastu=True,
                  has_lift=False, lift_capacity="4 Person", stair_type="dog_leg") -> Dict[int, Dict]:
    """For the form: how 1, 2 and 4 flats per floor fit this plot ('fits' / 'comfortable' / unit size)."""
    *_, report = plan_floor(plot_w, plot_h, facing, vastu, bhk, family, "Apartment", num_floors, "Standard",
                            has_lift, lift_capacity, stair_type)
    return report


# =============================================================================
# Generation (called by LayoutEngine.generate / generate_alternatives for apartments)
# =============================================================================

def generate(eng, alternatives: int = 0):
    mode = eng._stack_mode() or "apartment"
    if mode == "duplex":
        return importlib.import_module("floorplan_generator.duplex").generate(eng, alternatives)
    le = _le()
    members = 4
    if eng.family_profile is not None:
        fp = eng.family_profile
        members = max(1, getattr(fp, "adults", 2) + getattr(fp, "children", 0) + getattr(fp, "elderly", 0))
    base = eng.base_layout or {}
    wanted = (base.get("apartment") or {}).get("flats_per_floor") or eng.flats_per_floor
    lift_out, lift_design = None, None
    if mode == "house" and eng.has_lift:
        lift_design = importlib.import_module("floorplan_generator.space_program").design_lift(eng.lift_capacity)
        if eng.lift_position in ("entrance", "side"):
            lift_out = {"position": eng.lift_position, "w": lift_design.shaft_w, "d": lift_design.shaft_d, "door": 0.9}
    def plate(boost=1.0, lo=lift_out):
        return plan_floor(eng.plot_width, eng.plot_height, eng.orientation, eng.vastu_enabled, eng.bhk_name,
                          eng.family_profile, eng.building_type, eng.num_floors, eng.quality_level, eng.has_lift,
                          eng.lift_capacity, eng.stair_type, eng.additional_components, members, wanted, mode, lo,
                          None, boost)

    lay, site, planner, program, stair, lift, report = plate()
    if lift_out and getattr(site, "lift_side", None) is None:      # no yard for the tower: lift in the band
        lay, site, planner, program, stair, lift, report = plan_floor(
            eng.plot_width, eng.plot_height, eng.orientation, eng.vastu_enabled, eng.bhk_name, eng.family_profile,
            eng.building_type, eng.num_floors, eng.quality_level, eng.has_lift, eng.lift_capacity, eng.stair_type,
            eng.additional_components, members, wanted, mode, None)
        site.notes.append("The lift could not go outside the house on this plot — it stands in the stair lobby.")
        lift_out = None
    if lay is None:                      # too narrow for a separate core: the stair stays inside the flat
        eng._apartment_fallback = True
        plans = eng.generate_alternatives(alternatives) if alternatives else [eng.generate()]
        for p in plans:
            p.design_notes.insert(0, ("Apartment: the plot is too narrow for a separate stair / lift core, so the "
                                      "stair is inside the flat (one family per building).") if mode == "apartment" else
                                     ("The plot is too narrow for a stair lobby at the front, so the stair is inside "
                                      "the home and the floors are one house."))
        return plans if alternatives else plans[0]

    unit_base = {k: v for k, v in base.items() if k != "apartment"} if base else None

    def solve_units(lay):
        types = {}
        for turned in sorted({s.turned for s in lay.slots}):
            slot = next(s for s in lay.slots if s.turned == turned)
            facing = OPPOSITE[eng.orientation] if turned else eng.orientation
            sub = le.LayoutEngine(slot.rect[2], slot.rect[3], eng.bhk_name, eng.vastu_enabled, facing,
                                  eng.quality_level, eng.project_name, eng.family_description, None, False,
                                  eng.lift_capacity, eng.stair_type, 1, "Independent House", {}, eng.family_profile,
                                  design_intents=None if turned else eng.design_intents,
                                  base_layout=None if turned else unit_base)
            sub._fixed_site = le.SitePlanFixed(slot.rect[2], slot.rect[3], facing,
                                               None if turned else _local_exclusion(lay))
            if alternatives and not turned:
                types[turned] = sub.generate_alternatives(alternatives)
            else:
                types[turned] = [sub.generate()]
        return types

    types = solve_units(lay)
    if mode == "house":                  # a home the solver could not lay out: give the plate a little more room
        for boost in (1.15, 1.3):
            if all(u.layout_state for u in types[False]):
                break
            got = plate(boost)
            if got[0] is None:
                break
            t2 = solve_units(got[0])
            if all(u.layout_state for u in t2[False]):
                lay, site, planner, program, stair, lift, report = got
                types = t2
        if not all(u.layout_state for u in types[False]):
            # the plot's coverage leaves too little behind a stair lobby: one house with the stair inside
            eng._apartment_fallback = True
            try:
                old = eng.generate_alternatives(alternatives) if alternatives else [eng.generate()]
            finally:
                eng._apartment_fallback = False
            if all(p.layout_state for p in old):
                for p in old:
                    p.design_notes.insert(0, "This plot is too small for a separate stair lobby with a full home "
                                             "behind it, so the stair is inside the home (the floors share it).")
                return old if alternatives else old[0]
    plans = []
    for unit in types[False]:
        plan = compose(eng, copy.deepcopy(lay), copy.deepcopy(site), copy.deepcopy(planner), unit,
                       (types.get(True) or [None])[0], stair, lift, report, mode=mode,
                       lift_design=lift_design if lift_out else None)
        plans.append(plan)
    return plans if alternatives else plans[0]


def _local_exclusion(lay: FloorLayout):
    return lay.door_exclusion


# ---- transforms -----------------------------------------------------------------
class _T:
    """Unit frame → plan frame: optional 180° turn inside the unit rectangle, then a shift."""

    def __init__(self, slot: Slot):
        self.x0, self.y0, self.w, self.h = slot.rect
        self.turned = slot.turned

    def pt(self, x, y):
        if self.turned:
            x, y = self.w - x, self.h - y
        return round(self.x0 + x, 3), round(self.y0 + y, 3)

    def rect(self, x, y, w, h):
        if self.turned:
            x, y = self.w - x - w, self.h - y - h
        return round(self.x0 + x, 3), round(self.y0 + y, 3), w, h

    def side(self, s):
        return OPPOSITE[s] if self.turned else s


def _owner(rooms, owner_of, x, y):
    for r in rooms:
        if r.x - 1e-6 < x < r.right + 1e-6 and r.y - 1e-6 < y < r.top + 1e-6:
            return owner_of.get(r.name)
    return None


def compose(eng, lay: FloorLayout, site, planner, unit, unit_turned, stair, lift, report,
            mode: str = "apartment", floor: str = "ground", lift_design=None, hall_name: Optional[str] = None):
    """
    Assemble the typical floor plan from the flat plan(s), the core and the common spaces.
    mode 'house': one home a floor, its main door on the stair lobby (a lift tower outside lands there too);
    mode 'duplex': one home over two floors — `floor` 'ground' (foyer + entrance) or 'upper' (upper lobby,
    no entrance, no site); the band is part of the home and opens to it through an arch.
    """
    le = _le()
    fx, fy, fw, fd = lay.footprint
    duplex = mode == "duplex"
    upper = duplex and floor == "upper"
    if hall_name and hall_name != lay.lobby:
        lay.common = [((hall_name if c[0] == lay.lobby else c[0]),) + tuple(c[1:]) for c in lay.common]
        lay.lobby = hall_name
    rooms, doors, windows, openings = [], [], [], []
    owner_of: Dict[str, str] = {}
    carpet = 0.0
    for slot in lay.slots:
        up = unit_turned if slot.turned else unit
        T = _T(slot)
        pre = f"{slot.label} · " if slot.label else ""
        for r in up.rooms:
            if le._is_site_room(r):
                continue
            x, y, w, h = T.rect(r.x, r.y, r.w, r.h)
            nr = le.RoomRect(name=pre + r.name, category=r.category, x=x, y=y, w=w, h=h, color=r.color,
                             area=r.area, role=r.role, key="" if (slot.turned or upper) else r.key)
            rooms.append(nr)
            owner_of[nr.name] = "home" if duplex else (slot.label or "flat")
        carpet += up.total_carpet_area
        for d in up.doors:
            if d.wall_side in ("N", "S"):
                x, y = (T.w - d.x - d.width, T.h - d.y) if T.turned else (d.x, d.y)
            else:
                x, y = (T.w - d.x, T.h - d.y - d.width) if T.turned else (d.x, d.y)
            if duplex and d.is_main_entrance:       # the home's own foyer / upper lobby: an open arch, no door
                X, Y = round(T.x0 + x, 3), round(T.y0 + y, 3)
                openings.append((X, Y, round(X + d.width, 3), Y) if d.wall_side in ("N", "S")
                                else (X, Y, X, round(Y + d.width, 3)))
                continue
            doors.append(le.DoorPlacement(x=round(T.x0 + x, 3), y=round(T.y0 + y, 3), width=d.width,
                                          wall_side=T.side(d.wall_side), swing_direction="RIGHT",
                                          room_name=pre + d.room_name, is_main_entrance=False))
        for wd in up.windows:
            x, y = T.pt(wd.x, wd.y)
            windows.append(le.WindowPlacement(x=x, y=y, width=wd.width, height=wd.height,
                                              wall_side=T.side(wd.wall_side), room_name=pre + wd.room_name))
        for (x1, y1, x2, y2) in (up.openings or []):
            a, b = T.pt(x1, y1), T.pt(x2, y2)
            openings.append((min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])))

    # ---- core and common spaces
    core = [le.RoomRect(name="Staircase", category="STAIRS", x=lay.stair[0], y=lay.stair[1], w=lay.stair[2],
                        h=lay.stair[3], color="#B8C4D6", role="STAIRS", key="")]
    if lay.lift:
        core.append(le.RoomRect(name="Lift / Elevator", category="LIFT", x=lay.lift[0], y=lay.lift[1],
                                w=lay.lift[2], h=lay.lift[3], color="#A9B7C9", role="LIFT", key=""))
    for name, cat, (x, y, w, h), col in lay.common:
        core.append(le.RoomRect(name=name, category=cat, x=x, y=y, w=w, h=h, color=col,
                                role="" if duplex else "COMMON", key=""))
    names = {}
    for r in core:                                  # unique names (several ducts)
        k = names.get(r.name, 0)
        names[r.name] = k + 1
        if k:
            r.name = f"{r.name} {k + 1}"
        owner_of[r.name] = "home" if duplex else "common"
    rooms += core
    lobby = next(r for r in core if r.name == lay.lobby)

    def edge(room, side):
        """(lo, hi, line) of a room edge."""
        if side == "N":
            return room.x, room.right, room.top
        if side == "S":
            return room.x, room.right, room.y
        if side == "E":
            return room.y, room.top, room.right
        return room.y, room.top, room.x

    # stair open to the lobby along its shared edge; lift door on the lobby
    st_room = core[0]
    lo, hi, line = edge(st_room, lay.stair_side)
    pad = 0.12
    if lay.stair_side in ("N", "S"):
        openings.append((round(lo + pad, 3), line, round(hi - pad, 3), line))
    else:
        openings.append((line, round(lo + pad, 3), line, round(hi - pad, 3)))
    if lay.lift:
        lr = core[1]
        lo, hi, line = edge(lr, lay.lift_side)
        mid = (lo + hi) / 2
        if lay.lift_side in ("N", "S"):
            doors.append(le.DoorPlacement(x=round(mid - 0.45, 3), y=line, width=0.9, wall_side=lay.lift_side,
                                          swing_direction="RIGHT", room_name=lr.name))
        else:
            doors.append(le.DoorPlacement(x=line, y=round(mid - 0.45, 3), width=0.9, wall_side=lay.lift_side,
                                          swing_direction="RIGHT", room_name=lr.name))
    # building entrance (ground floor): the lobby's front, or the corridor's end
    side, a, b = lay.entrance
    if upper:                                        # a duplex's first floor: a window where the door is below
        door_x = (a, b)
        windows.append(le.WindowPlacement(x=round((a + b) / 2, 3), y=round(lobby.top, 3), width=1.2, height=1.2,
                                          wall_side="N", room_name=lobby.name))
    elif side == "N":
        main = le.DoorPlacement(x=a, y=round(lobby.top, 3), width=round(b - a, 3), wall_side="N",
                                swing_direction="RIGHT", room_name=lobby.name, is_main_entrance=True)
        door_x = (a, b)
    else:
        main = le.DoorPlacement(x=round(lobby.x if side == "W" else lobby.right, 3), y=a, width=round(b - a, 3),
                                wall_side=side, swing_direction="RIGHT", room_name=lobby.name, is_main_entrance=True)
        sb_left = fx
        door_x = ((max(0.25, fx - sb_left + 0.25), max(0.9, fx - 0.25)) if fx >= 1.15 else (fx + 0.3, fx + 1.3))
    if not upper:
        doors.append(main)
    if lay.n == 4:                                   # the corridor's far end gets a window
        windows.append(le.WindowPlacement(x=round(lobby.right, 3), y=round(lobby.cy, 3), width=1.2,
                                          height=round(min(1.2, lobby.h - 0.3), 3), wall_side="E",
                                          room_name=lobby.name))

    # ---- a lift tower outside (a house of homes on every floor): it lands on the stair lobby
    tower, tnotes = None, []
    if lift_design is not None and getattr(site, "lift_side", None):
        tower = eng._outside_lift(SimpleNamespace(lift=lift_design), site, rooms, doors, door_x,
                                  entry_rooms={lobby.name})
        if tower is not None:
            eng._tower_into(tower, rooms, doors, site, tnotes)
        else:
            tnotes.append("The lift tower could not be fitted outside the walls of this layout.")

    # ---- site amenities around the whole building (the ground floor of a duplex carries them)
    if not upper:
        eng._gate_brief(planner, site, rooms)
        planner.place_amenities(site, door_x[0], door_x[1], None)
        for it in site.items:
            x, y, w, h = it.rect
            rooms.append(le.RoomRect(name=it.name, category=it.category, x=round(x, 3), y=round(y, 3),
                                     w=round(w, 3), h=round(h, 3), color=it.color))

    # ---- walls: envelope + partitions, party walls, parapet on the open gallery
    walls = eng._generate_walls(rooms)
    if upper:
        walls = [w for w in walls if getattr(w, "wall_type", "") != "compound"]
    if tower is not None:
        walls.extend(eng._tower_walls(tower))
    cuts = [le.LayoutEngine._door_cut(d) for d in doors] + list(openings)
    top = eng.plot_height
    if not upper:
        cuts += [(g0, top, g1, top) for g0, g1 in site.gates]
    walls = eng._cut_openings(walls, cuts)
    walls = _split_open_edges(walls, lay.open_edges, le)
    building = [r for r in rooms if not le._is_site_room(r)]
    for w in walls:
        if getattr(w, "wall_type", "") != "internal":
            continue
        mx, my = (w.x1 + w.x2) / 2, (w.y1 + w.y2) / 2
        nx, ny = (0.0, 0.2) if abs(w.y1 - w.y2) < 1e-6 else (0.2, 0.0)
        o1 = _owner(building, owner_of, mx + nx, my + ny)
        o2 = _owner(building, owner_of, mx - nx, my - ny)
        if o1 and o2 and o1 != o2:            # between two homes, or a home and the core
            w.thickness = 0.23

    # ---- windows only where they see open air: outside, the open gallery or a duct
    open_rooms = {r.name for r in core if "Duct" in r.name or ((lay.n == 2 or mode == "house") and r.name == lay.lobby)}
    kept = []
    for wd in windows:
        ox, oy = {"N": (0, 1), "S": (0, -1), "E": (1, 0), "W": (-1, 0)}[wd.wall_side]
        px, py = wd.x + ox * 0.35, wd.y + oy * 0.35
        if not (fx < px < fx + fw and fy < py < fy + fd):
            kept.append(wd)
            continue
        host = next((r for r in building if r.x < px < r.right and r.y < py < r.top), None)
        if host is not None and host.name in open_rooms:
            kept.append(wd)
    windows = kept
    if tower is not None:
        windows = eng._clear_windows(windows, rooms, doors, tower["outer"])

    core_prog = SimpleNamespace(stair=stair, lift=lift, external_stair=None)
    staircases = eng._place_staircases(rooms, core_prog)
    dimensions = eng._generate_dimensions(rooms)
    eng._set_clear_areas(core, lay.footprint)
    built_up = (fw + eng.wall.external) * (fd + eng.wall.external)

    plan = le.GeneratedFloorPlan(
        plot_width=eng.plot_width, plot_height=eng.plot_height, orientation=eng.orientation,
        vastu_enabled=eng.vastu_enabled, bhk_config=unit.bhk_config, family_description=eng.family_description,
        quality_level=eng.quality_level, project_name=eng.project_name, has_lift=bool(lift),
        lift_capacity=eng.lift_capacity, building_type=eng.building_type, num_floors=eng.num_floors,
        additional_components=eng.additional_components, north_edge=eng.north_edge, stair_type=eng.stair_type)
    plan.rooms, plan.walls, plan.doors, plan.windows = rooms, walls, doors, windows
    plan.staircases, plan.dimensions, plan.openings = staircases, dimensions, openings
    if duplex:
        plan.project_name = f"{eng.project_name} — {'First' if upper else 'Ground'} floor"
    plan.vastu_score = unit.vastu_score                       # every flat of this type scores the same
    plan.vastu_breakdown = copy.deepcopy(unit.vastu_breakdown)
    plan.total_carpet_area = round(carpet, 2)
    plan.total_built_up_area = round(built_up, 2)
    plan.building_footprint = (round(fx, 3), round(fy, 3), round(fw, 3), round(fd, 3))
    plan.setbacks = {eng.sides[k]: v for k, v in site.setbacks.items()}
    engineering = eng._engineering_report(rooms, walls, core_prog, site, carpet, built_up)
    if not upper:
        le.LayoutEngine._gate_report(engineering, site, planner)
    flats = len(lay.slots)
    if lift_design is not None:                      # the lift is a tower outside, not in the band
        engineering["lift"] = {"persons": lift_design.persons, "load_kg": lift_design.rated_load_kg,
                               "shaft_m": [lift_design.shaft_w, lift_design.shaft_d], "position": "inside"}
        if tower is not None:
            engineering["lift"].update(eng._tower_info(tower))
    elif engineering.get("lift"):
        engineering["lift"]["position"] = {"apartment": "core", "house": "lobby"}.get(mode, "inside")
    if not duplex:
        engineering["apartment"] = {
            "kind": mode, "flats_per_floor": flats, "flats_total": flats * eng.num_floors, "flat_bhk": unit.bhk_config,
            "flat_carpet_m2": round(unit.total_carpet_area, 2),
            "labels": [s.label for s in lay.slots], "turned": [s.turned for s in lay.slots],
            "slots": [list(s.rect) for s in lay.slots], "lobby": lay.lobby, "stair_open": lay.stair_side,
            "options": {str(k): v for k, v in report.items()}}
    plan.engineering = engineering
    unit_notes = [n for n in unit.design_notes if not n.startswith(("Setbacks:", "Single storey", "Layout:",
                                                                     "Gates on the"))]
    if mode == "house":
        notes = lay.notes + [
            f"{eng.num_floors} floors = {eng.num_floors} homes, each {unit.bhk_config} of "
            f"{unit.total_carpet_area:.1f} m² carpet with its own front door; 230 mm walls round the stair lobby.",
            f"Vastu of each home, judged for the way its own front door faces: {unit.vastu_score:.0f}/100."]
    elif duplex:
        notes = list(lay.notes) if not upper else []
    else:
        notes = lay.notes + [
            f"{flats} flat{'s' if flats > 1 else ''} per floor × {eng.num_floors} floor{'s' if eng.num_floors > 1 else ''}"
            f" = {flats * eng.num_floors} homes; each {unit.bhk_config}, {unit.total_carpet_area:.1f} m² carpet "
            f"(RERA). 230 mm walls between homes and around the core.",
            f"Every flat's Vastu is judged for the way its own main door faces: {unit.vastu_score:.0f}/100"
            + (f" (front flats {unit_turned.vastu_score:.0f}/100)." if unit_turned is not None else ".")]
    plan.design_notes = notes + tnotes + unit_notes + site.notes + engineering.pop("notes", [])
    plan.generation_params = dict(eng.generation_params)
    ls = copy.deepcopy(unit.layout_state) if unit.layout_state else {}
    if ls:
        ls["apartment"] = {"flats_per_floor": flats, "unit_rect": list(lay.slots[0].rect)}
        lq = importlib.import_module("floorplan_generator.quantities")
        ref = (eng.base_layout or {}).get("ref_quantities")
        ls["ref_quantities"] = dict(ref) if ref else lq.layout_quantities(plan)
    plan.layout_state = ls
    plan.edit_report = copy.deepcopy(getattr(unit, "edit_report", {}) or {})
    return plan


def _split_open_edges(walls, open_edges, le):
    """Walls on the open gallery's edges become a 1 m parapet (150 mm) — split where they run on."""
    if not open_edges:
        return walls
    out = []
    for w in walls:
        if getattr(w, "wall_type", "") != "external":
            out.append(w)
            continue
        horiz = abs(w.y1 - w.y2) < 1e-6
        lo, hi = sorted((w.x1, w.x2)) if horiz else sorted((w.y1, w.y2))
        line = w.y1 if horiz else w.x1
        spans = []
        for (x1, y1, x2, y2) in open_edges:
            e_h = abs(y1 - y2) < 1e-6
            if e_h != horiz:
                continue
            el = y1 if e_h else x1
            if abs(el - line) > 0.02:
                continue
            a, b = sorted((x1, x2)) if e_h else sorted((y1, y2))
            a, b = max(a, lo), min(b, hi)
            if b - a > 0.05:
                spans.append((a, b))
        if not spans:
            out.append(w)
            continue
        pos = lo
        for a, b in sorted(spans):
            for (p0, p1, kind) in ((pos, a, None), (a, b, "parapet")):
                if p1 - p0 < 0.05:
                    continue
                seg = (p0, line, p1, line) if horiz else (line, p0, line, p1)
                if kind:
                    out.append(le.WallSegment(*seg, 0.15, True, "parapet"))
                else:
                    out.append(le.WallSegment(*seg, w.thickness, True, w.wall_type))
            pos = max(pos, b)
        if hi - pos > 0.05:
            seg = (pos, line, hi, line) if horiz else (line, pos, line, hi)
            out.append(le.WallSegment(*seg, w.thickness, True, w.wall_type))
    return out
