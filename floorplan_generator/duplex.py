"""
Duplex — one home on two floors
================================
building_type 'Duplex' (LayoutEngine._stack_mode() == 'duplex') is one family's home split over two
floors, the way Indian duplexes are lived in:

  ground floor   living room, dining, kitchen (+ utility), pooja, store / servant, a toilet for guests,
                 and one bedroom that must not need the stair — the elders' room when there are elders
                 (with its accessible bath), else a guest bedroom when guests stay often, else (3 BHK and
                 up) a spare bedroom
  first floor    the master suite, the other bedrooms, a study, a family lounge (and a common bath if a
                 bedroom up there has none)

Both floors share the front band of apartment.floor_layout(mode='duplex'): the stair (a dog-leg turned to
run along the front) rising from the foyer, and the lift beside it when there is one — inside the home.
So the stair lands on exactly the same spot upstairs, by construction. Each floor's rooms are planned by
the engine (space program → band-spine solver → doors, windows) in the same rectangle behind the band:
downstairs the living room opens onto the foyer, upstairs the family lounge onto the upper lobby, both
through an open arch. Wet rooms are not forced to stack; the engineering notes list the pipe stacks.

The returned plan is the ground floor (drawing, site, cost basis); plan.upper_plan is the first floor
(its own drawing, and the 3D view's first level). Smart Re-Plan edits the ground floor's rooms.
"""

import copy
import importlib
from typing import List, Optional, Tuple


def _sp():
    return importlib.import_module("floorplan_generator.space_program")


def _ap():
    return importlib.import_module("floorplan_generator.apartment")


def split_program(full, family=None) -> Tuple[object, object, List[str]]:
    """Ground-floor and first-floor programs of one home (the full BHK program, divided)."""
    sp = _sp()
    beds = [g for g in full.groups if g.key.startswith("bed")]
    role = {g.key: g.primary.role for g in beds}
    down = (next((g for g in beds if role[g.key] == "ELDER_BEDROOM"), None)
            or next((g for g in beds if role[g.key] == "GUEST_BEDROOM"), None)
            or (beds[-1] if len(beds) >= 3 else None))
    up_beds = [g for g in beds if g is not down]
    if not up_beds:                                   # a 1BHK duplex: the bedroom goes up
        up_beds, down = beds, None

    ground_keys = {"kitchen", "pooja", "store", "servant", "cbath"}
    ground = [copy.deepcopy(g) for g in full.groups if g.key in ground_keys]
    if down is not None:
        ground.append(copy.deepcopy(down))
    if not any(g.key == "cbath" for g in ground):    # a toilet for guests downstairs
        ground.append(sp.RoomGroup(key="cbath", kind="single",
                                   rooms=[sp._room("cbath", "Common Toilet", "bathroom", 2.6, wet=True,
                                                   min_area=1.8, min_width=1.1)],
                                   bands=("rear", "private", "front"), growth=0.1, min_depth=1.5))

    upper = [copy.deepcopy(g) for g in up_beds]
    upper += [copy.deepcopy(g) for g in full.groups if g.key == "study"]
    if len(upper) < 2:                                # one bedroom up there: a study keeps the floor a floor
        upper.append(sp.RoomGroup(key="study", kind="single",
                                  rooms=[sp._room("study", "Study / Home Office", "study_room", 8.0, role="STUDY",
                                                  habitable=True, min_area=6.0, min_width=2.1)],
                                  bands=("front", "rear", "private"), growth=0.6, min_depth=2.4))
    if any(g.kind != "suite" for g in up_beds):      # a bedroom upstairs without its own bath
        upper.append(sp.RoomGroup(key="cbath", kind="single",
                                  rooms=[sp._room("cbath", "Common Bathroom", "bathroom", 3.0, wet=True,
                                                  min_area=2.2, min_width=1.2)],
                                  bands=("rear", "private", "front"), growth=0.1, min_depth=1.5))

    lounge = sp._room("living", "Family Lounge", "living_room", max(9.5, full.living.target_area * 0.6),
                      role="LIVING", circulation=True, habitable=True, min_area=7.5, min_width=2.7)
    hall = sp._room("dining", "Upper Hall", "dining_room", 4.0, role="DINING", circulation=True,
                    min_area=2.5, min_width=1.2)
    g_prog = sp.SpaceProgram(bhk=full.bhk, groups=ground, living=copy.deepcopy(full.living),
                             dining=copy.deepcopy(full.dining), notes=[], elder_friendly=full.elder_friendly)
    u_prog = sp.SpaceProgram(bhk=full.bhk, groups=upper, living=lounge, dining=hall, notes=[])
    names = lambda gs: ", ".join(g.primary.name.lower() for g in gs)    # noqa: E731
    notes = [f"Duplex ground floor: living, dining, kitchen{', ' + names([down]) if down else ''} and a toilet "
             "for guests" + (" — the elders never need the stair." if down is not None and role.get(down.key) ==
                             "ELDER_BEDROOM" else "."),
             f"Duplex first floor: {names(up_beds)}, a family lounge at the head of the stair"
             + (", a study" if any(g.key == "study" for g in upper) else "") + "."]
    return g_prog, u_prog, notes


def generate(eng, alternatives: int = 0):
    """Plan both floors of a duplex; returns the ground-floor plan with .upper_plan (or a list of them)."""
    le = importlib.import_module("floorplan_generator.layout_engine")
    sp, ap = _sp(), _ap()
    members = 4
    if eng.family_profile is not None:
        fp = eng.family_profile
        members = max(1, getattr(fp, "adults", 2) + getattr(fp, "children", 0) + getattr(fp, "elderly", 0))
    full = sp.build_program(eng.bhk_name, eng.family_profile, "Independent House", 1, eng.quality_level,
                            False, eng.lift_capacity, eng.stair_type)
    g_prog, u_prog, split_notes = split_program(full, eng.family_profile)
    for p in (g_prog, u_prog):
        p.external_stair = None
    sizing = g_prog if g_prog.target_area >= u_prog.target_area else u_prog
    base = eng.base_layout or {}
    unit_base = {k: v for k, v in base.items() if k not in ("apartment", "duplex")} if base else None
    attempt = None
    for boost in (1.0, 1.15, 1.3):                   # both floors must solve (no template fallback upstairs)
        lay, site, planner, _, stair, lift, report = ap.plan_floor(
            eng.plot_width, eng.plot_height, eng.orientation, eng.vastu_enabled, eng.bhk_name, eng.family_profile,
            eng.building_type, 2, eng.quality_level, eng.has_lift, eng.lift_capacity, eng.stair_type,
            eng.additional_components, members, 1, "duplex", None, copy.deepcopy(sizing), boost)
        if lay is None:
            break
        slot = lay.slots[0]

        def sub(program, intents=None, base_layout=None, slot=slot, lay=lay):
            e = le.LayoutEngine(slot.rect[2], slot.rect[3], eng.bhk_name, eng.vastu_enabled, eng.orientation,
                                eng.quality_level, eng.project_name, eng.family_description, None, False,
                                eng.lift_capacity, eng.stair_type, 1, "Independent House", {}, eng.family_profile,
                                design_intents=intents, base_layout=base_layout)
            e._fixed_site = le.SitePlanFixed(slot.rect[2], slot.rect[3], eng.orientation, ap._local_exclusion(lay))
            e._program_override = program
            return e

        up_unit = sub(u_prog).generate()
        if not up_unit.layout_state:
            continue
        g_engine = sub(g_prog, eng.design_intents, unit_base)
        grounds = g_engine.generate_alternatives(alternatives) if alternatives else [g_engine.generate()]
        if all(g.layout_state for g in grounds):
            attempt = (lay, site, planner, stair, lift, report, grounds, up_unit)
            break
    if attempt is None:                              # no foyer band fits, or a floor cannot be laid out
        eng._apartment_fallback = True
        plans = eng.generate_alternatives(alternatives) if alternatives else [eng.generate()]
        for p in plans:
            p.design_notes.insert(0, "Duplex: this plot cannot hold a foyer with the stair at the front and both "
                                     "floors' rooms, so the stair is inside and both floors repeat one plan.")
        return plans if alternatives else plans[0]
    lay, site, planner, stair, lift, report, grounds, up_unit = attempt
    for r in up_unit.rooms:                          # upstairs: the living is the family lounge, the spine a hall
        if r.role == "LIVING" or r.category == "LIVING":
            r.name, r.category = "Family Lounge", "LIVING"
        elif r.name == "Upper Hall" or r.category == "DINING":
            r.name, r.category = "Upper Hall", "CORRIDOR"
    for d in up_unit.doors:
        if d.room_name in ("Living & Dining", "Living Room"):
            d.room_name = "Family Lounge"
    for w in up_unit.windows:
        if w.room_name in ("Living & Dining", "Living Room"):
            w.room_name = "Family Lounge"

    plans = []
    for g_unit in grounds:
        ground = ap.compose(eng, copy.deepcopy(lay), copy.deepcopy(site), copy.deepcopy(planner), g_unit, None,
                            stair, lift, report, mode="duplex", floor="ground")
        upper = ap.compose(eng, copy.deepcopy(lay), copy.deepcopy(site), copy.deepcopy(planner), up_unit, None,
                           stair, lift, report, mode="duplex", floor="upper", hall_name="Upper Lobby")
        _finish(eng, ground, upper, g_unit, up_unit, split_notes)
        plans.append(ground)
    return plans if alternatives else plans[0]


def _finish(eng, ground, upper, g_unit, up_unit, split_notes) -> None:
    """One home: both floors' areas, notes, engineering and the link ground → upper."""
    hall_g = sum(r.w * r.h for r in ground.rooms if r.name == "Entrance Foyer")
    hall_u = sum(r.w * r.h for r in upper.rooms if r.name == "Upper Lobby")
    carpet = g_unit.total_carpet_area + up_unit.total_carpet_area + 0.9 * (hall_g + hall_u)
    ground.total_carpet_area = round(carpet, 2)
    upper.total_carpet_area = round(up_unit.total_carpet_area + 0.9 * hall_u, 2)
    ground.num_floors = upper.num_floors = 2
    info = {"floors": 2, "ground_rooms": [r.name for r in ground.rooms if r.role != "COMMON" and r.w * r.h > 0
                                          and r.category not in ("PARKING", "BIKE_PARKING", "GARDEN", "POOL",
                                                                  "WATER_TANK", "GENERATOR", "SECURITY")],
            "upper_rooms": [r.name for r in upper.rooms],
            "ground_vastu": g_unit.vastu_score, "upper_vastu": up_unit.vastu_score,
            "carpet_ground_m2": round(g_unit.total_carpet_area + 0.9 * hall_g, 1),
            "carpet_upper_m2": round(up_unit.total_carpet_area + 0.9 * hall_u, 1)}
    ground.engineering["duplex"] = info
    upper.engineering["duplex"] = dict(info)
    ground.upper_plan = upper
    ground.design_notes = (split_notes + [
        f"One home of {carpet:.1f} m² carpet: ground floor {info['carpet_ground_m2']:.1f} m², first floor "
        f"{info['carpet_upper_m2']:.1f} m². Vastu — ground floor {g_unit.vastu_score:.0f}/100 (judged at the main "
        f"door), bedrooms upstairs {up_unit.vastu_score:.0f}/100."] + ground.design_notes)
    upper.design_notes = split_notes[1:] + upper.design_notes
    ls = ground.layout_state or {}
    if ls:
        ls.pop("apartment", None)
        ls["duplex"] = {"floors": 2}
