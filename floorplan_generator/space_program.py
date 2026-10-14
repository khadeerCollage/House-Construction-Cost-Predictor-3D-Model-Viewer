"""
Space Program Builder
=====================
Turns (BHK + family profile + lifestyle spaces + building type + quality) into a
concrete *space program*: the list of rooms the house must contain, their target
and NBC-minimum sizes, how they are grouped into suites (bedroom + attached bath,
kitchen + utility, stair + lift core), and which rooms are reached through which.

Also holds the vertical-circulation engineering:
  - StairDesign: riser/tread/flight geometry per NBC 2016 Part 3 & 4
  - LiftDesign:  IS 14665 shaft sizing from the selected capacity

All dimensions in metres / m².
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import re

from floorplan_generator.room_specs import RoomCategory, NBC_ROOM_SPECS


# =============================================================================
# Quality / scale factors
# =============================================================================

QUALITY_FACTORS = {
    "economy": 0.88, "basic": 0.88,
    "standard": 1.00,
    "premium": 1.15,
    "luxury": 1.30,
}


def quality_factor(quality_level: str) -> float:
    q = (quality_level or "standard").lower()
    for key, val in QUALITY_FACTORS.items():
        if key in q:
            return val
    return 1.0


# =============================================================================
# Vertical circulation engineering
# =============================================================================

FLOOR_TO_FLOOR_M = 3.0       # typical Indian residential floor-to-floor height
MAX_RISER_M = 0.19           # NBC 2016 Part 4: residential max riser 190 mm
MIN_TREAD_M = 0.25           # NBC 2016 Part 4: residential min tread 250 mm
MAX_RISERS_PER_FLIGHT = 15   # NBC 2016 Part 4


@dataclass
class StairDesign:
    """Engineering design of one staircase between two floors."""
    stair_type: str
    flight_width: float        # clear width of one flight (m)
    footprint_w: float         # size across the flights (m)
    footprint_l: float         # size along the walking line incl. landings (m)
    risers: int
    riser_height: float
    tread: float
    landing_depth: float
    flights: int
    notes: List[str] = field(default_factory=list)

    @property
    def blondel(self) -> float:
        """2R + T comfort check (NBC recommends 550–700 mm)."""
        return 2 * self.riser_height + self.tread


def design_staircase(stair_type: str, building_type: str = "Independent House",
                     floor_height: float = FLOOR_TO_FLOOR_M,
                     external: bool = False) -> StairDesign:
    """
    Size a staircase from first principles.

    Risers: n = ceil(H / 0.175) so each riser ≤ 175 mm (comfortable, ≤ NBC 190).
    Treads: 250 mm (NBC minimum for residential).
    Flight width: 1.0 m dwellings, 1.25 m apartments (NBC Part 4 Table 7),
                  0.9 m for an external terrace stair.
    """
    st = (stair_type or "dog_leg").lower()
    is_apartment = "apartment" in (building_type or "").lower()
    width = 0.9 if external else (1.25 if is_apartment else 1.0)
    risers = max(2, math.ceil(floor_height / 0.175))
    riser_h = floor_height / risers
    tread = MIN_TREAD_M
    landing = width            # landing depth ≥ flight width (NBC)
    notes: List[str] = []

    if st == "straight":
        flights = 2 if risers > MAX_RISERS_PER_FLIGHT else 1
        r1 = math.ceil(risers / flights)
        run = (r1 - 1) * tread
        fw, fl = width, flights * run + (flights - 1) * landing
        if flights > 1:
            notes.append(f"Mid-landing provided: {risers} risers exceed {MAX_RISERS_PER_FLIGHT} per flight (NBC).")
    elif st == "l_shaped":
        flights = 2
        r1 = math.ceil(risers / 2)
        run = (r1 - 1) * tread
        fw, fl = width + run, width + run          # quarter-turn landing at the corner
    elif st == "open_well":
        flights = 3
        r1 = math.ceil(risers / 3)
        run = (r1 - 1) * tread
        well = 0.6
        fw = 2 * width + well
        fl = run + 2 * landing
    elif st == "spiral":
        flights = 1
        dia = max(1.6, 2 * width + 0.2)
        fw = fl = dia
        notes.append("Spiral stair: permitted as a secondary/terrace stair only; not as the sole fire-escape route (NBC Part 4).")
    elif st == "zigzag":
        flights = 2
        r1 = math.ceil(risers / 2)
        run = (r1 - 1) * tread
        fw, fl = 2 * width + 0.15, run + 2 * landing
    else:  # dog_leg (default)
        st = "dog_leg"
        flights = 2
        r1 = math.ceil(risers / 2)
        run = (r1 - 1) * tread
        fw = 2 * width + 0.1        # 100 mm gap between flights
        fl = run + landing

    design = StairDesign(
        stair_type=st, flight_width=width,
        footprint_w=round(fw, 2), footprint_l=round(fl, 2),
        risers=risers, riser_height=round(riser_h, 4), tread=tread,
        landing_depth=landing, flights=flights, notes=notes,
    )
    if not (0.55 <= design.blondel <= 0.70):
        design.notes.append(f"2R+T = {design.blondel*1000:.0f} mm (outside 550–700 mm comfort band).")
    return design


@dataclass
class LiftDesign:
    persons: int
    rated_load_kg: int
    shaft_w: float
    shaft_d: float
    pit_depth: float = 1.5
    overhead: float = 4.2
    notes: List[str] = field(default_factory=list)


def design_lift(capacity: str) -> LiftDesign:
    """IS 14665: 68 kg per person; shaft sizes from the selected capacity."""
    m = re.search(r"(\d+)\s*Person", capacity or "", re.IGNORECASE)
    persons = int(m.group(1)) if m else 4
    shafts = {4: (1.5, 1.5), 6: (1.6, 1.7), 8: (1.8, 2.0)}
    sw, sd = shafts.get(persons, (1.5, 1.5) if persons <= 4 else (1.8, 2.0))
    return LiftDesign(
        persons=persons, rated_load_kg=persons * 68, shaft_w=sw, shaft_d=sd,
        notes=[f"IS 14665: {persons} persons × 68 kg = {persons * 68} kg rated load; "
               f"pit 1.5 m, overhead 4.2 m, machine-room-less traction drive."],
    )


# Lift car (inside size, width along the door × depth) by persons — IS 14665 / EN 81-20 car area for the
# rated load (68 kg a person), and what it can carry (EN 81-70 accessible car types).
LIFT_CARS = {4: (0.85, 0.95), 6: (1.0, 1.25), 8: (1.1, 1.4)}
LIFT_DOOR_CLEAR = {4: 0.8, 6: 0.8, 8: 0.9}
CAR_NEEDS = [("wheelchair", "a wheelchair user", 1.0, 1.25),
             ("helper", "a wheelchair with a helper", 1.1, 1.4),
             ("stretcher", "a stretcher", 1.1, 2.1)]


def lift_car(persons: int) -> Dict:
    """Car inside size and what fits in it (plain words for the owner)."""
    p = 4 if persons <= 4 else (6 if persons <= 6 else 8)
    w, d = LIFT_CARS[p]
    fits = {k: (w >= cw - 1e-6 and d >= cd - 1e-6) for k, _, cw, cd in CAR_NEEDS}
    yes = [txt for k, txt, _, _ in CAR_NEEDS if fits[k]]
    no = [(txt, cw, cd) for k, txt, cw, cd in CAR_NEEDS if not fits[k]]
    words = (f"Car {w:.2f} × {d:.2f} m inside · door {LIFT_DOOR_CLEAR[p]:.1f} m. "
             + (f"Fits {yes[-1]}." if yes else "Too small for a wheelchair (needs 1.0 × 1.25 m).")
             + (f" {no[0][0][0].upper() + no[0][0][1:]} needs {no[0][1]:.1f} × {no[0][2]:.2f} m." if yes and no else ""))
    return {"w": w, "d": d, "door": LIFT_DOOR_CLEAR[p], "fits": fits, "words": words}


# =============================================================================
# Program data model
# =============================================================================

@dataclass
class ProgramRoom:
    key: str
    name: str
    category: str               # RoomCategory value
    target_area: float
    min_area: float
    min_width: float
    color: str
    role: str = ""              # vastu role key (MASTER_BEDROOM, CHILD_BEDROOM, ...)
    parent: Optional[str] = None        # reached through this room (attached)
    circulation: bool = False
    habitable: bool = False     # needs daylight / exterior window (NBC Part 3 §8)
    wet: bool = False


@dataclass
class RoomGroup:
    """A set of rooms that occupy one column of a band (a 'suite')."""
    key: str
    kind: str                   # single | suite | service | servant | core | passage
    rooms: List[ProgramRoom]
    bands: Tuple[str, ...]      # allowed bands: front / private / rear / cell
    growth: float = 1.0         # share of surplus width this group absorbs
    min_depth: float = 2.4
    stair: Optional[StairDesign] = None
    lift: Optional[LiftDesign] = None

    @property
    def primary(self) -> ProgramRoom:
        return self.rooms[0]

    @property
    def target_area(self) -> float:
        return sum(r.target_area for r in self.rooms)


@dataclass
class SpaceProgram:
    bhk: str
    groups: List[RoomGroup]
    living: ProgramRoom
    dining: ProgramRoom
    notes: List[str] = field(default_factory=list)
    stair: Optional[StairDesign] = None
    lift: Optional[LiftDesign] = None
    external_stair: Optional[StairDesign] = None
    elder_friendly: bool = False
    lift_outside: Optional[str] = None      # 'entrance' | 'side': a lift tower outside the house walls

    @property
    def target_area(self) -> float:
        return self.living.target_area + self.dining.target_area + sum(g.target_area for g in self.groups)

    @property
    def min_area(self) -> float:
        base = self.living.min_area + 6.0 + sum(r.min_area for g in self.groups for r in g.rooms)
        return base * 1.18   # walls + residual circulation


def _spec(key: str):
    return NBC_ROOM_SPECS[key]


def _room(key: str, name: str, spec_key: str, target: float, *, role: str = "",
          parent: Optional[str] = None, circulation: bool = False,
          habitable: bool = False, wet: bool = False,
          min_area: Optional[float] = None, min_width: Optional[float] = None) -> ProgramRoom:
    s = _spec(spec_key)
    return ProgramRoom(
        key=key, name=name, category=s.category.value,
        target_area=round(target, 2),
        min_area=min_area if min_area is not None else s.min_area,
        min_width=min_width if min_width is not None else s.min_width,
        color=s.color, role=role, parent=parent, circulation=circulation,
        habitable=habitable, wet=wet,
    )


# Base targets (m², Standard quality) per BHK
_BASE = {
    1: dict(living=13.0, dining=7.0, master=12.5, bed=10.5, kitchen=7.5, mbath=3.6, bath=3.0),
    2: dict(living=15.5, dining=9.0, master=13.5, bed=11.0, kitchen=8.5, mbath=4.0, bath=3.2),
    3: dict(living=19.0, dining=11.0, master=15.0, bed=12.0, kitchen=10.0, mbath=4.5, bath=3.4),
    4: dict(living=23.0, dining=13.0, master=16.5, bed=12.5, kitchen=11.5, mbath=5.0, bath=3.6),
}


def build_program(bhk: str,
                  family=None,
                  building_type: str = "Independent House",
                  num_floors: int = 1,
                  quality_level: str = "Standard",
                  has_lift: bool = False,
                  lift_capacity: str = "4 Person",
                  stair_type: str = "dog_leg",
                  lift_position: str = "inside") -> SpaceProgram:
    """
    Build the room program.

    `family` is a FamilyProfile (duck-typed: adults, children, elderly,
    guests_frequent, needs_home_office, needs_pooja_room, needs_servant_quarter,
    needs_store_room). When absent, sensible defaults for the BHK are used.
    """
    m = re.search(r"(\d)", bhk or "2")
    n_bed = max(1, min(4, int(m.group(1)) if m else 2))
    bhk_name = f"{n_bed}BHK"
    base = _BASE[n_bed]
    qf = quality_factor(quality_level)
    notes: List[str] = []

    adults = getattr(family, "adults", 2) if family else 2
    children = getattr(family, "children", 0) if family else 0
    elderly = getattr(family, "elderly", 0) if family else 0
    guests = getattr(family, "guests_frequent", False) if family else False
    want_office = getattr(family, "needs_home_office", False) if family else False
    want_pooja = getattr(family, "needs_pooja_room", n_bed >= 3) if family else (n_bed >= 3)
    want_servant = getattr(family, "needs_servant_quarter", False) if family else False
    want_store = getattr(family, "needs_store_room", False) if family else False
    members = adults + children + elderly
    fam_f = 1.0 + min(0.30, max(0, members - 4) * 0.05)

    btype = (building_type or "").lower()
    is_apartment = "apartment" in btype
    is_duplex = "duplex" in btype
    floors = max(1, int(num_floors or 1))

    living = _room("living", "Living Room", "living_room", base["living"] * qf * fam_f,
                   role="LIVING", circulation=True, habitable=True, min_area=9.5, min_width=3.0)
    dining = _room("dining", "Dining & Family Lounge", "dining_room",
                   (base["dining"] + max(0, members - 4) * 0.8) * qf,
                   role="DINING", circulation=True, min_area=6.0, min_width=2.4)

    groups: List[RoomGroup] = []

    # ── Bedroom roles ─────────────────────────────────────────────────────
    roles: List[str] = ["MASTER_BEDROOM"]
    if elderly and len(roles) < n_bed:
        roles.append("ELDER_BEDROOM")
    if children and len(roles) < n_bed:
        roles.append("CHILD_BEDROOM")
    if guests and len(roles) < n_bed:
        roles.append("GUEST_BEDROOM")
    while len(roles) < n_bed:
        roles.append("CHILD_BEDROOM" if "CHILD_BEDROOM" not in roles else "BEDROOM")

    # Attached baths: master always; elders always (accessibility); 3BHK+: one more
    attached_roles = {"MASTER_BEDROOM", "ELDER_BEDROOM"}
    if n_bed >= 3:
        for r in roles:
            if r not in attached_roles:
                attached_roles.add(r)
                break
    if n_bed >= 4 and qf >= 1.0:
        attached_roles.update(roles[:3])

    bed_names = {
        "MASTER_BEDROOM": "Master Bedroom", "ELDER_BEDROOM": "Elders' Bedroom",
        "CHILD_BEDROOM": "Children's Bedroom", "GUEST_BEDROOM": "Guest Bedroom",
        "BEDROOM": "Bedroom",
    }
    used_names: Dict[str, int] = {}

    def unique(name: str) -> str:
        used_names[name] = used_names.get(name, 0) + 1
        return name if used_names[name] == 1 else f"{name} {used_names[name]}"

    att_count = 0
    for i, role in enumerate(roles):
        key = f"bed{i + 1}"
        is_master = role == "MASTER_BEDROOM"
        tgt = (base["master"] if is_master else base["bed"]) * qf
        if role == "ELDER_BEDROOM":
            tgt = max(tgt, 12.0 * qf)      # wheelchair turning circle 1.5 m
        bed = _room(key, unique(bed_names[role]), "master_bedroom" if is_master else "bedroom", tgt,
                    role=role, habitable=True,
                    min_area=9.5 if is_master else 7.5,
                    min_width=2.7 if is_master else 2.6)
        rooms = [bed]
        kind = "single"
        if role in attached_roles:
            att_count += 1
            bath_t = (base["mbath"] if is_master else base["bath"]) * qf
            if role == "ELDER_BEDROOM":
                bath_t = max(bath_t, 4.0)  # accessible toilet ≥ 1.5 m turning space
            bath_name = "Attached Bathroom" if att_count == 1 else f"Attached Bathroom {att_count}"
            rooms.append(_room(f"{key}_bath", bath_name, "attached_bath", bath_t, parent=key,
                               wet=True, min_area=2.8, min_width=1.2 if role != "ELDER_BEDROOM" else 1.5))
            rooms.append(_room(f"{key}_dress", "Dress / Wardrobe" if is_master else unique("Wardrobe"),
                               "corridor", 2.4 * qf, parent=key, min_area=1.2, min_width=1.0))
            kind = "suite"
        growth = 2.0 if is_master else 1.5
        groups.append(RoomGroup(key=key, kind=kind, rooms=rooms,
                                bands=("rear", "front", "private"), growth=growth,
                                min_depth=3.0 if is_master else 2.7))

    # ── Common bathroom (when some bedroom lacks an attached bath) ────────
    if len(attached_roles & set(roles)) < len(roles) or n_bed == 1:
        groups.append(RoomGroup(
            key="cbath", kind="single",
            rooms=[_room("cbath", "Common Bathroom", "bathroom", base["bath"] * qf, wet=True,
                         min_area=2.2, min_width=1.2)],
            bands=("rear", "private", "front"), growth=0.1, min_depth=1.5))

    # ── Kitchen + utility ─────────────────────────────────────────────────
    k_rooms = [_room("kitchen", "Kitchen", "kitchen", base["kitchen"] * qf * fam_f,
                     role="KITCHEN", habitable=True, wet=True, min_area=5.0, min_width=2.1)]
    k_rooms.append(_room("utility", "Utility / Wash", "utility_room", (2.6 + 0.4 * (n_bed - 1)) * qf,
                         parent="kitchen", wet=True, min_area=1.8, min_width=1.2))
    groups.append(RoomGroup(key="kitchen", kind="service", rooms=k_rooms,
                            bands=("rear", "front", "private"), growth=0.9, min_depth=2.7))

    # ── Lifestyle spaces ──────────────────────────────────────────────────
    if want_pooja:
        groups.append(RoomGroup(
            key="pooja", kind="single",
            rooms=[_room("pooja", "Pooja Room", "pooja_room", max(2.2, 2.6 * qf), role="POOJA",
                         min_area=1.5, min_width=1.2)],
            bands=("front", "rear", "private"), growth=0.1, min_depth=1.5))
    if want_office:
        groups.append(RoomGroup(
            key="study", kind="single",
            rooms=[_room("study", "Study / Home Office", "study_room", 8.0 * qf, role="STUDY",
                         habitable=True, min_area=6.0, min_width=2.1)],
            bands=("front", "rear", "private"), growth=0.6, min_depth=2.4))
    if want_store:
        groups.append(RoomGroup(
            key="store", kind="single",
            rooms=[_room("store", "Store Room", "utility_room", 3.0, role="STORE",
                         min_area=2.0, min_width=1.2)],
            bands=("rear", "private", "front"), growth=0.05, min_depth=1.5))
        groups[-1].rooms[0].category = RoomCategory.STORE.value
        groups[-1].rooms[0].color = "#D6C6A8"
    if want_servant:
        groups.append(RoomGroup(
            key="servant", kind="servant",
            rooms=[_room("servant", "Servant Quarter", "servant_room", 7.0, role="SERVANT",
                         habitable=True, min_area=5.0, min_width=2.1),
                   _room("servant_wc", "Servant Toilet", "bathroom", 2.0, parent="servant", wet=True,
                         min_area=1.2, min_width=0.9)],
            bands=("rear", "front", "private"), growth=0.3, min_depth=2.7))

    # ── Vertical circulation ──────────────────────────────────────────────
    stair = None
    external_stair = None
    lift = None
    lift_outside = None
    multi_level = floors > 1 or is_duplex or is_apartment
    if multi_level:
        stair = design_staircase(stair_type, building_type)
        if has_lift and (floors > 1 or is_apartment or is_duplex):
            lift = design_lift(lift_capacity)
            if not is_apartment and (lift_position or "inside") in ("entrance", "side"):
                lift_outside = lift_position          # a tower outside the walls, see site_planner / layout_engine
        core_rooms = [_room("stairs", "Staircase", "staircase", stair.footprint_w * stair.footprint_l,
                            role="STAIRS", min_area=3.0, min_width=stair.footprint_w)]
        if lift and not lift_outside:
            core_rooms.append(_room("lift", "Lift / Elevator", "lift_shaft", lift.shaft_w * lift.shaft_d,
                                    role="LIFT", parent=None, min_area=2.25, min_width=lift.shaft_w))
        groups.append(RoomGroup(key="core", kind="core", rooms=core_rooms,
                                bands=("cell", "rear", "front", "private"), growth=0.0,
                                min_depth=stair.footprint_w, stair=stair, lift=None if lift_outside else lift))
        level = "Duplex" if is_duplex else ("Apartment" if is_apartment else f"G+{floors - 1}")
        notes.append(
            f"Staircase ({stair.stair_type.replace('_', '-')}, {level}): {stair.risers} risers × "
            f"{stair.riser_height * 1000:.0f} mm, tread {stair.tread * 1000:.0f} mm, flight width "
            f"{stair.flight_width:.2f} m, 2R+T = {stair.blondel * 1000:.0f} mm (NBC 2016 Part 4).")
        notes.extend(stair.notes)
        if lift:
            notes.extend(lift.notes)
            notes.append("Lift car: " + lift_car(lift.persons)["words"])
        if lift_outside:
            notes.append("Lift outside the home " + ("beside the main door" if lift_outside == "entrance"
                                                     else "against the side wall")
                         + ": a 230 mm RCC tower with its own landing on every floor, so each floor is reached "
                           "without walking through another — and the rooms keep the shaft's floor space.")
    else:
        if has_lift:
            notes.append("Lift omitted: single-storey house has no upper level to serve.")
        external_stair = design_staircase("straight", building_type, external=True)
        notes.append(
            "Single storey: terrace access by an external straight stair (0.9 m flight) in the "
            "S/W setback — keeps the Brahmasthan and NE free and saves ~10 m² of carpet area.")

    program = SpaceProgram(
        bhk=bhk_name, groups=groups, living=living, dining=dining, notes=notes,
        stair=stair, lift=lift, external_stair=external_stair,
        elder_friendly=elderly > 0, lift_outside=lift_outside,
    )
    if elderly:
        notes.append("Elders' bedroom on ground level with attached accessible toilet "
                     "(1.5 m turning circle, 900 mm door, grab bars) per NBC Part 3 Annex B.")
    return program


def trim_program(program: SpaceProgram, available_area: float) -> List[str]:
    """
    Drop/shrink optional spaces when the buildable footprint is too small.
    Returns human-readable notes describing what was trimmed.
    """
    removed: List[str] = []
    order = ["servant", "store", "study", "pooja"]
    for key in order:
        if program.min_area <= available_area:
            break
        for g in list(program.groups):
            if g.key == key:
                program.groups.remove(g)
                removed.append(g.primary.name)
    # Drop dress/wardrobe areas next
    if program.min_area > available_area:
        for g in program.groups:
            if g.kind == "suite":
                g.rooms = [r for r in g.rooms if "Wardrobe" not in r.name and "Dress" not in r.name]
    # Then downgrade bedrooms (keep master)
    while program.min_area > available_area:
        beds = [g for g in program.groups if g.primary.category == RoomCategory.BEDROOM.value
                and g.primary.role != "MASTER_BEDROOM"]
        if not beds:
            break
        g = beds[-1]
        program.groups.remove(g)
        removed.append(g.primary.name)
        n = sum(1 for gg in program.groups if gg.primary.category == RoomCategory.BEDROOM.value)
        program.bhk = f"{n}BHK"
        if not any(gg.key == "cbath" for gg in program.groups):
            program.groups.append(RoomGroup(
                key="cbath", kind="single",
                rooms=[_room("cbath", "Common Bathroom", "bathroom", 3.0, wet=True, min_area=2.2, min_width=1.2)],
                bands=("rear", "private", "front"), growth=0.1, min_depth=1.5))
    notes = []
    if removed:
        notes.append("Buildable footprint too small for the full brief — omitted: " + ", ".join(removed) + ".")
    return notes
