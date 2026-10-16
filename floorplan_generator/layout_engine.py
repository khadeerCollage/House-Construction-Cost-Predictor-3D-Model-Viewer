"""
V-HSP Layout Engine
====================
Vastu-Guided Hierarchical Space Partitioning — converts plot dimensions, BHK,
family intent, lifestyle spaces, engineering options and site amenities into a
complete, professional floor plan (rooms, walls with real openings, doors,
windows, staircases, site amenities, dimensions) plus an engineering report.

Pipeline:
  Stage 1: Space program        — rooms the family needs (space_program.py)
  Stage 2: Site planning        — setbacks, Vastu open space, footprint (site_planner.py)
  Stage 3: Zoning & geometry    — band-spine solver, optimised for Vastu + NBC (layout_solver.py)
  Stage 4: Doors & arches       — every room reachable from the main door
  Stage 5: Main entrance        — best auspicious pada on the road facade
  Stage 6: Site amenities       — parking, sump, genset, pool, garden, security, terrace stair
  Stage 7: Walls & openings     — 230 mm envelope, 115 mm partitions, compound wall, cut openings
  Stage 8: Windows              — NBC Part 3 glazing (≥ 1/10 floor area), more on N/E
  Stage 9: Dimensions & audit   — 3-tier dimensions, Vastu score, RERA carpet, engineering notes

Plan frame: metres, origin at the plot's bottom-left corner, +x right, +y up,
and the road / entrance side ("orientation" = facing) is the TOP edge.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math

from floorplan_generator.room_specs import (
    BHKConfig, RoomTemplate, WallThickness, RoomCategory,
    get_bhk_config, recommend_bhk_for_plot,
)
from floorplan_generator.vastu_engine import (
    VastuEngine, VastuZone, ROOM_IMPORTANCE,
    north_edge_for_facing, plan_sides_for_facing, pada_for_point,
)


# =============================================================================
# Output Dataclasses
# =============================================================================

@dataclass
class RoomRect:
    """A placed room with its physical bounds (wall centre-line rectangle)."""
    name: str
    category: str       # RoomCategory value string
    x: float            # left edge, meters from plot origin
    y: float            # bottom edge, meters from plot origin
    w: float            # width in meters
    h: float            # height in meters
    color: str           # hex color code for visualization
    area: float = 0.0    # carpet area in m² (computed)
    role: str = ""       # Vastu role (MASTER_BEDROOM, CHILD_BEDROOM, …) when known
    key: str = ""        # stable space-program key ('bed1', 'kitchen', …) for intent editing

    def __post_init__(self):
        if self.area == 0.0:
            self.area = round(self.w * self.h, 2)

    @property
    def cx(self) -> float:
        """Center X coordinate."""
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        """Center Y coordinate."""
        return self.y + self.h / 2

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def top(self) -> float:
        return self.y + self.h

    def overlaps(self, other: "RoomRect", tol: float = 1e-6) -> bool:
        """Check if two rooms overlap (excluding touching edges, float-tolerant)."""
        return (
            self.x < other.right - tol and self.right > other.x + tol and
            self.y < other.top - tol and self.top > other.y + tol
        )

    def shares_wall_with(self, other: "RoomRect", tolerance: float = 0.05) -> Optional[str]:
        """
        Check if this room shares a wall segment with another room.

        Returns:
            Wall side ('N', 'S', 'E', 'W') from this room's perspective,
            or None if no shared wall.
        """
        min_shared = 0.5  # minimum shared length in meters for a door

        if abs(self.right - other.x) < tolerance:
            overlap = min(self.top, other.top) - max(self.y, other.y)
            if overlap >= min_shared:
                return "E"
        if abs(self.x - other.right) < tolerance:
            overlap = min(self.top, other.top) - max(self.y, other.y)
            if overlap >= min_shared:
                return "W"
        if abs(self.top - other.y) < tolerance:
            overlap = min(self.right, other.right) - max(self.x, other.x)
            if overlap >= min_shared:
                return "N"
        if abs(self.y - other.top) < tolerance:
            overlap = min(self.right, other.right) - max(self.x, other.x)
            if overlap >= min_shared:
                return "S"
        return None


@dataclass
class WallSegment:
    """A wall segment with physical coordinates and properties."""
    x1: float
    y1: float
    x2: float
    y2: float
    thickness: float  # meters (0.23 external, 0.115 internal)
    is_external: bool
    wall_type: str = ""   # 'external' | 'internal' | 'compound' (informational)

    @property
    def length(self) -> float:
        return math.sqrt((self.x2 - self.x1) ** 2 + (self.y2 - self.y1) ** 2)

    @property
    def is_horizontal(self) -> bool:
        return abs(self.y2 - self.y1) < 0.01

    @property
    def is_vertical(self) -> bool:
        return abs(self.x2 - self.x1) < 0.01


@dataclass
class DoorPlacement:
    """
    A door placed in a wall opening.

    (x, y) is the low-coordinate end of the opening on the wall line; the
    opening spans [x, x+width] for N/S walls and [y, y+width] for E/W walls.
    The hinge is at (x, y) and the leaf swings into `room_name`.
    """
    x: float
    y: float
    width: float        # door width in meters
    wall_side: str      # 'N', 'S', 'E', 'W' — which wall of the room (plan-relative)
    swing_direction: str  # 'LEFT', 'RIGHT'
    room_name: str      # which room this door belongs to
    is_main_entrance: bool = False


@dataclass
class WindowPlacement:
    """A window placed in an external wall."""
    x: float            # center X position in meters
    y: float            # center Y position in meters
    width: float        # window width in meters
    height: float       # window height in meters (for E/W windows: length along the wall)
    wall_side: str      # 'N', 'S', 'E', 'W'
    room_name: str


@dataclass
class StaircaseRect:
    """A placed staircase with its physical bounds."""
    name: str  # "Staircase", "Lift"
    stair_type: str  # "straight", "dog_leg", "open_well", "lift"
    x: float
    y: float
    w: float  # width in meters
    h: float  # height/depth in meters
    direction: str  # "up" direction — 'N', 'S', 'E', 'W'
    num_steps: int = 15
    tread_depth_m: float = 0.25  # 250mm per step
    riser_height_m: float = 0.175  # 175mm per step
    has_landing: bool = True
    landing_depth_m: float = 1.0


@dataclass
class DimLine:
    """A dimension annotation line."""
    x1: float
    y1: float
    x2: float
    y2: float
    offset: float       # perpendicular offset from wall in meters (negative = outside, below/left)
    label: str          # dimension text like "3,500"
    tier: int           # 1=rooms/setbacks, 2=building envelope, 3=overall plot


@dataclass
class GeneratedFloorPlan:
    """Complete output of the layout engine — everything needed to render."""
    plot_width: float
    plot_height: float
    orientation: str          # road / entrance facing direction (N/E/S/W) — drawn at the top
    rooms: List[RoomRect] = field(default_factory=list)
    walls: List[WallSegment] = field(default_factory=list)
    doors: List[DoorPlacement] = field(default_factory=list)
    windows: List[WindowPlacement] = field(default_factory=list)
    staircases: List[StaircaseRect] = field(default_factory=list)
    dimensions: List[DimLine] = field(default_factory=list)
    vastu_score: float = 0.0
    vastu_enabled: bool = True
    total_carpet_area: float = 0.0
    total_built_up_area: float = 0.0
    bhk_config: str = ""       # '1BHK', '2BHK', etc.
    family_description: str = ""
    quality_level: str = "Standard"
    project_name: str = "Residential Floor Plan"
    has_lift: bool = False
    lift_capacity: str = "4 Person"
    building_type: str = "Independent House"
    num_floors: int = 1
    additional_components: Dict = field(default_factory=dict)
    # ── Engineering upgrade fields ─────────────────────────────────────────
    north_edge: str = ""                       # plan edge pointing North ('N' top, 'E' right, …)
    building_footprint: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    setbacks: Dict[str, float] = field(default_factory=dict)   # by compass direction
    stair_type: str = "dog_leg"
    openings: List[Tuple[float, float, float, float]] = field(default_factory=list)
    design_notes: List[str] = field(default_factory=list)
    engineering: Dict = field(default_factory=dict)
    vastu_breakdown: Dict = field(default_factory=dict)
    # ── Intent editing (see intent_engine.py) ──────────────────────────────
    generation_params: Dict = field(default_factory=dict)   # LayoutEngine kwargs that rebuild this plan
    layout_state: Dict = field(default_factory=dict)        # solver arrangement + room centroids by key
    edit_report: Dict = field(default_factory=dict)         # how an intent re-plan met the user's wishes


# =============================================================================
# Layout Engine
# =============================================================================

# Architectural grid module (snap dimensions to this increment)
GRID_MODULE = 0.15  # 150mm = half-brick module

# Room categories that live outside the house envelope
SITE_CATEGORIES = {
    RoomCategory.PARKING.value, RoomCategory.BIKE_PARKING.value, RoomCategory.GARDEN.value,
    RoomCategory.POOL.value, RoomCategory.WATER_TANK.value, RoomCategory.GENERATOR.value,
    RoomCategory.SECURITY.value,
}

# Program relaxation steps (_relax_program): 0-3 always tried; 4-9 = compact-home steps (an attached bath shared,
# one at a time ×3 · pooja shelf · NBC minimum widths · one bathroom), only when no band layout exists at all.
# A plan's layout_state['relax_level'] = how many were applied (a re-plan replays them).
MAX_RELAX = 10


def _snap_to_grid(value: float, module: float = GRID_MODULE) -> float:
    """Snap a dimension to the nearest architectural grid module."""
    return round(value / module) * module


def _clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max."""
    return max(min_val, min(max_val, value))


LIFT_TOWER = "Lift / Elevator (External)"     # a lift tower outside the house walls
TERRACE_STAIR = "Terrace Staircase (External)"
# where the owner wants the outside terrace stair, in the words of the drawing (road at the top)
STAIR_SIDES = {"front": "top", "back": "bottom", "left": "left", "right": "right"}
ZONE_NAMES = {"N": "North", "NE": "North-East", "E": "East", "SE": "South-East", "S": "South",
              "SW": "South-West", "W": "West", "NW": "North-West", "CENTER": "centre"}


def _is_site_room(r) -> bool:
    return r.category in SITE_CATEGORIES or "External" in r.name


def audit_vastu(rooms, doors, plot_w: float, plot_h: float, facing: str,
                footprint=None, setbacks_compass: Optional[Dict[str, float]] = None,
                vastu_enabled: bool = True) -> Tuple[float, Dict]:
    """
    Plan-level Vastu score shared by the generator and the interactive editor.

    House rooms are scored on the building's mandala, site amenities on the plot
    mandala (role-aware for bedrooms), plus the main door's 32-pada score,
    Brahmasthan violations (−8 each) and the N/E vs S/W open-space balance (−4).
    """
    north = north_edge_for_facing(facing)
    building = [r for r in rooms if not _is_site_room(r)]
    if not footprint or footprint[2] <= 0:
        if building:
            x0 = min(r.x for r in building)
            y0 = min(r.y for r in building)
            footprint = (x0, y0, max(r.right for r in building) - x0, max(r.top for r in building) - y0)
        else:
            footprint = (0.0, 0.0, plot_w, plot_h)
    fx, fy, fw, fd = footprint
    house = VastuEngine(fw, fd, north, fx, fy)
    plot = VastuEngine(plot_w, plot_h, north)
    placements, weights, rows = [], [], []
    for r in rooms:
        if "Dress" in r.name or "Wardrobe" in r.name or "Duct" in r.name:
            continue
        role = getattr(r, "role", "")
        key = role if role in ("MASTER_BEDROOM", "CHILD_BEDROOM", "GUEST_BEDROOM", "ELDER_BEDROOM") else r.category
        eng = plot if _is_site_room(r) else house
        zone = eng.get_zone_for_position(r.cx, r.cy).value
        placements.append((key, zone))
        weights.append(ROOM_IMPORTANCE.get(key, 1.0))
        rows.append({"room": r.name, "key": key, "zone": zone,
                     "score": eng.get_room_score(key, zone), "taboo": eng.is_taboo(key, zone)})
    main = next((d for d in doors if d.is_main_entrance), None)
    extra = []
    pada = None
    if main is not None:
        if main.wall_side in ("N", "S"):
            px, py = main.x + main.width / 2, main.y
        else:
            px, py = main.x, main.y + main.width / 2
        pada = pada_for_point(px, py, footprint, facing)
        extra.append((pada.score, ROOM_IMPORTANCE["ENTRANCE"]))
    score = house.compute_plan_score(placements, weights, extra_scores=extra)
    bcx, bcy = fx + fw / 2, fy + fd / 2
    brahma = [r.name for r in building if r.x < bcx < r.right and r.y < bcy < r.top
              and r.category in ("BATHROOM", "KITCHEN", "STAIRS", "LIFT", "STORE", "UTILITY")]
    score -= 8 * len(brahma)
    sb = setbacks_compass or {}
    ne_open, sw_open = sb.get("N", 0) + sb.get("E", 0), sb.get("S", 0) + sb.get("W", 0)
    if vastu_enabled and sb and ne_open + 0.05 < sw_open:
        score -= 4
    score = max(0.0, min(100.0, score))
    return score, {
        "rooms": rows,
        "entrance_pada": pada.pada_id if pada else None,
        "entrance_score": pada.score if pada else None,
        "brahmasthan_violations": brahma,
        "open_space_ne_m": round(ne_open, 2), "open_space_sw_m": round(sw_open, 2),
    }


def SitePlanFixed(width: float, depth: float, facing: str, door_exclusion=None):
    """Site plan of a flat inside an apartment floor: the whole rectangle is the home, no setbacks."""
    from floorplan_generator.site_planner import SitePlan
    return SitePlan(plot_w=width, plot_h=depth, facing=facing,
                    setbacks={"top": 0.0, "bottom": 0.0, "left": 0.0, "right": 0.0},
                    footprint=(0.0, 0.0, width, depth), door_exclusion=door_exclusion)


class LayoutEngine:
    """
    Vastu-Guided Hierarchical Space Partitioning (V-HSP) Layout Engine.

    Usage:
        engine = LayoutEngine(plot_width=12.0, plot_height=15.0, bhk_config="2BHK",
                              vastu_enabled=True, orientation="N")
        plan = engine.generate()
    """

    def __init__(
        self,
        plot_width: float,
        plot_height: float,
        bhk_config: str = "auto",
        vastu_enabled: bool = True,
        orientation: str = "N",
        quality_level: str = "Standard",
        project_name: str = "Residential Floor Plan",
        family_description: str = "",
        custom_rooms: Optional[List[RoomTemplate]] = None,
        has_lift: bool = False,
        lift_capacity: str = "4 Person",
        stair_type: str = "dog_leg",
        num_floors: int = 1,
        building_type: str = "Independent House",
        additional_components: Optional[Dict] = None,
        family_profile=None,
        design_intents: Optional[List[Dict]] = None,
        base_layout: Optional[Dict] = None,
        flats_per_floor="auto",
        lift_position: str = "inside",
        road_width: float = 9.0,
        terrace_stair: str = "auto",
        **kwargs,
    ):
        """
        Args:
            plot_width: Plot width in meters (X axis, along the road).
            plot_height: Plot depth in meters (Y axis).
            bhk_config: '1BHK', '2BHK', '3BHK', '4BHK', or 'auto'.
            vastu_enabled: Whether to optimise for Vastu Shastra.
            orientation: Road / entrance facing direction 'N', 'E', 'S', 'W'
                         (the road side is drawn at the top of the plan).
            quality_level: 'Economy', 'Standard', 'Premium', 'Luxury' (scales room sizes).
            project_name: Project title for title block.
            family_description: Human-readable family description.
            custom_rooms: Optional custom room list (legacy fallback only).
            has_lift: Whether to include an IS 14665 lift shaft.
            lift_capacity: '4 Person', '6 Person', or '8 Person'.
            stair_type: 'dog_leg', 'straight', 'l_shaped', 'open_well', 'spiral', 'zigzag'.
            num_floors: Number of floors (stairs/lift, setbacks, genset, cost).
            building_type: 'Independent House', 'Apartment', or 'Duplex'.
            additional_components: Site amenities dict (parking, garden, pool, ...).
            family_profile: Optional FamilyProfile — drives bedroom roles and lifestyle rooms.
            design_intents: Optional user wishes for an edit re-plan (intent_engine dicts, e.g.
                            {"type": "zone", "room": "kitchen", "zones": ["NW"]}).
            base_layout: The `layout_state` of the plan being edited — the re-plan stays close to it.
            flats_per_floor: Apartments only — 1, 2, 4 or 'auto' flats per typical floor around a
                             common stair / lift core (floorplan_generator/apartment.py).
            lift_position: Houses with a lift — 'inside' (next to the stair, opens into the home),
                           'entrance' (a tower beside the main door) or 'side' (a tower against a
                           side wall with its own landing). Apartments always use the common core.
            road_width: Width of the road in front of the plot (m) — sizes the car gate (gates.py).
            terrace_stair: Single-storey houses — side of the house for the outside terrace stair:
                           'auto' (South / West, never the road side) or 'front' | 'back' | 'left' |
                           'right' as on the drawing (road at the top). A {"type": "terrace_stair",
                           "side": …} wish in `design_intents` sets it too (Smart Re-Plan).
        """
        self.plot_width = max(plot_width, 4.0)
        self.plot_height = max(plot_height, 4.0)
        self.vastu_enabled = vastu_enabled
        self.orientation = (orientation or "N").upper()[0]
        if self.orientation not in "NESW":
            self.orientation = "N"
        self.quality_level = quality_level
        self.project_name = project_name
        self.family_description = family_description
        self.has_lift = has_lift
        self.lift_capacity = lift_capacity
        valid_stairs = {"dog_leg", "straight", "l_shaped", "open_well", "spiral", "zigzag"}
        self.stair_type = stair_type if stair_type in valid_stairs else "dog_leg"
        self.num_floors = max(1, int(num_floors or 1))
        self.building_type = building_type
        if additional_components is None and "additional_components" in kwargs:
            additional_components = kwargs["additional_components"]
        self.additional_components = additional_components or {}
        self.family_profile = family_profile
        self.custom_rooms = custom_rooms
        # the terrace-stair wish is a site choice, not a room arrangement: it sets the side, the room
        # solver never sees it (its report line is added in generate_alternatives)
        wishes = list(design_intents or [])
        self.site_intents = [i for i in wishes if isinstance(i, dict) and i.get("type") == "terrace_stair"]
        self.design_intents = [i for i in wishes if not (isinstance(i, dict) and i.get("type") == "terrace_stair")]
        side = str(terrace_stair or "auto").lower()
        for it in self.site_intents:
            side = str(it.get("side", side)).lower()
        self.terrace_stair = side if side in STAIR_SIDES else "auto"
        self.base_layout = base_layout or {}
        # Everything needed to rebuild this exact plan (the intent editor re-runs the pipeline)
        self.generation_params = dict(
            plot_width=plot_width, plot_height=plot_height, bhk_config=bhk_config,
            vastu_enabled=vastu_enabled, orientation=orientation, quality_level=quality_level,
            project_name=project_name, family_description=family_description,
            custom_rooms=custom_rooms, has_lift=has_lift, lift_capacity=lift_capacity,
            stair_type=stair_type, num_floors=num_floors, building_type=building_type,
            additional_components=self.additional_components, family_profile=family_profile,
        )
        self.flats_per_floor = flats_per_floor
        self._fixed_site = None              # a flat inside an apartment floor: plan exactly this rectangle
        self._apartment_fallback = False
        self._program_override = None        # a duplex floor: plan this room program instead of the BHK's
        if "apartment" in (building_type or "").lower():
            self.generation_params["flats_per_floor"] = flats_per_floor
        self.lift_position = lift_position if lift_position in ("entrance", "side") else "inside"
        if "apartment" in (building_type or "").lower() or "duplex" in (building_type or "").lower():
            self.lift_position = "inside"            # flats share the core's lift; a duplex keeps it inside
        if self.lift_position != "inside":
            self.generation_params["lift_position"] = self.lift_position
        try:
            self.road_width = float(road_width or 9.0)
        except (TypeError, ValueError):
            self.road_width = 9.0
        if abs(self.road_width - 9.0) > 1e-6:
            self.generation_params["road_width"] = self.road_width
        if self.terrace_stair != "auto":
            self.generation_params["terrace_stair"] = self.terrace_stair

        self.wall = WallThickness()
        self.north_edge = north_edge_for_facing(self.orientation)
        self.sides = plan_sides_for_facing(self.orientation)

        if bhk_config.lower() == "auto":
            plot_area = self.plot_width * self.plot_height
            self.bhk_name = recommend_bhk_for_plot(plot_area)
        else:
            self.bhk_name = bhk_config.upper().replace(" ", "")
        self.config = get_bhk_config(self.bhk_name)
        self.room_templates = list(custom_rooms) if custom_rooms else list(self.config.rooms)

        # Vastu engine over the whole plot (kept for backward compatibility)
        self.vastu = VastuEngine(self.plot_width, self.plot_height, self.north_edge) if self.vastu_enabled else None

    # =========================================================================
    # Main pipeline
    # =========================================================================

    def _stack_mode(self) -> Optional[str]:
        """
        How the floors are stacked (floorplan_generator/apartment.py):
          'apartment' — flats around a common stair + lift core;
          'house'     — a house of 2+ floors: one home on every floor, each floor's main door on a stair
                        lobby at the front (nobody walks through another floor's home);
          'duplex'    — one home on two floors: living below, bedrooms above, the stair rising from the foyer;
          None        — one floor (or a flat / unit planned inside a given rectangle).
        """
        if self._fixed_site is not None or self._apartment_fallback:
            return None
        bt = (self.building_type or "").lower()
        if "apartment" in bt:
            return "apartment"
        if "duplex" in bt:
            return "duplex"
        if self.num_floors > 1:
            return "house"
        return None

    def _apartment_mode(self) -> bool:
        return self._stack_mode() is not None

    def generate(self) -> GeneratedFloorPlan:
        """Execute the full pipeline and return a complete floor plan."""
        if self._apartment_mode():           # floors stacked round a stair core (apartment.py)
            from floorplan_generator import apartment
            return apartment.generate(self)
        ctx = self._solve_stage()
        return self._assemble(ctx, ctx["result"])

    def generate_alternatives(self, k: int = 3) -> List[GeneratedFloorPlan]:
        """
        Intent editing: run the pipeline once with `design_intents`, then build the
        k best *distinct* layouts the intent solver found (balanced, best Vastu,
        least change). Each plan carries an `edit_report` explaining how it met the
        user's wishes and what else had to move.
        """
        import copy as _copy
        if self._apartment_mode():
            from floorplan_generator import apartment
            return apartment.generate(self, alternatives=k)
        ctx = self._solve_stage(alternatives=k)
        solver = ctx["solver"]
        options = ctx["alternatives"] or ([(ctx["result"], "Re-planned layout")] if ctx["result"] else [])
        if not options:
            plan = self._assemble(ctx, None)
            plan.edit_report = {"label": "Fallback layout", "intents": [], "satisfied": 0, "total": 0,
                                "changes": [], "solver_vastu": None, "base_vastu": None, "mode": None,
                                "warnings": ["No band layout could be found for these wishes on this footprint."]}
            return [plan]
        plans = []
        for res, label in options[:max(1, k)]:
            c = dict(ctx)
            for part in ("program", "planner", "site"):
                c[part] = _copy.deepcopy(ctx[part])       # amenity placement mutates the site
            plan = self._assemble(c, res)
            report = solver.report(res) if hasattr(solver, "report") else {}
            report["label"] = label
            self._site_intent_report(report, plan)
            plan.edit_report = report
            plans.append(plan)
        return plans

    def _site_intent_report(self, report: Dict, plan) -> None:
        """The terrace-stair wish in the option's report: met when the stair stands on that side."""
        if not self.site_intents:
            return
        from floorplan_generator.intent_engine import describe_intent
        info = (plan.engineering or {}).get("terrace_stair") or {}
        want = STAIR_SIDES.get(self.terrace_stair)
        ok = bool(info) and info.get("side") == want
        if ok:
            detail = f"on the {info['compass_name']} side, in the {info['zone_name']} · Vastu {info['vastu_word']}"
        elif info:
            detail = f"stays on the {info['compass_name']} side"
        else:
            detail = "this house has no outside terrace stair"
        entry = {"text": describe_intent(self.site_intents[-1], plan.layout_state or {}), "satisfied": ok,
                 "detail": detail}
        if not ok:
            entry["blocked_by"] = info.get("refused") or "that side has no room for it"
        report.setdefault("intents", []).append(entry)
        report["satisfied"] = sum(1 for d in report["intents"] if d.get("satisfied"))
        report["total"] = len(report["intents"])

    def _make_solver(self, program, footprint, seed: int):
        from floorplan_generator.layout_solver import BandSpineSolver
        if self.design_intents or self.base_layout:
            from floorplan_generator.intent_engine import IntentSolver
            return IntentSolver(program, footprint, self.orientation, self.vastu_enabled, seed=seed,
                                intents=self.design_intents, base=self.base_layout)
        return BandSpineSolver(program, footprint, self.orientation, self.vastu_enabled, seed=seed)

    def _solve_stage(self, alternatives: int = 0) -> Dict:
        """Stages 1-3: space program, site plan and the zoning solve (with relaxation)."""
        from floorplan_generator.space_program import build_program, trim_program
        from floorplan_generator.site_planner import SitePlanner

        # ── Stage 1: space program ────────────────────────────────────────
        if self._program_override is not None:          # one floor of a duplex
            import copy as _copy
            program = _copy.deepcopy(self._program_override)
        else:
            program = build_program(
                self.bhk_name, self.family_profile, self.building_type, self.num_floors,
                self.quality_level, self.has_lift, self.lift_capacity, self.stair_type,
                lift_position=self.lift_position)
        members = 4
        fp_prof = self.family_profile
        if fp_prof is not None:
            members = max(1, getattr(fp_prof, "adults", 2) + getattr(fp_prof, "children", 0)
                          + getattr(fp_prof, "elderly", 0))

        # ── Stage 2: site planning ────────────────────────────────────────
        comps = dict(self.additional_components)
        comps["_has_lift"] = bool(program.lift)
        planner = SitePlanner(self.plot_width, self.plot_height, self.orientation, self.vastu_enabled,
                              self.building_type, self.num_floors, comps, members)
        n_bed = int(program.bhk[0])
        min_w = {1: 5.6, 2: 6.6, 3: 7.6, 4: 8.4}.get(n_bed, 6.6)
        min_d = 6.8
        if self._fixed_site is not None:    # a flat of an apartment floor: its rectangle is given
            import copy as _copy
            program.external_stair = None
            site = _copy.deepcopy(self._fixed_site)
        else:
            ext_len = program.external_stair.footprint_l if program.external_stair else 0.0
            lift_out = None
            if program.lift_outside and program.lift:
                lift_out = {"position": program.lift_outside, "w": program.lift.shaft_w,
                            "d": program.lift.shaft_d, "door": 0.9}
            stair_want = STAIR_SIDES.get(self.terrace_stair)
            site = planner.plan_footprint(program.target_area * 1.12, program.min_area, min_w, min_d, ext_len,
                                          lift_out=lift_out, ext_stair_want=stair_want)
            if lift_out and site.lift_side is None:
                # no yard can take the tower: the lift goes back inside, next to the stair
                failed = [n for n in site.notes if n.startswith("The lift could not go outside")]
                program = build_program(
                    self.bhk_name, self.family_profile, self.building_type, self.num_floors,
                    self.quality_level, self.has_lift, self.lift_capacity, self.stair_type)
                planner = SitePlanner(self.plot_width, self.plot_height, self.orientation, self.vastu_enabled,
                                      self.building_type, self.num_floors, dict(comps), members)
                site = planner.plan_footprint(program.target_area * 1.12, program.min_area, min_w, min_d, ext_len,
                                              ext_stair_want=stair_want)
                site.notes.extend(failed)

        # User size wishes re-balance rooms inside the same envelope (footprint unchanged)
        intent_notes: List[str] = []
        if self.design_intents:
            from floorplan_generator.intent_engine import apply_size_intents
            intent_notes = apply_size_intents(program, self.design_intents, self.base_layout)

        # ── Stage 3: zoning & geometry (with progressive relaxation) ─────
        fx, fy, fw, fd = site.footprint
        trim_notes = trim_program(program, fw * fd) if fw * fd < program.min_area else []
        seed = int(self.plot_width * 100) * 31 + int(self.plot_height * 100) * 17 + "NESW".index(self.orientation)
        result = None
        solver = None
        relax_notes: List[str] = []
        # An edit re-plan replays the relaxations its base plan needed, so the program matches
        start = int(self.base_layout.get("relax_level", 0) or 0) if self.base_layout else 0
        for attempt in range(min(start, MAX_RELAX)):
            relax_notes.extend(self._relax_program(program, attempt))
        relax_level = start
        attempt = min(start, MAX_RELAX)
        while True:                   # the always-tried steps 0-3 (an edit re-plan starts where its base was)
            relax_level = attempt
            solver = self._make_solver(program, site.footprint, seed)
            solver.door_exclusion = getattr(site, "door_exclusion", None)
            result = solver.solve()
            if result is not None and result.penalties.get("access", 0) == 0 and                     result.penalties.get("circulation", 0) == 0:
                break
            if attempt >= 4:
                break
            step = self._relax_program(program, attempt)
            relax_notes.extend(step)
            attempt += 1
        retry = getattr(self, "_lift_retry", False)
        if getattr(program, "lift_outside", None) and result is None and self.lift_position != "inside" and not retry:
            # the rooms cannot be laid out around the tower's yard: plan the lift inside, next to the stair
            ctx = self._solve_lift_as("inside", alternatives)
            ctx["site"].notes.append("The lift could not go outside the house on this plot: with the tower's yard "
                                     "the rooms do not fit — it stays inside, next to the stair.")
            return ctx
        if program.lift and not getattr(program, "lift_outside", None) and result is None and not retry                 and self._fixed_site is None and "apartment" not in (self.building_type or "").lower():
            # no layout keeps the stair and the lift inside: a tower beside the main door, rather than no lift
            ctx = self._solve_lift_as("entrance", alternatives)
            if ctx["result"] is not None and getattr(ctx["program"], "lift_outside", None):
                ctx["site"].notes.append("The lift could not fit inside the house on this plot (with the stair the "
                                         "rooms do not fit) — it stands outside, beside the main door.")
                return ctx
        # Still no layout at all: the compact-home steps (an attached bath shared, a pooja shelf, NBC minimum
        # widths, one bathroom) — after moving the lift, because a room given up is worse than a lift outside;
        # each step that changes something is solved again, and the first layout found is kept
        if result is None:
            attempt = max(attempt, 4)
            while attempt < MAX_RELAX:
                step = self._relax_program(program, attempt)
                attempt += 1
                if not step:
                    continue
                relax_notes.extend(step)
                relax_level = attempt
                solver = self._make_solver(program, site.footprint, seed)
                solver.door_exclusion = getattr(site, "door_exclusion", None)
                result = solver.solve()
                if result is not None:
                    break
        alts = []
        if alternatives and result is not None and hasattr(solver, "alternatives"):
            alts = solver.alternatives(alternatives)
        return {"program": program, "planner": planner, "site": site, "result": result, "solver": solver,
                "alternatives": alts, "trim_notes": trim_notes, "relax_notes": relax_notes,
                "intent_notes": intent_notes, "relax_level": relax_level}

    def _solve_lift_as(self, position: str, alternatives: int = 0) -> Dict:
        """Solve again with the lift at `position` (one retry, never recursive)."""
        keep = self.lift_position
        self.lift_position, self._lift_retry = position, True
        try:
            return self._solve_stage(alternatives)
        finally:
            self.lift_position, self._lift_retry = keep, False

    def site_preview(self, door_x: Optional[Tuple[float, float]] = None) -> Dict:
        """
        Stages 1-2 + the amenities only (no room solve, a few milliseconds): the footprint, setbacks,
        the outside terrace stair and everything in the yard exactly as a re-plan with these settings
        builds them. `door_x` = where the main door is expected on the road face (the room solve picks
        it; the current design's door is a good guess). Single-storey houses (the ones with a terrace
        stair outside); returns {} for anything else.
        """
        from floorplan_generator.space_program import build_program
        from floorplan_generator.site_planner import SitePlanner
        if self._stack_mode() is not None or self._fixed_site is not None:
            return {}
        program = build_program(self.bhk_name, self.family_profile, self.building_type, self.num_floors,
                                self.quality_level, self.has_lift, self.lift_capacity, self.stair_type,
                                lift_position=self.lift_position)
        if program.external_stair is None:
            return {}
        members = 4
        if self.family_profile is not None:
            fp = self.family_profile
            members = max(1, getattr(fp, "adults", 2) + getattr(fp, "children", 0) + getattr(fp, "elderly", 0))
        comps = dict(self.additional_components)
        comps["_has_lift"] = False
        planner = SitePlanner(self.plot_width, self.plot_height, self.orientation, self.vastu_enabled,
                              self.building_type, self.num_floors, comps, members)
        n_bed = int(program.bhk[0])
        min_w = {1: 5.6, 2: 6.6, 3: 7.6, 4: 8.4}.get(n_bed, 6.6)
        site = planner.plan_footprint(program.target_area * 1.12, program.min_area, min_w, 6.8,
                                      program.external_stair.footprint_l,
                                      ext_stair_want=STAIR_SIDES.get(self.terrace_stair))
        fx, fy, fw, fd = site.footprint
        if door_x is None:
            door_x = (fx + fw / 2 - 0.5, fx + fw / 2 + 0.5)
        w = door_x[1] - door_x[0]
        x0 = min(max(door_x[0], fx + 0.45), fx + fw - 0.45 - w)
        ex = site.door_exclusion
        if ex and x0 < ex[1] and x0 + w > ex[0]:            # the door keeps off a front stair / car bay
            x0 = ex[1] + 0.05 if ex[0] <= 0.05 else ex[0] - 0.05 - w
        self._gate_brief(planner, site, [])
        planner.place_amenities(site, x0, x0 + w, program.external_stair)
        return {"site": site, "program": program, "door_x": (x0, x0 + w)}

    def _assemble(self, ctx: Dict, result) -> GeneratedFloorPlan:
        """Stages 4-9 for one solved layout: doors, amenities, walls, windows, audit, report."""
        program, planner, site = ctx["program"], ctx["planner"], ctx["site"]
        plan = GeneratedFloorPlan(
            plot_width=self.plot_width, plot_height=self.plot_height,
            orientation=self.orientation, vastu_enabled=self.vastu_enabled,
            bhk_config=self.bhk_name, family_description=self.family_description,
            quality_level=self.quality_level, project_name=self.project_name,
            has_lift=self.has_lift, lift_capacity=self.lift_capacity,
            building_type=self.building_type, num_floors=self.num_floors,
            additional_components=self.additional_components,
            north_edge=self.north_edge, stair_type=self.stair_type,
        )
        notes: List[str] = []
        fx, fy, fw, fd = site.footprint

        if result is None:
            # Last resort: legacy functional subdivision inside the footprint
            notes.append("Compact plot: rooms arranged by functional subdivision (no feasible band layout).")
            rooms = self._legacy_rooms(site.footprint)
            door_x = (fx + fw / 2 - 0.5, fx + fw / 2 + 0.5)
            placed = None
        else:
            rooms = [RoomRect(name=p.name, category=p.category, x=p.x, y=p.y, w=p.w, h=p.h,
                              color=p.color, role=p.role, key=p.key) for p in result.rooms]
            door_x = result.door
            placed = result.rooms
        bhk_now = program.bhk
        if bhk_now != self.bhk_name:
            plan.bhk_config = bhk_now

        # ── Stages 4-5: doors, arches, main entrance ─────────────────────
        doors, openings = self._place_doors(rooms, placed, door_x, site.footprint)

        # ── Stage 5b: a lift tower outside the house (beside the main door / against a side wall) ──
        tower = None
        if getattr(program, "lift_outside", None) and getattr(site, "lift_side", None) and result is not None:
            tower = self._outside_lift(program, site, rooms, doors, door_x)
            if tower is None:
                notes.append("The lift tower could not be fitted outside the walls of this layout.")
            else:
                self._tower_into(tower, rooms, doors, site, notes)

        # ── Stage 6: site amenities and gates ────────────────────────────
        self._gate_brief(planner, site, rooms)
        planner.place_amenities(site, door_x[0], door_x[1], program.external_stair)
        for it in site.items:
            x, y, w, h = it.rect
            rooms.append(RoomRect(name=it.name, category=it.category, x=round(x, 3), y=round(y, 3),
                                  w=round(w, 3), h=round(h, 3), color=it.color))

        # ── Stage 7: walls with openings ─────────────────────────────────
        walls = self._generate_walls(rooms)
        if tower is not None:                 # the tower's own 230 mm RCC walls
            walls.extend(self._tower_walls(tower))
        cuts = [self._door_cut(d) for d in doors] + list(openings)
        top = self.plot_height
        cuts += [(g0, top, g1, top) for g0, g1 in site.gates]
        walls = self._cut_openings(walls, cuts)

        # ── Stage 8: windows ─────────────────────────────────────────────
        windows = self._place_windows(rooms, doors, site.footprint)
        if tower is not None:
            windows = self._clear_windows(windows, rooms, doors, tower["outer"])

        # Staircases (internal + external terrace stair)
        staircases = self._place_staircases(rooms, program, site.footprint)

        # ── Stage 9: dimensions, areas, Vastu audit, engineering notes ───
        dimensions = self._generate_dimensions(rooms)
        self._set_clear_areas(rooms, site.footprint)
        lift_area = sum(r.w * r.h for r in rooms if r.category == RoomCategory.LIFT.value and not _is_site_room(r))
        duct_area = sum(r.w * r.h for r in rooms if "Duct" in r.name)
        carpet = max(0.0, (fw - self.wall.external) * (fd - self.wall.external) - lift_area - duct_area)
        built_up = (fw + self.wall.external) * (fd + self.wall.external)

        vastu_score, breakdown = self._vastu_audit(rooms, doors, site)
        plan.rooms = rooms
        plan.walls = walls
        plan.doors = doors
        plan.windows = windows
        plan.staircases = staircases
        plan.dimensions = dimensions
        plan.openings = list(openings)
        plan.vastu_score = round(vastu_score, 1)
        plan.vastu_breakdown = breakdown
        plan.total_carpet_area = round(carpet, 2)
        plan.total_built_up_area = round(built_up, 2)
        plan.building_footprint = (round(fx, 3), round(fy, 3), round(fw, 3), round(fd, 3))
        plan.setbacks = {self.sides[k]: v for k, v in site.setbacks.items()}

        engineering = self._engineering_report(rooms, walls, program, site, carpet, built_up)
        stair_info = self._terrace_stair_info(rooms, staircases, site)
        if stair_info:
            engineering["terrace_stair"] = stair_info
        if engineering.get("lift"):
            engineering["lift"]["position"] = "inside"
            if tower is not None:
                engineering["lift"].update(self._tower_info(tower))
        self._gate_report(engineering, site, planner)
        plan.engineering = engineering
        notes = (program.notes + ctx["intent_notes"] + ctx["trim_notes"] + ctx["relax_notes"] + site.notes
                 + notes + engineering.pop("notes", []))
        if result is not None:
            notes.insert(0, f"Layout: {result.mode}-band zoning optimised over the building's Vastu "
                            f"Purusha Mandala (raw objective {result.score:.1f}).")
            plan.layout_state = self._layout_state(result, program, site.footprint)
            plan.layout_state["relax_level"] = ctx.get("relax_level", 0)
            if stair_info:                     # the wish box and the side chooser read it
                plan.layout_state["terrace_stair"] = {k: stair_info[k] for k in
                                                      ("side", "word", "compass", "zone", "score", "want")}
            # quantities of the design the owner started from — edits are priced against it
            from floorplan_generator.quantities import layout_quantities
            ref = (self.base_layout or {}).get("ref_quantities")
            plan.layout_state["ref_quantities"] = dict(ref) if ref else layout_quantities(plan)
        plan.design_notes = notes
        plan.generation_params = dict(self.generation_params)
        return plan

    def _gate_brief(self, planner, site, rooms) -> None:
        """What the gate designer needs to know: the road, the design car, the stilt bays (gates.py)."""
        planner.road, planner.quality = self.road_width, self.quality_level
        if getattr(site, "car_mode", None) == "stilt":
            from floorplan_generator import gates as _gates
            info = _gates.stilt_plan(rooms, site.footprint, self.additional_components, self.quality_level,
                                     self.building_type, _is_site_room, grid_x=getattr(site, "grid_x", None),
                                     walkways=getattr(site, "walkways", None))
            planner.stilt_info = info
            if info.get("note"):
                site.notes.append(info["note"])

    @staticmethod
    def _gate_report(engineering: Dict, site, planner) -> None:
        engineering["gates"] = [dict(g) for g in (getattr(site, "gate_specs", None) or [])]
        engineering["road_m"] = getattr(site, "road", 9.0)
        if getattr(site, "drive", None):
            engineering["drive"] = site.drive
        info = getattr(planner, "stilt_info", None)
        if info and getattr(site, "car_mode", None) == "stilt":
            engineering["stilt"] = {k: v for k, v in info.items() if k != "note"}

    @staticmethod
    def _tower_into(tower: Dict, rooms, doors, site, notes) -> None:
        """Add a placed lift tower to the plan: its room, its door (+ side door), the yard it keeps."""
        x, y, w, h = tower["rect"]
        rooms.append(RoomRect(name=LIFT_TOWER, category=RoomCategory.LIFT.value, x=round(x, 3), y=round(y, 3),
                              w=round(w, 3), h=round(h, 3), color="#38BDF8", role="LIFT"))
        doors.append(tower["door"])
        if tower["entry"] is not None:
            doors.append(tower["entry"])
        site.reserved = [tower["outer"], tower["apron"]]
        notes.append(tower["note"])

    def _tower_walls(self, tower: Dict) -> List[WallSegment]:
        x0, y0, w, h = tower["rect"]
        x1, y1 = x0 + w, y0 + h
        t = self.wall.external
        return [WallSegment(round(a, 3), round(b, 3), round(c, 3), round(d, 3), t, True, "external")
                for a, b, c, d in ((x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0))]

    @staticmethod
    def _tower_info(tower: Dict) -> Dict:
        return {"position": "entrance" if tower["how"] == "entrance" else "side",
                "layout": tower["how"], "side": tower["side"],
                "tower": [round(v, 3) for v in tower["outer"]],
                "landing": [round(v, 3) for v in tower["landing"]],
                "entry": "side door" if tower["entry"] is not None else "main door"}

    # Rooms a lift landing may open into through a side door (never a bedroom, bath or pooja)
    _ENTRY_PREF = {"DINING": 3.0, "CORRIDOR": 3.0, "FOYER": 3.0, "LIVING": 2.6, "UTILITY": 1.8, "KITCHEN": 1.5}

    def _outside_lift(self, program, site, rooms, doors, door_x, entry_rooms=None) -> Optional[Dict]:
        """
        Place the lift tower outside the house walls (230 mm RCC shaft) after the rooms are solved:
          'top'  — beside the main door, its door facing the entrance landing (a porch at ground,
                   a covered landing on every floor joining that floor's front door);
          'left' / 'right' — against that side wall, its door facing the road, a 1.5 m landing in
                   front with a side door into a living space (dining / passage / foyer / living,
                   kitchen or utility as a service entry) — or, when that wall has none, at the
                   front end of the side yard with a verandah along the facade to the main door.
        Vastu: lift zones on the plot mandala (S / W / NW good, NE never).
        """
        from floorplan_generator.site_planner import MARGIN_PLOT
        lift = program.lift
        sw, sd, dw = lift.shaft_w, lift.shaft_d, 0.9
        fx, fy, fw, fd = site.footprint
        W, H = self.plot_width, self.plot_height
        top = fy + fd
        e = 0.115
        plot_eng = VastuEngine.for_facing(W, H, self.orientation)

        def vastu_at(cx, cy):
            if not self.vastu_enabled:
                return 0
            return plot_eng.get_room_score("LIFT", plot_eng.get_zone_for_position(cx, cy).value)

        def in_plot(o):
            return (o[0] >= MARGIN_PLOT - 1e-6 and o[1] >= MARGIN_PLOT - 1e-6
                    and o[2] <= W - MARGIN_PLOT + 1e-6 and o[3] <= H - MARGIN_PLOT + 1e-6)

        def pack(rect, door, entry, landing, how, side, apron=None):
            x0, y0, x1, y1 = rect
            zone = plot_eng.get_zone_for_position((x0 + x1) / 2, (y0 + y1) / 2).value
            where = {"entrance": "beside the main door",
                     "side door": f"against the {ZONE_NAMES.get(self.sides.get(side, ''), 'side')} side wall",
                     "verandah": f"at the front of the {ZONE_NAMES.get(self.sides.get(side, ''), 'side')} side"}[how]
            host = entry.room_name if entry is not None else "house"
            reach = {"entrance": "its landing is the entrance porch, and on every floor a covered landing joins "
                                 "that floor's front door",
                     "side door": f"a 1.5 m landing in front opens into the {host} through a side door on every floor",
                     "verandah": "a 1.5 m verandah along the front joins the main door on every floor"}[how]
            note = (f"Lift tower {where} ({ZONE_NAMES.get(zone, zone)} of the site): {sw:.1f} × {sd:.1f} m shaft in "
                    f"230 mm RCC walls, door {dw:.1f} m — {reach}.")
            return {"rect": (x0, y0, x1 - x0, y1 - y0), "outer": (x0 - e, y0 - e, x1 + e, y1 + e), "door": door,
                    "entry": entry, "landing": landing, "apron": apron or landing, "how": how, "side": side,
                    "note": note}

        def entrance():
            d0, d1 = door_x
            depth = max(1.2, 0.23 + (sw + dw) / 2 + 0.15)
            y0, y1 = top + 0.23, top + 0.23 + sw
            best = None
            for side in ("right", "left"):
                if side == "right":
                    x0 = d1 + 0.6 + e
                    x1, face, ws = x0 + sd, x0, "W"
                    landing = (d0 - 0.3, top, x0 - e, top + depth)
                else:
                    x1 = d0 - 0.6 - e
                    x0, face, ws = x1 - sd, x1, "E"
                    landing = (x1 + e, top, d1 + 0.3, top + depth)
                if not in_plot((x0 - e, y0 - e, x1 + e, y1 + e)):
                    continue
                key = vastu_at((x0 + x1) / 2, (y0 + y1) / 2) + (0.1 if side == "right" else 0.0)
                if best is None or key > best[0]:
                    best = (key, (x0, y0, x1, y1), face, ws, landing)
            if best is None:
                return None
            _, rect, face, ws, landing = best
            door = DoorPlacement(x=round(face, 3), y=round(rect[1] + (sw - dw) / 2, 3), width=dw, wall_side=ws,
                                 swing_direction="RIGHT", room_name=LIFT_TOWER)
            apron = (min(landing[0], d0 - 0.45), top, max(landing[2], d1 + 0.45), top + depth + 0.9)
            return pack(rect, door, None, landing, "entrance", "top", apron)

        def side_door(side):
            left = side == "left"
            x1 = fx - 0.23 if left else fx + fw + 0.23 + sw
            x0 = x1 - sw
            wall_x = fx if left else fx + fw
            ws_entry = "W" if left else "E"
            lx = (x0 - e, fx - e) if left else (fx + fw + e, x1 + e)
            edge = [r for r in rooms if not _is_site_room(r) and abs((r.x if left else r.right) - wall_x) < 0.03
                    and "Dress" not in r.name and "Wardrobe" not in r.name
                    and (entry_rooms is None or r.name in entry_rooms)]
            taken = [(d.y, d.y + d.width) for d in doors if d.wall_side in ("E", "W") and abs(d.x - wall_x) < 0.03]
            best = None
            ty = fy
            while ty + sd + e + 1.5 <= top + 1e-6:
                ly0 = ty + sd + e
                e0, e1 = ly0 + 0.3, ly0 + 1.2
                host = next((r for r in edge if r.y + 0.12 <= e0 and e1 <= r.top - 0.12), None)
                pref = self._ENTRY_PREF.get(host.category) if host is not None else None
                if pref and not any(a < e1 + 0.3 and b > e0 - 0.3 for a, b in taken):
                    key = 10 * pref + 0.5 * vastu_at((x0 + x1) / 2, ty + sd / 2) + 2.0 * (ty - fy) / max(fd, 1.0)
                    if best is None or key > best[0]:
                        best = (key, ty, host, e0)
                ty = round(ty + 0.05, 3)
            if best is None:
                return None
            _, ty, host, e0 = best
            y0, y1 = ty, ty + sd
            door = DoorPlacement(x=round((x0 + x1) / 2 - dw / 2, 3), y=round(y1, 3), width=dw, wall_side="N",
                                 swing_direction="RIGHT", room_name=LIFT_TOWER)
            entry = DoorPlacement(x=round(wall_x, 3), y=round(e0, 3), width=0.9, wall_side=ws_entry,
                                  swing_direction="RIGHT", room_name=host.name)
            landing = (lx[0], y1 + e, lx[1], y1 + e + 1.5)
            apron = (lx[0], y1 + e, lx[1], min(H - MARGIN_PLOT, y1 + e + 2.4))
            return pack((x0, y0, x1, y1), door, entry, landing, "side door", side, apron)

        def verandah(side):
            left = side == "left"
            x1 = fx - 0.23 if left else fx + fw + 0.23 + sw
            x0 = x1 - sw
            y0, y1 = top - sd, top
            d0, d1 = door_x
            landing = (x0 - e, top, d1 + 0.3, top + 1.5) if left else (d0 - 0.3, top, x1 + e, top + 1.5)
            if landing[2] - landing[0] > 9.0:
                return None
            door = DoorPlacement(x=round((x0 + x1) / 2 - dw / 2, 3), y=round(y1, 3), width=dw, wall_side="N",
                                 swing_direction="RIGHT", room_name=LIFT_TOWER)
            apron = (landing[0], top, landing[2], top + 2.4)
            return pack((x0, y0, x1, y1), door, None, landing, "verandah", side, apron)

        if site.lift_side == "top":
            return entrance()
        return side_door(site.lift_side) or verandah(site.lift_side) or entrance()

    @staticmethod
    def _clear_windows(windows, rooms, doors, outer) -> List:
        """Windows the lift tower would cover slide along their wall (or go) — never glass into a shaft wall."""
        bx0, by0, bx1, by1 = outer
        by_name = {r.name: r for r in rooms}
        out = []
        for wd in windows:
            horiz = wd.wall_side in ("N", "S")
            o = {"N": (0, 1), "S": (0, -1), "E": (1, 0), "W": (-1, 0)}[wd.wall_side]
            line = wd.y if horiz else wd.x
            c = wd.x if horiz else wd.y
            half = wd.width / 2
            probe = line + (o[1] if horiz else o[0]) * 0.35
            lo_b, hi_b = (bx0, bx1) if horiz else (by0, by1)
            near = (by0 - 0.01 <= probe <= by1 + 0.01) if horiz else (bx0 - 0.01 <= probe <= bx1 + 0.01)
            if not near or c + half <= lo_b or c - half >= hi_b:
                out.append(wd)
                continue
            r = by_name.get(wd.room_name)
            if r is None:
                continue
            lo_r, hi_r = (r.x, r.right) if horiz else (r.y, r.top)
            busy = [(lo_b - 0.1, hi_b + 0.1)]
            for d in doors:
                if (d.wall_side in ("N", "S")) == horiz and abs((d.y if horiz else d.x) - line) < 0.05:
                    a = d.x if horiz else d.y
                    busy.append((a - 0.3, a + d.width + 0.3))
            best = None
            for width in (wd.width, max(0.6, wd.width * 0.75)):
                h2 = width / 2
                for cand in (lo_b - 0.1 - h2, hi_b + 0.1 + h2):
                    if cand - h2 < lo_r + 0.25 or cand + h2 > hi_r - 0.25:
                        continue
                    if any(cand + h2 > a and cand - h2 < b for a, b in busy):
                        continue
                    if best is None or abs(cand - c) < abs(best[0] - c):
                        best = (cand, width)
                if best:
                    break
            if best is None:
                continue
            cand, width = best
            if horiz:
                out.append(WindowPlacement(x=round(cand, 3), y=wd.y, width=round(width, 2), height=wd.height,
                                           wall_side=wd.wall_side, room_name=wd.room_name))
            else:
                out.append(WindowPlacement(x=wd.x, y=round(cand, 3), width=round(width, 2), height=round(width, 2),
                                           wall_side=wd.wall_side, room_name=wd.room_name))
        return out

    def _layout_state(self, result, program, footprint) -> Dict:
        """Solver arrangement + per-room geometry, the base an intent re-plan stays close to."""
        groups = {g.key: g for g in program.groups}
        rooms: Dict[str, Dict] = {}
        for p in result.rooms:
            # small rooms stacked in one column keep their own group (the solver moves them separately)
            gk = p.key if p.key in groups else p.group
            g = groups.get(gk)
            if p.key.startswith("filler_"):           # foyer / passage end of a small-room column
                kind, bands = "filler", [p.band]
            elif g is not None:
                kind, bands = g.kind, list(g.bands)
            elif p.group == "living":
                kind, bands = "living", ["front"]
            else:
                kind, bands = ("spine" if p.band == "spine" else p.group or "other"), [p.band]
            rooms[p.key] = {"name": p.name, "category": p.category, "role": p.role, "group": gk,
                            "parent": p.parent, "band": p.band, "kind": kind, "bands": bands,
                            "cx": round(p.cx, 3), "cy": round(p.cy, 3), "w": round(p.w, 3), "h": round(p.h, 3),
                            "area": round(p.area, 2)}
        st = result.state
        return {"mode": st.mode, "bands": {k: list(v) for k, v in st.bands.items()},
                "cells": {k: list(v) for k, v in st.cells.items()}, "flip": dict(st.flip),
                "rooms": rooms, "footprint": [round(v, 3) for v in footprint],
                "facing": self.orientation, "vastu": round(result.vastu_score, 1)}

    # =========================================================================
    # Program relaxation & legacy fallback
    # =========================================================================

    def _relax_program(self, program, attempt: int) -> List[str]:
        """Progressively simplify the program when no access-complete layout exists."""
        if attempt == 0:
            for g in program.groups:
                if g.kind == "suite":
                    g.rooms = [r for r in g.rooms if "Wardrobe" not in r.name and "Dress" not in r.name]
            return ["Wardrobe/dress areas folded into bedrooms to fit the footprint."]
        if attempt == 1:
            for g in program.groups:
                if g.kind == "service" and len(g.rooms) > 1:
                    g.rooms = g.rooms[:1]
                    g.kind = "single"
            return ["Utility/wash area merged into the kitchen (compact footprint)."]
        if attempt == 2:
            for key in ("servant", "store", "study"):
                for g in list(program.groups):
                    if g.key == key:
                        program.groups.remove(g)
                        return [f"{g.primary.name} omitted: footprint cannot give it a door and daylight."]
            return []
        if attempt == 3:
            for r in [g.primary for g in program.groups]:
                if r.category == RoomCategory.BEDROOM.value:
                    r.min_width = min(r.min_width, 2.4)
            program.living.min_width = 2.7
            return ["Bedroom minimum widths relaxed to the NBC 2.4 m limit."]
        # Compact-home steps — only when no band layout exists at all (else the old legacy subdivision would
        # drop rooms, put a bedroom in the Brahmasthan and lose Smart Re-Plan). What an architect gives up
        # first on a small plot, keeping every bedroom the owner asked for:
        from floorplan_generator.space_program import RoomGroup, _room
        if 4 <= attempt <= 6:                # one attached bath at a time → the common bathroom (guest's first)
            order = {"GUEST_BEDROOM": 0, "BEDROOM": 1, "CHILD_BEDROOM": 2}
            suites = [g for g in program.groups if g.kind == "suite" and len(g.rooms) > 1
                      and g.primary.role not in ("MASTER_BEDROOM", "ELDER_BEDROOM")]
            if not suites:
                return []
            g = min(suites, key=lambda s: (order.get(s.primary.role, 1), -program.groups.index(s)))
            g.rooms = g.rooms[:1]
            g.kind = "single"
            self._ensure_common_bath(program, RoomGroup, _room)
            return [f"The {g.primary.name.lower()} shares the common bathroom — this footprint has no room for "
                    "another attached bath."]
        if attempt == 7:                     # a pooja shelf in the hall instead of a pooja room
            for g in list(program.groups):
                if g.key == "pooja":
                    program.groups.remove(g)
                    return ["Pooja: a pooja shelf in the North-East corner of the hall instead of a separate room "
                            "(compact footprint)."]
            return []
        if attempt == 8:                     # kitchen and hall at the NBC minimum widths
            for g in program.groups:
                for r in g.rooms:
                    if r.category == RoomCategory.KITCHEN.value:
                        r.min_width = min(r.min_width, 1.8)
            program.living.min_width = min(program.living.min_width, 2.4)
            return ["Kitchen and hall at the NBC minimum widths (1.8 m and 2.4 m)."]
        if attempt == 9:                     # one bathroom for the home, off the passage
            if not any(g.key == "cbath" for g in program.groups):
                return []
            for g in program.groups:
                if g.kind == "suite" and len(g.rooms) > 1:
                    g.rooms = g.rooms[:1]
                    g.kind = "single"
                    return [f"One bathroom for the home: the {g.primary.name.lower()} uses the common bathroom "
                            "off the passage, as in most compact homes."]
            return []
        return []

    @staticmethod
    def _ensure_common_bath(program, RoomGroup, _room) -> None:
        if not any(g.key == "cbath" for g in program.groups):
            program.groups.append(RoomGroup(
                key="cbath", kind="single",
                rooms=[_room("cbath", "Common Bathroom", "bathroom", 3.0, wet=True, min_area=2.2, min_width=1.2)],
                bands=("rear", "private", "front"), growth=0.1, min_depth=1.5))

    def _legacy_rooms(self, footprint) -> List[RoomRect]:
        fx, fy, fw, fd = footprint
        entries = self._build_room_entries()
        rooms = self._place_rooms_functional(entries, fx, fy, fw, fd)
        return self._clamp_nbc_dimensions(rooms, footprint)

    def _build_room_entries(self) -> List[dict]:
        """Build list of room entries with unique names (legacy fallback)."""
        entries = []
        name_counts: Dict[str, int] = {}
        for template in self.room_templates:
            if template.is_optional:
                continue
            base_name = template.spec.name
            name_counts[base_name] = name_counts.get(base_name, 0) + 1
            count = name_counts[base_name]
            name = f"{base_name} {count}" if count > 1 else base_name
            entries.append({
                "name": name,
                "category": template.spec.category.value,
                "target_area": template.target_width * template.target_height,
                "min_width": template.spec.min_width,
                "target_width": template.target_width,
                "target_height": template.target_height,
                "color": template.spec.color,
                "preferred_zone": template.preferred_vastu_zone,
                "attached_to": template.attached_to,
            })
        return entries

    def _place_rooms_functional(self, entries: List[dict],
                                ix: float, iy: float,
                                iw: float, ih: float) -> List[RoomRect]:
        """Place rooms using functional zoning (legacy fallback)."""
        public = [e for e in entries if e["category"] in
                  (RoomCategory.LIVING.value, RoomCategory.DINING.value, RoomCategory.FOYER.value)]
        private = [e for e in entries if e["category"] in
                   (RoomCategory.BEDROOM.value, RoomCategory.STUDY.value, RoomCategory.POOJA.value)]
        service = [e for e in entries if e["category"] in
                   (RoomCategory.KITCHEN.value, RoomCategory.BATHROOM.value,
                    RoomCategory.UTILITY.value, RoomCategory.SERVANT.value, RoomCategory.STORE.value)]
        classified = {e["name"] for grp in (public, private, service) for e in grp}
        public.extend(e for e in entries if e["name"] not in classified)

        total_area = sum(e["target_area"] for e in entries) or 1.0
        public_ratio = sum(e["target_area"] for e in public) / total_area
        service_ratio = sum(e["target_area"] for e in service) / total_area

        rooms = []
        if iw >= ih:
            sw = _snap_to_grid(iw * service_ratio)
            pw = _snap_to_grid(iw * public_ratio)
            prw = iw - sw - pw
            if service:
                rooms.extend(self._subdivide_zone(service, ix, iy, sw, ih))
            if public:
                rooms.extend(self._subdivide_zone(public, ix + sw, iy, pw, ih))
            if private:
                rooms.extend(self._subdivide_zone(private, ix + sw + pw, iy, prw, ih))
        else:
            sh = _snap_to_grid(ih * service_ratio)
            ph = _snap_to_grid(ih * public_ratio)
            prh = ih - sh - ph
            if service:
                rooms.extend(self._subdivide_zone(service, ix, iy, iw, sh))
            if private:
                rooms.extend(self._subdivide_zone(private, ix, iy + sh, iw, prh))
            if public:
                rooms.extend(self._subdivide_zone(public, ix, iy + sh + prh, iw, ph))
        return rooms

    def _subdivide_zone(self, entries: List[dict], zx: float, zy: float,
                        zw: float, zh: float) -> List[RoomRect]:
        """Recursively subdivide a rectangular zone (area-ratio binary splitting)."""
        if not entries or zw <= 0 or zh <= 0:
            return []
        if len(entries) == 1:
            e = entries[0]
            return [RoomRect(name=e["name"], category=e["category"], x=round(zx, 3), y=round(zy, 3),
                             w=round(zw, 3), h=round(zh, 3), color=e["color"])]
        sorted_entries = sorted(entries, key=lambda e: e["target_area"], reverse=True)
        mid = len(sorted_entries) // 2
        left_group, right_group = sorted_entries[:mid], sorted_entries[mid:]
        la = sum(e["target_area"] for e in left_group)
        ra = sum(e["target_area"] for e in right_group)
        ratio = la / (la + ra) if la + ra > 0 else 0.5
        rooms = []
        if zw >= zh:
            lw = _clamp(_snap_to_grid(zw * ratio), min(1.0, zw / 2), max(zw - 1.0, zw / 2))
            rooms.extend(self._subdivide_zone(left_group, zx, zy, lw, zh))
            rooms.extend(self._subdivide_zone(right_group, zx + lw, zy, zw - lw, zh))
        else:
            lh = _clamp(_snap_to_grid(zh * ratio), min(1.0, zh / 2), max(zh - 1.0, zh / 2))
            rooms.extend(self._subdivide_zone(left_group, zx, zy, zw, lh))
            rooms.extend(self._subdivide_zone(right_group, zx, zy + lh, zw, zh - lh))
        return rooms

    def _split_two(self, entries: List[dict], zx: float, zy: float,
                   zw: float, zh: float) -> List[RoomRect]:
        """Split a zone between exactly two rooms (kept for API compatibility)."""
        return self._subdivide_zone(entries[:2], zx, zy, zw, zh)

    def _clamp_nbc_dimensions(self, rooms: List[RoomRect], footprint=None) -> List[RoomRect]:
        """Keep rooms inside the footprint and recompute areas."""
        if footprint is None:
            ext = self.wall.external
            footprint = (ext, ext, self.plot_width - 2 * ext, self.plot_height - 2 * ext)
        fx, fy, fw, fd = footprint
        for room in rooms:
            room.x = _clamp(room.x, fx, fx + fw - 0.5)
            room.y = _clamp(room.y, fy, fy + fd - 0.5)
            room.w = max(0.5, min(room.w, fx + fw - room.x))
            room.h = max(0.5, min(room.h, fy + fd - room.y))
            room.area = round(room.w * room.h, 2)
        return rooms

    # =========================================================================
    # Doors, arches and the main entrance
    # =========================================================================

    @staticmethod
    def _shared(a, b, tol: float = 0.02):
        if abs(a.top - b.y) < tol:
            lo, hi = max(a.x, b.x), min(a.right, b.right)
            if hi - lo > tol:
                return "N", lo, hi, a.top
        if abs(a.y - b.top) < tol:
            lo, hi = max(a.x, b.x), min(a.right, b.right)
            if hi - lo > tol:
                return "S", lo, hi, a.y
        if abs(a.right - b.x) < tol:
            lo, hi = max(a.y, b.y), min(a.top, b.top)
            if hi - lo > tol:
                return "E", lo, hi, a.right
        if abs(a.x - b.right) < tol:
            lo, hi = max(a.y, b.y), min(a.top, b.top)
            if hi - lo > tol:
                return "W", lo, hi, a.x
        return None

    def _make_door(self, room, side, lo, hi, coord, width, toward=None) -> DoorPlacement:
        """Door near one end of the shared segment (keeps wall space for furniture)."""
        width = min(width, max(0.6, hi - lo - 0.1))
        margin = 0.15 if hi - lo >= width + 0.45 else max(0.0, (hi - lo - width) / 2)
        if toward is not None and toward > (lo + hi) / 2:
            pos = hi - margin - width
        else:
            pos = lo + margin
        pos = round(pos, 3)
        if side in ("N", "S"):
            return DoorPlacement(x=pos, y=round(coord, 3), width=round(width, 3), wall_side=side,
                                 swing_direction="RIGHT", room_name=room.name)
        return DoorPlacement(x=round(coord, 3), y=pos, width=round(width, 3), wall_side=side,
                             swing_direction="RIGHT", room_name=room.name)

    def _place_doors(self, rooms: List[RoomRect], placed, door_x, footprint):
        """
        Doors: attached rooms open into their parent (bath → dress/bedroom,
        utility → kitchen); every other room opens to the best circulation room
        (bedrooms → dining/passage before living for privacy, kitchen → dining,
        pooja → living). Living↔dining and stairs↔circulation get open arches.
        """
        from floorplan_generator.layout_solver import door_width_for
        doors: List[DoorPlacement] = []
        openings: List[Tuple[float, float, float, float]] = []
        fx, fy, fw, fd = footprint

        if placed is None:
            # Legacy fallback: a door on any shared wall + main door on the top facade
            for i, r in enumerate(rooms):
                for o in rooms[i + 1:]:
                    se = self._shared(r, o)
                    if se and se[2] - se[1] >= 0.9:
                        doors.append(self._make_door(r, se[0], se[1], se[2], se[3], 0.8))
                        break
            top_rooms = [r for r in rooms if abs(r.top - (fy + fd)) < 0.05]
            lr = next((r for r in top_rooms if r.category == RoomCategory.LIVING.value), top_rooms[0] if top_rooms else None)
            if lr:
                doors.append(DoorPlacement(x=round(lr.cx - 0.5, 3), y=round(lr.top, 3), width=1.0, wall_side="N",
                                           swing_direction="RIGHT", room_name=lr.name, is_main_entrance=True))
            return doors, openings

        by_key = {p.key: (p, r) for p, r in zip(placed, rooms)}

        # Main entrance on the living room's front facade
        if "living" in by_key:
            lp, lr = by_key["living"]
            doors.append(DoorPlacement(x=round(door_x[0], 3), y=round(lr.top, 3),
                                       width=round(door_x[1] - door_x[0], 3), wall_side="N",
                                       swing_direction="RIGHT", room_name=lr.name, is_main_entrance=True))

        circ_pref = {
            "BEDROOM": ["dining", "passage", "living"], "STUDY": ["passage", "dining", "living"],
            "KITCHEN": ["dining", "living", "passage"], "POOJA": ["living", "dining", "passage"],
            "BATHROOM": ["dining", "passage", "living"], "STORE": ["dining", "passage", "living"],
            "SERVANT": ["dining", "passage", "living"], "LIFT": ["dining", "passage", "living"],
        }
        circulation = [p for p in placed if p.circulation]

        for p, r in zip(placed, rooms):
            if p.circulation:
                continue
            dw = door_width_for(p)
            if p.category == RoomCategory.STAIRS.value:
                # open flight onto circulation: cut an arch the width of one flight
                for c in sorted(circulation, key=lambda c: 0 if c.key == "dining" else 1):
                    se = self._shared(r, by_key[c.key][1])
                    if se and se[2] - se[1] >= 0.9:
                        w = min(se[2] - se[1] - 0.1, 1.2)
                        lo = (se[1] + se[2]) / 2 - w / 2
                        openings.append(self._segment(se[0], lo, lo + w, se[3]))
                        break
                continue
            target = None
            if p.parent and p.parent in by_key:
                parent_room = by_key[p.parent][1]
                # baths prefer to open into the dress area when there is one
                dress = next((q for q in placed if q.parent == p.parent and q.key != p.key
                              and ("Dress" in q.name or "Wardrobe" in q.name)), None)
                if p.category == RoomCategory.BATHROOM.value and dress is not None:
                    se = self._shared(r, by_key[dress.key][1])
                    if se and se[2] - se[1] >= dw + 0.05:
                        target = (by_key[dress.key][1], se)
                if target is None:
                    se = self._shared(r, parent_room)
                    if se and se[2] - se[1] >= 0.65:
                        target = (parent_room, se)
            if target is None:
                order = circ_pref.get(p.category, ["dining", "passage", "living"])
                cands = []
                for c in circulation:
                    se = self._shared(r, by_key[c.key][1])
                    if se and se[2] - se[1] >= 0.65:
                        rank = order.index(c.key) if c.key in order else 9
                        cands.append((rank, -(se[2] - se[1]), c, se))
                if cands:
                    cands.sort(key=lambda t: (t[0], t[1]))
                    target = (by_key[cands[0][2].key][1], cands[0][3])
            if target is None:
                # anything shared at all (keeps every room reachable on tiny plots)
                for o in rooms:
                    if o is r or _is_site_room(o):
                        continue
                    se = self._shared(r, o)
                    if se and se[2] - se[1] >= 0.65:
                        target = (o, se)
                        break
            if target is None:
                continue
            other, (side, lo, hi, coord) = target
            toward = other.cx if side in ("N", "S") else other.cy
            if p.category == RoomCategory.LIFT.value:
                c = (lo + hi) / 2
                doors.append(self._make_door(r, side, c - dw / 2, c + dw / 2, coord, dw))
            else:
                doors.append(self._make_door(r, side, lo, hi, coord, dw, toward))

        # Living ↔ dining / passage: wide open arch (open-plan Indian hall)
        if "living" in by_key:
            lr = by_key["living"][1]
            for c in circulation:
                if c.key == "living":
                    continue
                se = self._shared(lr, by_key[c.key][1])
                if se and se[2] - se[1] >= 1.0:
                    w = min(2.4, (se[2] - se[1]) * 0.6)
                    mid = (se[1] + se[2]) / 2
                    openings.append(self._segment(se[0], mid - w / 2, mid + w / 2, se[3]))
        # Passages / foyers ↔ dining or living: open arches
        hubs = [k for k in ("dining", "living") if k in by_key]
        for c in circulation:
            if c.key in ("living", "dining"):
                continue
            for hk in hubs:
                se = self._shared(by_key[c.key][1], by_key[hk][1])
                if se and se[2] - se[1] >= 0.9:
                    w = min(1.2, se[2] - se[1] - 0.1)
                    mid = (se[1] + se[2]) / 2
                    openings.append(self._segment(se[0], mid - w / 2, mid + w / 2, se[3]))
                    break

        # Utility service door to outside when it touches the envelope (drying yard access)
        for p, r in zip(placed, rooms):
            if p.category == RoomCategory.UTILITY.value:
                ext = self._exterior_edges(r, footprint)
                if ext:
                    side, lo, hi, coord = max(ext, key=lambda e: e[2] - e[1])
                    if hi - lo >= 1.0 and side != "N":
                        doors.append(self._make_door(r, side, lo, hi, coord, 0.8))
        return doors, openings

    @staticmethod
    def _segment(side, lo, hi, coord):
        if side in ("N", "S"):
            return (round(lo, 3), round(coord, 3), round(hi, 3), round(coord, 3))
        return (round(coord, 3), round(lo, 3), round(coord, 3), round(hi, 3))

    @staticmethod
    def _door_cut(d: DoorPlacement):
        if d.wall_side in ("N", "S"):
            return (d.x, d.y, d.x + d.width, d.y)
        return (d.x, d.y, d.x, d.y + d.width)

    # =========================================================================
    # Walls
    # =========================================================================

    def _generate_walls(self, rooms: List[RoomRect]) -> List[WallSegment]:
        """
        Walls on room centre-lines:
          - 230 mm external envelope around the house footprint
          - 115 mm partitions, merged into continuous runs (no duplicates)
          - 150 mm compound wall on the plot boundary
        Works on any room list (also used by the interactive editor).
        """
        walls: List[WallSegment] = []
        ext_t, int_t = self.wall.external, self.wall.internal
        W, H = self.plot_width, self.plot_height

        building = [r for r in rooms if not _is_site_room(r) and r.w > 0 and r.h > 0]
        if building:
            bx0 = min(r.x for r in building)
            by0 = min(r.y for r in building)
            bx1 = max(r.right for r in building)
            by1 = max(r.top for r in building)
        else:
            bx0, by0, bx1, by1 = 0.0, 0.0, W, H

        # Compound wall (skip when the house sits on the boundary)
        if bx0 > 0.3 or by0 > 0.3 or bx1 < W - 0.3 or by1 < H - 0.3:
            for seg in ((0, 0, W, 0), (W, 0, W, H), (W, H, 0, H), (0, H, 0, 0)):
                walls.append(WallSegment(*seg, 0.15, True, "compound"))

        # External envelope
        walls.extend([
            WallSegment(round(bx0, 3), round(by0, 3), round(bx1, 3), round(by0, 3), ext_t, True, "external"),
            WallSegment(round(bx1, 3), round(by0, 3), round(bx1, 3), round(by1, 3), ext_t, True, "external"),
            WallSegment(round(bx1, 3), round(by1, 3), round(bx0, 3), round(by1, 3), ext_t, True, "external"),
            WallSegment(round(bx0, 3), round(by1, 3), round(bx0, 3), round(by0, 3), ext_t, True, "external"),
        ])

        # Interior partitions: union of room edges per line, minus the envelope
        horiz: Dict[float, List[Tuple[float, float]]] = {}
        vert: Dict[float, List[Tuple[float, float]]] = {}
        for r in building:
            for y in (r.y, r.top):
                if abs(y - by0) > 0.01 and abs(y - by1) > 0.01:
                    horiz.setdefault(round(y, 3), []).append((r.x, r.right))
            for x in (r.x, r.right):
                if abs(x - bx0) > 0.01 and abs(x - bx1) > 0.01:
                    vert.setdefault(round(x, 3), []).append((r.y, r.top))

        def merge(iv):
            iv = sorted(iv)
            out = [list(iv[0])]
            for a, b in iv[1:]:
                if a <= out[-1][1] + 0.01:
                    out[-1][1] = max(out[-1][1], b)
                else:
                    out.append([a, b])
            return out

        for y, iv in horiz.items():
            for a, b in merge(iv):
                if b - a > 0.05:
                    walls.append(WallSegment(round(a, 3), y, round(b, 3), y, int_t, False, "internal"))
        for x, iv in vert.items():
            for a, b in merge(iv):
                if b - a > 0.05:
                    walls.append(WallSegment(x, round(a, 3), x, round(b, 3), int_t, False, "internal"))
        return walls

    @staticmethod
    def _cut_openings(walls: List[WallSegment], cuts) -> List[WallSegment]:
        """Split walls where doors, arches and gates create openings."""
        out: List[WallSegment] = []
        for w in walls:
            pieces = [w]
            for (x1, y1, x2, y2) in cuts:
                nxt = []
                for p in pieces:
                    if p.is_horizontal and abs(y1 - y2) < 1e-6 and abs(p.y1 - y1) < 0.02:
                        lo, hi = sorted((p.x1, p.x2))
                        a, b = sorted((x1, x2))
                        if b > lo + 1e-6 and a < hi - 1e-6:
                            if a - lo > 0.05:
                                nxt.append(WallSegment(lo, p.y1, a, p.y2, p.thickness, p.is_external, p.wall_type))
                            if hi - b > 0.05:
                                nxt.append(WallSegment(b, p.y1, hi, p.y2, p.thickness, p.is_external, p.wall_type))
                            continue
                    if p.is_vertical and abs(x1 - x2) < 1e-6 and abs(p.x1 - x1) < 0.02:
                        lo, hi = sorted((p.y1, p.y2))
                        a, b = sorted((y1, y2))
                        if b > lo + 1e-6 and a < hi - 1e-6:
                            if a - lo > 0.05:
                                nxt.append(WallSegment(p.x1, lo, p.x2, a, p.thickness, p.is_external, p.wall_type))
                            if hi - b > 0.05:
                                nxt.append(WallSegment(p.x1, b, p.x2, hi, p.thickness, p.is_external, p.wall_type))
                            continue
                    nxt.append(p)
                pieces = nxt
            out.extend(pieces)
        return out

    # =========================================================================
    # Windows
    # =========================================================================

    @staticmethod
    def _exterior_edges(r, footprint, tol: float = 0.03):
        fx, fy, fw, fd = footprint
        edges = []
        if abs(r.top - (fy + fd)) < tol:
            edges.append(("N", r.x, r.right, r.top))
        if abs(r.y - fy) < tol:
            edges.append(("S", r.x, r.right, r.y))
        if abs(r.right - (fx + fw)) < tol:
            edges.append(("E", r.y, r.top, r.right))
        if abs(r.x - fx) < tol:
            edges.append(("W", r.y, r.top, r.x))
        return edges

    def _place_windows(self, rooms: List[RoomRect], doors: List[DoorPlacement], footprint) -> List[WindowPlacement]:
        """
        NBC 2016 Part 3 §8: habitable rooms need openings ≥ 1/10 of the floor area
        (kitchens 1/10, bath ventilators ≥ 0.37 m²). Glazing is split across the
        room's exterior walls, favouring the walls that face compass N and E.
        """
        windows: List[WindowPlacement] = []
        spec = {
            "LIVING": (1.2, 1.2, 2.4), "DINING": (1.2, 1.2, 2.1), "BEDROOM": (1.2, 1.0, 2.1),
            "STUDY": (1.2, 1.0, 1.8), "KITCHEN": (1.0, 0.9, 1.5), "SERVANT": (1.0, 0.75, 1.2),
            "BATHROOM": (0.45, 0.6, 0.9), "UTILITY": (0.9, 0.6, 1.2), "POOJA": (0.9, 0.6, 0.9),
            "STORE": (0.45, 0.6, 0.6),
        }
        good = {"N": 2, "E": 2, "W": 1, "S": 0}
        for r in rooms:
            if _is_site_room(r) or r.category not in spec:
                continue
            if "Dress" in r.name or "Wardrobe" in r.name or "Duct" in r.name:
                continue
            edges = [e for e in self._exterior_edges(r, footprint) if e[2] - e[1] >= 0.9]
            if not edges:
                continue
            h, wmin, wmax = spec[r.category]
            need = max(r.w * r.h / 10.0, 0.37 if r.category == "BATHROOM" else 0.8)
            compass = {e[0]: self.sides[{"N": "top", "S": "bottom", "E": "right", "W": "left"}[e[0]]] for e in edges}
            edges.sort(key=lambda e: (-good.get(compass[e[0]], 0), -(e[2] - e[1])))
            max_windows = 2 if r.category in ("LIVING", "BEDROOM", "DINING", "STUDY") else 1
            chosen = edges[:max_windows]
            per = need / len(chosen)
            for side, lo, hi, coord in chosen:
                L = hi - lo
                width = round(_clamp(per / h, wmin, min(wmax, 0.6 * L)), 2)
                c = (lo + hi) / 2
                # keep clear of a door on the same wall
                for d in doors:
                    if d.wall_side == side or d.is_main_entrance:
                        if side in ("N", "S") and abs(d.y - coord) < 0.05:
                            d0, d1 = d.x, d.x + d.width
                        elif side in ("E", "W") and abs(d.x - coord) < 0.05:
                            d0, d1 = d.y, d.y + d.width
                        else:
                            continue
                        if c + width / 2 > d0 - 0.3 and c - width / 2 < d1 + 0.3:
                            left_room = d0 - 0.3 - lo
                            right_room = hi - d1 - 0.3
                            if right_room >= left_room:
                                c = d1 + 0.3 + min(width, right_room) / 2
                                width = round(min(width, max(wmin * 0.75, right_room)), 2)
                            else:
                                c = lo + left_room / 2
                                width = round(min(width, max(wmin * 0.75, left_room)), 2)
                if width < 0.4:
                    continue
                if side in ("N", "S"):
                    windows.append(WindowPlacement(x=round(c, 3), y=round(coord, 3), width=width,
                                                   height=h, wall_side=side, room_name=r.name))
                else:
                    windows.append(WindowPlacement(x=round(coord, 3), y=round(c, 3), width=width,
                                                   height=width, wall_side=side, room_name=r.name))
        return windows

    # =========================================================================
    # Staircases
    # =========================================================================

    def _place_staircases(self, rooms: List[RoomRect], program=None, footprint=None) -> List[StaircaseRect]:
        staircases = []
        for r in rooms:
            if r.category != RoomCategory.STAIRS.value:
                continue
            external = "External" in r.name
            design = None
            if program is not None:
                design = program.external_stair if external else program.stair
            stype = "straight" if external else self.stair_type
            direction = "N" if r.h >= r.w else "E"
            if external and footprint is not None:
                direction = self._ext_stair_climb(r, footprint)
            if design is not None:
                staircases.append(StaircaseRect(
                    name=r.name, stair_type=stype, x=r.x, y=r.y, w=r.w, h=r.h, direction=direction,
                    num_steps=design.risers, tread_depth_m=design.tread,
                    riser_height_m=design.riser_height, has_landing=design.flights > 1,
                    landing_depth_m=design.landing_depth))
            else:
                staircases.append(StaircaseRect(name=r.name, stair_type=stype, x=r.x, y=r.y,
                                                 w=r.w, h=r.h, direction=direction))
        return staircases

    def _ext_stair_climb(self, r: RoomRect, footprint) -> str:
        """Climb direction of the outside stair (SitePlanner.ext_stair_climb: top alongside, S / W)."""
        from floorplan_generator.site_planner import SitePlanner
        sp = SitePlanner(self.plot_width, self.plot_height, self.orientation, self.vastu_enabled)
        return sp.ext_stair_climb((r.x, r.y, r.w, r.h), footprint)

    def _terrace_stair_info(self, rooms: List[RoomRect], staircases, site) -> Dict:
        """Where the outside terrace stair stands, measured, in the owner's words (engineering report)."""
        r = next((x for x in rooms if x.name == TERRACE_STAIR), None)
        side = getattr(site, "ext_stair_side", None)
        if r is None or side is None:
            return {}
        from floorplan_generator.vastu_engine import VastuEngine
        from floorplan_generator.intent_engine import ZONE_NAMES
        eng = VastuEngine.for_facing(self.plot_width, self.plot_height, self.orientation)
        zone = eng.get_zone_for_position(r.cx, r.cy).value
        score = eng.get_room_score("STAIRS", zone)
        sc = next((x for x in staircases if x.name == TERRACE_STAIR), None)
        plan_side = {"N": "top", "S": "bottom", "E": "right", "W": "left"}
        climb = self.sides[plan_side[sc.direction]] if sc is not None and sc.direction in plan_side else None
        word = {v: k for k, v in STAIR_SIDES.items()}[side]
        compass = self.sides[side]
        return {"side": side, "word": word, "compass": compass, "compass_name": ZONE_NAMES[compass],
                "zone": zone, "zone_name": ZONE_NAMES.get(zone, zone), "score": score,
                "vastu_word": "good" if score >= 5 else ("fine" if score >= 0 else "not good"),
                "taboo": eng.is_taboo("STAIRS", zone),
                "climb": climb, "climb_ok": climb in ("S", "W"),
                "rect": [round(r.x, 3), round(r.y, 3), round(r.w, 3), round(r.h, 3)],
                "want": {v: k for k, v in STAIR_SIDES.items()}.get(getattr(site, "ext_stair_want", None)),
                "refused": getattr(site, "ext_stair_refused", None)}

    # =========================================================================
    # Dimensions
    # =========================================================================

    def _generate_dimensions(self, rooms: List[RoomRect]) -> List[DimLine]:
        """
        Three-tier dimensions along the bottom and left plot edges:
          Tier 1 — setbacks + room partitions on the rear / left facade
          Tier 2 — setbacks + building envelope
          Tier 3 — overall plot
        """
        dims: List[DimLine] = []
        W, H = self.plot_width, self.plot_height
        building = [r for r in rooms if not _is_site_room(r)]
        if building:
            bx0 = min(r.x for r in building)
            by0 = min(r.y for r in building)
            bx1 = max(r.right for r in building)
            by1 = max(r.top for r in building)
        else:
            bx0, by0, bx1, by1 = 0.0, 0.0, W, H

        def chain(points, horizontal, tier, offset):
            pts = sorted({round(p, 3) for p in points})
            for a, b in zip(pts, pts[1:]):
                if b - a < 0.3:
                    continue
                if horizontal:
                    dims.append(DimLine(a, 0, b, 0, offset, f"{(b - a) * 1000:.0f}", tier))
                else:
                    dims.append(DimLine(0, a, 0, b, offset, f"{(b - a) * 1000:.0f}", tier))

        rear = [r for r in building if abs(r.y - by0) < 0.03]
        left = [r for r in building if abs(r.x - bx0) < 0.03]
        chain([0, W, bx0, bx1] + [r.x for r in rear] + [r.right for r in rear], True, 1, -0.6)
        chain([0, H, by0, by1] + [r.y for r in left] + [r.top for r in left], False, 1, -0.6)
        if bx0 > 0.3 or bx1 < W - 0.3:
            chain([0, bx0, bx1, W], True, 2, -1.2)
        if by0 > 0.3 or by1 < H - 0.3:
            chain([0, by0, by1, H], False, 2, -1.2)
        dims.append(DimLine(0, 0, W, 0, -1.8, f"{W * 1000:.0f}", 3))
        dims.append(DimLine(0, 0, 0, H, -1.8, f"{H * 1000:.0f}", 3))
        return dims

    # =========================================================================
    # Areas, Vastu audit, engineering report
    # =========================================================================

    def _set_clear_areas(self, rooms: List[RoomRect], footprint) -> None:
        """Room areas become clear (carpet) areas: centre-line rectangle minus half-walls."""
        fx, fy, fw, fd = footprint
        ext, inn = self.wall.external, self.wall.internal
        for r in rooms:
            if _is_site_room(r):
                r.area = round(r.w * r.h, 2)
                continue
            tl = ext if abs(r.x - fx) < 0.03 else inn
            tr = ext if abs(r.right - (fx + fw)) < 0.03 else inn
            tb = ext if abs(r.y - fy) < 0.03 else inn
            tt = ext if abs(r.top - (fy + fd)) < 0.03 else inn
            r.area = round(max(0.0, (r.w - tl / 2 - tr / 2) * (r.h - tb / 2 - tt / 2)), 2)

    def _vastu_audit(self, rooms: List[RoomRect], doors: List[DoorPlacement], site) -> Tuple[float, Dict]:
        """Weighted Vastu score: house mandala + main-door pada + site mandala + Brahmasthan."""
        setbacks = {self.sides[k]: v for k, v in site.setbacks.items()}
        return audit_vastu(rooms, doors, self.plot_width, self.plot_height, self.orientation,
                           site.footprint, setbacks, self.vastu_enabled)

    def _engineering_report(self, rooms, walls, program, site, carpet, built_up) -> Dict:
        fx, fy, fw, fd = site.footprint
        building = [r for r in rooms if not _is_site_room(r)]
        notes: List[str] = []

        # Structural frame: RCC columns on a grid set out along wall centre-lines (bays 2.4–4.5 m)
        from floorplan_generator.structure import structural_grid
        grid_x = getattr(site, "grid_x", None)            # columns set out for the cars under a stilt (apartment.py)
        grid = structural_grid(rooms, site.footprint, x_lines=grid_x)
        columns = len(grid.columns)
        span = max((min(r.w, r.h) for r in building
                    if r.category in ("LIVING", "DINING", "BEDROOM", "KITCHEN", "STUDY")), default=3.0)
        slab = max(125, int(math.ceil(span * 1000 / 32 / 5.0) * 5))
        beam_d = max(300, int(math.ceil(max(span, grid.max_bay) * 1000 / 12 / 25.0) * 25))
        notes.append(f"RCC frame: {columns} columns (230 × 300 mm) on a {len(grid.xs)} × {len(grid.ys)} grid "
                     f"(max bay {grid.max_bay:.2f} m); largest clear span {span:.2f} m → {slab} mm two-way slab, "
                     f"230 × {beam_d} mm beams (L/12), M20 concrete, Fe500 steel.")
        if grid_x:
            notes.append("Column lines across the front are set out for the cars on the stilt floor (between the bays, "
                         "never in a car's way); walls above that are off these lines stand on the beams and slab, and a "
                         "few columns show inside rooms as exposed columns — usual in stilt buildings.")

        # Plumbing stacks
        wet = [r for r in building if r.category in ("BATHROOM", "KITCHEN", "UTILITY")]
        groups = []
        for r in wet:
            g = next((gg for gg in groups if any(self._shared(r, o) for o in gg)), None)
            if g is None:
                groups.append([r])
            else:
                g.append(r)
        if wet:
            notes.append(f"Plumbing: {len(wet)} wet rooms grouped into {len(groups)} vertical stack(s) "
                         "(shared shafts cut pipe runs and leak points).")
        internal_baths = [r.name for r in building if r.category == "BATHROOM"
                          and not self._exterior_edges(r, site.footprint)]
        if internal_baths:
            notes.append("Mechanical exhaust (≥ 10 air changes/h) needed for internal toilets without an "
                         "outside wall: " + ", ".join(internal_baths) + ".")
        habitable_dark = [r.name for r in building if r.category in ("BEDROOM", "LIVING", "STUDY")
                          and not self._exterior_edges(r, site.footprint)]
        if habitable_dark:
            notes.append("Daylight warning (NBC Part 3 §8): no outside wall for " + ", ".join(habitable_dark) + ".")
        notes.append("Room labels show centre-line dimensions; areas are clear carpet areas (RERA definition).")

        floors = self.num_floors
        return {
            "footprint_m": [round(fw, 2), round(fd, 2)],
            "setbacks_m": {self.sides[k]: v for k, v in site.setbacks.items()},
            "ground_coverage_pct": round(site.coverage * 100, 1),
            "far": round(built_up * floors / (self.plot_width * self.plot_height), 2),
            "carpet_m2_per_floor": round(carpet, 2),
            "built_up_m2_per_floor": round(built_up, 2),
            "built_up_m2_total": round(built_up * floors, 2),
            "columns": columns,
            "grid": {"x": grid.xs, "y": grid.ys, "max_bay_m": round(grid.max_bay, 2)},
            **({"grid_x": [round(v, 3) for v in grid_x]} if grid_x else {}),
            "max_span_m": round(span, 2),
            "slab_mm": slab,
            "beam_mm": [230, beam_d],
            "stair": None if not program.stair else {
                "type": program.stair.stair_type, "risers": program.stair.risers,
                "riser_mm": round(program.stair.riser_height * 1000), "tread_mm": round(program.stair.tread * 1000),
                "flight_width_m": program.stair.flight_width},
            "lift": None if not program.lift else {
                "persons": program.lift.persons, "load_kg": program.lift.rated_load_kg,
                "shaft_m": [program.lift.shaft_w, program.lift.shaft_d]},
            "car_parking": site.car_mode,
            "site_items": [it.name for it in site.items],
            "notes": notes,
        }
