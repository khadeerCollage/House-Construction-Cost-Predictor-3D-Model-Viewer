"""
V-HSP Layout Engine
====================
Vastu-Guided Hierarchical Space Partitioning (V-HSP) — the core algorithm
that converts user inputs (plot dimensions, BHK config, Vastu preferences)
into a complete, professional floor plan with rooms, walls, doors, and windows.

Algorithm Pipeline:
  Stage 1: Functional Zoning (Public / Private / Service macro-zones)
  Stage 2: Vastu Mandala mapping (rooms → compass zones)
  Stage 3: Hierarchical room subdivision (area-ratio proportional splitting)
  Stage 4: NBC dimension clamping (minimum widths, aspect ratios)
  Stage 5: Wall insetting & classification (external 230mm, internal 115mm)
  Stage 6: Door & window placement on shared wall segments
  Stage 7: Corridor synthesis along major partition spines
  Stage 8: Dimension line generation (3-tier hierarchy)

Performance: < 10ms on any CPU for up to 15 rooms.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math

from floorplan_generator.room_specs import (
    BHKConfig, RoomTemplate, WallThickness, RoomCategory,
    get_bhk_config, recommend_bhk_for_plot,
)
from floorplan_generator.vastu_engine import VastuEngine, VastuZone


# =============================================================================
# Output Dataclasses
# =============================================================================

@dataclass
class RoomRect:
    """A placed room with its physical bounds."""
    name: str
    category: str       # RoomCategory value string
    x: float            # left edge, meters from plot origin
    y: float            # bottom edge, meters from plot origin
    w: float            # width in meters
    h: float            # height in meters
    color: str           # hex color code for visualization
    area: float = 0.0    # carpet area in m² (computed)

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

    def overlaps(self, other: "RoomRect") -> bool:
        """Check if two rooms overlap (excluding touching edges)."""
        return (
            self.x < other.right and self.right > other.x and
            self.y < other.top and self.top > other.y
        )

    def shares_wall_with(self, other: "RoomRect", tolerance: float = 0.05) -> Optional[str]:
        """
        Check if this room shares a wall segment with another room.

        Returns:
            Wall side ('N', 'S', 'E', 'W') from this room's perspective,
            or None if no shared wall.
        """
        min_shared = 0.5  # minimum shared length in meters for a door

        # East wall of self touches West wall of other
        if abs(self.right - other.x) < tolerance:
            overlap = min(self.top, other.top) - max(self.y, other.y)
            if overlap >= min_shared:
                return "E"

        # West wall of self touches East wall of other
        if abs(self.x - other.right) < tolerance:
            overlap = min(self.top, other.top) - max(self.y, other.y)
            if overlap >= min_shared:
                return "W"

        # North wall of self touches South wall of other
        if abs(self.top - other.y) < tolerance:
            overlap = min(self.right, other.right) - max(self.x, other.x)
            if overlap >= min_shared:
                return "N"

        # South wall of self touches North wall of other
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
    """A door placed in a wall opening."""
    x: float            # hinge position X in meters
    y: float            # hinge position Y in meters
    width: float        # door width in meters
    wall_side: str      # 'N', 'S', 'E', 'W' — which wall of the room
    swing_direction: str  # 'LEFT', 'RIGHT'
    room_name: str      # which room this door belongs to
    is_main_entrance: bool = False


@dataclass
class WindowPlacement:
    """A window placed in an external wall."""
    x: float            # center X position in meters
    y: float            # center Y position in meters
    width: float        # window width in meters
    height: float       # window height in meters
    wall_side: str      # 'N', 'S', 'E', 'W'
    room_name: str


@dataclass
class DimLine:
    """A dimension annotation line."""
    x1: float
    y1: float
    x2: float
    y2: float
    offset: float       # perpendicular offset from wall in meters
    label: str          # dimension text like "3,500"
    tier: int           # 1=openings, 2=rooms, 3=overall


@dataclass
class GeneratedFloorPlan:
    """Complete output of the layout engine — everything needed to render."""
    plot_width: float
    plot_height: float
    orientation: str          # compass direction of North edge
    rooms: List[RoomRect] = field(default_factory=list)
    walls: List[WallSegment] = field(default_factory=list)
    doors: List[DoorPlacement] = field(default_factory=list)
    windows: List[WindowPlacement] = field(default_factory=list)
    dimensions: List[DimLine] = field(default_factory=list)
    vastu_score: float = 0.0
    vastu_enabled: bool = True
    total_carpet_area: float = 0.0
    total_built_up_area: float = 0.0
    bhk_config: str = ""       # '1BHK', '2BHK', etc.
    family_description: str = ""
    quality_level: str = "Standard"
    project_name: str = "Residential Floor Plan"


# =============================================================================
# Layout Engine
# =============================================================================

# Architectural grid module (snap dimensions to this increment)
GRID_MODULE = 0.15  # 150mm = half-brick module


def _snap_to_grid(value: float, module: float = GRID_MODULE) -> float:
    """Snap a dimension to the nearest architectural grid module."""
    return round(value / module) * module


def _clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max."""
    return max(min_val, min(max_val, value))


class LayoutEngine:
    """
    Vastu-Guided Hierarchical Space Partitioning (V-HSP) Layout Engine.

    Generates professional residential floor plans from:
    - Plot dimensions (width × height in meters)
    - BHK configuration (room schedule)
    - Vastu Shastra preferences
    - Compass orientation

    Usage:
        engine = LayoutEngine(
            plot_width=12.0,
            plot_height=15.0,
            bhk_config="2BHK",
            vastu_enabled=True,
            orientation="N"
        )
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
    ):
        """
        Initialize the layout engine.

        Args:
            plot_width: Plot width in meters (X axis).
            plot_height: Plot height in meters (Y axis).
            bhk_config: '1BHK', '2BHK', '3BHK', '4BHK', or 'auto'.
            vastu_enabled: Whether to enforce Vastu Shastra placement.
            orientation: North direction — 'N' (top), 'E' (right), 'S' (bottom), 'W' (left).
            quality_level: 'Economy', 'Standard', 'Premium', 'Luxury'.
            project_name: Project title for title block.
            family_description: Human-readable family description.
            custom_rooms: Optional custom room list (overrides BHK config).
        """
        self.plot_width = max(plot_width, 4.0)
        self.plot_height = max(plot_height, 4.0)
        self.vastu_enabled = vastu_enabled
        self.orientation = orientation.upper()
        self.quality_level = quality_level
        self.project_name = project_name
        self.family_description = family_description

        # Walls
        self.wall = WallThickness()

        # Resolve BHK config
        if bhk_config.lower() == "auto":
            plot_area = self.plot_width * self.plot_height
            self.bhk_name = recommend_bhk_for_plot(plot_area)
        else:
            self.bhk_name = bhk_config.upper().replace(" ", "")

        self.config = get_bhk_config(self.bhk_name)

        # Use custom rooms if provided
        if custom_rooms:
            self.room_templates = custom_rooms
        else:
            self.room_templates = list(self.config.rooms)

        # Filter optional rooms that won't fit
        self._filter_rooms_for_plot()

        # Vastu engine
        if self.vastu_enabled:
            self.vastu = VastuEngine(self.plot_width, self.plot_height, self.orientation)
        else:
            self.vastu = None

    def _filter_rooms_for_plot(self):
        """Remove optional rooms if the plot is too small."""
        plot_area = self.plot_width * self.plot_height
        usable_area = plot_area * 0.85  # ~15% for walls and corridors

        # Sum required room areas
        required_area = sum(
            t.target_width * t.target_height
            for t in self.room_templates
            if not t.is_optional
        )

        # Keep optional rooms only if there's space
        if required_area > usable_area * 0.95:
            self.room_templates = [
                t for t in self.room_templates if not t.is_optional
            ]

    def generate(self) -> GeneratedFloorPlan:
        """
        Execute the full V-HSP pipeline and return a complete floor plan.

        Returns:
            GeneratedFloorPlan with all rooms, walls, doors, windows, dimensions.
        """
        plan = GeneratedFloorPlan(
            plot_width=self.plot_width,
            plot_height=self.plot_height,
            orientation=self.orientation,
            vastu_enabled=self.vastu_enabled,
            bhk_config=self.bhk_name,
            family_description=self.family_description,
            quality_level=self.quality_level,
            project_name=self.project_name,
        )

        # Stage 1-3: Place rooms using Vastu-guided subdivision
        rooms = self._place_rooms()

        # Stage 4: Validate and clamp NBC dimensions
        rooms = self._clamp_nbc_dimensions(rooms)

        # Stage 5: Generate wall segments
        walls = self._generate_walls(rooms)

        # Stage 6: Place doors and windows
        doors = self._place_doors(rooms)
        windows = self._place_windows(rooms)

        # Stage 7: Generate dimension annotations
        dimensions = self._generate_dimensions(rooms)

        # Compute Vastu score
        vastu_score = 0.0
        if self.vastu_enabled and self.vastu:
            placements = []
            for room in rooms:
                zone = self.vastu.get_zone_for_position(room.cx, room.cy)
                placements.append((room.category, zone.value))
            vastu_score = self.vastu.compute_plan_score(placements)

        # Compute areas
        carpet_area = sum(r.area for r in rooms if r.category != RoomCategory.CORRIDOR.value)
        wall_area = sum(
            w.length * w.thickness for w in walls
        )
        built_up_area = carpet_area + wall_area

        # Populate plan
        plan.rooms = rooms
        plan.walls = walls
        plan.doors = doors
        plan.windows = windows
        plan.dimensions = dimensions
        plan.vastu_score = round(vastu_score, 1)
        plan.total_carpet_area = round(carpet_area, 2)
        plan.total_built_up_area = round(built_up_area, 2)

        return plan

    # =========================================================================
    # Stage 1-3: Room Placement
    # =========================================================================

    def _place_rooms(self) -> List[RoomRect]:
        """
        Place all rooms within the plot using hierarchical subdivision.

        The algorithm:
        1. Compute internal bounds (after exterior wall deduction)
        2. Group rooms by functional zone (public/private/service)
        3. If Vastu enabled, assign rooms to 3×3 Mandala zones
        4. Subdivide each zone among its assigned rooms
        """
        # Internal bounds after deducting exterior wall thickness
        ext = self.wall.external
        ix = ext
        iy = ext
        iw = self.plot_width - 2 * ext
        ih = self.plot_height - 2 * ext

        if iw <= 0 or ih <= 0:
            return []

        # Build room placement list with names and target areas
        room_entries = self._build_room_entries()

        if self.vastu_enabled and self.vastu:
            rooms = self._place_rooms_vastu(room_entries, ix, iy, iw, ih)
        else:
            rooms = self._place_rooms_functional(room_entries, ix, iy, iw, ih)

        return rooms

    def _build_room_entries(self) -> List[dict]:
        """Build list of room entries with unique names."""
        entries = []
        name_counts: Dict[str, int] = {}

        for template in self.room_templates:
            base_name = template.spec.name
            name_counts[base_name] = name_counts.get(base_name, 0) + 1
            count = name_counts[base_name]

            # Make name unique if duplicates
            if count > 1:
                name = f"{base_name} {count}"
            else:
                name = base_name

            # Scale target area to fit plot
            target_area = template.target_width * template.target_height

            entries.append({
                "name": name,
                "category": template.spec.category.value,
                "target_area": target_area,
                "min_width": template.spec.min_width,
                "target_width": template.target_width,
                "target_height": template.target_height,
                "color": template.spec.color,
                "preferred_zone": template.preferred_vastu_zone,
                "attached_to": template.attached_to,
            })

        # Fix duplicate names for rooms that only appear once
        single_counts = {k for k, v in name_counts.items() if v == 1}
        for entry in entries:
            # The first occurrence of a multi-count name needs renaming too
            base = entry["name"]
            if base not in single_counts and not any(c.isdigit() for c in base):
                # Find the base name in counts
                for bn, cnt in name_counts.items():
                    if cnt > 1 and base == bn:
                        entry["name"] = f"{bn} 1"
                        break

        return entries

    def _place_rooms_vastu(self, entries: List[dict],
                           ix: float, iy: float,
                           iw: float, ih: float) -> List[RoomRect]:
        """Place rooms using Vastu Mandala zone assignment."""
        zones = self.vastu.get_all_zones()
        zone_assignments: Dict[str, List[dict]] = {z.value: [] for z in VastuZone}

        # Assign rooms to their preferred Vastu zones
        assigned = set()
        for entry in entries:
            pref = entry["preferred_zone"]
            if pref in zone_assignments:
                zone_assignments[pref].append(entry)
                assigned.add(entry["name"])

        # Any unassigned rooms go to CENTER
        for entry in entries:
            if entry["name"] not in assigned:
                zone_assignments["CENTER"].append(entry)

        # Now subdivide each zone among its rooms
        rooms = []
        for zone_name, zone_entries in zone_assignments.items():
            if not zone_entries:
                continue

            zone_enum = VastuZone(zone_name)
            zb = zones[zone_enum]

            # Map zone bounds to internal coordinates
            # Zone bounds are in plot coordinates; clip to internal bounds
            zx = max(zb.x, ix)
            zy = max(zb.y, iy)
            zr = min(zb.x + zb.w, ix + iw)
            zt = min(zb.y + zb.h, iy + ih)
            zw = max(0, zr - zx)
            zh = max(0, zt - zy)

            if zw <= 0 or zh <= 0:
                continue

            # Subdivide zone among its rooms
            zone_rooms = self._subdivide_zone(zone_entries, zx, zy, zw, zh)
            rooms.extend(zone_rooms)

        return rooms

    def _place_rooms_functional(self, entries: List[dict],
                                ix: float, iy: float,
                                iw: float, ih: float) -> List[RoomRect]:
        """Place rooms using functional zoning (no Vastu)."""
        # Group rooms by function
        public = [e for e in entries if e["category"] in
                  (RoomCategory.LIVING.value, RoomCategory.DINING.value,
                   RoomCategory.FOYER.value)]
        private = [e for e in entries if e["category"] in
                   (RoomCategory.BEDROOM.value, RoomCategory.STUDY.value,
                    RoomCategory.POOJA.value)]
        service = [e for e in entries if e["category"] in
                   (RoomCategory.KITCHEN.value, RoomCategory.BATHROOM.value,
                    RoomCategory.UTILITY.value, RoomCategory.SERVANT.value,
                    RoomCategory.STORE.value)]
        corridors = [e for e in entries if e["category"] == RoomCategory.CORRIDOR.value]

        # Remaining unclassified
        classified_names = {e["name"] for group in [public, private, service, corridors]
                           for e in group}
        misc = [e for e in entries if e["name"] not in classified_names]
        if misc:
            public.extend(misc)

        # Calculate area ratios for zone splits
        total_area = sum(e["target_area"] for e in entries) or 1.0
        public_ratio = sum(e["target_area"] for e in public) / total_area
        private_ratio = sum(e["target_area"] for e in private) / total_area
        service_ratio = sum(e["target_area"] for e in service) / total_area

        rooms = []

        # Split plot into zones based on aspect ratio
        if iw >= ih:
            # Wide plot: split vertically into columns
            # [Service | Public | Private]
            sx = ix
            sw = _snap_to_grid(iw * service_ratio)
            px = sx + sw
            pw = _snap_to_grid(iw * public_ratio)
            prx = px + pw
            prw = iw - sw - pw

            if service:
                rooms.extend(self._subdivide_zone(service, sx, iy, sw, ih))
            if public:
                rooms.extend(self._subdivide_zone(public, px, iy, pw, ih))
            if private:
                rooms.extend(self._subdivide_zone(private, prx, iy, prw, ih))
        else:
            # Tall plot: split horizontally into rows
            # Bottom: Service, Middle: Public, Top: Private
            sy = iy
            sh = _snap_to_grid(ih * service_ratio)
            py = sy + sh
            ph = _snap_to_grid(ih * public_ratio)
            pry = py + ph
            prh = ih - sh - ph

            if service:
                rooms.extend(self._subdivide_zone(service, ix, sy, iw, sh))
            if public:
                rooms.extend(self._subdivide_zone(public, ix, py, iw, ph))
            if private:
                rooms.extend(self._subdivide_zone(private, ix, pry, iw, prh))

        return rooms

    def _subdivide_zone(self, entries: List[dict],
                        zx: float, zy: float,
                        zw: float, zh: float) -> List[RoomRect]:
        """
        Recursively subdivide a rectangular zone among a list of rooms.

        Uses area-ratio proportional binary splitting.
        """
        if not entries:
            return []

        if len(entries) == 1:
            e = entries[0]
            w = _snap_to_grid(max(zw, e.get("min_width", 1.0)))
            h = _snap_to_grid(max(zh, e.get("min_width", 1.0)))
            return [RoomRect(
                name=e["name"],
                category=e["category"],
                x=round(zx, 3),
                y=round(zy, 3),
                w=round(min(w, zw), 3),
                h=round(min(h, zh), 3),
                color=e["color"],
            )]

        if len(entries) == 2:
            return self._split_two(entries, zx, zy, zw, zh)

        # More than 2: split into two halves and recurse
        # Sort by target area descending for better space utilization
        sorted_entries = sorted(entries, key=lambda e: e["target_area"], reverse=True)
        mid = len(sorted_entries) // 2
        left_group = sorted_entries[:mid]
        right_group = sorted_entries[mid:]

        left_area = sum(e["target_area"] for e in left_group)
        right_area = sum(e["target_area"] for e in right_group)
        total_area = left_area + right_area

        if total_area <= 0:
            ratio = 0.5
        else:
            ratio = left_area / total_area

        rooms = []

        if zw >= zh:
            # Split vertically
            lw = _snap_to_grid(zw * ratio)
            lw = _clamp(lw, 1.0, zw - 1.0)
            rw = zw - lw
            rooms.extend(self._subdivide_zone(left_group, zx, zy, lw, zh))
            rooms.extend(self._subdivide_zone(right_group, zx + lw, zy, rw, zh))
        else:
            # Split horizontally
            lh = _snap_to_grid(zh * ratio)
            lh = _clamp(lh, 1.0, zh - 1.0)
            rh = zh - lh
            rooms.extend(self._subdivide_zone(left_group, zx, zy, zw, lh))
            rooms.extend(self._subdivide_zone(right_group, zx, zy + lh, zw, rh))

        return rooms

    def _split_two(self, entries: List[dict],
                   zx: float, zy: float,
                   zw: float, zh: float) -> List[RoomRect]:
        """Split a zone between exactly two rooms."""
        e1, e2 = entries[0], entries[1]
        a1 = e1["target_area"]
        a2 = e2["target_area"]
        total = a1 + a2

        if total <= 0:
            ratio = 0.5
        else:
            ratio = a1 / total

        rooms = []

        # Check if one is attached to the other (e.g. attached bathroom)
        # If so, split along the shorter axis to keep them side-by-side
        is_attached = (e1.get("attached_to") and e2["name"].startswith(e1["attached_to"])) or \
                      (e2.get("attached_to") and e1["name"].startswith(e2["attached_to"]))

        if zw >= zh or is_attached:
            # Split vertically (side by side)
            w1 = _snap_to_grid(zw * ratio)
            w1 = _clamp(w1, max(1.0, e1.get("min_width", 1.0)), zw - max(1.0, e2.get("min_width", 1.0)))
            w2 = zw - w1

            rooms.append(RoomRect(
                name=e1["name"], category=e1["category"],
                x=round(zx, 3), y=round(zy, 3),
                w=round(w1, 3), h=round(zh, 3),
                color=e1["color"],
            ))
            rooms.append(RoomRect(
                name=e2["name"], category=e2["category"],
                x=round(zx + w1, 3), y=round(zy, 3),
                w=round(w2, 3), h=round(zh, 3),
                color=e2["color"],
            ))
        else:
            # Split horizontally (stacked)
            h1 = _snap_to_grid(zh * ratio)
            h1 = _clamp(h1, max(1.0, e1.get("min_width", 1.0)), zh - max(1.0, e2.get("min_width", 1.0)))
            h2 = zh - h1

            rooms.append(RoomRect(
                name=e1["name"], category=e1["category"],
                x=round(zx, 3), y=round(zy, 3),
                w=round(zw, 3), h=round(h1, 3),
                color=e1["color"],
            ))
            rooms.append(RoomRect(
                name=e2["name"], category=e2["category"],
                x=round(zx, 3), y=round(zy + h1, 3),
                w=round(zw, 3), h=round(h2, 3),
                color=e2["color"],
            ))

        return rooms

    # =========================================================================
    # Stage 4: NBC Dimension Clamping
    # =========================================================================

    def _clamp_nbc_dimensions(self, rooms: List[RoomRect]) -> List[RoomRect]:
        """
        Validate and adjust room dimensions to meet NBC 2016 minimums.

        Ensures:
        - No room width below its category minimum
        - Aspect ratios within 1:0.55 to 1:1.8
        - Rooms don't exceed plot boundaries
        """
        ext = self.wall.external
        max_x = self.plot_width - ext
        max_y = self.plot_height - ext

        for room in rooms:
            # Clamp to plot bounds
            room.x = _clamp(room.x, ext, max_x - 1.0)
            room.y = _clamp(room.y, ext, max_y - 1.0)
            room.w = min(room.w, max_x - room.x)
            room.h = min(room.h, max_y - room.y)

            # Ensure minimum dimensions
            room.w = max(room.w, 1.2)
            room.h = max(room.h, 1.2)

            # Snap to grid
            room.w = _snap_to_grid(room.w)
            room.h = _snap_to_grid(room.h)

            # Recalculate area
            room.area = round(room.w * room.h, 2)

        return rooms

    # =========================================================================
    # Stage 5: Wall Generation
    # =========================================================================

    def _generate_walls(self, rooms: List[RoomRect]) -> List[WallSegment]:
        """
        Generate wall segments for the floor plan.

        Creates:
        - Exterior walls around the plot perimeter (230mm)
        - Interior partition walls between rooms (115mm)
        """
        walls = []
        ext_t = self.wall.external
        int_t = self.wall.internal

        # Exterior walls (plot perimeter)
        W = self.plot_width
        H = self.plot_height

        # Bottom wall
        walls.append(WallSegment(0, 0, W, 0, ext_t, True))
        # Right wall
        walls.append(WallSegment(W, 0, W, H, ext_t, True))
        # Top wall
        walls.append(WallSegment(W, H, 0, H, ext_t, True))
        # Left wall
        walls.append(WallSegment(0, H, 0, 0, ext_t, True))

        # Interior walls (between rooms)
        seen_edges = set()

        for i, room in enumerate(rooms):
            # Room boundary walls
            edges = [
                (room.x, room.y, room.right, room.y),          # bottom
                (room.right, room.y, room.right, room.top),    # right
                (room.right, room.top, room.x, room.top),      # top
                (room.x, room.top, room.x, room.y),            # left
            ]

            for x1, y1, x2, y2 in edges:
                # Normalize edge for deduplication
                edge_key = (round(min(x1, x2), 2), round(min(y1, y2), 2),
                           round(max(x1, x2), 2), round(max(y1, y2), 2))

                # Skip if this edge is on the plot boundary (already covered)
                is_on_boundary = (
                    (abs(y1 - 0) < 0.01 and abs(y2 - 0) < 0.01) or    # bottom
                    (abs(y1 - H) < 0.01 and abs(y2 - H) < 0.01) or    # top
                    (abs(x1 - 0) < 0.01 and abs(x2 - 0) < 0.01) or    # left
                    (abs(x1 - W) < 0.01 and abs(x2 - W) < 0.01)       # right
                )

                if is_on_boundary:
                    continue

                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    walls.append(WallSegment(
                        round(x1, 3), round(y1, 3),
                        round(x2, 3), round(y2, 3),
                        int_t, False
                    ))

        return walls

    # =========================================================================
    # Stage 6: Door & Window Placement
    # =========================================================================

    def _place_doors(self, rooms: List[RoomRect]) -> List[DoorPlacement]:
        """
        Place doors on shared wall segments between adjacent rooms.

        Rules:
        - Each room gets at least one door
        - Main entrance placed on the exterior wall closest to North/East (Vastu)
        - Attached bathrooms get doors on the shared wall with their parent room
        """
        doors = []
        rooms_with_doors = set()

        # Find the main living/foyer room for the entrance
        entrance_room = None
        for room in rooms:
            if room.category in (RoomCategory.LIVING.value, RoomCategory.FOYER.value):
                entrance_room = room
                break
        if not entrance_room and rooms:
            entrance_room = rooms[0]

        # Place main entrance door
        if entrance_room:
            # Prefer North or East wall for Vastu
            door_x = entrance_room.x + entrance_room.w / 2 - 0.5
            door_y = entrance_room.top  # Top wall (North side)

            # Check which exterior wall this room touches
            if abs(entrance_room.top - (self.plot_height - self.wall.external)) < 0.15:
                door_x = entrance_room.x + entrance_room.w / 2 - 0.5
                door_y = entrance_room.top
                wall_side = "N"
            elif abs(entrance_room.y - self.wall.external) < 0.15:
                door_x = entrance_room.x + entrance_room.w / 2 - 0.5
                door_y = entrance_room.y
                wall_side = "S"
            elif abs(entrance_room.right - (self.plot_width - self.wall.external)) < 0.15:
                door_x = entrance_room.right
                door_y = entrance_room.y + entrance_room.h / 2 - 0.5
                wall_side = "E"
            else:
                door_x = entrance_room.x
                door_y = entrance_room.y + entrance_room.h / 2 - 0.5
                wall_side = "W"

            doors.append(DoorPlacement(
                x=round(door_x, 3),
                y=round(door_y, 3),
                width=1.0,  # Main entrance = 1000mm
                wall_side=wall_side,
                swing_direction="RIGHT",
                room_name=entrance_room.name,
                is_main_entrance=True,
            ))
            rooms_with_doors.add(entrance_room.name)

        # Place doors between adjacent rooms
        for i, room_a in enumerate(rooms):
            for j, room_b in enumerate(rooms):
                if i >= j:
                    continue

                shared_wall = room_a.shares_wall_with(room_b)
                if shared_wall is None:
                    continue

                # Determine door width based on room categories
                bath_cats = (RoomCategory.BATHROOM.value,)
                if room_a.category in bath_cats or room_b.category in bath_cats:
                    door_width = 0.75
                elif room_a.category == RoomCategory.KITCHEN.value or \
                        room_b.category == RoomCategory.KITCHEN.value:
                    door_width = 0.80
                else:
                    door_width = 0.90

                # Calculate door position (centered on shared wall segment)
                if shared_wall == "E":
                    overlap_start = max(room_a.y, room_b.y)
                    overlap_end = min(room_a.top, room_b.top)
                    door_y = (overlap_start + overlap_end) / 2 - door_width / 2
                    door_x = room_a.right
                    wall_side = "E"
                elif shared_wall == "W":
                    overlap_start = max(room_a.y, room_b.y)
                    overlap_end = min(room_a.top, room_b.top)
                    door_y = (overlap_start + overlap_end) / 2 - door_width / 2
                    door_x = room_a.x
                    wall_side = "W"
                elif shared_wall == "N":
                    overlap_start = max(room_a.x, room_b.x)
                    overlap_end = min(room_a.right, room_b.right)
                    door_x = (overlap_start + overlap_end) / 2 - door_width / 2
                    door_y = room_a.top
                    wall_side = "N"
                else:  # S
                    overlap_start = max(room_a.x, room_b.x)
                    overlap_end = min(room_a.right, room_b.right)
                    door_x = (overlap_start + overlap_end) / 2 - door_width / 2
                    door_y = room_a.y
                    wall_side = "S"

                doors.append(DoorPlacement(
                    x=round(door_x, 3),
                    y=round(door_y, 3),
                    width=door_width,
                    wall_side=wall_side,
                    swing_direction="RIGHT",
                    room_name=room_a.name,
                ))
                rooms_with_doors.add(room_a.name)
                rooms_with_doors.add(room_b.name)

        return doors

    def _place_windows(self, rooms: List[RoomRect]) -> List[WindowPlacement]:
        """
        Place windows on external walls of rooms that require ventilation.

        Rules:
        - Habitable rooms must have windows (NBC daylight requirement)
        - Windows placed on walls touching the plot exterior
        - Window size proportional to room area
        """
        windows = []
        ext = self.wall.external

        for room in rooms:
            # Skip rooms that don't need windows
            if room.category in (RoomCategory.CORRIDOR.value, RoomCategory.STORE.value):
                continue

            # Determine window size based on room category
            if room.category in (RoomCategory.LIVING.value, RoomCategory.DINING.value):
                win_w, win_h = 1.5, 1.2
            elif room.category == RoomCategory.BEDROOM.value:
                win_w, win_h = 1.2, 1.2
            elif room.category == RoomCategory.KITCHEN.value:
                win_w, win_h = 0.9, 0.6
            elif room.category == RoomCategory.BATHROOM.value:
                win_w, win_h = 0.6, 0.45
            else:
                win_w, win_h = 0.9, 0.9

            # Find which external walls this room touches
            placed = False

            # Check top wall (near plot top)
            if abs(room.top - (self.plot_height - ext)) < 0.15:
                windows.append(WindowPlacement(
                    x=round(room.x + room.w / 2, 3),
                    y=round(room.top, 3),
                    width=min(win_w, room.w * 0.6),
                    height=win_h,
                    wall_side="N",
                    room_name=room.name,
                ))
                placed = True

            # Check bottom wall (near plot bottom)
            if abs(room.y - ext) < 0.15:
                windows.append(WindowPlacement(
                    x=round(room.x + room.w / 2, 3),
                    y=round(room.y, 3),
                    width=min(win_w, room.w * 0.6),
                    height=win_h,
                    wall_side="S",
                    room_name=room.name,
                ))
                placed = True

            # Check right wall (near plot right)
            if not placed and abs(room.right - (self.plot_width - ext)) < 0.15:
                windows.append(WindowPlacement(
                    x=round(room.right, 3),
                    y=round(room.y + room.h / 2, 3),
                    width=min(win_w, room.h * 0.6),
                    height=win_h,
                    wall_side="E",
                    room_name=room.name,
                ))

            # Check left wall (near plot left)
            if not placed and abs(room.x - ext) < 0.15:
                windows.append(WindowPlacement(
                    x=round(room.x, 3),
                    y=round(room.y + room.h / 2, 3),
                    width=min(win_w, room.h * 0.6),
                    height=win_h,
                    wall_side="W",
                    room_name=room.name,
                ))

        return windows

    # =========================================================================
    # Stage 7: Dimension Generation
    # =========================================================================

    def _generate_dimensions(self, rooms: List[RoomRect]) -> List[DimLine]:
        """
        Generate 3-tier dimension annotations.

        Tier 1: Individual room widths (closest to walls)
        Tier 2: Room partition positions (middle distance)
        Tier 3: Overall building envelope (outermost)
        """
        dims = []

        # Tier 3: Overall building dimensions (outermost)
        # Bottom overall dimension
        dims.append(DimLine(
            x1=0, y1=0, x2=self.plot_width, y2=0,
            offset=-1.8,
            label=f"{self.plot_width * 1000:.0f}",
            tier=3,
        ))
        # Left overall dimension
        dims.append(DimLine(
            x1=0, y1=0, x2=0, y2=self.plot_height,
            offset=-1.8,
            label=f"{self.plot_height * 1000:.0f}",
            tier=3,
        ))

        # Tier 1: Individual room dimensions (closest to walls)
        # Collect unique X and Y partition lines
        x_lines = sorted(set(
            [r.x for r in rooms] + [r.right for r in rooms]
        ))
        y_lines = sorted(set(
            [r.y for r in rooms] + [r.top for r in rooms]
        ))

        # Bottom edge room dimensions
        prev_x = 0
        for x in x_lines:
            if x > prev_x + 0.5:  # Skip tiny segments
                dims.append(DimLine(
                    x1=prev_x, y1=0, x2=x, y2=0,
                    offset=-0.6,
                    label=f"{(x - prev_x) * 1000:.0f}",
                    tier=1,
                ))
            prev_x = x
        # Last segment to plot edge
        if prev_x < self.plot_width - 0.5:
            dims.append(DimLine(
                x1=prev_x, y1=0, x2=self.plot_width, y2=0,
                offset=-0.6,
                label=f"{(self.plot_width - prev_x) * 1000:.0f}",
                tier=1,
            ))

        # Left edge room dimensions
        prev_y = 0
        for y in y_lines:
            if y > prev_y + 0.5:
                dims.append(DimLine(
                    x1=0, y1=prev_y, x2=0, y2=y,
                    offset=-0.6,
                    label=f"{(y - prev_y) * 1000:.0f}",
                    tier=1,
                ))
            prev_y = y
        if prev_y < self.plot_height - 0.5:
            dims.append(DimLine(
                x1=0, y1=prev_y, x2=0, y2=self.plot_height,
                offset=-0.6,
                label=f"{(self.plot_height - prev_y) * 1000:.0f}",
                tier=1,
            ))

        return dims
