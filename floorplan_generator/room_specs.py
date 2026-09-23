"""
Indian Residential Architecture Standards Database
====================================================
NBC 2016 (National Building Code of India, Part 3) compliant room specifications,
standard Indian BHK configurations, wall thicknesses, door/window standards.

This module serves as the single source of truth for all architectural dimensions
and constraints used by the layout engine.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


# =============================================================================
# Room Categories
# =============================================================================

class RoomCategory(str, Enum):
    """Architectural room classification."""
    LIVING = "LIVING"
    DINING = "DINING"
    KITCHEN = "KITCHEN"
    BEDROOM = "BEDROOM"
    BATHROOM = "BATHROOM"
    POOJA = "POOJA"
    STUDY = "STUDY"
    CORRIDOR = "CORRIDOR"
    UTILITY = "UTILITY"
    BALCONY = "BALCONY"
    FOYER = "FOYER"
    SERVANT = "SERVANT"
    STORE = "STORE"


# =============================================================================
# Core Dataclasses
# =============================================================================

@dataclass
class RoomSpec:
    """
    NBC 2016 compliant room specification.
    All dimensions in meters. Areas in square meters.
    """
    name: str
    category: RoomCategory
    min_area: float          # NBC minimum carpet area (m²)
    target_area: float       # Typical comfortable area (m²)
    max_area: float          # Maximum reasonable area (m²)
    min_width: float         # NBC minimum clear width (m)
    min_height: float        # NBC minimum clear ceiling height (m)
    min_aspect_ratio: float  # Minimum width/height ratio (avoids slivers)
    max_aspect_ratio: float  # Maximum width/height ratio
    requires_external_wall: bool  # Must touch plot exterior (NBC daylight)
    requires_ventilation: bool    # Must have window
    color: str               # Hex color for visualization

    def validate_dimensions(self, width: float, height: float) -> bool:
        """Check if given dimensions meet NBC requirements."""
        area = width * height
        aspect = width / height if height > 0 else 0
        return (
            area >= self.min_area
            and min(width, height) >= self.min_width
            and self.min_aspect_ratio <= aspect <= self.max_aspect_ratio
        )


@dataclass
class WallThickness:
    """Standard Indian wall thicknesses in meters."""
    external: float = 0.230      # 230mm — 9" clay brick masonry
    internal: float = 0.115      # 115mm — 4.5" half-brick partition
    internal_wet: float = 0.150  # 150mm — 6" AAC block (bathrooms/kitchens)
    plaster: float = 0.015       # 15mm cement plaster per side


@dataclass
class DoorSpec:
    """Standard Indian door dimensions in meters."""
    width: float   # clear opening width
    height: float  # clear opening height
    frame_thickness: float = 0.050  # 50mm door frame

    @property
    def rough_opening_width(self) -> float:
        return self.width + 2 * self.frame_thickness

    @property
    def rough_opening_height(self) -> float:
        return self.height + self.frame_thickness


@dataclass
class WindowSpec:
    """Standard Indian window dimensions in meters."""
    width: float   # clear opening width
    height: float  # clear opening height
    sill_height: float  # height from floor to window sill


@dataclass
class RoomTemplate:
    """
    A room with its specification and target dimensions for a specific BHK config.
    """
    spec: RoomSpec
    target_width: float   # target width in meters
    target_height: float  # target height in meters
    door_spec: DoorSpec
    window_spec: Optional[WindowSpec]
    preferred_vastu_zone: str  # 'NE', 'SE', 'SW', 'NW', 'N', 'E', 'S', 'W', 'CENTER'
    attached_to: Optional[str] = None  # name of room this must be adjacent to
    is_optional: bool = False  # can be omitted on small plots


@dataclass
class BHKConfig:
    """
    Complete room schedule for a BHK configuration.
    """
    name: str                        # '1BHK', '2BHK', '3BHK', '4BHK'
    min_plot_area: float             # minimum plot area needed (m²)
    recommended_plot_area: float     # recommended plot area (m²)
    typical_carpet_area: float       # typical total carpet area (m²)
    rooms: List[RoomTemplate] = field(default_factory=list)
    description: str = ""


# =============================================================================
# NBC 2016 Room Specifications
# =============================================================================

# Standard door sizes
DOOR_MAIN_ENTRANCE = DoorSpec(width=1.000, height=2.100)
DOOR_BEDROOM = DoorSpec(width=0.900, height=2.100)
DOOR_BATHROOM = DoorSpec(width=0.750, height=2.000)
DOOR_KITCHEN = DoorSpec(width=0.800, height=2.000)
DOOR_INTERNAL = DoorSpec(width=0.800, height=2.100)

# Standard window sizes
WINDOW_LIVING = WindowSpec(width=1.500, height=1.200, sill_height=0.900)
WINDOW_BEDROOM = WindowSpec(width=1.200, height=1.200, sill_height=0.900)
WINDOW_KITCHEN = WindowSpec(width=0.900, height=0.600, sill_height=1.200)
WINDOW_BATHROOM = WindowSpec(width=0.600, height=0.450, sill_height=1.500)
WINDOW_DINING = WindowSpec(width=1.200, height=1.200, sill_height=0.900)

# Master NBC room specifications
NBC_ROOM_SPECS: Dict[str, RoomSpec] = {
    "living_room": RoomSpec(
        name="Living Room",
        category=RoomCategory.LIVING,
        min_area=9.5, target_area=16.0, max_area=35.0,
        min_width=2.4, min_height=2.75,
        min_aspect_ratio=0.55, max_aspect_ratio=1.80,
        requires_external_wall=True,
        requires_ventilation=True,
        color="#8DD3C7"
    ),
    "living_dining": RoomSpec(
        name="Living & Dining",
        category=RoomCategory.LIVING,
        min_area=14.0, target_area=22.0, max_area=45.0,
        min_width=3.0, min_height=2.75,
        min_aspect_ratio=0.50, max_aspect_ratio=2.00,
        requires_external_wall=True,
        requires_ventilation=True,
        color="#8DD3C7"
    ),
    "dining_room": RoomSpec(
        name="Dining Room",
        category=RoomCategory.DINING,
        min_area=7.5, target_area=11.0, max_area=20.0,
        min_width=2.4, min_height=2.75,
        min_aspect_ratio=0.55, max_aspect_ratio=1.80,
        requires_external_wall=False,
        requires_ventilation=True,
        color="#FFFFB3"
    ),
    "kitchen": RoomSpec(
        name="Kitchen",
        category=RoomCategory.KITCHEN,
        min_area=5.0, target_area=7.5, max_area=15.0,
        min_width=1.8, min_height=2.60,
        min_aspect_ratio=0.55, max_aspect_ratio=1.80,
        requires_external_wall=True,
        requires_ventilation=True,
        color="#FB8072"
    ),
    "kitchen_dining": RoomSpec(
        name="Kitchen cum Dining",
        category=RoomCategory.KITCHEN,
        min_area=9.5, target_area=12.0, max_area=20.0,
        min_width=2.4, min_height=2.60,
        min_aspect_ratio=0.55, max_aspect_ratio=1.80,
        requires_external_wall=True,
        requires_ventilation=True,
        color="#FB8072"
    ),
    "master_bedroom": RoomSpec(
        name="Master Bedroom",
        category=RoomCategory.BEDROOM,
        min_area=9.5, target_area=15.5, max_area=28.0,
        min_width=3.0, min_height=2.75,
        min_aspect_ratio=0.55, max_aspect_ratio=1.70,
        requires_external_wall=True,
        requires_ventilation=True,
        color="#BEBADA"
    ),
    "bedroom": RoomSpec(
        name="Bedroom",
        category=RoomCategory.BEDROOM,
        min_area=7.5, target_area=12.0, max_area=20.0,
        min_width=2.4, min_height=2.75,
        min_aspect_ratio=0.55, max_aspect_ratio=1.70,
        requires_external_wall=True,
        requires_ventilation=True,
        color="#BC80BD"
    ),
    "guest_bedroom": RoomSpec(
        name="Guest Bedroom",
        category=RoomCategory.BEDROOM,
        min_area=7.5, target_area=12.0, max_area=18.0,
        min_width=2.4, min_height=2.75,
        min_aspect_ratio=0.55, max_aspect_ratio=1.70,
        requires_external_wall=True,
        requires_ventilation=True,
        color="#FDB462"
    ),
    "bathroom": RoomSpec(
        name="Bathroom",
        category=RoomCategory.BATHROOM,
        min_area=1.8, target_area=3.0, max_area=6.0,
        min_width=1.2, min_height=2.10,
        min_aspect_ratio=0.50, max_aspect_ratio=2.00,
        requires_external_wall=False,
        requires_ventilation=True,
        color="#FCCDE5"
    ),
    "attached_bath": RoomSpec(
        name="Attached Bathroom",
        category=RoomCategory.BATHROOM,
        min_area=2.8, target_area=3.5, max_area=8.0,
        min_width=1.2, min_height=2.10,
        min_aspect_ratio=0.50, max_aspect_ratio=2.00,
        requires_external_wall=False,
        requires_ventilation=True,
        color="#FCCDE5"
    ),
    "pooja_room": RoomSpec(
        name="Pooja Room",
        category=RoomCategory.POOJA,
        min_area=1.5, target_area=2.7, max_area=5.0,
        min_width=1.2, min_height=2.75,
        min_aspect_ratio=0.50, max_aspect_ratio=2.00,
        requires_external_wall=False,
        requires_ventilation=False,
        color="#D9D9D9"
    ),
    "study_room": RoomSpec(
        name="Study / Home Office",
        category=RoomCategory.STUDY,
        min_area=6.0, target_area=9.0, max_area=15.0,
        min_width=2.1, min_height=2.75,
        min_aspect_ratio=0.55, max_aspect_ratio=1.70,
        requires_external_wall=True,
        requires_ventilation=True,
        color="#B3DE69"
    ),
    "corridor": RoomSpec(
        name="Corridor",
        category=RoomCategory.CORRIDOR,
        min_area=2.0, target_area=4.0, max_area=12.0,
        min_width=0.90, min_height=2.10,
        min_aspect_ratio=0.15, max_aspect_ratio=6.00,
        requires_external_wall=False,
        requires_ventilation=False,
        color="#E8E8E8"
    ),
    "utility_room": RoomSpec(
        name="Utility / Balcony",
        category=RoomCategory.UTILITY,
        min_area=2.0, target_area=3.5, max_area=8.0,
        min_width=1.2, min_height=2.10,
        min_aspect_ratio=0.30, max_aspect_ratio=3.00,
        requires_external_wall=True,
        requires_ventilation=True,
        color="#CCEBC5"
    ),
    "servant_room": RoomSpec(
        name="Servant Quarter",
        category=RoomCategory.SERVANT,
        min_area=5.0, target_area=6.0, max_area=10.0,
        min_width=2.1, min_height=2.60,
        min_aspect_ratio=0.55, max_aspect_ratio=1.70,
        requires_external_wall=True,
        requires_ventilation=True,
        color="#FFD8B1"
    ),
    "foyer": RoomSpec(
        name="Foyer / Entrance",
        category=RoomCategory.FOYER,
        min_area=2.0, target_area=4.0, max_area=8.0,
        min_width=1.5, min_height=2.75,
        min_aspect_ratio=0.40, max_aspect_ratio=2.50,
        requires_external_wall=True,
        requires_ventilation=False,
        color="#F0F0F0"
    ),
}


# =============================================================================
# Standard Indian BHK Configurations
# =============================================================================

def _build_1bhk() -> BHKConfig:
    """1 BHK: Ideal for 1-2 persons. 37-52 m² carpet area."""
    return BHKConfig(
        name="1BHK",
        min_plot_area=45.0,
        recommended_plot_area=60.0,
        typical_carpet_area=42.0,
        description="1 Bedroom, Hall, Kitchen — ideal for singles or couples",
        rooms=[
            RoomTemplate(
                spec=NBC_ROOM_SPECS["living_dining"],
                target_width=3.35, target_height=4.25,
                door_spec=DOOR_MAIN_ENTRANCE,
                window_spec=WINDOW_LIVING,
                preferred_vastu_zone="N",
            ),
            RoomTemplate(
                spec=NBC_ROOM_SPECS["kitchen"],
                target_width=2.10, target_height=2.75,
                door_spec=DOOR_KITCHEN,
                window_spec=WINDOW_KITCHEN,
                preferred_vastu_zone="SE",
            ),
            RoomTemplate(
                spec=NBC_ROOM_SPECS["master_bedroom"],
                target_width=3.05, target_height=3.65,
                door_spec=DOOR_BEDROOM,
                window_spec=WINDOW_BEDROOM,
                preferred_vastu_zone="SW",
            ),
            RoomTemplate(
                spec=NBC_ROOM_SPECS["attached_bath"],
                target_width=1.35, target_height=2.10,
                door_spec=DOOR_BATHROOM,
                window_spec=WINDOW_BATHROOM,
                preferred_vastu_zone="W",
                attached_to="Master Bedroom",
            ),
            RoomTemplate(
                spec=NBC_ROOM_SPECS["utility_room"],
                target_width=1.20, target_height=2.40,
                door_spec=DOOR_INTERNAL,
                window_spec=None,
                preferred_vastu_zone="NW",
                is_optional=True,
            ),
        ],
    )


def _build_2bhk() -> BHKConfig:
    """2 BHK: Ideal for small families (3-4 persons). 60-85 m² carpet area."""
    return BHKConfig(
        name="2BHK",
        min_plot_area=70.0,
        recommended_plot_area=95.0,
        typical_carpet_area=72.0,
        description="2 Bedrooms, Hall, Kitchen — ideal for small families",
        rooms=[
            RoomTemplate(
                spec=NBC_ROOM_SPECS["living_dining"],
                target_width=3.65, target_height=5.50,
                door_spec=DOOR_MAIN_ENTRANCE,
                window_spec=WINDOW_LIVING,
                preferred_vastu_zone="N",
            ),
            RoomTemplate(
                spec=NBC_ROOM_SPECS["kitchen"],
                target_width=2.40, target_height=3.05,
                door_spec=DOOR_KITCHEN,
                window_spec=WINDOW_KITCHEN,
                preferred_vastu_zone="SE",
            ),
            RoomTemplate(
                spec=NBC_ROOM_SPECS["master_bedroom"],
                target_width=3.65, target_height=4.25,
                door_spec=DOOR_BEDROOM,
                window_spec=WINDOW_BEDROOM,
                preferred_vastu_zone="SW",
            ),
            RoomTemplate(
                spec=NBC_ROOM_SPECS["attached_bath"],
                target_width=1.35, target_height=2.25,
                door_spec=DOOR_BATHROOM,
                window_spec=WINDOW_BATHROOM,
                preferred_vastu_zone="SW",
                attached_to="Master Bedroom",
            ),
            RoomTemplate(
                spec=NBC_ROOM_SPECS["bedroom"],
                target_width=3.05, target_height=3.65,
                door_spec=DOOR_BEDROOM,
                window_spec=WINDOW_BEDROOM,
                preferred_vastu_zone="NW",
            ),
            RoomTemplate(
                spec=NBC_ROOM_SPECS["bathroom"],
                target_width=1.35, target_height=2.10,
                door_spec=DOOR_BATHROOM,
                window_spec=WINDOW_BATHROOM,
                preferred_vastu_zone="W",
            ),
            RoomTemplate(
                spec=NBC_ROOM_SPECS["utility_room"],
                target_width=1.20, target_height=3.00,
                door_spec=DOOR_INTERNAL,
                window_spec=None,
                preferred_vastu_zone="NW",
                is_optional=True,
            ),
        ],
    )


def _build_3bhk() -> BHKConfig:
    """3 BHK: Ideal for medium families (4-6 persons). 95-140 m² carpet area."""
    return BHKConfig(
        name="3BHK",
        min_plot_area=110.0,
        recommended_plot_area=150.0,
        typical_carpet_area=120.0,
        description="3 Bedrooms, Hall, Kitchen, Dining — ideal for medium families",
        rooms=[
            RoomTemplate(
                spec=NBC_ROOM_SPECS["living_room"],
                target_width=4.25, target_height=5.80,
                door_spec=DOOR_MAIN_ENTRANCE,
                window_spec=WINDOW_LIVING,
                preferred_vastu_zone="N",
            ),
            RoomTemplate(
                spec=NBC_ROOM_SPECS["dining_room"],
                target_width=3.05, target_height=3.65,
                door_spec=DOOR_INTERNAL,
                window_spec=WINDOW_DINING,
                preferred_vastu_zone="E",
            ),
            RoomTemplate(
                spec=NBC_ROOM_SPECS["kitchen"],
                target_width=2.75, target_height=3.65,
                door_spec=DOOR_KITCHEN,
                window_spec=WINDOW_KITCHEN,
                preferred_vastu_zone="SE",
            ),
            RoomTemplate(
                spec=NBC_ROOM_SPECS["master_bedroom"],
                target_width=4.25, target_height=4.85,
                door_spec=DOOR_BEDROOM,
                window_spec=WINDOW_BEDROOM,
                preferred_vastu_zone="SW",
            ),
            RoomTemplate(
                spec=NBC_ROOM_SPECS["attached_bath"],
                target_width=1.80, target_height=2.40,
                door_spec=DOOR_BATHROOM,
                window_spec=WINDOW_BATHROOM,
                preferred_vastu_zone="SW",
                attached_to="Master Bedroom",
            ),
            RoomTemplate(
                spec=NBC_ROOM_SPECS["bedroom"],
                target_width=3.65, target_height=3.95,
                door_spec=DOOR_BEDROOM,
                window_spec=WINDOW_BEDROOM,
                preferred_vastu_zone="NW",
            ),
            RoomTemplate(
                spec=NBC_ROOM_SPECS["bathroom"],
                target_width=1.35, target_height=2.25,
                door_spec=DOOR_BATHROOM,
                window_spec=WINDOW_BATHROOM,
                preferred_vastu_zone="W",
                attached_to="Bedroom 2",
            ),
            RoomTemplate(
                spec=NBC_ROOM_SPECS["guest_bedroom"],
                target_width=3.35, target_height=3.65,
                door_spec=DOOR_BEDROOM,
                window_spec=WINDOW_BEDROOM,
                preferred_vastu_zone="W",
            ),
            RoomTemplate(
                spec=NBC_ROOM_SPECS["bathroom"],
                target_width=1.35, target_height=2.10,
                door_spec=DOOR_BATHROOM,
                window_spec=WINDOW_BATHROOM,
                preferred_vastu_zone="NW",
            ),
            RoomTemplate(
                spec=NBC_ROOM_SPECS["pooja_room"],
                target_width=1.50, target_height=1.80,
                door_spec=DOOR_INTERNAL,
                window_spec=None,
                preferred_vastu_zone="NE",
            ),
        ],
    )


def _build_4bhk() -> BHKConfig:
    """4 BHK: Ideal for large/joint families (6+ persons). 150-250+ m² carpet area."""
    return BHKConfig(
        name="4BHK",
        min_plot_area=170.0,
        recommended_plot_area=220.0,
        typical_carpet_area=190.0,
        description="4 Bedrooms, Living Hall, Dining, Kitchen — ideal for large families",
        rooms=[
            RoomTemplate(
                spec=NBC_ROOM_SPECS["living_room"],
                target_width=4.85, target_height=6.70,
                door_spec=DOOR_MAIN_ENTRANCE,
                window_spec=WINDOW_LIVING,
                preferred_vastu_zone="N",
            ),
            RoomTemplate(
                spec=NBC_ROOM_SPECS["dining_room"],
                target_width=3.65, target_height=4.25,
                door_spec=DOOR_INTERNAL,
                window_spec=WINDOW_DINING,
                preferred_vastu_zone="E",
            ),
            RoomTemplate(
                spec=NBC_ROOM_SPECS["kitchen"],
                target_width=3.05, target_height=4.25,
                door_spec=DOOR_KITCHEN,
                window_spec=WINDOW_KITCHEN,
                preferred_vastu_zone="SE",
            ),
            RoomTemplate(
                spec=NBC_ROOM_SPECS["master_bedroom"],
                target_width=4.55, target_height=5.50,
                door_spec=DOOR_BEDROOM,
                window_spec=WINDOW_BEDROOM,
                preferred_vastu_zone="SW",
            ),
            RoomTemplate(
                spec=NBC_ROOM_SPECS["attached_bath"],
                target_width=2.40, target_height=3.05,
                door_spec=DOOR_BATHROOM,
                window_spec=WINDOW_BATHROOM,
                preferred_vastu_zone="SW",
                attached_to="Master Bedroom",
            ),
            RoomTemplate(
                spec=NBC_ROOM_SPECS["bedroom"],
                target_width=3.95, target_height=4.55,
                door_spec=DOOR_BEDROOM,
                window_spec=WINDOW_BEDROOM,
                preferred_vastu_zone="W",
            ),
            RoomTemplate(
                spec=NBC_ROOM_SPECS["bedroom"],
                target_width=3.65, target_height=4.25,
                door_spec=DOOR_BEDROOM,
                window_spec=WINDOW_BEDROOM,
                preferred_vastu_zone="NW",
            ),
            RoomTemplate(
                spec=NBC_ROOM_SPECS["guest_bedroom"],
                target_width=3.35, target_height=3.65,
                door_spec=DOOR_BEDROOM,
                window_spec=WINDOW_BEDROOM,
                preferred_vastu_zone="N",
            ),
            RoomTemplate(
                spec=NBC_ROOM_SPECS["attached_bath"],
                target_width=1.50, target_height=2.40,
                door_spec=DOOR_BATHROOM,
                window_spec=WINDOW_BATHROOM,
                preferred_vastu_zone="W",
                attached_to="Bedroom 2",
            ),
            RoomTemplate(
                spec=NBC_ROOM_SPECS["bathroom"],
                target_width=1.50, target_height=2.40,
                door_spec=DOOR_BATHROOM,
                window_spec=WINDOW_BATHROOM,
                preferred_vastu_zone="NW",
                attached_to="Bedroom 3",
            ),
            RoomTemplate(
                spec=NBC_ROOM_SPECS["bathroom"],
                target_width=1.35, target_height=2.10,
                door_spec=DOOR_BATHROOM,
                window_spec=WINDOW_BATHROOM,
                preferred_vastu_zone="NW",
            ),
            RoomTemplate(
                spec=NBC_ROOM_SPECS["pooja_room"],
                target_width=1.80, target_height=2.10,
                door_spec=DOOR_INTERNAL,
                window_spec=None,
                preferred_vastu_zone="NE",
            ),
            RoomTemplate(
                spec=NBC_ROOM_SPECS["servant_room"],
                target_width=2.10, target_height=2.75,
                door_spec=DOOR_INTERNAL,
                window_spec=WINDOW_BATHROOM,
                preferred_vastu_zone="NW",
                is_optional=True,
            ),
        ],
    )


# Pre-built configurations lookup
_BHK_CONFIGS: Dict[str, BHKConfig] = {
    "1BHK": _build_1bhk(),
    "2BHK": _build_2bhk(),
    "3BHK": _build_3bhk(),
    "4BHK": _build_4bhk(),
}


def get_bhk_config(bhk: str) -> BHKConfig:
    """
    Get a standard BHK configuration by name.

    Args:
        bhk: One of '1BHK', '2BHK', '3BHK', '4BHK'

    Returns:
        BHKConfig with all room templates for the configuration.

    Raises:
        ValueError: If bhk is not a recognized configuration.
    """
    key = bhk.upper().replace(" ", "")
    if key not in _BHK_CONFIGS:
        raise ValueError(
            f"Unknown BHK configuration: '{bhk}'. "
            f"Available: {list(_BHK_CONFIGS.keys())}"
        )
    return _BHK_CONFIGS[key]


def get_all_configs() -> Dict[str, BHKConfig]:
    """Return all available BHK configurations."""
    return dict(_BHK_CONFIGS)


def recommend_bhk_for_plot(plot_area: float) -> str:
    """
    Recommend the best BHK configuration for a given plot area.

    Args:
        plot_area: Total plot area in square meters.

    Returns:
        BHK configuration name (e.g., '2BHK').
    """
    if plot_area < 55:
        return "1BHK"
    elif plot_area < 100:
        return "2BHK"
    elif plot_area < 160:
        return "3BHK"
    else:
        return "4BHK"
