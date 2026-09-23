"""
Floor Plan Generator Package
=============================
AI-powered residential floor plan generation engine with Vastu Shastra compliance
and NBC 2016 Indian Building Code standards.

Core Components:
- room_specs: Indian residential architecture standards database
- vastu_engine: Vastu Shastra compass-based room placement engine
- layout_engine: V-HSP (Vastu-Guided Hierarchical Space Partitioning) algorithm
- family_analyzer: Family intent → room schedule converter
"""

from floorplan_generator.room_specs import (
    RoomSpec,
    WallThickness,
    DoorSpec,
    WindowSpec,
    BHKConfig,
    get_bhk_config,
    NBC_ROOM_SPECS,
)
from floorplan_generator.vastu_engine import VastuEngine, VastuZone
from floorplan_generator.layout_engine import (
    LayoutEngine,
    GeneratedFloorPlan,
    RoomRect,
    WallSegment,
    DoorPlacement,
    WindowPlacement,
    DimLine,
)
from floorplan_generator.family_analyzer import FamilyAnalyzer, FamilyProfile
from floorplan_generator.land_units import LandUnitsEngine, LandParcel

__version__ = "1.0.0"
__all__ = [
    "RoomSpec", "WallThickness", "DoorSpec", "WindowSpec", "BHKConfig",
    "get_bhk_config", "NBC_ROOM_SPECS",
    "VastuEngine", "VastuZone",
    "LayoutEngine", "GeneratedFloorPlan", "RoomRect", "WallSegment",
    "DoorPlacement", "WindowPlacement", "DimLine",
    "FamilyAnalyzer", "FamilyProfile",
    "LandUnitsEngine", "LandParcel",
]
