# Floor Plan Detection Package
# Advanced floor plan analysis with accurate wall and room detection

__version__ = "1.1.0"

# Import main classes for easy access
from .smart_detection import SmartFloorPlanDetector, SmartWall, DetectedSpace, SpaceType
from .preprocessing import FloorPlanPreprocessor
from .wall_detection import WallDetector, WallSegment
from .room_detection import RoomDetector, Room, RoomType
from .floor_plan_analyzer import FloorPlanAnalyzer, RoomLabel, ROOM_NAMES

# Conditionally import 3D model generator (requires Open3D)
try:
    from .model_3d import FloorPlan3DGenerator, ModelConfig
    HAS_3D = True
except ImportError:
    HAS_3D = False
    FloorPlan3DGenerator = None
    ModelConfig = None

__all__ = [
    'SmartFloorPlanDetector',
    'SmartWall', 
    'DetectedSpace',
    'SpaceType',
    'FloorPlanPreprocessor',
    'WallDetector',
    'WallSegment',
    'RoomDetector', 
    'Room',
    'RoomType',
    'FloorPlanAnalyzer',
    'RoomLabel',
    'ROOM_NAMES',
    'FloorPlan3DGenerator',
    'ModelConfig',
    'HAS_3D'
]
