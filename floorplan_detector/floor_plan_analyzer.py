"""
Advanced Floor Plan Analyzer
=============================
Uses multiple techniques for accurate floor plan analysis:

1. Wall Detection - Using line detection with door gap identification
2. Door Detection - Finding gaps in walls that represent doors
3. Room Segmentation - Using contour analysis and semantic features
4. Room Classification - Using visual features, size, and context

This is designed to work like a human would - looking at the overall
structure and using context to determine room types.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

class RoomLabel(Enum):
    """Room type labels based on CubiCasa5K categories."""
    BACKGROUND = 0
    OUTDOOR = 1
    WALL = 2
    KITCHEN = 3
    LIVING_DINING = 4
    BEDROOM = 5
    BATHROOM = 6
    HALLWAY = 7
    RAILING = 8
    STORAGE = 9
    GARAGE = 10
    OTHER = 11
    STAIRS = 12
    DOOR = 13
    WINDOW = 14


# Human-readable names
ROOM_NAMES = {
    RoomLabel.BACKGROUND: "Background",
    RoomLabel.OUTDOOR: "Outdoor",
    RoomLabel.WALL: "Wall",
    RoomLabel.KITCHEN: "Kitchen",
    RoomLabel.LIVING_DINING: "Living/Dining",
    RoomLabel.BEDROOM: "Bedroom",
    RoomLabel.BATHROOM: "Bathroom",
    RoomLabel.HALLWAY: "Hallway",
    RoomLabel.RAILING: "Railing",
    RoomLabel.STORAGE: "Storage",
    RoomLabel.GARAGE: "Garage",
    RoomLabel.OTHER: "Room",
    RoomLabel.STAIRS: "Stairs",
    RoomLabel.DOOR: "Door",
    RoomLabel.WINDOW: "Window"
}

# Colors for visualization (BGR)
ROOM_COLORS = {
    RoomLabel.BACKGROUND: (200, 200, 200),
    RoomLabel.OUTDOOR: (180, 230, 180),
    RoomLabel.WALL: (50, 50, 50),
    RoomLabel.KITCHEN: (100, 180, 255),  # Orange
    RoomLabel.LIVING_DINING: (180, 255, 180),  # Light green
    RoomLabel.BEDROOM: (255, 200, 150),  # Light blue
    RoomLabel.BATHROOM: (255, 255, 150),  # Cyan
    RoomLabel.HALLWAY: (200, 200, 255),  # Light pink
    RoomLabel.RAILING: (100, 100, 100),
    RoomLabel.STORAGE: (180, 180, 220),
    RoomLabel.GARAGE: (150, 150, 150),
    RoomLabel.OTHER: (220, 220, 220),
    RoomLabel.STAIRS: (150, 200, 255),
    RoomLabel.DOOR: (0, 255, 0),  # Green
    RoomLabel.WINDOW: (255, 0, 0)  # Blue
}


@dataclass
class Wall:
    """Represents a wall segment."""
    x1: int
    y1: int
    x2: int
    y2: int
    thickness: int = 3
    has_door: bool = False
    door_position: Optional[Tuple[int, int]] = None
    
    @property
    def length(self) -> float:
        return np.sqrt((self.x2 - self.x1)**2 + (self.y2 - self.y1)**2)
    
    @property
    def is_horizontal(self) -> bool:
        angle = abs(np.degrees(np.arctan2(self.y2 - self.y1, self.x2 - self.x1)))
        return angle < 20 or angle > 160
    
    @property
    def is_vertical(self) -> bool:
        angle = abs(np.degrees(np.arctan2(self.y2 - self.y1, self.x2 - self.x1)))
        return 70 < angle < 110


@dataclass
class Door:
    """Represents a door opening."""
    x: int
    y: int
    width: int
    height: int
    orientation: str = "unknown"  # "horizontal" or "vertical"
    connected_rooms: List[int] = field(default_factory=list)
    
    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)


@dataclass
class Room:
    """Represents a detected room with semantic label."""
    id: int
    contour: np.ndarray
    label: RoomLabel
    area: float
    center: Tuple[int, int]
    bounding_box: Tuple[int, int, int, int]  # x, y, w, h
    confidence: float = 1.0
    connected_doors: List[int] = field(default_factory=list)
    features: Dict = field(default_factory=dict)
    
    @property
    def name(self) -> str:
        return ROOM_NAMES.get(self.label, "Unknown")
    
    @property
    def aspect_ratio(self) -> float:
        x, y, w, h = self.bounding_box
        return min(w, h) / max(w, h) if max(w, h) > 0 else 1.0


# =============================================================================
# WALL AND DOOR DETECTOR
# =============================================================================

class WallDoorDetector:
    """
    Detects walls and doors in floor plans.
    Doors are identified as gaps in walls.
    """
    
    def __init__(
        self,
        min_wall_length: int = 50,
        wall_thickness_range: Tuple[int, int] = (2, 20),
        door_width_range: Tuple[int, int] = (15, 80),
    ):
        self.min_wall_length = min_wall_length
        self.wall_thickness_range = wall_thickness_range
        self.door_width_range = door_width_range
    
    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image to extract wall structure."""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Denoise
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=15, C=5
        )
        
        # Clean up
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return binary
    
    def detect_walls(self, binary: np.ndarray) -> List[Wall]:
        """Detect wall segments using Hough transform."""
        # Edge detection
        edges = cv2.Canny(binary, 50, 150, apertureSize=3)
        
        # Hough lines
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi/180,
            threshold=60,
            minLineLength=self.min_wall_length,
            maxLineGap=20
        )
        
        if lines is None:
            return []
        
        walls = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            wall = Wall(x1, y1, x2, y2)
            
            # Only keep horizontal and vertical walls
            if wall.is_horizontal or wall.is_vertical:
                if wall.length >= self.min_wall_length:
                    walls.append(wall)
        
        # Merge nearby parallel walls
        walls = self._merge_walls(walls)
        
        return walls
    
    def _merge_walls(self, walls: List[Wall]) -> List[Wall]:
        """Merge nearby parallel wall segments."""
        if not walls:
            return []
        
        # Separate by orientation
        h_walls = [w for w in walls if w.is_horizontal]
        v_walls = [w for w in walls if w.is_vertical]
        
        merged = []
        merged.extend(self._merge_parallel(h_walls, horizontal=True))
        merged.extend(self._merge_parallel(v_walls, horizontal=False))
        
        return merged
    
    def _merge_parallel(self, walls: List[Wall], horizontal: bool) -> List[Wall]:
        """Merge parallel walls that are close together."""
        if not walls:
            return []
        
        # Sort by position
        if horizontal:
            walls = sorted(walls, key=lambda w: (min(w.y1, w.y2), min(w.x1, w.x2)))
        else:
            walls = sorted(walls, key=lambda w: (min(w.x1, w.x2), min(w.y1, w.y2)))
        
        merged = []
        used = set()
        merge_dist = 25  # pixels
        
        for i, w1 in enumerate(walls):
            if i in used:
                continue
            
            group = [w1]
            used.add(i)
            
            for j, w2 in enumerate(walls):
                if j in used:
                    continue
                
                # Check if on same line
                if horizontal:
                    pos_diff = abs((w1.y1 + w1.y2) / 2 - (w2.y1 + w2.y2) / 2)
                else:
                    pos_diff = abs((w1.x1 + w1.x2) / 2 - (w2.x1 + w2.x2) / 2)
                
                if pos_diff < merge_dist:
                    # Check for overlap or proximity
                    if self._walls_overlap_or_close(w1, w2, horizontal):
                        group.append(w2)
                        used.add(j)
            
            # Combine group into single wall
            if len(group) == 1:
                merged.append(w1)
            else:
                merged.append(self._combine_wall_group(group, horizontal))
        
        return merged
    
    def _walls_overlap_or_close(self, w1: Wall, w2: Wall, horizontal: bool) -> bool:
        """Check if walls overlap or are close enough to merge."""
        gap_threshold = 40  # Maximum gap to still merge
        
        if horizontal:
            x1_range = (min(w1.x1, w1.x2), max(w1.x1, w1.x2))
            x2_range = (min(w2.x1, w2.x2), max(w2.x1, w2.x2))
            gap = max(0, max(x1_range[0], x2_range[0]) - min(x1_range[1], x2_range[1]))
        else:
            y1_range = (min(w1.y1, w1.y2), max(w1.y1, w1.y2))
            y2_range = (min(w2.y1, w2.y2), max(w2.y1, w2.y2))
            gap = max(0, max(y1_range[0], y2_range[0]) - min(y1_range[1], y2_range[1]))
        
        return gap < gap_threshold
    
    def _combine_wall_group(self, walls: List[Wall], horizontal: bool) -> Wall:
        """Combine multiple walls into one extended wall."""
        if horizontal:
            y_avg = int(np.mean([w.y1 for w in walls] + [w.y2 for w in walls]))
            x_min = min(min(w.x1, w.x2) for w in walls)
            x_max = max(max(w.x1, w.x2) for w in walls)
            return Wall(x_min, y_avg, x_max, y_avg)
        else:
            x_avg = int(np.mean([w.x1 for w in walls] + [w.x2 for w in walls]))
            y_min = min(min(w.y1, w.y2) for w in walls)
            y_max = max(max(w.y1, w.y2) for w in walls)
            return Wall(x_avg, y_min, x_avg, y_max)
    
    def detect_doors(self, binary: np.ndarray, walls: List[Wall]) -> List[Door]:
        """
        Detect doors as gaps in walls.
        A door is a gap of appropriate width where walls would otherwise connect.
        """
        doors = []
        door_id = 0
        
        # Create wall mask
        wall_mask = np.zeros(binary.shape, dtype=np.uint8)
        for wall in walls:
            cv2.line(wall_mask, (wall.x1, wall.y1), (wall.x2, wall.y2), 255, 5)
        
        # Find potential door locations by looking for gaps
        # Check horizontal walls for vertical doors
        h_walls = sorted([w for w in walls if w.is_horizontal], 
                        key=lambda w: (w.y1 + w.y2) / 2)
        
        # Check vertical walls for horizontal doors
        v_walls = sorted([w for w in walls if w.is_vertical],
                        key=lambda w: (w.x1 + w.x2) / 2)
        
        # Find gaps between wall endpoints that could be doors
        doors.extend(self._find_gaps_as_doors(h_walls, horizontal=True))
        doors.extend(self._find_gaps_as_doors(v_walls, horizontal=False))
        
        # Also detect door symbols (arcs) in the image
        doors.extend(self._detect_door_arcs(binary))
        
        logger.info(f"Detected {len(doors)} doors")
        return doors
    
    def _find_gaps_as_doors(self, walls: List[Wall], horizontal: bool) -> List[Door]:
        """Find gaps between wall endpoints that could be doors."""
        doors = []
        min_door, max_door = self.door_width_range
        
        # Group walls by their position (same line)
        position_groups = defaultdict(list)
        for wall in walls:
            if horizontal:
                pos = (wall.y1 + wall.y2) // 2
            else:
                pos = (wall.x1 + wall.x2) // 2
            position_groups[pos // 20 * 20].append(wall)  # Group within 20px
        
        for pos, group in position_groups.items():
            if len(group) < 2:
                continue
            
            # Sort by start position
            if horizontal:
                group = sorted(group, key=lambda w: min(w.x1, w.x2))
            else:
                group = sorted(group, key=lambda w: min(w.y1, w.y2))
            
            # Check gaps between consecutive walls
            for i in range(len(group) - 1):
                w1, w2 = group[i], group[i + 1]
                
                if horizontal:
                    end1 = max(w1.x1, w1.x2)
                    start2 = min(w2.x1, w2.x2)
                    gap = start2 - end1
                    
                    if min_door < gap < max_door:
                        door = Door(
                            x=end1, y=pos - 5,
                            width=gap, height=10,
                            orientation="vertical"
                        )
                        doors.append(door)
                else:
                    end1 = max(w1.y1, w1.y2)
                    start2 = min(w2.y1, w2.y2)
                    gap = start2 - end1
                    
                    if min_door < gap < max_door:
                        door = Door(
                            x=pos - 5, y=end1,
                            width=10, height=gap,
                            orientation="horizontal"
                        )
                        doors.append(door)
        
        return doors
    
    def _detect_door_arcs(self, binary: np.ndarray) -> List[Door]:
        """Detect door swing arcs (quarter circles) in the image."""
        doors = []
        
        # Use Hough circles to find arcs
        # Door arcs are typically quarter circles
        blurred = cv2.GaussianBlur(binary, (5, 5), 0)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, 1, 50,
            param1=50, param2=30,
            minRadius=20, maxRadius=60
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)
            for (x, y, r) in circles:
                # Check if this looks like a door arc
                # Door arcs are usually partial circles near walls
                door = Door(
                    x=x - r, y=y - r,
                    width=r * 2, height=r * 2,
                    orientation="arc"
                )
                doors.append(door)
        
        return doors


# =============================================================================
# ROOM SEGMENTATION AND CLASSIFICATION
# =============================================================================

class RoomAnalyzer:
    """
    Analyzes and classifies rooms in floor plans.
    Uses multiple features to determine room types:
    - Size and shape
    - Position in floor plan
    - Connected fixtures (if visible)
    - Relationship to other rooms
    """
    
    # Room size thresholds (in pixels, adjust based on scale)
    SIZE_THRESHOLDS = {
        'tiny': 3000,      # < 3000 = closet/utility
        'small': 8000,     # 3000-8000 = bathroom
        'medium': 20000,   # 8000-20000 = bedroom/kitchen
        'large': 40000,    # 20000-40000 = living room
        'huge': 80000      # > 40000 = combined spaces
    }
    
    def __init__(self, min_room_area: int = 3000):
        self.min_room_area = min_room_area
    
    def segment_rooms(self, binary: np.ndarray, walls: List[Wall]) -> List[Room]:
        """Segment floor plan into rooms using walls as boundaries."""
        
        # Create wall mask (thicker for better separation)
        wall_mask = np.zeros(binary.shape, dtype=np.uint8)
        for wall in walls:
            cv2.line(wall_mask, (wall.x1, wall.y1), (wall.x2, wall.y2), 255, 7)
        
        # Close gaps in walls
        kernel = np.ones((9, 9), np.uint8)
        wall_mask = cv2.dilate(wall_mask, kernel, iterations=2)
        
        # Find rooms as connected components in the inverted wall mask
        room_mask = cv2.bitwise_not(wall_mask)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            room_mask, connectivity=4
        )
        
        rooms = []
        image_area = binary.shape[0] * binary.shape[1]
        
        for i in range(1, num_labels):  # Skip background
            area = stats[i, cv2.CC_STAT_AREA]
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            # Filter by area
            if area < self.min_room_area:
                continue
            
            # Skip outer boundary
            if area > 0.5 * image_area:
                continue
            
            # Skip if dimensions match image (boundary)
            if w > binary.shape[1] * 0.85 and h > binary.shape[0] * 0.85:
                continue
            
            # Get contour
            component_mask = (labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                continue
            
            contour = contours[0]
            center = (int(centroids[i][0]), int(centroids[i][1]))
            
            room = Room(
                id=len(rooms),
                contour=contour,
                label=RoomLabel.OTHER,  # Will be classified later
                area=area,
                center=center,
                bounding_box=(x, y, w, h)
            )
            
            # Extract features for classification
            room.features = self._extract_features(room, binary, component_mask)
            
            rooms.append(room)
        
        return rooms
    
    def _extract_features(self, room: Room, binary: np.ndarray, 
                         room_mask: np.ndarray) -> Dict:
        """Extract features from room for classification."""
        x, y, w, h = room.bounding_box
        
        features = {
            'area': room.area,
            'aspect_ratio': room.aspect_ratio,
            'width': w,
            'height': h,
            'perimeter': cv2.arcLength(room.contour, True),
            'solidity': self._calc_solidity(room.contour),
            'position_x': room.center[0] / binary.shape[1],  # Normalized position
            'position_y': room.center[1] / binary.shape[0],
            'is_corner': self._is_corner_room(room, binary.shape),
            'has_fixtures': False,  # Would need fixture detection
            'num_vertices': len(cv2.approxPolyDP(
                room.contour, 0.02 * cv2.arcLength(room.contour, True), True
            ))
        }
        
        # Check for internal patterns (stairs, fixtures)
        roi = binary[y:y+h, x:x+w]
        if roi.size > 0:
            features['internal_density'] = np.sum(roi > 0) / roi.size
            features['has_parallel_lines'] = self._has_parallel_lines(roi)
        
        return features
    
    def _calc_solidity(self, contour: np.ndarray) -> float:
        """Calculate solidity (area / convex hull area)."""
        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        return area / hull_area if hull_area > 0 else 0
    
    def _is_corner_room(self, room: Room, img_shape: Tuple[int, int]) -> bool:
        """Check if room is in a corner of the floor plan."""
        h, w = img_shape
        cx, cy = room.center
        margin = 0.2
        
        in_corner = (
            (cx < w * margin or cx > w * (1 - margin)) and
            (cy < h * margin or cy > h * (1 - margin))
        )
        return in_corner
    
    def _has_parallel_lines(self, roi: np.ndarray) -> bool:
        """Check if ROI has parallel lines (stairs pattern)."""
        if roi.size == 0:
            return False
        
        edges = cv2.Canny(roi, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 20, 
                               minLineLength=roi.shape[1]//3, maxLineGap=5)
        
        if lines is None:
            return False
        
        # Count horizontal lines
        h_count = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if angle < 20 or angle > 160:
                h_count += 1
        
        return h_count >= 4  # Stairs typically have 4+ treads
    
    def classify_rooms(self, rooms: List[Room], doors: List[Door]) -> List[Room]:
        """
        Classify rooms based on features and context.
        Uses a rule-based approach inspired by how humans identify rooms.
        """
        if not rooms:
            return rooms
            
        # Sort rooms by area (largest first)
        rooms = sorted(rooms, key=lambda r: r.area, reverse=True)
        
        # Calculate statistics for relative classification
        total_area = sum(r.area for r in rooms)
        avg_area = total_area / len(rooms)
        largest_area = rooms[0].area if rooms else 0
        smallest_area = rooms[-1].area if rooms else 0
        
        # Use relative thresholds based on average room size
        REL_THRESHOLDS = {
            'tiny': avg_area * 0.3,
            'small': avg_area * 0.6,
            'medium': avg_area * 1.0,
            'large': avg_area * 1.2  # Lowered from 1.5
        }
        
        # Track assignments to ensure realistic room distribution
        assigned = {'kitchen': 0, 'bathroom': 0, 'bedroom': 0, 'living': 0}
        
        # PASS 1: The LARGEST room is typically Living Room (if squarish enough)
        if rooms:
            largest = rooms[0]
            # Check if significantly larger than average (at least 1.15x)
            if largest.area > avg_area * 1.15 and largest.aspect_ratio > 0.45:
                largest.label = RoomLabel.LIVING_DINING
                assigned['living'] += 1
        
        # PASS 2: Identify bathrooms (small, roughly square)
        for room in rooms:
            if room.label != RoomLabel.OTHER:
                continue
            if room.area < REL_THRESHOLDS['small']:
                if room.aspect_ratio > 0.5 and assigned['bathroom'] < 2:
                    room.label = RoomLabel.BATHROOM
                    assigned['bathroom'] += 1
        
        # PASS 3: Identify kitchen (second or third largest with good aspect ratio)
        for room in rooms[1:4]:  # Check 2nd to 4th largest
            if room.label != RoomLabel.OTHER:
                continue
            if room.area >= REL_THRESHOLDS['small'] and assigned['kitchen'] == 0:
                if room.aspect_ratio > 0.55:  # Kitchen should be fairly square
                    room.label = RoomLabel.KITCHEN
                    assigned['kitchen'] += 1
                    break
        
        # PASS 4: Remaining medium/large rooms are bedrooms
        for room in rooms:
            if room.label != RoomLabel.OTHER:
                continue
            if room.area >= REL_THRESHOLDS['tiny']:
                if room.aspect_ratio > 0.35:
                    room.label = RoomLabel.BEDROOM
                    assigned['bedroom'] += 1
        
        # PASS 5: Check for stairs (parallel lines pattern)
        for room in rooms:
            features = room.features
            if features.get('has_parallel_lines', False) and room.aspect_ratio < 0.4:
                room.label = RoomLabel.STAIRS
        
        # PASS 6: Hallways (very elongated)
        for room in rooms:
            if room.label == RoomLabel.OTHER:
                if room.aspect_ratio < 0.3:
                    room.label = RoomLabel.HALLWAY
        
        # PASS 7: Small remaining spaces
        for room in rooms:
            if room.label == RoomLabel.OTHER:
                if room.area < REL_THRESHOLDS['tiny']:
                    room.label = RoomLabel.STORAGE
                else:
                    room.label = RoomLabel.BEDROOM  # Default to bedroom
        
        return rooms
    
    def _classify_by_features(self, room: Room) -> RoomLabel:
        """Classify room based on its features."""
        area = room.area
        aspect = room.aspect_ratio
        features = room.features
        
        # Check for stairs first
        if features.get('has_parallel_lines', False) and aspect < 0.4:
            return RoomLabel.STAIRS
        
        # Very small = storage/closet
        if area < self.SIZE_THRESHOLDS['tiny']:
            return RoomLabel.STORAGE
        
        # Small and roughly square = bathroom
        if area < self.SIZE_THRESHOLDS['small']:
            if aspect > 0.5:
                return RoomLabel.BATHROOM
            else:
                return RoomLabel.STORAGE
        
        # Medium size
        if area < self.SIZE_THRESHOLDS['medium']:
            # Check position - kitchens often near edges
            if features.get('position_y', 0.5) > 0.6:  # Lower part of plan
                if aspect > 0.6:
                    return RoomLabel.KITCHEN
            return RoomLabel.BEDROOM
        
        # Large = living room
        if area < self.SIZE_THRESHOLDS['large']:
            if aspect > 0.5:
                return RoomLabel.LIVING_DINING
            else:
                return RoomLabel.BEDROOM
        
        # Very large = combined living/dining
        return RoomLabel.LIVING_DINING
    
    def _refine_classifications(self, rooms: List[Room], doors: List[Door]):
        """Refine room classifications using context."""
        # Count room types
        type_counts = defaultdict(int)
        for room in rooms:
            type_counts[room.label] += 1
        
        # Ensure at least one of each major room type
        has_kitchen = type_counts[RoomLabel.KITCHEN] > 0
        has_bathroom = type_counts[RoomLabel.BATHROOM] > 0
        has_bedroom = type_counts[RoomLabel.BEDROOM] > 0
        has_living = type_counts[RoomLabel.LIVING_DINING] > 0
        
        # If no kitchen, find best candidate
        if not has_kitchen:
            candidates = [r for r in rooms if r.label in 
                         [RoomLabel.BEDROOM, RoomLabel.OTHER] and
                         self.SIZE_THRESHOLDS['small'] < r.area < self.SIZE_THRESHOLDS['medium']]
            if candidates:
                # Pick the one with best kitchen characteristics
                best = max(candidates, key=lambda r: r.features.get('position_y', 0))
                best.label = RoomLabel.KITCHEN
        
        # Hallways are elongated spaces connecting rooms
        for room in rooms:
            if room.label == RoomLabel.OTHER:
                if room.aspect_ratio < 0.35:  # Very elongated
                    room.label = RoomLabel.HALLWAY


# =============================================================================
# MAIN ANALYZER CLASS
# =============================================================================

class FloorPlanAnalyzer:
    """
    Complete floor plan analyzer combining all detection modules.
    """
    
    def __init__(self):
        self.wall_detector = WallDoorDetector(
            min_wall_length=50,
            door_width_range=(15, 80)
        )
        self.room_analyzer = RoomAnalyzer(min_room_area=3000)
    
    def analyze(self, image: np.ndarray) -> Dict:
        """
        Analyze floor plan image.
        
        Returns:
            Dictionary with walls, doors, rooms, and visualization
        """
        logger.info(f"Analyzing floor plan: {image.shape}")
        
        # Preprocess
        binary = self.wall_detector.preprocess(image)
        
        # Detect walls
        walls = self.wall_detector.detect_walls(binary)
        logger.info(f"Detected {len(walls)} walls")
        
        # Detect doors
        doors = self.wall_detector.detect_doors(binary, walls)
        logger.info(f"Detected {len(doors)} doors")
        
        # Segment and classify rooms
        rooms = self.room_analyzer.segment_rooms(binary, walls)
        rooms = self.room_analyzer.classify_rooms(rooms, doors)
        logger.info(f"Detected {len(rooms)} rooms")
        
        # Log room breakdown
        room_counts = defaultdict(int)
        for room in rooms:
            room_counts[room.name] += 1
        for name, count in room_counts.items():
            logger.info(f"  {name}: {count}")
        
        return {
            'walls': walls,
            'doors': doors,
            'rooms': rooms,
            'binary': binary,
            'shape': image.shape[:2]
        }
    
    def visualize(self, image: np.ndarray, result: Dict, 
                  show_labels: bool = True) -> np.ndarray:
        """Create visualization of analysis results."""
        vis = image.copy()
        if len(vis.shape) == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        
        # Draw room fills with transparency
        overlay = vis.copy()
        for room in result['rooms']:
            color = ROOM_COLORS.get(room.label, (200, 200, 200))
            cv2.drawContours(overlay, [room.contour], -1, color, -1)
        cv2.addWeighted(overlay, 0.4, vis, 0.6, 0, vis)
        
        # Draw room contours
        for room in result['rooms']:
            color = ROOM_COLORS.get(room.label, (200, 200, 200))
            cv2.drawContours(vis, [room.contour], -1, color, 2)
            
            if show_labels:
                cx, cy = room.center
                label = room.name
                # Background for text
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(vis, (cx - tw//2 - 5, cy - th - 5), 
                             (cx + tw//2 + 5, cy + 5), (255, 255, 255), -1)
                cv2.putText(vis, label, (cx - tw//2, cy), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Draw walls
        for wall in result['walls']:
            cv2.line(vis, (wall.x1, wall.y1), (wall.x2, wall.y2), (0, 0, 255), 2)
        
        # Draw doors
        for door in result['doors']:
            cv2.rectangle(vis, (door.x, door.y), 
                         (door.x + door.width, door.y + door.height),
                         (0, 255, 0), 2)
        
        # Add legend
        y_offset = 30
        for label in [RoomLabel.KITCHEN, RoomLabel.LIVING_DINING, RoomLabel.BEDROOM,
                     RoomLabel.BATHROOM, RoomLabel.STORAGE, RoomLabel.HALLWAY]:
            color = ROOM_COLORS[label]
            cv2.rectangle(vis, (10, y_offset - 15), (30, y_offset), color, -1)
            cv2.putText(vis, ROOM_NAMES[label], (35, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            y_offset += 25
        
        return vis


# =============================================================================
# TEST FUNCTION
# =============================================================================

def test_analyzer(image_path: str):
    """Test the floor plan analyzer."""
    import os
    
    print("=" * 70)
    print("FLOOR PLAN ANALYZER TEST")
    print("=" * 70)
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load: {image_path}")
        return
    
    print(f"Image: {image_path}")
    print(f"Size: {img.shape[1]}x{img.shape[0]}")
    
    # Analyze
    analyzer = FloorPlanAnalyzer()
    result = analyzer.analyze(img)
    
    # Print results
    print(f"\nResults:")
    print(f"  Walls: {len(result['walls'])}")
    print(f"  Doors: {len(result['doors'])}")
    print(f"  Rooms: {len(result['rooms'])}")
    
    print("\nRoom breakdown:")
    for room in result['rooms']:
        x, y, w, h = room.bounding_box
        print(f"  [{room.id}] {room.name}: {room.area:.0f}px² ({w}x{h})")
    
    # Visualize
    vis = analyzer.visualize(img, result)
    
    # Save
    output_dir = os.path.dirname(image_path)
    vis_path = os.path.join(output_dir, "analyzer_result.png")
    cv2.imwrite(vis_path, vis)
    print(f"\nVisualization saved: {vis_path}")
    
    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_analyzer(sys.argv[1])
    else:
        print("Usage: python floor_plan_analyzer.py <image_path>")
