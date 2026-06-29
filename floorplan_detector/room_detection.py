"""
Room Detection Module
======================
Advanced room detection using:
1. Contour hierarchy analysis
2. Wall boundary detection
3. Room polygon extraction
4. Room classification
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging

from .wall_detection import WallSegment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RoomType(Enum):
    """Room type classification."""
    BEDROOM = "bedroom"
    BATHROOM = "bathroom"
    KITCHEN = "kitchen"
    LIVING_ROOM = "living_room"
    DINING_ROOM = "dining_room"
    HALLWAY = "hallway"
    BALCONY = "balcony"
    UNKNOWN = "unknown"


@dataclass
class Room:
    """Represents a detected room with properties."""
    contour: np.ndarray
    room_type: RoomType = RoomType.UNKNOWN
    area: float = 0.0
    center: Tuple[float, float] = (0.0, 0.0)
    bounding_box: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h
    vertices: List[Tuple[int, int]] = field(default_factory=list)
    confidence: float = 1.0
    label: str = ""
    
    def __post_init__(self):
        if self.contour is not None and len(self.contour) > 0:
            self.area = cv2.contourArea(self.contour)
            M = cv2.moments(self.contour)
            if M['m00'] > 0:
                self.center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
            self.bounding_box = cv2.boundingRect(self.contour)
            
            # Simplify contour to polygon
            epsilon = 0.02 * cv2.arcLength(self.contour, True)
            approx = cv2.approxPolyDP(self.contour, epsilon, True)
            self.vertices = [(int(p[0][0]), int(p[0][1])) for p in approx]


class RoomDetector:
    """
    Advanced room detection for floor plans.
    Detects closed regions bounded by walls.
    """
    
    # Room classification criteria
    ROOM_CRITERIA = {
        RoomType.BATHROOM: {
            'min_area': 3000,
            'max_area': 25000,
            'aspect_range': (0.5, 2.0)
        },
        RoomType.BEDROOM: {
            'min_area': 20000,
            'max_area': 100000,
            'aspect_range': (0.6, 1.8)
        },
        RoomType.LIVING_ROOM: {
            'min_area': 30000,
            'max_area': 150000,
            'aspect_range': (0.5, 2.5)
        },
        RoomType.KITCHEN: {
            'min_area': 15000,
            'max_area': 50000,
            'aspect_range': (0.7, 2.0)
        },
        RoomType.HALLWAY: {
            'min_area': 5000,
            'max_area': 30000,
            'aspect_range': (0.1, 0.4)  # Very elongated
        }
    }
    
    def __init__(
        self,
        min_room_area: int = 2000,
        max_room_area: int = 200000,
        simplify_epsilon: float = 0.02
    ):
        """
        Initialize room detector.
        
        Args:
            min_room_area: Minimum area for a region to be considered a room
            max_room_area: Maximum area for a room
            simplify_epsilon: Epsilon for polygon simplification (as fraction of perimeter)
        """
        self.min_room_area = min_room_area
        self.max_room_area = max_room_area
        self.simplify_epsilon = simplify_epsilon
    
    def create_wall_mask(self, walls: List[WallSegment], shape: Tuple[int, int], thickness: int = 3) -> np.ndarray:
        """
        Create a binary mask from wall segments.
        """
        mask = np.zeros(shape, dtype=np.uint8)
        
        for wall in walls:
            cv2.line(mask, (wall.x1, wall.y1), (wall.x2, wall.y2), 255, thickness)
        
        # Close gaps in walls
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def find_enclosed_regions(self, wall_mask: np.ndarray) -> List[np.ndarray]:
        """
        Find enclosed regions (rooms) from wall mask.
        """
        # Invert mask - rooms are the white areas between walls
        inverted = cv2.bitwise_not(wall_mask)
        
        # Find contours of enclosed regions
        contours, hierarchy = cv2.findContours(
            inverted, 
            cv2.RETR_CCOMP,  # Two-level hierarchy
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if hierarchy is None or len(contours) == 0:
            logger.warning("No enclosed regions found")
            return []
        
        # Filter contours by area and hierarchy
        # We want inner contours (rooms), not the outer boundary
        rooms = []
        h = hierarchy[0]
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Skip if area is too small or too large
            if area < self.min_room_area or area > self.max_room_area:
                continue
            
            # Check if this is a valid room (not the outer boundary)
            # A room should have a parent contour (the outer boundary)
            parent_idx = h[i][3]
            
            # If no parent and very large area, it's likely the outer boundary
            if parent_idx == -1 and area > 0.7 * (wall_mask.shape[0] * wall_mask.shape[1]):
                continue
            
            rooms.append(contour)
        
        logger.info(f"Found {len(rooms)} potential rooms")
        return rooms
    
    def detect_rooms_from_binary(self, binary: np.ndarray) -> List[np.ndarray]:
        """
        Detect rooms directly from binary image.
        Uses flood fill to find enclosed regions.
        """
        # Ensure walls are thick enough
        kernel = np.ones((3, 3), np.uint8)
        thick_walls = cv2.dilate(binary, kernel, iterations=1)
        
        # Invert to get room areas
        inverted = cv2.bitwise_not(thick_walls)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted, connectivity=8)
        
        rooms = []
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            
            if area < self.min_room_area or area > self.max_room_area:
                continue
            
            # Create mask for this component
            component_mask = (labels == i).astype(np.uint8) * 255
            
            # Find contour
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                rooms.append(contours[0])
        
        return rooms
    
    def classify_room(self, room: Room) -> RoomType:
        """
        Classify room type based on area and aspect ratio.
        """
        x, y, w, h = room.bounding_box
        aspect_ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 1.0
        
        # Check against each room type criteria
        for room_type, criteria in self.ROOM_CRITERIA.items():
            if (criteria['min_area'] <= room.area <= criteria['max_area'] and
                criteria['aspect_range'][0] <= aspect_ratio <= criteria['aspect_range'][1]):
                return room_type
        
        # Default classification based on area
        if room.area < 5000:
            return RoomType.BATHROOM
        elif room.area < 20000:
            return RoomType.HALLWAY
        elif room.area < 50000:
            return RoomType.BEDROOM
        else:
            return RoomType.LIVING_ROOM
    
    def simplify_room_polygon(self, contour: np.ndarray) -> np.ndarray:
        """
        Simplify room contour to a cleaner polygon.
        Prefer right angles for rooms.
        """
        # Basic simplification
        epsilon = self.simplify_epsilon * cv2.arcLength(contour, True)
        simplified = cv2.approxPolyDP(contour, epsilon, True)
        
        # Try to make right angles
        simplified = self._orthogonalize(simplified)
        
        return simplified
    
    def _orthogonalize(self, polygon: np.ndarray, angle_threshold: float = 15.0) -> np.ndarray:
        """
        Adjust polygon vertices to create more orthogonal edges.
        """
        if len(polygon) < 4:
            return polygon
        
        vertices = polygon.reshape(-1, 2)
        n = len(vertices)
        new_vertices = vertices.copy()
        
        for i in range(n):
            prev_idx = (i - 1) % n
            next_idx = (i + 1) % n
            
            # Calculate vectors
            v1 = vertices[i] - vertices[prev_idx]
            v2 = vertices[next_idx] - vertices[i]
            
            # Calculate angle between vectors
            angle = np.degrees(np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0]))
            angle = angle % 180
            
            # If close to 90 degrees, snap to it
            if abs(angle - 90) < angle_threshold:
                # Adjust vertex to make it exactly 90 degrees
                # This is a simplified version - just keeps the vertex as is
                pass
        
        return new_vertices.reshape(-1, 1, 2).astype(np.int32)
    
    def detect_from_walls(self, walls: List[WallSegment], shape: Tuple[int, int]) -> List[Room]:
        """
        Detect rooms from wall segments.
        
        Args:
            walls: List of wall segments
            shape: Image shape (height, width)
            
        Returns:
            List of detected rooms
        """
        # Create wall mask
        wall_mask = self.create_wall_mask(walls, shape, thickness=3)
        
        # Find enclosed regions
        contours = self.find_enclosed_regions(wall_mask)
        
        # Create Room objects
        rooms = []
        room_counts = {}
        
        for contour in contours:
            room = Room(contour)
            room.room_type = self.classify_room(room)
            
            # Generate label
            if room.room_type not in room_counts:
                room_counts[room.room_type] = 0
            room_counts[room.room_type] += 1
            room.label = f"{room.room_type.value}_{room_counts[room.room_type]}"
            
            rooms.append(room)
        
        logger.info(f"Detected {len(rooms)} rooms")
        for rt in room_counts:
            logger.info(f"  {rt.value}: {room_counts[rt]}")
        
        return rooms
    
    def detect_from_binary(self, binary: np.ndarray) -> List[Room]:
        """
        Detect rooms directly from binary image.
        
        Args:
            binary: Binary image with walls in white
            
        Returns:
            List of detected rooms
        """
        contours = self.detect_rooms_from_binary(binary)
        
        rooms = []
        room_counts = {}
        
        for contour in contours:
            room = Room(contour)
            room.room_type = self.classify_room(room)
            
            if room.room_type not in room_counts:
                room_counts[room.room_type] = 0
            room_counts[room.room_type] += 1
            room.label = f"{room.room_type.value}_{room_counts[room.room_type]}"
            
            rooms.append(room)
        
        return rooms


def detect_rooms(
    binary_image: np.ndarray = None,
    walls: List[WallSegment] = None,
    shape: Tuple[int, int] = None,
    **kwargs
) -> List[Room]:
    """
    Convenience function for room detection.
    
    Args:
        binary_image: Binary image with walls (optional)
        walls: Wall segments (optional)
        shape: Image shape if using walls
        **kwargs: Parameters for RoomDetector
        
    Returns:
        List of detected rooms
    """
    detector = RoomDetector(**kwargs)
    
    if walls is not None and shape is not None:
        return detector.detect_from_walls(walls, shape)
    elif binary_image is not None:
        return detector.detect_from_binary(binary_image)
    else:
        raise ValueError("Either binary_image or (walls and shape) must be provided")


def rooms_to_image(rooms: List[Room], shape: Tuple[int, int]) -> np.ndarray:
    """
    Draw rooms on an image for visualization.
    """
    img = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    
    colors = {
        RoomType.BEDROOM: (255, 255, 150),    # Light yellow
        RoomType.BATHROOM: (150, 150, 255),   # Light purple
        RoomType.KITCHEN: (150, 200, 255),    # Light orange
        RoomType.LIVING_ROOM: (200, 255, 200), # Light green
        RoomType.DINING_ROOM: (255, 200, 150), # Light cyan
        RoomType.HALLWAY: (200, 200, 200),    # Light gray
        RoomType.UNKNOWN: (180, 180, 180)     # Gray
    }
    
    for room in rooms:
        color = colors.get(room.room_type, (180, 180, 180))
        cv2.drawContours(img, [room.contour], -1, color, -1)
        cv2.drawContours(img, [room.contour], -1, (0, 0, 0), 2)
        
        # Draw label
        cv2.putText(
            img, room.label,
            (int(room.center[0]) - 30, int(room.center[1])),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
        )
    
    return img
