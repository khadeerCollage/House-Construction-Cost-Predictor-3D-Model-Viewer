"""
Smart Floor Plan Detection Module
==================================
Human-like detection of floor plan elements:
1. Distinguish walls from noise/text/dimensions
2. Identify actual rooms vs stairs/small spaces
3. Classify rooms based on size, shape, position
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpaceType(Enum):
    """Types of spaces in a floor plan."""
    ROOM = "room"
    STAIRS = "stairs"
    CORRIDOR = "corridor"
    BATHROOM = "bathroom"
    KITCHEN = "kitchen"
    BEDROOM = "bedroom"
    LIVING_ROOM = "living_room"
    BALCONY = "balcony"
    CLOSET = "closet"
    UTILITY = "utility"
    UNKNOWN = "unknown"


@dataclass
class DetectedSpace:
    """Represents a detected space with smart classification."""
    contour: np.ndarray
    space_type: SpaceType = SpaceType.UNKNOWN
    area: float = 0.0
    center: Tuple[float, float] = (0.0, 0.0)
    bounding_box: Tuple[int, int, int, int] = (0, 0, 0, 0)
    aspect_ratio: float = 1.0
    solidity: float = 1.0
    is_room: bool = True
    confidence: float = 1.0
    name: str = ""
    
    def __post_init__(self):
        if self.contour is not None and len(self.contour) > 0:
            self.area = cv2.contourArea(self.contour)
            M = cv2.moments(self.contour)
            if M['m00'] > 0:
                self.center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
            
            x, y, w, h = cv2.boundingRect(self.contour)
            self.bounding_box = (x, y, w, h)
            self.aspect_ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 1.0
            
            # Solidity = contour area / convex hull area
            hull = cv2.convexHull(self.contour)
            hull_area = cv2.contourArea(hull)
            self.solidity = self.area / hull_area if hull_area > 0 else 0


@dataclass  
class SmartWall:
    """Represents a wall with additional properties."""
    x1: int
    y1: int
    x2: int
    y2: int
    thickness: float = 3.0
    is_structural: bool = True
    confidence: float = 1.0
    
    @property
    def length(self) -> float:
        return np.sqrt((self.x2 - self.x1)**2 + (self.y2 - self.y1)**2)
    
    @property
    def angle(self) -> float:
        return np.degrees(np.arctan2(self.y2 - self.y1, self.x2 - self.x1))
    
    @property
    def is_horizontal(self) -> bool:
        angle = abs(self.angle) % 180
        return angle < 15 or angle > 165
    
    @property
    def is_vertical(self) -> bool:
        angle = abs(self.angle) % 180
        return 75 < angle < 105


class SmartFloorPlanDetector:
    """
    Smart floor plan detector that thinks like a human.
    
    Key principles:
    1. Walls are thick, continuous lines (not text or dimensions)
    2. Rooms are large enclosed spaces (not small boxes)
    3. Stairs have a distinctive pattern (parallel lines)
    4. Context matters (room positions, relative sizes)
    """
    
    def __init__(
        self,
        # Wall detection parameters
        min_wall_length: int = 50,        # Minimum wall length in pixels
        wall_thickness_range: Tuple[int, int] = (2, 15),  # Expected wall thickness
        
        # Room detection parameters  
        min_room_area: int = 8000,        # Minimum room area (filters small boxes)
        max_room_area: int = 500000,      # Maximum room area
        min_room_dimension: int = 60,     # Minimum width/height for a room
        
        # Classification thresholds
        stairs_aspect_ratio: float = 0.3,  # Stairs are usually narrow and long
        corridor_aspect_ratio: float = 0.25,  # Corridors are very elongated
        bathroom_max_area: int = 25000,    # Bathrooms are typically small
    ):
        self.min_wall_length = min_wall_length
        self.wall_thickness_range = wall_thickness_range
        self.min_room_area = min_room_area
        self.max_room_area = max_room_area
        self.min_room_dimension = min_room_dimension
        self.stairs_aspect_ratio = stairs_aspect_ratio
        self.corridor_aspect_ratio = corridor_aspect_ratio
        self.bathroom_max_area = bathroom_max_area
        
    def preprocess_image(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess image to extract clean wall lines.
        Optimized for hand-drawn floor plans with gaps.
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # 1. Apply bilateral filter to smooth while keeping edges
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # 2. Adaptive threshold - better for varying lighting
        binary = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=15,
            C=5
        )
        
        # 3. Remove small noise (text, dimension numbers)
        binary = self._remove_small_components(binary, min_area=80)
        
        # 4. Close gaps in walls - CRITICAL for hand-drawn plans
        # First close small gaps
        kernel_small = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_small)
        
        # Then use line-connecting morphology for larger gaps
        # Horizontal closing
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        closed_h = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_h)
        
        # Vertical closing  
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        closed_v = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_v)
        
        # Combine
        binary = cv2.bitwise_or(closed_h, closed_v)
        
        # 5. Dilate slightly to make walls thicker
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=1)
        
        return gray, binary
    
    def _remove_small_components(self, binary: np.ndarray, min_area: int = 100) -> np.ndarray:
        """Remove small connected components (noise, text)."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        result = np.zeros_like(binary)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                result[labels == i] = 255
                
        return result
    
    def _keep_thick_elements(self, binary: np.ndarray) -> np.ndarray:
        """Keep only thick elements (walls), remove thin lines."""
        # Erode to remove thin elements
        kernel = np.ones((2, 2), np.uint8)
        eroded = cv2.erode(binary, kernel, iterations=1)
        
        # Dilate back
        dilated = cv2.dilate(eroded, kernel, iterations=1)
        
        return dilated
    
    def detect_walls(self, binary: np.ndarray) -> List[SmartWall]:
        """
        Detect walls using smart line detection.
        Filters out noise and merges nearby segments.
        """
        # 1. Detect edges
        edges = cv2.Canny(binary, 50, 150, apertureSize=3)
        
        # 2. Hough line detection with strict parameters
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=80,              # Higher threshold = fewer false positives
            minLineLength=self.min_wall_length,
            maxLineGap=15
        )
        
        if lines is None:
            logger.warning("No lines detected")
            return []
        
        # 3. Convert to SmartWall objects
        raw_walls = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            wall = SmartWall(x1, y1, x2, y2)
            if wall.length >= self.min_wall_length:
                raw_walls.append(wall)
        
        logger.info(f"Detected {len(raw_walls)} raw line segments")
        
        # 4. Filter to keep only horizontal and vertical walls
        # (Real walls are usually orthogonal in floor plans)
        filtered_walls = [w for w in raw_walls if w.is_horizontal or w.is_vertical]
        logger.info(f"After filtering non-orthogonal: {len(filtered_walls)} walls")
        
        # 5. Merge nearby parallel walls
        merged_walls = self._merge_walls(filtered_walls)
        logger.info(f"After merging: {len(merged_walls)} walls")
        
        return merged_walls
    
    def _merge_walls(self, walls: List[SmartWall]) -> List[SmartWall]:
        """Merge nearby parallel wall segments."""
        if not walls:
            return []
        
        # Separate horizontal and vertical walls
        horizontal = [w for w in walls if w.is_horizontal]
        vertical = [w for w in walls if w.is_vertical]
        
        merged = []
        merged.extend(self._merge_parallel_walls(horizontal, is_horizontal=True))
        merged.extend(self._merge_parallel_walls(vertical, is_horizontal=False))
        
        return merged
    
    def _merge_parallel_walls(self, walls: List[SmartWall], is_horizontal: bool) -> List[SmartWall]:
        """Merge parallel walls that are close together."""
        if not walls:
            return []
        
        # Sort by position
        if is_horizontal:
            walls = sorted(walls, key=lambda w: (min(w.y1, w.y2), min(w.x1, w.x2)))
        else:
            walls = sorted(walls, key=lambda w: (min(w.x1, w.x2), min(w.y1, w.y2)))
        
        merged = []
        used = set()
        
        for i, wall1 in enumerate(walls):
            if i in used:
                continue
            
            # Find walls to merge with wall1
            to_merge = [wall1]
            used.add(i)
            
            for j, wall2 in enumerate(walls):
                if j in used:
                    continue
                
                if self._should_merge(wall1, wall2, is_horizontal):
                    to_merge.append(wall2)
                    used.add(j)
            
            # Create merged wall
            if len(to_merge) == 1:
                merged.append(wall1)
            else:
                merged.append(self._combine_walls(to_merge, is_horizontal))
        
        return merged
    
    def _should_merge(self, w1: SmartWall, w2: SmartWall, is_horizontal: bool) -> bool:
        """Check if two walls should be merged."""
        distance_threshold = 20  # pixels
        
        if is_horizontal:
            # Check if on same horizontal band
            y_diff = abs((w1.y1 + w1.y2) / 2 - (w2.y1 + w2.y2) / 2)
            if y_diff > distance_threshold:
                return False
            
            # Check if x ranges overlap or are close
            x1_min, x1_max = min(w1.x1, w1.x2), max(w1.x1, w1.x2)
            x2_min, x2_max = min(w2.x1, w2.x2), max(w2.x1, w2.x2)
            
            # Check for overlap or proximity
            gap = max(0, max(x1_min, x2_min) - min(x1_max, x2_max))
            return gap < distance_threshold * 2
        else:
            # Vertical walls
            x_diff = abs((w1.x1 + w1.x2) / 2 - (w2.x1 + w2.x2) / 2)
            if x_diff > distance_threshold:
                return False
            
            y1_min, y1_max = min(w1.y1, w1.y2), max(w1.y1, w1.y2)
            y2_min, y2_max = min(w2.y1, w2.y2), max(w2.y1, w2.y2)
            
            gap = max(0, max(y1_min, y2_min) - min(y1_max, y2_max))
            return gap < distance_threshold * 2
    
    def _combine_walls(self, walls: List[SmartWall], is_horizontal: bool) -> SmartWall:
        """Combine multiple walls into one."""
        if is_horizontal:
            y_avg = int(np.mean([w.y1 for w in walls] + [w.y2 for w in walls]))
            x_min = min(min(w.x1, w.x2) for w in walls)
            x_max = max(max(w.x1, w.x2) for w in walls)
            return SmartWall(x_min, y_avg, x_max, y_avg)
        else:
            x_avg = int(np.mean([w.x1 for w in walls] + [w.x2 for w in walls]))
            y_min = min(min(w.y1, w.y2) for w in walls)
            y_max = max(max(w.y1, w.y2) for w in walls)
            return SmartWall(x_avg, y_min, x_avg, y_max)
    
    def detect_spaces(self, binary: np.ndarray, walls: List[SmartWall]) -> List[DetectedSpace]:
        """
        Detect enclosed spaces (rooms, stairs, etc.) from wall structure.
        Uses multiple methods to find rooms.
        """
        # Method 1: Try using walls to find enclosed regions
        wall_mask = self._create_wall_mask(binary.shape, walls)
        spaces1 = self._find_enclosed_spaces(wall_mask)
        
        # Method 2: Detect directly from binary image
        spaces2 = self._find_spaces_from_binary(binary)
        
        # Combine results - use the method that found more rooms
        if len(spaces1) >= len(spaces2):
            spaces = spaces1
            logger.info(f"Using wall-based detection: {len(spaces)} spaces")
        else:
            spaces = spaces2
            logger.info(f"Using binary-based detection: {len(spaces)} spaces")
        
        # 3. Classify each space
        classified_spaces = []
        for space in spaces:
            classified = self._classify_space(space, binary)
            if classified.is_room or classified.space_type == SpaceType.STAIRS:
                classified_spaces.append(classified)
        
        # 4. Filter overlapping/duplicate spaces
        final_spaces = self._filter_overlapping_spaces(classified_spaces)
        
        logger.info(f"Detected {len(final_spaces)} spaces")
        return final_spaces
    
    def _find_spaces_from_binary(self, binary: np.ndarray) -> List[DetectedSpace]:
        """
        Find spaces directly from binary image using morphological operations.
        Better for real floor plans where walls might have gaps.
        """
        # AGGRESSIVE gap closing for hand-drawn plans
        # Step 1: Close small gaps
        kernel_small = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_small)
        
        # Step 2: Connect walls in horizontal and vertical directions
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        closed_h = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel_h)
        closed_v = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel_v)
        closed = cv2.bitwise_or(closed_h, closed_v)
        
        # Step 3: Dilate to make walls thicker
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(closed, kernel, iterations=2)
        
        # Invert to get room areas
        inverted = cv2.bitwise_not(dilated)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            inverted, connectivity=4
        )
        
        spaces = []
        image_area = binary.shape[0] * binary.shape[1]
        
        logger.info(f"Found {num_labels - 1} connected components")
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                        stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            
            # Filter by minimum area
            min_area = max(2500, self.min_room_area * 0.4)  
            if area < min_area:
                continue
            
            # Minimum dimensions for a real room
            if w < 40 or h < 40:
                continue
            
            # Skip outer boundary (entire floor plan)
            if area > 0.45 * image_area:
                logger.info(f"  Skipping space {i} - too large (likely boundary)")
                continue
            
            # Skip if both dimensions are close to image size (likely boundary)
            if w > binary.shape[1] * 0.8 and h > binary.shape[0] * 0.8:
                logger.info(f"  Skipping space {i} - dimensions match image size")
                continue
            
            # Create contour
            component_mask = (labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                space = DetectedSpace(contour=contours[0])
                logger.info(f"  Space {i}: area={area}, size={w}x{h}, aspect={space.aspect_ratio:.2f}")
                spaces.append(space)
        
        return spaces
    
    def _create_wall_mask(self, shape: Tuple[int, int], walls: List[SmartWall]) -> np.ndarray:
        """Create binary mask from walls."""
        mask = np.zeros(shape, dtype=np.uint8)
        
        for wall in walls:
            cv2.line(mask, (wall.x1, wall.y1), (wall.x2, wall.y2), 255, 3)
        
        # Close small gaps
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def _find_enclosed_spaces(self, wall_mask: np.ndarray) -> List[DetectedSpace]:
        """Find enclosed spaces using flood fill."""
        # Invert mask - spaces are white areas between walls
        inverted = cv2.bitwise_not(wall_mask)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            inverted, connectivity=4
        )
        
        spaces = []
        image_area = wall_mask.shape[0] * wall_mask.shape[1]
        
        for i in range(1, num_labels):  # Skip background
            area = stats[i, cv2.CC_STAT_AREA]
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                        stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            
            # Filter by area
            if area < self.min_room_area or area > self.max_room_area:
                continue
            
            # Filter by dimensions
            if w < self.min_room_dimension or h < self.min_room_dimension:
                continue
            
            # Skip if this is the outer boundary (very large)
            if area > 0.7 * image_area:
                continue
            
            # Create contour for this space
            component_mask = (labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                space = DetectedSpace(contour=contours[0])
                spaces.append(space)
        
        return spaces
    
    def _classify_space(self, space: DetectedSpace, binary: np.ndarray) -> DetectedSpace:
        """
        Classify a space as room, stairs, corridor, etc.
        Uses shape analysis and pattern detection.
        """
        x, y, w, h = space.bounding_box
        aspect_ratio = space.aspect_ratio
        area = space.area
        
        # Check for stairs pattern (parallel lines inside)
        if self._is_stairs(space, binary):
            space.space_type = SpaceType.STAIRS
            space.is_room = False
            space.name = "Stairs"
            return space
        
        # Very elongated = corridor
        if aspect_ratio < self.corridor_aspect_ratio:
            space.space_type = SpaceType.CORRIDOR
            space.is_room = False
            space.name = "Corridor"
            return space
        
        # Small space classification
        if area < self.bathroom_max_area:
            # Very small and elongated = likely closet or utility
            if area < 5000:
                if aspect_ratio < 0.4:
                    space.space_type = SpaceType.CLOSET
                    space.name = "Closet"
                else:
                    space.space_type = SpaceType.UTILITY
                    space.name = "Utility"
                space.is_room = True
                return space
            
            # Small but reasonable size
            if aspect_ratio > 0.6:  # Roughly square = bathroom
                space.space_type = SpaceType.BATHROOM
                space.name = "Bathroom"
            elif aspect_ratio < 0.35:  # Very elongated = closet
                space.space_type = SpaceType.CLOSET
                space.name = "Closet"
            else:
                space.space_type = SpaceType.ROOM
                space.name = "Room"
            space.is_room = True
            return space
        
        # Medium to large spaces
        if area > 35000:
            # Large space = living room or bedroom
            if aspect_ratio > 0.55:
                space.space_type = SpaceType.LIVING_ROOM
                space.name = "Living Room"
            else:
                space.space_type = SpaceType.BEDROOM
                space.name = "Bedroom"
        elif area > 20000:
            # Medium-large = bedroom
            space.space_type = SpaceType.BEDROOM
            space.name = "Bedroom"
        else:
            # Medium space - could be kitchen, bedroom, etc.
            if aspect_ratio > 0.6:
                space.space_type = SpaceType.KITCHEN
                space.name = "Kitchen"
            else:
                space.space_type = SpaceType.ROOM
                space.name = "Room"
        
        space.is_room = True
        return space
    
    def _is_stairs(self, space: DetectedSpace, binary: np.ndarray) -> bool:
        """
        Detect if a space contains stairs.
        Stairs have a distinctive pattern of parallel lines (treads).
        """
        x, y, w, h = space.bounding_box
        
        # Stairs are usually narrow and elongated
        if space.aspect_ratio > 0.5:
            return False
        
        # Extract region of interest
        roi = binary[y:y+h, x:x+w]
        if roi.size == 0:
            return False
        
        # Look for parallel horizontal lines (stair treads)
        # Use Hough transform to detect lines
        edges = cv2.Canny(roi, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=w//3, maxLineGap=5)
        
        if lines is None:
            return False
        
        # Count horizontal lines
        horizontal_lines = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < 20 or abs(angle) > 160:
                horizontal_lines += 1
        
        # Stairs typically have 5+ treads
        is_stairs = horizontal_lines >= 4
        
        if is_stairs:
            logger.info(f"Detected stairs with {horizontal_lines} treads")
        
        return is_stairs
    
    def _filter_overlapping_spaces(self, spaces: List[DetectedSpace]) -> List[DetectedSpace]:
        """Remove overlapping/duplicate spaces, keeping the best ones."""
        if len(spaces) <= 1:
            return spaces
        
        # Sort by area (larger first)
        spaces = sorted(spaces, key=lambda s: s.area, reverse=True)
        
        filtered = []
        for space in spaces:
            # Check if this space overlaps significantly with any kept space
            overlap = False
            for kept in filtered:
                iou = self._calculate_iou(space, kept)
                if iou > 0.5:  # More than 50% overlap
                    overlap = True
                    break
            
            if not overlap:
                filtered.append(space)
        
        return filtered
    
    def _calculate_iou(self, space1: DetectedSpace, space2: DetectedSpace) -> float:
        """Calculate Intersection over Union of two spaces."""
        x1, y1, w1, h1 = space1.bounding_box
        x2, y2, w2, h2 = space2.bounding_box
        
        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def detect(self, image: np.ndarray) -> Dict:
        """
        Main detection method.
        Returns walls and spaces (rooms, stairs, etc.)
        """
        # Preprocess
        gray, binary = self.preprocess_image(image)
        
        # Detect walls
        walls = self.detect_walls(binary)
        
        # Detect spaces
        spaces = self.detect_spaces(binary, walls)
        
        # Separate rooms and non-rooms
        rooms = [s for s in spaces if s.is_room]
        non_rooms = [s for s in spaces if not s.is_room]
        
        logger.info(f"Final: {len(walls)} walls, {len(rooms)} rooms, {len(non_rooms)} non-room spaces")
        
        return {
            'walls': walls,
            'rooms': rooms,
            'non_rooms': non_rooms,
            'all_spaces': spaces,
            'binary': binary,
            'shape': image.shape[:2]
        }
    
    def visualize(self, image: np.ndarray, detection_result: Dict, 
                  show_walls: bool = True, show_rooms: bool = True,
                  show_labels: bool = True) -> np.ndarray:
        """
        Create visualization of detection results.
        """
        vis = image.copy()
        if len(vis.shape) == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        
        # Draw walls
        if show_walls:
            for wall in detection_result['walls']:
                cv2.line(vis, (wall.x1, wall.y1), (wall.x2, wall.y2), (0, 0, 255), 2)
        
        # Draw rooms
        if show_rooms:
            room_count = 0
            for space in detection_result['all_spaces']:
                if space.is_room:
                    # Draw filled polygon with transparency
                    overlay = vis.copy()
                    cv2.drawContours(overlay, [space.contour], -1, (255, 200, 100), -1)
                    cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)
                    
                    # Draw contour
                    cv2.drawContours(vis, [space.contour], -1, (255, 150, 0), 2)
                    
                    room_count += 1
                    
                    if show_labels:
                        cx, cy = int(space.center[0]), int(space.center[1])
                        label = f"{space.name}"
                        cv2.putText(vis, label, (cx - 40, cy), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                else:
                    # Non-room spaces (stairs, corridors)
                    cv2.drawContours(vis, [space.contour], -1, (0, 255, 255), 2)
                    if show_labels:
                        cx, cy = int(space.center[0]), int(space.center[1])
                        cv2.putText(vis, space.name, (cx - 30, cy),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 2)
        
        # Add summary
        walls_count = len(detection_result['walls'])
        rooms_count = len(detection_result['rooms'])
        cv2.putText(vis, f"Walls: {walls_count} | Rooms: {rooms_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        return vis
