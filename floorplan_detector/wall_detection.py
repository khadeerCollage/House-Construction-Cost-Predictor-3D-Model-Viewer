"""
Wall Detection Module
======================
Advanced wall detection using multiple techniques:
1. Hough Line Transform with intelligent filtering
2. Line segment merging and extension
3. Wall intersection detection
4. Wall connectivity analysis
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class WallSegment:
    """Represents a wall segment with endpoints and properties."""
    x1: int
    y1: int
    x2: int
    y2: int
    thickness: float = 1.0
    orientation: str = "unknown"  # "horizontal", "vertical", or "diagonal"
    confidence: float = 1.0
    
    @property
    def length(self) -> float:
        return np.sqrt((self.x2 - self.x1)**2 + (self.y2 - self.y1)**2)
    
    @property
    def angle(self) -> float:
        """Returns angle in degrees from horizontal."""
        return np.degrees(np.arctan2(self.y2 - self.y1, self.x2 - self.x1))
    
    @property
    def midpoint(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)
    
    def get_normalized_endpoints(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Returns endpoints ordered (left-to-right or top-to-bottom)."""
        if self.x1 < self.x2 or (self.x1 == self.x2 and self.y1 < self.y2):
            return (self.x1, self.y1), (self.x2, self.y2)
        return (self.x2, self.y2), (self.x1, self.y1)


class WallDetector:
    """
    Advanced wall detection for floor plans.
    Uses multiple techniques to accurately detect and merge wall segments.
    """
    
    def __init__(
        self,
        min_line_length: int = 20,
        max_line_gap: int = 10,
        angle_tolerance: float = 5.0,  # degrees
        merge_distance: float = 15.0,
        min_wall_length: int = 30
    ):
        """
        Initialize wall detector.
        
        Args:
            min_line_length: Minimum line length for Hough detection
            max_line_gap: Maximum gap for line segment connection
            angle_tolerance: Tolerance for considering lines parallel (degrees)
            merge_distance: Maximum distance for merging parallel lines
            min_wall_length: Minimum length for final wall segments
        """
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.angle_tolerance = angle_tolerance
        self.merge_distance = merge_distance
        self.min_wall_length = min_wall_length
    
    def detect_lines(self, binary: np.ndarray) -> List[WallSegment]:
        """
        Detect line segments using Probabilistic Hough Transform.
        """
        # Apply Canny edge detection
        edges = cv2.Canny(binary, 50, 150, apertureSize=3)
        
        # Detect lines
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        
        if lines is None:
            logger.warning("No lines detected")
            return []
        
        segments = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            segment = WallSegment(x1, y1, x2, y2)
            segment.orientation = self._classify_orientation(segment.angle)
            segments.append(segment)
        
        logger.info(f"Detected {len(segments)} initial line segments")
        return segments
    
    def _classify_orientation(self, angle: float) -> str:
        """Classify line orientation based on angle."""
        angle = abs(angle) % 180
        if angle < 10 or angle > 170:
            return "horizontal"
        elif 80 < angle < 100:
            return "vertical"
        else:
            return "diagonal"
    
    def _are_collinear(self, seg1: WallSegment, seg2: WallSegment) -> bool:
        """Check if two segments are roughly collinear."""
        # Check angle similarity
        angle_diff = abs(seg1.angle - seg2.angle)
        angle_diff = min(angle_diff, 180 - angle_diff)
        
        if angle_diff > self.angle_tolerance:
            return False
        
        # Check if they're on the same line (perpendicular distance)
        mid1 = seg1.midpoint
        mid2 = seg2.midpoint
        
        # Calculate perpendicular distance from mid2 to line of seg1
        dx = seg1.x2 - seg1.x1
        dy = seg1.y2 - seg1.y1
        length = seg1.length
        
        if length == 0:
            return False
        
        # Distance from point to line
        dist = abs(dy * mid2[0] - dx * mid2[1] + seg1.x2 * seg1.y1 - seg1.y2 * seg1.x1) / length
        
        return dist < self.merge_distance
    
    def _can_merge(self, seg1: WallSegment, seg2: WallSegment) -> bool:
        """Check if two segments can be merged."""
        if not self._are_collinear(seg1, seg2):
            return False
        
        # Check if segments are close enough
        p1, p2 = seg1.get_normalized_endpoints()
        p3, p4 = seg2.get_normalized_endpoints()
        
        # Check endpoint distances
        min_dist = min(
            np.sqrt((p1[0] - p3[0])**2 + (p1[1] - p3[1])**2),
            np.sqrt((p1[0] - p4[0])**2 + (p1[1] - p4[1])**2),
            np.sqrt((p2[0] - p3[0])**2 + (p2[1] - p3[1])**2),
            np.sqrt((p2[0] - p4[0])**2 + (p2[1] - p4[1])**2)
        )
        
        # Also check if segments overlap
        overlap = self._check_overlap(seg1, seg2)
        
        return min_dist < self.merge_distance * 3 or overlap
    
    def _check_overlap(self, seg1: WallSegment, seg2: WallSegment) -> bool:
        """Check if two segments overlap along their direction."""
        if seg1.orientation == "horizontal" and seg2.orientation == "horizontal":
            # Check x-overlap
            return not (max(seg1.x1, seg1.x2) < min(seg2.x1, seg2.x2) or
                       max(seg2.x1, seg2.x2) < min(seg1.x1, seg1.x2))
        elif seg1.orientation == "vertical" and seg2.orientation == "vertical":
            # Check y-overlap
            return not (max(seg1.y1, seg1.y2) < min(seg2.y1, seg2.y2) or
                       max(seg2.y1, seg2.y2) < min(seg1.y1, seg1.y2))
        return False
    
    def _merge_segments(self, seg1: WallSegment, seg2: WallSegment) -> WallSegment:
        """Merge two collinear segments into one."""
        # Get all endpoints
        points = [
            (seg1.x1, seg1.y1),
            (seg1.x2, seg1.y2),
            (seg2.x1, seg2.y1),
            (seg2.x2, seg2.y2)
        ]
        
        # For horizontal-ish lines, sort by x
        # For vertical-ish lines, sort by y
        if seg1.orientation == "horizontal":
            points.sort(key=lambda p: p[0])
        elif seg1.orientation == "vertical":
            points.sort(key=lambda p: p[1])
        else:
            # For diagonal, sort by x then y
            points.sort(key=lambda p: (p[0], p[1]))
        
        # Use the extreme points
        merged = WallSegment(
            points[0][0], points[0][1],
            points[-1][0], points[-1][1]
        )
        merged.orientation = seg1.orientation
        merged.confidence = max(seg1.confidence, seg2.confidence)
        
        return merged
    
    def merge_collinear_segments(self, segments: List[WallSegment]) -> List[WallSegment]:
        """Merge collinear and close segments."""
        if not segments:
            return []
        
        # Group by orientation first
        groups = defaultdict(list)
        for seg in segments:
            groups[seg.orientation].append(seg)
        
        merged_all = []
        
        for orientation, segs in groups.items():
            merged = self._merge_group(segs)
            merged_all.extend(merged)
        
        logger.info(f"Merged {len(segments)} segments into {len(merged_all)}")
        return merged_all
    
    def _merge_group(self, segments: List[WallSegment]) -> List[WallSegment]:
        """Merge segments within an orientation group."""
        if not segments:
            return []
        
        # Iteratively merge until no more merges possible
        merged = segments.copy()
        changed = True
        
        while changed:
            changed = False
            new_merged = []
            used = set()
            
            for i, seg1 in enumerate(merged):
                if i in used:
                    continue
                
                current = seg1
                for j, seg2 in enumerate(merged[i+1:], i+1):
                    if j in used:
                        continue
                    
                    if self._can_merge(current, seg2):
                        current = self._merge_segments(current, seg2)
                        used.add(j)
                        changed = True
                
                new_merged.append(current)
                used.add(i)
            
            merged = new_merged
        
        return merged
    
    def snap_to_grid(self, segments: List[WallSegment], grid_size: int = 5) -> List[WallSegment]:
        """
        Snap wall endpoints to a grid for cleaner geometry.
        This helps create proper wall intersections.
        """
        snapped = []
        for seg in segments:
            x1 = round(seg.x1 / grid_size) * grid_size
            y1 = round(seg.y1 / grid_size) * grid_size
            x2 = round(seg.x2 / grid_size) * grid_size
            y2 = round(seg.y2 / grid_size) * grid_size
            
            new_seg = WallSegment(x1, y1, x2, y2)
            new_seg.orientation = seg.orientation
            new_seg.confidence = seg.confidence
            
            if new_seg.length >= self.min_wall_length:
                snapped.append(new_seg)
        
        return snapped
    
    def extend_to_intersections(self, segments: List[WallSegment], max_extension: int = 20) -> List[WallSegment]:
        """
        Extend wall segments to meet at intersections.
        This creates closed room polygons.
        """
        extended = []
        
        for i, seg in enumerate(segments):
            best_ext_start = (seg.x1, seg.y1)
            best_ext_end = (seg.x2, seg.y2)
            
            for other in segments:
                if seg == other:
                    continue
                
                # Check if extending would create intersection
                intersection = self._find_intersection(seg, other)
                if intersection:
                    ix, iy = intersection
                    
                    # Check if intersection is near segment endpoints
                    dist_to_start = np.sqrt((ix - seg.x1)**2 + (iy - seg.y1)**2)
                    dist_to_end = np.sqrt((ix - seg.x2)**2 + (iy - seg.y2)**2)
                    
                    if dist_to_start < max_extension and dist_to_start > 5:
                        best_ext_start = (int(ix), int(iy))
                    if dist_to_end < max_extension and dist_to_end > 5:
                        best_ext_end = (int(ix), int(iy))
            
            new_seg = WallSegment(
                best_ext_start[0], best_ext_start[1],
                best_ext_end[0], best_ext_end[1]
            )
            new_seg.orientation = seg.orientation
            new_seg.confidence = seg.confidence
            extended.append(new_seg)
        
        return extended
    
    def _find_intersection(self, seg1: WallSegment, seg2: WallSegment) -> Optional[Tuple[float, float]]:
        """Find intersection point of two line segments (extended as lines)."""
        x1, y1, x2, y2 = seg1.x1, seg1.y1, seg1.x2, seg1.y2
        x3, y3, x4, y4 = seg2.x1, seg2.y1, seg2.x2, seg2.y2
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None  # Parallel lines
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        
        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)
        
        return (ix, iy)
    
    def filter_by_length(self, segments: List[WallSegment]) -> List[WallSegment]:
        """Remove segments shorter than minimum wall length."""
        filtered = [s for s in segments if s.length >= self.min_wall_length]
        logger.info(f"Filtered to {len(filtered)} segments (min length: {self.min_wall_length})")
        return filtered
    
    def remove_duplicate_walls(self, segments: List[WallSegment]) -> List[WallSegment]:
        """Remove duplicate or very similar wall segments."""
        if not segments:
            return []
        
        unique = []
        for seg in segments:
            is_duplicate = False
            for existing in unique:
                # Check if segments are essentially the same
                if (self._are_collinear(seg, existing) and 
                    abs(seg.length - existing.length) < 10):
                    mid1 = seg.midpoint
                    mid2 = existing.midpoint
                    mid_dist = np.sqrt((mid1[0] - mid2[0])**2 + (mid1[1] - mid2[1])**2)
                    if mid_dist < 10:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique.append(seg)
        
        return unique
    
    def detect(self, binary: np.ndarray) -> List[WallSegment]:
        """
        Complete wall detection pipeline.
        
        Args:
            binary: Binary image with walls in white
            
        Returns:
            List of detected and processed wall segments
        """
        # Step 1: Detect initial line segments
        segments = self.detect_lines(binary)
        
        # Step 2: Merge collinear segments
        segments = self.merge_collinear_segments(segments)
        
        # Step 3: Snap to grid
        segments = self.snap_to_grid(segments)
        
        # Step 4: Extend to intersections
        segments = self.extend_to_intersections(segments)
        
        # Step 5: Merge again after extension
        segments = self.merge_collinear_segments(segments)
        
        # Step 6: Filter short segments
        segments = self.filter_by_length(segments)
        
        # Step 7: Remove duplicates
        segments = self.remove_duplicate_walls(segments)
        
        logger.info(f"Final wall count: {len(segments)}")
        return segments


def detect_walls(binary_image: np.ndarray, **kwargs) -> List[WallSegment]:
    """
    Convenience function for wall detection.
    
    Args:
        binary_image: Binary image with walls
        **kwargs: Parameters for WallDetector
        
    Returns:
        List of detected wall segments
    """
    detector = WallDetector(**kwargs)
    return detector.detect(binary_image)


def walls_to_image(walls: List[WallSegment], shape: Tuple[int, int], thickness: int = 2) -> np.ndarray:
    """
    Draw walls on an image for visualization.
    
    Args:
        walls: List of wall segments
        shape: Image shape (height, width)
        thickness: Line thickness
        
    Returns:
        Image with walls drawn
    """
    img = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    
    colors = {
        "horizontal": (0, 255, 0),  # Green
        "vertical": (0, 0, 255),    # Red
        "diagonal": (255, 0, 0)     # Blue
    }
    
    for wall in walls:
        color = colors.get(wall.orientation, (255, 255, 255))
        cv2.line(img, (wall.x1, wall.y1), (wall.x2, wall.y2), color, thickness)
    
    return img
