"""
YOLO-Based Room Labeling
==========================
Uses YOLOv8 to detect fixtures and objects in floor plans to help identify room types.

Approach:
1. Detect objects/fixtures in the floor plan (toilet, sink, stove, bed symbols)
2. Use detected objects to infer room types
3. Combine with geometry-based room detection

This works best with floor plans that have furniture/fixture symbols.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import ultralytics (YOLOv8)
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
    logger.warning("ultralytics not installed. Install with: pip install ultralytics")


# Mapping of COCO classes to room types
# These are objects that might appear in floor plan symbols
OBJECT_TO_ROOM = {
    # Bathroom indicators
    'toilet': 'bathroom',
    'sink': 'bathroom',  # Could also be kitchen
    'bathtub': 'bathroom',
    
    # Kitchen indicators
    'oven': 'kitchen',
    'refrigerator': 'kitchen',
    'microwave': 'kitchen',
    'toaster': 'kitchen',
    
    # Bedroom indicators
    'bed': 'bedroom',
    
    # Living room indicators
    'couch': 'living_room',
    'tv': 'living_room',
    'sofa': 'living_room',
    
    # Dining indicators
    'dining table': 'dining_room',
    'chair': 'dining_room',  # Multiple chairs = dining
    
    # Garage indicators
    'car': 'garage',
    
    # General
    'potted plant': None,
    'clock': None,
}


@dataclass
class DetectedObject:
    """Object detected by YOLO."""
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]
    suggested_room: Optional[str] = None


class YOLORoomLabeler:
    """
    Uses YOLO to detect objects/fixtures that help identify room types.
    """
    
    def __init__(self, model_path: str = "yolov8n.pt"):
        """
        Initialize YOLO model.
        
        Args:
            model_path: Path to YOLO model weights
        """
        self.model = None
        self.model_path = model_path
        
        if HAS_YOLO:
            try:
                self.model = YOLO(model_path)
                logger.info(f"Loaded YOLO model: {model_path}")
            except Exception as e:
                logger.error(f"Failed to load YOLO model: {e}")
    
    def detect_objects(self, image: np.ndarray, conf_threshold: float = 0.3) -> List[DetectedObject]:
        """
        Detect objects in the floor plan image.
        
        Args:
            image: Input image (BGR)
            conf_threshold: Minimum confidence threshold
            
        Returns:
            List of detected objects
        """
        if self.model is None:
            logger.warning("YOLO model not loaded")
            return []
        
        # Run detection
        results = self.model(image, conf=conf_threshold, verbose=False)
        
        objects = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                class_name = result.names[cls_id]
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                
                # Get suggested room type
                suggested_room = OBJECT_TO_ROOM.get(class_name.lower())
                
                obj = DetectedObject(
                    class_name=class_name,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    center=center,
                    suggested_room=suggested_room
                )
                objects.append(obj)
        
        logger.info(f"Detected {len(objects)} objects")
        return objects
    
    def label_rooms_by_objects(self, rooms: List, objects: List[DetectedObject]) -> List:
        """
        Update room labels based on detected objects.
        
        Args:
            rooms: List of Room objects (must have center, label attributes)
            objects: List of detected objects
            
        Returns:
            Updated rooms list
        """
        from .floor_plan_analyzer import RoomLabel
        
        # For each room, check which objects are inside
        for room in rooms:
            room_objects = []
            
            for obj in objects:
                # Check if object center is inside room
                if self._point_in_room(obj.center, room):
                    room_objects.append(obj)
            
            if room_objects:
                # Determine room type from objects
                suggested_type = self._infer_room_type(room_objects)
                if suggested_type:
                    room.label = suggested_type
                    logger.info(f"Room {room.id} labeled as {room.name} based on objects")
        
        return rooms
    
    def _point_in_room(self, point: Tuple[int, int], room) -> bool:
        """Check if point is inside room contour."""
        if room.contour is None:
            return False
        result = cv2.pointPolygonTest(room.contour, point, False)
        return result >= 0
    
    def _infer_room_type(self, objects: List[DetectedObject]):
        """Infer room type from detected objects."""
        from .floor_plan_analyzer import RoomLabel
        
        room_votes = {}
        
        for obj in objects:
            if obj.suggested_room:
                room_votes[obj.suggested_room] = room_votes.get(obj.suggested_room, 0) + obj.confidence
        
        if not room_votes:
            return None
        
        # Get highest voted room type
        best_type = max(room_votes, key=room_votes.get)
        
        # Map to RoomLabel
        type_mapping = {
            'bathroom': RoomLabel.BATHROOM,
            'kitchen': RoomLabel.KITCHEN,
            'bedroom': RoomLabel.BEDROOM,
            'living_room': RoomLabel.LIVING_DINING,
            'dining_room': RoomLabel.LIVING_DINING,
            'garage': RoomLabel.GARAGE,
        }
        
        return type_mapping.get(best_type)


class FloorPlanSymbolDetector:
    """
    Detects common floor plan symbols using template matching and shape analysis.
    Works without YOLO for floor plans with standard architectural symbols.
    """
    
    def __init__(self):
        self.templates = {}
    
    def detect_toilet_symbols(self, binary: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect toilet symbols (usually oval/ellipse shapes).
        """
        toilets = []
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Toilet symbols are typically small ovals
            if 200 < area < 3000:
                # Check if ellipse-like
                if len(contour) >= 5:
                    ellipse = cv2.fitEllipse(contour)
                    (cx, cy), (w, h), angle = ellipse
                    
                    # Toilet is usually elongated oval
                    aspect = min(w, h) / max(w, h) if max(w, h) > 0 else 1
                    if 0.4 < aspect < 0.8:
                        toilets.append((int(cx), int(cy)))
        
        return toilets
    
    def detect_sink_symbols(self, binary: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect sink symbols (usually small rectangles or circles).
        """
        sinks = []
        
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Sink symbols are small
            if 100 < area < 2000:
                # Approximate to polygon
                epsilon = 0.04 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Rectangular sink = 4 vertices
                if 4 <= len(approx) <= 6:
                    M = cv2.moments(contour)
                    if M['m00'] > 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        sinks.append((cx, cy))
        
        return sinks
    
    def detect_stair_symbols(self, binary: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect stair symbols (parallel horizontal lines in a rectangle).
        Returns bounding boxes of detected stairs.
        """
        stairs = []
        
        # Find rectangular regions with parallel lines
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            
            # Stairs have specific aspect ratio
            if area > 1000:
                aspect = min(w, h) / max(w, h) if max(w, h) > 0 else 1
                
                if aspect < 0.5:  # Elongated
                    # Check for parallel lines inside
                    roi = binary[y:y+h, x:x+w]
                    if self._has_stair_pattern(roi):
                        stairs.append((x, y, w, h))
        
        return stairs
    
    def _has_stair_pattern(self, roi: np.ndarray) -> bool:
        """Check if ROI has stair-like pattern."""
        if roi.size == 0:
            return False
        
        h, w = roi.shape[:2]
        if h == 0 or w == 0:
            return False
        
        # Look for horizontal lines
        edges = cv2.Canny(roi, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 15, 
                               minLineLength=w//3, maxLineGap=5)
        
        if lines is None:
            return False
        
        # Count horizontal lines
        h_count = sum(1 for line in lines 
                     if abs(np.degrees(np.arctan2(
                         line[0][3] - line[0][1], 
                         line[0][2] - line[0][0]))) < 20)
        
        return h_count >= 3
    
    def detect_door_arcs(self, binary: np.ndarray) -> List[Tuple[int, int, int]]:
        """
        Detect door swing arcs (quarter circles).
        Returns list of (x, y, radius).
        """
        doors = []
        
        # Detect arcs using Hough circles with relaxed parameters
        blurred = cv2.GaussianBlur(binary, (5, 5), 0)
        
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, 1, 30,
            param1=50, param2=25,
            minRadius=15, maxRadius=50
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)
            for (x, y, r) in circles:
                doors.append((x, y, r))
        
        return doors


def test_yolo_labeling(image_path: str):
    """Test YOLO-based room labeling."""
    print("=" * 70)
    print("YOLO ROOM LABELING TEST")
    print("=" * 70)
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load: {image_path}")
        return
    
    print(f"Image: {image_path}")
    print(f"Size: {img.shape[1]}x{img.shape[0]}")
    
    # Initialize YOLO labeler
    labeler = YOLORoomLabeler()
    
    if labeler.model is None:
        print("\nYOLO model not available. Testing symbol detector instead.")
        
        # Test symbol detector
        detector = FloorPlanSymbolDetector()
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 15, 5)
        
        toilets = detector.detect_toilet_symbols(binary)
        sinks = detector.detect_sink_symbols(binary)
        stairs = detector.detect_stair_symbols(binary)
        doors = detector.detect_door_arcs(binary)
        
        print(f"\nDetected symbols:")
        print(f"  Possible toilets: {len(toilets)}")
        print(f"  Possible sinks: {len(sinks)}")
        print(f"  Stair areas: {len(stairs)}")
        print(f"  Door arcs: {len(doors)}")
        
        return
    
    # Detect objects
    objects = labeler.detect_objects(img)
    
    print(f"\nDetected {len(objects)} objects:")
    for obj in objects:
        print(f"  {obj.class_name}: {obj.confidence:.2f} at {obj.center}")
        if obj.suggested_room:
            print(f"    -> Suggests: {obj.suggested_room}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_yolo_labeling(sys.argv[1])
    else:
        print("Usage: python yolo_room_labeler.py <image_path>")
