"""
Utility functions for room detection and visualization
This module now uses the improved FloorPlanAnalyzer for accurate detection
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import os
import sys
import time

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import the improved FloorPlanAnalyzer
from floorplan_detector.floor_plan_analyzer import (
    FloorPlanAnalyzer, RoomLabel, ROOM_NAMES
)

# Room configuration for frontend display
ROOM_CONFIG = {
    "Living/Dining": {
        "color": "#8DD3C7",  # Teal
        "icon": "🛋️",
        "features": ["Sofa", "TV", "Coffee Table", "Dining Table"],
        "description": "Main living and dining space"
    },
    "Living Room": {
        "color": "#8DD3C7",  # Teal
        "icon": "🛋️",
        "features": ["Sofa", "TV", "Coffee Table"],
        "description": "Main living space for family activities"
    },
    "Bedroom": {
        "color": "#FFFFB3",  # Light yellow
        "icon": "🛏️",
        "features": ["Bed", "Wardrobe", "Nightstand"],
        "description": "Private sleeping area"
    },
    "Bathroom": {
        "color": "#BEBADA",  # Lavender
        "icon": "🚿",
        "features": ["Shower", "Toilet", "Sink"],
        "description": "Personal hygiene space"
    },
    "Kitchen": {
        "color": "#FB8072",  # Salmon pink
        "icon": "🍳",
        "features": ["Stove", "Fridge", "Sink"],
        "description": "Cooking and food preparation area"
    },
    "Hallway": {
        "color": "#D9D9D9",  # Gray
        "icon": "🚶",
        "features": ["Passage", "Corridor"],
        "description": "Connecting passage between rooms"
    },
    "Storage": {
        "color": "#FDB462",  # Orange
        "icon": "📦",
        "features": ["Closet", "Shelves"],
        "description": "Storage space"
    },
    "Stairs": {
        "color": "#B3DE69",  # Green
        "icon": "🪜",
        "features": ["Steps", "Railing"],
        "description": "Staircase connecting floors"
    },
    "Room": {
        "color": "#80B1D3",  # Light blue
        "icon": "🚪",
        "features": ["Multi-purpose", "Generic space"],
        "description": "General purpose room"
    },
    "Other": {
        "color": "#CCEBC5",  # Light green
        "icon": "🏠",
        "features": ["Multi-purpose"],
        "description": "Other space"
    }
}

# Map RoomLabel enum to display names
LABEL_TO_NAME = {
    RoomLabel.LIVING_DINING: "Living/Dining",
    RoomLabel.KITCHEN: "Kitchen",
    RoomLabel.BEDROOM: "Bedroom",
    RoomLabel.BATHROOM: "Bathroom",
    RoomLabel.HALLWAY: "Hallway",
    RoomLabel.STORAGE: "Storage",
    RoomLabel.STAIRS: "Stairs",
    RoomLabel.GARAGE: "Room",
    RoomLabel.OTHER: "Room",
    RoomLabel.BACKGROUND: "Room",
    RoomLabel.OUTDOOR: "Room",
    RoomLabel.WALL: "Room",
}

def detect_rooms_and_walls(image_path, output_dir=None, min_area=500, max_area=50000, plot_width=None, plot_length=None, **kwargs):
    """
    Detect walls and rooms in a floorplan image using improved FloorPlanAnalyzer
    
    Args:
        image_path: Path to the input image
        output_dir: Directory to save visualization (optional)
        min_area: Minimum room area (not used, kept for compatibility)
        max_area: Maximum room area (not used, kept for compatibility)
        plot_width: Real-world plot width in meters
        plot_length: Real-world plot length in meters
        
    Returns:
        dict: Detection results including room data and visualization path
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    
    # Resize large images
    max_dimension = 1200
    height, width = img.shape[:2]
    if height > max_dimension or width > max_dimension:
        scale = max_dimension / max(height, width)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
    # Use the improved FloorPlanAnalyzer
    analyzer = FloorPlanAnalyzer()
    result = analyzer.analyze(img)
    
    walls = result['walls']
    doors = result['doors']
    rooms = result['rooms']
    
    wall_count = len(walls)
    room_count = len(rooms)
    
    # Initialize scale calibration
    try:
        from scale_calibration import ScaleCalibrator
    except ImportError:
        from model_folder.scale_calibration import ScaleCalibrator
    
    # Treat 0.0 as "not provided"
    effective_width = plot_width if plot_width and plot_width > 0 else None
    effective_length = plot_length if plot_length and plot_length > 0 else None
        
    calibrator = ScaleCalibrator(
        image_shape=img.shape,
        plot_width_m=effective_width,
        plot_length_m=effective_length
    )
    
    # Fallback scale estimation if no dimensions are supplied
    if not effective_width and not effective_length:
        calibrator.ppm = ScaleCalibrator.get_fallback_ppm_for_rooms(room_count, img.shape)
        
    # Calculate estimated total area from the calibrator
    real_width = calibrator.pixel_length_to_meters(img.shape[1])
    real_length = calibrator.pixel_length_to_meters(img.shape[0])
    # Total canvas area in sqm
    total_canvas_area_sqm = real_width * real_length
    # Estimated build-up area is usually a fraction of the plot
    estimated_build_area = total_canvas_area_sqm * 0.7
    
    # Initialize room data structures
    room_data = []
    room_type_counts = {room_type: 0 for room_type in ROOM_CONFIG.keys()}

    
    # Create visualization
    plt.figure(figsize=(12, 10), dpi=120)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    
    # Draw walls in red
    for wall in walls:
        plt.plot([wall.x1, wall.x2], [wall.y1, wall.y2], color='red', linewidth=2, alpha=0.8)
    
    # Draw doors in green
    for door in doors:
        rect = patches.Rectangle(
            (door.x, door.y), door.width, door.height,
            linewidth=2,
            edgecolor='green',
            facecolor='green',
            alpha=0.3
        )
        plt.gca().add_patch(rect)
        plt.text(door.x + 2, door.y + door.height//2, "🚪", fontsize=10)
    
    # Process each room
    for i, room in enumerate(rooms):
        x, y, w, h = room.bounding_box
        area = room.area
        
        # Get room type name from label
        room_type = LABEL_TO_NAME.get(room.label, room.name)
        if room_type not in ROOM_CONFIG:
            room_type = "Room"
        
        # Get room configuration
        config = ROOM_CONFIG[room_type]
        color = config["color"]
        icon = config["icon"]
        
        # Update room count
        room_type_counts[room_type] = room_type_counts.get(room_type, 0) + 1
        room_id = room_type_counts[room_type]
        
        # Draw room contour with color fill
        if room.contour is not None and len(room.contour) > 0:
            contour_points = room.contour.reshape(-1, 2)
            polygon = patches.Polygon(
                contour_points,
                linewidth=2,
                edgecolor=color,
                facecolor=color,
                alpha=0.4
            )
            plt.gca().add_patch(polygon)
        else:
            # Fallback to rectangle
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=2,
                edgecolor=color,
                facecolor=color,
                alpha=0.4
            )
            plt.gca().add_patch(rect)
        
        # Add room label with icon
        label = f"{icon} {room_type}"
        if room_id > 1 or room_type_counts[room_type] > 1:
            label += f" {room_id}"
        
        # Position label at room center
        cx, cy = room.center
        plt.text(cx - 30, cy, label, color='black', fontsize=11,
                 fontweight='bold', bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))
        
        # Store detailed room info
        room_data.append({
            "id": i + 1,
            "type": room_type,
            "type_id": room_id,
            "label": label,
            "icon": icon,
            "area": float(calibrator.pixel_area_to_sqm(area)),
            "area_px": float(area),
            "bounds": [int(x), int(y), int(w), int(h)],
            "aspect_ratio": float(room.aspect_ratio),
            "features": config["features"],
            "description": config["description"],
            "color": color
        })
    
    # Add title with stats
    plt.title(f"Floor Plan Analysis\nWalls: {wall_count} | Doors: {len(doors)} | Rooms: {room_count}", 
              fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    # Save visualization if output directory provided
    vis_path = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = int(time.time())
        filename = f'detection_{timestamp}.png'
        vis_path = os.path.join(output_dir, filename)
        plt.savefig(vis_path, format='png', dpi=120, bbox_inches='tight')
    
    plt.close()
    
    # Organize rooms by type
    rooms_by_type = {room_type: [] for room_type in ROOM_CONFIG.keys()}
    for room in room_data:
        room_type = room["type"]
        if room_type in rooms_by_type:
            rooms_by_type[room_type].append(room)
            
    # Calculate total room area in square meters
    total_room_area_sqm = sum(room["area"] for room in room_data)
    # Ensure a minimum reasonable building area if detection was noisy
    if total_room_area_sqm <= 0:
        total_room_area_sqm = estimated_build_area
    
    # Return detection results
    return {
        'wall_count': wall_count,
        'door_count': len(doors),
        'room_count': room_count,
        'visualization': vis_path,
        'rooms': room_data,
        'rooms_by_type': rooms_by_type,
        'room_type_counts': room_type_counts,
        'room_config': ROOM_CONFIG,
        'estimated_area': float(total_room_area_sqm),
        'pixels_per_meter': float(calibrator.ppm)
    }

