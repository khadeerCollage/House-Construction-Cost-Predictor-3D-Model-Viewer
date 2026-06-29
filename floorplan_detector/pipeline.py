"""
Floor Plan Analysis Pipeline
=============================
Main entry point for floor plan analysis.
Combines preprocessing, wall detection, room detection, and 3D generation.
"""

import cv2
import numpy as np
import os
import json
import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, asdict

from .preprocessing import FloorPlanPreprocessor, preprocess_floorplan
from .wall_detection import WallDetector, WallSegment, detect_walls, walls_to_image
from .room_detection import RoomDetector, Room, RoomType, detect_rooms, rooms_to_image

try:
    from .model_3d import FloorPlan3DGenerator, ModelConfig, generate_3d_model
    HAS_3D = True
except ImportError:
    HAS_3D = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Results from floor plan analysis."""
    success: bool
    walls: List[WallSegment]
    rooms: List[Room]
    wall_count: int
    room_count: int
    room_breakdown: Dict[str, int]
    visualization_path: Optional[str] = None
    model_path: Optional[str] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'success': self.success,
            'wall_count': self.wall_count,
            'room_count': self.room_count,
            'room_breakdown': self.room_breakdown,
            'visualization_path': self.visualization_path,
            'model_path': self.model_path,
            'error': self.error,
            'walls': [
                {
                    'x1': w.x1, 'y1': w.y1, 'x2': w.x2, 'y2': w.y2,
                    'length': w.length, 'orientation': w.orientation
                }
                for w in self.walls
            ],
            'rooms': [
                {
                    'type': r.room_type.value,
                    'label': r.label,
                    'area': r.area,
                    'center': r.center,
                    'bounding_box': r.bounding_box
                }
                for r in self.rooms
            ]
        }


class FloorPlanAnalyzer:
    """
    Complete floor plan analysis pipeline.
    """
    
    def __init__(
        self,
        target_size: int = 800,
        min_wall_length: int = 30,
        min_room_area: int = 2000,
        wall_height: float = 2.5,
        wall_thickness: float = 0.15
    ):
        """
        Initialize the analyzer.
        
        Args:
            target_size: Target image size for processing
            min_wall_length: Minimum wall segment length
            min_room_area: Minimum room area
            wall_height: Wall height for 3D model (meters)
            wall_thickness: Wall thickness for 3D model (meters)
        """
        self.preprocessor = FloorPlanPreprocessor(target_size=target_size)
        self.wall_detector = WallDetector(min_wall_length=min_wall_length)
        self.room_detector = RoomDetector(min_room_area=min_room_area)
        
        if HAS_3D:
            self.model_config = ModelConfig(
                wall_height=wall_height,
                wall_thickness=wall_thickness
            )
            self.model_generator = FloorPlan3DGenerator(self.model_config)
        else:
            self.model_generator = None
    
    def analyze(
        self,
        image_path: str,
        output_dir: Optional[str] = None,
        generate_3d: bool = True,
        save_visualization: bool = True
    ) -> AnalysisResult:
        """
        Analyze a floor plan image.
        
        Args:
            image_path: Path to floor plan image
            output_dir: Directory for output files
            generate_3d: Whether to generate 3D model
            save_visualization: Whether to save visualization images
            
        Returns:
            AnalysisResult with detection results
        """
        try:
            logger.info(f"Analyzing floor plan: {image_path}")
            
            # Create output directory
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Step 1: Preprocess image
            logger.info("Step 1: Preprocessing image...")
            original, processed, binary, scale = self.preprocessor.preprocess(
                image_path, 
                remove_text=True
            )
            img_shape = original.shape[:2]
            
            # Step 2: Detect walls
            logger.info("Step 2: Detecting walls...")
            walls = self.wall_detector.detect(binary)
            
            # Step 3: Detect rooms
            logger.info("Step 3: Detecting rooms...")
            rooms = self.room_detector.detect_from_walls(walls, img_shape)
            
            # Count rooms by type
            room_breakdown = {}
            for room in rooms:
                rt = room.room_type.value
                room_breakdown[rt] = room_breakdown.get(rt, 0) + 1
            
            # Step 4: Generate visualization
            vis_path = None
            if save_visualization and output_dir:
                vis_path = self._save_visualization(
                    original, walls, rooms, output_dir, image_path
                )
            
            # Step 5: Generate 3D model
            model_path = None
            if generate_3d and self.model_generator and output_dir:
                logger.info("Step 5: Generating 3D model...")
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                model_path = os.path.join(output_dir, f"{base_name}_3d.ply")
                
                try:
                    mesh = self.model_generator.generate(walls, rooms, img_shape)
                    self.model_generator.save(mesh, model_path)
                except Exception as e:
                    logger.error(f"3D generation failed: {e}")
                    model_path = None
            
            result = AnalysisResult(
                success=True,
                walls=walls,
                rooms=rooms,
                wall_count=len(walls),
                room_count=len(rooms),
                room_breakdown=room_breakdown,
                visualization_path=vis_path,
                model_path=model_path
            )
            
            logger.info(f"Analysis complete: {len(walls)} walls, {len(rooms)} rooms")
            
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            
            return AnalysisResult(
                success=False,
                walls=[],
                rooms=[],
                wall_count=0,
                room_count=0,
                room_breakdown={},
                error=str(e)
            )
    
    def _save_visualization(
        self,
        original: np.ndarray,
        walls: List[WallSegment],
        rooms: List[Room],
        output_dir: str,
        image_path: str
    ) -> str:
        """Save visualization of detected walls and rooms."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Floor Plan')
        axes[0].axis('off')
        
        # Wall detection
        wall_img = original.copy()
        for wall in walls:
            color = (0, 255, 0) if wall.orientation == "horizontal" else (0, 0, 255)
            cv2.line(wall_img, (wall.x1, wall.y1), (wall.x2, wall.y2), color, 2)
        axes[1].imshow(cv2.cvtColor(wall_img, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f'Detected Walls ({len(walls)})')
        axes[1].axis('off')
        
        # Room detection with labels
        room_img = original.copy()
        colors = {
            RoomType.BEDROOM: (255, 255, 150),
            RoomType.BATHROOM: (150, 150, 255),
            RoomType.KITCHEN: (150, 200, 255),
            RoomType.LIVING_ROOM: (200, 255, 200),
            RoomType.DINING_ROOM: (255, 200, 150),
            RoomType.HALLWAY: (200, 200, 200),
            RoomType.UNKNOWN: (180, 180, 180)
        }
        
        overlay = room_img.copy()
        for room in rooms:
            color = colors.get(room.room_type, (180, 180, 180))
            cv2.drawContours(overlay, [room.contour], -1, color, -1)
        
        room_img = cv2.addWeighted(room_img, 0.4, overlay, 0.6, 0)
        
        for room in rooms:
            cv2.drawContours(room_img, [room.contour], -1, (0, 0, 0), 2)
            cx, cy = int(room.center[0]), int(room.center[1])
            cv2.putText(
                room_img, room.label,
                (cx - 40, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1
            )
        
        axes[2].imshow(cv2.cvtColor(room_img, cv2.COLOR_BGR2RGB))
        axes[2].set_title(f'Detected Rooms ({len(rooms)})')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        vis_path = os.path.join(output_dir, f"{base_name}_analysis.png")
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved visualization to: {vis_path}")
        return vis_path


def analyze_floorplan(
    image_path: str,
    output_dir: str = None,
    generate_3d: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function for floor plan analysis.
    
    Args:
        image_path: Path to floor plan image
        output_dir: Output directory (default: same as image)
        generate_3d: Whether to generate 3D model
        **kwargs: Additional parameters for FloorPlanAnalyzer
        
    Returns:
        Dictionary with analysis results
    """
    if output_dir is None:
        output_dir = os.path.dirname(image_path)
    
    analyzer = FloorPlanAnalyzer(**kwargs)
    result = analyzer.analyze(
        image_path,
        output_dir=output_dir,
        generate_3d=generate_3d
    )
    
    return result.to_dict()


# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Floor Plan Analysis")
    parser.add_argument("--input", "-i", required=True, help="Input image path")
    parser.add_argument("--output", "-o", help="Output directory")
    parser.add_argument("--no-3d", action="store_true", help="Skip 3D generation")
    parser.add_argument("--wall-height", type=float, default=2.5, help="Wall height (meters)")
    parser.add_argument("--min-wall-length", type=int, default=30, help="Minimum wall length")
    
    args = parser.parse_args()
    
    result = analyze_floorplan(
        args.input,
        output_dir=args.output,
        generate_3d=not args.no_3d,
        wall_height=args.wall_height,
        min_wall_length=args.min_wall_length
    )
    
    print(json.dumps(result, indent=2))
