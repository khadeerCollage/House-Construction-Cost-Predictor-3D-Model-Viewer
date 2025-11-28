"""
Floor Plan 2D to 3D Converter
==============================
Improved version using advanced wall and room detection.
This script replaces the old 2d-3d.py with much better accuracy.

Usage:
    python 2d-3d-improved.py --input <image_path> --output <output.ply> [--view]

Features:
- Advanced wall detection with line merging
- Accurate room detection and classification
- Clean 3D geometry without random blocks
- Proper wall intersections
"""

import argparse
import os
import sys
import cv2
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports (floorplan_detector is in vit_project)
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
logger.info(f"Added to sys.path: {parent_dir}")

# Use the improved FloorPlanAnalyzer for better room classification
from floorplan_detector.floor_plan_analyzer import FloorPlanAnalyzer, RoomLabel, ROOM_NAMES

# Also import SmartWall for type compatibility
from floorplan_detector.smart_detection import SmartWall

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    logger.warning("Open3D not installed. Install with: pip install open3d")


class ImprovedFloorPlan3D:
    """
    Improved 2D to 3D floor plan converter using FloorPlanAnalyzer.
    - Better wall detection (filters noise, merges segments)
    - Smarter room detection (excludes stairs, small boxes)
    - Proper room classification (bedroom, bathroom, kitchen, etc.)
    """
    
    def __init__(
        self,
        wall_height: float = 2.5,
        wall_thickness: float = 0.12,
        scale: float = 0.01
    ):
        """
        Initialize converter.
        
        Args:
            wall_height: Height of walls in meters
            wall_thickness: Thickness of walls in meters
            scale: Scale factor (pixels to meters)
        """
        self.wall_height = wall_height
        self.wall_thickness = wall_thickness
        self.scale = scale
        
        # Use FloorPlanAnalyzer for improved detection and classification
        self.analyzer = FloorPlanAnalyzer()
        
        # Colors for different room types
        self.room_colors = {
            RoomLabel.LIVING_DINING: np.array([0.8, 0.95, 0.8]),   # Light green
            RoomLabel.KITCHEN: np.array([1.0, 0.85, 0.6]),         # Orange
            RoomLabel.BEDROOM: np.array([0.75, 0.85, 1.0]),        # Light blue
            RoomLabel.BATHROOM: np.array([0.7, 1.0, 0.95]),        # Cyan
            RoomLabel.HALLWAY: np.array([0.9, 0.85, 0.95]),        # Light purple
            RoomLabel.STORAGE: np.array([0.85, 0.85, 0.75]),       # Tan
            RoomLabel.GARAGE: np.array([0.7, 0.7, 0.7]),           # Gray
            RoomLabel.STAIRS: np.array([0.9, 0.8, 0.7]),           # Light brown
            RoomLabel.OTHER: np.array([0.88, 0.88, 0.88]),         # Light gray
        }
        
        # Wall and floor colors
        self.wall_color = np.array([0.95, 0.92, 0.88])  # Off-white
        self.floor_color = np.array([0.75, 0.68, 0.58])  # Wood-like
    
    def process(self, image_path: str) -> dict:
        """
        Process floor plan image and detect walls/rooms using FloorPlanAnalyzer.
        
        Args:
            image_path: Path to floor plan image
            
        Returns:
            Dictionary with detection results
        """
        logger.info(f"Processing: {image_path}")
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        self.img_shape = img.shape[:2]
        logger.info(f"Image shape: {self.img_shape}")
        
        # Use FloorPlanAnalyzer for detection
        result = self.analyzer.analyze(img)
        
        logger.info(f"Detected {len(result['walls'])} walls")
        logger.info(f"Detected {len(result['doors'])} doors")
        logger.info(f"Detected {len(result['rooms'])} rooms")
        
        # Log room classification
        for room in result['rooms']:
            logger.info(f"  Room {room.id}: {room.name} ({room.area:.0f}px²)")
        
        return {
            'original': img,
            'binary': result['binary'],
            'walls': result['walls'],
            'doors': result['doors'],
            'rooms': result['rooms'],
            'shape': self.img_shape
        }
    
    def create_wall_mesh(self, wall):
        """Create 3D mesh for a single wall (works with Wall from floor_plan_analyzer)."""
        h, w = self.img_shape
        
        # Convert to 3D coordinates
        x1 = wall.x1 * self.scale
        z1 = wall.y1 * self.scale
        x2 = wall.x2 * self.scale
        z2 = wall.y2 * self.scale
        
        # Direction and normal
        dx = x2 - x1
        dz = z2 - z1
        length = np.sqrt(dx*dx + dz*dz)
        
        if length < 0.001:
            return None
        
        dx /= length
        dz /= length
        
        # Normal (perpendicular)
        nx = -dz
        nz = dx
        
        ht = self.wall_thickness / 2
        wh = self.wall_height
        
        # 8 vertices for wall box
        vertices = np.array([
            # Bottom
            [x1 + nx*ht, 0, z1 + nz*ht],
            [x1 - nx*ht, 0, z1 - nz*ht],
            [x2 - nx*ht, 0, z2 - nz*ht],
            [x2 + nx*ht, 0, z2 + nz*ht],
            # Top
            [x1 + nx*ht, wh, z1 + nz*ht],
            [x1 - nx*ht, wh, z1 - nz*ht],
            [x2 - nx*ht, wh, z2 - nz*ht],
            [x2 + nx*ht, wh, z2 + nz*ht]
        ], dtype=np.float64)
        
        # 12 triangles (2 per face)
        triangles = np.array([
            [0, 2, 1], [0, 3, 2],  # Bottom
            [4, 5, 6], [4, 6, 7],  # Top
            [0, 4, 7], [0, 7, 3],  # Front
            [1, 2, 6], [1, 6, 5],  # Back
            [0, 1, 5], [0, 5, 4],  # Left
            [3, 7, 6], [3, 6, 2]   # Right
        ], dtype=np.int32)
        
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(self.wall_color)
        
        return mesh
    
    def create_floor_mesh(self):
        """Create floor plane."""
        h, w = self.img_shape
        
        vertices = np.array([
            [0, 0, 0],
            [w * self.scale, 0, 0],
            [w * self.scale, 0, h * self.scale],
            [0, 0, h * self.scale]
        ], dtype=np.float64)
        
        triangles = np.array([
            [0, 1, 2],
            [0, 2, 3]
        ], dtype=np.int32)
        
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(self.floor_color)
        
        return mesh
    
    def create_room_floor(self, room):
        """Create colored floor for a room."""
        if room.contour is None or len(room.contour) < 3:
            return None
        
        h, w = self.img_shape
        
        # Get color based on room label
        color = self.room_colors.get(room.label, np.array([0.88, 0.88, 0.88]))
        
        # Simplify contour to polygon
        epsilon = 0.02 * cv2.arcLength(room.contour, True)
        approx = cv2.approxPolyDP(room.contour, epsilon, True)
        vertices_2d = [(int(p[0][0]), int(p[0][1])) for p in approx]
        
        if len(vertices_2d) < 3:
            return None
        
        # Convert to 3D coordinates
        vertices_3d = []
        for vx, vy in vertices_2d:
            vertices_3d.append([vx * self.scale, 0.001, vy * self.scale])
        
        vertices = np.array(vertices_3d, dtype=np.float64)
        
        # Triangulate using fan triangulation
        n = len(vertices)
        triangles = []
        for i in range(1, n - 1):
            triangles.append([0, i, i + 1])
        
        if not triangles:
            return None
        
        triangles = np.array(triangles, dtype=np.int32)
        
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(color)
        
        return mesh
    
    def generate_3d(self, detection_result: dict) -> o3d.geometry.TriangleMesh:
        """
        Generate 3D model from detection results.
        
        Args:
            detection_result: Output from process()
            
        Returns:
            Combined triangle mesh
        """
        if not HAS_OPEN3D:
            raise ImportError("Open3D required for 3D generation")
        
        meshes = []
        
        # Floor
        floor = self.create_floor_mesh()
        meshes.append(floor)
        
        # Walls
        wall_count = 0
        for wall in detection_result['walls']:
            mesh = self.create_wall_mesh(wall)
            if mesh is not None:
                meshes.append(mesh)
                wall_count += 1
        
        logger.info(f"Created {wall_count} wall meshes")
        
        # Room floors
        room_count = 0
        for room in detection_result['rooms']:
            mesh = self.create_room_floor(room)
            if mesh is not None:
                meshes.append(mesh)
                room_count += 1
        
        logger.info(f"Created {room_count} room floor meshes")
        
        # Combine
        combined = meshes[0]
        for mesh in meshes[1:]:
            combined += mesh
        
        combined.compute_vertex_normals()
        
        return combined
    
    def save(self, mesh, output_path: str):
        """Save mesh to file."""
        success = o3d.io.write_triangle_mesh(
            output_path,
            mesh,
            write_vertex_normals=True,
            write_ascii=True
        )
        
        if success:
            logger.info(f"Saved 3D model to: {output_path}")
        else:
            raise RuntimeError(f"Failed to save: {output_path}")
        
        return success
    
    def run(self, image_path: str, output_path: str, view: bool = False):
        """
        Complete pipeline: process image and generate 3D model.
        
        Args:
            image_path: Input floor plan image
            output_path: Output PLY file path
            view: Whether to open 3D viewer
        """
        # Process image
        result = self.process(image_path)
        
        # Generate 3D
        mesh = self.generate_3d(result)
        
        # Save
        self.save(mesh, output_path)
        
        # View if requested
        if view:
            logger.info("Opening 3D viewer...")
            o3d.visualization.draw_geometries(
                [mesh],
                window_name="Floor Plan 3D Model",
                width=1024,
                height=768
            )
        
        # Print summary
        print("\n" + "="*50)
        print("FLOOR PLAN ANALYSIS COMPLETE")
        print("="*50)
        print(f"Walls detected: {len(result['walls'])}")
        print(f"Rooms detected: {len(result['rooms'])}")
        
        # Room breakdown using room name
        room_types = {}
        for room in result['rooms']:
            rt = room.name  # Use name property from Room
            room_types[rt] = room_types.get(rt, 0) + 1
        
        print("\nRoom breakdown:")
        for rt, count in room_types.items():
            print(f"  {rt}: {count}")
        
        print(f"\n3D model saved to: {output_path}")
        print("="*50)
        
        return result


def main():
    parser = argparse.ArgumentParser(
        description="Convert 2D floor plan to 3D model with accurate detection"
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input floor plan image path'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output PLY file path'
    )
    parser.add_argument(
        '--view',
        action='store_true',
        help='Open 3D viewer after generation'
    )
    parser.add_argument(
        '--wall-height',
        type=float,
        default=2.5,
        help='Wall height in meters (default: 2.5)'
    )
    parser.add_argument(
        '--wall-thickness',
        type=float,
        default=0.12,
        help='Wall thickness in meters (default: 0.12)'
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.isfile(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Run pipeline
    converter = ImprovedFloorPlan3D(
        wall_height=args.wall_height,
        wall_thickness=args.wall_thickness
    )
    
    converter.run(args.input, args.output, view=args.view)


if __name__ == "__main__":
    main()
