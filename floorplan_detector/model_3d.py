"""
3D Model Generation Module
===========================
Creates proper 3D models from detected walls and rooms.
Features:
1. Clean wall geometry with proper thickness
2. Room floor generation
3. Wall-wall intersections
4. Export to multiple formats (PLY, OBJ, GLTF)
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    o3d = None  # Placeholder
    logger.warning("Open3D not installed. 3D generation will be limited.")

from .wall_detection import WallSegment
from .room_detection import Room, RoomType


@dataclass
class ModelConfig:
    """Configuration for 3D model generation."""
    wall_height: float = 2.5  # meters
    wall_thickness: float = 0.15  # meters
    floor_thickness: float = 0.1  # meters
    scale_factor: float = 0.01  # pixels to meters
    
    # Colors (RGB 0-1)
    wall_color: Tuple[float, float, float] = (0.95, 0.92, 0.88)  # Off-white
    floor_color: Tuple[float, float, float] = (0.7, 0.6, 0.5)    # Wood-like
    ceiling_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # White


class FloorPlan3DGenerator:
    """
    Generates 3D models from floor plan detection results.
    """
    
    ROOM_COLORS = {
        RoomType.BEDROOM: (0.8, 0.85, 0.95),     # Light blue
        RoomType.BATHROOM: (0.85, 0.95, 0.9),    # Light cyan
        RoomType.KITCHEN: (0.95, 0.9, 0.8),      # Light orange
        RoomType.LIVING_ROOM: (0.9, 0.95, 0.85), # Light green
        RoomType.DINING_ROOM: (0.95, 0.92, 0.85), # Cream
        RoomType.HALLWAY: (0.88, 0.88, 0.88),    # Light gray
        RoomType.UNKNOWN: (0.85, 0.85, 0.85)     # Gray
    }
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize the 3D generator.
        
        Args:
            config: Model configuration
        """
        self.config = config or ModelConfig()
        
        if not HAS_OPEN3D:
            raise ImportError("Open3D is required for 3D model generation. Install with: pip install open3d")
    
    def create_wall_mesh(self, wall: WallSegment, img_shape: Tuple[int, int]) -> o3d.geometry.TriangleMesh:
        """
        Create a 3D mesh for a single wall segment.
        """
        # Convert pixel coordinates to 3D coordinates
        h, w = img_shape
        scale = self.config.scale_factor
        
        # Normalize coordinates (0-1 range, then scale)
        x1 = (wall.x1 / w) * w * scale
        y1 = (wall.y1 / h) * h * scale
        x2 = (wall.x2 / w) * w * scale
        y2 = (wall.y2 / h) * h * scale
        
        # Calculate wall direction and normal
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx*dx + dy*dy)
        
        if length < 0.001:
            return None
        
        # Normalize direction
        dx /= length
        dy /= length
        
        # Normal vector (perpendicular to wall direction)
        nx = -dy
        ny = dx
        
        # Wall half-thickness
        half_thick = self.config.wall_thickness / 2
        wall_h = self.config.wall_height
        
        # Create 8 vertices for wall box
        # Bottom face
        p0 = [x1 + nx * half_thick, 0, y1 + ny * half_thick]
        p1 = [x1 - nx * half_thick, 0, y1 - ny * half_thick]
        p2 = [x2 - nx * half_thick, 0, y2 - ny * half_thick]
        p3 = [x2 + nx * half_thick, 0, y2 + ny * half_thick]
        
        # Top face
        p4 = [x1 + nx * half_thick, wall_h, y1 + ny * half_thick]
        p5 = [x1 - nx * half_thick, wall_h, y1 - ny * half_thick]
        p6 = [x2 - nx * half_thick, wall_h, y2 - ny * half_thick]
        p7 = [x2 + nx * half_thick, wall_h, y2 + ny * half_thick]
        
        vertices = np.array([p0, p1, p2, p3, p4, p5, p6, p7], dtype=np.float64)
        
        # Create triangles for all 6 faces
        triangles = np.array([
            # Bottom
            [0, 2, 1], [0, 3, 2],
            # Top
            [4, 5, 6], [4, 6, 7],
            # Front (outer)
            [0, 4, 7], [0, 7, 3],
            # Back (inner)
            [1, 2, 6], [1, 6, 5],
            # Left end
            [0, 1, 5], [0, 5, 4],
            # Right end
            [3, 7, 6], [3, 6, 2]
        ], dtype=np.int32)
        
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(self.config.wall_color)
        
        return mesh
    
    def create_room_floor(self, room: Room, img_shape: Tuple[int, int]) -> o3d.geometry.TriangleMesh:
        """
        Create a floor mesh for a room.
        """
        if len(room.vertices) < 3:
            return None
        
        h, w = img_shape
        scale = self.config.scale_factor
        
        # Convert room vertices to 3D coordinates
        vertices_3d = []
        for vx, vy in room.vertices:
            x = (vx / w) * w * scale
            z = (vy / h) * h * scale
            vertices_3d.append([x, 0, z])
        
        vertices = np.array(vertices_3d, dtype=np.float64)
        
        # Triangulate the polygon (simple fan triangulation)
        n = len(vertices)
        if n < 3:
            return None
        
        triangles = []
        for i in range(1, n - 1):
            triangles.append([0, i, i + 1])
        
        triangles = np.array(triangles, dtype=np.int32)
        
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.compute_vertex_normals()
        
        # Color by room type
        color = self.ROOM_COLORS.get(room.room_type, self.config.floor_color)
        mesh.paint_uniform_color(color)
        
        return mesh
    
    def create_floor_plane(self, img_shape: Tuple[int, int]) -> o3d.geometry.TriangleMesh:
        """
        Create a base floor plane for the entire floor plan.
        """
        h, w = img_shape
        scale = self.config.scale_factor
        
        # Floor covers entire image area
        vertices = np.array([
            [0, 0, 0],
            [w * scale, 0, 0],
            [w * scale, 0, h * scale],
            [0, 0, h * scale]
        ], dtype=np.float64)
        
        triangles = np.array([
            [0, 1, 2],
            [0, 2, 3]
        ], dtype=np.int32)
        
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(self.config.floor_color)
        
        return mesh
    
    def generate(
        self,
        walls: List[WallSegment],
        rooms: List[Room],
        img_shape: Tuple[int, int],
        add_base_floor: bool = True
    ) -> o3d.geometry.TriangleMesh:
        """
        Generate complete 3D model from walls and rooms.
        
        Args:
            walls: Detected wall segments
            rooms: Detected rooms
            img_shape: Original image shape (height, width)
            add_base_floor: Whether to add a base floor plane
            
        Returns:
            Combined triangle mesh
        """
        meshes = []
        
        # Create base floor
        if add_base_floor:
            floor = self.create_floor_plane(img_shape)
            if floor is not None:
                meshes.append(floor)
        
        # Create walls
        wall_count = 0
        for wall in walls:
            mesh = self.create_wall_mesh(wall, img_shape)
            if mesh is not None:
                meshes.append(mesh)
                wall_count += 1
        
        logger.info(f"Created {wall_count} wall meshes")
        
        # Create room floors (with room-specific colors)
        room_count = 0
        for room in rooms:
            mesh = self.create_room_floor(room, img_shape)
            if mesh is not None:
                # Offset room floor slightly above base floor to avoid z-fighting
                vertices = np.asarray(mesh.vertices)
                vertices[:, 1] += 0.001
                mesh.vertices = o3d.utility.Vector3dVector(vertices)
                meshes.append(mesh)
                room_count += 1
        
        logger.info(f"Created {room_count} room floor meshes")
        
        # Combine all meshes
        if not meshes:
            raise ValueError("No meshes created. Check input data.")
        
        combined = meshes[0]
        for mesh in meshes[1:]:
            combined += mesh
        
        combined.compute_vertex_normals()
        
        return combined
    
    def save(self, mesh: o3d.geometry.TriangleMesh, filepath: str) -> bool:
        """
        Save mesh to file.
        
        Supports: .ply, .obj, .stl, .gltf, .glb
        """
        try:
            success = o3d.io.write_triangle_mesh(
                filepath, 
                mesh, 
                write_vertex_normals=True,
                write_ascii=True
            )
            
            if success:
                logger.info(f"Saved 3D model to: {filepath}")
            else:
                logger.error(f"Failed to save 3D model to: {filepath}")
            
            return success
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False


def generate_3d_model(
    walls: List[WallSegment],
    rooms: List[Room],
    img_shape: Tuple[int, int],
    output_path: str,
    config: Optional[ModelConfig] = None
) -> str:
    """
    Convenience function to generate and save a 3D model.
    
    Args:
        walls: Detected wall segments
        rooms: Detected rooms
        img_shape: Original image shape
        output_path: Path to save the model
        config: Model configuration
        
    Returns:
        Path to saved model
    """
    generator = FloorPlan3DGenerator(config)
    mesh = generator.generate(walls, rooms, img_shape)
    generator.save(mesh, output_path)
    
    return output_path


class SimpleModelGenerator:
    """
    Simpler 3D model generation that doesn't require Open3D.
    Outputs data in a format that can be used with other 3D libraries.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
    
    def generate_wall_vertices(
        self, 
        wall: WallSegment, 
        img_shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate vertices and faces for a wall.
        
        Returns:
            Tuple of (vertices, faces)
        """
        h, w = img_shape
        scale = self.config.scale_factor
        
        x1 = wall.x1 * scale
        y1 = wall.y1 * scale
        x2 = wall.x2 * scale
        y2 = wall.y2 * scale
        
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx*dx + dy*dy)
        
        if length < 0.001:
            return None, None
        
        nx = -dy / length
        ny = dx / length
        
        half_thick = self.config.wall_thickness / 2
        wall_h = self.config.wall_height
        
        vertices = np.array([
            [x1 + nx * half_thick, 0, y1 + ny * half_thick],
            [x1 - nx * half_thick, 0, y1 - ny * half_thick],
            [x2 - nx * half_thick, 0, y2 - ny * half_thick],
            [x2 + nx * half_thick, 0, y2 + ny * half_thick],
            [x1 + nx * half_thick, wall_h, y1 + ny * half_thick],
            [x1 - nx * half_thick, wall_h, y1 - ny * half_thick],
            [x2 - nx * half_thick, wall_h, y2 - ny * half_thick],
            [x2 + nx * half_thick, wall_h, y2 + ny * half_thick]
        ], dtype=np.float64)
        
        faces = np.array([
            [0, 2, 1], [0, 3, 2],
            [4, 5, 6], [4, 6, 7],
            [0, 4, 7], [0, 7, 3],
            [1, 2, 6], [1, 6, 5],
            [0, 1, 5], [0, 5, 4],
            [3, 7, 6], [3, 6, 2]
        ], dtype=np.int32)
        
        return vertices, faces
    
    def to_obj_format(
        self,
        walls: List[WallSegment],
        img_shape: Tuple[int, int]
    ) -> str:
        """
        Generate OBJ file content.
        """
        all_vertices = []
        all_faces = []
        vertex_offset = 0
        
        for wall in walls:
            verts, faces = self.generate_wall_vertices(wall, img_shape)
            if verts is None:
                continue
            
            all_vertices.extend(verts)
            all_faces.extend(faces + vertex_offset)
            vertex_offset += len(verts)
        
        # Generate OBJ content
        obj_lines = ["# Floor Plan 3D Model", "# Generated by FloorPlan3DGenerator", ""]
        
        # Vertices
        for v in all_vertices:
            obj_lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
        
        obj_lines.append("")
        
        # Faces (OBJ uses 1-based indexing)
        for f in all_faces:
            obj_lines.append(f"f {f[0]+1} {f[1]+1} {f[2]+1}")
        
        return "\n".join(obj_lines)
