# Step 2: Import dependencies
import torch
import cv2
import numpy as np
import open3d as o3d
import gc
import sys
import os
import argparse

# Parse command-line arguments for input and output paths
parser = argparse.ArgumentParser(description="2D floorplan to 3D model generator")
parser.add_argument('--input', type=str, required=True, help='Input image path')
parser.add_argument('--output', type=str, required=True, help='Output .ply file path')
parser.add_argument('--view', action='store_true', help='Open 3D viewer after generation')
args = parser.parse_args()

image_path = args.input
ply_path = args.output

if not os.path.isfile(image_path):
    raise ValueError(f"Failed to load image: {image_path}")

img = cv2.imread(image_path)
if img is None:
    raise ValueError(f"Failed to load image: {image_path}")

# Step 3: Load MiDaS model (MiDaS_small for lower memory usage)
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device).eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

# Step 4: Downscale image
scale_factor = 512 / max(img.shape[:2])
img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

# Step 5: Preprocess the floor plan image for MiDaS
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_rgb = cv2.bitwise_not(img_rgb)
img_rgb = cv2.convertScaleAbs(img_rgb, alpha=1.5, beta=0)
img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

# Step 6: Generate depth map with MiDaS
input_tensor = transform(img_bgr)
if len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1:
    input_tensor = input_tensor.squeeze(0)
input_batch = input_tensor.unsqueeze(0).to(device)

with torch.no_grad():
    prediction = midas(input_batch)

prediction = torch.nn.functional.interpolate(
    prediction.unsqueeze(1),
    size=img.shape[:2],
    mode="bicubic",
    align_corners=False
).squeeze()

depth_map = prediction.cpu().numpy()
depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

# Clean up memory
del input_batch, prediction, input_tensor
if device.type == "cuda":
    torch.cuda.empty_cache()
gc.collect()

# Step 7: Advanced wall detection with refined line filtering
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)
kernel = np.ones((3, 3), np.uint8)
thresh = cv2.dilate(thresh, kernel, iterations=1)
edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=10, maxLineGap=5)

# Step 8: Filter unnecessary lines while preserving main walls
filtered_lines = []
if lines is not None:
    line_list = [(line[0][0], line[0][1], line[0][2], line[0][3]) for line in lines]
    
    def are_lines_redundant(line1, line2, dist_threshold=5, angle_threshold=np.pi/18):
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        angle1 = np.arctan2(y2 - y1, x2 - x1)
        angle2 = np.arctan2(y4 - y3, x4 - x3)
        angle_diff = min(abs(angle1 - angle2), np.pi - abs(angle1 - angle2))
        mid1_x, mid1_y = (x1 + x2) / 2, (y1 + y2) / 2
        mid2_x, mid2_y = (x3 + x4) / 2, (y3 + y4) / 2
        dist = np.sqrt((mid1_x - mid2_x)**2 + (mid1_y - mid2_y)**2)
        return angle_diff < angle_threshold and dist < dist_threshold

    lengths = [np.sqrt((line[2] - line[0])**2 + (line[3] - line[1])**2) for line in line_list]
    length_threshold = np.percentile(lengths, 75)
    main_walls = []
    other_lines = []
    for i, line in enumerate(line_list):
        length = lengths[i]
        if length > length_threshold:
            main_walls.append(line)
        else:
            other_lines.append((line, i))
    
    filtered_lines.extend(main_walls)
    keep_indices = set()
    for i, (line, idx) in enumerate(other_lines):
        keep = True
        for j, (other_line, other_idx) in enumerate(other_lines[:i]):
            if other_idx in keep_indices and are_lines_redundant(line, other_line):
                keep = False
                break
        if keep:
            keep_indices.add(idx)
            filtered_lines.append(line)

# Step 9: Create geometries for walls
wall_height = 0.2
wall_thickness = 0.005
geometries = []
material_names = []

for x1, y1, x2, y2 in filtered_lines:
    depth_values = [depth_map[y1, x1], depth_map[y2, x2]]
    avg_depth = np.mean(depth_values) * 0.1

    x1n, y1n = x1 / img.shape[1], y1 / img.shape[0]
    x2n, y2n = x2 / img.shape[1], y2 / img.shape[0]

    dx, dy = x2n - x1n, y2n - y1n
    length = np.sqrt(dx**2 + dy**2)
    if length == 0:
        continue
    nx, ny = -dy / length, dx / length

    offset = wall_thickness / 2
    p1 = [x1n + nx * offset, y1n + ny * offset, avg_depth]
    p2 = [x2n + nx * offset, y2n + ny * offset, avg_depth]
    p3 = [x2n - nx * offset, y2n - ny * offset, avg_depth]
    p4 = [x1n - nx * offset, y1n - ny * offset, avg_depth]
    p5 = [x1n + nx * offset, y1n + ny * offset, avg_depth + wall_height]
    p6 = [x2n + nx * offset, y2n + ny * offset, avg_depth + wall_height]
    p7 = [x2n - nx * offset, y2n - ny * offset, avg_depth + wall_height]
    p8 = [x1n - nx * offset, y1n - ny * offset, avg_depth + wall_height]

    vertices = np.array([p1, p2, p3, p4, p5, p6, p7, p8], dtype=np.float64)
    triangles = np.array([
        [0, 1, 2], [0, 2, 3],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6],
        [3, 0, 4], [3, 4, 7]
    ], dtype=np.int32)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(np.array([0.96, 0.87, 0.70]))
    geometries.append(mesh)
    material_names.append("WallMaterial")

# Step 10: Room detection and labeling
_, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Room classification based on size and door/window presence
rooms = []
for contour in contours:
    area = cv2.contourArea(contour)
    if 100 < area < 2000:  # Adjusted for downscaled image
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        if 0.1 < aspect_ratio < 6.0:
            depth_region = depth_map[y:y+h, x:x+w]
            avg_depth = np.mean(depth_region) * 0.1

            xn, yn = x / img.shape[1], y / img.shape[0]
            wn, hn = w / img.shape[1], h / img.shape[0]

            # Classify room type
            is_bathroom = 200 < area < 500 and 0.5 < aspect_ratio < 2.0
            element_height = 0.15 if aspect_ratio < 0.8 else 0.1

            # Heuristics for other rooms based on area and aspect ratio
            if is_bathroom:
                room_type = "Bathroom"
                color = np.array([0.7, 0.9, 0.7])
                mat_name = "BathroomWallMaterial"
            elif area > 1000 and aspect_ratio > 1.5:  # Large, wide rooms
                room_type = "Living Room"
                color = np.array([0.9, 0.8, 0.8])
                mat_name = "LivingRoomMaterial"
            elif 500 < area < 1000 and 0.8 < aspect_ratio < 1.5:  # Medium, square-ish rooms
                room_type = "Bedroom"
                color = np.array([0.8, 0.8, 0.9])
                mat_name = "BedroomMaterial"
            elif area < 300 and aspect_ratio > 2.0:  # Small, narrow rooms
                room_type = "Kitchen"
                color = np.array([0.9, 0.9, 0.7])
                mat_name = "KitchenMaterial"
            else:
                room_type = "Other Room"
                color = np.array([0.85, 0.85, 0.85])
                mat_name = "OtherRoomMaterial"

            # Add room walls (for bathrooms, as before)
            if is_bathroom:
                points = [(xn, yn), (xn + wn, yn), (xn + wn, yn + hn), (xn, yn + hn)]
                for i in range(len(points)):
                    x1n, y1n = points[i]
                    x2n, y2n = points[(i + 1) % len(points)]
                    dx, dy = x2n - x1n, y2n - y1n
                    length = np.sqrt(dx**2 + dy**2)
                    if length == 0:
                        continue
                    nx, ny = -dy / length, dx / length

                    offset = wall_thickness / 2
                    p1 = [x1n + nx * offset, y1n + ny * offset, avg_depth]
                    p2 = [x2n + nx * offset, y2n + ny * offset, avg_depth]
                    p3 = [x2n - nx * offset, y2n - ny * offset, avg_depth]
                    p4 = [x1n - nx * offset, y1n - ny * offset, avg_depth]
                    p5 = [x1n + nx * offset, y1n + ny * offset, avg_depth + wall_height]
                    p6 = [x2n + nx * offset, y2n + ny * offset, avg_depth + wall_height]
                    p7 = [x2n - nx * offset, y2n - ny * offset, avg_depth + wall_height]
                    p8 = [x1n - nx * offset, y1n - ny * offset, avg_depth + wall_height]

                    vertices = np.array([p1, p2, p3, p4, p5, p6, p7, p8], dtype=np.float64)
                    triangles = np.array([
                        [0, 1, 2], [0, 2, 3],
                        [4, 5, 6], [4, 6, 7],
                        [0, 1, 5], [0, 5, 4],
                        [1, 2, 6], [1, 6, 5],
                        [2, 3, 7], [2, 7, 6],
                        [3, 0, 4], [3, 4, 7]
                    ], dtype=np.int32)

                    mesh = o3d.geometry.TriangleMesh()
                    mesh.vertices = o3d.utility.Vector3dVector(vertices)
                    mesh.triangles = o3d.utility.Vector3iVector(triangles)
                    mesh.compute_vertex_normals()
                    mesh.paint_uniform_color(color)
                    geometries.append(mesh)
                    material_names.append(mat_name)

            # Store room info for furniture placement and labeling
            rooms.append({
                "type": room_type,
                "center": (xn + wn / 2, yn + hn / 2, avg_depth),
                "width": wn,
                "height": hn,
                "color": color,
                "material": mat_name
            })

# Step 12: Create floor plane
h, w = img.shape[:2]
floor_vertices = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0]
], dtype=np.float64)
floor_triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

floor_mesh = o3d.geometry.TriangleMesh()
floor_mesh.vertices = o3d.utility.Vector3dVector(floor_vertices)
floor_mesh.triangles = o3d.utility.Vector3iVector(floor_triangles)
floor_mesh.compute_vertex_normals()
floor_mesh.paint_uniform_color(np.array([0.95, 0.92, 0.88]))
geometries.append(floor_mesh)
material_names.append("FloorMaterial")

# Step 13: Save to PLY file with enhanced validation
# Validate geometries before combining
if not geometries:
    raise ValueError("No geometries to combine. Ensure the floor plan processing generated valid meshes.")

combined_geometry = geometries[0]
for i, geom in enumerate(geometries[1:]):
    # Validate each geometry
    vertices = np.asarray(geom.vertices)
    triangles = np.asarray(geom.triangles)
    if vertices.size == 0 or triangles.size == 0:
        print(f"Skipping invalid geometry at index {i+1}: {geom}")
        continue
    print(f"Combining geometry {i+1}: {vertices.shape[0]} vertices, {triangles.shape[0]} triangles")
    combined_geometry += geom

# Validate combined geometry
vertices = np.asarray(combined_geometry.vertices)
triangles = np.asarray(combined_geometry.triangles)
if vertices.size == 0 or triangles.size == 0:
    raise ValueError(f"Combined geometry is invalid. Vertices: {vertices.shape}, Triangles: {triangles.shape}")

print(f"Final combined geometry: {vertices.shape[0]} vertices, {triangles.shape[0]} triangles")
combined_geometry.compute_vertex_normals()

# When saving the PLY file, use the output path provided
ply_success = o3d.io.write_triangle_mesh(ply_path, combined_geometry, write_vertex_normals=True, write_ascii=True)

if not ply_success:
    obj_path = ply_path.replace('.ply', '.obj')
    obj_success = o3d.io.write_triangle_mesh(obj_path, combined_geometry)
    if not obj_success:
        raise RuntimeError("Failed to write both PLY and OBJ files. Check geometry data or disk permissions.")
    print(f"3D model saved as '{obj_path}'. Update the Blender script to import OBJ instead.")
else:
    print(f"3D model saved as '{ply_path}'.")

# Open 3D viewer if requested
if args.view:
    mesh = o3d.io.read_triangle_mesh(ply_path)
    o3d.visualization.draw_geometries([mesh], window_name="3D Floorplan Viewer")