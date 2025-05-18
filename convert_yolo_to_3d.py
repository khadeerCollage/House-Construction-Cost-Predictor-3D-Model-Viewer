# Step 1: Import dependencies
import torch
import cv2
import numpy as np
import open3d as o3d

# Step 2: Load MiDaS model and transforms
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device).eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform

# Step 3: Load image from local path
image_path = "high_quality_3736_F1_original.png"
img = cv2.imread(image_path)
if img is None:
    raise ValueError(f"Failed to load image: {image_path}")

# Step 4: Preprocess the floor plan image for MiDaS
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_rgb = cv2.bitwise_not(img_rgb)
img_rgb = cv2.convertScaleAbs(img_rgb, alpha=1.5, beta=0)
img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

# Step 5: Generate depth map with MiDaS
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

# Step 6: Extract walls using image processing
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

# Step 7: Create 3D geometry for walls
wall_height = 0.5
geometries = []

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        depth_values = [depth_map[y1, x1], depth_map[y2, x2]]
        avg_depth = np.mean(depth_values)

        p1 = [x1 / img.shape[1], y1 / img.shape[0], avg_depth]
        p2 = [x2 / img.shape[1], y2 / img.shape[0], avg_depth]
        p3 = [x2 / img.shape[1], y2 / img.shape[0], avg_depth + wall_height]
        p4 = [x1 / img.shape[1], y1 / img.shape[0], avg_depth + wall_height]

        vertices = np.array([p1, p2, p3, p4], dtype=np.float64)
        triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.paint_uniform_color([0.5, 0.5, 0.5])
        geometries.append(mesh)

# Step 8: Extract doors and windows
_, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    area = cv2.contourArea(contour)
    if 500 < area < 5000:
        x, y, w, h = cv2.boundingRect(contour)
        depth_region = depth_map[y:y+h, x:x+w]
        avg_depth = np.mean(depth_region)

        door_height = 0.3
        p1 = [x / img.shape[1], y / img.shape[0], avg_depth]
        p2 = [(x + w) / img.shape[1], y / img.shape[0], avg_depth]
        p3 = [(x + w) / img.shape[1], (y + h) / img.shape[0], avg_depth]
        p4 = [x / img.shape[1], (y + h) / img.shape[0], avg_depth]
        p5 = [x / img.shape[1], y / img.shape[0], avg_depth + door_height]
        p6 = [(x + w) / img.shape[1], y / img.shape[0], avg_depth + door_height]
        p7 = [(x + w) / img.shape[1], (y + h) / img.shape[0], avg_depth + door_height]
        p8 = [x / img.shape[1], (y + h) / img.shape[0], avg_depth + door_height]

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
        mesh.paint_uniform_color([0.0, 0.0, 1.0])
        geometries.append(mesh)

# Step 9: Create floor plane
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
floor_mesh.paint_uniform_color([0.8, 0.8, 0.8])
geometries.append(floor_mesh)

# Step 10: Combine geometries and save to file
combined_geometry = geometries[0]
for geom in geometries[1:]:
    combined_geometry += geom

# Save the 3D model to a PLY file
o3d.io.write_triangle_mesh("floorplan_3d_model.ply", combined_geometry)
print("3D model saved as 'floorplan_3d_model.ply'. Visualize it on a system with a display.")