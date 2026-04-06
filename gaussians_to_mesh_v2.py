"""Convert Lyra Gaussian PLY to mesh - use ALL points (low opacity threshold)."""
import numpy as np
import open3d as o3d
from plyfile import PlyData
import time, os

print("Loading PLY...")
ply = PlyData.read("/workspace/lyra/outputs/demo/lyra_room/static_view_indices_fixed_0_1/room_photo/gaussians_orig/gaussians_0.ply")
v = ply["vertex"]
n = len(v)
print("Loaded %d Gaussians" % n)

positions = np.stack([np.array(v["x"]), np.array(v["y"]), np.array(v["z"])], axis=-1)
C0 = 0.28209479177387814
colors = np.clip(np.stack([
    np.array(v["f_dc_0"]) * C0 + 0.5,
    np.array(v["f_dc_1"]) * C0 + 0.5,
    np.array(v["f_dc_2"]) * C0 + 0.5,
], axis=-1), 0, 1)

# Use very low threshold - keep most points
opacity = 1.0 / (1.0 + np.exp(-np.clip(np.array(v["opacity"]), -20, 20)))
mask = opacity > 0.02
positions = positions[mask]
colors = colors[mask]
print("After filter (>0.02): %d points" % len(positions))

# Subsample for speed if too many
if len(positions) > 300000:
    idx = np.random.RandomState(42).choice(len(positions), 300000, replace=False)
    positions = positions[idx]
    colors = colors[idx]
    print("Subsampled to 300K")

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(positions.astype(np.float64))
pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

# Remove outliers
pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
print("After outlier removal: %d points" % len(pcd.points))

print("Estimating normals...")
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
pcd.orient_normals_consistent_tangent_plane(k=15)

print("Poisson reconstruction (depth=8)...")
t0 = time.time()
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1)
dt = time.time() - t0
print("Mesh: %d vertices, %d triangles (%.1fs)" % (len(mesh.vertices), len(mesh.triangles), dt))

# Remove low-density (noisy) regions
densities = np.asarray(densities)
threshold = np.quantile(densities, 0.02)
mesh.remove_vertices_by_mask(densities < threshold)
print("After cleanup: %d vertices, %d triangles" % (len(mesh.vertices), len(mesh.triangles)))

# Transfer colors
print("Transferring colors...")
mesh_verts = np.asarray(mesh.vertices)
pcd_tree = o3d.geometry.KDTreeFlann(pcd)
vcols = np.zeros((len(mesh_verts), 3))
for i in range(len(mesh_verts)):
    _, idx, _ = pcd_tree.search_knn_vector_3d(mesh_verts[i], 5)
    vcols[i] = np.mean(np.asarray(pcd.colors)[idx], axis=0)
mesh.vertex_colors = o3d.utility.Vector3dVector(vcols)

# Save
o3d.io.write_triangle_mesh("/workspace/lyra/room_mesh_v2.obj", mesh)
o3d.io.write_triangle_mesh("/workspace/lyra/room_mesh_v2.ply", mesh)

# Also save point cloud as PLY for Blender
o3d.io.write_point_cloud("/workspace/lyra/room_points.ply", pcd)

print("OBJ: %.1fMB" % (os.path.getsize("/workspace/lyra/room_mesh_v2.obj") / 1e6))
print("Points PLY: %.1fMB" % (os.path.getsize("/workspace/lyra/room_points.ply") / 1e6))
print("Done!")
