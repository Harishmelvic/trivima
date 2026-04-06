"""Convert Lyra Gaussian PLY to textured triangle mesh via Poisson reconstruction."""
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

opacity = 1.0 / (1.0 + np.exp(-np.clip(np.array(v["opacity"]), -20, 20)))
mask = opacity > 0.3
positions = positions[mask]
colors = colors[mask]
print("After opacity filter: %d points" % len(positions))

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(positions.astype(np.float64))
pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

print("Estimating normals...")
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
pcd.orient_normals_consistent_tangent_plane(k=15)

print("Poisson reconstruction (depth=9)...")
t0 = time.time()
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9, width=0, scale=1.1)
dt = time.time() - t0
print("Mesh: %d vertices, %d triangles (%.1fs)" % (len(mesh.vertices), len(mesh.triangles), dt))

densities = np.asarray(densities)
threshold = np.quantile(densities, 0.05)
mesh.remove_vertices_by_mask(densities < threshold)
print("After cleanup: %d vertices, %d triangles" % (len(mesh.vertices), len(mesh.triangles)))

print("Transferring colors to mesh...")
mesh_verts = np.asarray(mesh.vertices)
pcd_tree = o3d.geometry.KDTreeFlann(pcd)
vcols = np.zeros((len(mesh_verts), 3))
for i in range(len(mesh_verts)):
    _, idx, _ = pcd_tree.search_knn_vector_3d(mesh_verts[i], 3)
    vcols[i] = np.mean(np.asarray(pcd.colors)[idx], axis=0)
mesh.vertex_colors = o3d.utility.Vector3dVector(vcols)

o3d.io.write_triangle_mesh("/workspace/lyra/room_mesh.obj", mesh)
o3d.io.write_triangle_mesh("/workspace/lyra/room_mesh.ply", mesh)
print("OBJ: %.1fMB" % (os.path.getsize("/workspace/lyra/room_mesh.obj") / 1e6))
print("PLY: %.1fMB" % (os.path.getsize("/workspace/lyra/room_mesh.ply") / 1e6))
print("Done!")
