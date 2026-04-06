from plyfile import PlyData
import numpy as np

ply = PlyData.read("/workspace/lyra/outputs/demo/lyra_room/static_view_indices_fixed_0_1/room_photo/gaussians_orig/gaussians_0.ply")
v = ply["vertex"]
opacity = 1.0 / (1.0 + np.exp(-np.clip(np.array(v["opacity"]), -20, 20)))

for thresh in [0.05, 0.1, 0.2]:
    mask = opacity > thresh
    x, y, z = np.array(v["x"])[mask], np.array(v["y"])[mask], np.array(v["z"])[mask]
    print("Opacity > %.2f: %d points" % (thresh, mask.sum()))
    print("  X: [%.2f, %.2f]" % (x.min(), x.max()))
    print("  Y: [%.2f, %.2f]" % (y.min(), y.max()))
    print("  Z: [%.2f, %.2f]" % (z.min(), z.max()))
    print("  Center: (%.2f, %.2f, %.2f)" % (x.mean(), y.mean(), z.mean()))
