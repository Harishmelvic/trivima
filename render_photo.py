#!/usr/bin/env python3
"""
Trivima — Photo to 3D Render
=============================
One script: photo → Depth Pro → cells → render from original camera position.

Usage:
    python render_photo.py --image room.jpg
    python render_photo.py --image room.jpg --views 16 --output renders/

The camera starts at (0,0,0) looking into the scene — exactly where
the photo was taken from. The first render should match the original photo.

Requires: Depth Pro, moderngl, PIL, numpy
GPU: A40/A100/H100 with EGL support
"""

import argparse
import os
import sys
import time
import math
import numpy as np
from pathlib import Path
from PIL import Image


def run_perception(image_path: str, device: str = "cuda"):
    """Run Depth Pro on photo → depth map + focal length."""
    import torch

    print(f"[1/4] Loading Depth Pro...")
    os.environ["PYOPENGL_PLATFORM"] = "egl"

    from trivima.perception.depth_pro import DepthProEstimator
    from trivima.perception.depth_smoothing import bilateral_depth_smooth

    model = DepthProEstimator(device=device)
    model.load()

    image = np.array(Image.open(image_path).convert("RGB"))
    h, w = image.shape[:2]

    print(f"[2/4] Running depth estimation on {w}x{h} image...")
    result = model.estimate(image)
    depth = result["depth"]
    focal = result["focal_length"]

    model.unload()
    torch.cuda.empty_cache()

    # Bilateral smoothing
    smoothed = bilateral_depth_smooth(depth, image, spatial_sigma=2.5, color_sigma=25.0)

    print(f"  Depth range: {smoothed[smoothed>0].min():.2f}m - {smoothed.max():.2f}m")
    print(f"  Focal length: {focal:.1f}px")

    return image, smoothed, focal, h, w


def build_cells(image, depth, focal, h, w, cell_size=0.05):
    """Backproject depth to 3D points → build cell grid."""
    print(f"[3/4] Building cell grid (cell_size={cell_size}m)...")

    cx, cy = w / 2.0, h / 2.0

    # Create pixel grid
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    # Valid depth mask
    valid = depth > 0.1

    # Backproject: camera at origin, looking along +Z
    x = (u - cx) * depth / focal
    y = -(v - cy) * depth / focal  # flip Y (image Y is down, world Y is up)
    z = depth

    # Extract valid points
    positions = np.stack([x[valid], y[valid], z[valid]], axis=-1).astype(np.float32)
    colors = image[valid].astype(np.float32) / 255.0

    print(f"  Points: {len(positions):,}")

    # Bin into cells
    cell_indices = np.floor(positions / cell_size).astype(np.int32)
    cells = {}

    for i in range(len(positions)):
        key = tuple(cell_indices[i])
        if key not in cells:
            cells[key] = {"pos_sum": np.zeros(3), "col_sum": np.zeros(3), "count": 0}
        cells[key]["pos_sum"] += positions[i]
        cells[key]["col_sum"] += colors[i]
        cells[key]["count"] += 1

    # Finalize cells
    cell_positions = []
    cell_colors = []

    for key, cell in cells.items():
        n = cell["count"]
        cell_positions.append(cell["pos_sum"] / n)
        cell_colors.append(np.clip(cell["col_sum"] / n, 0, 1))

    cell_positions = np.array(cell_positions, dtype=np.float32)
    cell_colors = np.array(cell_colors, dtype=np.float32)

    print(f"  Cells: {len(cell_positions):,}")
    print(f"  Bounds: X[{cell_positions[:,0].min():.2f}, {cell_positions[:,0].max():.2f}] "
          f"Y[{cell_positions[:,1].min():.2f}, {cell_positions[:,1].max():.2f}] "
          f"Z[{cell_positions[:,2].min():.2f}, {cell_positions[:,2].max():.2f}]")

    return cell_positions, cell_colors, cell_size


def render_views(cell_positions, cell_colors, cell_size, output_dir, n_views=8, res=(1280, 720)):
    """Render cell grid from multiple viewpoints using EGL (NVIDIA GPU)."""
    import moderngl

    print(f"[4/4] Rendering {n_views} views on GPU...")

    ctx = moderngl.create_context(standalone=True, backend="egl")
    print(f"  GPU: {ctx.info['GL_RENDERER']}")

    VS = """
    #version 330
    in vec3 in_pos;
    in vec3 in_col;
    uniform mat4 u_mvp;
    out vec3 v_color;
    void main() {
        gl_Position = u_mvp * vec4(in_pos, 1.0);
        v_color = in_col;
    }
    """
    FS = """
    #version 330
    in vec3 v_color;
    out vec4 frag_color;
    void main() { frag_color = vec4(v_color, 1.0); }
    """

    prog = ctx.program(vertex_shader=VS, fragment_shader=FS)

    # Expand cells into cube triangles
    n = len(cell_positions)
    half = cell_size * 0.5

    # 8 corners of unit cube
    offsets = np.array([
        [-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
        [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1],
    ], dtype=np.float32) * half

    # 12 triangles (36 indices)
    tri_idx = [
        4,5,6, 4,6,7,  # front
        1,0,3, 1,3,2,  # back
        0,4,7, 0,7,3,  # left
        5,1,2, 5,2,6,  # right
        3,7,6, 3,6,2,  # top
        0,1,5, 0,5,4,  # bottom
    ]

    print(f"  Expanding {n} cells to {n*36} vertices...")
    t0 = time.time()

    all_verts = np.zeros((n * 36, 6), dtype=np.float32)
    for ci in range(n):
        p = cell_positions[ci]
        c = cell_colors[ci]
        base = ci * 36
        for ti, idx in enumerate(tri_idx):
            all_verts[base + ti, :3] = offsets[idx] + p
            all_verts[base + ti, 3:6] = c

    print(f"  Expanded in {time.time()-t0:.1f}s")

    vbo = ctx.buffer(all_verts.tobytes())
    vao = ctx.vertex_array(prog, [(vbo, "3f 3f", "in_pos", "in_col")])

    W, H = res
    fbo = ctx.simple_framebuffer((W, H))

    os.makedirs(output_dir, exist_ok=True)

    # Camera projection
    fov = 60.0
    aspect = W / H
    near, far = 0.01, 100.0
    f = 1.0 / math.tan(math.radians(fov) / 2.0)
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = (2 * far * near) / (near - far)
    proj[3, 2] = -1.0

    def view_matrix(pos, yaw, pitch):
        ry = math.radians(yaw)
        rp = math.radians(pitch)
        front = np.array([math.cos(rp)*math.cos(ry), math.sin(rp), math.cos(rp)*math.sin(ry)])
        right = np.cross(front, [0, 1, 0])
        rn = np.linalg.norm(right)
        right = right / rn if rn > 1e-6 else np.array([1, 0, 0])
        up = np.cross(right, front)
        m = np.eye(4, dtype=np.float32)
        m[0, :3] = right
        m[1, :3] = up
        m[2, :3] = -front
        m[0, 3] = -np.dot(right, pos)
        m[1, 3] = -np.dot(up, pos)
        m[2, 3] = np.dot(front, pos)
        return m

    # Camera at origin looking into the scene (+Z)
    # View 0 = original camera angle (should match the photo)
    views = [("original_angle", 90, 0)]
    step = 360 // max(n_views - 1, 1)
    for i in range(1, n_views):
        yaw = 90 + i * step
        views.append((f"rotate_{i*step}deg", yaw, 0))

    # Also add looking down and up
    if n_views > 4:
        views.append(("look_down", 90, -30))
        views.append(("look_up", 90, 20))

    cam_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    for i, (name, yaw, pitch) in enumerate(views[:n_views]):
        view = view_matrix(cam_pos, yaw, pitch)
        mvp = (proj @ view).astype(np.float32)
        prog["u_mvp"].write(mvp.tobytes())

        fbo.use()
        fbo.clear(0.1, 0.1, 0.15, 1.0)  # dark background
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.CULL_FACE)
        vao.render(moderngl.TRIANGLES)

        pixels = np.frombuffer(fbo.read(), dtype=np.uint8).reshape(H, W, 3)
        img = Image.fromarray(np.flipud(pixels))
        out_path = os.path.join(output_dir, f"view_{i:02d}_{name}.png")
        img.save(out_path)

        non_bg = np.sum(pixels.sum(axis=2) > 50)
        pct = 100 * non_bg / (W * H)
        print(f"  [{i}] {name}: {pct:.1f}% filled")

    # Save HTML viewer
    with open(os.path.join(output_dir, "index.html"), "w") as f:
        f.write('<!DOCTYPE html><html><head><title>Trivima Photo Render</title>\n')
        f.write('<style>body{background:#111;color:#fff;font-family:monospace;text-align:center;margin:20px}\n')
        f.write('img{max-width:95vw;border:2px solid #333;margin:5px;cursor:pointer}\n')
        f.write('.grid{display:flex;flex-wrap:wrap;justify-content:center;gap:8px}\n')
        f.write('.thumb{width:300px}h1{color:#e94560}</style></head>\n')
        f.write(f'<body><h1>Trivima - Photo to 3D ({n} cells)</h1>\n')
        f.write(f'<p>Rendered on {ctx.info["GL_RENDERER"]} via EGL</p>\n')
        f.write('<div class="grid">\n')
        for i, (name, _, _) in enumerate(views[:n_views]):
            f.write(f'<img class="thumb" src="view_{i:02d}_{name}.png" '
                    f'title="{name}" onclick="window.open(this.src)">\n')
        f.write('</div><p>View 0 = original camera angle</p></body></html>')

    print(f"\nDone! Output: {output_dir}/")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Trivima Photo to 3D Render")
    parser.add_argument("--image", type=str, required=True, help="Input room photo")
    parser.add_argument("--views", type=int, default=8, help="Number of views to render")
    parser.add_argument("--output", type=str, default="/workspace/renders", help="Output directory")
    parser.add_argument("--cell_size", type=float, default=0.03, help="Cell size in meters")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print("=" * 50)
    print("Trivima — Photo to 3D Render")
    print("=" * 50)

    image, depth, focal, h, w = run_perception(args.image, args.device)
    positions, colors, cs = build_cells(image, depth, focal, h, w, args.cell_size)
    output = render_views(positions, colors, cs, args.output, args.views)

    print(f"\nServing at http://localhost:8080")
    os.chdir(output)
    os.system("python3 -m http.server 8080 &")


if __name__ == "__main__":
    main()
