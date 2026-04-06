#!/usr/bin/env python3
"""
Trivima — Complete Pipeline
============================
Photo → Depth Pro → SAM → cells → volume fill → shell → render

Every step connected. No shortcuts. Validates at each stage.

Usage:
    python pipeline_complete.py test_room.jpg output_final
"""

import os, sys, math, time
import numpy as np
from PIL import Image

os.environ["PYOPENGL_PLATFORM"] = "egl"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CELL_SIZE = 0.01  # 1cm resolution


# ============================================================
# Step 1: Depth Pro — metric depth + focal length
# ============================================================
def step1_depth(image_path, device="cuda"):
    import torch
    from trivima.perception.depth_pro import DepthProEstimator
    from trivima.perception.depth_smoothing import bilateral_depth_smooth

    print("[Step 1/6] Depth Pro...")
    t0 = time.time()
    model = DepthProEstimator(device=device)
    model.load()
    image = np.array(Image.open(image_path).convert("RGB"))
    h, w = image.shape[:2]
    result = model.estimate(image)
    depth = result["depth"]
    focal = result["focal_length"]
    model.unload()
    torch.cuda.empty_cache()

    smoothed = bilateral_depth_smooth(depth, image, spatial_sigma=3.0, color_sigma=25.0)

    # Validate
    valid = smoothed > 0.1
    coverage = 100 * valid.sum() / (h * w)
    print(f"  {w}x{h}, focal={focal:.0f}")
    print(f"  Depth: {smoothed[valid].min():.2f} - {smoothed.max():.2f}m")
    print(f"  Coverage: {coverage:.1f}% pixels have depth")
    print(f"  Time: {time.time()-t0:.1f}s")

    return image, smoothed, focal, h, w


# ============================================================
# Step 2: SAM segmentation — label every pixel
# ============================================================
def step2_sam(image_path, h, w, device="cuda"):
    import torch
    from ultralytics import SAM

    print("\n[Step 2/6] SAM segmentation...")
    t0 = time.time()
    model = SAM("sam2.1_l.pt")
    results = model(image_path)
    r = results[0]

    pixel_labels = np.zeros((h, w), dtype=np.int32)
    n_segments = 0

    if r.masks:
        masks = r.masks.data.cpu().numpy()
        if masks.shape[1:] != (h, w):
            resized = []
            for m in masks:
                rm = np.array(Image.fromarray(m.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)) > 127
                resized.append(rm)
            masks = np.array(resized)

        # Assign labels: largest masks first, smaller override
        areas = [(i, m.sum()) for i, m in enumerate(masks)]
        areas.sort(key=lambda x: -x[1])
        for mask_idx, area in areas:
            pixel_labels[masks[mask_idx]] = mask_idx + 1
        n_segments = len(masks)

    del model
    torch.cuda.empty_cache()

    # Validate
    labeled = (pixel_labels > 0).sum()
    print(f"  {n_segments} segments found")
    print(f"  {100*labeled/(h*w):.1f}% pixels labeled")
    print(f"  Time: {time.time()-t0:.1f}s")

    # Return raw masks for Qwen object estimation
    raw_masks = masks if r.masks else np.zeros((0, h, w), dtype=bool)
    return pixel_labels, n_segments, raw_masks


# ============================================================
# Step 3: Build cells — backproject + bin + proper normals
# ============================================================
def step3_cells(image, depth, focal, h, w, pixel_labels):
    print("\n[Step 3/6] Building cells at {:.1f}cm...".format(CELL_SIZE * 100))
    t0 = time.time()
    cs = CELL_SIZE

    # Backproject every pixel to 3D
    cx, cy = w / 2.0, h / 2.0
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    valid = depth > 0.1
    px = (u - cx) * depth / focal
    py = -(v - cy) * depth / focal
    pz = -depth  # OpenGL: -Z

    positions = np.stack([px[valid], py[valid], pz[valid]], axis=-1).astype(np.float32)
    colors = image[valid].astype(np.float32) / 255.0
    labels = pixel_labels[valid].astype(np.int32)

    # Normals from depth gradient (Sobel)
    try:
        import cv2
        dz_dx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=5)
        dz_dy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=5)
    except ImportError:
        from scipy.ndimage import convolve
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        dz_dx = convolve(depth, sobel_x)
        dz_dy = convolve(depth, sobel_x.T)

    nx = -dz_dx / focal
    ny = dz_dy / focal
    nz = np.ones_like(depth)
    norm = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-8
    nx /= norm; ny /= norm; nz /= norm
    pixel_normals = np.stack([nx[valid], ny[valid], -nz[valid]], axis=-1).astype(np.float32)

    # Bin into cells
    cell_idx = np.floor(positions / cs).astype(np.int32)
    bins = {}
    for i in range(len(positions)):
        key = tuple(cell_idx[i])
        if key not in bins:
            bins[key] = {"ps": np.zeros(3, dtype=np.float64),
                         "cs": np.zeros(3, dtype=np.float64),
                         "ns": np.zeros(3, dtype=np.float64),
                         "ls": [], "n": 0}
        bins[key]["ps"] += positions[i]
        bins[key]["cs"] += colors[i]
        bins[key]["ns"] += pixel_normals[i]
        bins[key]["ls"].append(labels[i])
        bins[key]["n"] += 1

    n_cells = len(bins)
    cell_pos = np.zeros((n_cells, 3), dtype=np.float32)
    cell_col = np.zeros((n_cells, 3), dtype=np.float32)
    cell_nrm = np.zeros((n_cells, 3), dtype=np.float32)
    cell_labels = np.zeros(n_cells, dtype=np.int32)

    for i, (key, cell) in enumerate(bins.items()):
        cell_pos[i] = (cell["ps"] / cell["n"]).astype(np.float32)
        cell_col[i] = np.clip(cell["cs"] / cell["n"], 0, 1).astype(np.float32)
        avg_nrm = cell["ns"] / cell["n"]
        nm = np.linalg.norm(avg_nrm)
        cell_nrm[i] = (avg_nrm / nm).astype(np.float32) if nm > 1e-6 else np.array([0, 0, 1], dtype=np.float32)
        cell_labels[i] = int(np.bincount(np.array(cell["ls"], dtype=np.int64)).argmax())

    fov_deg = math.degrees(2.0 * math.atan(h / (2.0 * focal)))

    # Validate
    print(f"  {n_cells:,} cells")
    print(f"  X: [{cell_pos[:,0].min():.2f}, {cell_pos[:,0].max():.2f}]")
    print(f"  Y: [{cell_pos[:,1].min():.2f}, {cell_pos[:,1].max():.2f}]")
    print(f"  Z: [{cell_pos[:,2].min():.2f}, {cell_pos[:,2].max():.2f}]")
    print(f"  Unique labels: {len(np.unique(cell_labels))}")
    print(f"  FOV: {fov_deg:.1f}°")
    print(f"  Time: {time.time()-t0:.1f}s")

    return cell_pos, cell_col, cell_nrm, cell_labels, fov_deg


# ============================================================
# Step 4: Volume fill — per-object with SAM labels
# ============================================================
def step4_fill(cell_pos, cell_col, cell_nrm, cell_labels, image, sam_masks):
    print("\n[Step 4/6] AI volume fill (Qwen estimates per object)...")
    t0 = time.time()
    from trivima.construction.ai_volume_fill import estimate_objects_with_qwen, ai_volume_fill

    # Qwen estimates dimensions for each SAM segment
    estimates = estimate_objects_with_qwen(image, sam_masks, device="cuda")

    # Free CUDA before next step
    import torch
    torch.cuda.empty_cache()

    # Fill using AI estimates
    n_before = len(cell_pos)
    cell_pos, cell_col, cell_nrm = ai_volume_fill(
        cell_pos, cell_col, cell_nrm,
        cell_labels=cell_labels,
        estimates=estimates,
        cell_size=CELL_SIZE,
    )
    print(f"  Time: {time.time()-t0:.1f}s")
    return cell_pos, cell_col, cell_nrm


# ============================================================
# Step 5: Shell extension — enclose the room
# ============================================================
def step5_shell(cell_pos, cell_col, cell_nrm):
    print("\n[Step 5/6] Shell extension...")
    t0 = time.time()
    from trivima.construction.shell_extension import extend_shell

    n_before = len(cell_pos)
    cell_pos, cell_col, cell_nrm = extend_shell(
        cell_pos, cell_col, cell_nrm,
        cell_size=CELL_SIZE,
        room_height=2.7,
        extend_behind=1.5,
        extend_sides=0.5,
    )
    n_shell = len(cell_pos) - n_before
    print(f"  +{n_shell:,} shell cells = {len(cell_pos):,} total")
    print(f"  Time: {time.time()-t0:.1f}s")
    return cell_pos, cell_col, cell_nrm


# ============================================================
# Step 6: Render from multiple viewpoints
# ============================================================
def step6_render(cell_pos, cell_col, cell_nrm, fov_deg, output_dir):
    print("\n[Step 6/6] Rendering...")
    t0 = time.time()
    import moderngl

    n_cells = len(cell_pos)
    cs = CELL_SIZE

    # Cube geometry
    cube_v = np.array([
        [.5,.5,.5],[-.5,.5,.5],[-.5,-.5,.5],[.5,.5,.5],[-.5,-.5,.5],[.5,-.5,.5],
        [-.5,.5,-.5],[.5,.5,-.5],[.5,-.5,-.5],[-.5,.5,-.5],[.5,-.5,-.5],[-.5,-.5,-.5],
        [.5,.5,-.5],[.5,.5,.5],[.5,-.5,.5],[.5,.5,-.5],[.5,-.5,.5],[.5,-.5,-.5],
        [-.5,.5,.5],[-.5,.5,-.5],[-.5,-.5,-.5],[-.5,.5,.5],[-.5,-.5,-.5],[-.5,-.5,.5],
        [-.5,.5,.5],[.5,.5,.5],[.5,.5,-.5],[-.5,.5,.5],[.5,.5,-.5],[-.5,.5,-.5],
        [-.5,-.5,-.5],[.5,-.5,-.5],[.5,-.5,.5],[-.5,-.5,-.5],[.5,-.5,.5],[-.5,-.5,.5],
    ], dtype=np.float32)

    def expand(pos, col, sizes):
        n = len(pos)
        s = np.asarray(sizes, dtype=np.float32)
        if s.ndim == 0: s = np.full(n, float(s), dtype=np.float32)
        scaled = cube_v[None,:,:] * s[:,None,None]
        translated = scaled + pos[:,None,:]
        out = np.zeros((n * 36, 6), dtype=np.float32)
        out[:, 0:3] = translated.reshape(-1, 3)
        out[:, 3:6] = np.broadcast_to(col[:,None,:], (n,36,3)).reshape(-1, 3)
        return out

    def view_mat(pos, yaw, pitch):
        ry, rp = math.radians(yaw), math.radians(pitch)
        fwd = np.array([math.cos(rp)*math.cos(ry), math.sin(rp), math.cos(rp)*math.sin(ry)])
        right = np.cross(fwd, [0,1,0])
        rn = np.linalg.norm(right)
        right = right/rn if rn > 1e-6 else np.array([1.,0.,0.])
        up = np.cross(right, fwd)
        m = np.eye(4, dtype=np.float32)
        m[0,:3]=right; m[1,:3]=up; m[2,:3]=-fwd
        m[0,3]=-np.dot(right,pos); m[1,3]=-np.dot(up,pos); m[2,3]=np.dot(fwd,pos)
        return m

    def proj_mat(fov, aspect):
        f = 1.0/math.tan(math.radians(fov)/2)
        m = np.zeros((4,4), dtype=np.float32)
        m[0,0]=f/aspect; m[1,1]=f
        m[2,2]=-1.0004; m[2,3]=-0.02; m[3,2]=-1.0
        return m

    ctx = moderngl.create_context(standalone=True, backend="egl")
    print(f"  GPU: {ctx.info['GL_RENDERER']}")

    VS = """
#version 330
in vec3 in_pos;
in vec3 in_col;
uniform mat4 u_mvp;
out vec3 v_col;
void main() { gl_Position = u_mvp * vec4(in_pos, 1.0); v_col = in_col; }
"""
    FS = """
#version 330
in vec3 v_col;
out vec4 fc;
void main() { fc = vec4(v_col, 1.0); }
"""
    prog = ctx.program(vertex_shader=VS, fragment_shader=FS)

    # Batched VBOs
    sizes = np.full(n_cells, cs, dtype=np.float32)
    MAX = 75000
    vaos = []
    for start in range(0, n_cells, MAX):
        end = min(start + MAX, n_cells)
        verts = expand(cell_pos[start:end], cell_col[start:end], sizes[start:end])
        vbo = ctx.buffer(verts.tobytes())
        vao = ctx.vertex_array(prog, [(vbo, "3f 3f", "in_pos", "in_col")])
        vaos.append(vao)
    print(f"  {len(vaos)} VBO batches, {n_cells:,} cells")

    W, H = 1280, 720
    fbo = ctx.simple_framebuffer((W, H))
    proj = proj_mat(fov_deg, W / H)

    os.makedirs(output_dir, exist_ok=True)

    views = [
        ("front",           0.0,  0.0,  0.0, -90,   0),
        ("look_down",       0.0,  0.0,  0.0, -90, -20),
        ("look_left",       0.0,  0.0,  0.0, -60,   0),
        ("look_right",      0.0,  0.0,  0.0,-120,   0),
        ("step_in",         0.0,  0.0, -0.4, -90,  -5),
        ("deep_in",         0.0,  0.0, -0.8, -90,  -5),
        ("left_side",      -0.4,  0.0, -0.5, -80,   0),
        ("right_side",      0.4,  0.0, -0.5,-100,   0),
        ("center_left",    -0.1,  0.05,-1.2, -45,  -5),
        ("center_right",   -0.1,  0.05,-1.2,-135,  -5),
        ("look_back",       0.0,  0.0, -1.0,  90,   0),
        ("corner",         -0.5,  0.0, -0.5, -70, -10),
    ]

    results = []
    for i, (name, cx, cy, cz, yaw, pitch) in enumerate(views):
        cam = np.array([cx, cy, cz], dtype=np.float64)
        view = view_mat(cam, yaw, pitch)
        mvp = (proj @ view).astype(np.float32)

        prog["u_mvp"].write(mvp.T.tobytes())
        fbo.use()
        fbo.clear(0.08, 0.08, 0.12, 1.0)
        ctx.enable(moderngl.DEPTH_TEST)

        for vao in vaos:
            vao.render(moderngl.TRIANGLES)
        ctx.finish()

        pixels = np.frombuffer(fbo.read(), dtype=np.uint8).reshape(H, W, 3)
        img = Image.fromarray(np.flipud(pixels))
        img.save(os.path.join(output_dir, f"{i:02d}_{name}.png"))

        bg = np.array([20, 20, 31])
        non_bg = np.sum(np.abs(pixels.astype(float) - bg).sum(axis=2) > 30)
        pct = 100 * non_bg / (W * H)
        results.append((name, pct))
        print(f"  [{i:2d}] {name:18s} {pct:5.1f}%")

    ctx.release()
    print(f"  Time: {time.time()-t0:.1f}s")
    return results


# ============================================================
# Main
# ============================================================
def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else "test_room.jpg"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output_final"

    print("=" * 60)
    print("  Trivima — Complete Pipeline")
    print(f"  Cell size: {CELL_SIZE*100:.1f}cm")
    print(f"  Image: {image_path}")
    print("=" * 60)

    t_total = time.time()

    # Step 1: Depth
    image, depth, focal, h, w = step1_depth(image_path)

    # Step 2: SAM
    pixel_labels, n_segments, sam_masks = step2_sam(image_path, h, w)

    # Step 3: Cells
    cell_pos, cell_col, cell_nrm, cell_labels, fov_deg = step3_cells(
        image, depth, focal, h, w, pixel_labels)
    n_surface = len(cell_pos)

    # Step 4: AI Volume fill (Qwen estimates object dimensions)
    cell_pos, cell_col, cell_nrm = step4_fill(
        cell_pos, cell_col, cell_nrm, cell_labels, image, sam_masks)
    n_filled = len(cell_pos)

    # Step 5: Shell
    cell_pos, cell_col, cell_nrm = step5_shell(cell_pos, cell_col, cell_nrm)
    n_total = len(cell_pos)

    # Save cell data for rendering in a separate process (EGL + CUDA conflict)
    os.makedirs(output_dir, exist_ok=True)
    cell_data_path = os.path.join(output_dir, "cells.npz")
    np.savez_compressed(cell_data_path, pos=cell_pos, col=cell_col, nrm=cell_nrm,
                        fov=np.array([fov_deg]))
    print(f"\n  Saved {n_total:,} cells to {cell_data_path}")

    # Free ALL CUDA memory before rendering
    import torch
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # Render in subprocess to get clean EGL context
    import subprocess
    render_script = f"""
import os, sys, numpy as np
os.environ["PYOPENGL_PLATFORM"] = "egl"
sys.path.insert(0, "{os.path.dirname(os.path.abspath(__file__))}")
from pipeline_complete import step6_render
data = np.load("{cell_data_path}")
step6_render(data["pos"], data["col"], data["nrm"], float(data["fov"][0]), "{output_dir}")
"""
    result = subprocess.run([sys.executable, "-c", render_script],
                           capture_output=True, text=True, timeout=120)
    print(result.stdout)
    if result.returncode != 0:
        print(f"  Render subprocess error: {result.stderr[-500:]}")

    # Parse results from stdout
    results = []
    for line in result.stdout.split("\n"):
        if "%" in line and "[" in line:
            parts = line.strip().split()
            for p in parts:
                if p.endswith("%"):
                    try:
                        results.append(("view", float(p.rstrip("%"))))
                    except ValueError:
                        pass

    # Summary
    dt = time.time() - t_total
    avg_coverage = np.mean([r[1] for r in results])

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Surface cells:  {n_surface:,}")
    print(f"  After fill:     {n_filled:,} (+{n_filled - n_surface:,})")
    print(f"  After shell:    {n_total:,} (+{n_total - n_filled:,})")
    print(f"  SAM segments:   {n_segments}")
    print(f"  Avg coverage:   {avg_coverage:.1f}%")
    print(f"  Total time:     {dt:.1f}s")
    print(f"  Output:         {output_dir}/")


if __name__ == "__main__":
    main()
