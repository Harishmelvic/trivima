#!/usr/bin/env python3
"""
Complete 3D room — proper cells, volume fill, shell, 1cm resolution.
No GAN. Just correct cell geometry rendered as-is.
"""
import os, sys, math, time
import numpy as np
from PIL import Image

os.environ["PYOPENGL_PLATFORM"] = "egl"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def expand_cells(cell_pos, cell_col, cell_nrm, cell_sizes):
    """Expand N cells into N*36 triangle vertices. Self-contained."""
    n = len(cell_pos)
    cube_v = np.array([
        [.5,.5,.5],[-.5,.5,.5],[-.5,-.5,.5],[.5,.5,.5],[-.5,-.5,.5],[.5,-.5,.5],
        [-.5,.5,-.5],[.5,.5,-.5],[.5,-.5,-.5],[-.5,.5,-.5],[.5,-.5,-.5],[-.5,-.5,-.5],
        [.5,.5,-.5],[.5,.5,.5],[.5,-.5,.5],[.5,.5,-.5],[.5,-.5,.5],[.5,-.5,-.5],
        [-.5,.5,.5],[-.5,.5,-.5],[-.5,-.5,-.5],[-.5,.5,.5],[-.5,-.5,-.5],[-.5,-.5,.5],
        [-.5,.5,.5],[.5,.5,.5],[.5,.5,-.5],[-.5,.5,.5],[.5,.5,-.5],[-.5,.5,-.5],
        [-.5,-.5,-.5],[.5,-.5,-.5],[.5,-.5,.5],[-.5,-.5,-.5],[.5,-.5,.5],[-.5,-.5,.5],
    ], dtype=np.float32)
    cube_n = np.array(
        [[0,0,1]]*6+[[0,0,-1]]*6+[[1,0,0]]*6+[[-1,0,0]]*6+[[0,1,0]]*6+[[0,-1,0]]*6,
        dtype=np.float32)
    sizes = np.asarray(cell_sizes, dtype=np.float32)
    if sizes.ndim == 0:
        sizes = np.full(n, float(sizes), dtype=np.float32)
    scaled = cube_v[None,:,:] * sizes[:,None,None]
    translated = scaled + cell_pos[:,None,:]
    out = np.zeros((n * 36, 9), dtype=np.float32)
    out[:, 0:3] = translated.reshape(-1, 3)
    out[:, 3:6] = np.broadcast_to(cell_col[:,None,:], (n,36,3)).reshape(-1, 3)
    out[:, 6:9] = np.broadcast_to(cube_n[None,:,:], (n,36,3)).reshape(-1, 3)
    return out


def make_view(pos, yaw, pitch):
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


def make_proj(fov_deg, aspect, near=0.01, far=50.0):
    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    m = np.zeros((4,4), dtype=np.float32)
    m[0,0]=f/aspect; m[1,1]=f
    m[2,2]=(far+near)/(near-far); m[2,3]=(2*far*near)/(near-far)
    m[3,2]=-1.0
    return m


def main():
    import torch

    image_path = sys.argv[1] if len(sys.argv) > 1 else "test_room.jpg"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output_complete"
    cell_size = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01  # 1cm default

    print("=" * 60)
    print("  Trivima — Complete 3D Room (Cells Only)")
    print(f"  Cell size: {cell_size*100:.1f}cm")
    print("=" * 60)

    # ============================================================
    # Stage 1: Depth Pro
    # ============================================================
    print("\n[1/4] Depth Pro...")
    from trivima.perception.depth_pro import DepthProEstimator
    from trivima.perception.depth_smoothing import bilateral_depth_smooth

    t0 = time.time()
    model = DepthProEstimator(device="cuda")
    model.load()
    image = np.array(Image.open(image_path).convert("RGB"))
    h, w = image.shape[:2]
    result = model.estimate(image)
    depth = result["depth"]
    focal = result["focal_length"]
    model.unload()
    torch.cuda.empty_cache()

    smoothed = bilateral_depth_smooth(depth, image, spatial_sigma=2.5, color_sigma=25.0)
    print(f"  {w}x{h}, focal={focal:.0f}, depth={smoothed[smoothed>0].min():.2f}-{smoothed.max():.2f}m ({time.time()-t0:.1f}s)")

    # ============================================================
    # Stage 2: Backproject + proper normals from depth gradient
    # ============================================================
    print("\n[2/4] Building cells with proper normals...")
    t0 = time.time()

    cx, cy = w / 2.0, h / 2.0
    u_grid, v_grid = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    valid = smoothed > 0.1

    # Backproject
    px = (u_grid - cx) * smoothed / focal
    py = -(v_grid - cy) * smoothed / focal
    pz = -smoothed  # OpenGL: camera looks along -Z

    positions = np.stack([px[valid], py[valid], pz[valid]], axis=-1).astype(np.float32)
    colors = image[valid].astype(np.float32) / 255.0

    # Compute per-pixel normals from depth gradient (Sobel)
    try:
        import cv2
        dz_dx = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=5)
        dz_dy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=5)
    except ImportError:
        from scipy.ndimage import convolve
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        dz_dx = convolve(smoothed, sobel_x)
        dz_dy = convolve(smoothed, sobel_x.T)

    # Surface normal in camera space
    nx = -dz_dx / focal
    ny = dz_dy / focal
    nz = np.ones_like(smoothed)
    norm = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-8
    nx /= norm; ny /= norm; nz /= norm
    # Transform: camera looks along -Z, so normal z component is negative
    pixel_normals = np.stack([nx[valid], ny[valid], -nz[valid]], axis=-1).astype(np.float32)

    # Bin into cells
    cs = cell_size
    cell_idx = np.floor(positions / cs).astype(np.int32)
    bins = {}
    for i in range(len(positions)):
        key = tuple(cell_idx[i])
        if key not in bins:
            bins[key] = {"ps": np.zeros(3, dtype=np.float64), "cs": np.zeros(3, dtype=np.float64),
                         "ns": np.zeros(3, dtype=np.float64), "n": 0}
        bins[key]["ps"] += positions[i]
        bins[key]["cs"] += colors[i]
        bins[key]["ns"] += pixel_normals[i]
        bins[key]["n"] += 1

    n_cells = len(bins)
    cell_pos = np.zeros((n_cells, 3), dtype=np.float32)
    cell_col = np.zeros((n_cells, 3), dtype=np.float32)
    cell_nrm = np.zeros((n_cells, 3), dtype=np.float32)

    for i, (key, cell) in enumerate(bins.items()):
        cell_pos[i] = (cell["ps"] / cell["n"]).astype(np.float32)
        cell_col[i] = np.clip(cell["cs"] / cell["n"], 0, 1).astype(np.float32)
        avg_nrm = cell["ns"] / cell["n"]
        nm = np.linalg.norm(avg_nrm)
        cell_nrm[i] = (avg_nrm / nm).astype(np.float32) if nm > 1e-6 else np.array([0, 0, 1], dtype=np.float32)

    print(f"  {n_cells:,} surface cells ({time.time()-t0:.1f}s)")
    print(f"  X: [{cell_pos[:,0].min():.2f}, {cell_pos[:,0].max():.2f}]")
    print(f"  Y: [{cell_pos[:,1].min():.2f}, {cell_pos[:,1].max():.2f}]")
    print(f"  Z: [{cell_pos[:,2].min():.2f}, {cell_pos[:,2].max():.2f}]")

    # ============================================================
    # Stage 2b: SAM segmentation — assign per-pixel labels
    # ============================================================
    print("\n[2b/4] SAM segmentation...")
    t0 = time.time()
    from ultralytics import SAM
    sam_model = SAM("sam2.1_l.pt")
    sam_results = sam_model(image_path)
    sam_r = sam_results[0]

    # Build per-pixel label map from SAM masks
    pixel_labels = np.zeros((h, w), dtype=np.int32)
    if sam_r.masks:
        sam_masks = sam_r.masks.data.cpu().numpy()
        # Resize masks to image size if needed
        if sam_masks.shape[1:] != (h, w):
            from PIL import Image as PILImage
            resized_masks = []
            for m in sam_masks:
                rm = np.array(PILImage.fromarray(m.astype(np.uint8) * 255).resize((w, h), PILImage.NEAREST)) > 127
                resized_masks.append(rm)
            sam_masks = np.array(resized_masks)

        # Assign labels: larger masks first (background), smaller masks override
        mask_areas = [(i, m.sum()) for i, m in enumerate(sam_masks)]
        mask_areas.sort(key=lambda x: -x[1])  # largest first
        for mask_idx, area in mask_areas:
            pixel_labels[sam_masks[mask_idx]] = mask_idx + 1

        print(f"  SAM found {len(sam_masks)} segments ({time.time()-t0:.1f}s)")
    else:
        print(f"  SAM found 0 segments, using uniform labels")

    del sam_model
    torch.cuda.empty_cache()

    # Assign labels to cells from pixel labels
    cell_labels = pixel_labels[valid].astype(np.int32)
    cell_label_idx = np.floor(positions / cs).astype(np.int32)
    cell_label_map = np.zeros(n_cells, dtype=np.int32)
    # Use majority vote per cell
    label_bins = {}
    for i in range(len(positions)):
        key = tuple(cell_label_idx[i])
        if key not in label_bins:
            label_bins[key] = []
        label_bins[key].append(cell_labels[i])

    for i, (key, cell) in enumerate(bins.items()):
        if key in label_bins:
            labels_list = label_bins[key]
            cell_label_map[i] = int(np.bincount(np.array(labels_list, dtype=np.int64)).argmax())

    # ============================================================
    # Stage 3: Volume fill + shell extension
    # ============================================================
    print("\n[3/4] Volume fill + shell extension...")
    t0 = time.time()

    from trivima.construction.volume_fill import fill_volume
    from trivima.construction.shell_extension import extend_shell

    n_surface = len(cell_pos)
    cell_pos, cell_col, cell_nrm = fill_volume(
        cell_pos, cell_col, cell_nrm,
        cell_labels=cell_label_map,
        cell_size=cs,
        default_depth=0.12,
    )
    n_after_fill = len(cell_pos)

    cell_pos, cell_col, cell_nrm = extend_shell(
        cell_pos, cell_col, cell_nrm,
        cell_size=cs,
        room_height=2.7,
        extend_behind=1.0,
        extend_sides=0.3,
    )
    n_total = len(cell_pos)
    print(f"  {n_surface:,} surface + {n_after_fill - n_surface:,} filled + {n_total - n_after_fill:,} shell = {n_total:,} total ({time.time()-t0:.1f}s)")

    # ============================================================
    # Stage 4: Render from multiple positions
    # ============================================================
    print("\n[4/4] Rendering...")
    import moderngl

    ctx = moderngl.create_context(standalone=True, backend="egl")
    print(f"  GPU: {ctx.info['GL_RENDERER']}")

    VS = """
#version 330
in vec3 in_position;
in vec3 in_color;
in vec3 in_normal;
uniform mat4 u_projection;
uniform mat4 u_view;
uniform vec3 u_light_dir;
uniform vec3 u_camera_pos;
out vec3 v_color;
out vec3 v_world_pos;
out vec3 v_normal;
void main() {
    gl_Position = u_projection * u_view * vec4(in_position, 1.0);
    v_world_pos = in_position;
    v_color = in_color;
    v_normal = in_normal;
}
"""
    FS = """
#version 330
in vec3 v_color;
in vec3 v_world_pos;
in vec3 v_normal;
uniform vec3 u_light_dir;
uniform vec3 u_camera_pos;
out vec4 frag_color;
void main() {
    // Direct cell color — the photo's actual pixel colors
    frag_color = vec4(v_color, 1.0);
}
"""
    prog = ctx.program(vertex_shader=VS, fragment_shader=FS)

    # Batched VBO upload
    cell_sizes = np.full(n_total, cs, dtype=np.float32)
    MAX_CELLS = 75000
    n_batches = (n_total + MAX_CELLS - 1) // MAX_CELLS
    # Build VAO format — handle optimized-out attributes
    active = set(prog)
    fmt_parts, attr_names = [], []
    for name, fmt in [("in_position", "3f"), ("in_color", "3f"), ("in_normal", "3f")]:
        if name in active:
            fmt_parts.append(fmt)
            attr_names.append(name)
        else:
            n_floats = int(fmt[0]) if fmt[0].isdigit() else 1
            fmt_parts.append(f"{n_floats * 4}x")
    fmt_str = " ".join(fmt_parts)
    print(f"  VAO format: {fmt_str}, attrs: {attr_names}")

    vaos = []
    for bi in range(n_batches):
        start = bi * MAX_CELLS
        end = min(start + MAX_CELLS, n_total)
        verts = expand_cells(
            cell_pos[start:end], cell_col[start:end],
            cell_nrm[start:end], cell_sizes[start:end])
        vbo = ctx.buffer(verts.tobytes())
        vao = ctx.vertex_array(prog, [(vbo, fmt_str, *attr_names)])
        vaos.append(vao)
    print(f"  {n_batches} VBO batches uploaded")

    W, H = 1280, 720
    fbo = ctx.simple_framebuffer((W, H))
    fov_deg = math.degrees(2.0 * math.atan(h / (2.0 * focal)))
    proj = make_proj(fov_deg, W / H)

    os.makedirs(output_dir, exist_ok=True)

    # Camera positions — walk through the room
    views = [
        ("original",          0.0,  0.0,  0.0,  -90,   0),
        ("look_down",         0.0,  0.0,  0.0,  -90, -15),
        ("look_left",         0.0,  0.0,  0.0,  -60,   0),
        ("look_right",        0.0,  0.0,  0.0, -120,   0),
        ("step_in",           0.0,  0.0, -0.4,  -90,   0),
        ("step_in_deep",      0.0,  0.0, -0.7,  -90,  -5),
        ("step_left",        -0.3,  0.0, -0.3,  -90,   0),
        ("step_right",        0.3,  0.0, -0.3,  -90,   0),
        ("inside_look_left", -0.1,  0.05, -1.2, -45,  -5),
        ("inside_look_right",-0.1,  0.05, -1.2,-135,  -5),
        ("inside_look_back",  0.0,  0.0, -1.0,   90,   0),
        ("corner_view",      -0.5,  0.0, -0.5,  -70, -10),
    ]

    print(f"\n  Rendering {len(views)} views...")
    for i, (name, cx, cy, cz, yaw, pitch) in enumerate(views):
        cam = np.array([cx, cy, cz], dtype=np.float64)
        view = make_view(cam, yaw, pitch)

        active = set(prog)
        prog["u_projection"].write(proj.T.tobytes())
        prog["u_view"].write(view.T.tobytes())
        if "u_light_dir" in active:
            prog["u_light_dir"].value = (0.3, 0.8, 0.4)
        if "u_camera_pos" in active:
            prog["u_camera_pos"].value = tuple(cam.astype(np.float32))

        fbo.use()
        fbo.clear(0.08, 0.08, 0.12, 1.0)
        ctx.enable(moderngl.DEPTH_TEST)

        for vao in vaos:
            vao.render(moderngl.TRIANGLES)

        pixels = np.frombuffer(fbo.read(), dtype=np.uint8).reshape(H, W, 3)
        img = Image.fromarray(np.flipud(pixels))
        img.save(os.path.join(output_dir, f"{i:02d}_{name}.png"))

        bg = np.array([20, 20, 31])
        non_bg = np.sum(np.abs(pixels.astype(float) - bg).sum(axis=2) > 30)
        pct = 100 * non_bg / (W * H)
        print(f"  [{i:2d}] {name:22s} cam=({cx:.1f},{cy:.1f},{cz:.1f}) yaw={yaw:4d} -> {pct:.1f}%")

    ctx.release()
    print(f"\nDone! {n_total:,} cells at {cs*100:.1f}cm resolution")
    print(f"Output: {output_dir}/")


if __name__ == "__main__":
    main()
