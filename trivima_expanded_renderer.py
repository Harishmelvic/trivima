#!/usr/bin/env python3
"""
Trivima — Expanded Vertex Renderer (No Instancing)
====================================================
Every cell -> 36 vertices (6 faces x 2 triangles x 3 verts) in one VBO.
One draw call. No instancing. Works everywhere.

Usage:
    python trivima_expanded_renderer.py --image room.jpg --headless --frames 8
    python trivima_expanded_renderer.py --synthetic --headless --frames 8
"""

import argparse, os, sys, math, time
import numpy as np
from PIL import Image

os.environ["PYOPENGL_PLATFORM"] = "egl"

VERTEX_SHADER = """
#version 330
in vec3 in_position;
in vec3 in_color;
in vec3 in_normal;

uniform mat4 u_projection;
uniform mat4 u_view;

out vec3 v_color;
out vec3 v_normal;
out vec3 v_world_pos;

void main() {
    gl_Position = u_projection * u_view * vec4(in_position, 1.0);
    v_world_pos = in_position;
    v_color = in_color;
    v_normal = in_normal;
}
"""

FRAGMENT_SHADER = """
#version 330
in vec3 v_color;
in vec3 v_normal;
in vec3 v_world_pos;

uniform vec3 u_light_dir;
uniform vec3 u_camera_pos;

out vec4 frag_color;

void main() {
    vec3 N = normalize(v_normal);
    vec3 L = normalize(u_light_dir);
    vec3 V = normalize(u_camera_pos - v_world_pos);

    float ambient = 0.18;
    float diffuse = max(dot(N, L), 0.0) * 0.55;
    float spec = pow(max(dot(reflect(-L, N), V), 0.0), 32.0) * 0.12;
    vec3 color = v_color * (ambient + diffuse) + vec3(spec);
    frag_color = vec4(color, 1.0);
}
"""


def expand_cells_to_triangles(cell_pos, cell_col, cell_nrm, cell_sizes):
    """Expand N cells into N*36 vertices. No instancing needed.

    Each cell becomes a cube: 6 faces x 2 triangles x 3 vertices = 36 vertices.
    Returns array of shape (N*36, 9) with interleaved pos(3) + col(3) + nrm(3).
    """
    n = len(cell_pos)

    # Unit cube faces: 6 faces, each with 6 vertices (2 triangles)
    # Vertices in [-0.5, 0.5] range
    cube_verts = np.array([
        # +Z face
        [ 0.5,  0.5,  0.5], [-0.5,  0.5,  0.5], [-0.5, -0.5,  0.5],
        [ 0.5,  0.5,  0.5], [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5],
        # -Z face
        [-0.5,  0.5, -0.5], [ 0.5,  0.5, -0.5], [ 0.5, -0.5, -0.5],
        [-0.5,  0.5, -0.5], [ 0.5, -0.5, -0.5], [-0.5, -0.5, -0.5],
        # +X face
        [ 0.5,  0.5, -0.5], [ 0.5,  0.5,  0.5], [ 0.5, -0.5,  0.5],
        [ 0.5,  0.5, -0.5], [ 0.5, -0.5,  0.5], [ 0.5, -0.5, -0.5],
        # -X face
        [-0.5,  0.5,  0.5], [-0.5,  0.5, -0.5], [-0.5, -0.5, -0.5],
        [-0.5,  0.5,  0.5], [-0.5, -0.5, -0.5], [-0.5, -0.5,  0.5],
        # +Y face
        [-0.5,  0.5,  0.5], [ 0.5,  0.5,  0.5], [ 0.5,  0.5, -0.5],
        [-0.5,  0.5,  0.5], [ 0.5,  0.5, -0.5], [-0.5,  0.5, -0.5],
        # -Y face
        [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5], [ 0.5, -0.5,  0.5],
        [-0.5, -0.5, -0.5], [ 0.5, -0.5,  0.5], [-0.5, -0.5,  0.5],
    ], dtype=np.float32)

    cube_normals = np.array([
        [0,0,1]]*6 + [[0,0,-1]]*6 + [[1,0,0]]*6 +
        [[-1,0,0]]*6 + [[0,1,0]]*6 + [[0,-1,0]]*6,
        dtype=np.float32)

    # Vectorized: broadcast across all cells at once
    # cell_pos: (N, 3), cell_sizes: (N,) or scalar
    sizes = np.asarray(cell_sizes, dtype=np.float32)
    if sizes.ndim == 0:
        sizes = np.full(n, float(sizes), dtype=np.float32)

    # Output: (N*36, 9) interleaved pos + col + nrm
    out = np.zeros((n * 36, 9), dtype=np.float32)

    # Scale cube verts by each cell size and translate to cell center
    # cube_verts: (36, 3), sizes: (N,), cell_pos: (N, 3)
    # Result positions: (N, 36, 3)
    scaled = cube_verts[None, :, :] * sizes[:, None, None]  # (N, 36, 3)
    translated = scaled + cell_pos[:, None, :]  # (N, 36, 3)

    # Colors: (N, 36, 3) — same color for all 36 verts of a cell
    colors_exp = np.broadcast_to(cell_col[:, None, :], (n, 36, 3))

    # Normals: use cube face normals for proper lighting
    normals_exp = np.broadcast_to(cube_normals[None, :, :], (n, 36, 3))

    # Flatten and interleave
    out[:, 0:3] = translated.reshape(-1, 3)
    out[:, 3:6] = colors_exp.reshape(-1, 3)
    out[:, 6:9] = normals_exp.reshape(-1, 3)

    return out


def view_matrix(pos, yaw, pitch):
    ry, rp = math.radians(yaw), math.radians(pitch)
    fwd = np.array([math.cos(rp)*math.cos(ry), math.sin(rp), math.cos(rp)*math.sin(ry)])
    right = np.cross(fwd, [0, 1, 0])
    rn = np.linalg.norm(right)
    right = right / rn if rn > 1e-6 else np.array([1.0, 0.0, 0.0])
    up = np.cross(right, fwd)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = right
    m[1, :3] = up
    m[2, :3] = -fwd
    m[0, 3] = -np.dot(right, pos)
    m[1, 3] = -np.dot(up, pos)
    m[2, 3] = np.dot(fwd, pos)
    return m


def projection_matrix(fov_deg, aspect, near=0.01, far=50.0):
    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def build_synthetic_room():
    """Build a simple box room for testing."""
    cells = []
    step = 0.05

    # Floor (y=-1, green)
    for x in np.arange(-2.0, 2.01, step):
        for z in np.arange(-4.0, -0.5, step):
            cells.append(([x, -1.0, z], [0.3, 0.5, 0.3], [0, 1, 0]))

    # Back wall (z=-4, blue-gray)
    for x in np.arange(-2.0, 2.01, step):
        for y in np.arange(-1.0, 1.51, step):
            cells.append(([x, y, -4.0], [0.4, 0.4, 0.55], [0, 0, 1]))

    # Left wall (x=-2, warm)
    for y in np.arange(-1.0, 1.51, step):
        for z in np.arange(-4.0, -0.5, step):
            cells.append(([-2.0, y, z], [0.5, 0.45, 0.4], [1, 0, 0]))

    # Right wall (x=2, warm)
    for y in np.arange(-1.0, 1.51, step):
        for z in np.arange(-4.0, -0.5, step):
            cells.append(([2.0, y, z], [0.5, 0.45, 0.4], [-1, 0, 0]))

    n = len(cells)
    pos = np.array([c[0] for c in cells], dtype=np.float32)
    col = np.array([c[1] for c in cells], dtype=np.float32)
    nrm = np.array([c[2] for c in cells], dtype=np.float32)
    return pos, col, nrm, n


def build_cells_from_photo(image_path, cell_size=0.02, device="cuda"):
    """Depth Pro -> backproject -> bin into cells."""
    import torch
    sys.path.insert(0, "/workspace/trivima")
    from trivima.perception.depth_pro import DepthProEstimator
    from trivima.perception.depth_smoothing import bilateral_depth_smooth

    print("[1/3] Depth Pro...")
    model = DepthProEstimator(device=device)
    model.load()
    image = np.array(Image.open(image_path).convert("RGB"))
    h, w = image.shape[:2]
    result = model.estimate(image)
    depth = result["depth"]
    focal = result["focal_length"]
    model.unload()
    torch.cuda.empty_cache()

    smoothed = bilateral_depth_smooth(depth, image, spatial_sigma=2.5, color_sigma=25.0)
    print(f"  {w}x{h}, focal={focal:.0f}, depth range: {smoothed[smoothed>0].min():.2f} - {smoothed.max():.2f}m")

    print("[2/3] Backproject + bin cells...")
    cx, cy = w / 2.0, h / 2.0
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    valid = smoothed > 0.1
    px = (u - cx) * smoothed / focal
    py = -(v - cy) * smoothed / focal
    pz = -smoothed  # OpenGL convention: camera looks along -Z

    positions = np.stack([px[valid], py[valid], pz[valid]], axis=-1).astype(np.float32)
    colors = image[valid].astype(np.float32) / 255.0
    cs = cell_size

    cell_idx = np.floor(positions / cs).astype(np.int32)
    bins = {}
    for i in range(len(positions)):
        key = tuple(cell_idx[i])
        if key not in bins:
            bins[key] = {"ps": np.zeros(3, dtype=np.float64), "cs": np.zeros(3, dtype=np.float64), "n": 0}
        bins[key]["ps"] += positions[i]
        bins[key]["cs"] += colors[i]
        bins[key]["n"] += 1

    n_cells = len(bins)
    cell_pos = np.zeros((n_cells, 3), dtype=np.float32)
    cell_col = np.zeros((n_cells, 3), dtype=np.float32)
    cell_nrm = np.zeros((n_cells, 3), dtype=np.float32)

    for i, (key, cell) in enumerate(bins.items()):
        cell_pos[i] = (cell["ps"] / cell["n"]).astype(np.float32)
        cell_col[i] = np.clip(cell["cs"] / cell["n"], 0, 1).astype(np.float32)
        to_cam = -cell_pos[i]
        nm = np.linalg.norm(to_cam)
        cell_nrm[i] = to_cam / nm if nm > 1e-6 else np.array([0, 0, 1], dtype=np.float32)

    print(f"  {n_cells:,} cells from {len(positions):,} points")
    return cell_pos, cell_col, cell_nrm, n_cells, focal, h, w


def main():
    parser = argparse.ArgumentParser(description="Trivima Expanded Renderer")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no_shell", action="store_true", help="Skip shell extension")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--output", type=str, default="/workspace/renders")
    parser.add_argument("--cell_size", type=float, default=0.02)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print("=" * 60)
    print("Trivima -- Expanded Vertex Renderer (No Instancing)")
    print("=" * 60)

    if args.synthetic:
        cell_pos, cell_col, cell_nrm, n_cells = build_synthetic_room()
        cell_sizes = np.full(n_cells, 0.05, dtype=np.float32)
        fov_deg = 70.0
    else:
        if not args.image:
            print("ERROR: provide --image or --synthetic")
            sys.exit(1)
        cell_pos, cell_col, cell_nrm, n_cells, focal, img_h, img_w = build_cells_from_photo(
            args.image, args.cell_size, args.device
        )
        cell_sizes = np.full(n_cells, args.cell_size, dtype=np.float32)
        fov_deg = math.degrees(2.0 * math.atan(img_h / (2.0 * focal)))

    # Shell extension — fill in walls, floor, ceiling
    if not args.no_shell and not args.synthetic:
        print("\n[Shell] Extending room shell...")
        from trivima.construction.shell_extension import extend_shell
        n_before = len(cell_pos)
        cell_pos, cell_col, cell_nrm = extend_shell(
            cell_pos, cell_col, cell_nrm,
            cell_size=args.cell_size,
            room_height=2.7,
            extend_behind=1.5,
            extend_sides=0.5,
        )
        n_cells = len(cell_pos)
        cell_sizes = np.full(n_cells, args.cell_size, dtype=np.float32)
        print(f"  {n_before:,} observed + {n_cells - n_before:,} generated = {n_cells:,} total cells")

    # Data validation
    print(f"\n--- Data Validation ---")
    print(f"  Cells: {n_cells:,}")
    print(f"  Position X: [{cell_pos[:,0].min():.3f}, {cell_pos[:,0].max():.3f}]")
    print(f"  Position Y: [{cell_pos[:,1].min():.3f}, {cell_pos[:,1].max():.3f}]")
    print(f"  Position Z: [{cell_pos[:,2].min():.3f}, {cell_pos[:,2].max():.3f}]")
    print(f"  Color range: [{cell_col.min():.3f}, {cell_col.max():.3f}]")
    print(f"  NaN check: pos={np.any(np.isnan(cell_pos))}, col={np.any(np.isnan(cell_col))}")
    print(f"  FOV: {fov_deg:.1f} degrees")

    # Expand cells to triangle vertices in chunks
    # EGL has a ~3M vertex limit per draw call, so split into 80K-cell batches
    MAX_CELLS_PER_BATCH = 75000
    print(f"\n[3/3] Expanding {n_cells:,} cells -> {n_cells*36:,} vertices...")
    t0 = time.time()

    n_batches = (n_cells + MAX_CELLS_PER_BATCH - 1) // MAX_CELLS_PER_BATCH
    vbo_batches = []
    for bi in range(n_batches):
        start = bi * MAX_CELLS_PER_BATCH
        end = min(start + MAX_CELLS_PER_BATCH, n_cells)
        batch_verts = expand_cells_to_triangles(
            cell_pos[start:end], cell_col[start:end], cell_nrm[start:end],
            cell_sizes[start:end]
        )
        vbo_batches.append(batch_verts)

    total_verts = sum(len(b) for b in vbo_batches)
    total_mb = sum(b.nbytes for b in vbo_batches) / (1024 * 1024)
    expand_ms = (time.time() - t0) * 1000
    print(f"  Done in {expand_ms:.0f}ms, {n_batches} batches, {total_verts:,} verts, {total_mb:.1f}MB")

    # Camera position: center of observed cells at eye height
    # For photos, camera was at origin looking along -Z. With shell extension,
    # place camera inside the room so it can see all walls.
    W, H = 1280, 720
    proj = projection_matrix(fov_deg, W / H)

    if args.synthetic or args.no_shell:
        cam_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    else:
        # Position camera at the centroid of observed cells (first n_before cells),
        # at eye height (floor + 1.6m)
        obs_center_x = cell_pos[:n_cells, 0].mean() if not hasattr(args, '_n_observed') else 0.0
        obs_center_z = cell_pos[:n_cells, 2].mean() if not hasattr(args, '_n_observed') else 0.0
        floor_y = cell_pos[:, 1].min()
        cam_pos = np.array([0.0, floor_y + 1.0, cell_pos[:, 2].mean() * 0.3], dtype=np.float64)
        print(f"  Camera at: ({cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f})")
    test_view = view_matrix(cam_pos, -90, 0)
    mvp = proj @ test_view

    print(f"\n--- NDC Check (original view, first 5 cells) ---")
    for i in range(min(5, n_cells)):
        p = np.append(cell_pos[i], 1.0).astype(np.float32)
        clip = mvp @ p
        if abs(clip[3]) > 1e-6:
            ndc = clip[:3] / clip[3]
        else:
            ndc = clip[:3]
        ok = all(-1 <= ndc[j] <= 1 for j in range(3)) and clip[3] > 0
        print(f"  [{i}] pos=({cell_pos[i][0]:.3f}, {cell_pos[i][1]:.3f}, {cell_pos[i][2]:.3f}) -> NDC=({ndc[0]:.3f}, {ndc[1]:.3f}, {ndc[2]:.3f}) w={clip[3]:.3f} {'OK' if ok else 'OUTSIDE'}")

    # Count cells in frustum
    ones = np.ones((n_cells, 1), dtype=np.float32)
    pos4 = np.hstack([cell_pos, ones])
    clip_all = (mvp @ pos4.T).T
    w_all = clip_all[:, 3]
    valid_w = w_all > 0
    ndc_all = np.zeros_like(clip_all[:, :3])
    ndc_all[valid_w] = clip_all[valid_w, :3] / w_all[valid_w, None]
    in_frustum = (
        valid_w &
        (ndc_all[:, 0] >= -1) & (ndc_all[:, 0] <= 1) &
        (ndc_all[:, 1] >= -1) & (ndc_all[:, 1] <= 1) &
        (ndc_all[:, 2] >= -1) & (ndc_all[:, 2] <= 1)
    )
    print(f"  Total in frustum: {in_frustum.sum()}/{n_cells} ({100*in_frustum.sum()/max(n_cells,1):.1f}%)")

    # GPU rendering
    print(f"\n--- GPU Rendering ---")
    import moderngl
    ctx = moderngl.create_context(standalone=True, backend="egl")
    print(f"  GPU: {ctx.info['GL_RENDERER']}")

    prog = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)

    # Build format string, handling optimized-out attributes
    active = set(prog)
    print(f"  Active shader attributes: {active}")

    fmt_parts = []
    attr_names = []
    attr_map = [("in_position", "3f"), ("in_color", "3f"), ("in_normal", "3f")]
    for name, fmt in attr_map:
        if name in active:
            fmt_parts.append(fmt)
            attr_names.append(name)
        else:
            n_floats = int(fmt[0]) if fmt[0].isdigit() else 1
            fmt_parts.append(f"{n_floats * 4}x")

    fmt_str = " ".join(fmt_parts)
    print(f"  VAO format: {fmt_str}, attrs: {attr_names}")

    # Upload batched VBOs and create VAOs
    vaos = []
    for bi, batch in enumerate(vbo_batches):
        vbo = ctx.buffer(batch.tobytes())
        vao = ctx.vertex_array(prog, [(vbo, fmt_str, *attr_names)])
        vaos.append(vao)
    print(f"  Uploaded {len(vaos)} VBO batches")

    fbo = ctx.simple_framebuffer((W, H))
    os.makedirs(args.output, exist_ok=True)

    # Camera views
    views = [("original", -90, 0)]
    for i in range(1, args.frames):
        angle = i * (360 // args.frames)
        views.append((f"rot_{angle}", -90 + angle, 0))

    print(f"\n--- Rendering {len(views)} views ---")
    for i, (name, yaw, pitch) in enumerate(views):
        view = view_matrix(cam_pos, yaw, pitch)
        # OpenGL expects column-major; numpy is row-major -> transpose
        prog["u_projection"].write(proj.T.tobytes())
        prog["u_view"].write(view.T.tobytes())
        prog["u_light_dir"].value = (0.3, 0.8, 0.4)
        prog["u_camera_pos"].value = tuple(cam_pos.astype(np.float32))

        fbo.use()
        fbo.clear(0.12, 0.12, 0.18, 1.0)
        ctx.enable(moderngl.DEPTH_TEST)

        t0 = time.time()
        # Draw each batch — shared depth buffer ensures correct occlusion
        for vao in vaos:
            vao.render(moderngl.TRIANGLES)
        render_ms = (time.time() - t0) * 1000

        pixels = np.frombuffer(fbo.read(), dtype=np.uint8).reshape(H, W, 3)
        img = Image.fromarray(np.flipud(pixels))
        fname = f"expanded_{i:02d}_{name}.png"
        img.save(os.path.join(args.output, fname))

        # Count non-background pixels
        bg = np.array([31, 31, 46])
        non_bg = np.sum(np.abs(pixels.astype(float) - bg).sum(axis=2) > 30)
        pct = 100 * non_bg / (W * H)
        size_kb = os.path.getsize(os.path.join(args.output, fname)) // 1024
        print(f"  [{i}] {name}: {pct:.1f}% filled, {render_ms:.1f}ms, {size_kb}KB")

    # HTML viewer
    html = "<!DOCTYPE html><html><head><title>Trivima Expanded</title>\n"
    html += "<style>body{background:#111;color:#fff;font-family:monospace;text-align:center;margin:20px}\n"
    html += "img{max-width:95vw;border:2px solid #333;margin:5px;cursor:pointer}\n"
    html += ".grid{display:flex;flex-wrap:wrap;justify-content:center;gap:8px}\n"
    html += ".thumb{width:300px}h1{color:#e94560}</style></head>\n"
    html += "<body><h1>Trivima Expanded (%d cells, %d tris)</h1>\n" % (n_cells, n_cells * 12)
    html += "<div class='grid'>\n"
    for i, (name, _, _) in enumerate(views):
        fname = "expanded_%02d_%s.png" % (i, name)
        html += "<img class='thumb' src='%s' title='%s' onclick='window.open(this.src)'>\n" % (fname, name)
    html += "</div></body></html>"
    with open(os.path.join(args.output, "expanded_index.html"), "w") as f:
        f.write(html)

    print(f"\nDone! {n_cells:,} cells, {n_cells*36:,} vertices, {len(views)} views.")
    ctx.release()


if __name__ == "__main__":
    main()
