#!/usr/bin/env python3
"""
Trivima — Photo to 3D Render (Instanced)
==========================================
photo → Depth Pro → cell grid → instanced cube rendering on GPU via EGL.

Camera starts at (0,0,0) — the original photo viewpoint.
View 0 should match the original photo.

Usage:
    python render_photo.py --image room.jpg
    python render_photo.py --image room.jpg --views 12 --output renders/
"""

import argparse, os, sys, math, time
import numpy as np
from PIL import Image

os.environ["PYOPENGL_PLATFORM"] = "egl"

VERTEX_SHADER = """
#version 330
in vec3 in_position;
in vec3 in_normal;

in vec3 inst_position;
in vec3 inst_color;
in vec3 inst_normal;
in float inst_size;
in float inst_density;
in float inst_cell_id;

uniform mat4 u_projection;
uniform mat4 u_view;

out vec3 v_color;
out vec3 v_normal;
out vec3 v_world_pos;
out float v_density;

void main() {
    vec3 world_pos = in_position * inst_size * 0.5 + inst_position;
    gl_Position = u_projection * u_view * vec4(world_pos, 1.0);
    v_world_pos = world_pos;
    v_color = inst_color;
    v_normal = inst_normal;
    v_density = inst_density;
}
"""

FRAGMENT_SHADER = """
#version 330
in vec3 v_color;
in vec3 v_normal;
in vec3 v_world_pos;
in float v_density;

uniform vec3 u_light_dir;
uniform vec3 u_camera_pos;

out vec4 frag_color;

void main() {
    vec3 N = normalize(v_normal);
    vec3 L = normalize(u_light_dir);
    vec3 V = normalize(u_camera_pos - v_world_pos);

    float ambient = 0.15;
    float diffuse = max(dot(N, L), 0.0) * 0.6;
    float spec = pow(max(dot(reflect(-L, N), V), 0.0), 32.0) * 0.15;
    float rim = pow(1.0 - max(dot(N, V), 0.0), 3.0) * 0.1;
    vec3 color = v_color * (ambient + diffuse) + vec3(spec + rim);
    frag_color = vec4(color, 1.0);
}
"""


def unit_cube():
    """Unit cube vertices, normals, indices."""
    v = np.array([
        [-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
        [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1],
    ], dtype=np.float32)
    faces = [
        ([4,5,6,7],[0,0,1]),([1,0,3,2],[0,0,-1]),
        ([0,4,7,3],[-1,0,0]),([5,1,2,6],[1,0,0]),
        ([3,7,6,2],[0,1,0]),([0,1,5,4],[0,-1,0]),
    ]
    verts, norms, inds = [], [], []
    for fv, n in faces:
        b = len(verts)
        for vi in fv:
            verts.append(v[vi])
            norms.append(n)
        inds.extend([b,b+1,b+2,b,b+2,b+3])
    return np.array(verts,dtype=np.float32), np.array(norms,dtype=np.float32), np.array(inds,dtype=np.int32)


def view_matrix(pos, yaw, pitch):
    ry, rp = math.radians(yaw), math.radians(pitch)
    fwd = np.array([math.cos(rp)*math.cos(ry), math.sin(rp), math.cos(rp)*math.sin(ry)])
    right = np.cross(fwd, [0,1,0])
    rn = np.linalg.norm(right)
    right = right/rn if rn > 1e-6 else np.array([1,0,0])
    up = np.cross(right, fwd)
    m = np.eye(4, dtype=np.float32)
    m[0,:3]=right; m[1,:3]=up; m[2,:3]=-fwd
    m[0,3]=-np.dot(right,pos); m[1,3]=-np.dot(up,pos); m[2,3]=np.dot(fwd,pos)
    return m


def projection_matrix(fov_deg, aspect, near=0.01, far=50.0):
    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    m = np.zeros((4,4), dtype=np.float32)
    m[0,0] = f/aspect; m[1,1] = f
    m[2,2] = (far+near)/(near-far); m[2,3] = (2*far*near)/(near-far)
    m[3,2] = -1.0
    return m


def main():
    parser = argparse.ArgumentParser(description="Trivima Photo to 3D Render")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--output", type=str, default="/workspace/renders")
    parser.add_argument("--cell_size", type=float, default=0.02)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print("=" * 50)
    print("Trivima — Photo to 3D (Instanced Rendering)")
    print("=" * 50)

    # 1. Perception
    import torch
    from trivima.perception.depth_pro import DepthProEstimator
    from trivima.perception.depth_smoothing import bilateral_depth_smooth

    print("[1/4] Depth Pro...")
    model = DepthProEstimator(device=args.device)
    model.load()
    image = np.array(Image.open(args.image).convert("RGB"))
    h, w = image.shape[:2]
    result = model.estimate(image)
    depth = result["depth"]
    focal = result["focal_length"]
    model.unload(); torch.cuda.empty_cache()

    smoothed = bilateral_depth_smooth(depth, image, spatial_sigma=2.5, color_sigma=25.0)
    print(f"  {w}x{h}, focal={focal:.0f}, depth={smoothed[smoothed>0].min():.2f}-{smoothed.max():.2f}m")

    # 2. Backproject + bin into cells
    print("[2/4] Building cells...")
    cx, cy = w/2.0, h/2.0
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    valid = smoothed > 0.1
    px = (u - cx) * smoothed / focal
    py = -(v - cy) * smoothed / focal
    pz = -smoothed  # OpenGL convention: camera looks along -Z

    positions = np.stack([px[valid], py[valid], pz[valid]], axis=-1).astype(np.float32)
    colors = image[valid].astype(np.float32) / 255.0
    cs = args.cell_size

    cell_idx = np.floor(positions / cs).astype(np.int32)
    cells = {}
    for i in range(len(positions)):
        key = tuple(cell_idx[i])
        if key not in cells:
            cells[key] = {"ps": np.zeros(3), "cs": np.zeros(3), "ns": np.zeros(3), "n": 0}
        cells[key]["ps"] += positions[i]
        cells[key]["cs"] += colors[i]
        cells[key]["n"] += 1

    # Compute normals from position gradient
    n_cells = len(cells)
    cell_pos = np.zeros((n_cells, 3), dtype=np.float32)
    cell_col = np.zeros((n_cells, 3), dtype=np.float32)
    cell_nrm = np.zeros((n_cells, 3), dtype=np.float32)
    cell_size = np.full(n_cells, cs, dtype=np.float32)
    cell_dens = np.ones(n_cells, dtype=np.float32)
    cell_ids = np.arange(n_cells, dtype=np.float32)

    for i, (key, cell) in enumerate(cells.items()):
        cell_pos[i] = cell["ps"] / cell["n"]
        cell_col[i] = np.clip(cell["cs"] / cell["n"], 0, 1)
        # Normal points toward camera at origin (cells are at -Z, so normal is +Z)
        to_cam = -cell_pos[i]
        nm = np.linalg.norm(to_cam)
        cell_nrm[i] = to_cam / nm if nm > 1e-6 else [0, 0, 1]

    print(f"  {n_cells:,} cells from {len(positions):,} points")

    # 3. Render
    print("[3/4] GPU Rendering...")
    import moderngl

    ctx = moderngl.create_context(standalone=True, backend="egl")
    gpu = ctx.info["GL_RENDERER"]
    print(f"  GPU: {gpu}")

    prog = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
    cube_v, cube_n, cube_i = unit_cube()

    # Build VAO with only attributes the compiled shader actually uses
    # (GLSL optimizer removes unused attributes → KeyError if we try to bind them)
    active_attrs = set(prog)
    buffers = [(ctx.buffer(cube_v.tobytes()), "3f", "in_position")]
    if "in_normal" in active_attrs:
        buffers.append((ctx.buffer(cube_n.tobytes()), "3f", "in_normal"))
    inst_bufs = [
        (cell_pos, "3f/i", "inst_position"),
        (cell_col, "3f/i", "inst_color"),
        (cell_nrm, "3f/i", "inst_normal"),
        (cell_size, "f/i", "inst_size"),
        (cell_dens, "f/i", "inst_density"),
        (cell_ids, "f/i", "inst_cell_id"),
    ]
    for data, fmt, name in inst_bufs:
        if name in active_attrs:
            buffers.append((ctx.buffer(data.tobytes()), fmt, name))

    vao = ctx.vertex_array(prog, buffers, index_buffer=ctx.buffer(cube_i.tobytes()))

    W, H = 1280, 720
    fbo = ctx.simple_framebuffer((W, H))

    # FOV from actual focal length
    fov_deg = math.degrees(2.0 * math.atan(h / (2.0 * focal)))
    proj = projection_matrix(fov_deg, W / H)

    os.makedirs(args.output, exist_ok=True)
    cam_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    # Camera looks along -Z (OpenGL convention). yaw=-90 = looking along -Z.
    views = [("original", -90, 0)]
    for i in range(1, args.views):
        angle = i * (360 // args.views)
        views.append((f"rot_{angle}", -90 + angle, 0))

    print("[4/4] Writing views...")
    for i, (name, yaw, pitch) in enumerate(views):
        view = view_matrix(cam_pos, yaw, pitch)
        prog["u_projection"].write(proj.tobytes())
        prog["u_view"].write(view.tobytes())
        prog["u_light_dir"].value = (0.3, 0.8, 0.4)
        prog["u_camera_pos"].value = tuple(cam_pos.astype(np.float32))

        fbo.use()
        fbo.clear(0.12, 0.12, 0.18, 1.0)
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.CULL_FACE)

        t0 = time.time()
        vao.render(moderngl.TRIANGLES, instances=n_cells)
        render_ms = (time.time() - t0) * 1000

        pixels = np.frombuffer(fbo.read(), dtype=np.uint8).reshape(H, W, 3)
        img = Image.fromarray(np.flipud(pixels))
        fname = "view_%02d_%s.png" % (i, name)
        img.save(os.path.join(args.output, fname))

        # Validate: count non-background pixels
        bg = np.array([31, 31, 46])
        non_bg = np.sum(np.abs(pixels.astype(float) - bg).sum(axis=2) > 30)
        pct = 100 * non_bg / (W * H)
        size_kb = os.path.getsize(os.path.join(args.output, fname)) // 1024
        print(f"  [{i}] {name}: {pct:.1f}% filled, {render_ms:.1f}ms, {size_kb}KB")

    # HTML viewer
    html = '<!DOCTYPE html><html><head><title>Trivima</title>\n'
    html += '<style>body{background:#111;color:#fff;font-family:monospace;text-align:center;margin:20px}\n'
    html += 'img{max-width:95vw;border:2px solid #333;margin:5px;cursor:pointer}\n'
    html += '.grid{display:flex;flex-wrap:wrap;justify-content:center;gap:8px}\n'
    html += '.thumb{width:300px}h1{color:#e94560}</style></head>\n'
    html += '<body><h1>Trivima Photo to 3D (%d cells)</h1>\n' % n_cells
    html += '<p>%s | %dx%d | FOV %.0f | View 0 = original angle</p>\n' % (gpu, W, H, fov_deg)
    html += '<div class="grid">\n'
    for i, (name, _, _) in enumerate(views):
        fname = "view_%02d_%s.png" % (i, name)
        html += '<img class="thumb" src="%s" title="%s" onclick="window.open(this.src)">\n' % (fname, name)
    html += '</div></body></html>'
    with open(os.path.join(args.output, "index.html"), "w") as f:
        f.write(html)

    print(f"\nDone! {n_cells} cells, {len(views)} views.")


if __name__ == "__main__":
    main()
