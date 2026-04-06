#!/usr/bin/env python3
"""
Render the room from multiple positions inside it.
Walk through the 3D cell world — move the camera, look around.
"""
import os, sys, math, time
import numpy as np
from PIL import Image

os.environ["PYOPENGL_PLATFORM"] = "egl"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trivima_expanded_renderer import (
    expand_cells_to_triangles, view_matrix, projection_matrix
)


def main():
    import torch
    from trivima.perception.depth_pro import DepthProEstimator
    from trivima.perception.depth_smoothing import bilateral_depth_smooth

    image_path = sys.argv[1] if len(sys.argv) > 1 else "test_room.jpg"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output_3d_walk"

    print("=" * 60)
    print("  Trivima — 3D Room Walk-Through")
    print("=" * 60)

    # Depth Pro
    print("[1/3] Depth Pro...")
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
    print(f"  {w}x{h}, focal={focal:.0f}")

    # Build cells
    print("[2/3] Building cells...")
    cx, cy = w / 2.0, h / 2.0
    cs = 0.02
    u_grid, v_grid = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    valid = smoothed > 0.1
    px = (u_grid - cx) * smoothed / focal
    py = -(v_grid - cy) * smoothed / focal
    pz = -smoothed

    positions = np.stack([px[valid], py[valid], pz[valid]], axis=-1).astype(np.float32)
    colors = image[valid].astype(np.float32) / 255.0

    # Compute per-pixel normals from depth gradient (actual surface direction)
    try:
        import cv2
        dz_dx = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=5)
        dz_dy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=5)
    except ImportError:
        from scipy.ndimage import convolve
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        dz_dx = convolve(smoothed, sobel_x)
        dz_dy = convolve(smoothed, sobel_x.T)

    # Normal = cross product of surface tangents in camera space
    # Tangent along x: (1/focal, 0, dz/dx), tangent along y: (0, -1/focal, dz/dy)
    # Normal = normalize(-dz/dx, dz/dy, 1) roughly
    nx = -dz_dx / focal
    ny = dz_dy / focal
    nz = np.ones_like(smoothed)
    norm = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-8
    nx /= norm; ny /= norm; nz /= norm
    # Transform to world space: nz should be negative (facing toward camera at +Z)
    pixel_normals = np.stack([nx[valid], ny[valid], -nz[valid]], axis=-1).astype(np.float32)

    # Bin into cells
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

    print(f"  {n_cells:,} surface cells")
    print(f"  X: [{cell_pos[:,0].min():.2f}, {cell_pos[:,0].max():.2f}]")
    print(f"  Y: [{cell_pos[:,1].min():.2f}, {cell_pos[:,1].max():.2f}]")
    print(f"  Z: [{cell_pos[:,2].min():.2f}, {cell_pos[:,2].max():.2f}]")

    # Volume fill — give objects thickness
    print("  Filling volumes behind surfaces...")
    from trivima.construction.volume_fill import fill_volume
    cell_pos, cell_col, cell_nrm = fill_volume(
        cell_pos, cell_col, cell_nrm,
        cell_size=cs,
        default_depth=0.15,  # 15cm default thickness
    )
    n_cells = len(cell_pos)

    # Room center and dimensions
    room_cx = (cell_pos[:,0].min() + cell_pos[:,0].max()) / 2
    room_cz = (cell_pos[:,2].min() + cell_pos[:,2].max()) / 2
    floor_y = cell_pos[:,1].min()
    eye_y = floor_y + 0.3  # eye height above floor

    # Setup GL
    print("[3/3] Rendering walk-through...")
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
    FS = """
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
    prog = ctx.program(vertex_shader=VS, fragment_shader=FS)

    cell_sizes = np.full(n_cells, cs, dtype=np.float32)
    verts = expand_cells_to_triangles(cell_pos, cell_col, cell_nrm, cell_sizes)
    vbo = ctx.buffer(verts.tobytes())
    vao = ctx.vertex_array(prog, [(vbo, "3f 3f 3f", "in_position", "in_color", "in_normal")])

    W, H = 1280, 720
    fbo = ctx.simple_framebuffer((W, H))
    fov_deg = math.degrees(2.0 * math.atan(h / (2.0 * focal)))
    proj = projection_matrix(fov_deg, W / H)

    os.makedirs(output_dir, exist_ok=True)

    # Camera positions and angles — walk through the room
    views = [
        # (name, cam_x, cam_y, cam_z, yaw, pitch)
        ("front_original", 0.0, 0.0, 0.0, -90, 0),
        ("front_look_down", 0.0, 0.0, 0.0, -90, -15),
        ("front_look_up", 0.0, 0.0, 0.0, -90, 15),
        ("front_look_left", 0.0, 0.0, 0.0, -60, 0),
        ("front_look_right", 0.0, 0.0, 0.0, -120, 0),
        ("step_forward", 0.0, 0.0, -0.3, -90, 0),
        ("step_forward_2", 0.0, 0.0, -0.6, -90, 0),
        ("step_left", -0.3, 0.0, -0.3, -90, 0),
        ("step_right", 0.3, 0.0, -0.3, -90, 0),
        ("walk_into_room", 0.0, 0.0, -0.8, -90, -5),
        ("at_center_look_left", room_cx, eye_y, room_cz, -45, 0),
        ("at_center_look_right", room_cx, eye_y, room_cz, -135, 0),
    ]

    print(f"\n  Rendering {len(views)} views...")
    for i, (name, cx, cy, cz, yaw, pitch) in enumerate(views):
        cam = np.array([cx, cy, cz], dtype=np.float64)
        view = view_matrix(cam, yaw, pitch)

        prog["u_projection"].write(proj.T.tobytes())
        prog["u_view"].write(view.T.tobytes())
        prog["u_light_dir"].value = (0.3, 0.8, 0.4)
        prog["u_camera_pos"].value = tuple(cam.astype(np.float32))

        fbo.use()
        fbo.clear(0.12, 0.12, 0.18, 1.0)
        ctx.enable(moderngl.DEPTH_TEST)
        vao.render(moderngl.TRIANGLES)

        pixels = np.frombuffer(fbo.read(), dtype=np.uint8).reshape(H, W, 3)
        img = Image.fromarray(np.flipud(pixels))
        fname = f"walk_{i:02d}_{name}.png"
        img.save(os.path.join(output_dir, fname))

        # Count filled pixels
        bg = np.array([31, 31, 46])
        non_bg = np.sum(np.abs(pixels.astype(float) - bg).sum(axis=2) > 30)
        pct = 100 * non_bg / (W * H)
        print(f"  [{i:2d}] {name:25s} cam=({cx:.1f},{cy:.1f},{cz:.1f}) yaw={yaw:4d} -> {pct:.1f}%")

    # HTML viewer
    html = "<!DOCTYPE html><html><head><title>Trivima 3D Walk</title>\n"
    html += "<style>body{background:#111;color:#fff;font-family:monospace;text-align:center;margin:20px}\n"
    html += "img{max-width:95vw;border:2px solid #333;margin:5px}\n"
    html += ".grid{display:flex;flex-wrap:wrap;justify-content:center;gap:8px}\n"
    html += ".view{width:400px}.label{font-size:11px;color:#aaa;margin-top:2px}\n"
    html += "h1{color:#e94560}</style></head>\n"
    html += "<body><h1>Trivima 3D Room Walk-Through</h1>\n"
    html += "<p>%d cells | FOV %.0f | Moving inside the 3D voxel world</p>\n" % (n_cells, fov_deg)
    html += "<div class='grid'>\n"
    for i, (name, cx, cy, cz, yaw, pitch) in enumerate(views):
        fname = "walk_%02d_%s.png" % (i, name)
        html += "<div><img class='view' src='%s'><div class='label'>%s<br>cam=(%.1f,%.1f,%.1f) yaw=%d</div></div>\n" % (
            fname, name, cx, cy, cz, yaw)
    html += "</div></body></html>"
    with open(os.path.join(output_dir, "index.html"), "w") as f:
        f.write(html)

    ctx.release()
    print(f"\nDone! {output_dir}/index.html")


if __name__ == "__main__":
    main()
