#!/usr/bin/env python3
"""
Trivima — Photo to Photorealistic 3D World
============================================
Single command: photo -> depth -> cells -> shell -> buffers -> GAN -> output

Usage:
    python trivima.py --image room.jpg
    python trivima.py --image room.jpg --views 12 --output renders/
    python trivima.py --image room.jpg --no-shell
    python trivima.py --image room.jpg --no-gan
"""

import argparse, os, sys, math, time
import numpy as np
from PIL import Image

os.environ["PYOPENGL_PLATFORM"] = "egl"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser(description="Trivima — Photo to 3D World")
    p.add_argument("--image", type=str, required=True, help="Input room photo")
    p.add_argument("--output", type=str, default="output", help="Output directory")
    p.add_argument("--views", type=int, default=8, help="Number of render views")
    p.add_argument("--cell_size", type=float, default=0.02, help="Cell size in meters")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--no_shell", action="store_true", help="Skip shell extension")
    p.add_argument("--no_gan", action="store_true", help="Skip GAN texturing (flat shading only)")
    p.add_argument("--gan_checkpoint", type=str, default="checkpoints/best.pt")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    return p.parse_args()


# ============================================================
# Stage 1: Perception — Depth Pro + bilateral smoothing
# ============================================================
def run_perception(image_path, device="cuda"):
    """Photo -> depth map + focal length."""
    import torch
    from trivima.perception.depth_pro import DepthProEstimator
    from trivima.perception.depth_smoothing import bilateral_depth_smooth

    print("[1/5] Depth Pro...")
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

    smoothed = bilateral_depth_smooth(depth, image, spatial_sigma=2.5, color_sigma=25.0)
    dt = time.time() - t0
    print(f"  {w}x{h}, focal={focal:.0f}, depth={smoothed[smoothed>0].min():.2f}-{smoothed.max():.2f}m ({dt:.1f}s)")
    return image, smoothed, focal, h, w


# ============================================================
# Stage 2: Cell Grid — backproject + bin into cells
# ============================================================
def build_cells(image, depth, focal, h, w, cell_size=0.02):
    """Depth map -> cell grid (positions, colors, normals)."""
    print("[2/5] Building cells...")
    t0 = time.time()

    cx, cy = w / 2.0, h / 2.0
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    valid = depth > 0.1

    px = (u - cx) * depth / focal
    py = -(v - cy) * depth / focal
    pz = -depth  # OpenGL: camera looks along -Z

    positions = np.stack([px[valid], py[valid], pz[valid]], axis=-1).astype(np.float32)
    colors = image[valid].astype(np.float32) / 255.0

    # Bin into cells
    cell_idx = np.floor(positions / cell_size).astype(np.int32)
    bins = {}
    for i in range(len(positions)):
        key = tuple(cell_idx[i])
        if key not in bins:
            bins[key] = {"ps": np.zeros(3, dtype=np.float64), "cs": np.zeros(3, dtype=np.float64), "n": 0}
        bins[key]["ps"] += positions[i]
        bins[key]["cs"] += colors[i]
        bins[key]["n"] += 1

    n = len(bins)
    cell_pos = np.zeros((n, 3), dtype=np.float32)
    cell_col = np.zeros((n, 3), dtype=np.float32)
    cell_nrm = np.zeros((n, 3), dtype=np.float32)

    for i, (key, cell) in enumerate(bins.items()):
        cell_pos[i] = (cell["ps"] / cell["n"]).astype(np.float32)
        cell_col[i] = np.clip(cell["cs"] / cell["n"], 0, 1).astype(np.float32)
        to_cam = -cell_pos[i]
        nm = np.linalg.norm(to_cam)
        cell_nrm[i] = to_cam / nm if nm > 1e-6 else np.array([0, 0, 1], dtype=np.float32)

    dt = time.time() - t0
    print(f"  {n:,} cells from {len(positions):,} points ({dt:.1f}s)")
    return cell_pos, cell_col, cell_nrm


# ============================================================
# Stage 3: Shell Extension — enclose the room
# ============================================================
def run_shell_extension(cell_pos, cell_col, cell_nrm, cell_size=0.02):
    """Extend cells to fill walls, floor, ceiling."""
    print("[3/5] Shell extension...")
    from trivima.construction.shell_extension import extend_shell

    n_before = len(cell_pos)
    cell_pos, cell_col, cell_nrm = extend_shell(
        cell_pos, cell_col, cell_nrm,
        cell_size=cell_size,
        room_height=2.7,
        extend_behind=1.5,
        extend_sides=0.5,
    )
    n_after = len(cell_pos)
    print(f"  {n_before:,} observed + {n_after - n_before:,} shell = {n_after:,} total")
    return cell_pos, cell_col, cell_nrm


# ============================================================
# Stage 4: Render flat buffers for each view
# ============================================================
VERTEX_SHADER = """
#version 330
in vec3 in_position;
in vec3 in_color;
in vec3 in_normal;
uniform mat4 u_projection;
uniform mat4 u_view;
uniform vec3 u_light_dir;
uniform vec3 u_camera_pos;
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

FRAGMENT_SHADER_FLAT = """
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
    """Expand N cells into N*36 triangle vertices."""
    n = len(cell_pos)
    cube_v = np.array([
        [0.5,0.5,0.5],[-0.5,0.5,0.5],[-0.5,-0.5,0.5],[0.5,0.5,0.5],[-0.5,-0.5,0.5],[0.5,-0.5,0.5],
        [-0.5,0.5,-0.5],[0.5,0.5,-0.5],[0.5,-0.5,-0.5],[-0.5,0.5,-0.5],[0.5,-0.5,-0.5],[-0.5,-0.5,-0.5],
        [0.5,0.5,-0.5],[0.5,0.5,0.5],[0.5,-0.5,0.5],[0.5,0.5,-0.5],[0.5,-0.5,0.5],[0.5,-0.5,-0.5],
        [-0.5,0.5,0.5],[-0.5,0.5,-0.5],[-0.5,-0.5,-0.5],[-0.5,0.5,0.5],[-0.5,-0.5,-0.5],[-0.5,-0.5,0.5],
        [-0.5,0.5,0.5],[0.5,0.5,0.5],[0.5,0.5,-0.5],[-0.5,0.5,0.5],[0.5,0.5,-0.5],[-0.5,0.5,-0.5],
        [-0.5,-0.5,-0.5],[0.5,-0.5,-0.5],[0.5,-0.5,0.5],[-0.5,-0.5,-0.5],[0.5,-0.5,0.5],[-0.5,-0.5,0.5],
    ], dtype=np.float32)
    cube_n = np.array(
        [[0,0,1]]*6+[[0,0,-1]]*6+[[1,0,0]]*6+[[-1,0,0]]*6+[[0,1,0]]*6+[[0,-1,0]]*6,
        dtype=np.float32)

    sizes = np.asarray(cell_sizes, dtype=np.float32)
    if sizes.ndim == 0:
        sizes = np.full(n, float(sizes), dtype=np.float32)

    scaled = cube_v[None, :, :] * sizes[:, None, None]
    translated = scaled + cell_pos[:, None, :]
    colors_exp = np.broadcast_to(cell_col[:, None, :], (n, 36, 3))
    normals_exp = np.broadcast_to(cube_n[None, :, :], (n, 36, 3))

    out = np.zeros((n * 36, 9), dtype=np.float32)
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
    m[0, :3] = right; m[1, :3] = up; m[2, :3] = -fwd
    m[0, 3] = -np.dot(right, pos)
    m[1, 3] = -np.dot(up, pos)
    m[2, 3] = np.dot(fwd, pos)
    return m


def projection_matrix(fov_deg, aspect, near=0.01, far=50.0):
    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect; m[1, 1] = f
    m[2, 2] = (far + near) / (near - far); m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def setup_gl(cell_pos, cell_col, cell_nrm, cell_sizes):
    """Create GL context, upload VBOs in batches. Single shader."""
    import moderngl

    ctx = moderngl.create_context(standalone=True, backend="egl")
    print(f"  GPU: {ctx.info['GL_RENDERER']}")

    prog = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER_FLAT)

    # Batched VBOs (EGL ~3M vertex limit per draw call)
    MAX_CELLS = 75000
    n_cells = len(cell_pos)
    n_batches = (n_cells + MAX_CELLS - 1) // MAX_CELLS

    vbo_list = []
    for bi in range(n_batches):
        start = bi * MAX_CELLS
        end = min(start + MAX_CELLS, n_cells)
        verts = expand_cells_to_triangles(
            cell_pos[start:end], cell_col[start:end],
            cell_nrm[start:end], cell_sizes[start:end])
        vbo_list.append(ctx.buffer(verts.tobytes()))

    # All attributes should be active in the flat shader
    vaos = [ctx.vertex_array(prog, [(vbo, "3f 3f 3f", "in_position", "in_color", "in_normal")])
            for vbo in vbo_list]

    return ctx, prog, vaos


def render_view(ctx, prog, vaos, fbo, proj, cam_pos, yaw, pitch, W, H):
    """Render one view, return pixel array (H, W, 3)."""
    import moderngl

    view = view_matrix(cam_pos, yaw, pitch)
    active = set(prog)

    prog["u_projection"].write(proj.T.tobytes())
    prog["u_view"].write(view.T.tobytes())
    if "u_light_dir" in active:
        prog["u_light_dir"].value = (0.3, 0.8, 0.4)
    if "u_camera_pos" in active:
        prog["u_camera_pos"].value = tuple(cam_pos.astype(np.float32))

    fbo.use()
    fbo.clear(0.12, 0.12, 0.18, 1.0)
    ctx.enable(moderngl.DEPTH_TEST)

    for vao in vaos:
        vao.render(moderngl.TRIANGLES)

    pixels = np.frombuffer(fbo.read(), dtype=np.uint8).reshape(H, W, 3)
    return np.flipud(pixels)


# ============================================================
# Stage 5: GAN Texturing
# ============================================================
def load_gan(checkpoint_path, device="cuda"):
    """Load trained GAN model."""
    import torch
    from trivima.texturing.models.pix2pix_lite import Pix2PixLiteGenerator

    gen = Pix2PixLiteGenerator().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    gen.load_state_dict(ckpt["generator"])
    gen.eval()
    return gen


def apply_gan(gen, buf_8ch, output_w, output_h, device="cuda"):
    """Apply GAN to 8-channel buffer projection -> photorealistic output.

    Input: buf_8ch (256, 256, 8) float32 — cell data projected as smooth 2D maps
    Output: (output_h, output_w, 3) uint8 — photorealistic image
    """
    import torch

    buf_tensor = torch.from_numpy(buf_8ch.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

    with torch.no_grad():
        fake_rgb, _ = gen(buf_tensor)

    fake_np = ((fake_rgb[0].cpu().numpy().transpose(1, 2, 0) + 1) / 2).clip(0, 1)
    fake_np = (fake_np * 255).astype(np.uint8)

    result = np.array(Image.fromarray(fake_np).resize((output_w, output_h), Image.LANCZOS))
    return result


# ============================================================
# Main Pipeline
# ============================================================
def main():
    args = parse_args()
    t_total = time.time()

    print("=" * 60)
    print("  Trivima — Photo to Photorealistic 3D World")
    print("=" * 60)
    print(f"  Image: {args.image}")
    print(f"  Output: {args.output}/")
    print(f"  Views: {args.views}")
    print(f"  GAN: {'ON' if not args.no_gan else 'OFF'}")
    print(f"  Shell: {'ON' if not args.no_shell else 'OFF'}")
    print()

    # Stage 1: Perception
    image, depth, focal, h, w = run_perception(args.image, args.device)
    fov_deg = math.degrees(2.0 * math.atan(h / (2.0 * focal)))

    # Stage 2: Cell Grid
    cell_pos, cell_col, cell_nrm = build_cells(image, depth, focal, h, w, args.cell_size)
    n_observed = len(cell_pos)

    # Stage 3: Shell Extension
    if not args.no_shell:
        cell_pos, cell_col, cell_nrm = run_shell_extension(cell_pos, cell_col, cell_nrm, args.cell_size)

    n_cells = len(cell_pos)
    cell_sizes = np.full(n_cells, args.cell_size, dtype=np.float32)

    # Stage 4: Load GAN BEFORE creating GL context (CUDA + EGL conflict)
    use_gan = not args.no_gan and os.path.exists(args.gan_checkpoint)
    gan_model = None
    if use_gan:
        print("[4/5] Loading GAN model...")
        import torch
        gan_model = load_gan(args.gan_checkpoint, args.device)
        torch.cuda.synchronize()
        print(f"  Loaded: {args.gan_checkpoint}")
    else:
        print("[4/5] GAN skipped (flat shading only)")

    # Stage 5: GPU rendering
    # Buffer renderer: projects cells as smooth point splats for GAN input
    # Cube renderer: for flat-shaded preview (fallback / debug)
    print("[5/5] GPU rendering...")
    from trivima.texturing.gpu_buffer_renderer import GPUBufferRenderer
    buf_renderer = GPUBufferRenderer(width=256, height=256)

    # Also set up cube renderer for flat preview
    ctx, prog, vaos = setup_gl(cell_pos, cell_col, cell_nrm, cell_sizes)

    W, H = args.width, args.height
    fbo = ctx.simple_framebuffer((W, H))
    proj = projection_matrix(fov_deg, W / H)
    cam_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    views = [("original", -90, 0)]
    for i in range(1, args.views):
        angle = i * (360 // args.views)
        views.append((f"rot_{angle}", -90 + angle, 0))

    os.makedirs(args.output, exist_ok=True)
    print(f"\n--- Rendering {len(views)} views ---")

    for i, (name, yaw, pitch) in enumerate(views):
        t0 = time.time()

        # Render flat-shaded cube view (for preview/comparison)
        flat_pixels = render_view(ctx, prog, vaos, fbo, proj, cam_pos, yaw, pitch, W, H)

        if gan_model is not None:
            # Project cells as smooth buffers for GAN (not cubes)
            buf_result = buf_renderer.render(
                cell_pos, cell_col, cell_nrm, args.cell_size, fov_deg,
                cam_pos=cam_pos, yaw=yaw, pitch=pitch)
            buf_8ch = buf_result["combined_8ch"]

            # GAN: buffer projection -> photorealistic
            gan_pixels = apply_gan(gan_model, buf_8ch, W, H, args.device)
            import torch
            torch.cuda.synchronize()

            Image.fromarray(gan_pixels).save(os.path.join(args.output, f"view_{i:02d}_{name}.png"))
            Image.fromarray(flat_pixels).save(os.path.join(args.output, f"flat_{i:02d}_{name}.png"))

            # Also save the buffer projection for debug
            albedo_vis = (buf_result["albedo"] * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(albedo_vis).save(os.path.join(args.output, f"buf_{i:02d}_{name}.png"))
        else:
            Image.fromarray(flat_pixels).save(os.path.join(args.output, f"view_{i:02d}_{name}.png"))

        dt = (time.time() - t0) * 1000
        size_kb = os.path.getsize(os.path.join(args.output, f"view_{i:02d}_{name}.png")) // 1024
        gan_tag = " +GAN" if gan_model else ""
        print(f"  [{i}] {name}: {dt:.0f}ms, {size_kb}KB{gan_tag}")

    # HTML viewer
    html = '<!DOCTYPE html><html><head><title>Trivima</title>\n'
    html += '<style>body{background:#111;color:#fff;font-family:monospace;text-align:center;margin:20px}\n'
    html += 'img{max-width:95vw;border:2px solid #333;margin:5px;cursor:pointer}\n'
    html += '.grid{display:flex;flex-wrap:wrap;justify-content:center;gap:8px}\n'
    html += '.thumb{width:400px}h1{color:#e94560}\n'
    html += '.pair{display:flex;gap:4px;align-items:center}\n'
    html += '.label{font-size:10px;color:#888}</style></head>\n'
    html += '<body><h1>Trivima &mdash; Photo to 3D (%d cells)</h1>\n' % n_cells
    html += '<p>%d observed + %d shell | FOV %.0f&deg; | %s</p>\n' % (
        n_observed, n_cells - n_observed, fov_deg,
        'GAN enhanced' if gan_model else 'Flat shading')
    html += '<div class="grid">\n'
    for i, (name, _, _) in enumerate(views):
        html += '<div>'
        if gan_model:
            html += '<div class="pair">'
            html += '<div><div class="label">Flat</div><img class="thumb" src="flat_%02d_%s.png" style="width:195px"></div>' % (i, name)
            html += '<div><div class="label">GAN</div><img class="thumb" src="view_%02d_%s.png" style="width:195px"></div>' % (i, name)
            html += '</div>'
        else:
            html += '<img class="thumb" src="view_%02d_%s.png" onclick="window.open(this.src)">' % (i, name)
        html += '</div>\n'
    html += '</div></body></html>'
    with open(os.path.join(args.output, "index.html"), "w") as f:
        f.write(html)

    total_s = time.time() - t_total
    print(f"\nDone! {n_cells:,} cells, {len(views)} views, {total_s:.1f}s total")
    print(f"Output: {os.path.abspath(args.output)}/index.html")

    ctx.release()


if __name__ == "__main__":
    main()
