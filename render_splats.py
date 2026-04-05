#!/usr/bin/env python3
"""Render photo as point splats — correct FOV, correct camera position."""
import os, sys, math, time
import numpy as np
from PIL import Image

os.environ["PYOPENGL_PLATFORM"] = "egl"

def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else "test_room.jpg"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "/workspace/renders"

    print("=" * 50)
    print("Trivima — Photo to 3D Point Splats")
    print("=" * 50)

    # 1. Depth Pro
    print("[1] Depth Pro...")
    import torch
    from trivima.perception.depth_pro import DepthProEstimator
    from trivima.perception.depth_smoothing import bilateral_depth_smooth

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
    print(f"  {w}x{h}, focal={focal:.0f}, depth={smoothed[smoothed>0].min():.2f}-{smoothed.max():.2f}m")

    # 2. Backproject
    print("[2] Backproject...")
    cx, cy = w / 2.0, h / 2.0
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    valid = smoothed > 0.1
    x = (u - cx) * smoothed / focal
    y = -(v - cy) * smoothed / focal
    z = smoothed
    positions = np.stack([x[valid], y[valid], z[valid]], axis=-1).astype(np.float32)
    colors = image[valid].astype(np.float32) / 255.0

    # Subsample every 2nd point for speed
    step = 2
    pos = positions[::step]
    col = colors[::step]
    print(f"  {len(pos):,} points (subsampled from {len(positions):,})")

    # 3. Render
    print("[3] Rendering...")
    import moderngl

    VS = """
    #version 330
    in vec3 in_pos;
    in vec3 in_col;
    uniform mat4 u_mvp;
    out vec3 v_color;
    void main() {
        gl_Position = u_mvp * vec4(in_pos, 1.0);
        gl_PointSize = max(2.0, 800.0 / gl_Position.w);
        v_color = in_col;
    }
    """
    FS = """
    #version 330
    in vec3 v_color;
    out vec4 frag_color;
    void main() { frag_color = vec4(v_color, 1.0); }
    """

    ctx = moderngl.create_context(standalone=True, backend="egl")
    gpu_name = ctx.info["GL_RENDERER"]
    print(f"  GPU: {gpu_name}")

    prog = ctx.program(vertex_shader=VS, fragment_shader=FS)
    vbo = ctx.buffer(np.hstack([pos, col]).tobytes())
    vao = ctx.vertex_array(prog, [(vbo, "3f 3f", "in_pos", "in_col")])

    W, H = 1280, 720
    fbo = ctx.simple_framebuffer((W, H))

    # Projection from actual focal length
    fov_rad = 2.0 * math.atan(h / (2.0 * focal))
    aspect = W / H
    near, far = 0.01, 50.0
    f = 1.0 / math.tan(fov_rad / 2.0)
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = (2 * far * near) / (near - far)
    proj[3, 2] = -1.0

    def view_mat(pos, yaw, pitch):
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

    os.makedirs(output_dir, exist_ok=True)
    cam = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    views = [
        ("original", 90, 0),
        ("right_15", 105, 0),
        ("left_15", 75, 0),
        ("down_20", 90, -20),
        ("up_15", 90, 15),
        ("right_45", 135, -5),
        ("left_45", 45, -5),
        ("behind", 270, 0),
    ]

    for i, (name, yaw, pitch) in enumerate(views):
        mvp = (proj @ view_mat(cam, yaw, pitch)).astype(np.float32)
        prog["u_mvp"].write(mvp.tobytes())
        fbo.use()
        fbo.clear(0.15, 0.15, 0.2, 1.0)
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        vao.render(moderngl.POINTS)
        pixels = np.frombuffer(fbo.read(), dtype=np.uint8).reshape(H, W, 3)
        img = Image.fromarray(np.flipud(pixels))
        fname = "view_%02d_%s.png" % (i, name)
        img.save(os.path.join(output_dir, fname))
        non_bg = np.sum(pixels.sum(axis=2) > 80)
        pct = 100 * non_bg / (W * H)
        print(f"  [{i}] {name}: {pct:.1f}%")

    # HTML
    html = '<!DOCTYPE html><html><head><title>Trivima</title>\n'
    html += '<style>body{background:#111;color:#fff;font-family:monospace;text-align:center;margin:20px}\n'
    html += 'img{max-width:95vw;border:2px solid #333;margin:5px;cursor:pointer}\n'
    html += '.grid{display:flex;flex-wrap:wrap;justify-content:center;gap:8px}\n'
    html += '.thumb{width:300px}h1{color:#e94560}</style></head>\n'
    html += '<body><h1>Trivima Photo to 3D (%d points)</h1>\n' % len(pos)
    html += '<p>Rendered on %s | View 0 = original angle</p>\n' % gpu_name
    html += '<div class="grid">\n'
    for i, (name, _, _) in enumerate(views):
        fname = "view_%02d_%s.png" % (i, name)
        html += '<img class="thumb" src="%s" title="%s" onclick="window.open(this.src)">\n' % (fname, name)
    html += '</div></body></html>'
    with open(os.path.join(output_dir, "index.html"), "w") as f:
        f.write(html)

    print("\nDone!")

if __name__ == "__main__":
    main()
