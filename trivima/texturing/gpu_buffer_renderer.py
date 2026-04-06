"""
GPU Buffer Renderer — projects cells as point splats into clean 2D buffers.

Each cell is rendered as a screen-space point sized to cover its projected area.
No cubes, no gaps, no grid. Produces smooth continuous 2D maps:
  - albedo (H x W x 3): flat surface color
  - depth  (H x W):     metric depth
  - normals (H x W x 3): surface normals as RGB

Uses ModernGL GL_POINTS with gl_PointSize for coverage.
Output goes directly to the GAN as 8-channel input.
"""

import numpy as np
import math
import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


class GPUBufferRenderer:
    """Renders cell grid as point splats into clean 2D buffers for GAN input."""

    def __init__(self, width=256, height=256):
        self.W = width
        self.H = height
        self._ctx = None
        self._prog_albedo = None
        self._prog_depth = None
        self._prog_normal = None

    def _ensure_context(self):
        if self._ctx is not None:
            return

        import moderngl
        self._ctx = moderngl.create_context(standalone=True, backend="egl")

        # Albedo shader — outputs cell color directly
        VS = """
#version 330
in vec3 in_position;
in vec3 in_color;
in vec3 in_normal;
in float in_size;
uniform mat4 u_projection;
uniform mat4 u_view;
uniform float u_point_scale;
out vec3 v_color;
out vec3 v_normal;
out float v_depth;
void main() {
    vec4 view_pos = u_view * vec4(in_position, 1.0);
    gl_Position = u_projection * view_pos;
    // Point size: cell covers its projected area, no gaps
    gl_PointSize = max(2.0, u_point_scale * in_size / (-view_pos.z));
    v_color = in_color;
    v_normal = in_normal;
    v_depth = -view_pos.z;
}
"""
        FS_ALBEDO = """
#version 330
in vec3 v_color;
out vec4 frag_color;
void main() {
    // Circular splat — discard corners for smooth coverage
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    if (dot(coord, coord) > 1.0) discard;
    frag_color = vec4(v_color, 1.0);
}
"""
        FS_NORMAL = """
#version 330
in vec3 v_normal;
out vec4 frag_color;
void main() {
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    if (dot(coord, coord) > 1.0) discard;
    // Encode normal as RGB: [-1,1] -> [0,1]
    frag_color = vec4(v_normal * 0.5 + 0.5, 1.0);
}
"""
        FS_DEPTH = """
#version 330
in float v_depth;
out vec4 frag_color;
void main() {
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    if (dot(coord, coord) > 1.0) discard;
    // Depth normalized: assume 0-10m range
    float d = clamp(v_depth / 10.0, 0.0, 1.0);
    frag_color = vec4(d, d, d, 1.0);
}
"""
        self._prog_albedo = self._ctx.program(vertex_shader=VS, fragment_shader=FS_ALBEDO)
        self._prog_normal = self._ctx.program(vertex_shader=VS, fragment_shader=FS_NORMAL)
        self._prog_depth = self._ctx.program(vertex_shader=VS, fragment_shader=FS_DEPTH)

        self._fbo = self._ctx.simple_framebuffer((self.W, self.H))

    def render(self, cell_pos, cell_col, cell_nrm, cell_size, fov_deg,
               cam_pos=None, yaw=-90, pitch=0):
        """Render cells as point splats into albedo/depth/normal buffers.

        Args:
            cell_pos: (N, 3) float32 — cell positions
            cell_col: (N, 3) float32 — cell colors [0,1]
            cell_nrm: (N, 3) float32 — cell normals
            cell_size: float — cell size in meters
            fov_deg: float — field of view
            cam_pos: (3,) camera position (default: origin)
            yaw, pitch: camera angles

        Returns:
            dict with 'albedo' (H,W,3), 'depth' (H,W), 'normals' (H,W,3),
            'combined_8ch' (H,W,8) ready for GAN
        """
        import moderngl

        self._ensure_context()

        if cam_pos is None:
            cam_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        n = len(cell_pos)

        # Build VBO: pos(3) + col(3) + nrm(3) + size(1) = 10 floats per point
        sizes = np.full(n, cell_size, dtype=np.float32)
        vbo_data = np.zeros((n, 10), dtype=np.float32)
        vbo_data[:, 0:3] = cell_pos
        vbo_data[:, 3:6] = cell_col
        vbo_data[:, 6:9] = cell_nrm
        vbo_data[:, 9] = sizes

        vbo = self._ctx.buffer(vbo_data.tobytes())

        # Matrices
        proj = self._projection(fov_deg, self.W / self.H)
        view = self._view_matrix(cam_pos, yaw, pitch)

        # Point scale: adjust so points cover their projected cell area
        # At distance d, a cell of size s covers s/d * focal_pixels
        # focal_pixels = H / (2 * tan(fov/2))
        focal_px = self.H / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
        point_scale = focal_px * 1.5  # 1.5x to slightly overlap and fill gaps

        results = {}
        for name, prog in [("albedo", self._prog_albedo),
                           ("normals", self._prog_normal),
                           ("depth", self._prog_depth)]:

            active = set(prog)
            # Build format, padding inactive attributes
            fmt_parts, attr_names = [], []
            for aname, fmt in [("in_position", "3f"), ("in_color", "3f"),
                               ("in_normal", "3f"), ("in_size", "f")]:
                if aname in active:
                    fmt_parts.append(fmt)
                    attr_names.append(aname)
                else:
                    n_floats = int(fmt[0]) if fmt[0].isdigit() else 1
                    fmt_parts.append(f"{n_floats * 4}x")

            fmt_str = " ".join(fmt_parts)
            vao = self._ctx.vertex_array(prog, [(vbo, fmt_str, *attr_names)])

            prog["u_projection"].write(proj.T.tobytes())
            prog["u_view"].write(view.T.tobytes())
            if "u_point_scale" in active:
                prog["u_point_scale"].value = point_scale

            self._fbo.use()
            self._fbo.clear(0.0, 0.0, 0.0, 1.0)
            self._ctx.enable(moderngl.DEPTH_TEST)
            self._ctx.enable(moderngl.PROGRAM_POINT_SIZE)
            vao.render(moderngl.POINTS)
            self._ctx.finish()

            pixels = np.frombuffer(self._fbo.read(), dtype=np.uint8).reshape(self.H, self.W, 3)
            results[name] = np.flipud(pixels).astype(np.float32) / 255.0

            vao.release()

        vbo.release()

        # Build 8-channel GAN input
        albedo = results["albedo"]
        depth = results["depth"][:, :, 0]  # single channel
        normals = results["normals"]
        label = np.zeros((self.H, self.W), dtype=np.float32)

        combined = np.stack([
            albedo[:,:,0], albedo[:,:,1], albedo[:,:,2],
            depth,
            normals[:,:,0], normals[:,:,1], normals[:,:,2],
            label,
        ], axis=-1).astype(np.float32)

        return {
            "albedo": albedo,
            "depth": depth,
            "normals": normals,
            "combined_8ch": combined,
        }

    def _projection(self, fov_deg, aspect, near=0.01, far=50.0):
        f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
        m = np.zeros((4, 4), dtype=np.float32)
        m[0, 0] = f / aspect
        m[1, 1] = f
        m[2, 2] = (far + near) / (near - far)
        m[2, 3] = (2 * far * near) / (near - far)
        m[3, 2] = -1.0
        return m

    def _view_matrix(self, pos, yaw, pitch):
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
