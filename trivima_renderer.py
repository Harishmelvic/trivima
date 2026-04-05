#!/usr/bin/env python3
"""
Trivima — Cell Grid Renderer
=============================
ModernGL-based renderer for navigating cell grids.
Supports real cell grids from the perception pipeline or synthetic test rooms.

Usage:
    # Synthetic test room (no GPU models needed)
    python trivima_renderer.py --synthetic

    # From a real image (needs Depth Pro + SAM 3)
    python trivima_renderer.py --image room.jpg

    # From a serialized grid
    python trivima_renderer.py --grid room.bin

    # Headless mode (renders frames to disk)
    python trivima_renderer.py --synthetic --headless --frames 60

Controls:
    WASD        — Move forward/left/back/right
    Mouse       — Look around
    Space       — Move up
    Shift       — Move down (or sprint)
    Scroll      — Adjust speed
    ESC         — Quit
    Tab         — Toggle wireframe
    F1          — Toggle debug overlay
    C           — Toggle collision

Requirements:
    pip install moderngl moderngl-window numpy
"""

import argparse
import math
import struct
import sys
import time
from pathlib import Path

import numpy as np

try:
    import moderngl
    import moderngl_window as mglw
    from moderngl_window import geometry
except ImportError:
    print("Install dependencies: pip install moderngl moderngl-window")
    sys.exit(1)


# ============================================================================
# Shaders
# ============================================================================

VERTEX_SHADER = """
#version 330

// Per-vertex (unit cube)
in vec3 in_position;
in vec3 in_normal;

// Per-instance (cell data)
in vec3 inst_position;   // cell center in world coords
in vec3 inst_color;      // cell albedo RGB
in vec3 inst_normal;     // cell surface normal
in float inst_size;      // cell size (usually 0.05m)
in float inst_density;   // 0-1 solidity
in float inst_cell_id;   // cell index (float to avoid glVertexAttribIPointer bug)

uniform mat4 u_projection;
uniform mat4 u_view;
uniform vec3 u_light_dir;

out vec3 v_color;
out vec3 v_normal;
out vec3 v_world_pos;
out float v_density;
flat out int v_cell_id;

void main() {
    // Scale unit cube by cell size, position at cell center
    vec3 world_pos = in_position * inst_size * 0.5 + inst_position;

    gl_Position = u_projection * u_view * vec4(world_pos, 1.0);

    v_world_pos = world_pos;
    v_color = inst_color;
    v_normal = inst_normal;
    v_density = inst_density;
    v_cell_id = int(inst_cell_id);
}
"""

FRAGMENT_SHADER = """
#version 330

in vec3 v_color;
in vec3 v_normal;
in vec3 v_world_pos;
in float v_density;
flat in int v_cell_id;

uniform vec3 u_light_dir;
uniform vec3 u_camera_pos;
uniform int u_render_mode;  // 0=lit, 1=albedo, 2=normals, 3=density

out vec4 frag_color;

void main() {
    vec3 N = normalize(v_normal);
    vec3 L = normalize(u_light_dir);
    vec3 V = normalize(u_camera_pos - v_world_pos);

    if (u_render_mode == 0) {
        // Lit shading: ambient + diffuse + specular + rim
        float ambient = 0.15;
        float diffuse = max(dot(N, L), 0.0) * 0.6;
        float spec = pow(max(dot(reflect(-L, N), V), 0.0), 32.0) * 0.15;
        float rim = pow(1.0 - max(dot(N, V), 0.0), 3.0) * 0.1;
        vec3 color = v_color * (ambient + diffuse) + vec3(spec + rim);
        frag_color = vec4(color, 1.0);
    } else if (u_render_mode == 1) {
        frag_color = vec4(v_color, 1.0);           // Albedo only
    } else if (u_render_mode == 2) {
        frag_color = vec4(N * 0.5 + 0.5, 1.0);    // Normals
    } else {
        frag_color = vec4(vec3(v_density), 1.0);   // Density
    }
}
"""


# ============================================================================
# Cell Grid Data Structure
# ============================================================================

class CellGrid:
    """Minimal cell grid for rendering. Compatible with the full Trivima grid."""
    
    def __init__(self):
        self.positions = []    # (N, 3) float32 — cell centers
        self.colors = []       # (N, 3) float32 — albedo RGB [0,1]
        self.normals = []      # (N, 3) float32 — surface normals
        self.densities = []    # (N,)   float32 — solidity [0,1]
        self.sizes = []        # (N,)   float32 — cell size in meters
        self.labels = []       # (N,)   str     — semantic labels
        self.cell_size = 0.05  # default 5cm
        self._lookup = None    # spatial hash for collision
    
    @property
    def count(self):
        return len(self.positions)
    
    def finalize(self):
        """Convert lists to numpy arrays and build spatial lookup."""
        self.positions = np.array(self.positions, dtype=np.float32)
        self.colors = np.array(self.colors, dtype=np.float32)
        self.normals = np.array(self.normals, dtype=np.float32)
        self.densities = np.array(self.densities, dtype=np.float32)
        self.sizes = np.array(self.sizes, dtype=np.float32)
        self._build_lookup()
    
    def _build_lookup(self):
        """Build spatial hash map for O(1) collision queries."""
        self._lookup = {}
        cs = self.cell_size
        for i in range(len(self.positions)):
            key = (
                int(round(self.positions[i, 0] / cs)),
                int(round(self.positions[i, 1] / cs)),
                int(round(self.positions[i, 2] / cs)),
            )
            self._lookup[key] = i
    
    def query_cell(self, x, y, z):
        """Return cell index at world position, or -1 if empty."""
        if self._lookup is None:
            return -1
        cs = self.cell_size
        key = (int(round(x / cs)), int(round(y / cs)), int(round(z / cs)))
        return self._lookup.get(key, -1)
    
    def query_density(self, x, y, z):
        """Return density at world position (0 if empty)."""
        idx = self.query_cell(x, y, z)
        if idx < 0:
            return 0.0
        return float(self.densities[idx])
    
    def query_floor(self, x, z, max_height=5.0):
        """Scan downward from max_height to find the highest surface cell."""
        cs = self.cell_size
        ix = int(round(x / cs))
        iz = int(round(z / cs))
        
        best_y = -999.0
        for iy in range(int(max_height / cs), -int(max_height / cs) - 1, -1):
            key = (ix, iy, iz)
            idx = self._lookup.get(key, -1)
            if idx >= 0 and self.densities[idx] > 0.3:
                cell_y = self.positions[idx, 1]
                if cell_y > best_y:
                    best_y = cell_y
                    # Once we find the first solid from above, that's the floor
                    break
        
        return best_y if best_y > -900 else None


# ============================================================================
# Synthetic Room Generator
# ============================================================================

def generate_synthetic_room(
    width=5.0, depth=5.0, height=2.7,
    cell_size=0.05,
    add_furniture=True
):
    """Generate a test room with walls, floor, ceiling, and optional furniture."""
    grid = CellGrid()
    grid.cell_size = cell_size
    cs = cell_size
    
    # Colors
    floor_color = np.array([0.45, 0.35, 0.25])   # wood brown
    wall_color = np.array([0.85, 0.82, 0.78])     # off-white
    ceiling_color = np.array([0.95, 0.95, 0.95])   # white
    
    # Floor
    for x in np.arange(0, width, cs):
        for z in np.arange(0, depth, cs):
            # Add subtle wood grain variation
            grain = 0.03 * np.sin(x * 20) * np.cos(z * 3)
            color = floor_color + np.array([grain, grain * 0.5, 0])
            
            grid.positions.append([x, 0.0, z])
            grid.colors.append(np.clip(color, 0, 1))
            grid.normals.append([0.0, 1.0, 0.0])
            grid.densities.append(1.0)
            grid.sizes.append(cs)
            grid.labels.append('floor')
    
    # Walls (4 walls)
    for y in np.arange(0, height, cs):
        # Back wall (z=0)
        for x in np.arange(0, width, cs):
            grid.positions.append([x, y, 0.0])
            grid.colors.append(wall_color)
            grid.normals.append([0.0, 0.0, 1.0])
            grid.densities.append(1.0)
            grid.sizes.append(cs)
            grid.labels.append('wall')
        
        # Front wall (z=depth)
        for x in np.arange(0, width, cs):
            grid.positions.append([x, y, depth])
            grid.colors.append(wall_color)
            grid.normals.append([0.0, 0.0, -1.0])
            grid.densities.append(1.0)
            grid.sizes.append(cs)
            grid.labels.append('wall')
        
        # Left wall (x=0)
        for z in np.arange(0, depth, cs):
            grid.positions.append([0.0, y, z])
            grid.colors.append(wall_color * 0.95)  # slightly darker
            grid.normals.append([1.0, 0.0, 0.0])
            grid.densities.append(1.0)
            grid.sizes.append(cs)
            grid.labels.append('wall')
        
        # Right wall (x=width)
        for z in np.arange(0, depth, cs):
            grid.positions.append([width, y, z])
            grid.colors.append(wall_color * 0.95)
            grid.normals.append([-1.0, 0.0, 0.0])
            grid.densities.append(1.0)
            grid.sizes.append(cs)
            grid.labels.append('wall')
    
    # Ceiling
    for x in np.arange(0, width, cs):
        for z in np.arange(0, depth, cs):
            grid.positions.append([x, height, z])
            grid.colors.append(ceiling_color)
            grid.normals.append([0.0, -1.0, 0.0])
            grid.densities.append(1.0)
            grid.sizes.append(cs)
            grid.labels.append('ceiling')
    
    if add_furniture:
        _add_sofa(grid, cs, x=1.0, z=0.3, width_m=1.8, depth_m=0.8, height_m=0.85)
        _add_table(grid, cs, x=1.5, z=2.0, width_m=1.0, depth_m=0.6, height_m=0.45)
        _add_lamp(grid, cs, x=0.3, z=4.0, height_m=1.5)
        _add_bookshelf(grid, cs, x=4.5, z=1.0, width_m=0.3, depth_m=1.2, height_m=1.8)
    
    grid.finalize()
    print(f"Synthetic room: {grid.count} cells, {width}×{depth}×{height}m")
    return grid


def _add_box(grid, cs, x, y, z, w, h, d, color, label):
    """Add a solid box of cells."""
    for cx in np.arange(x, x + w, cs):
        for cy in np.arange(y, y + h, cs):
            for cz in np.arange(z, z + d, cs):
                # Only surface cells (faces of the box)
                is_surface = (
                    abs(cx - x) < cs or abs(cx - (x + w - cs)) < cs or
                    abs(cy - y) < cs or abs(cy - (y + h - cs)) < cs or
                    abs(cz - z) < cs or abs(cz - (z + d - cs)) < cs
                )
                if not is_surface and w > 3 * cs and h > 3 * cs and d > 3 * cs:
                    continue  # Skip interior cells for large objects
                
                normal = [0, 1, 0]
                if abs(cy - y) < cs: normal = [0, -1, 0]
                elif abs(cy - (y + h - cs)) < cs: normal = [0, 1, 0]
                elif abs(cx - x) < cs: normal = [-1, 0, 0]
                elif abs(cx - (x + w - cs)) < cs: normal = [1, 0, 0]
                elif abs(cz - z) < cs: normal = [0, 0, -1]
                elif abs(cz - (z + d - cs)) < cs: normal = [0, 0, 1]
                
                grid.positions.append([cx, cy, cz])
                grid.colors.append(color + np.random.uniform(-0.02, 0.02, 3))
                grid.normals.append(normal)
                grid.densities.append(1.0)
                grid.sizes.append(cs)
                grid.labels.append(label)


def _add_sofa(grid, cs, x, z, width_m, depth_m, height_m):
    seat_color = np.array([0.35, 0.38, 0.45])  # blue-gray
    _add_box(grid, cs, x, 0, z, width_m, 0.45, depth_m, seat_color, 'sofa')
    _add_box(grid, cs, x, 0.45, z, width_m, height_m - 0.45, cs * 3, seat_color * 0.9, 'sofa')


def _add_table(grid, cs, x, z, width_m, depth_m, height_m):
    table_color = np.array([0.55, 0.40, 0.28])  # warm wood
    _add_box(grid, cs, x, height_m - cs * 2, z, width_m, cs * 2, depth_m, table_color, 'table')
    leg_w = cs * 2
    for lx, lz in [(x, z), (x + width_m - leg_w, z), (x, z + depth_m - leg_w), (x + width_m - leg_w, z + depth_m - leg_w)]:
        _add_box(grid, cs, lx, 0, lz, leg_w, height_m - cs * 2, leg_w, table_color * 0.8, 'table')


def _add_lamp(grid, cs, x, z, height_m):
    lamp_color = np.array([0.2, 0.2, 0.2])  # dark metal
    shade_color = np.array([0.9, 0.85, 0.7])  # warm white
    _add_box(grid, cs, x, 0, z, cs * 4, cs * 2, cs * 4, lamp_color, 'lamp')
    _add_box(grid, cs, x + cs, 0, z + cs, cs * 2, height_m * 0.8, cs * 2, lamp_color, 'lamp')
    _add_box(grid, cs, x - cs, height_m * 0.7, z - cs, cs * 6, cs * 4, cs * 6, shade_color, 'lamp')


def _add_bookshelf(grid, cs, x, z, width_m, depth_m, height_m):
    shelf_color = np.array([0.50, 0.38, 0.25])
    _add_box(grid, cs, x, 0, z, width_m, height_m, depth_m, shelf_color, 'bookshelf')


# ============================================================================
# Camera
# ============================================================================

class Camera:
    def __init__(self, position=None, yaw=-90.0, pitch=0.0):
        self.position = np.array(position or [2.5, 1.6, 3.5], dtype=np.float64)
        self.yaw = yaw
        self.pitch = pitch
        self.speed = 2.0
        self.sensitivity = 0.15
        self.fov = 70.0
        self.near = 0.01
        self.far = 100.0
    
    @property
    def front(self):
        rad_yaw = math.radians(self.yaw)
        rad_pitch = math.radians(self.pitch)
        return np.array([
            math.cos(rad_pitch) * math.cos(rad_yaw),
            math.sin(rad_pitch),
            math.cos(rad_pitch) * math.sin(rad_yaw),
        ], dtype=np.float64)
    
    @property
    def right(self):
        f = self.front
        up = np.array([0, 1, 0], dtype=np.float64)
        r = np.cross(f, up)
        norm = np.linalg.norm(r)
        return r / norm if norm > 1e-6 else np.array([1, 0, 0])
    
    @property
    def up(self):
        return np.cross(self.right, self.front)
    
    def view_matrix(self):
        f = self.front
        r = self.right
        u = self.up
        p = self.position
        
        m = np.eye(4, dtype=np.float32)
        m[0, :3] = r
        m[1, :3] = u
        m[2, :3] = -f
        m[0, 3] = -np.dot(r, p)
        m[1, 3] = -np.dot(u, p)
        m[2, 3] = np.dot(f, p)
        return m
    
    def projection_matrix(self, aspect):
        fov_rad = math.radians(self.fov)
        f = 1.0 / math.tan(fov_rad / 2.0)
        n, far = self.near, self.far
        
        m = np.zeros((4, 4), dtype=np.float32)
        m[0, 0] = f / aspect
        m[1, 1] = f
        m[2, 2] = (far + n) / (n - far)
        m[2, 3] = (2 * far * n) / (n - far)
        m[3, 2] = -1.0
        return m


# ============================================================================
# Collision System
# ============================================================================

class CollisionSystem:
    def __init__(self, grid: CellGrid, radius=0.2, height=1.7):
        self.grid = grid
        self.radius = radius
        self.height = height
        self.enabled = True
        self.eye_height = 1.6
        self.smooth_y = None
        self.smooth_alpha = 0.15
    
    def check_move(self, current_pos, desired_pos):
        """Check if movement is valid. Returns adjusted position."""
        if not self.enabled:
            return desired_pos.copy()
        
        result = desired_pos.copy()
        
        # Check horizontal collision (XZ plane)
        probe_points = [
            [self.radius, 0, 0], [-self.radius, 0, 0],
            [0, 0, self.radius], [0, 0, -self.radius],
            [self.radius * 0.7, 0, self.radius * 0.7],
            [-self.radius * 0.7, 0, self.radius * 0.7],
            [self.radius * 0.7, 0, -self.radius * 0.7],
            [-self.radius * 0.7, 0, -self.radius * 0.7],
        ]
        
        # Check at multiple heights (feet, waist, head)
        for height_offset in [0.3, 0.9, 1.5]:
            for dx, _, dz in probe_points:
                probe_x = result[0] + dx
                probe_y = result[1] - self.eye_height + height_offset
                probe_z = result[2] + dz
                
                density = self.grid.query_density(probe_x, probe_y, probe_z)
                if density > 0.5:
                    # Block movement in this direction — wall slide
                    # Check if X or Z is the problem
                    dx_density = self.grid.query_density(
                        current_pos[0] + dx, probe_y, current_pos[2] + dz
                    )
                    dz_density = self.grid.query_density(
                        result[0] + dx, probe_y, current_pos[2] + dz
                    )
                    
                    if dz_density > 0.5:
                        result[0] = current_pos[0]  # block X
                    if dx_density > 0.5:
                        result[2] = current_pos[2]  # block Z
                    
                    if dx_density > 0.5 and dz_density > 0.5:
                        result[0] = current_pos[0]
                        result[2] = current_pos[2]
        
        return result
    
    def get_floor_height(self, x, z):
        """Get the floor height at XZ position with smoothing."""
        floor_y = self.grid.query_floor(x, z)
        
        if floor_y is None:
            return self.smooth_y if self.smooth_y is not None else 0.0
        
        target_y = floor_y + self.eye_height
        
        if self.smooth_y is None:
            self.smooth_y = target_y
        else:
            self.smooth_y += (target_y - self.smooth_y) * self.smooth_alpha
        
        return self.smooth_y


# ============================================================================
# Renderer Window
# ============================================================================

class TrivimRenderer(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "Trivima — Cell Grid Navigator"
    window_size = (1280, 720)
    aspect_ratio = None
    resizable = True
    cursor = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Get grid from class variable (set before window creation)
        self.grid = TrivimRenderer._grid
        
        # Camera starts at room center
        positions = self.grid.positions
        center_x = (positions[:, 0].min() + positions[:, 0].max()) / 2
        center_z = (positions[:, 2].min() + positions[:, 2].max()) / 2
        
        self.camera = Camera(position=[center_x, 1.6, center_z * 0.7])
        self.collision = CollisionSystem(self.grid)
        
        # Render state
        self.render_mode = 0  # 0=lit, 1=albedo, 2=normals, 3=density
        self.wireframe = False
        self.show_debug = True
        self.frame_count = 0
        self.fps_timer = time.time()
        self.fps = 0.0
        
        # Movement state
        self.keys = set()
        self.mouse_captured = True
        self.first_mouse = True
        self.last_mx = 0
        self.last_my = 0
        
        # Build GPU buffers
        self._setup_rendering()
    
    def _setup_rendering(self):
        """Upload cell data to GPU and create shader program."""
        ctx = self.ctx
        
        # Compile shaders
        self.prog = ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=FRAGMENT_SHADER,
        )
        
        # Unit cube geometry
        cube_verts, cube_normals, cube_indices = _unit_cube()
        
        self.vbo_cube_pos = ctx.buffer(cube_verts.astype(np.float32).tobytes())
        self.vbo_cube_nrm = ctx.buffer(cube_normals.astype(np.float32).tobytes())
        self.ibo = ctx.buffer(cube_indices.astype(np.int32).tobytes())
        
        # Instance data
        n = self.grid.count
        print(f"Uploading {n} cells to GPU...")
        
        inst_positions = self.grid.positions.astype(np.float32)
        inst_colors = self.grid.colors.astype(np.float32)
        inst_normals = self.grid.normals.astype(np.float32)
        inst_sizes = self.grid.sizes.astype(np.float32)
        inst_densities = self.grid.densities.astype(np.float32)
        inst_cell_ids = np.arange(n, dtype=np.float32)
        
        self.vbo_inst_pos = ctx.buffer(inst_positions.tobytes())
        self.vbo_inst_col = ctx.buffer(inst_colors.tobytes())
        self.vbo_inst_nrm = ctx.buffer(inst_normals.tobytes())
        self.vbo_inst_size = ctx.buffer(inst_sizes.tobytes())
        self.vbo_inst_density = ctx.buffer(inst_densities.tobytes())
        self.vbo_inst_id = ctx.buffer(inst_cell_ids.tobytes())
        
        # Vertex Array Object
        self.vao = ctx.vertex_array(
            self.prog,
            [
                (self.vbo_cube_pos, '3f', 'in_position'),
                (self.vbo_cube_nrm, '3f', 'in_normal'),
                (self.vbo_inst_pos, '3f/i', 'inst_position'),
                (self.vbo_inst_col, '3f/i', 'inst_color'),
                (self.vbo_inst_nrm, '3f/i', 'inst_normal'),
                (self.vbo_inst_size, 'f/i', 'inst_size'),
                (self.vbo_inst_density, 'f/i', 'inst_density'),
                (self.vbo_inst_id, 'f/i', 'inst_cell_id'),
            ],
            index_buffer=self.ibo,
        )
        
        # Cell ID framebuffer (integer attachment)
        self.cell_id_texture = ctx.texture(
            self.window_size, 1, dtype='i4'
        )
        self.depth_texture = ctx.depth_texture(self.window_size)
        self.fbo_cell_id = ctx.framebuffer(
            color_attachments=[self.cell_id_texture],
            depth_attachment=self.depth_texture,
        )
        
        # Enable depth test
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.CULL_FACE)
        
        self.instance_count = n
        print(f"GPU ready: {n} cells, {n * 36} triangles")
    
    def render(self, time_val, frametime):
        """Called every frame."""
        self._update_movement(frametime)
        self._update_fps()
        
        ctx = self.ctx
        aspect = self.window_size[0] / max(self.window_size[1], 1)
        
        proj = self.camera.projection_matrix(aspect)
        view = self.camera.view_matrix()
        
        # Set uniforms
        self.prog['u_projection'].write(proj.tobytes())
        self.prog['u_view'].write(view.tobytes())
        self.prog['u_light_dir'].value = (0.4, 0.8, 0.3)
        self.prog['u_camera_pos'].value = tuple(self.camera.position.astype(np.float32))
        self.prog['u_render_mode'].value = self.render_mode
        
        # Clear
        ctx.clear(0.6, 0.75, 0.9, 1.0)  # sky blue
        
        if self.wireframe:
            ctx.wireframe = True
        
        # Render instanced cubes
        self.vao.render(moderngl.TRIANGLES, instances=self.instance_count)
        
        if self.wireframe:
            ctx.wireframe = False
        
        self.frame_count += 1
        
        # Debug text overlay (simple — just print to console every 60 frames)
        if self.show_debug and self.frame_count % 60 == 0:
            pos = self.camera.position
            print(
                f"\r  FPS: {self.fps:.0f} | "
                f"Pos: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) | "
                f"Cells: {self.instance_count} | "
                f"Collision: {'ON' if self.collision.enabled else 'OFF'} | "
                f"Mode: {['Lit','Albedo','Normals','Density'][self.render_mode]}",
                end='', flush=True
            )
    
    def _update_movement(self, dt):
        """Process keyboard input for camera movement."""
        speed = self.camera.speed * dt
        
        front = self.camera.front.copy()
        front[1] = 0  # no vertical component for walking
        front_norm = np.linalg.norm(front)
        if front_norm > 1e-6:
            front /= front_norm
        
        right = self.camera.right
        
        desired = self.camera.position.copy()
        
        if self.wnd.keys.W in self.keys:
            desired += front * speed
        if self.wnd.keys.S in self.keys:
            desired -= front * speed
        if self.wnd.keys.A in self.keys:
            desired -= right * speed
        if self.wnd.keys.D in self.keys:
            desired += right * speed
        if self.wnd.keys.SPACE in self.keys:
            desired[1] += speed
        if self.wnd.keys.LEFT_SHIFT in self.keys:
            desired[1] -= speed
        
        # Apply collision
        new_pos = self.collision.check_move(self.camera.position, desired)
        
        # Apply floor following (only if not flying with Space/Shift)
        if self.wnd.keys.SPACE not in self.keys and self.wnd.keys.LEFT_SHIFT not in self.keys:
            floor_y = self.collision.get_floor_height(new_pos[0], new_pos[2])
            if floor_y is not None:
                new_pos[1] = floor_y
        
        self.camera.position = new_pos
    
    def _update_fps(self):
        now = time.time()
        if now - self.fps_timer >= 1.0:
            self.fps = self.frame_count / (now - self.fps_timer)
            self.frame_count = 0
            self.fps_timer = now
    
    def key_event(self, key, action, modifiers):
        if action == self.wnd.keys.ACTION_PRESS:
            self.keys.add(key)
            
            if key == self.wnd.keys.ESCAPE:
                self.wnd.close()
            elif key == self.wnd.keys.TAB:
                self.wireframe = not self.wireframe
            elif key == self.wnd.keys.F1:
                self.show_debug = not self.show_debug
            elif key == self.wnd.keys.C:
                self.collision.enabled = not self.collision.enabled
                print(f"\n  Collision: {'ON' if self.collision.enabled else 'OFF'}")
            elif key == self.wnd.keys.NUMBER_1:
                self.render_mode = 0
            elif key == self.wnd.keys.NUMBER_2:
                self.render_mode = 1
            elif key == self.wnd.keys.NUMBER_3:
                self.render_mode = 2
            elif key == self.wnd.keys.NUMBER_4:
                self.render_mode = 3
        
        elif action == self.wnd.keys.ACTION_RELEASE:
            self.keys.discard(key)
    
    def mouse_position_event(self, x, y, dx, dy):
        if self.mouse_captured:
            self.camera.yaw += dx * self.camera.sensitivity
            self.camera.pitch -= dy * self.camera.sensitivity
            self.camera.pitch = max(-89, min(89, self.camera.pitch))
    
    def mouse_scroll_event(self, x_offset, y_offset):
        self.camera.speed = max(0.5, min(10.0, self.camera.speed + y_offset * 0.3))
        print(f"\n  Speed: {self.camera.speed:.1f} m/s")


# ============================================================================
# Unit Cube Geometry
# ============================================================================

def _unit_cube():
    """Generate unit cube vertices, normals, and indices."""
    # 8 corners of a unit cube centered at origin
    v = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # back
        [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1],  # front
    ], dtype=np.float32)
    
    # 6 faces, each with 4 vertices and a normal
    faces = [
        ([4, 5, 6, 7], [0, 0, 1]),   # front +Z
        ([1, 0, 3, 2], [0, 0, -1]),  # back -Z
        ([0, 4, 7, 3], [-1, 0, 0]),  # left -X
        ([5, 1, 2, 6], [1, 0, 0]),   # right +X
        ([3, 7, 6, 2], [0, 1, 0]),   # top +Y
        ([0, 1, 5, 4], [0, -1, 0]),  # bottom -Y
    ]
    
    verts = []
    normals = []
    indices = []
    
    for face_verts, normal in faces:
        base = len(verts)
        for vi in face_verts:
            verts.append(v[vi])
            normals.append(normal)
        indices.extend([base, base + 1, base + 2, base, base + 2, base + 3])
    
    return np.array(verts), np.array(normals), np.array(indices, dtype=np.int32)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Trivima Cell Grid Renderer")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic test room")
    parser.add_argument("--image", type=str, help="Build grid from room photo")
    parser.add_argument("--grid", type=str, help="Load serialized grid")
    parser.add_argument("--cell_size", type=float, default=0.05, help="Cell size in meters")
    parser.add_argument("--width", type=float, default=5.0, help="Synthetic room width")
    parser.add_argument("--depth", type=float, default=5.0, help="Synthetic room depth")
    parser.add_argument("--height", type=float, default=2.7, help="Synthetic room height")
    parser.add_argument("--headless", action="store_true", help="Render offscreen")
    parser.add_argument("--frames", type=int, default=0, help="Frames to render in headless mode")
    args = parser.parse_args()
    
    # Build or load grid
    if args.image:
        print(f"Building grid from image: {args.image}")
        import torch
        from trivima.perception.pipeline import PerceptionPipeline
        from trivima.construction.point_to_grid import build_cell_grid, apply_failure_mode_density_forcing

        pipeline = PerceptionPipeline()
        pipeline.load_models()
        result = pipeline.run(args.image)

        grid_data, stats = build_cell_grid(
            result.positions, result.colors, result.normals,
            result.labels, result.confidence, cell_size=args.cell_size,
        )
        apply_failure_mode_density_forcing(
            grid_data, None, result.label_names, result.positions, args.cell_size,
        )
        pipeline.unload()
        torch.cuda.empty_cache()

        # Convert dict grid to CellGrid for renderer
        grid = CellGrid()
        grid.cell_size = args.cell_size
        for (ix, iy, iz), cell in grid_data.items():
            cs = args.cell_size
            grid.positions.append([(ix + 0.5) * cs, (iy + 0.5) * cs, (iz + 0.5) * cs])
            grid.colors.append(cell["albedo"])
            grid.normals.append(cell["normal"])
            grid.densities.append(cell["density"])
            grid.sizes.append(cs)
            grid.labels.append(str(cell.get("label", "")))
        grid.finalize()
        
    elif args.grid:
        print(f"Loading grid from: {args.grid}")
        # Placeholder for grid deserialization
        raise NotImplementedError("Grid loading not yet implemented")
        
    else:
        # Default to synthetic
        print("Generating synthetic test room...")
        grid = generate_synthetic_room(
            width=args.width, depth=args.depth, height=args.height,
            cell_size=args.cell_size
        )
    
    # Store grid for the window class to access
    TrivimRenderer._grid = grid
    
    print(f"\nStarting renderer...")
    print(f"  Controls: WASD=move, Mouse=look, Tab=wireframe, C=collision, 1-4=render mode, ESC=quit")
    print(f"  Cell count: {grid.count}")
    
    # Run window
    mglw.run_window_config(TrivimRenderer)


if __name__ == "__main__":
    main()
