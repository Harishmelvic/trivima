#version 430 core

// Per-cell data from SSBO (CellGeo)
layout(std430, binding = 0) buffer GeoBuffer {
    // Each cell: 16 floats = 64 bytes
    // [0] density, [1-3] density_gradient, [4] density_integral,
    // [5-7] normal, [8-10] normal_gradient,
    // [11] cell_type (as float bits), [12] confidence, [13] collision_margin,
    // [14-15] reserved
    float geo_data[];
};

// Per-cell data from SSBO (CellVisual) — only albedo needed for vertex shader
layout(std430, binding = 1) buffer VisBuffer {
    float vis_data[];
};

// Cell keys for position computation
layout(std430, binding = 2) buffer KeyBuffer {
    // Each key: 4 ints = level, x, y, z
    int key_data[];
};

uniform mat4 u_projection;
uniform mat4 u_view;
uniform float u_base_cell_size;
uniform int u_debug_confidence;  // 1 = show orange tint for low-confidence cells

// Unit cube vertex (8 vertices × 6 faces = 36 vertices via index buffer, or use geometry)
layout(location = 0) in vec3 a_cube_vertex;  // unit cube [-0.5, 0.5]

out vec3 v_color;
out vec3 v_normal;
out float v_confidence;
flat out int v_cell_id;

void main() {
    int cell_id = gl_InstanceID;

    // Read cell key
    int key_offset = cell_id * 4;
    int level = key_data[key_offset + 0];
    int cx = key_data[key_offset + 1];
    int cy = key_data[key_offset + 2];
    int cz = key_data[key_offset + 3];

    // Cell size at this level
    float cell_size = u_base_cell_size * pow(2.0, float(level));

    // Cell center in world space
    vec3 cell_center = vec3(
        (float(cx) + 0.5) * cell_size,
        (float(cy) + 0.5) * cell_size,
        (float(cz) + 0.5) * cell_size
    );

    // World position of this vertex
    vec3 world_pos = cell_center + a_cube_vertex * cell_size;

    gl_Position = u_projection * u_view * vec4(world_pos, 1.0);

    // Read cell data
    int geo_offset = cell_id * 16;
    float nx = geo_data[geo_offset + 5];
    float ny = geo_data[geo_offset + 6];
    float nz = geo_data[geo_offset + 7];
    float confidence = geo_data[geo_offset + 12];

    // Read albedo from visual buffer
    // CellVisual layout: offset 16 = albedo_r, 17 = albedo_g, 18 = albedo_b
    int vis_offset = cell_id * 112;  // 448 bytes / 4 bytes per float
    float albedo_r = vis_data[vis_offset + 4];  // after normal_curvature (4 floats)
    float albedo_g = vis_data[vis_offset + 5];
    float albedo_b = vis_data[vis_offset + 6];

    v_color = vec3(albedo_r, albedo_g, albedo_b);
    v_normal = normalize(vec3(nx, ny, nz));
    v_confidence = confidence;
    v_cell_id = cell_id;
}
