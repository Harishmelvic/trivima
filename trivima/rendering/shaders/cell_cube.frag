#version 430 core

in vec3 v_color;
in vec3 v_normal;
in float v_confidence;
flat in int v_cell_id;

uniform vec3 u_light_dir;       // directional light (normalized)
uniform int u_debug_confidence;  // 1 = orange tint for low-confidence cells
uniform int u_output_cell_id;    // 1 = write cell_id to second attachment

layout(location = 0) out vec4 frag_color;
layout(location = 1) out int frag_cell_id;  // cell ID buffer for AI write-back

void main() {
    // Basic directional lighting
    float ndotl = max(dot(v_normal, u_light_dir), 0.1);
    vec3 lit_color = v_color * ndotl;

    // Debug: orange tint for low-confidence cells (theory doc Section 2.5)
    if (u_debug_confidence == 1 && v_confidence < 0.5) {
        // Blend toward orange proportional to how low the confidence is
        float low_conf_strength = 1.0 - v_confidence * 2.0;  // 0 at conf=0.5, 1 at conf=0
        vec3 orange = vec3(1.0, 0.5, 0.0);
        lit_color = mix(lit_color, orange, low_conf_strength * 0.6);
    }

    frag_color = vec4(lit_color, 1.0);

    // Cell ID buffer (for AI texturing write-back)
    if (u_output_cell_id == 1) {
        frag_cell_id = v_cell_id;
    }
}
