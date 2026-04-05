#pragma once
#include "trivima/types.h"

namespace trivima {

// ============================================================
// Neighbor summary — compact view of an adjacent cell (16 bytes)
// ============================================================

struct NeighborSummary {
    uint8_t type;       // CellType
    uint8_t _pad[3];
    float density;
    float normal_y;
    float light_luma;
};

static_assert(sizeof(NeighborSummary) == 16, "NeighborSummary must be 16 bytes");

// ============================================================
// CellGeo — HOT data, read every frame (64 bytes)
//
// Raw floats only — no vec3 members — guarantees packed layout.
// ============================================================

struct alignas(64) CellGeo {
    // Density: value + gradient + integral  (5 floats = 20 bytes)
    float density;
    float density_gx, density_gy, density_gz;
    float density_integral;

    // Normal: value + gradient  (6 floats = 24 bytes)
    float normal_x, normal_y, normal_z;
    float normal_gx, normal_gy, normal_gz;

    // Classification (4 bytes)
    uint8_t cell_type_raw;
    uint8_t _pad0[3];

    // Confidence + collision margin + reserved (4 floats = 16 bytes)
    float confidence;        // [0,1] reliability of this cell's data
                             // Derived from: depth smoothness, point density,
                             // semantic penalty (glass/mirror → low), DUSt3R agreement
    float collision_margin;  // expanded margin for low-confidence cells (meters)
    float _reserved[2];     // future use

    // 20 + 24 + 4 + 16 = 64 bytes

    TRIVIMA_HD CellType type() const { return static_cast<CellType>(cell_type_raw); }
    TRIVIMA_HD void set_type(CellType t) { cell_type_raw = static_cast<uint8_t>(t); }
    TRIVIMA_HD bool is_solid() const { return type() == CellType::Solid || type() == CellType::Surface; }
    TRIVIMA_HD bool is_empty() const { return type() == CellType::Empty; }
    TRIVIMA_HD bool is_low_confidence() const { return confidence < 0.5f; }

    TRIVIMA_HD vec3 get_density_gradient() const { return {density_gx, density_gy, density_gz}; }
    TRIVIMA_HD vec3 get_normal() const { return {normal_x, normal_y, normal_z}; }
    TRIVIMA_HD vec3 get_normal_gradient() const { return {normal_gx, normal_gy, normal_gz}; }

    TRIVIMA_HD void set_density_gradient(const vec3& g) { density_gx = g.x; density_gy = g.y; density_gz = g.z; }
    TRIVIMA_HD void set_normal(const vec3& n) { normal_x = n.x; normal_y = n.y; normal_z = n.z; }
    TRIVIMA_HD void set_normal_gradient(const vec3& g) { normal_gx = g.x; normal_gy = g.y; normal_gz = g.z; }

    TRIVIMA_HD float density_at_offset(float ox, float oy, float oz) const {
        return density + density_gx * ox + density_gy * oy + density_gz * oz;
    }

    // Effective collision radius: base cell half-size + expanded margin for uncertain cells
    TRIVIMA_HD float effective_collision_margin(float cell_half_size) const {
        return cell_half_size + collision_margin;
    }
};

static_assert(sizeof(CellGeo) == 64, "CellGeo must be exactly 64 bytes");

// ============================================================
// CellVisual — COLD data, read during rendering (448 bytes)
//
// Section bytes:
//   Normal curvature      16
//   Albedo               48
//   Neural texture       128
//   Lighting              48
//   Material              32
//   Animation             80
//   Spatial (neighbors)   96
//   TOTAL                448
// ============================================================

struct alignas(64) CellVisual {
    // --- Normal curvature: 3 floats + 1 pad = 16 bytes ---
    float normal_cx, normal_cy, normal_cz;
    float _pad_nc;

    // --- Albedo: 12 floats = 48 bytes ---
    float albedo_r, albedo_g, albedo_b;
    float _pad_a0;
    float albedo_gx, albedo_gy, albedo_gz;
    float _pad_a1;
    float albedo_2dx, albedo_2dy, albedo_2dz;
    float albedo_integral;

    // --- Neural texture features: 32 floats = 128 bytes ---
    float texture_features[NEURAL_TEXTURE_DIM];

    // --- Lighting: 12 floats = 48 bytes ---
    float light_r, light_g, light_b, light_a;
    float light_gx, light_gy, light_gz;
    float light_integral;
    float light_temporal_deriv;
    float _pad_lt[3];

    // --- Material: 8 floats = 32 bytes ---
    SemanticLabel semantic_label;       // 2 bytes
    uint16_t _pad_m0;                  // 2 bytes
    float roughness;                   // 4 bytes -> 8 total
    float roughness_gx, roughness_gy, roughness_gz; // 12 bytes -> 20
    float reflectance;                 // 4 bytes -> 24
    float reflectance_gradient;        // 4 bytes -> 28
    float _pad_m1;                     // 4 bytes -> 32

    // --- Animation: 20 floats = 80 bytes ---
    float vel_x, vel_y, vel_z;        // 12
    float acc_x, acc_y, acc_z;        // 12 -> 24
    float deform_params[6];           // 24 -> 48
    float tex_shift_x, tex_shift_y, tex_shift_z; // 12 -> 60
    float _pad_an[5];                 // 20 -> 80

    // --- Spatial: 6 × NeighborSummary = 96 bytes ---
    NeighborSummary neighbors[6];

    // Verify: 16 + 48 + 128 + 48 + 32 + 80 + 96 = 448

    // --- Accessors ---
    TRIVIMA_HD vec3 get_normal_curvature() const { return {normal_cx, normal_cy, normal_cz}; }
    TRIVIMA_HD vec3 get_albedo() const { return {albedo_r, albedo_g, albedo_b}; }
    TRIVIMA_HD vec3 get_albedo_gradient() const { return {albedo_gx, albedo_gy, albedo_gz}; }
    TRIVIMA_HD vec3 get_albedo_2nd_deriv() const { return {albedo_2dx, albedo_2dy, albedo_2dz}; }
    TRIVIMA_HD vec3 get_light() const { return {light_r, light_g, light_b}; }
    TRIVIMA_HD vec3 get_light_gradient() const { return {light_gx, light_gy, light_gz}; }
    TRIVIMA_HD vec3 get_velocity() const { return {vel_x, vel_y, vel_z}; }
    TRIVIMA_HD vec3 get_acceleration() const { return {acc_x, acc_y, acc_z}; }
    TRIVIMA_HD vec3 get_texture_shift() const { return {tex_shift_x, tex_shift_y, tex_shift_z}; }

    TRIVIMA_HD void set_normal_curvature(const vec3& c) { normal_cx = c.x; normal_cy = c.y; normal_cz = c.z; }
    TRIVIMA_HD void set_albedo(const vec3& c) { albedo_r = c.x; albedo_g = c.y; albedo_b = c.z; }
    TRIVIMA_HD void set_albedo_gradient(const vec3& g) { albedo_gx = g.x; albedo_gy = g.y; albedo_gz = g.z; }
    TRIVIMA_HD void set_albedo_2nd_deriv(const vec3& d) { albedo_2dx = d.x; albedo_2dy = d.y; albedo_2dz = d.z; }
    TRIVIMA_HD void set_light(const vec3& l) { light_r = l.x; light_g = l.y; light_b = l.z; light_a = 1.0f; }
    TRIVIMA_HD void set_light_gradient(const vec3& g) { light_gx = g.x; light_gy = g.y; light_gz = g.z; }
    TRIVIMA_HD void set_velocity(const vec3& v) { vel_x = v.x; vel_y = v.y; vel_z = v.z; }
    TRIVIMA_HD void set_acceleration(const vec3& a) { acc_x = a.x; acc_y = a.y; acc_z = a.z; }
    TRIVIMA_HD void set_texture_shift(const vec3& s) { tex_shift_x = s.x; tex_shift_y = s.y; tex_shift_z = s.z; }
};

static_assert(sizeof(CellVisual) == 448, "CellVisual must be exactly 448 bytes");
static_assert(sizeof(CellGeo) + sizeof(CellVisual) == 512, "Total cell data must be 512 bytes");

} // namespace trivima
