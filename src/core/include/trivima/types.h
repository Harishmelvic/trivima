#pragma once
#include <cstdint>
#include <cmath>
#include <functional>

#ifdef __CUDACC__
#define TRIVIMA_HD __host__ __device__
#else
#define TRIVIMA_HD
#endif

namespace trivima {

// ============================================================
// Vector types — GPU-compatible, minimal, no dependencies
// ============================================================

struct alignas(16) vec3 {
    float x, y, z;
    float _pad;  // pad to 16 bytes for GPU alignment

    TRIVIMA_HD vec3() : x(0), y(0), z(0), _pad(0) {}
    TRIVIMA_HD vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_), _pad(0) {}

    TRIVIMA_HD float& operator[](int i) { return (&x)[i]; }
    TRIVIMA_HD const float& operator[](int i) const { return (&x)[i]; }

    TRIVIMA_HD vec3 operator+(const vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    TRIVIMA_HD vec3 operator-(const vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    TRIVIMA_HD vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
    TRIVIMA_HD vec3 operator/(float s) const { float inv = 1.0f / s; return {x * inv, y * inv, z * inv}; }

    TRIVIMA_HD vec3& operator+=(const vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    TRIVIMA_HD vec3& operator-=(const vec3& o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
    TRIVIMA_HD vec3& operator*=(float s) { x *= s; y *= s; z *= s; return *this; }
};

TRIVIMA_HD inline float dot(const vec3& a, const vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

TRIVIMA_HD inline float length(const vec3& v) {
    return sqrtf(dot(v, v));
}

TRIVIMA_HD inline vec3 normalize(const vec3& v) {
    float len = length(v);
    if (len < 1e-8f) return {0.0f, 0.0f, 0.0f};
    return v / len;
}

TRIVIMA_HD inline vec3 cross(const vec3& a, const vec3& b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

struct alignas(16) vec4 {
    float x, y, z, w;

    TRIVIMA_HD vec4() : x(0), y(0), z(0), w(0) {}
    TRIVIMA_HD vec4(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_) {}
    TRIVIMA_HD vec4(const vec3& v, float w_) : x(v.x), y(v.y), z(v.z), w(w_) {}

    TRIVIMA_HD vec3 xyz() const { return {x, y, z}; }
};

struct ivec3 {
    int32_t x, y, z;

    bool operator==(const ivec3& o) const {
        return x == o.x && y == o.y && z == o.z;
    }
};

// ============================================================
// Cell classification
// ============================================================

enum class CellType : uint8_t {
    Empty       = 0,
    Surface     = 1,
    Solid       = 2,
    Transparent = 3
};

// Semantic label from SAM — 16-bit to cover 270K+ categories
using SemanticLabel = uint16_t;

// ============================================================
// Grid key — encodes resolution level + 3D coordinates
// ============================================================

struct CellKey {
    int32_t level;  // 0 = base (5cm), -1 = 2.5cm, -2 = 1.25cm, +1 = 10cm, +2 = 20cm
    int32_t x, y, z;

    bool operator==(const CellKey& o) const {
        return level == o.level && x == o.x && y == o.y && z == o.z;
    }

    // Pack into uint64 for GPU hash map: level in high 8 bits, x/y/z each 18 bits
    uint64_t pack() const {
        uint64_t l = static_cast<uint64_t>(level + 128) & 0xFF;         // 8 bits (level offset to unsigned)
        uint64_t cx = static_cast<uint64_t>(x + 131072) & 0x3FFFF;     // 18 bits
        uint64_t cy = static_cast<uint64_t>(y + 131072) & 0x3FFFF;     // 18 bits
        uint64_t cz = static_cast<uint64_t>(z + 131072) & 0x3FFFF;     // 18 bits
        return (l << 54) | (cx << 36) | (cy << 18) | cz;               // total: 62 bits
    }

    static CellKey unpack(uint64_t packed) {
        CellKey k;
        k.level = static_cast<int32_t>((packed >> 54) & 0xFF) - 128;
        k.x = static_cast<int32_t>((packed >> 36) & 0x3FFFF) - 131072;
        k.y = static_cast<int32_t>((packed >> 18) & 0x3FFFF) - 131072;
        k.z = static_cast<int32_t>(packed & 0x3FFFF) - 131072;
        return k;
    }

    // Children at next finer level
    CellKey child(int dx, int dy, int dz) const {
        return {level - 1, x * 2 + dx, y * 2 + dy, z * 2 + dz};
    }

    // Parent at next coarser level
    CellKey parent() const {
        return {level + 1, x >> 1, y >> 1, z >> 1};
    }
};

struct CellKeyHash {
    size_t operator()(const CellKey& k) const {
        // Spatial hash with prime multipliers for good distribution
        size_t h = static_cast<size_t>(k.level) * 73856093ULL
                 ^ static_cast<size_t>(k.x + 131072) * 19349663ULL
                 ^ static_cast<size_t>(k.y + 131072) * 83492791ULL
                 ^ static_cast<size_t>(k.z + 131072) * 50331653ULL;
        return h;
    }
};

// ============================================================
// Constants
// ============================================================

constexpr float BASE_CELL_SIZE = 0.05f;           // 5cm base resolution
constexpr float EYE_HEIGHT = 1.6f;                // camera height above floor
constexpr float COLLISION_DENSITY_THRESHOLD = 0.5f;
constexpr int   MAX_VISIBLE_CELLS = 200000;        // LOD budget cap
constexpr int   NEURAL_TEXTURE_DIM = 32;           // texture feature vector size

// Cell size at a given level: base_size * 2^level
TRIVIMA_HD inline float cell_size_at_level(int level) {
    // level 0 = 5cm, level -1 = 2.5cm, level +1 = 10cm, level +2 = 20cm
    return BASE_CELL_SIZE * powf(2.0f, static_cast<float>(level));
}

// World-space center of a cell
TRIVIMA_HD inline vec3 cell_center(const CellKey& key) {
    float s = cell_size_at_level(key.level);
    return {
        (key.x + 0.5f) * s,
        (key.y + 0.5f) * s,
        (key.z + 0.5f) * s
    };
}

// CellKey from a world-space position at a given level
inline CellKey position_to_key(const vec3& pos, int level) {
    float s = cell_size_at_level(level);
    return {
        level,
        static_cast<int32_t>(floorf(pos.x / s)),
        static_cast<int32_t>(floorf(pos.y / s)),
        static_cast<int32_t>(floorf(pos.z / s))
    };
}

} // namespace trivima
