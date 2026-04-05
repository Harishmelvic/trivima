#include "trivima/cell_grid.h"
#include <fstream>
#include <cstring>

namespace trivima {

// ============================================================
// Binary serialization format:
//   Header (32 bytes):
//     magic:       "TRVM" (4 bytes)
//     version:     uint32_t (4 bytes)
//     base_size:   float (4 bytes)
//     cell_count:  uint64_t (8 bytes)
//     aabb_min:    3 × float (12 bytes)
//
//   Cell table (cell_count × (16 + 64 + 448) bytes):
//     For each cell:
//       key:     CellKey (16 bytes: level + x + y + z)
//       geo:     CellGeo (64 bytes)
//       visual:  CellVisual (448 bytes)
// ============================================================

static constexpr uint32_t MAGIC = 0x4D565254; // "TRVM" little-endian
static constexpr uint32_t FORMAT_VERSION = 1;
static constexpr size_t HEADER_SIZE = 32;
static constexpr size_t CELL_RECORD_SIZE = sizeof(CellKey) + sizeof(CellGeo) + sizeof(CellVisual);
// 16 + 64 + 448 = 528 bytes per cell record

bool save_grid(const CellGridCPU& grid, const char* path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;

    // Header
    uint32_t magic = MAGIC;
    uint32_t version = FORMAT_VERSION;
    float base_size = grid.base_cell_size;
    uint64_t count = grid.size();

    vec3 aabb_min, aabb_max;
    grid.compute_bounds(aabb_min, aabb_max);

    out.write(reinterpret_cast<const char*>(&magic), 4);
    out.write(reinterpret_cast<const char*>(&version), 4);
    out.write(reinterpret_cast<const char*>(&base_size), 4);
    out.write(reinterpret_cast<const char*>(&count), 8);
    out.write(reinterpret_cast<const char*>(&aabb_min.x), 12);

    // Cell records
    for (size_t i = 0; i < count; ++i) {
        const CellKey& k = grid.key(static_cast<uint32_t>(i));
        out.write(reinterpret_cast<const char*>(&k), sizeof(CellKey));
        out.write(reinterpret_cast<const char*>(&grid.geo(static_cast<uint32_t>(i))), sizeof(CellGeo));
        out.write(reinterpret_cast<const char*>(&grid.vis(static_cast<uint32_t>(i))), sizeof(CellVisual));
    }

    return out.good();
}

bool load_grid(CellGridCPU& grid, const char* path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;

    // Header
    uint32_t magic, version;
    float base_size;
    uint64_t count;
    float aabb_min[3];

    in.read(reinterpret_cast<char*>(&magic), 4);
    if (magic != MAGIC) return false;

    in.read(reinterpret_cast<char*>(&version), 4);
    if (version != FORMAT_VERSION) return false;

    in.read(reinterpret_cast<char*>(&base_size), 4);
    in.read(reinterpret_cast<char*>(&count), 8);
    in.read(reinterpret_cast<char*>(aabb_min), 12);

    grid.clear();
    grid.base_cell_size = base_size;
    grid.reserve(static_cast<size_t>(count));

    // Cell records
    CellKey key;
    CellGeo geo;
    CellVisual vis;

    for (uint64_t i = 0; i < count; ++i) {
        in.read(reinterpret_cast<char*>(&key), sizeof(CellKey));
        in.read(reinterpret_cast<char*>(&geo), sizeof(CellGeo));
        in.read(reinterpret_cast<char*>(&vis), sizeof(CellVisual));
        if (!in) return false;
        grid.insert(key, geo, vis);
    }

    return true;
}

} // namespace trivima
