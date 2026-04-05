#include "trivima/types.h"
#include "trivima/cell.h"
#include "trivima/cell_grid.h"
#include "trivima/serialization.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <cmath>

// ============================================================
// Test 1: Struct sizes match theory doc (512 bytes total)
// ============================================================
void test_struct_sizes() {
    assert(sizeof(trivima::NeighborSummary) == 16);
    assert(sizeof(trivima::CellGeo) == 64);
    assert(sizeof(trivima::CellVisual) == 448);
    assert(sizeof(trivima::CellGeo) + sizeof(trivima::CellVisual) == 512);
    assert(alignof(trivima::CellGeo) == 64);
    assert(alignof(trivima::CellVisual) == 64);
    printf("[PASS] struct sizes: CellGeo=%zu CellVisual=%zu total=%zu\n",
           sizeof(trivima::CellGeo), sizeof(trivima::CellVisual),
           sizeof(trivima::CellGeo) + sizeof(trivima::CellVisual));
}

// ============================================================
// Test 2: CellKey pack/unpack round-trip
// ============================================================
void test_cellkey_pack_unpack() {
    trivima::CellKey keys[] = {
        {0, 0, 0, 0},
        {0, 100, -50, 200},
        {-2, -1000, 500, -300},
        {2, 131071, -131072, 0},  // max range
    };

    for (const auto& k : keys) {
        uint64_t packed = k.pack();
        trivima::CellKey unpacked = trivima::CellKey::unpack(packed);
        assert(unpacked.level == k.level);
        assert(unpacked.x == k.x);
        assert(unpacked.y == k.y);
        assert(unpacked.z == k.z);
    }
    printf("[PASS] CellKey pack/unpack round-trip\n");
}

// ============================================================
// Test 3: CellKey child/parent relationships
// ============================================================
void test_cellkey_hierarchy() {
    trivima::CellKey parent = {1, 5, 3, 7};

    // Children should be at level 0
    trivima::CellKey c000 = parent.child(0, 0, 0);
    assert(c000.level == 0);
    assert(c000.x == 10 && c000.y == 6 && c000.z == 14);

    trivima::CellKey c111 = parent.child(1, 1, 1);
    assert(c111.x == 11 && c111.y == 7 && c111.z == 15);

    // Child's parent should be the original
    trivima::CellKey back = c000.parent();
    assert(back == parent);

    printf("[PASS] CellKey child/parent hierarchy\n");
}

// ============================================================
// Test 4: Grid insert, find, remove
// ============================================================
void test_grid_basic() {
    trivima::CellGridCPU grid;

    trivima::CellGeo geo{};
    geo.density = 0.8f;
    geo.set_type(trivima::CellType::Solid);
    geo.normal_y = 1.0f;

    trivima::CellVisual vis{};
    vis.albedo_r = 0.5f;
    vis.albedo_g = 0.3f;
    vis.albedo_b = 0.1f;
    vis.semantic_label = 42;

    trivima::CellKey key = {0, 10, 5, 20};
    uint32_t idx = grid.insert(key, geo, vis);
    assert(idx == 0);
    assert(grid.size() == 1);

    // Find
    uint32_t found = grid.find(key);
    assert(found == 0);
    assert(grid.geo(found).density == 0.8f);
    assert(grid.geo(found).type() == trivima::CellType::Solid);
    assert(grid.vis(found).albedo_r == 0.5f);
    assert(grid.vis(found).semantic_label == 42);

    // Not found
    trivima::CellKey missing = {0, 99, 99, 99};
    assert(grid.find(missing) == UINT32_MAX);

    // Remove
    grid.remove(key);
    assert(grid.size() == 0);
    assert(grid.find(key) == UINT32_MAX);

    printf("[PASS] grid insert/find/remove\n");
}

// ============================================================
// Test 5: Grid swap-remove maintains consistency
// ============================================================
void test_grid_swap_remove() {
    trivima::CellGridCPU grid;

    // Insert 3 cells
    trivima::CellGeo g{};
    trivima::CellVisual v{};

    trivima::CellKey k0 = {0, 0, 0, 0};
    trivima::CellKey k1 = {0, 1, 0, 0};
    trivima::CellKey k2 = {0, 2, 0, 0};

    g.density = 0.1f; grid.insert(k0, g, v);
    g.density = 0.2f; grid.insert(k1, g, v);
    g.density = 0.3f; grid.insert(k2, g, v);
    assert(grid.size() == 3);

    // Remove middle element — last element should swap in
    grid.remove(k1);
    assert(grid.size() == 2);
    assert(grid.find(k1) == UINT32_MAX);

    // k0 should still be at index 0
    uint32_t i0 = grid.find(k0);
    assert(i0 != UINT32_MAX);
    assert(grid.geo(i0).density == 0.1f);

    // k2 should still be findable (was swapped to index 1)
    uint32_t i2 = grid.find(k2);
    assert(i2 != UINT32_MAX);
    assert(grid.geo(i2).density == 0.3f);

    printf("[PASS] grid swap-remove consistency\n");
}

// ============================================================
// Test 6: Cell center and position_to_key
// ============================================================
void test_spatial() {
    // At level 0, cell size = 5cm
    trivima::CellKey k = {0, 10, 20, 30};
    trivima::vec3 c = trivima::cell_center(k);
    assert(std::abs(c.x - 0.525f) < 0.001f);  // (10 + 0.5) * 0.05
    assert(std::abs(c.y - 1.025f) < 0.001f);  // (20 + 0.5) * 0.05
    assert(std::abs(c.z - 1.525f) < 0.001f);  // (30 + 0.5) * 0.05

    // position_to_key should round-trip
    trivima::CellKey back = trivima::position_to_key(c, 0);
    assert(back == k);

    // Level 1 = 10cm cells
    float s1 = trivima::cell_size_at_level(1);
    assert(std::abs(s1 - 0.1f) < 0.001f);

    // Level -1 = 2.5cm cells
    float sm1 = trivima::cell_size_at_level(-1);
    assert(std::abs(sm1 - 0.025f) < 0.001f);

    printf("[PASS] spatial: cell_center, position_to_key, cell_size_at_level\n");
}

// ============================================================
// Test 7: Sub-cell density from gradient
// ============================================================
void test_subcell_density() {
    trivima::CellGeo geo{};
    geo.density = 0.5f;
    geo.density_gx = 10.0f;  // strong gradient in x
    geo.density_gy = 0.0f;
    geo.density_gz = 0.0f;

    // At center (offset 0): density = 0.5
    float d0 = geo.density_at_offset(0, 0, 0);
    assert(std::abs(d0 - 0.5f) < 0.001f);

    // Offset +0.025 in x (half a 5cm cell): density = 0.5 + 10*0.025 = 0.75
    float dp = geo.density_at_offset(0.025f, 0, 0);
    assert(std::abs(dp - 0.75f) < 0.001f);

    // Offset -0.025 in x: density = 0.5 - 10*0.025 = 0.25
    float dm = geo.density_at_offset(-0.025f, 0, 0);
    assert(std::abs(dm - 0.25f) < 0.001f);

    printf("[PASS] sub-cell density from gradient\n");
}

// ============================================================
// Test 8: Serialization round-trip
// ============================================================
void test_serialization() {
    trivima::CellGridCPU grid;

    // Insert some cells
    for (int i = 0; i < 100; ++i) {
        trivima::CellGeo g{};
        g.density = static_cast<float>(i) / 100.0f;
        g.set_type(trivima::CellType::Surface);
        g.normal_y = 1.0f;

        trivima::CellVisual v{};
        v.albedo_r = static_cast<float>(i) / 100.0f;
        v.semantic_label = static_cast<uint16_t>(i);

        trivima::CellKey k = {0, i, 0, 0};
        grid.insert(k, g, v);
    }

    // Save
    const char* path = "test_grid.bin";
    bool saved = trivima::save_grid(grid, path);
    assert(saved);

    // Load into new grid
    trivima::CellGridCPU loaded;
    bool ok = trivima::load_grid(loaded, path);
    assert(ok);
    assert(loaded.size() == grid.size());

    // Verify data
    for (int i = 0; i < 100; ++i) {
        trivima::CellKey k = {0, i, 0, 0};
        uint32_t idx = loaded.find(k);
        assert(idx != UINT32_MAX);
        assert(std::abs(loaded.geo(idx).density - static_cast<float>(i) / 100.0f) < 0.001f);
        assert(loaded.vis(idx).semantic_label == static_cast<uint16_t>(i));
    }

    // Cleanup
    remove(path);

    printf("[PASS] serialization round-trip (100 cells)\n");
}

// ============================================================
int main() {
    printf("=== Trivima Core Tests ===\n");
    test_struct_sizes();
    test_cellkey_pack_unpack();
    test_cellkey_hierarchy();
    test_grid_basic();
    test_grid_swap_remove();
    test_spatial();
    test_subcell_density();
    test_serialization();
    printf("=== All tests passed ===\n");
    return 0;
}
