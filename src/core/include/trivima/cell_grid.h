#pragma once
#include "trivima/cell.h"
#include <unordered_map>
#include <vector>
#include <cstring>
#include <cassert>

namespace trivima {

// ============================================================
// CellGridCPU — Sparse cell grid with SoA storage
//
// Cells are stored in parallel arrays (CellGeo[], CellVisual[])
// indexed by a hash map from CellKey → uint32_t index.
// This gives O(1) lookup and cache-friendly iteration.
// ============================================================

class CellGridCPU {
public:
    float base_cell_size;

    CellGridCPU(float base_size = BASE_CELL_SIZE)
        : base_cell_size(base_size) {}

    // --- Core operations ---

    // Insert a cell. Returns the index of the inserted cell.
    uint32_t insert(const CellKey& key, const CellGeo& geo, const CellVisual& vis) {
        auto it = key_to_index_.find(key);
        if (it != key_to_index_.end()) {
            // Overwrite existing
            uint32_t idx = it->second;
            geo_[idx] = geo;
            vis_[idx] = vis;
            return idx;
        }
        uint32_t idx = static_cast<uint32_t>(geo_.size());
        geo_.push_back(geo);
        vis_.push_back(vis);
        keys_.push_back(key);
        key_to_index_[key] = idx;
        return idx;
    }

    // Lookup by key. Returns index or UINT32_MAX if not found.
    uint32_t find(const CellKey& key) const {
        auto it = key_to_index_.find(key);
        if (it == key_to_index_.end()) return UINT32_MAX;
        return it->second;
    }

    bool contains(const CellKey& key) const {
        return key_to_index_.count(key) > 0;
    }

    // Remove a cell. Swaps with last element to keep arrays dense.
    void remove(const CellKey& key) {
        auto it = key_to_index_.find(key);
        if (it == key_to_index_.end()) return;

        uint32_t idx = it->second;
        uint32_t last = static_cast<uint32_t>(geo_.size()) - 1;

        if (idx != last) {
            // Swap with last element
            geo_[idx] = geo_[last];
            vis_[idx] = vis_[last];
            keys_[idx] = keys_[last];
            key_to_index_[keys_[last]] = idx;
        }

        geo_.pop_back();
        vis_.pop_back();
        keys_.pop_back();
        key_to_index_.erase(it);
    }

    // --- Direct array access (for GPU upload, iteration, etc.) ---

    size_t size() const { return geo_.size(); }
    bool empty() const { return geo_.empty(); }

    CellGeo* geo_data() { return geo_.data(); }
    const CellGeo* geo_data() const { return geo_.data(); }

    CellVisual* vis_data() { return vis_.data(); }
    const CellVisual* vis_data() const { return vis_.data(); }

    const CellKey* key_data() const { return keys_.data(); }

    CellGeo& geo(uint32_t idx) { return geo_[idx]; }
    const CellGeo& geo(uint32_t idx) const { return geo_[idx]; }

    CellVisual& vis(uint32_t idx) { return vis_[idx]; }
    const CellVisual& vis(uint32_t idx) const { return vis_[idx]; }

    const CellKey& key(uint32_t idx) const { return keys_[idx]; }

    // --- Spatial queries ---

    // Get cell at a world position at the given level
    uint32_t find_at(const vec3& pos, int level = 0) const {
        CellKey k = position_to_key(pos, level);
        return find(k);
    }

    // Get the neighbor in a given direction (+X=0, -X=1, +Y=2, -Y=3, +Z=4, -Z=5)
    uint32_t find_neighbor(const CellKey& key, int direction) const {
        static const int dx[] = {1, -1, 0, 0, 0, 0};
        static const int dy[] = {0, 0, 1, -1, 0, 0};
        static const int dz[] = {0, 0, 0, 0, 1, -1};
        CellKey nk = {key.level, key.x + dx[direction], key.y + dy[direction], key.z + dz[direction]};
        return find(nk);
    }

    // --- Cell size at a given key ---
    float cell_size(const CellKey& key) const {
        return cell_size_at_level(key.level);
    }

    vec3 cell_center_pos(const CellKey& key) const {
        return trivima::cell_center(key);
    }

    // --- Bounding box ---
    void compute_bounds(vec3& out_min, vec3& out_max) const {
        if (keys_.empty()) {
            out_min = {0, 0, 0};
            out_max = {0, 0, 0};
            return;
        }
        out_min = {1e30f, 1e30f, 1e30f};
        out_max = {-1e30f, -1e30f, -1e30f};
        for (size_t i = 0; i < keys_.size(); ++i) {
            vec3 c = trivima::cell_center(keys_[i]);
            float hs = cell_size_at_level(keys_[i].level) * 0.5f;
            if (c.x - hs < out_min.x) out_min.x = c.x - hs;
            if (c.y - hs < out_min.y) out_min.y = c.y - hs;
            if (c.z - hs < out_min.z) out_min.z = c.z - hs;
            if (c.x + hs > out_max.x) out_max.x = c.x + hs;
            if (c.y + hs > out_max.y) out_max.y = c.y + hs;
            if (c.z + hs > out_max.z) out_max.z = c.z + hs;
        }
    }

    // --- Bulk operations ---

    void clear() {
        geo_.clear();
        vis_.clear();
        keys_.clear();
        key_to_index_.clear();
    }

    void reserve(size_t n) {
        geo_.reserve(n);
        vis_.reserve(n);
        keys_.reserve(n);
        key_to_index_.reserve(n);
    }

private:
    std::vector<CellGeo> geo_;
    std::vector<CellVisual> vis_;
    std::vector<CellKey> keys_;
    std::unordered_map<CellKey, uint32_t, CellKeyHash> key_to_index_;
};

} // namespace trivima
