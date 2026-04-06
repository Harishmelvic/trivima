"""
Volume filling — make objects solid by filling cells behind front surfaces.

Key insight: fill along the DOMINANT surface normal per connected region,
not per-cell normals. A wall fills straight backward. A sofa fills straight
backward. Per-cell normals are noisy and create layered gaps.

Algorithm:
  1. Cluster surface cells into connected regions (flood fill in 3D)
  2. RANSAC fit a dominant normal per region
  3. Fill all cells in that region along the single dominant direction
  4. Result: solid blocks, not layered pancakes
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from collections import deque


# Default object depths in meters
OBJECT_DEPTHS = {
    "sofa": 0.85, "couch": 0.85, "chair": 0.50, "armchair": 0.70,
    "table": 0.60, "desk": 0.60, "coffee table": 0.45, "bed": 2.00,
    "nightstand": 0.40, "dresser": 0.50, "cabinet": 0.40, "bookshelf": 0.30,
    "shelf": 0.25, "tv stand": 0.40, "wardrobe": 0.60, "ottoman": 0.50,
    "door": 0.05, "window": 0.15, "curtain": 0.05,
    "wall": 0.15, "floor": 0.10, "ceiling": 0.10,
    "rug": 0.02, "painting": 0.03, "mirror": 0.03,
    "refrigerator": 0.70, "oven": 0.60, "television": 0.10, "lamp": 0.25,
    "default": 0.15,
}


def estimate_object_depth(label: str) -> float:
    label_lower = label.lower().strip()
    if label_lower in OBJECT_DEPTHS:
        return OBJECT_DEPTHS[label_lower]
    for key, depth in OBJECT_DEPTHS.items():
        if key in label_lower or label_lower in key:
            return depth
    return OBJECT_DEPTHS["default"]


def _cluster_cells(cell_pos, cell_size):
    """Cluster surface cells into connected regions using flood fill.

    Two cells are connected if they are within 2 cell sizes of each other
    in any axis. Returns list of cluster indices per cell.
    """
    n = len(cell_pos)
    cell_keys = {}
    for i in range(n):
        key = tuple(np.floor(cell_pos[i] / cell_size).astype(int))
        cell_keys[key] = i

    cluster_id = np.full(n, -1, dtype=np.int32)
    current_cluster = 0

    # 26-connected neighbors (3x3x3 cube minus center)
    offsets = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                offsets.append((dx, dy, dz))

    for i in range(n):
        if cluster_id[i] >= 0:
            continue

        # BFS flood fill
        queue = deque([i])
        cluster_id[i] = current_cluster

        while queue:
            ci = queue.popleft()
            key = tuple(np.floor(cell_pos[ci] / cell_size).astype(int))

            for dx, dy, dz in offsets:
                nkey = (key[0] + dx, key[1] + dy, key[2] + dz)
                if nkey in cell_keys:
                    ni = cell_keys[nkey]
                    if cluster_id[ni] < 0:
                        cluster_id[ni] = current_cluster
                        queue.append(ni)

        current_cluster += 1

    return cluster_id, current_cluster


def _ransac_dominant_normal(normals, n_iterations=50):
    """Find the dominant normal direction from a set of normals.

    Uses voting: pick a random normal, count how many others agree (dot > 0.8).
    The normal with most votes wins.
    """
    n = len(normals)
    if n == 0:
        return np.array([0, 0, 1], dtype=np.float32)
    if n < 3:
        avg = normals.mean(axis=0)
        nm = np.linalg.norm(avg)
        return (avg / nm).astype(np.float32) if nm > 1e-6 else np.array([0, 0, 1], dtype=np.float32)

    best_normal = normals[0]
    best_votes = 0
    rng = np.random.RandomState(42)

    for _ in range(min(n_iterations, n)):
        candidate = normals[rng.randint(n)]
        nm = np.linalg.norm(candidate)
        if nm < 1e-6:
            continue
        candidate = candidate / nm

        # Count inliers: normals that agree (dot > 0.7)
        dots = np.abs(normals @ candidate)
        votes = (dots > 0.7).sum()

        if votes > best_votes:
            best_votes = votes
            # Average the inlier normals for a cleaner result
            inliers = dots > 0.7
            avg = normals[inliers].mean(axis=0)
            nm2 = np.linalg.norm(avg)
            best_normal = (avg / nm2).astype(np.float32) if nm2 > 1e-6 else candidate

    return best_normal


def fill_volume(
    cell_pos: np.ndarray,
    cell_col: np.ndarray,
    cell_nrm: np.ndarray,
    cell_labels: Optional[np.ndarray] = None,
    label_names: Optional[Dict[int, str]] = None,
    cell_size: float = 0.02,
    default_depth: float = 0.15,
    min_cluster_size: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fill cells behind front surfaces using per-segment dominant normals.

    If SAM labels are provided, uses them as clusters (each segment = one object).
    Otherwise falls back to spatial clustering.
    Each cluster gets ONE dominant fill direction via RANSAC.
    """
    import time
    t0 = time.time()
    n_original = len(cell_pos)

    # Step 1: Use SAM labels as clusters if available, else spatial clustering
    if cell_labels is not None and len(np.unique(cell_labels)) > 1:
        cluster_id = cell_labels.copy()
        n_clusters = int(cluster_id.max()) + 1
        cluster_sizes = np.bincount(cluster_id.astype(np.int64), minlength=n_clusters)
        n_valid_clusters = (cluster_sizes >= min_cluster_size).sum()
        print(f"  Volume fill: using SAM labels — {n_valid_clusters} segments (>={min_cluster_size} cells)")
    else:
        cluster_id, n_clusters = _cluster_cells(cell_pos, cell_size)
        cluster_sizes = np.bincount(cluster_id[cluster_id >= 0], minlength=n_clusters)
        n_valid_clusters = (cluster_sizes >= min_cluster_size).sum()
        print(f"  Volume fill: spatial clustering — {n_clusters} clusters, {n_valid_clusters} large enough")

    # Step 2: Find dominant normal per cluster
    cluster_normals = {}
    cluster_depths = {}
    for cid in range(n_clusters):
        if cluster_sizes[cid] < min_cluster_size:
            continue
        mask = cluster_id == cid
        dominant = _ransac_dominant_normal(cell_nrm[mask])
        cluster_normals[cid] = dominant

        # Determine fill depth from labels or default
        if cell_labels is not None and label_names is not None:
            # Majority vote for label in this cluster
            labels_in_cluster = cell_labels[mask]
            label_idx = int(np.bincount(labels_in_cluster.astype(np.int64)).argmax())
            label_name = label_names.get(label_idx, "default")
            cluster_depths[cid] = estimate_object_depth(label_name)
        else:
            cluster_depths[cid] = default_depth

    # Step 3: Fill along dominant normal per cluster
    existing = set()
    for i in range(n_original):
        key = tuple(np.floor(cell_pos[i] / cell_size).astype(int))
        existing.add(key)

    generated_pos = []
    generated_col = []

    for i in range(n_original):
        cid = cluster_id[i]
        if cid < 0 or cid not in cluster_normals:
            continue

        dominant_normal = cluster_normals[cid]
        fill_depth = cluster_depths[cid]
        fill_dir = -dominant_normal  # into the object (opposite of surface normal)

        n_fill = max(1, int(fill_depth / cell_size))

        for step in range(1, n_fill + 1):
            new_pos = cell_pos[i] + fill_dir * (step * cell_size)
            new_key = tuple(np.floor(new_pos / cell_size).astype(int))

            if new_key not in existing:
                existing.add(new_key)
                darken = max(0.85, 1.0 - step * 0.005)
                generated_pos.append(new_pos.astype(np.float32))
                generated_col.append((cell_col[i] * darken).astype(np.float32))

    if not generated_pos:
        print(f"  Volume fill: no cells generated")
        return cell_pos, cell_col, cell_nrm

    gen_pos = np.array(generated_pos, dtype=np.float32)
    gen_col = np.array(generated_col, dtype=np.float32)
    # Normals for filled cells: same as the dominant normal of their cluster
    # (pointing outward from the fill direction)
    gen_nrm = np.zeros_like(gen_pos)
    # For simplicity, use the fill direction reversed (surface faces outward)
    # We'll recompute from neighbors after
    gen_nrm[:] = [0, 0, 1]  # placeholder, will be overridden by neighbor recompute

    all_pos = np.concatenate([cell_pos, gen_pos])
    all_col = np.concatenate([cell_col, gen_col])
    all_nrm = np.concatenate([cell_nrm, gen_nrm])

    # Step 4: Recompute normals for filled cells from neighbor occupancy
    _recompute_normals(all_pos, all_nrm, cell_size, n_original)

    dt = time.time() - t0
    print(f"  Volume fill: {n_original:,} surface + {len(gen_pos):,} filled = {len(all_pos):,} total ({dt:.1f}s)")

    return all_pos, all_col, all_nrm


def _recompute_normals(positions, normals, cell_size, start_idx):
    """Recompute normals for filled cells based on neighbor occupancy."""
    cell_map = set()
    for i in range(len(positions)):
        key = tuple(np.floor(positions[i] / cell_size).astype(int))
        cell_map.add(key)

    directions = np.array([
        [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]
    ], dtype=np.float32)

    for i in range(start_idx, len(positions)):
        key = tuple(np.floor(positions[i] / cell_size).astype(int))
        ix, iy, iz = key

        empty_sum = np.zeros(3, dtype=np.float32)
        for di, (dx, dy, dz) in enumerate([(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]):
            nkey = (ix+dx, iy+dy, iz+dz)
            if nkey not in cell_map:
                empty_sum += directions[di]

        nm = np.linalg.norm(empty_sum)
        if nm > 1e-6:
            normals[i] = empty_sum / nm
