# Trivima Testing — Stage 1 & 2
## Perception + Cell Grid Foundation Tests

---

# Chapter 1: What These Tests Cover

These tests validate the foundation of the pipeline: can we take a photograph and produce a correct, high-quality cell grid? Nothing about rendering, navigation, or AI texturing. If the cell grid is wrong, everything downstream is wrong.

The tests cover four areas: the cell data structure itself, the perception pipeline (Depth Pro + bilateral smoothing + SAM + scale calibration + failure modes), the cell grid construction (point-to-cell conversion + gradients + neighbors + integrals + confidence), and the shell extension (plane detection + room completion).

## Test Image Set

5 ScanNet images with ground truth depth and segmentation (quantitative comparison).
3 personal smartphone photos (real-world conditions).
1 image with a glass coffee table (failure mode test).
1 image with a large mirror (failure mode test).
1 dark image, mean brightness < 30/255 (failure mode test).

Total: 11 images. All tests use this same set unless otherwise noted.

---

# Chapter 2: Cell Data Structure

## Test 2.1 — Struct Size and Alignment

Verify CellGeo is exactly 64 bytes, CellVisual is exactly 448 bytes, both 64-byte aligned.

```
static_assert(sizeof(CellGeo) == 64)
static_assert(sizeof(CellVisual) == 448)
static_assert(alignof(CellGeo) == 64)
static_assert(alignof(CellVisual) == 64)

Allocate 1000 CellGeo, verify consecutive addresses are exactly 64 bytes apart.
Allocate 1000 CellVisual, verify consecutive addresses are exactly 448 bytes apart.
```

Pass: All assertions pass. Zero tolerance.

## Test 2.2 — Grid Insert and Lookup

Insert 10,000 cells with known values. Retrieve each by CellKey. Verify all fields match exactly. Query 1,000 non-existent keys, verify all return "not found." Verify geo array index equals visual array index for every cell (SoA consistency).

Pass: 100% retrieval accuracy, zero false positives.

## Test 2.3 — GPU Grid Consistency

Build a cell grid on CPU. Copy to GPU (cuco::static_map). Query 10,000 random positions on both. Verify every query returns the same cell index and identical field values.

Pass: 100% agreement between CPU and GPU.

## Test 2.4 — Serialization Round-Trip

Build a cell grid from a test image (~80K cells). Serialize to binary. Deserialize. Verify cell count matches, every CellGeo field matches (exact), every CellVisual field matches (exact). Verify file size equals 32 (header) + cell_count × 528 bytes.

Pass: Bit-exact round-trip.

## Test 2.5 — Confidence Field Accessibility

Verify the confidence field in CellGeo is readable and writable without breaking alignment or padding.

```
Create a cell. Set confidence = 0.73.
Serialize, deserialize. Read confidence.
Verify: value == 0.73 (exact float match).
Verify: no other CellGeo fields were corrupted by writing confidence.
```

Pass: Confidence survives round-trip, no field corruption.

---

# Chapter 3: Perception Pipeline

## Test 3.1 — Depth Pro Output Validation

Run Depth Pro on 5 ScanNet images with ground truth depth.

```
For each image:
  Verify: output shape == input shape (H × W)
  Verify: all depth values > 0
  Verify: depth range between 0.1m and 20m
  Compute AbsRel error against ground truth
```

Pass: Mean AbsRel < 0.10 (10%) across 5 images.

Report per-image AbsRel for investigation — any image above 15% needs manual inspection.

## Test 3.2 — Bilateral Smoothing Effectiveness

Take Depth Pro output for a room with textured floor and clear object boundaries.

```
PRE-SMOOTHING:
  100 points on flat floor: measure local depth variance in 7×7 window → noise_before
  50 object boundary pixels: measure depth gradient magnitude → edge_before

Apply bilateral filter (spatial_sigma=3.5, color_sigma=25.0)

POST-SMOOTHING:
  Same 100 floor points: measure variance → noise_after
  Same 50 boundary pixels: measure gradient magnitude → edge_after

noise_reduction = 1 - (noise_after / noise_before)
edge_degradation = 1 - (edge_after / edge_before)
```

Pass: noise_reduction between 40% and 65%. edge_degradation below 20%.

If noise_reduction < 40%: sigma too small, not smoothing enough.
If edge_degradation > 20%: sigma too large, destroying boundaries.

## Test 3.3 — Bilateral Sigma Preserves Surface Detail

Specific test for the concern that sigma 3-4px erases brick mortar and wood grain.

```
Take an image with a visible brick wall or wood floor.
Extract a 100×100 pixel patch of the textured surface.

Compute depth map of the patch.
Compute local depth standard deviation in 3×3 windows across the patch → texture_signal_before

Apply bilateral filter.
Compute local depth standard deviation again → texture_signal_after

texture_preservation = texture_signal_after / texture_signal_before
```

Pass: texture_preservation > 0.60 (at least 60% of surface texture detail survives smoothing).

If below 0.60: reduce spatial_sigma to 2.5 and retest.

## Test 3.4 — Scale Calibration Accuracy

Select 5 ScanNet images where a full door is visible.

```
For each:
  Run Depth Pro → raw depth
  Compute uncalibrated_scale_error = |mean(predicted / gt) - 1.0|
  
  Run scale calibration → detect door → compute correction
  Apply correction
  Compute calibrated_scale_error
```

Pass: Mean calibrated_scale_error < 0.005 (0.5%). Calibration improves over uncalibrated for all 5 images.

## Test 3.5 — SAM Segmentation Quality

Run SAM on all 8 normal test images (excluding the 3 failure mode images).

```
For each:
  Verify: at least one "floor" region detected
  Verify: at least one "wall" region detected  
  Verify: segmentation covers > 95% of pixels
  Verify: unique label count between 5 and 100
  
For 5 ScanNet images with ground truth:
  Compute IoU for 3 largest objects against GT masks
```

Pass: Floor and wall detected in all 8. Mean IoU > 0.70 on ScanNet subset.

## Test 3.6 — Failure Mode Detection: Mirror

Run pipeline on the mirror test image.

```
Verify: mirror region detected by SAM
Verify: mirror cells assigned confidence ≤ 0.1
Verify: mirror cells forced to density = 1.0
Verify: mirror cells marked as Solid type
Verify: no subdivision allowed on mirror cells
```

Pass: All 5 checks pass. The mirror is treated as a solid wall despite depth showing a phantom room behind it.

## Test 3.7 — Failure Mode Detection: Glass

Run pipeline on the glass table test image.

```
Verify: glass surface detected by SAM
Verify: glass cells assigned confidence ≤ 0.2
Verify: glass cells forced to density = 1.0
Verify: glass cells marked as Solid type
```

Pass: All 4 checks pass. The glass table exists in the cell grid as a solid barrier despite being invisible to depth estimation.

## Test 3.8 — Failure Mode Detection: Dark Scene

Run pipeline on the dark test image (mean brightness < 30/255).

```
Verify: dark scene flag raised
Verify: all cells have confidence multiplied by ≤ 0.4
Verify: mean confidence across all cells < 0.4
Verify: pipeline still completes (does not crash)
Verify: cell grid has at least 1000 cells (something was reconstructed, even if low quality)
```

Pass: All 5 checks pass. The system degrades gracefully rather than crashing or producing an empty grid.

## Test 3.9 — Perception Pipeline Timing

Run the full perception pipeline on 5 test images, recording time per stage.

```
For each image:
  Record: depth_pro_time (model inference only)
  Record: bilateral_smooth_time
  Record: sam_time
  Record: scale_calibration_time
  Record: failure_mode_detection_time
  Record: total_perception_time
```

Pass: Mean total_perception_time < 3 seconds (on A100/H100). Report per-stage breakdown to identify bottlenecks.

## Test 3.10 — Perception Memory Profile

Run each perception model separately, recording peak GPU memory.

```
torch.cuda.reset_peak_memory_stats()
Run Depth Pro
depth_pro_peak = torch.cuda.max_memory_allocated()
torch.cuda.empty_cache()

torch.cuda.reset_peak_memory_stats()
Run SAM
sam_peak = torch.cuda.max_memory_allocated()
torch.cuda.empty_cache()

Report: depth_pro_peak, sam_peak
Verify: each peak < 16 GB (fits on RTX 4090)
Verify: sequential execution with empty_cache prevents OOM
```

Pass: No OOM. Each model fits in 16 GB individually. Sequential execution with cache clearing works.

---

# Chapter 4: Cell Grid Construction

## Test 4.1 — Point-to-Cell Conversion (Synthetic)

Create a synthetic point cloud: 100,000 points on a flat floor at Y=0, uniformly distributed across X=[0,5m], Z=[0,5m], white color.

```
Run point_to_cell at 5cm resolution.

Verify: cell count ≈ 10,000 (100×100 grid)
Verify: all cells have type = Surface
Verify: all normals ≈ (0, 1, 0) within 5 degrees
Verify: all density ≈ 1.0
Verify: all albedo ≈ (1, 1, 1) within 0.05
Verify: all density_integral ≈ density × cell_volume
```

Pass: Cell count within 5% of 10,000. All property checks pass.

## Test 4.2 — Point-to-Cell Conversion (Real Photo)

Run point-to-cell on a real test image point cloud.

```
Verify: cell count between 30,000 and 200,000 (plausible room range)
Verify: no cells with NaN or Inf values in any field
Verify: all density values in [0, 1]
Verify: all normal vectors approximately unit length (magnitude 0.9-1.1)
Verify: all albedo values in [0, 1] per channel
Verify: memory usage ≈ cell_count × 512 bytes (within 20%)
```

Pass: All sanity checks pass on all 8 normal test images.

## Test 4.3 — Gradient Quality (Synthetic, Most Important Test)

This is the most critical test in the entire suite. If gradients are wrong, Taylor expansion is wrong, and the cell architecture's core mechanism fails.

```
Test Case A — Uniform wall:
  Create a flat wall, all cells identical white albedo.
  Compute albedo gradient.
  Verify: gradient magnitude < 0.05 everywhere (should be ~0).

Test Case B — Linear color ramp:
  Create a wall with albedo linearly varying from black to white across X.
  Compute albedo gradient.
  Verify: gradient direction is (1,0,0) ± 5 degrees.
  Verify: gradient magnitude matches analytical value ± 10%.

Test Case C — Surface boundary:
  Create solid wall ending at X=2.5m, empty space beyond.
  Compute density gradient at boundary cells.
  Verify: gradient points from solid toward empty (+X direction).
  Verify: gradient magnitude > 0.5 / cell_width.

Test Case D — Curved surface:
  Create a sphere of radius 1m, represented as cells.
  Compute normal gradient (curvature).
  Verify: curvature approximately uniform (constant for sphere).
  Verify: curvature magnitude ≈ 1/radius ± 20%.
```

Pass: All four cases pass their specific criteria.

## Test 4.4 — Gradient Quality (Real Photo)

Run gradient computation on a cell grid from a real ScanNet image with ground truth.

```
Select 50 cells on the floor surface (known to be flat).
Verify: mean density gradient magnitude < 0.15 (flat surface → low gradient)
Verify: mean normal gradient magnitude < 0.10 (flat surface → low curvature)

Select 50 cells at object boundaries (sofa edge, table edge).
Verify: mean density gradient magnitude > 0.3 (boundary → high gradient)
Verify: gradient direction generally points from object interior toward empty space
```

Pass: Flat surfaces have low gradients, boundaries have high gradients with correct direction.

## Test 4.5 — 5×5 Sobel Kernel vs Simple Finite Difference

Compare gradient quality between simple 2-point finite difference and 5×5 Sobel kernel.

```
Build cell grid from a real test image.

Compute gradients using simple finite difference: g_fd[i] = (value[i+1] - value[i-1]) / (2 × cell_size)
Compute gradients using 5×5 Sobel kernel.

For 100 cells on flat floor surfaces:
  Record gradient magnitude for both methods.
  
sobel_noise = mean(sobel_gradient_magnitude on flat surface)
fd_noise = mean(fd_gradient_magnitude on flat surface)
```

Pass: sobel_noise < fd_noise (Sobel produces cleaner gradients on flat surfaces). Expected improvement: 30-50% noise reduction.

## Test 4.6 — Neighbor Summary Correctness

Build a cell grid from a test image. For 1000 random cells, check all 6 neighbor summaries.

```
For each cell, for each of 6 directions:
  Look up the actual neighbor at (position + direction × cell_size)
  Compare summary {type, density, normal_y, light_luma} against actual values
  If no neighbor exists: verify summary indicates empty
```

Pass: 100% match across all 6000 checks.

## Test 4.7 — Integral Conservation (Subdivision + Merge)

Build a cell grid. Record total density integral (sum of density × volume for all cells).

```
Step 1: Record total_mass_original

Step 2: Subdivide all cells one level (each → 8 children via Taylor expansion)
        Record total_mass_subdivided

Step 3: Merge all children back to parents (8 → 1 via integral-weighted average)
        Record total_mass_merged

subdivision_error = |total_mass_subdivided - total_mass_original| / total_mass_original
merge_error = |total_mass_merged - total_mass_original| / total_mass_original
```

Pass: subdivision_error < 1%. merge_error < 0.1%.

## Test 4.8 — Taylor Expansion Child Prediction Accuracy

The critical test for whether subdivision produces useful children, not noise.

```
Build a cell grid from a ScanNet image (has GT depth for comparison).
Select 100 cells on well-textured surfaces (confidence > 0.7).

For each cell:
  Subdivide into 8 children using Taylor expansion:
    child_value = parent_value + gradient · offset + 0.5 × second_deriv × offset²
  
  Also compute "ground truth" children by re-sampling the original point cloud 
  at child resolution and computing actual values.
  
  Compute prediction_error = |taylor_predicted - actual| for each child property.

Report: mean albedo prediction error, mean density prediction error, mean normal prediction error.
```

Pass: Mean albedo prediction error < 0.10 (10% color error). Mean density prediction error < 0.15. Mean normal error < 15 degrees. These thresholds come from the error propagation analysis — ±1.25cm child prediction error translates to approximately these property errors.

## Test 4.9 — Confidence Assignment

Build a cell grid from a test image containing textured floor, featureless wall, and (if present) glass surface.

```
Sample 20 cells from textured floor → mean confidence
Sample 20 cells from featureless white wall → mean confidence  
Sample glass cells (if present) → mean confidence

Verify: textured_confidence > wall_confidence > glass_confidence
Verify: textured_confidence > 0.70
Verify: glass_confidence ≤ 0.20
Verify: all confidence values in [0.0, 1.0]
Verify: no NaN confidence values
```

Pass: Confidence ordering correct, all values in valid range.

## Test 4.10 — Cell Grid Memory

Verify cell grid memory usage matches expectations.

```
Build cell grids from 5 test images.

For each:
  Record cell_count
  Record actual_memory (total allocated for geo + visual arrays)
  Compute expected_memory = cell_count × 512
  Compute memory_ratio = actual_memory / expected_memory
  
  Verify: 0.9 < memory_ratio < 1.2 (within 20% of theoretical)
  Verify: actual_memory < 100 MB for a typical room
```

Pass: Memory within 20% of theoretical. No room exceeds 100 MB at base resolution.

---

# Chapter 5: Shell Extension

## Test 5.1 — Plane Detection

Build a cell grid from a room photograph. Run RANSAC plane detection.

```
Verify: floor plane detected, normal within 10° of (0, 1, 0)
Verify: at least one wall plane detected, normal within 10° of horizontal
Verify: floor height consistent with lowest surface cells (within 5cm)
Verify: ceiling height plausible (200-350cm above floor)
Verify: wall planes are approximately vertical (within 10° of vertical)
Verify: detected planes don't intersect furniture (they should fit the room structure, not objects)
```

Pass: Floor and at least one wall detected in all 8 normal test images. Heights plausible.

## Test 5.2 — Extension Cell Quality

Run shell extension on a cell grid.

```
Record cell_count_before and cell_count_after.

Verify: cell_count_after > cell_count_before (new cells were generated)
Verify: extension increased count by at least 20%

For 100 random extension cells:
  Verify: type is Surface or Solid (not Empty)
  Verify: density = 1.0 (solid surface)
  Verify: normal matches the parent plane direction (within 5°)
  Verify: albedo is plausible (not NaN, not negative, in [0,1] range)
  Verify: confidence is reasonable (extension cells should have moderate confidence, ~0.5-0.7)

Verify: no extension cells overlap with existing cells (no duplication)
```

Pass: Extension generates new cells with correct properties and no overlaps.

## Test 5.3 — Extended Room Closure

Verify the extended cell grid forms a plausible closed room.

```
After shell extension:
  Count floor cells → floor_area = count × cell_size²
  Count wall cells in each detected wall → wall_area per wall
  
  Verify: floor_area is plausible for a room (4-100 m²)
  Verify: at least 3 wall planes have cells (a room needs at least 3 walls to be partially closed)
  
  Cast 100 random rays downward from ceiling height:
    Verify: at least 90% hit a floor cell (the floor is mostly complete)
  
  Cast 100 random rays horizontally from room center:
    Verify: at least 70% hit a wall cell within 10m (walls exist in most directions)
```

Pass: Floor covers most of the room area. Walls exist in most directions. The room feels enclosed rather than having large void gaps.

---

# Chapter 6: Test Execution

## 6.1 Test Order

Run in this order. Stop if any test marked CRITICAL fails.

```
PHASE 1 — Cell Struct (must pass first, 2 minutes):
  [CRITICAL] 2.1  Struct size and alignment
  [CRITICAL] 2.2  Grid insert/lookup
  [CRITICAL] 2.3  GPU grid consistency
             2.4  Serialization round-trip
             2.5  Confidence field accessibility

PHASE 2 — Perception (must pass before cell grid, 10 minutes):
  [CRITICAL] 3.1  Depth Pro output
  [CRITICAL] 3.2  Bilateral smoothing effectiveness
             3.3  Bilateral sigma preserves surface detail
             3.4  Scale calibration accuracy
  [CRITICAL] 3.5  SAM segmentation quality
             3.6  Failure mode: mirror
             3.7  Failure mode: glass
             3.8  Failure mode: dark scene
             3.9  Perception timing
             3.10 Perception memory profile

PHASE 3 — Cell Grid (must pass before shell extension, 10 minutes):
  [CRITICAL] 4.1  Point-to-cell synthetic
  [CRITICAL] 4.2  Point-to-cell real photo
  [CRITICAL] 4.3  Gradient quality synthetic (MOST IMPORTANT)
             4.4  Gradient quality real photo
             4.5  Sobel vs finite difference comparison
  [CRITICAL] 4.6  Neighbor summary correctness
  [CRITICAL] 4.7  Integral conservation
             4.8  Taylor expansion child accuracy
             4.9  Confidence assignment
             4.10 Cell grid memory

PHASE 4 — Shell Extension (3 minutes):
             5.1  Plane detection
             5.2  Extension cell quality
             5.3  Extended room closure
```

## 6.2 Summary

| Phase | Tests | Critical | Time |
|---|---|---|---|
| Cell struct | 5 | 3 | 2 min |
| Perception | 10 | 3 | 10 min |
| Cell grid | 10 | 5 | 10 min |
| Shell extension | 3 | 0 | 3 min |
| **Total** | **28** | **11** | **~25 min** |

The 11 critical tests run in approximately 10 minutes. If all 11 pass, the foundation is solid and Stage 3+ (rendering, navigation, AI texturing) can begin.

## 6.3 What Failure Means

If Test 2.1-2.3 fail: the data structure is broken. Fix the C++ code before anything else.

If Test 3.1 fails: Depth Pro is not producing valid output. Check model loading, input preprocessing, GPU compatibility.

If Test 3.2 fails: bilateral smoothing is not working correctly. Check sigma values, verify the RGB guide image is being used.

If Test 4.3 fails (gradient quality): this is the most serious failure. It means the cell architecture's core mechanism (Taylor expansion) will not work. Investigate: are the finite differences computed correctly? Is the bilateral smoothing insufficient? Is the cell resolution too coarse for the scene? This test failing blocks all downstream development until resolved.

If Test 4.7 fails (integral conservation): mass is being created or destroyed during subdivision/merge. The integral-weighted averaging has a bug. Fix before proceeding — conservation is the basis for the validation layer.

If shell extension tests fail: this is not critical. The system works without shell extension (user just can't turn around). Fix later, proceed with rendering.
