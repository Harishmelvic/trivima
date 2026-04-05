# Trivima Testing — Complete 3D World Pipeline
## End-to-End Tests: Photograph to Navigable World

---

# Chapter 1: What These Tests Cover

This document tests the complete pipeline described in `complete_3d_world_theory.md` — all 8 stages from photograph input to navigable 3D world. Previous testing documents validated individual stages in isolation (Stage 1-2: 71/71 passing, VLM Stage 4: 29 tests designed). This document validates them working together as a single system.

## Dependencies

All Stage 1-2 tests must pass (71/71). The rendering pipeline (ModernGL + cell ID buffer + AI texturing) must be integrated. Shell extension, navigation, and progressive entry must be implemented.

## Test Data

The 20-photo test suite:
- 10 ScanNet rooms (ground truth 3D available for SSIM comparison)
- 5 Matterport3D rooms (larger rooms, multiple wall segments)
- 5 personal photos (real-world conditions — varying lighting, phone cameras, clutter)

Each photo is processed through the full pipeline and evaluated at multiple stages.

---

# Chapter 2: Stage 1-2 Integration (Perception → Cell Grid)

Already validated by 71/71 existing tests. These integration tests check the handoff works end-to-end.

## Test 2.1 — Photo to Navigable Cells in Under 2 Seconds

```
Input: single room photograph (1920×1080 or higher)

Time the full pipeline:
  t1 = Depth Pro inference
  t2 = SAM 3 inference  
  t3 = bilateral smoothing + scale calibration
  t4 = point-to-cell conversion + gradients + integrals + neighbors

total = t1 + t2 + t3 + t4

Verify: total < 2.0 seconds on A40
Verify: cell grid has 30,000-150,000 cells (reasonable for a room)
Verify: cell grid passes conservation check (mass integral stable)
```

Pass: perception + cell grid construction completes in under 2 seconds. This is the "time to first frame" — the user can start navigating after this.

## Test 2.2 — Perception Output Consistency

```
Run the full perception pipeline on the same image 3 times.

Verify: cell count varies by < 1% between runs (deterministic)
Verify: total density integral varies by < 0.1% between runs
Verify: same objects detected with same labels in all runs
```

Pass: pipeline is deterministic. Same input always produces the same cell grid.

## Test 2.3 — Memory Cleanup Between Models

```
Run pipeline on 5 images sequentially.
Record GPU memory after each:
  - After Depth Pro (before unload)
  - After empty_cache()
  - After SAM 3 (before unload)
  - After empty_cache()
  - After cell grid construction

Verify: GPU memory after empty_cache() returns to within 500MB of baseline
Verify: no OOM on any image
Verify: no memory growth across the 5 images (no leak)
```

Pass: sequential model loading works cleanly with no memory leaks.

---

# Chapter 3: Stage 3 (Shell Extension)

## Test 3.1 — Shell Extends to Complete Room

```
Input: cell grid from a room photo showing 2 walls + floor

Run shell extension.

Verify: floor extends to cover estimated room footprint
Verify: at least 2 additional wall segments generated
Verify: ceiling generated at estimated height (240-270cm)
Verify: extended cells have density 1.0 and correct normals
  (wall normals are horizontal, floor normal is up, ceiling normal is down)
Verify: extended cells have confidence 0.5-0.7 (lower than photo-derived cells)
```

Pass: shell extension produces a closed architectural envelope.

## Test 3.2 — Shell Extension Geometry Quality

```
For extended wall cells:
  Compute planarity error: distance of each cell center from the fitted plane

Verify: mean planarity error < 1mm (walls should be perfectly planar)
Verify: wall cells form a continuous surface with no gaps (no missing cells)
Verify: wall-floor junction is seamless (no gap between floor plane and wall plane)
```

Pass: extended geometry is geometrically perfect.

## Test 3.3 — Shell Extension Doesn't Break Existing Cells

```
Record cell grid state before shell extension (positions, densities, albedos).
Run shell extension.
Compare photo-derived cells before and after.

Verify: zero photo-derived cells were modified by shell extension
Verify: total density integral increased (new cells added) but original mass unchanged
```

Pass: shell extension only adds — never modifies existing content.

## Test 3.4 — Shell Extension Timing

```
Time shell extension on 5 test images.

Verify: shell extension completes in < 15 seconds
Record: time per image, number of extension cells generated
```

Pass: completes within the progressive entry budget (user navigates visible portion while extension runs in background).

---

# Chapter 4: Stage 4 (AI Texturing)

## Test 4.1 — Buffer Renderer Produces Valid Input

```
Render the cell grid to 5 aligned buffers:
  flat albedo (3ch), depth (1ch), normals (3ch), semantic labels (1ch), cell ID (1ch integer)

Verify: all buffers are the same resolution
Verify: albedo buffer has no NaN or Inf values
Verify: depth buffer values are positive and plausible (0.1m - 20m)
Verify: normal buffer vectors are unit length (magnitude within 1% of 1.0)
Verify: cell ID buffer has valid indices (all map to existing cells)
Verify: no cell ID appears in a region where density is zero (no phantom cells)
```

Pass: buffer renderer produces clean input for the AI texturing model.

## Test 4.2 — Cell ID Buffer Round-Trip

```
Render cell ID buffer from a viewpoint.
For 100 random pixels, read the cell ID.
Look up the cell's 3D position from the grid.
Project the 3D position back to screen coordinates.

Verify: projected position is within 2 pixels of the original pixel
  (accounting for cell size quantization)
```

Pass: cell ID buffer correctly maps pixels to cells. This is essential for AI texturing write-back.

## Test 4.3 — AI Texturing Write-Back

```
Run AI texturing model on the buffer inputs.
Write the output back to cells via the cell ID buffer.

Verify: cells that were visible have updated light values
Verify: cells that were NOT visible have unchanged light values
Verify: no cell receives light from a pixel that doesn't belong to it
  (cell ID mapping is correct)
Verify: light values are in plausible range (0-1 per channel)
```

Pass: AI model output correctly writes back to the 3D cell grid.

## Test 4.4 — AI Texturing Visual Quality (Requires Trained GAN)

```
For 5 ScanNet test images (with ground truth):
  Render cell grid from original camera angle
  Apply AI texturing
  Compare AI-textured output to original photograph

Compute: SSIM between AI-textured render and original photo
Compute: LPIPS perceptual distance

Verify: mean SSIM > 0.75 (target: 0.85)
Verify: mean LPIPS < 0.20 (target: 0.15)
```

Pass: AI texturing produces output that recognizably matches the original photo from the same viewpoint. **This test can only run after the GAN is trained.**

## Test 4.5 — Temporal Consistency During Navigation

```
Render a slow camera movement (10 frames, 5° rotation between each).
Apply AI texturing to each frame.
Write back to cells.

For each cell visible in all 10 frames:
  Compute: light value variance across the 10 frames

Verify: mean light variance < 0.03 (cells shouldn't flicker)
Verify: cells with light_temporal_deriv ≈ 0 are NOT re-lit (dirty mask works)
Verify: only 10-30% of cells are re-lit per frame during slow navigation
```

Pass: per-cell temporal blending prevents flickering. Stable cells are skipped.

---

# Chapter 5: Stage 5 (Auto-Furnishing Integration)

## Test 5.1 — VLM Gap Detection on Cell Grid

```
Build cell grid from a sparsely furnished room (sofa only, no other furniture).

Run VLM gap detection.

Verify: VLM identifies at least 2 missing items appropriate for the room type
Verify: VLM does NOT suggest items that already exist as cells in the grid
Verify: suggested positions are within the room's cell grid bounds
```

Pass: VLM correctly reads the cell grid content and identifies gaps.

## Test 5.2 — Object Voxelization

```
Take a 3D furniture model (OBJ/GLB from database).
Voxelize it into cells at 5cm resolution.

Verify: cell count is reasonable (50-500 for typical furniture)
Verify: cells form a connected volume (no floating fragments)
Verify: cells have correct density (1.0 for solid, 0 for air)
Verify: cell albedo matches the model's texture colors
Verify: cell normals match the model's surface normals
```

Pass: 3D models convert cleanly to cell clusters.

## Test 5.3 — Placed Object Updates Grid

```
Place a voxelized chair at a validated position.

Verify: chair cells added to the grid
Verify: collision field updated (can't walk through the chair)
Verify: BFS clearance query near the chair returns shorter distance
Verify: floor cells beneath the chair are still present (not deleted)
Verify: total cell count increased by exactly the chair's cell count
```

Pass: placed objects integrate correctly into the cell grid.

## Test 5.4 — Object Removal Reveals Floor

```
Place a table. Verify table cells exist.
Remove the table.

Verify: table cells deleted from grid
Verify: floor cells beneath the table are now exposed (they existed all along)
Verify: collision field updated (can walk through where the table was)
Verify: total density integral decreased by exactly the table's mass
```

Pass: object removal is clean and reveals underlying surfaces.

---

# Chapter 6: Stage 6 (Conservation Validation)

## Test 6.1 — Energy Conservation During AI Texturing

```
Before AI texturing: compute total light energy = sum(albedo × light × cell_area)
After AI texturing: compute total light energy again

Verify: energy change < 5% per cell
Verify: energy change < 1% scene-wide
```

Pass: AI texturing doesn't create or destroy energy.

## Test 6.2 — Mass Conservation During LOD Changes

```
Record total density integral at base resolution.
Subdivide 1000 cells (Taylor expansion).
Record total density integral at subdivided resolution.
Merge back to base resolution (integral-weighted averaging).
Record total density integral after merge.

Verify: density integral change during subdivision < 0.5%
Verify: density integral change during merge < 0.1%
Verify: final density integral within 0.5% of original
```

Pass: LOD changes preserve mass.

## Test 6.3 — Shadow Direction Consistency

```
After AI texturing, for each cell with a cast shadow:
  Compute: light gradient direction in that cell
  Compute: vector from cell to nearest light source (estimated from bright cells)
  Compute: dot product between the two vectors

Verify: dot product > 0.5 for at least 70% of shadow-receiving cells
  (shadows point away from light sources, not toward them)
```

Pass: AI-generated shadows point in physically plausible directions.

## Test 6.4 — Deliberately Introduced Errors

```
Introduce 10 deliberate violations:
  - 3 cells with reflected light > incoming light (energy violation)
  - 3 cells with reversed shadow direction
  - 2 cells with density changed without integral update (mass violation)
  - 2 cells with light gradient opposite to shadow direction

Run conservation validation.

Verify: at least 5/10 violations detected (50% catch rate)
Record: which violations were caught, which were missed
```

Pass: conservation checks catch at least half of deliberately introduced errors. This validates that the checks are functional, not just that the system never produces errors.

---

# Chapter 7: Stage 7 (Navigation)

## Test 7.1 — Wall Collision

```
Position camera 0.5m from a wall. Move toward the wall.

Verify: camera stops before penetrating the wall
Verify: camera slides along the wall when moving at an angle
Verify: no frame where the camera position is inside a wall cell
Verify: collision check takes < 0.01ms per frame
```

Pass: cannot walk through walls.

## Test 7.2 — Furniture Collision

```
Position camera near a sofa. Walk toward the sofa.

Verify: camera stops before penetrating the sofa
Verify: collision works on all sides of the sofa (front, back, sides)
Verify: sub-cell gradient precision prevents the camera from getting
  closer than the actual surface (not just the cell center)
```

Pass: cannot walk through furniture.

## Test 7.3 — Floor Following

```
Navigate across the room floor.

Verify: camera height stays at floor_height + 1.6m (±5cm)
Verify: camera height is smooth (no sudden jumps between frames)
Verify: camera cannot fall through the floor
```

Pass: floor following is smooth and reliable.

## Test 7.4 — Floor Following on Steps/Slopes

```
If test scene has steps or level changes:
  Walk across the level change.

Verify: camera height adjusts smoothly (over 3-4 frames)
Verify: no sudden teleportation to the new height
```

Pass: height transitions are smooth. If no level changes in test data, skip.

## Test 7.5 — World Boundary

```
Walk toward the edge of the generated world (past the shell extension).

Verify: camera stops at the boundary (cannot walk into the void)
Verify: visual fog increases near the boundary (soft indication of edge)
```

Pass: cannot walk off the edge of the world.

## Test 7.6 — Navigation Performance

```
Run a 60-second automated camera path through the room.
Record frame times.

Verify: mean frame time < 33ms (>30 FPS)
Verify: no frame exceeds 100ms (no major stutters)
Verify: 95th percentile frame time < 50ms
```

Pass: navigation is smooth at 30+ FPS.

---

# Chapter 8: Stage 8 (Rendering)

## Test 8.1 — Visual Match from Original Angle

```
Render the cell grid from the original camera position.
Compare to the original photograph.

Compute: SSIM
Compute: LPIPS

Verify: SSIM > 0.65 (without AI texturing)
Verify: SSIM > 0.75 (with AI texturing, target 0.85)
```

Pass: the rendered scene recognizably looks like the original photo from the same viewpoint.

## Test 8.2 — Subdivision Reveals Detail

```
Render a surface cell at base resolution (5cm).
Subdivide it (Taylor expansion to 2.5cm children).
Render the children.

Verify: children's albedo values are within 10% of what sub-sampling
  the original photo would show at those positions
Verify: no visible artifacts at subdivision boundaries
Verify: total color energy (albedo integral) is preserved within 5%
```

Pass: Taylor expansion produces plausible detail, not noise or artifacts.

## Test 8.3 — LOD Transitions

```
Render a scene while moving the camera forward (approaching objects).
Objects transition from far LOD (merged) to near LOD (base) to close LOD (subdivided).

Verify: no visible popping during LOD transitions
Verify: total cell count stays within budget (< 200K visible)
Verify: adaptive quality maintains target frame rate
```

Pass: LOD transitions are smooth. Distant objects are simplified without visible quality jumps.

## Test 8.4 — Cell ID Buffer Accuracy

```
Render the scene. For each pixel in the cell ID buffer:
  If cell_id > 0: verify that the corresponding cell exists and is visible
  If cell_id = 0: verify that the pixel is background (sky/void)

Verify: 100% of non-zero cell IDs map to valid, visible cells
Verify: no cell is rendered but missing from the cell ID buffer
```

Pass: cell ID buffer is a perfect 1:1 map between pixels and cells.

## Test 8.5 — Frustum Culling Effectiveness

```
Render from several viewpoints. Record:
  total_cells: all cells in the grid
  visible_cells: cells within the frustum and rendered
  culled_cells: cells outside the frustum

Verify: culled_cells / total_cells > 0.5 (at least 50% culled for typical views)
Verify: no visible cell was incorrectly culled (no holes in the render)
```

Pass: frustum culling eliminates significant work without visual errors.

---

# Chapter 9: End-to-End Integration

## Test 9.1 — Photo to Navigable World (Must-Pass)

The single most important test.

```
Input: single room photograph

Step 1: Perception → cell grid (< 2 seconds)
Step 2: Begin navigation (flat-shaded, visible portion only)
Step 3: Shell extension (background, < 15 seconds)
Step 4: AI texturing (background, < 30 seconds)
Step 5: Navigate freely through complete room

Verify: pipeline completes without crash
Verify: navigation begins within 2 seconds of upload
Verify: no wall clipping during navigation
Verify: no floor falling during navigation
Verify: visible portion recognizably matches the original photo
Verify: extended portions form a complete room (no void when turning around)
Verify: frame rate > 20 FPS throughout

Record: time_to_first_frame, time_to_complete, cell_count, fps_mean, fps_min
```

Pass: a user can upload a photo and walk through the room within seconds. This is the product.

## Test 9.2 — 20-Photo Acceptance Test

```
Run Test 9.1 on all 20 test photos.

Record pass/fail for each photo.
Record failure reasons for failed photos.

Verify: at least 15/20 photos pass (75% acceptance rate)
```

Pass: the system works on the majority of real-world photos.

## Test 9.3 — Failure Case Analysis

```
For each photo that fails Test 9.2:
  Categorize the failure:
    A. Perception failure (depth map is wrong — glass, mirrors, dark)
    B. Shell extension failure (room geometry misestimated)
    C. Collision failure (walk through walls or furniture)
    D. Rendering failure (visual artifacts, wrong colors)
    E. Performance failure (< 20 FPS)
    F. AI texturing failure (flickering, wrong lighting)

Document: which failure modes are most common
Document: which are fixable vs fundamental limitations
```

Pass: failures are understood and categorized. No "mystery" failures.

## Test 9.4 — Progressive Entry Timing

```
Instrument each pipeline stage with timestamps.
Upload a photo and record when each stage completes:

  t_perception:     perception finishes
  t_grid:           cell grid built, navigation begins
  t_shell:          shell extension complete
  t_texturing:      AI texturing applied
  t_furnishing:     auto-furnishing complete (if activated)

Verify: t_grid < 2.0 seconds
Verify: t_shell < 15 seconds
Verify: t_texturing < 30 seconds
Verify: t_furnishing < 60 seconds
Verify: navigation is uninterrupted during background processing
  (no frame drops when shell extension or texturing completes)
```

Pass: progressive entry works — user navigates immediately, content appears seamlessly.

## Test 9.5 — Memory Budget

```
Run the complete pipeline on the largest test image.
Record peak GPU memory at each stage.

Verify: peak memory < 40 GB (fits on A40 48GB with headroom)
Verify: after pipeline completes, steady-state memory < 20 GB
  (navigation uses only the cell grid + renderer + AI texturing model)
```

Pass: complete pipeline fits on a single A40 48GB GPU.

## Test 9.6 — Object Placement in Navigable World

```
Build a complete navigable world from a room photo.
Place a furniture object via the validation + VLM pipeline.
Navigate to the placed object.

Verify: object is visible and correctly positioned
Verify: object has collision (can't walk through it)
Verify: object sits on the floor (not floating)
Verify: AI texturing renders the object with consistent lighting
Verify: the object does not clip through existing furniture
```

Pass: placed objects are physically present in the navigable world.

---

# Chapter 10: Test Execution

## 10.1 Test Order and Dependencies

```
PHASE 1 — Perception + Grid (must pass, 5 min):
  [CRITICAL] 2.1  Photo to cells in < 2 seconds
             2.2  Perception consistency
             2.3  Memory cleanup

PHASE 2 — Shell Extension (must pass, 10 min):
  [CRITICAL] 3.1  Shell extends to complete room
             3.2  Geometry quality
             3.3  Doesn't break existing cells
             3.4  Timing

PHASE 3 — AI Texturing (10 min, partially requires trained GAN):
  [CRITICAL] 4.1  Buffer renderer produces valid input
  [CRITICAL] 4.2  Cell ID buffer round-trip
  [CRITICAL] 4.3  AI texturing write-back
             4.4  Visual quality (needs GAN)
             4.5  Temporal consistency (needs GAN)

PHASE 4 — Auto-Furnishing Integration (10 min):
             5.1  VLM gap detection on cell grid
  [CRITICAL] 5.2  Object voxelization
  [CRITICAL] 5.3  Placed object updates grid
             5.4  Object removal reveals floor

PHASE 5 — Conservation (5 min):
             6.1  Energy conservation during texturing
             6.2  Mass conservation during LOD
             6.3  Shadow direction consistency
             6.4  Deliberately introduced errors

PHASE 6 — Navigation (10 min):
  [CRITICAL] 7.1  Wall collision
  [CRITICAL] 7.2  Furniture collision
  [CRITICAL] 7.3  Floor following
             7.4  Steps/slopes
             7.5  World boundary
  [CRITICAL] 7.6  Navigation performance (> 30 FPS)

PHASE 7 — Rendering (10 min):
  [CRITICAL] 8.1  Visual match from original angle
             8.2  Subdivision reveals detail
             8.3  LOD transitions
             8.4  Cell ID buffer accuracy
             8.5  Frustum culling effectiveness

PHASE 8 — End-to-End (30 min):
  [CRITICAL] 9.1  Photo to navigable world
  [CRITICAL] 9.2  20-photo acceptance (≥15/20 pass)
             9.3  Failure case analysis
             9.4  Progressive entry timing
             9.5  Memory budget
             9.6  Object placement in navigable world
```

## 10.2 Summary

| Phase | Tests | Critical | Time |
|---|---|---|---|
| Perception + Grid | 3 | 1 | 5 min |
| Shell Extension | 4 | 1 | 10 min |
| AI Texturing | 5 | 3 | 10 min |
| Auto-Furnishing | 4 | 2 | 10 min |
| Conservation | 4 | 0 | 5 min |
| Navigation | 6 | 4 | 10 min |
| Rendering | 5 | 1 | 10 min |
| End-to-End | 6 | 2 | 30 min |
| **Total** | **37** | **14** | **~90 min** |

## 10.3 What Failure Means

If perception+grid fails (Phase 1): core pipeline is broken. Check Depth Pro/SAM 3 model loading, bilateral smoothing parameters, point-to-cell conversion thresholds.

If shell extension fails (Phase 2): RANSAC plane fitting is misconfigured. Check the minimum point count for plane detection, the normal clustering threshold, and the room dimension estimation.

If AI texturing fails (Phase 3): the buffer renderer or cell ID write-back has bugs. Tests 4.1-4.3 isolate the issue. Tests 4.4-4.5 require the trained GAN — if those fail, it's a training data or model architecture issue.

If navigation fails (Phase 6): collision thresholds are wrong (too permissive = walk through walls, too aggressive = stuck everywhere) or floor detection is failing (wrong floor plane, or floor cells have incorrect normals). Check cell types and density values for walls and floor.

If end-to-end fails (Phase 8): individual phases work but the pipeline integration has issues. Check data flow between stages — format mismatches, timing of background tasks, memory management during progressive entry.

## 10.4 Relationship to Success Criteria

| Success Criterion | Test(s) That Verify It |
|---|---|
| Navigate within 5 seconds | 2.1 (< 2s), 9.1 (< 2s), 9.4 (progressive timing) |
| Visible portion looks like the room | 8.1 (SSIM > 0.75), 9.1 (visual match) |
| No wall clipping | 7.1, 7.2, 9.1 |
| No floor falling | 7.3, 9.1 |
| Frame rate > 20 FPS | 7.6 (> 30 FPS target), 9.1 (> 20 FPS minimum) |
| Subdivision without artifacts | 8.2 (Taylor expansion quality) |
| ≥15/20 photos pass | 9.2 (20-photo acceptance) |
| Frame rate > 30 FPS | 7.6 |
| Convincing shadows/lighting | 4.4 (SSIM), 6.3 (shadow direction) |
| Math validation catches ≥50% errors | 6.4 (deliberate violations) |
| Taylor within 10% color error | 8.2 (subdivision accuracy) |

## 10.5 Pre-Requisites

Before running these tests:

1. All 71 Stage 1-2 tests passing
2. ModernGL renderer integrated with cell grid
3. Cell ID buffer implemented
4. Shell extension implemented
5. Navigation (collision + floor following) implemented
6. AI texturing GAN trained (for Phase 3 tests 4.4-4.5 and Phase 7 test 8.1 with texturing)
7. 20-photo test suite prepared (10 ScanNet + 5 Matterport3D + 5 personal)

Tests 2.1-3.4, 4.1-4.3, 5.2-5.4, 6.1-6.2, 7.1-7.6, and 8.2-8.5 can run WITHOUT the trained GAN. Tests 4.4-4.5 and 8.1 (with texturing) require the GAN. Plan accordingly.
