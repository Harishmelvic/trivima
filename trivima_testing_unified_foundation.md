# Trivima Testing — Unified Pipeline Foundation
## Tests for Surface Field, Functional Field, Soft Collision, Conservation Wiring, Confidence Fix

---

# Chapter 1: What These Tests Cover

Five items that bridge the validated cell grid (Stage 1-2) to the validation fields needed for placement and auto-furnishing (Stage 3+). Each item is a query layer built on top of the existing cell grid. The cell grid itself is already tested and passing (27/28).

Items under test:

1. Surface Support Field — "is there a surface here to hold an object?"
2. Functional Field — "does this position serve the object's purpose?"
3. Soft Collision BFS — "how far is this position from the nearest object?"
4. Conservation Validation Wiring — connect existing conservation.py to app.py
5. Confidence Formula Fix — multiplicative instead of geometric mean

## Test Data

All tests reuse the same 11-image test set from Stage 1-2 testing. Synthetic scenes are added for items 1-3 where analytical ground truth is needed.

---

# Chapter 2: Surface Support Field

## Test 2.1 — Floor Detection (Synthetic)

Create a synthetic cell grid: flat floor at Y=0 spanning X=[0,5m], Z=[0,5m]. All cells are Surface type with normal=(0,1,0) and semantic_label=FLOOR.

```
Run surface field detection.

Verify: exactly one floor surface detected
Verify: floor height = 0.0 ± 0.01m
Verify: floor extent covers X=[0,5], Z=[0,5]
Verify: floor labeled as "floor"

Query query_support(grid, 2.5, 0.0, 2.5):
  Verify: has_support = True
  Verify: surface_type = "floor"
  Verify: height = 0.0 ± 0.01

Query query_support(grid, 2.5, 1.0, 2.5):
  Verify: has_support = False (1m above floor, no surface there)

Query query_support(grid, 6.0, 0.0, 2.5):
  Verify: has_support = False (outside floor extent)
```

Pass: All queries return correct results on the trivial case.

## Test 2.2 — Elevated Surface Detection (Synthetic)

Create a synthetic cell grid: floor at Y=0 plus a table surface at Y=0.75m spanning X=[1,2], Z=[1,2]. Table cells have semantic_label=TABLE, normal=(0,1,0).

```
Run surface field detection.

Verify: two surfaces detected (floor + table)
Verify: floor height = 0.0 ± 0.01
Verify: table height = 0.75 ± 0.02
Verify: table extent covers approximately X=[1,2], Z=[1,2]

Query query_support(grid, 1.5, 0.0, 1.5):
  Verify: has_support = True, surface_type = "floor"

Query query_support(grid, 1.5, 0.75, 1.5):
  Verify: has_support = True, surface_type = "table"

Query query_support(grid, 1.5, 0.40, 1.5):
  Verify: has_support = False (between floor and table — no surface)

Query query_support(grid, 3.0, 0.75, 3.0):
  Verify: has_support = False (correct height but outside table extent)
```

Pass: Both surfaces detected. Queries at correct heights within correct extents return True. All other queries return False.

## Test 2.3 — Tolerance Parameter

Verify the 2cm height tolerance works correctly.

```
Using the synthetic floor at Y=0:

Query query_support(grid, 2.5, 0.01, 2.5, tolerance=0.02):
  Verify: has_support = True (1cm above floor, within 2cm tolerance)

Query query_support(grid, 2.5, 0.03, 2.5, tolerance=0.02):
  Verify: has_support = False (3cm above floor, outside 2cm tolerance)

Query query_support(grid, 2.5, -0.015, 2.5, tolerance=0.02):
  Verify: has_support = True (1.5cm below floor surface, within tolerance)
```

Pass: Tolerance correctly gates surface matching.

## Test 2.4 — Confidence-Weighted Plane Fitting

Create a synthetic floor where 80% of cells have confidence=0.9 and are at Y=0.0, and 20% of cells have confidence=0.2 and are at Y=0.15 (bad depth — 15cm off).

```
Run surface field detection with confidence weighting.

Verify: detected floor height is close to 0.0, NOT pulled toward 0.15
Specifically: floor_height < 0.02 (low-confidence outliers did not corrupt the fit)

Run the same detection WITHOUT confidence weighting (uniform weights):
Verify: detected floor height is pulled toward outliers (floor_height > 0.02)
```

Pass: Confidence weighting produces a floor height closer to the true value (0.0) than uniform weighting. This proves that low-confidence cells are correctly downweighted.

## Test 2.5 — Real Photo Floor Detection

Run surface field detection on 5 ScanNet test images.

```
For each:
  Verify: floor detected
  Verify: floor height is plausible (between -0.1m and 0.1m in room coordinates)
  Verify: floor extent covers at least 50% of the room's XZ footprint
  
  If the room contains a table or desk:
    Verify: at least one elevated surface detected
    Verify: elevated surface height is plausible (0.4m - 1.2m for typical furniture)
```

Pass: Floor detected in all 5 images with plausible heights. Elevated surfaces detected where furniture exists.

## Test 2.6 — Slope Rejection

Verify that steeply angled cells are not classified as support surfaces.

```
Create a synthetic ramp: cells from Y=0 to Y=1.5m across X=[0,3m], making a 30° slope.
Cell normals are tilted 30° from vertical.

Run surface field detection.

Verify: ramp is NOT detected as a support surface (normal_y = cos(30°) ≈ 0.87 — 
        borderline with the 0.85 threshold)

Create a steeper ramp at 40°: normal_y = cos(40°) ≈ 0.77
Verify: NOT detected as support surface (below 0.85 threshold)

Create a gentle slope at 10°: normal_y = cos(10°) ≈ 0.98
Verify: IS detected as support surface
```

Pass: Only surfaces with normal_y > 0.85 are classified as support. The 30° case is borderline — document whether the system accepts or rejects it.

---

# Chapter 3: Functional Field

## Test 3.1 — Plant Near Window (Synthetic)

Create a synthetic room: floor cells, wall cells, and a window cell cluster at position (4.0, 1.5, 2.5) labeled "window."

```
Query query_functional(grid, 3.5, 0.0, 2.5, category="plant"):
  Record score_near (0.5m from window)

Query query_functional(grid, 1.0, 0.0, 2.5, category="plant"):
  Record score_far (3.0m from window)

Verify: score_near > score_far
Verify: score_near > 0.7 (close to window is good for plants)
Verify: score_far < 0.5 (far from window is less ideal)
```

Pass: Plant functional score correctly correlates with window proximity.

## Test 3.2 — Lamp Near Seating (Synthetic)

Create a synthetic room: floor, walls, and a sofa cell cluster at position (2.0, 0.0, 3.0) labeled "sofa."

```
Query query_functional(grid, 2.5, 0.0, 3.0, category="lamp"):
  Record score_near (0.5m from sofa)

Query query_functional(grid, 0.5, 0.0, 0.5, category="lamp"):
  Record score_far (corner, ~3m from sofa)

Verify: score_near > score_far
Verify: score_near > 0.7 (close to seating is good for lamps)
```

Pass: Lamp functional score correctly correlates with seating proximity.

## Test 3.3 — Storage Near Wall (Synthetic)

Create a synthetic room. Query functional score for storage furniture.

```
Query query_functional(grid, 0.1, 0.0, 2.5, category="bookshelf"):
  Record score_wall (10cm from wall — wall-adjacent)

Query query_functional(grid, 2.5, 0.0, 2.5, category="bookshelf"):
  Record score_center (center of room)

Verify: score_wall > score_center
Verify: score_wall > 0.7 (bookshelves belong against walls)
Verify: score_center < 0.5 (bookshelves in room center is wrong)
```

Pass: Storage functional score correctly correlates with wall adjacency. This tests the neighbor summary check — cells near walls have wall-type neighbors.

## Test 3.4 — Unknown Category Returns Neutral Score

```
Query query_functional(grid, 2.5, 0.0, 2.5, category="unknown_object"):

Verify: returns a neutral score (0.5) rather than crashing
Verify: no exception thrown
```

Pass: Graceful handling of unknown categories.

## Test 3.5 — Real Photo Functional Queries

Run functional queries on a ScanNet image that contains a window and seating.

```
Detect window position from cell grid (cells with label "window")
Detect seating position from cell grid (cells with label "sofa" or "chair")

Query plant score at 10 positions across the room
Verify: positions nearest the window have highest plant scores

Query lamp score at 10 positions across the room
Verify: positions nearest the seating have highest lamp scores
```

Pass: Functional scores on real data show the expected spatial pattern — scores are highest near the relevant anchor objects.

## Test 3.6 — Functional Field Performance

```
Run query_functional 10,000 times at random positions for category="plant"
Record total time.

Verify: total_time < 1 second (< 0.1ms per query)
```

Pass: Functional field queries are fast enough for real-time heatmap generation (need ~5000 queries for a heatmap at 10cm spacing).

---

# Chapter 4: Soft Collision BFS Distance

## Test 4.1 — Distance to Wall (Synthetic)

Create a synthetic room: a single wall at X=0 (solid cells at X=0), empty space from X=0.05 to X=5.0.

```
Query query_clearance(grid, 0.10, 0.5, 2.5):
  Verify: distance ≈ 0.05m (one cell away from wall)

Query query_clearance(grid, 0.50, 0.5, 2.5):
  Verify: distance ≈ 0.45m (9 cells from wall)

Query query_clearance(grid, 2.50, 0.5, 2.5):
  Verify: distance ≈ 2.45m (49 cells from wall)
```

Pass: Reported distances match expected values within ±1 cell (±5cm).

## Test 4.2 — Distance to Furniture (Synthetic)

Create a synthetic room with a table (solid cell cluster) at X=[2.0, 2.5], Z=[2.0, 2.5].

```
Query query_clearance(grid, 1.5, 0.0, 2.25):
  Verify: distance ≈ 0.45m (9 cells to nearest table edge)

Query query_clearance(grid, 2.25, 0.0, 2.25):
  Verify: distance ≈ 0.0m (inside the table — zero clearance)

Query query_clearance(grid, 3.0, 0.0, 2.25):
  Verify: distance ≈ 0.45m (9 cells to nearest table edge, from the other side)
```

Pass: Clearance correctly computed from all directions.

## Test 4.3 — Clearance Inside Empty Room

Create a synthetic room with only walls (no furniture). Query clearance at the center.

```
Room is 5m × 5m. Center is at (2.5, 0.5, 2.5).
Nearest wall is 2.5m away.

Query query_clearance(grid, 2.5, 0.5, 2.5):
  Verify: distance ≈ 2.5m
```

Pass: BFS correctly traverses to the nearest wall from the room center.

## Test 4.4 — BFS Performance

```
Run query_clearance 1,000 times at random positions in a real cell grid (~80K cells).
Record total time.

Verify: total_time < 5 seconds (< 5ms per query)
```

Pass: BFS distance queries are fast enough for batch heatmap computation. Note: if BFS is too slow, a precomputed distance field (computed once per scene) can replace per-query BFS.

## Test 4.5 — Clearance Scoring for Spacing

Test that clearance values produce reasonable spacing scores.

```
Define spacing_score(distance) = clamp(distance / target_clearance, 0, 1)
where target_clearance = 0.5m (50cm comfortable spacing)

Verify: spacing_score(0.0) = 0.0 (touching object — bad)
Verify: spacing_score(0.25) = 0.5 (25cm — marginal)
Verify: spacing_score(0.50) = 1.0 (50cm — good)
Verify: spacing_score(1.00) = 1.0 (clamped — more than enough)
```

Pass: Spacing score correctly maps clearance to a [0,1] quality score.

---

# Chapter 5: Conservation Validation Wiring

## Test 5.1 — Reference Mass Set on Startup

```
Run app.py with a test image.

Verify: conservation validator initializes
Verify: reference_mass is set (total density integral > 0)
Verify: reference_mass matches manually computed sum of all cell.density_integral
```

Pass: Reference mass correctly computed at startup.

## Test 5.2 — Conservation Check Runs

```
Run app.py with a test image for 100 frames.

Verify: conservation check runs at least once during 100 frames
Verify: conservation results are logged or reported
Verify: no crash from conservation check
```

Pass: Conservation validation executes without errors.

## Test 5.3 — Mass Conservation Passes on Static Scene

```
Run app.py with a test image. Navigate for 300 frames (10 seconds).
No objects added or removed — just camera movement with LOD changes.

Record mass at frame 1 and frame 300.

Verify: |mass_300 - mass_1| / mass_1 < 0.001 (0.1% tolerance)
```

Pass: Total mass is conserved during navigation despite subdivision and merge operations.

## Test 5.4 — Conservation Detects Injected Error

```
Run app.py, wait for grid to be built.
Programmatically set 50 random cells' density to 2× their current value (injecting mass).

Run conservation check.

Verify: mass violation detected
Verify: violation is reported in output/logs
```

Pass: Conservation check catches artificially injected mass increase.

## Test 5.5 — Stats Output

```
Run: python trivima/app.py --image room.jpg --stats

Verify: CLI output includes conservation stats:
  - total_mass (float)
  - mass_drift (percentage)
  - energy_check (pass/fail)
  - frames_since_last_check (integer)
```

Pass: Stats flag produces readable conservation output.

---

# Chapter 6: Confidence Formula Fix

## Test 6.1 — Formula Change Verification

```
Create two cells with known inputs:
  Cell A: density_conf = 0.8, propagated_conf = 0.6 (includes semantic penalty)
  Cell B: density_conf = 0.4, propagated_conf = 0.3

OLD formula (geometric mean):
  Cell A: sqrt(0.8 × 0.6) = sqrt(0.48) = 0.693
  Cell B: sqrt(0.4 × 0.3) = sqrt(0.12) = 0.346

NEW formula (multiplicative):
  Cell A: 0.8 × 0.6 = 0.48
  Cell B: 0.4 × 0.3 = 0.12

Build cell grid using the new formula.

Verify: Cell A confidence = 0.48 (not 0.693)
Verify: Cell B confidence = 0.12 (not 0.346)
```

Pass: Confidence values match the multiplicative formula, not the geometric mean.

## Test 6.2 — Multiplicative Is More Conservative

The multiplicative formula produces lower confidence values than the geometric mean for the same inputs (since sqrt(a×b) > a×b when both a,b < 1). Verify this makes the system more conservative.

```
Build a cell grid from a test image using both formulas (old and new).

Compare:
  mean_confidence_old (geometric mean)
  mean_confidence_new (multiplicative)

Verify: mean_confidence_new < mean_confidence_old
Verify: the ratio is between 0.5 and 0.9 (the new values are lower but not drastically different)
```

Pass: New formula is more conservative (lower values) but not so aggressive that everything becomes low-confidence.

## Test 6.3 — Subdivision Gating Still Works

```
Build a cell grid with the new confidence formula.

Count cells with confidence > 0.5 → subdivisible_cells
Count cells with confidence < 0.5 → non_subdivisible_cells

Verify: both counts are > 0 (some cells pass, some don't)
Verify: glass/mirror cells are in non_subdivisible (confidence well below 0.5)
Verify: well-textured floor cells are in subdivisible (confidence above 0.5)
```

Pass: The 0.5 subdivision threshold still produces a reasonable split between subdivisible and non-subdivisible cells with the new formula.

## Test 6.4 — Collision Margins Update

```
Build a cell grid with the new confidence formula.

Find a glass cell (confidence ≤ 0.2 after formula change).
Read its collision_margin.

Verify: collision_margin is expanded (larger than default)
Verify: the expansion is appropriate for the lower confidence value

Find a high-confidence floor cell (confidence > 0.5).
Read its collision_margin.

Verify: collision_margin is at default (not expanded)
```

Pass: Collision margin expansion correctly adapts to the new confidence values.

## Test 6.5 — Existing Tests Still Pass

```
Run the full Stage 1-2 test suite (27 existing tests).

Some tests check specific confidence values that will change with the new formula.
Update the expected values in those tests to match the multiplicative formula.

Verify: all 27 tests pass with updated thresholds.
Specifically check:
  Test 4.9 (confidence assignment) — thresholds may need adjustment
  Test 3.6-3.8 (failure modes) — confidence values will be lower
```

Pass: All existing tests pass with adjusted thresholds. No regressions in non-confidence-related tests.

---

# Chapter 7: Integration Test

## Test 7.1 — All Five Validation Fields Query Together

The integration test verifies that all five fields can be queried for the same position and produce a meaningful composite score.

```
Build a cell grid from a real ScanNet image.
Select a position on the visible floor, 1m from a window, 0.5m from a sofa, 
with no furniture within 30cm.

Query all five fields for category="plant":
  surface = query_support(grid, x, y, z)           → should be True (on floor)
  collision = query_clearance(grid, x, y, z)        → should be ~0.3m
  boundary = cell exists at (x, y, z)               → should be True (inside room)
  functional = query_functional(grid, x, y, z, "plant") → should be high (near window)
  aesthetic = geometric_aesthetic_score(grid, x, y, z)  → (if implemented)

Compute composite: if surface and boundary, score = collision_score × functional

Verify: composite score > 0.5 (this is a reasonable position for a plant)

Repeat for a position inside the sofa:
  surface = True (sofa surface)
  collision = 0.0 (inside an object)
  
  Verify: composite score = 0 (collision kills the score)

Repeat for a position 3m from any window:
  functional = lower score for plants
  
  Verify: composite score is lower than the first position
```

Pass: The composite score correctly differentiates good positions (near window, clear of furniture) from bad positions (inside furniture, far from functional anchors).

## Test 7.2 — Heatmap Data Generation

Test that the validation fields can produce a heatmap-ready data grid.

```
Build a cell grid from a test image.
Define a 2D grid of query positions on the floor at 10cm spacing.

For each query position:
  Compute composite validation score for category="plant"
  Store as heatmap[x][z] = score

Verify: heatmap has reasonable spatial pattern:
  - Zero scores where furniture exists (collision)
  - High scores near windows (functional)
  - Moderate scores in open floor areas (clear but far from windows)
  - Zero scores outside room boundary

Verify: total computation time < 3 seconds for a typical room
  (a 5m×5m room at 10cm spacing = 2,500 queries)
```

Pass: Heatmap shows expected spatial pattern. Computation is fast enough for interactive use.

---

# Chapter 8: Test Execution

## 8.1 Test Order

```
PHASE 1 — Confidence Formula Fix (must pass first, 5 minutes):
  [CRITICAL] 6.1  Formula change verification
             6.2  Multiplicative is more conservative
  [CRITICAL] 6.3  Subdivision gating still works
             6.4  Collision margins update
  [CRITICAL] 6.5  Existing tests still pass (regression check)

PHASE 2 — Surface Support Field (5 minutes):
  [CRITICAL] 2.1  Floor detection synthetic
  [CRITICAL] 2.2  Elevated surface detection
             2.3  Tolerance parameter
             2.4  Confidence-weighted plane fitting
             2.5  Real photo floor detection
             2.6  Slope rejection

PHASE 3 — Functional Field (5 minutes):
  [CRITICAL] 3.1  Plant near window
  [CRITICAL] 3.2  Lamp near seating
             3.3  Storage near wall
             3.4  Unknown category handling
             3.5  Real photo functional queries
             3.6  Performance

PHASE 4 — Soft Collision BFS (3 minutes):
  [CRITICAL] 4.1  Distance to wall
             4.2  Distance to furniture
             4.3  Clearance in empty room
             4.4  BFS performance
             4.5  Clearance scoring

PHASE 5 — Conservation Wiring (5 minutes):
  [CRITICAL] 5.1  Reference mass set
             5.2  Conservation check runs
  [CRITICAL] 5.3  Mass conservation on static scene
             5.4  Detects injected error
             5.5  Stats output

PHASE 6 — Integration (5 minutes):
  [CRITICAL] 7.1  All five fields query together
             7.2  Heatmap data generation
```

## 8.2 Summary

| Phase | Tests | Critical | Time |
|---|---|---|---|
| Confidence fix | 5 | 3 | 5 min |
| Surface field | 6 | 2 | 5 min |
| Functional field | 6 | 2 | 5 min |
| Soft collision | 5 | 1 | 3 min |
| Conservation | 5 | 2 | 5 min |
| Integration | 2 | 1 | 5 min |
| **Total** | **29** | **11** | **~28 min** |

## 8.3 What Failure Means

If confidence formula tests fail (Phase 1): the new formula isn't applied correctly, or existing behavior broke. Fix before proceeding — everything downstream depends on correct confidence values.

If surface field fails (Phase 2): floor/surface detection is wrong. Check the normal threshold (0.85), check that RANSAC plane fitting is using confidence weights, check that semantic labels are correctly propagated to cells.

If functional field fails (Phase 3): semantic label search isn't finding the right cells, or distance computation is wrong. Check that SAM labels ("window", "sofa", "chair") match what the functional field expects. Label vocabulary mismatch is the most likely cause.

If BFS fails (Phase 4): the grid traversal isn't working. Check cell neighbor connectivity — BFS depends on being able to walk from cell to cell through the sparse grid. If cells have gaps (missing neighbors), BFS will report incorrect distances.

If conservation wiring fails (Phase 5): the validator isn't being called from app.py, or the reference mass computation doesn't match the cell grid's actual density integrals. Check that the validator reads from the same grid instance that the renderer uses.

If integration fails (Phase 6): individual fields work but don't combine correctly. Check the composite score formula — multiplicative composition means one zero kills everything, which is the correct behavior for collision but might be too aggressive if any field has a bug that produces zeros incorrectly.
