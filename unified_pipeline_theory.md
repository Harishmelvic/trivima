# The Unified Pipeline Theory
## Connecting Perception to Validation to Interaction

---

# Gap 1: From Perception Output to Validation Fields on Cells

---

## 1.1 The Central Connection

The entire spatial placement system rests on one data transformation: converting the perception output (Depth Pro depth map + SAM 3 semantic labels) into the five validation fields that score every candidate placement position. In the current architecture, this transformation goes through the cell grid — the perception output becomes cells, and the validation fields are computed from cell properties.

The transformation chain: photograph → Depth Pro → depth map → bilateral smoothing → backprojection → 3D point cloud → cell grid → validation fields. Each step refines the data. Each step can introduce error. The validation fields inherit whatever accuracy the cell grid has, which inherits whatever accuracy the perception pipeline provides.

## 1.2 Cell Grid → Surface Support Field

The surface support field answers: "at position (x, y, z), is there a surface to hold the placed object?"

Construction from cells: scan the cell grid for surface cells with upward-pointing normals (normal.y > 0.85). The floor is the largest contiguous cluster of such cells at the lowest height. Elevated surfaces (tables, shelves, counters) are smaller clusters at higher heights. Each surface is characterized by its height (the Y coordinate of its cells), its horizontal extent (the X-Z region covered by its cells), its semantic label (from the cells' SAM labels), and its estimated load capacity (derived from the semantic category).

The support field at any query position is computed by checking: does any surface cell exist at this height (within 2cm tolerance) with horizontal extent covering this X-Z position?

Because the cell grid already stores surface normals and semantic labels, the surface field construction is a simple filter — no geometric computation is needed. The cells ARE the surfaces.

## 1.3 Cell Grid → Collision Field

The collision field answers: "at position (x, y, z), would the placed object overlap with anything?"

Construction from cells: the collision check is a direct cell lookup. Query the cell at position (x, y, z). If the cell type is Solid or Surface with density above the threshold, there is a collision. If the cell is Empty or does not exist, the position is free.

For an object with a footprint spanning multiple cells, each cell within the footprint is checked. An object spanning 6×6 cells requires 36 lookups — approximately 0.036 milliseconds.

Sub-cell precision comes from the density gradient. A surface cell's density gradient indicates where within the cell the surface boundary is located. The position within the cell where density crosses 0.5 is the precise surface boundary: surface_offset = (0.5 - density) / gradient_magnitude. This gives approximately 5-15mm precision within a 5cm cell.

Low-confidence cells (glass, mirror, transparent surfaces) have expanded collision margins — the system treats uncertain cells as larger than their actual size, preventing placement near surfaces whose geometry may be unreliable.

The soft collision field also computes the minimum distance from the query position to any non-empty cell. Positions farther from existing objects score higher, encouraging comfortable spacing. This distance is computed from the cell grid using a breadth-first search from the query cell — counting empty cells in each direction until a non-empty cell is reached, then multiplying by cell size.

## 1.4 Cell Grid → Room Boundary Field

The boundary field answers: "is position (x, y, z) inside the navigable world?"

Construction from cells: the room boundary is defined by the shell extension planes (floor, walls, ceiling detected via RANSAC). The boundary field is 1 inside the polyhedral volume formed by these planes and 0 outside. A margin based on the object's dimensions ensures the entire object fits within the boundary.

For terrain and outdoor environments, the boundary is the extent of the generated cell grid. Beyond the last cell, the boundary field is 0.

The cell grid directly encodes the boundary — cells exist inside the room and do not exist outside. The boundary check is therefore a cell existence check: if the cell at position (x, y, z) exists in the sparse grid, the position is inside the world. If it does not exist, the position is outside or in ungenerated space.

## 1.5 Cell Grid → Functional Field

The functional field answers: "does this position serve the object's purpose?"

Construction from cells: the functional field requires semantic knowledge about spatial relationships. It queries the cell grid for cells with specific semantic labels and computes distances.

For plants: search the cell grid for cells labeled "window." Compute distance from the placement position to the nearest window cell cluster's centroid. Closer positions score higher for visibility and spatial contribution.

For lamps: search for cells labeled "sofa," "chair," "bed" (seating areas). Compute distance to the nearest seating cluster. Positions within 1-2 meters of seating score highest.

For storage furniture: search for wall cells adjacent to the placement position. Storage furniture should be near walls. Check the neighbor summaries of cells at the placement position — if multiple neighbors are wall cells, the position is wall-adjacent.

Each functional sub-score is derived purely from cell properties — semantic labels, positions, and neighbor summaries. No language processing is involved.

## 1.6 Cell Grid → Aesthetic Field

The aesthetic field answers: "does this placement look right?"

The aesthetic field has two components in the current architecture.

The geometric aesthetic score computes three measurable properties from the cell grid. Spatial balance: compute the center of mass of all non-empty cells, weighted by density. Compare to the room's geometric center. Positions that improve balance score higher. Spacing rhythm: compute the distances between adjacent object clusters in the cell grid. The coefficient of variation of these distances indicates spacing consistency. Lower variation scores higher. Symmetry alignment: detect the room's dominant visual axes from the wall cell orientations and major furniture cluster positions. Placements aligned with or symmetrically straddling these axes score higher.

The VLM aesthetic judgment: the VLM (Qwen2.5-VL + 3D-RoPE) evaluates the top 20-50 candidate positions identified by the geometric score. It considers style coherence, cultural appropriateness, contextual sensitivity, and design sophistication — qualities that pure geometry cannot assess. The VLM outputs a re-ranked list of top positions with brief justifications.

The combined score is: A = A_geometric^0.4 × A_vlm^0.6, weighting the VLM's design intelligence more heavily.

---

# Gap 1B: Per-Cell Confidence and Uncertainty Handling

---

## 1.7 The Five Sources of Uncertainty

Not all cells have equally reliable data. Five phenomena create uncertainty, each affecting the validation fields differently.

Transparent and reflective surfaces (glass, mirrors, polished metal) are the most problematic. The depth model cannot distinguish between the surface itself and what is visible through or reflected in it. A glass coffee table might be estimated at the depth of the floor beneath it, making the table invisible to the collision field. Mitigation: SAM 3 detects these materials, and the failure mode system forces density to 1.0 and sets confidence to 0.1-0.2.

Textureless regions (white walls, plain ceilings) lack the visual features needed for precise depth estimation. The model falls back on scene-level priors which are usually correct in direction but imprecise in exact position — typically 5-15 centimeters off. The gradients in these cells are unreliable. Mitigation: depth smoothness analysis assigns moderate confidence (0.5-0.7).

Object boundaries (where one object meets another) produce unreliable depth because the depth model's feature patches span the boundary, creating mixed-pixel depth values. The density gradient direction at boundaries is usually correct but the magnitude is unreliable. Mitigation: boundary cells receive slightly reduced confidence based on depth gradient discontinuity analysis.

Far distances (beyond 4-5 meters from the camera) have inherently lower precision because each pixel subtends a larger angle. Depth resolution degrades proportionally with distance. Mitigation: distant cells receive lower confidence based on the estimated depth value. Cells beyond 5 meters get a distance-based confidence penalty.

Repeated patterns (tiled floors, patterned wallpaper, brick walls) can confuse the depth model into assigning similar depths to tiles at different distances. This creates staircase-like depth artifacts where the true depth is smooth. Mitigation: the bilateral filter smooths these artifacts. Remaining staircasing reduces confidence.

## 1.8 Per-Cell Confidence Computation

Each cell carries a confidence value between 0 and 1 computed from four signals.

Depth smoothness: the local variance of depth values within and around this cell before bilateral filtering. Low variance (smooth depth on a uniform surface) indicates the depth model was confident. High variance (noisy depth, spikes, oscillations) indicates uncertainty. The confidence contribution is: conf_depth = 1 - clamp(local_variance / variance_threshold, 0, 1).

Point density: the number of source points that fell within this cell during point-to-cell conversion. More points indicate more pixels observed this region, giving higher statistical confidence. The confidence contribution is: conf_points = clamp(point_count / expected_points, 0, 1).

Semantic penalty: cells with semantic labels known to confuse depth estimation receive a hard confidence penalty regardless of other signals. Glass: confidence × 0.2. Mirror: confidence × 0.1. Transparent: confidence × 0.3. Specular metal: confidence × 0.4. Dark region: confidence × 0.4.

DUSt3R agreement (multi-image only): when both Depth Pro and DUSt3R provide depth estimates, cells where the two agree (within 5% of each other) receive high confidence. Cells where they disagree receive low confidence. Disagreement between independent models is a strong uncertainty signal.

The final confidence is: confidence = conf_depth × conf_points × semantic_penalty × (conf_dustsr if available).

## 1.9 How Confidence Drives Downstream Behavior

Each validation field uses confidence differently.

The collision field uses the most conservative strategy: low-confidence cells get expanded collision margins. If a cell's confidence is below 0.5, its effective collision radius is increased by 10-15 centimeters. This means the system assumes uncertain objects are larger than they appear. A missed collision (placing an object inside an existing one) is the worst failure mode, so the system errs on the side of caution. A false collision (rejecting a valid position) is much less harmful — the system simply chooses a different position.

The surface support field uses confidence-weighted plane fitting. When fitting a plane to floor cells, each cell's contribution is weighted by its confidence. High-confidence cells on well-textured floors dominate the fit. Low-confidence cells near glass or in dark regions barely contribute.

The boundary field ignores low-confidence wall cells entirely. If a wall cell's confidence is below 0.5, it is excluded from the RANSAC plane fitting. The wall position is extrapolated from the remaining high-confidence cells.

The functional field is relatively robust to uncertainty because it operates on cluster centroids (averages over many cells). Even if 20% of a window's cells have bad depth, the window's centroid position is only slightly affected.

The aesthetic field is nearly immune to per-cell uncertainty because it operates on the coarse spatial layout (object positions and sizes) which aggregates over hundreds of cells per object.

## 1.10 Subdivision and Confidence

Low-confidence cells are never subdivided via Taylor expansion, regardless of camera distance or input type. The gradients in low-confidence cells are unreliable — subdividing them produces children with incorrect values that look worse than the unsplit parent.

Instead, low-confidence cells at close range use neural texture features for detail (if available) or receive full weight from the AI texturing model (which paints photorealistic appearance without relying on gradient data).

High-confidence cells are subdivided according to the input type cap: 1 level for single image, 3 for multi-image, 4 for video.

---

# Gap 2: Training the Aesthetic System

---

## 2.1 Why Aesthetics Cannot Be Encoded as Rules

The geometric validation fields (collision, surface, boundary) are deterministic — given the cell grid, the score can be computed exactly. But the aesthetic field captures subjective quality: does this arrangement "feel right"? Does the room look balanced? Is there a pleasing rhythm to the spacing?

These properties resist formalization. Consider symmetry: a bedroom with nightstands symmetrically flanking the bed looks intentional. But a living room with perfect left-right symmetry looks sterile. The "right" amount of symmetry depends on the room type, the furniture style, the number of objects, and cultural conventions. No single rule captures this — it must be learned.

## 2.2 The Geometric Aesthetic Network

The geometric component of the aesthetic field is a convolutional network that evaluates spatial balance, spacing rhythm, and compositional quality from a top-down layout.

The input image is rendered at 256×256 pixel resolution from the cell grid, viewed from above. It has five channels: occupancy map (cells present = 1, empty = 0, walls = 0.5), category map (each semantic category assigned a unique value), height map (cell heights for visual prominence reasoning), proposed-object channel (highlighted position for the new placement), and room structure channel (walls, doors, windows marked).

The network is a ResNet-18 adapted for 5 input channels, with approximately 11 million parameters. It processes one layout in approximately 2 milliseconds and outputs a quality score between 0 and 1.

Training uses perturbation-based data from the 3D-FRONT dataset (18,968 professionally designed rooms). For each room, 20 variants are generated: the original designer arrangement (score 0.85-1.0) and 19 perturbations at varying displacement distances (score scaled by Gaussian decay: Score = 0.9 × exp(-(displacement/sigma)²)). This produces approximately 380,000 training examples.

A subset of 2,000 layouts is additionally rated by human evaluators on crowd-sourcing platforms (5 raters per layout, median rating used). The network achieves approximately 0.85 Spearman rank correlation with human ratings — meaning it agrees with humans on which placements are better in 85% of pairwise comparisons.

Training time: 3-5 days on a single GPU. Cost: under $1,000.

## 2.3 The VLM Aesthetic Re-Ranking

The geometric network identifies the top 20-50 physically valid, geometrically reasonable positions. The VLM (Qwen2.5-VL + 3D-RoPE) then re-ranks these candidates based on design intelligence that geometry cannot capture.

The VLM receives: the original photograph, the cell grid's 3D layout, and the 20-50 candidate positions with their geometric scores. For each candidate, the VLM evaluates style coherence (does this object match the room's aesthetic?), functional logic (does a plant in this position make design sense?), visual relationships (does this position create interesting or cluttered compositions?), and cultural context (minimalist vs. eclectic density preferences).

The VLM outputs a re-ranked list of the top 5 positions with brief justifications. The highest-ranked position becomes the system's recommendation.

The VLM evaluation takes approximately 50 milliseconds for 20-50 candidates — fast enough for interactive use. The VLM is not in the per-frame rendering loop — it is invoked only when a new placement is being evaluated.

## 2.4 The Division of Labor

The geometric network and the VLM handle different aspects of aesthetics:

The geometric network handles: spatial balance (center of mass), spacing rhythm (inter-object distances), symmetry alignment (room axes), occupancy distribution (avoiding clustering). These are measurable from the top-down layout alone.

The VLM handles: style matching (modern vs. rustic vs. minimalist), contextual reasoning (a plant adds life to a plain corner but clutter to a busy shelf), cultural norms (Japanese rooms prefer open floor space), design sophistication (professional designers use asymmetry intentionally). These require visual understanding and world knowledge.

Neither alone achieves what the combination provides. The geometric network without the VLM produces spatially balanced but stylistically ignorant placements. The VLM without the geometric network produces stylistically appropriate positions that might violate spacing or balance constraints.

---

# Gap 3: The Interaction Model

---

## 3.1 The Heatmap Interface

The core interaction element for manual placement is the placement heatmap — a color-coded overlay on the 3D view showing the validation score at every position. Green indicates excellent positions (score above 0.8), yellow indicates acceptable positions (0.5-0.8), red indicates poor or invalid positions (below 0.5), and black indicates physically impossible positions (collision or out-of-bounds — score zero).

The heatmap replaces text-based placement descriptions entirely. Instead of the system saying "place the plant at position (1.5, 0.0, -0.8)" — which is meaningless to most users — the user sees all the good spots light up in the 3D scene. They can immediately understand why certain areas are green (on the floor, good spacing, near a window) and why others are red (inside furniture, too close to a wall, blocking a pathway).

The heatmap is computed by evaluating the validation score at a coarse grid (10cm spacing) across all visible surface cells, then interpolating for smooth visualization. This requires 2,000-5,000 evaluations — each a cell property lookup plus a composite score calculation — taking 0.2-0.5 seconds.

## 3.2 The Interaction Flow

The complete user interaction for manual placement follows seven steps.

Step 1 — Scene Construction: The user uploads a photograph. The system runs the perception pipeline and builds the cell grid. During the initial 1.5 seconds, the user sees a loading indicator. After 1.5 seconds, navigation begins in flat-shaded mode. Shell extension and AI texturing continue in the background.

Step 2 — Mode Selection: The user switches from navigation mode to placement mode. This activates the object catalog and validation system.

Step 3 — Object Selection: The user selects the object they want to place from a visual catalog — choosing by thumbnail appearance, not by typing names. Each catalog entry has pre-stored dimensions, weight, surface affinity, and functional requirements.

Step 4 — Heatmap Generation: The system evaluates candidate positions on a hierarchical grid (coarse 50cm → medium 10cm → fine 2cm in promising areas) against all five validation fields. The functional and aesthetic fields are category-specific, so this computation is triggered by the object selection. The result is the placement heatmap overlaid on the 3D scene.

Step 5 — Exploration: The user navigates the 3D scene with the heatmap visible. Tapping any position shows the detailed score breakdown: "Score: 0.87. Collision: 0.95 (42cm clearance). Surface: floor (confidence 0.92). Functional: 0.82 (1.3m from seating). Pathway: 0.91 (68cm clear). Aesthetic: 0.79." The system highlights the top 3-5 recommended positions.

Step 6 — Adjustment: The user can refine the placement using three mechanisms. Tap a reference point on the scene to add a proximity constraint (the heatmap shifts toward the tapped point with a Gaussian falloff of 1 meter). Long-press a position to exclude it (the heatmap adds a penalty near that point). Adjust criterion weights using sliders (increasing "spacing" importance shifts the heatmap toward more open positions). Each adjustment triggers an instant heatmap update — under 100 milliseconds, since only the composite score is recalculated, not the underlying field values.

Step 7 — Confirmation: The user taps their chosen position. The object appears in the 3D scene at that position. The user can rotate the view to verify the placement from different angles. Accept or undo.

## 3.3 Coordinate-Based Constraints Replace Language

Instead of expressing placement constraints in natural language ("near the window, but not too close to the sofa"), the user expresses them through direct spatial interactions.

A single tap on a point in the 3D scene means: "add a proximity bonus centered at this 3D position." The heatmap shifts toward the tapped point.

A long press means: "add a proximity penalty centered here." The heatmap shifts away from the pressed point.

A pinch gesture on the floor defines a rectangular region meaning: "restrict the search to this area." Positions outside turn black.

A slider adjustment for "spacing" means: "increase the weight of the collision clearance criterion." The heatmap reweights to favor more open positions.

Each interaction is unambiguous (a tap has an exact 3D position, computed by raycasting through the cell grid), composable (multiple constraints combine multiplicatively), reversible (tap again to remove), and instant (recomputation takes under 100 milliseconds).

## 3.4 Precomputation Strategy

The critical performance design is which computations are precomputed and which are triggered by user actions.

Precomputed once per scene (during Step 1): the cell grid, the room boundary (shell extension planes), and the surface support field. These are scene properties independent of what object is being placed.

Precomputed as a distance field (during Step 1): the minimum distance from every cell to the nearest non-empty cell. This is the collision clearance field. It depends only on existing objects, not the new object. The collision check for a specific object is then a comparison: is the clearance at this position greater than half the object's diagonal? This comparison is instantaneous.

Computed per object category (during Step 4): the functional field and the aesthetic field. A plant's functional field differs from a lamp's. These are computed after the user selects an object category but remain cached if the user switches back.

Recomputed on adjustment (during Step 6): only the composite score formula. The individual field values remain cached. The heatmap update is instantaneous because it is just reweighting pre-computed values.

## 3.5 Auto-Furnishing Mode

In auto-furnishing mode (no user interaction), the VLM takes the role of the user. It identifies functional gaps in the cell grid ("this living room needs a coffee table"), selects the appropriate object category, and evaluates the top candidates from the validation system. The VLM provides lookahead planning — before placing the sofa (the anchor piece), it considers whether the proposed position leaves adequate space for the coffee table, TV, and lamps that will follow.

The validation heatmap is still computed internally — the VLM selects from the same set of validated positions that a human user would see. The difference is that the VLM makes the selection automatically based on its trained design intelligence.

Sequential multi-object placement updates the validation fields after each placement. The sofa's cells are added to the collision field. The coffee table's position changes the aesthetic balance for the next item. Each subsequent object is placed on updated fields.

---

# Gap 4: Automatic Scale Recovery

---

## 4.1 The Scale Problem

Depth Pro estimates metric depth, but this estimate has a systematic scale error of 0.5-2%. A 2% scale error means every distance in the reconstruction is off by 2% — a 3-meter room is reconstructed as 3.06 meters, a 45-centimeter table height becomes 45.9 centimeters. This error is small in percentage terms but compounds in the cell grid: cell positions, gradients, and integrals all inherit the scale error.

Scale calibration using a known-size object can reduce this error to below 0.3%, improving cell position accuracy by 50-70%.

## 4.2 The Reference Object Priority

The system searches for known-size objects in the following priority order, based on frequency of occurrence and reliability as scale references.

Priority 1 — Interior doors: standard interior doors are approximately 200 centimeters tall by 80 centimeters wide (regional variations: 203×81.3cm US, 200×80cm Europe and Asia). SAM 3 detects doors as a semantic category. If a door is fully visible, the system measures its apparent height in the cell grid (the vertical span of the door cell cluster), compares to the expected standard height, and computes the scale correction factor. Doors are the best reference because they are present in virtually every room, they are large (strong geometric signal), and their dimensions are highly standardized.

Priority 2 — Standard furniture heights: kitchen counter (90-92cm), dining table (74-76cm), desk (72-75cm), dining chair seat (43-47cm), sofa seat (38-45cm). If SAM 3 detects these categories, the system estimates the furniture's height from the cell grid (top surface cell height minus floor cell height) and compares to the standard height.

Priority 3 — Floor tiles: standard sizes (30×30, 45×45, 60×60cm). If the floor has visible tiles (detectable from the albedo pattern in floor cells), the apparent tile spacing provides X and Z axis calibration simultaneously.

Priority 4 — Ceiling height: standard residential heights (240cm most of world, 244cm US 8-foot, 270cm many European countries). Less precise because ceiling heights vary more than door sizes.

Priority 5 — Common objects: A4/Letter paper (21.0×29.7cm / 21.6×27.9cm), standard monitors (24-inch ≈ 53cm wide), standard outlet face plates (7×12cm US). Less reliable but useful backup references.

## 4.3 Resolving Conflicting Scale Estimates

When multiple reference objects are detected, they may suggest slightly different scale corrections. Conflicts arise from measurement noise (pixel-level uncertainty in boundary detection), non-uniform scale error (wide-angle lens distortion), or misidentification (SAM labels a bookshelf as a door).

Resolution strategy: compute the scale correction for each reference object independently. Discard outliers — any correction differing from the median by more than 10% is likely a misidentification. Compute the weighted average of remaining factors, weighted by the reference object's apparent size in the image (larger objects provide more reliable measurements) and its SAM detection confidence.

The final scale correction is applied to the entire depth map as a multiplicative adjustment before backprojection to 3D. After correction, the scale error is typically reduced to 0.2-0.5%, corresponding to 0.6-1.5 centimeters of positional error at 3 meters.

## 4.4 When No Reference Object Is Found

In some images, no standard-size reference object is visible. The system falls back on Depth Pro's raw metric estimation (0.5-2% scale error).

The system informs the user that scale calibration was not performed and that positional accuracy may be reduced. It suggests remedies: take a wider photograph that includes a door or furniture, or place a known-size object (like a sheet of paper) in the scene and retake.

## 4.5 Implementation in the Prototype

Scale calibration is implemented in the prototype as `scale_calibration.py`. The implementation detects the largest vertical rectangle in the SAM segmentation (likely a door), computes its height in the depth map, and calculates the correction factor. The correction is applied to the depth map before bilateral smoothing and backprojection.

The scale calibration step adds approximately 50 milliseconds to the perception pipeline — negligible compared to Depth Pro inference (300ms) and SAM inference (200ms).

---

# Gap 5: VGGT and the Path to Unified Perception

---

## 5.1 What VGGT Is

VGGT (Visual Geometry Grounded Transformer), released by Meta, represents the next evolution beyond DUSt3R and MASt3R. It is a single feed-forward transformer that takes multiple images and directly outputs camera poses, depth maps, point maps, and dense 3D point clouds — all in a single forward pass, without iterative optimization or multi-stage pipelines.

The key advance over DUSt3R is simultaneous processing: VGGT processes all images at once rather than in pairs, establishing global correspondences in a single pass. No pairwise alignment step. No global optimization that can fail on repetitive scenes.

## 5.2 Relevance to Our Pipeline

For the current single-image prototype, VGGT's significance is forward-looking. The specialized models (Depth Pro for metric depth, SAM 3 for segmentation) currently outperform VGGT for their specific tasks because they were individually optimized.

Three developments will change this. First, VGGT or its successors will likely match specialized models' single-image accuracy within 1-2 years. When this happens, the multi-model fusion pipeline (Depth Pro + SAM 3, optionally DUSt3R) can be replaced by a single VGGT forward pass. Second, VGGT naturally extends to multi-image input with linear cost growth — adding more images adds more tokens to the transformer, and accuracy improves smoothly. Third, VGGT's architecture accepts additional task heads (segmentation, scene graph, placement prediction) without fundamental redesign.

## 5.3 Why the Cell Architecture Is Model-Agnostic

The cell grid does not care which perception model produced the input point cloud. The cell construction (point binning, gradient computation, integral calculation, neighbor population) operates on 3D points with colors, normals, and semantic labels. Any model that outputs these can feed the cell grid.

When VGGT or its successors replace the current perception stack, the change is confined to `perception/pipeline.py`. The cell grid construction, rendering, navigation, AI texturing, and validation are all unchanged. The cells simply receive more accurate points, which produce more accurate cells with cleaner gradients.

This modularity protects the investment in the validation system, the AI texturing model, and the rendering engine — the most complex and hardest-to-build components — while allowing the perception stack to be upgraded as better models become available.

## 5.4 The Convergence Timeline

Based on the pace of recent developments (DUSt3R late 2023, MASt3R mid-2024, VGGT early 2025):

By late 2026 to early 2027: unified models that take a single image and output metric depth, segmentation, 3D point maps, and camera parameters in one forward pass, matching current specialized models. The pipeline reduces from three perception models to one.

By 2027-2028: models that additionally perform spatial reasoning and placement recommendation as part of the same forward pass. The full perception-to-placement pipeline becomes a single model call. But coordinate-validation remains valuable for its guarantees (no floating objects, no collisions, no out-of-bounds placements) that a pure neural network cannot provide.

---

# Summary: The Complete Pipeline

---

## The Full Flow

Step 1 — Capture: user photographs the scene with any camera.

Step 2 — Perception: Depth Pro produces metric depth (0.3s). Bilateral filter smooths noise (5ms). Scale calibration detects known-size objects and corrects depth (50ms). SAM 3 produces semantic segmentation (0.2s). Point cloud is constructed via backprojection.

Step 3 — Cell Grid: point cloud is binned into 5cm cells. Gradients, integrals, neighbors computed. Failure modes (glass, mirror, dark, sky, specular) detected and mitigated via confidence and forced density. Per-cell confidence assigned from depth smoothness, point density, semantic penalties. Result: ~80K cells, ~40MB.

Step 4 — Shell Extension: RANSAC detects floor/wall/ceiling planes. New cells generated along extended planes. Room becomes navigable from all angles. (Background, 10-15s.)

Step 5 — Navigation Begins: user can walk around in flat-shaded mode while AI texturing loads.

Step 6 — AI Texturing: Pix2PixHD-Lite GAN takes cell buffer renders and produces photorealistic output. Light values written back to cells via cell ID buffer with view-angle weighting. Per-cell temporal blending prevents flicker. Only dirty cells (10-30%) re-lit each frame.

Step 7 — Validation: conservation checks (energy, mass, shadow direction) run async, one frame behind. Corrections applied gradually over 3-5 frames.

Step 8 — Placement (if activated): user selects an object from catalog. Validation fields computed from cell grid. Heatmap displayed. User explores, adjusts constraints, selects position. VLM re-ranks top candidates. Object placed in cell grid.

Step 9 — Auto-Furnishing (if activated): VLM identifies gaps. Objects retrieved and style-adapted. Validation system finds optimal positions on cell grid. Objects placed sequentially with field updates between each.

Total time from photograph to navigation: ~1.5 seconds (flat shading), ~15 seconds (full photorealistic).

Cell grid accuracy: 5-10cm from single image (after calibration and smoothing), 2-5cm from multi-image, sub-centimeter from video.

Physics violation rate: below 1%, guaranteed by deterministic validation on the cell grid regardless of perception model accuracy.
