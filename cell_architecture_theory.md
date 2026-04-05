# Differential-Integral Cell Architecture
## The Universal Primitive for Perception, Rendering, Physics, and Animation

---

# Chapter 1: The Architectural Shift

## 1.1 One Primitive Replaces Everything

The entire system now operates on a single data primitive: the differential-integral cell. Every previous component — surfels, octrees, density fields, LOD hierarchies, streaming indices, normal maps, displacement maps, material maps, animation deformation fields — is replaced by properties and derivatives stored within cells.

A cell is a small cube of space that knows everything about itself: what it contains (density, material), what it looks like (color, texture, roughness), how it changes across its volume (gradients), how those changes change (curvature), the total of each property within its volume (integrals), and how everything evolves over time (temporal derivatives). From these properties alone, the system reconstructs, renders, animates, validates, and navigates arbitrary 3D environments at any resolution.

## 1.2 What a Cell Stores

Every cell in the scene stores the following data, organized by domain.

Geometry: density value (how solid this cell is, from 0.0 for air to 1.0 for solid wall), density gradient (the direction and rate at which solidity changes — a high gradient means a surface boundary passes through this cell, and the gradient direction points from solid toward empty), density integral (total mass within the cell volume — used for weight validation and LOD merge conservation), normal value (average surface direction within this cell), normal gradient (surface curvature — how fast the normal rotates across the cell — this IS the bump detail), and normal curvature (second derivative — rate of curvature change, distinguishing sharp edges from smooth curves).

Visual: albedo value (average true surface color, lighting removed), albedo gradient (color variation direction and rate — this IS the texture grain, the pattern flow, the material detail), albedo second derivative (how fast the pattern changes — fine grain versus coarse grain), albedo integral (total color energy — used for correct averaging when merging cells at lower LOD), and texture features (a 32-float neural encoding that, when decoded with a sub-cell position, reveals arbitrarily fine detail beyond what the explicit gradients capture).

Lighting: light value (current illumination color and intensity from the AI texturing model), light gradient (where shadows begin, where highlights peak — the gradient encodes the smooth transition of a penumbra without needing per-pixel shadow computation), light integral (total light energy received — used for energy conservation validation — a cell cannot reflect more light than this integral), and light temporal derivative (how fast the lighting is changing — if near zero, skip re-lighting this cell, saving computation).

Material: semantic label (what this cell contains — wood, glass, fabric, skin, concrete, foliage, water, sky, or any of SAM 3's 270,000+ categories), roughness value and gradient (average shininess and how it varies across the cell — varnished center versus raw edge), and reflectance value and gradient (average reflectivity and variation).

Animation: velocity (current movement speed in three dimensions — zero for static cells, nonzero for character or vehicle cells), acceleration (rate of velocity change — ensures smooth animation without sudden jumps), deformation function (the full trajectory of this cell over time, derived from the motion generation model), and texture shift function (how the cell's color changes over time — fabric folds shifting as a character moves).

Spatial: six neighbor summaries (the type, density, normal, and light value of each adjacent cell — enabling local queries without global search), and six flux values (the energy and material flow rate between this cell and each neighbor — light flux determines how much illumination passes between cells, density flux is zero at wall boundaries and high at air-air boundaries).

Hierarchy: eight children (sub-cells at half the resolution, created on demand when higher detail is needed), one parent (the cell at double the resolution that contains this cell), and the cell's current size (ranging from 0.5 centimeters at maximum detail to 20 centimeters at minimum detail).

The total storage per cell is approximately 512 bytes. For a typical room at 5-centimeter base resolution with 80,000 surface cells, the total scene data is approximately 40 megabytes.

---

# Chapter 2: The Pipeline With Cells

## 2.1 Stage 1 — Perception (Unchanged Inputs, New Output)

The perception models remain identical: Depth Pro estimates metric depth, DUSt3R ensures geometric consistency, SAM 3 provides semantic segmentation. These models process the photograph and produce a labeled 3D point cloud exactly as before.

The change is in what happens to the point cloud. Previously, points were converted to 2D Gaussian Surfels. Now, points are converted to cells.

The conversion process: divide the scene volume into a regular grid at the base resolution (5 centimeters). For each grid cell, collect all 3D points that fall within it. If a cell contains points, classify it based on the point density and semantic labels. Cells with many points from a solid object (wall, furniture) are classified as solid. Cells with points from a surface boundary (the top of a table, the face of a wall) are classified as surface. Cells with no points are classified as empty. Cells with points from transparent or semi-transparent objects (glass, foliage) are classified as transparent with a density between 0 and 1.

For each non-empty cell, compute the initial property values from the contained points. The albedo is the average color of the points. The normal is the average surface direction (computed from local point cloud normals). The density is proportional to the point count (more points indicate denser material). The semantic label is the majority vote of the points' SAM 3 labels.

The gradient computation uses finite differences between adjacent cells. The albedo gradient in the x-direction is (albedo of the cell to the east minus albedo of the cell to the west) divided by two cell widths. Similarly for y and z directions, and for all other properties. Second derivatives are computed as finite differences of the first derivatives.

The integral computation sums the property value times the cell volume for each cell. The density integral is density times cell volume, giving the approximate mass.

The neighbor summaries are populated by reading adjacent cells' type and value data. The flux values are initially set to zero and computed during the lighting stage.

This conversion takes 0.5-1 second for a typical room. The output is a sparse cell grid — only non-empty cells are stored.

## 2.2 Stage 2 — Room Shell Extension

The room shell extension works directly on the cell grid. Visible walls, floor, and ceiling are already represented as bands of surface cells. Extension means generating new cells beyond the observed region.

Wall extension: identify the plane equation of each visible wall (from the surface cells' positions and normals). Generate new surface cells along the continuation of this plane, filling in cells at the same density, normal direction, and albedo as the observed wall cells. The albedo gradient of the new cells is set to match the observed wall's gradient (continuing the same texture pattern).

Floor extension: the floor plane is extended in the same way. Floor cells are generated at the floor height across the entire room footprint. The albedo and gradient are matched to the observed floor (continuing the hardwood grain pattern, tile pattern, or carpet texture).

Ceiling extension: ceiling cells are generated at the estimated ceiling height with the observed ceiling properties.

For terrain environments, the extension generates cells along the terrain heightmap. The heightmap is extrapolated beyond the observed region using fractal terrain generation conditioned on the observed terrain characteristics (roughness, amplitude, dominant frequency). New terrain cells follow the extrapolated heightmap with appropriate albedo gradients (matching the observed ground texture).

## 2.3 Stage 3 — AI Texture Generation for Extended Cells

Extended cells have correct geometry (position, normal) but approximate visual properties (albedo is extrapolated from the observed portion). The AI texture generation refines these.

A diffusion-based texture synthesis model receives the extended cell grid and generates detailed albedo values, gradients, and neural texture features for each extended cell. The model is conditioned on the observed cells' properties — it learns to continue the texture pattern seamlessly from the observed to the generated region.

The model also generates the light response for all cells. Instead of computing lighting mathematically, the AI texturing model examines the cell grid (knowing what each cell is, where light sources are, what's around each cell) and paints the correct illumination. The light value, gradient, and integral are set by the AI model's output.

The light temporal derivative is initially zero (static lighting). If the scene includes dynamic lighting (time-of-day changes, moving lamps), the temporal derivative is updated each time the lighting conditions change.

## 2.4 Stage 4 — Object Backside Completion

Objects visible in the photograph have front-facing surface cells (from the photo) but no back-facing cells. SAM 3D reconstructs the complete 3D shape of each object, and the back-facing cells are generated from this reconstruction.

The new back-facing cells inherit their semantic label from the front-facing cells of the same object. Their albedo is generated by SAM 3D's texture prediction. Their gradients are computed from the generated albedo variation. Their density is set to match the front-facing cells (same solidity).

Backside completion is performed lazily — only when the camera approaches an object from behind. This saves computation for objects the user never walks behind.

## 2.5 Stage 5 — Auto-Furnishing

The VLM identifies functional gaps in the cell grid. It examines the semantic labels and spatial arrangement of cells and determines what's missing: "This living room has sofa cells and TV cells but no coffee table cells between them."

For each needed item, a 3D model is retrieved from the furniture database and converted to cells. The conversion is the same process as the initial point-to-cell conversion: divide the model's 3D volume into cells, classify each cell, compute properties and derivatives.

The retrieved furniture cells are style-adapted: their albedo values and gradients are adjusted by a neural style transfer to match the room's existing cells' visual style.

Placement uses the coordinate-validation system operating on the cell grid. Collision detection is a cell-type lookup — are the target cells empty? Surface detection is a cell-type check — are there surface cells below the placement position? The validation is near-instantaneous because every check is a cell property query, not a geometric computation.

The placed furniture cells are inserted into the cell grid. Their neighbors are updated. Adjacent cells' flux values are recalculated. The AI texturing model is re-run on the local region to generate correct lighting for the new furniture and its surroundings.

## 2.6 Stage 6 — Character Creation and 4D Animation

Characters are cell clusters marked as dynamic. Each character cell has a deformation function derived from text-to-motion generation.

The motion pipeline: the director types a text prompt ("walk to the window"). The VLM converts this to a motion plan with 3D waypoints. A text-to-motion model (MoMask or MotionGPT) generates skeletal animation data. The skeletal joint positions are converted to per-cell deformations using linear blend skinning — each character cell is influenced by its nearest skeleton joints, with weights determined during character creation.

The per-cell motion data is stored as velocity and acceleration rather than explicit positions at each frame. The velocity tells the renderer how to extrapolate the cell's position between keyframes, ensuring smooth motion. The acceleration ensures the velocity changes gradually, preventing jittery animation.

The velocity integral over any time interval gives the total displacement, which must match the actual position change. If drift accumulates (the integrated velocity doesn't match the target position from the skeletal animation), a correction term is applied to the velocity to bring the cell back on track. This integral-based validation ensures temporal consistency without explicit per-frame position storage.

Dynamic cells' texture shifts are stored as temporal derivatives of albedo: ∂color/∂t. As a character walks, the fabric folds change — the albedo gradient on clothing cells shifts direction. The temporal derivative encodes this shift so the renderer can interpolate smoothly between frames.

## 2.7 Stage 7 — Navigation and Learned Physics

Navigation uses cell properties directly. No separate physics system exists.

Camera collision: query the cell at the proposed camera position. If the cell type is solid or surface with density above the threshold, block movement. If empty, allow movement. The density gradient at the boundary tells the system the exact surface position within the cell, enabling sub-cell-precision collision without sub-cell-resolution cells.

The collision precision from density gradient: if a 5-centimeter cell has density gradient (0.8, 0, 0), the surface boundary is at approximately 60% from the left edge of the cell — a position known to within 1-2 centimeters even though the cell is 5 centimeters wide. The gradient provides sub-cell precision for free.

Floor following: query the cell column below the camera. The highest surface cell in that column is the floor at this position. The camera height is set to this floor cell's height plus eye height (1.6 meters). The normal of the floor cell determines the slope. The normal gradient determines if the slope is changing (approaching a hill crest or a valley bottom).

Movement speed: the semantic label of the floor cell determines movement speed. Road cells: full speed. Grass cells: 85% speed. Sand cells: 70% speed. Steep slope (from normal value): reduced speed proportional to incline. Water cells: blocked.

World boundary: cells exist only within the generated world. Beyond the last cell, the camera cannot proceed. The boundary is soft — the last few cells before the edge have a visual fade effect (fog increasing with proximity to the boundary).

## 2.8 Stage 8 — Rendering

The renderer draws all visible cells at their current resolution. The rendering pipeline per frame:

Determine visible cells by checking which cells fall within the camera's view frustum. The cell hierarchy enables fast culling — if a parent cell is entirely outside the frustum, all its children are skipped without individual testing.

Select resolution per cell based on distance from camera. Near cells (within 2 meters): subdivide to 2-centimeter resolution. The subdivision uses Taylor expansion from parent derivatives to predict child values, costing approximately 0.01 milliseconds per cell. Medium cells (2-10 meters): keep at base 5-centimeter resolution. Far cells (beyond 10 meters): merge to 10-20-centimeter resolution. The merge uses the children's integral values to compute correct parent averages.

For each visible cell at its selected resolution: decode the neural texture features at the cell's position to get the fine-detail color. Combine with the albedo value and gradient to get the final surface color at this point. Apply the normal value and gradient to determine the surface orientation for shading. Apply the light value and gradient to get the illumination at this point.

The AI texturing model enhances the raw cell render to photorealistic quality. In real-time mode, a lightweight model runs at 5-15 milliseconds per frame. In production mode, a heavy model runs at 2-10 seconds per frame. In high-detail mode, a full AI video regeneration model runs at 30-60 seconds per frame.

The math validation layer checks each frame against physical laws using the integral data. Light energy conservation: the total reflected light (sum of all visible cells' light values times their areas) must not exceed the total incoming light (sum of all light source cells' emissive values). Shadow direction: the light gradient direction must be consistent with the known light source positions. Density conservation: the total density integral of the scene must remain constant (matter does not appear or disappear). Any violations trigger a correction applied as a per-cell adjustment before display.

---

# Chapter 3: How Derivatives Replace Separate Systems

## 3.1 Normal Gradient Replaces Bump Maps

Previously, bump detail required a separate normal map texture stored alongside each surfel. The normal map encoded per-pixel perturbations to the surface normal, creating the illusion of surface roughness.

In the cell architecture, the normal gradient IS the bump detail. A cell with normal value (0, 1, 0) (pointing up — a floor surface) and normal gradient (0.1, 0, 0.05) has a surface that tilts slightly as you move across it in the x-direction. This tilt IS the bump. A brick wall has normal gradient that oscillates between the mortar grooves (normal tilts inward) and the brick faces (normal returns to flat). The oscillation frequency corresponds to the brick spacing.

When the cell subdivides, the children inherit these gradient-predicted normals. The left child gets a slightly different normal than the right child, creating visible bumps at the finer resolution. No separate bump map is needed — the bumps emerge from the derivatives.

For surface detail finer than the gradient can capture (individual pores in leather, thread-level weave in fabric), the neural texture features encode this information. The features are decoded at the sub-cell level during rendering, providing arbitrary detail resolution from a compact encoding.

## 3.2 Albedo Gradient Replaces Texture Maps

Previously, texture detail required neural texture networks or explicit texture maps. The cell's albedo gradient replaces this for the dominant texture direction.

A wood surface cell has albedo gradient (0.15, 0, 0) meaning the color gets lighter moving in the x-direction — this IS the wood grain direction. When subdivided, the left child is darker (following the grain) and the right child is lighter. The grain pattern appears automatically.

The second derivative of albedo captures grain SPACING. A high second derivative means the grain pattern oscillates rapidly (fine-grained wood). A low second derivative means the pattern changes slowly (wide-plank wood). When subdivided to fine resolution, the second derivative determines how many grain lines appear per centimeter.

For complex textures that cannot be captured by gradients alone (a Persian rug pattern, a painting on a wall, a complex fabric print), the neural texture features provide the additional detail. The gradient captures the dominant pattern direction and frequency; the neural features capture everything else.

## 3.3 Density Gradient Replaces Displacement Maps

Previously, displacement maps moved the surface position to create real geometric bumps visible in silhouette. The density gradient achieves the same effect.

A cell at a surface boundary has density that transitions from 1.0 (solid) to 0.0 (empty). The density gradient points from solid toward empty, and its magnitude indicates how sharp the transition is. A sharp gradient means a hard surface boundary. A gradual gradient means a fuzzy boundary (like foliage or fabric fringe).

The position within the cell where density crosses the 0.5 threshold is the precise surface position. This position can be computed from the gradient: surface_offset = (0.5 - density_value) / gradient_magnitude. This gives sub-cell surface positioning.

For bumpy surfaces (brick mortar, carved wood), the density gradient varies across the cell. In some sub-regions the surface is closer to one side (the mortar groove); in other sub-regions it's closer to the other side (the brick face). When subdivided, the children reveal these surface position variations as actual geometric displacement. The displacement IS the density gradient variation.

## 3.4 Light Gradient Replaces Shadow Maps

Previously, shadows required rendering the scene from each light source's perspective and computing depth comparisons. The light gradient replaces this.

The AI texturing model computes the lighting for each cell once (including shadows, indirect illumination, ambient occlusion). The result is stored as the light value, and the spatial variation of lighting is stored as the light gradient.

A shadow edge passing through a cell appears as a strong light gradient: bright on one side (lit), dark on the other side (shadowed). The gradient direction points from shadow toward light. The gradient magnitude indicates the shadow sharpness (soft penumbra has a small gradient over a large distance; hard shadow has a large gradient over a short distance).

When the cell subdivides, the children on the lit side are bright and the children on the shadowed side are dark — the shadow edge is naturally resolved at finer resolution. No shadow map computation is needed at render time.

The light temporal derivative enables efficient dynamic lighting. When the AI texturing model updates the lighting (due to camera movement or time-of-day change), it only needs to re-compute cells where ∂L/∂t is significantly non-zero. Cells where the lighting is stable (∂L/∂t ≈ 0) are skipped. For a typical scene, only 10-30% of cells need lighting updates per frame — the rest are in regions where the lighting doesn't change with small camera movements.

## 3.5 Integrals Replace Conservation Checks

Previously, energy conservation and physical validation required separate computation. The integral data built into each cell enables instant validation.

The density integral of the entire scene is the total mass. This value must remain constant. If a cell's density changes (corruption, rendering error, animation artifact), the density integral detects the violation. The correction is local — adjust the offending cell's density until its integral matches the expected value.

The light integral of each cell is the total light energy it contains. The reflected light (albedo times light integral) plus the absorbed light ((1 minus albedo) times light integral) must equal the incoming flux from neighbors. If reflected exceeds incoming, the AI texturing model made a physics error. The math validation corrects it by scaling the light value until conservation holds.

The velocity integral of each dynamic cell over a time interval is the total displacement. This must match the actual position change from the deformation function. If drift accumulates (integrated velocity diverges from target position), a correction velocity is added. This ensures animation remains smooth and accurate over arbitrarily long sequences.

---

# Chapter 4: Resolution Scaling

## 4.1 The Subdivision Protocol

When higher detail is needed (camera approaches, production render, high-detail mode), a cell subdivides into eight children. Each child covers one octant of the parent's volume at half the parent's size.

The children's properties are predicted from the parent's derivatives using Taylor expansion. For any property P with value v, gradient g, and second derivative h:

child_value = v + g · offset + 0.5 × h · offset²

where offset is the vector from the parent's center to the child's center.

This prediction is accurate for smoothly varying properties (most natural surfaces). The neural texture features provide correction for properties that vary in ways the Taylor expansion cannot predict (sharp discontinuities, complex patterns, stochastic textures).

The subdivision cost is approximately 0.01 milliseconds per cell (evaluating the Taylor expansion for all properties) plus 0.1 milliseconds if neural texture correction is applied. For a production render subdividing 200,000 cells from 5-centimeter to 1-centimeter resolution (each cell producing 125 children), the total subdivision time is approximately 20-30 seconds.

## 4.2 The Merge Protocol

When lower detail is sufficient (camera recedes, object moves to distance), children merge back into their parent. The parent's values are recomputed from the children's integrals.

The parent's value is the volume-weighted average of the children's values. The parent's gradient is the best-fit linear approximation across the children's values. The parent's integral is the sum of the children's integrals.

Because the integrals are conserved during merging, no information is lost at the aggregate level. The fine detail is lost (individual children's variations are averaged out), but the total color energy, total mass, and total light energy are preserved exactly. This means that a distant object that has been merged to low resolution has the correct overall brightness and color — it just lacks fine texture detail.

## 4.3 Resolution Tiers

The system operates at five resolution tiers, selected per cell based on distance from camera and render mode.

Tier 0 (exploration): 20-centimeter cells. Used for cells beyond 15 meters. Approximately 125 cells per cubic meter of surface. Minimal detail — shapes and colors are visible but texture is absent. Suitable for distant background.

Tier 1 (navigation): 5-centimeter cells. Used for cells at 3-15 meters during real-time navigation. Approximately 8,000 cells per cubic meter of surface. Dominant texture patterns visible (wood grain direction, brick pattern). Sufficient for comfortable navigation.

Tier 2 (inspection): 2-centimeter cells. Used for cells within 3 meters during real-time navigation, and all visible cells during production render. Approximately 125,000 cells per cubic meter. Fine texture detail visible (individual grain lines, fabric weave pattern). Suitable for close inspection and production output.

Tier 3 (production): 1-centimeter cells. Used for all visible cells during production render. Approximately 1,000,000 cells per cubic meter. Very fine detail (wood pores, fabric thread, skin texture). Suitable for close-up production shots.

Tier 4 (cinematic): 0.5-centimeter cells. Used during high-detail render mode. Approximately 8,000,000 cells per cubic meter. Maximum detail — sub-millimeter surface features. Suitable for extreme close-ups and cinematic hero shots.

## 4.4 Memory Budget by Tier

For a typical room (75 cubic meters total, approximately 15 cubic meters of surface volume):

Tier 0 at 20cm: 1,875 cells × 512 bytes = 1 MB
Tier 1 at 5cm: 120,000 cells × 512 bytes = 60 MB
Tier 2 at 2cm: 1,875,000 cells × 512 bytes = 960 MB
Tier 3 at 1cm: 15,000,000 cells × 512 bytes = 7.5 GB
Tier 4 at 0.5cm: 120,000,000 cells × 512 bytes = 60 GB

During real-time navigation, only cells near the camera are at high tiers. A typical view has approximately 5,000 cells at Tier 2, 20,000 cells at Tier 1, and 5,000 cells at Tier 0 — total approximately 15 MB in GPU memory. This is extremely efficient.

During production render, the visible portion (approximately half the room) at Tier 3 is approximately 3.5 GB — feasible on an A100 (80 GB) or even an RTX 4090 (24 GB) with streaming.

During cinematic render, Tier 4 is streamed in blocks — only the cells visible in the current frame are at full resolution. The streaming system loads and unloads cell blocks between frames.

---

# Chapter 5: Coordinate-Validation on Cells

## 5.1 Instant Collision

The collision field — previously the most critical validation component — becomes a simple cell lookup.

To check if position (x, y, z) is free for placement:

Step 1: Compute the cell index from the coordinates: ix = floor(x / cell_size), iy = floor(y / cell_size), iz = floor(z / cell_size).

Step 2: Look up the cell at (ix, iy, iz) in the sparse cell grid.

Step 3: If the cell is empty (type = empty), the position is free. If the cell is solid or surface (type = solid or surface), there is a collision.

Step 4: For sub-cell precision, use the density gradient. If the cell is a surface cell, the gradient tells us where within the cell the surface boundary is. The position may be on the empty side of the boundary (valid) or the solid side (collision) — the gradient resolves this to within 1-2 centimeters even at 5-centimeter cell resolution.

Total collision check time: one hash table lookup plus one gradient evaluation. Approximately 0.001 milliseconds. This is 100 times faster than the surfel density query (0.1 milliseconds) and 10,000 times faster than mesh-based collision detection.

For an object with a footprint spanning multiple cells, each cell within the footprint is checked. An object spanning 6×6 cells (a 30-centimeter object at 5-centimeter resolution) requires 36 cell lookups — approximately 0.036 milliseconds. Still effectively instant.

## 5.2 Surface Support from Cell Properties

The surface support check queries the cell type and normal below the placement position.

Step 1: Find the cell at the placement position.
Step 2: Check the cell below it (at y minus one cell). If that cell is a surface cell with an upward-pointing normal (ny > 0.85), there is a valid support surface.
Step 3: Read the surface cell's semantic label to determine the surface type (floor, table, shelf, terrain).
Step 4: Read the surface cell's density integral to estimate load capacity (denser cells can support more weight).

For terrain environments, the floor cell's normal gradient gives the local slope. Steep slopes (high normal gradient magnitude) reduce the surface support score.

## 5.3 Functional Queries from Neighbors

The functional field queries neighboring cells to determine spatial relationships.

To check if a placement position is near a window: search the cell grid for cells with semantic label "window." Compute the distance from the placement position to the nearest window cell. This is a spatial query on the cell grid — using the hierarchical structure, it takes approximately 0.01 milliseconds.

To check pathway clearance: trace a line of cells from the placement position to each doorway. If all cells along the line are empty, the pathway is clear. If any cell is solid, the pathway is blocked. This is a cell grid traversal — approximately 0.05 milliseconds per pathway.

To check neighbor relationships (is this placement near seating, near a wall, in an empty corner): read the neighbor summaries of the cells at and around the placement position. The summaries immediately tell you what's adjacent without searching the full grid.

## 5.4 Aesthetic Scoring

The geometric aesthetic scoring operates on the cell grid's spatial distribution.

Spatial balance: compute the center of mass of all non-empty cells, weighted by density. Compare to the room's geometric center. Positions that move the center of mass toward the room center score higher. This is a sum over all cells — approximately 0.1 milliseconds for 80,000 cells.

Spacing rhythm: compute the distances between adjacent object cell clusters. The coefficient of variation of these distances indicates spacing consistency. Lower variation scores higher.

The VLM aesthetic re-ranking remains unchanged — it receives the top 20-50 candidate positions and re-ranks them based on design intelligence. The cell grid provides the VLM with richer context than surfels did, because each cell's neighbor summaries give the VLM explicit information about spatial relationships.

---

# Chapter 6: AI Texturing on Cells

## 6.1 What the AI Model Receives

The AI texturing model receives the cell grid properties as input for each frame:

For each visible cell: position, normal value, albedo value, semantic label, neighbor types, and the cell's distance from detected light sources.

The model does not need gradients or integrals as input — those are used internally by the rendering and validation systems. The AI model only needs to know what each cell IS and what is around it to determine how it should look.

## 6.2 What the AI Model Outputs

For each visible cell, the AI model outputs the light value (how bright and what color this cell should appear under the current viewing conditions) and the light gradient (how the lighting varies across the cell — shadow edges, highlight positions).

These outputs are stored in the cell. They persist until the camera moves significantly or the scene changes, at which point the AI model is re-run on the affected cells.

## 6.3 Efficiency Through Temporal Derivatives

The AI model does not need to re-run on every cell every frame. The light temporal derivative (∂L/∂t) indicates how fast each cell's lighting is changing.

For cells in the middle of a wall (far from any shadow edge, far from any light source), the lighting barely changes as the camera moves slowly. Their ∂L/∂t is near zero. These cells are skipped — their existing light values are reused.

For cells near shadow edges, near light sources, or on specular surfaces, the lighting changes significantly with camera movement. Their ∂L/∂t is large. These cells are re-lit by the AI model.

Typically, only 10-30% of visible cells need re-lighting per frame during slow navigation, and 30-50% during fast movement. This reduces the AI model's workload proportionally, enabling higher frame rates.

---

# Chapter 7: Animation Through Cell Derivatives

## 7.1 Smooth Motion from Velocity and Acceleration

Dynamic cells (characters, vehicles) have velocity and acceleration stored as cell properties. The renderer uses these to compute the cell's position at any time without querying the deformation function.

Between keyframes (generated by the motion model at 30 frames per second), the cell's position is interpolated using its velocity and acceleration:

position(t + dt) = position(t) + velocity × dt + 0.5 × acceleration × dt²

This second-order integration produces smooth, natural-looking motion that correctly handles acceleration and deceleration. A character's foot slowing down as it lands is encoded as negative vertical acceleration — the cell decelerates smoothly rather than stopping abruptly.

## 7.2 Temporal Consistency Through Integral Validation

The velocity integral over any time interval must equal the actual displacement. If the integrated velocity diverges from the target position (from the skeletal animation), a correction is applied.

The correction is computed as:
error = target_position(t) - (initial_position + integral of velocity from 0 to t)
correction_velocity = error / remaining_time

This correction is distributed smoothly over future frames rather than applied as a sudden jump, maintaining smooth motion while preventing drift accumulation.

## 7.3 Texture Animation

As a character moves, their clothing wrinkles change, hair shifts, facial expressions evolve. These visual changes are encoded as temporal derivatives of cell properties.

The albedo temporal derivative (∂color/∂t) captures color changes: a shirt wrinkle deepening creates a darkening (∂brightness/∂t < 0) along the fold line. The normal temporal derivative captures surface shape changes: the wrinkle also changes the surface normal direction.

These temporal derivatives are generated by the motion model alongside the spatial deformation. The renderer applies them smoothly, producing visual changes that are synchronized with the motion.

---

# Chapter 8: Math Validation Layer

## 8.1 Conservation Laws

The integral data enables automatic validation of physical laws.

Energy conservation: for each cell, the reflected light energy (albedo times incoming light integral) plus absorbed energy ((1 minus albedo) times incoming light integral) must equal the total incoming energy (sum of neighbor flux toward this cell plus any emissive contribution). Violations indicate an AI texturing error.

Mass conservation: the total density integral of the scene must remain constant between frames. If a cell's density changes without a corresponding change in an adjacent cell, matter has been created or destroyed — a physics violation.

Momentum conservation for dynamic cells: the total momentum (sum of velocity times density integral for all character cells) should change only due to external forces (gravity, collision with static cells). If momentum changes without a corresponding force, the animation has a physics error.

## 8.2 Validation Speed

Each conservation check is a sum or comparison of pre-computed integral values. No re-computation is needed. The total validation cost is approximately 0.1-0.5 milliseconds per frame — evaluating a few thousand sums and comparisons on pre-stored values.

When a violation is detected, the correction is applied as a per-cell adjustment: scale the light value to satisfy energy conservation, adjust the velocity to satisfy momentum conservation, or clamp the density to satisfy mass conservation. These corrections are small (typically less than 5% adjustment) and visually imperceptible.

---

# Chapter 9: Complete System Specifications

## 9.1 The Full Pipeline

Stage 1: Perception. Depth Pro + DUSt3R + SAM 3. Input: photograph. Output: labeled 3D point cloud. Time: 0.5 seconds.

Stage 2: Cell Grid Construction. Point cloud to differential-integral cells. Compute values, gradients, second derivatives, integrals, neighbors. Time: 0.5-1 second. Navigation can begin.

Stage 3: Room/Environment Extension. Extend cell grid for unseen areas. Geometric extension for structure, AI generation for detail. Time: 10-20 seconds (background).

Stage 4: AI Texturing. Compute light values and gradients for all cells. Time: 2-5 seconds for initial pass (background).

Stage 5: Auto-Furnishing. VLM identifies gaps, retrieves furniture as cell clusters, validates placement on cell grid, inserts into grid. Time: 15-30 seconds (background).

Stage 6: Character Creation. Generate character cell clusters, assign motion deformation. Time: 5-15 seconds per character.

Stage 7: Navigation. Cell-based collision, floor following, resolution scaling. Frame rate: 60-120 FPS.

Stage 8: Rendering. Cell rendering at adaptive resolution, AI texturing enhancement, math validation. Three tiers: real-time (30-60 FPS), production (2-10 sec/frame), high-detail (30-60 sec/frame).

## 9.2 Time to Navigate

Time to first frame: approximately 1.5 seconds (perception + cell grid construction).
Time to full quality navigation: approximately 30-45 seconds (all background processes complete).
Frame rate during navigation: 60-120 FPS (depending on cell count and GPU).

## 9.3 Memory Requirements

Real-time navigation: 15-60 MB GPU memory for visible cells (adaptive resolution, streaming).
Production render: 1-4 GB for visible cells at Tier 3 resolution.
Cinematic render: 4-15 GB for visible cells at Tier 4 resolution (streamed).

Minimum GPU: RTX 3060 (12 GB) for real-time navigation.
Recommended GPU: RTX 4090 (24 GB) for production rendering.
Cinematic GPU: A100 (80 GB) for maximum resolution.

## 9.4 What Was Unified

The differential-integral cell architecture replaces ten previously separate systems with one universal primitive. Surfels for rendering, octrees for spatial indexing, density fields for collision, LOD hierarchies for performance, streaming indices for memory management, normal maps for bump detail, displacement maps for geometric detail, material maps for surface properties, shadow maps for lighting, and animation deformation fields for character motion — all are now properties, derivatives, and integrals of cells.

The result is a system that is simpler to implement (one data structure instead of ten), faster to run (cell lookups instead of geometric computations), more precise (gradient-based sub-cell accuracy), physically validated (integral-based conservation checks), and continuously scalable from real-time exploration to cinematic final output.
