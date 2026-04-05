# Centimeter-Precision 3D from Single and Multi-Frame Input
## Deep Theory of Perception Models, Depth Estimation, and Motion Capture for the Cell Architecture

---

# Part I: The Physics of Image-Based 3D

---

## Chapter 1: What a Single Image Contains and What It Loses

### 1.1 The Information Content of a Photograph

A photograph is a 2D projection of a 3D world, and this projection destroys exactly one dimension of information: depth. Every other spatial property — horizontal position, vertical position, color, texture, edges, shadows — is preserved (up to sensor resolution). The fundamental question of single-image 3D reconstruction is: can the lost depth dimension be recovered from the remaining information?

The answer is nuanced. Depth cannot be recovered exactly from a single image — this is a mathematical impossibility because infinitely many 3D scenes can produce the same photograph. However, depth can be estimated with high confidence by exploiting the statistical regularities of natural scenes. These regularities are the learned priors that modern AI models encode.

A single photograph of a typical scene contains an enormous amount of implicit 3D information, far more than is immediately obvious. There are at least seven distinct categories of depth cues available in a single image.

Perspective convergence is the most powerful cue: parallel lines in the real world (like the edges of walls, the lines of a tiled floor, the rails of a bookshelf) converge toward vanishing points in the image. The positions of these vanishing points encode the camera's orientation relative to the scene, and the rate of convergence encodes the depth gradient — how quickly depth changes across the image.

Texture gradient is the change in texture density with distance. A wooden floor's planks appear wider and more detailed near the camera and narrower and less detailed farther away. The rate of this change is directly related to the surface's orientation and distance. A trained neural network can learn to read these texture gradients with surprising precision.

Occlusion occurs when one object hides part of another, providing unambiguous depth ordering: the occluding object is closer. While occlusion alone does not give metric distance, it constrains the depth map by requiring that the occluding object's depth is less than the occluded object's depth at every overlapping pixel.

Relative size provides depth cues when the model recognizes objects of known typical size. If the model knows that a standard door is approximately 200 centimeters tall and 80 centimeters wide, the apparent size of a door in the image directly constrains its distance from the camera. This is how Apple's Depth Pro achieves metric depth estimation without camera intrinsics — it has learned the typical sizes of thousands of object categories.

Shading and shadows encode surface orientation through the way light interacts with surfaces. A surface perpendicular to the light source appears brightest; a surface at a steep angle to the light appears darker. The gradient of brightness across a curved surface encodes the surface's curvature and thus its 3D shape. Cast shadows additionally constrain the relative positions of objects and light sources.

Atmospheric perspective is the tendency for distant objects to appear hazier, bluer, and lower in contrast due to light scattering in the atmosphere. While less relevant for indoor scenes, it is a strong depth cue for outdoor photographs and large spaces.

Defocus blur causes objects at different distances from the camera's focal plane to appear blurred by different amounts. This blur pattern encodes depth, though it is only useful when the camera's aperture is large enough to produce noticeable depth of field.

### 1.2 The Precision Limits of Each Cue

Not all depth cues provide equal precision. Understanding the precision of each cue tells us the theoretical maximum accuracy for single-image depth estimation.

Perspective convergence provides the highest precision for planar surfaces. On a large, well-textured floor with clear perspective lines, the depth of any point on the floor can be estimated to within 1-2% of its true value, corresponding to 2-4 centimeters at a distance of 2 meters. The precision degrades for surfaces without clear perspective structure.

Relative size provides precision proportional to the accuracy of the size prior. If the model knows a door is 200 centimeters tall with an uncertainty of plus or minus 5 centimeters, and the door appears to be 500 pixels tall in the image, the distance can be estimated to within about 2.5% of its true value. For common indoor objects with well-known sizes, this corresponds to 3-5 centimeters of depth uncertainty at typical distances.

Texture gradient provides intermediate precision. On a well-textured surface, the depth gradient can be estimated to within 3-5% accuracy. On untextured surfaces, the texture gradient provides no information at all.

Shading provides low precision for depth (typically 5-15% error) because it depends on knowing the lighting conditions, which are usually unknown. However, it provides high precision for surface normals, which are useful for detecting support surfaces and wall orientations.

The theoretical combined precision of all monocular depth cues, exploited optimally by a neural network, is approximately 2-5% relative depth error for well-constrained scenes and 5-15% for poorly constrained scenes. For a typical living room photograph taken at a distance of 3 meters, this corresponds to 6-15 centimeters of absolute depth error — firmly in the centimeter range but not at the millimeter level.

### 1.3 How Modern AI Models Exceed Classical Limits

Classical computer vision estimated depth from explicit geometric reasoning: detecting vanishing points, measuring perspective convergence, and triangulating known object sizes. These methods were precise when the cues were clear but failed completely in their absence.

Modern deep learning models achieve better precision by combining all depth cues simultaneously through learned feature representations. A neural network does not explicitly detect vanishing points or measure texture gradients — it learns to extract features that implicitly encode all available depth information and combine them in a way that is optimal for the training data distribution.

The key insight is that the combination of multiple weak cues can produce a strong estimate. Even in a scene where no single cue provides centimeter-level precision, the combination of perspective, relative size, texture gradient, and shading can produce a combined estimate with lower error than any individual cue — if the errors are independent and the combination is optimal. A neural network trained on millions of images learns to perform this optimal combination implicitly.

Additionally, neural networks learn scene-level priors that go beyond individual cues. They learn that rooms have flat floors, vertical walls, and horizontal ceilings. They learn that furniture sits on floors. They learn typical room proportions and furniture arrangements. These priors act as additional constraints that improve depth estimation even in the absence of strong geometric cues.

---

## Chapter 2: Apple's Depth Pro — The Current Single-Image Champion

### 2.1 What Makes Depth Pro Different

Depth Pro, released by Apple Research, represents the current state of the art in monocular metric depth estimation. Three properties make it uniquely suited for our system.

First, it produces metric depth — absolute distances in meters, not just relative ordering. Most depth estimation models produce relative depth: they correctly rank pixels from near to far but do not assign real-world distances. Converting relative depth to metric depth requires knowing the camera's focal length, which is often unavailable. Depth Pro solves this by simultaneously estimating the camera's focal length from the image content, using learned priors about perspective and object sizes.

Second, it produces sharp boundary depth — the depth transitions at object edges are crisp rather than blurred. Most depth models produce smooth depth maps where the transition from one object to the next is gradual. Depth Pro achieves sharp boundaries by using a multi-scale architecture that processes the image at multiple resolutions and fuses the results, with the high-resolution path preserving edge detail.

Third, it is fast: 0.3 seconds per 2.25-megapixel image on a standard GPU. This makes it practical for interactive applications where the user expects near-instant feedback.

### 2.2 Depth Pro Architecture Theory

Depth Pro uses a multi-scale Vision Transformer (ViT) architecture. Understanding the architecture explains why it achieves both high metric accuracy and sharp boundaries — two properties that are normally in tension.

The encoder is a ViT backbone that processes the image at multiple scales. The image is input at its full resolution and at progressively down-sampled resolutions (half, quarter, eighth). Each scale is processed by the same shared ViT encoder, producing feature maps at each scale. The multi-scale processing is critical because depth estimation requires both local detail (for sharp boundaries at high resolution) and global context (for correct metric scale at low resolution).

The decoder fuses the multi-scale features using a feature pyramid network (FPN) design. The lowest-resolution features (which capture global context like room layout and perspective structure) provide the coarse depth estimate. Higher-resolution features refine this estimate, adding detail and sharpening boundaries. The fusion is performed through a series of upsampling and concatenation operations, with learned weights that determine how much each scale contributes at each spatial location.

The focal length estimation head is a separate branch that takes the global features and predicts the camera's focal length. This prediction is based on the perspective structure of the image — the convergence of parallel lines, the apparent sizes of recognized objects, and the overall field of view. The estimated focal length is used to convert the raw depth predictions into metric depths in meters.

The training protocol is two-stage. First, the model is pre-trained on a mix of synthetic datasets (with perfect ground-truth depth) and real datasets with pseudo-ground-truth depth (estimated from stereo cameras or LiDAR). Second, the model is fine-tuned with a boundary-aware loss that specifically penalizes blurred depth transitions at object edges.

### 2.3 Precision Analysis of Depth Pro

Depth Pro achieves approximately 5-8% absolute relative error on standard indoor benchmarks (NYU Depth V2). This means that for a surface at 3 meters distance, the depth estimate is accurate to within 15-24 centimeters on average.

This average masks significant variation across surface types. On favorable surfaces (textured floors with strong perspective cues, walls with clear vanishing points, objects of known size providing scale anchors), the error drops to approximately 3-5%, corresponding to 9-15 centimeters at 3 meters. On unfavorable surfaces (featureless white walls, reflective surfaces, transparent glass, textureless ceilings), the error can exceed 8-15%. On failure cases (mirrors reflecting phantom rooms, glass that is invisible to depth, extremely dark regions), the error can be catastrophic — 15-50% or complete depth inversion.

It is important to note that NYU Depth V2 is itself a well-constrained indoor benchmark. The 5-8% average is already on favorable data. Achieving 3-5% on the best surfaces within this already favorable distribution requires strong multi-cue convergence — multiple depth cues reinforcing each other — which does not always occur. A featureless white wall next to a textured floor can produce conflicting cues that prevent convergence.

Boundary precision requires careful distinction between two different measurements. Boundary localization precision refers to the ability to identify WHERE object edges are in the 2D image — Depth Pro achieves F1 scores above 0.9, correctly localizing 90% of depth discontinuities to within 1-2 pixels, corresponding to 2-5 millimeters in image space. Boundary depth precision refers to the accuracy of the depth VALUES at those edges — this is significantly worse because depth values at object boundaries suffer from mixed-pixel contamination (the depth is a blend of foreground and background). Typical boundary depth error is 5-15 millimeters, not 2-5 millimeters.

This distinction matters for the cell architecture. The cell density gradient at an object boundary depends on the depth being correct ON BOTH SIDES of the boundary, not just knowing where the boundary is. Mixed-pixel depth at edges produces cells with incorrect density values — somewhere between the foreground and background depths. The density gradient direction is usually correct (pointing from solid toward empty), but its magnitude is unreliable, making sub-cell surface localization at boundaries approximately 5-15 millimeters precise rather than the 2-5 millimeters that the localization F1 score alone would suggest.

The focal length estimation accuracy is within 0.5-2% of the true focal length for images taken with standard smartphone and DSLR cameras. This corresponds to a 0.5-2% multiplicative error in all metric depth values — a systematic scale error that can be calibrated out if the scene contains any object of known size.

### 2.4 Depth Noise and Its Impact on Cell Architecture

The per-pixel depth noise in Depth Pro's output (random errors of 3-5% after scale calibration on favorable surfaces) has a critical impact on the differential-integral cell architecture because cells rely on spatial derivatives for their core operations.

When the raw depth map is backprojected to a 3D point cloud and converted to cells, the cell properties include gradients computed by finite differences between adjacent cells. Finite differences amplify noise: if adjacent cells have depth values of 2.97m and 3.05m due to random noise (true depth is 3.00m for both), the computed gradient is large and points in a random direction — when the true gradient should be near zero (both cells are on the same flat surface).

This noise-amplified gradient corrupts the Taylor expansion used for cell subdivision. When a cell subdivides, its children's values are predicted from the parent's gradient. A noisy gradient produces children with incorrect values — wrong colors, wrong normals, wrong density boundaries. The subdivision reveals noise-generated detail rather than real detail.

The solution is a bilateral filter applied to the depth map before backprojection. The bilateral filter smooths depth values within a spatial neighborhood but only across pixels with similar RGB colors. This preserves sharp depth transitions at object boundaries (where colors change abruptly) while smoothing random noise within uniform surfaces (where colors are similar). The filter adds approximately 5 milliseconds and reduces the standard deviation of per-pixel depth noise by 40-65%.

After bilateral smoothing, the gradients computed by finite differences represent true spatial variation (texture grain direction, surface curvature, boundary position) rather than amplified sensor noise. This preprocessing step is not optional for cell-based representations — it is essential for meaningful gradient data.

### 2.5 Per-Cell Confidence

Each cell should carry a confidence value that indicates how reliable its data is. This confidence is derived from multiple signals.

Depth smoothness: cells constructed from regions where the raw depth varied smoothly (low local variance before bilateral filtering) receive high confidence. Cells from regions with noisy or inconsistent depth receive low confidence.

Point density: cells that received many source points during point-to-cell conversion have more statistical support and receive higher confidence. Cells with few points (at the edges of the scene or in poorly observed areas) receive lower confidence.

Semantic penalty: cells with semantic labels that are known to confuse depth estimation — glass, mirror, transparent, highly reflective — receive reduced confidence regardless of depth smoothness, because the depth model's output for these materials is structurally unreliable (the model may confidently predict a wrong depth for glass).

DUSt3R agreement: when DUSt3R is available (multi-image input), cells where Depth Pro and DUSt3R agree on depth receive high confidence. Cells where the two models disagree receive low confidence — the disagreement itself is a strong uncertainty signal.

The confidence value is stored in the CellGeo struct (using existing padding bytes — no struct size change required) and used throughout the pipeline. Low-confidence cells receive expanded collision margins (the system assumes uncertain objects may be larger than they appear). Low-confidence cells skip Taylor expansion subdivision (using neural texture features or AI texturing instead, since gradient-based prediction is unreliable). Low-confidence cells receive more weight from the AI texturing model and less from gradient-based shading. In debug mode, low-confidence cells can be rendered with a visual indicator (orange tint or wireframe overlay) so developers can see where the reconstruction is uncertain.

---

## Chapter 3: DUSt3R — 3D from Uncalibrated Images

### 3.1 The Paradigm Shift: Direct Point Map Regression

DUSt3R (Dense and Unconstrained Stereo 3D Reconstruction), developed by Naver Labs, introduced a radical departure from traditional 3D reconstruction. Instead of following the classical pipeline (detect features, match features across images, triangulate 3D points, optimize camera poses), DUSt3R treats the entire problem as a single regression task: given two images, directly predict the 3D position of every pixel.

The output of DUSt3R is called a point map: for each pixel in the image, the network predicts a 3D coordinate (X, Y, Z) in a canonical reference frame. The point map is dense (every pixel gets a 3D prediction) and pixel-aligned (the 3D prediction for each pixel corresponds exactly to the scene content at that pixel).

### 3.2 The CroCo Foundation: Learning 3D from Cross-View Completion

DUSt3R is built on a pre-training strategy called CroCo (Cross-View Completion) that explains how the network learns genuine 3D understanding.

Standard masked image modeling masks random patches in a single image and trains the network to reconstruct them. This teaches 2D image statistics — textures, shapes, colors — but not 3D geometry.

CroCo changes this by masking patches in one image and asking the network to reconstruct them using information from a different view of the same scene. This is fundamentally harder because the two views have different perspectives, different occlusions, and different scales. To predict what the masked region looks like from view 1 using information from view 2, the network must implicitly understand the 3D structure of the scene — it must know where surfaces are in 3D, how they would appear from different angles, and what parts are visible from each viewpoint.

This cross-view pretraining imbues the network with genuine 3D understanding, not just 2D pattern matching. When DUSt3R is then fine-tuned for point map prediction, it has a massive head start because it already understands 3D structure.

### 3.3 The DUSt3R Architecture in Detail

DUSt3R uses an asymmetric encoder-decoder architecture based on Vision Transformers. The encoder processes both input images using a shared ViT backbone. The decoder consists of cross-attention layers where tokens from one image attend to tokens from the other image — this cross-attention is the mechanism by which the network establishes correspondences between the two views, implicitly performing feature matching without any explicit matching step.

The decoder is asymmetric: one image is designated as the reference and the other as the source. The point maps for both images are predicted in the reference image's coordinate frame. The regression heads predict, for each pixel, a 3D coordinate and a confidence value. High-confidence predictions (on well-textured, non-occluded surfaces) have sub-centimeter accuracy; low-confidence predictions (on reflective or thin structures) may have errors of several centimeters.

### 3.4 Role in the Current Pipeline

DUSt3R is designed for image pairs and provides maximum value when multiple views are available. For single-image input, DUSt3R can operate in monocular mode but with reduced accuracy and no metric scale.

In the current pipeline, DUSt3R is optional for single-image use. When skipped, Depth Pro serves as the sole depth source, with bilateral depth smoothing compensating partially for the geometric consistency that DUSt3R would have provided. The smoothed Depth Pro output produces adequate cell grids for most indoor and outdoor scenes.

When the user provides two or more photographs, DUSt3R becomes essential. It establishes cross-view correspondences and produces a geometrically consistent point map that single-image depth estimation cannot match. Multi-view input also eliminates scale ambiguity and reveals surfaces hidden from any single viewpoint.

The system automatically activates DUSt3R when multiple images are available, seamlessly improving reconstruction quality without user intervention. The single-image path remains the minimum viable input.

---

## Chapter 4: MASt3R — Precision Matching

### 4.1 What MASt3R Adds

MASt3R (Matching And Stereo 3D Reconstruction) extends DUSt3R with a dedicated matching head that achieves sub-pixel correspondence accuracy. The matching head predicts dense local feature descriptors for every pixel, trained with an InfoNCE contrastive loss that rewards correct correspondences and penalizes incorrect ones.

The InfoNCE loss drives extreme spatial discrimination because it is a classification problem, not a regression problem. A pixel matched to its neighbor (1 pixel from the true match) is penalized just as heavily as a pixel matched to a completely wrong location. This forces the network to learn descriptors that can distinguish adjacent pixels on textured surfaces, enabling sub-pixel matching accuracy — 30% better than previous best methods on challenging datasets.

### 4.2 Relevance to Our Pipeline

MASt3R is not used in the current prototype because it requires image pairs and adds complexity. However, it represents a future upgrade path for multi-image input. When multiple photographs are available, MASt3R could replace DUSt3R for even more precise cross-view reconstruction. Its sub-pixel matching would directly improve cell gradient quality by providing more accurate point positions before the point-to-cell conversion.

---

## Chapter 5: Neural Radiance Fields — Historical Context

### 5.1 NeRF's Conceptual Contribution

Neural Radiance Fields (NeRF), introduced in 2020, pioneered the concept of representing scenes as continuous neural functions — mapping 3D coordinates to color and density. NeRF achieved breakthrough novel view synthesis quality, proving that neural networks can learn to represent 3D scenes with photorealistic fidelity. However, its rendering speed (seconds per frame due to per-ray network evaluations) prevents real-time use.

### 5.2 Influence on the Cell Architecture

NeRF is not used in our pipeline, but its conceptual influence is direct. The differential-integral cell can be understood as a discretized version of NeRF's continuous function. Where NeRF queries a neural network at arbitrary (x, y, z) to get color and density, our cells store the same information at discrete grid positions. The cell gradients provide the continuity that discretization would otherwise lose — they encode how color and density change between grid points, enabling smooth interpolation and accurate subdivision.

NeRF also demonstrated that view-dependent appearance can be learned from data. Our AI texturing model serves a similar role — producing view-dependent lighting effects without explicit physical computation. The philosophical connection is the same: learn what reality looks like from data rather than simulate it from equations.

---

## Chapter 6: From Gaussians to Cells — The Representation Evolution

### 6.1 3D Gaussian Splatting

3D Gaussian Splatting (3DGS), introduced in 2023, replaced NeRF's continuous neural function with explicit primitives — 3D Gaussian ellipsoids with position, shape, color, and opacity. These Gaussians are projected and blended onto the image plane, achieving real-time rendering at 100+ frames per second. For 3D understanding, each Gaussian's position encodes where a surface element exists in space.

### 6.2 2D Gaussian Surfels

The first evolution of 3DGS recognized that indoor scenes are dominated by flat surfaces. 2D Gaussian Surfels replaced ellipsoids with flat oriented discs — approximately 10x fewer primitives for surface-dominated scenes with sharper, more accurate geometry. Neural Shell Texture further separated geometry from appearance, enabling arbitrarily detailed textures without increasing primitive count.

### 6.3 Differential-Integral Cells

The cell architecture extends this evolution. Instead of individual primitives, the scene is divided into volumetric cells storing values, spatial derivatives, and integrals. The key advantage for perception: when the point cloud from Depth Pro is converted to cells, the depth precision is enriched with gradient information that encodes sub-cell detail. A 5-centimeter cell with a density gradient can locate the surface boundary within the cell to approximately 5-15 millimeter precision (limited by boundary depth accuracy, not boundary localization accuracy). The gradient is the extra information extracted from the depth map's spatial variation.

The precision profile of cells matches the underlying depth estimation. Nearby surfaces (within 1-2 meters): cells positioned with 1-3 centimeter accuracy, gradients clean and informative, Taylor expansion produces high-quality children. Mid-range surfaces (2-4 meters): cells positioned with 3-8 centimeter accuracy, gradients moderately noisy, Taylor expansion adequate but may need neural texture correction. Far surfaces (beyond 4 meters): cells positioned with 8-20 centimeter accuracy, gradients noisy, these cells rely more on AI texturing than gradient-based detail.

---

## Chapter 7: Failure Modes

### 7.1 Catastrophic Depth Failures

Certain scene elements produce structurally incorrect depth estimates — not just noisy but fundamentally wrong. These failures create cells with geometry that does not correspond to reality.

Mirrors: Depth Pro sees a room inside the mirror — the depth extends behind the wall into a phantom space. The cell grid creates a fake room behind the mirror surface. Cells in this phantom room have full density and plausible normals but represent geometry that does not exist. Detection: SAM 3 labels mirror surfaces. Mitigation: mark mirror cells as high-density solid (preventing navigation into the phantom room) with low confidence. Do not extend or subdivide mirror cells.

Glass surfaces: Depth Pro sees through glass to whatever is behind it. A glass coffee table becomes invisible — its cells have the depth of the floor beneath. The glass surface is completely absent from the cell grid, creating a collision-detection hole. Detection: SAM 3 labels glass surfaces. Mitigation: force glass cells to have density 1.0 regardless of depth estimate. Assign low confidence. The glass surface exists in the cell grid as a solid barrier even though the depth model missed it.

Transparent objects (vases, bottles, clear plastic): similar to glass but with more complex geometry. The depth behind the object is slightly distorted by refraction. Cells in and around transparent objects have unreliable geometry. Detection: SAM 3 labels transparent categories. Mitigation: assign low confidence, expand collision margins.

Very dark scenes: Depth Pro produces noisy, unreliable depth across the entire image because the sensor captures very little light. The cell grid is uniformly noisy — gradients are meaningless everywhere, not just in specific regions. Detection: mean image brightness below a threshold (approximately 30/255 for 8-bit images). Mitigation: warn user, request better lighting. If proceeding, set all cells to low confidence and rely entirely on AI texturing for visual quality rather than gradient-based detail.

Outdoor sky: Depth Pro assigns very large depth values to sky pixels (often 50-1000 meters). Sky cells fill the distant portion of the cell grid with nonsense geometry at extreme depth. Detection: SAM 3 labels sky. Mitigation: exclude sky pixels from cell grid construction entirely. Handle sky as a skybox — a separate rendering layer at infinite distance, not part of the cell grid.

Highly specular surfaces (polished metal, wet floors): depth estimates oscillate due to reflected highlights being confused with surface geometry. Cells on specular surfaces have noisy depth with periodic artifacts. Detection: high local depth variance combined with high pixel brightness. Mitigation: assign moderate-low confidence, smooth depth more aggressively on specular surfaces before cell construction.

### 7.2 Impact on the Prototype Success Criteria

The "15 out of 20 test photos produce acceptable results" success criterion will fail for predictable reasons. The 5 failures will likely be: a room with a large mirror (phantom room), a room with a glass table (invisible furniture), a very dark room (noisy everywhere), an outdoor scene with sky (extreme depth artifacts), and a room with a large window showing a bright exterior (mixed indoor/outdoor depth scales).

Each failure has a specific mitigation (described above). Implementing these mitigations in the prototype — primarily the SAM-3-label-based confidence reduction and forced density for glass/mirror — should raise the success rate from 15/20 to 18-19/20.

---

## Chapter 8: Multi-Frame Input and Motion Capture

### 8.1 The Input Spectrum

The system accepts any number of images, from one to thousands.

One photograph: the minimum viable input. Depth Pro provides metric depth. Bilateral smoothing cleans noise. The cell grid captures approximately 40-60% of the scene with 5-10 centimeter precision (after calibration). The remaining 40-60% is generated by the system.

Two to five photographs from different viewpoints: DUSt3R activates automatically. Cross-view correspondences resolve depth ambiguity. Scale is determined from the baseline between views. The cell grid captures 70-90% of the scene with 2-5 centimeter precision. Far less generation is needed.

Video or dense image sequence: multiple frames provide multiple views AND temporal information. Static elements are reconstructed with sub-centimeter precision from dozens of observations. Moving elements are detected and their motion is captured.

Each additional image improves both coverage and precision. The cell grid construction seamlessly handles any input count — more images produce more points, which produce denser cells with cleaner gradients.

### 8.2 Multi-Frame Reconstruction

When multiple images are available, the reconstruction pipeline extends. Depth Pro runs on each image for metric scale. DUSt3R runs on image pairs for geometric consistency. All point maps are aligned to a common coordinate frame. Merged points produce denser cells with cleaner gradients.

The precision improvement from multi-image input depends critically on the baseline — the physical distance between camera positions. Two photos taken from nearly the same position (baseline less than 10 centimeters) give almost no improvement over one photo because the parallax is too small for triangulation. The baseline should be at least 10-20% of the scene depth for meaningful improvement. For a room at 3 meters depth, the cameras should be separated by at least 30-60 centimeters. Two photos from opposite corners of a room (baseline of 3-5 meters) provide excellent triangulation with sub-centimeter depth precision.

### 8.3 Motion Detection from Multiple Timeframes

When the input includes images of the same scene at different moments, the system detects motion by comparing the reconstructions. This capability requires video or burst photos with known timestamps — not arbitrary separate photos taken at unknown times.

The temporal alignment requirement: computing velocity requires knowing the time delta between frames. Video provides this natively (known FPS, typically 24-60 frames per second). Burst photos provide timestamps in EXIF metadata (accurate to milliseconds on modern smartphones). Separate photos taken at different times have timestamps accurate only to approximately 1 second, which is too coarse for velocity computation unless the photos are minutes or hours apart (capturing large-scale changes like furniture rearrangement rather than human motion).

The detection algorithm: build a cell grid from each timeframe. Align all timeframe grids to a common coordinate system using the static structure (walls, floor, fixed furniture) as registration anchors. The alignment must be sub-centimeter — if the alignment error is 2-3 centimeters, static cells will be falsely flagged as dynamic. Iterative closest point (ICP) or similar registration on the static cells achieves sub-centimeter alignment because the static structure provides hundreds of thousands of redundant correspondence points.

For each cell position, compute the difference in density and albedo across timeframes. Cells where the difference exceeds a threshold are flagged as dynamic. Cells where the difference is near zero are confirmed as static. Adjacent dynamic cells with similar motion patterns are clustered into moving objects (a person, a car, a pet).

### 8.4 Real Motion Capture into Cell Animation Data

For each detected moving object, the system extracts actual motion data — real velocity and acceleration observed from the images, not generated by an AI motion model.

The motion extraction: at each timeframe T, the dynamic object's centroid position is known from the cell grid. The trajectory over time is the sequence of centroid positions. Velocity and acceleration are computed by finite differences on this trajectory.

The shape deformation — how the person's body changes shape as they walk — is substantially harder than centroid tracking. Tracking per-cell motion within a deforming body (arm cells swinging, leg cells alternating) is essentially non-rigid registration — matching a set of surface points across frames when the surface is changing shape. This requires dense correspondences on deforming surfaces, which is an active research problem.

Methods like DynamicFusion, non-rigid ICP, or learned non-rigid flow can perform this matching, but they require dense frame sequences (video at 24+ FPS) rather than sparse photos. The quality of per-cell deformation capture depends on the input frame rate: at 30 FPS, arm swing and leg movement are captured smoothly. At 5 FPS (sparse photos), only gross body translation is captured — the detailed deformation is lost.

For the prototype, centroid-level motion capture (tracking where objects move over time) is achievable and valuable. Per-cell deformation capture (tracking how bodies change shape) requires a dedicated non-rigid registration model and is better treated as a Phase 2 capability to be added after the core cell architecture is validated.

### 8.5 Combining Real and Generated Motion

Real captured motion and AI-generated motion coexist in the same cell structure. A scene can contain a real person's captured walking trajectory alongside an AI-generated character's text-to-motion animation. The cell architecture doesn't distinguish the source — both are velocity and acceleration values in the same CellVisual struct. The renderer interpolates identically for both.

The most powerful workflow combines both: reconstruct a real room from video, capture a real person's motion from that video, then re-target the captured motion into a different environment (with the trajectory adjusted to fit the new room's geometry, avoiding furniture, following the floor). The person looks real because they are real. The room looks real because it is real. Only the combination is synthetic.

---

## Chapter 9: The Fusion Architecture for Cell Construction

### 9.1 The Fusion Pipeline

The perception pipeline runs the following steps, adapted based on input count.

For single-image input: run Depth Pro for metric depth. Apply bilateral smoothing with spatial sigma 3-4 pixels (preserving surface detail) guided by the RGB image. Apply scale calibration from detected reference objects. Run SAM 3 for semantic segmentation. Backproject the smoothed, calibrated depth map to a 3D point cloud. Assign semantic labels and compute per-pixel confidence from depth smoothness, semantic penalties, and point density. Convert to cell grid.

For multi-image input: run Depth Pro on each image for metric scale. Run DUSt3R on image pairs for geometric consistency. Align all point maps with metric scale from Depth Pro. Run SAM 3 on each image for labels. Merge into a single cell grid with higher confidence from multi-view redundancy.

For temporal input: process each timeframe group through multi-image pipeline. Align timeframes using static structure. Detect dynamic cells. Extract motion data. Build final cell grid with static cells (permanent) and dynamic cells (time-varying).

### 9.2 Depth Smoothing — The Critical Preprocessing Step

The bilateral depth smoothing step is the single most impactful preprocessing step for cell quality. It operates on the depth map with the RGB image as guide.

For each pixel, the smoothed depth is a weighted average of neighboring pixels' depths, where the weight depends on both spatial distance (nearby pixels contribute more) and color similarity (pixels with similar RGB values contribute more). Pixels across an object boundary have different colors, so their depths are not averaged together — the sharp boundary is preserved. Pixels within the same surface have similar colors, so their depths are averaged, smoothing out noise.

Parameters: spatial sigma of 3-4 pixels (at 2MP resolution, this spans approximately 6-12 millimeters at 2 meters, preserving surface features like brick mortar grooves at 5-10mm while smoothing per-pixel noise). Color sigma of 20-30 intensity units on a 0-255 scale.

For gradient computation, an additional smoothing step is applied: the gradients are computed using a 5x5 Sobel kernel rather than simple 2-point finite differences. The larger kernel provides implicit smoothing during differentiation, further reducing noise in the gradient values without requiring more aggressive depth smoothing that would erase real surface detail. This separates the two concerns: depth should be as accurate as possible (mild bilateral smoothing), gradients should be as clean as possible (smooth during differentiation via larger kernel).

Processing time for bilateral filter: approximately 5 milliseconds for a 2-megapixel image. Quality impact: gradient noise reduced by 40-65%, directly improving Taylor expansion subdivision quality.

### 9.3 Precision Budget of the Fused System

The fused system's precision at each step:

Depth estimation (Depth Pro): 5-8% average relative error, 3-5% on favorable surfaces after scale calibration. At 3 meters: 9-15 centimeters favorable, 15-24 centimeters average.

Bilateral smoothing: reduces random depth noise by 40-65%. After smoothing: 5-10 centimeters random error on favorable surfaces at 3 meters.

Scale calibration: removes systematic scale error (0.5-2% multiplicative). After calibration: systematic error below 0.3%.

Boundary precision: boundary localization at 2-5 millimeters (where the edge is in 2D). Boundary depth at 5-15 millimeters (the depth value at the edge).

Cell position accuracy: 5-10 centimeters from single image (after calibration and smoothing). 2-5 centimeters from multi-image input with adequate baseline. Sub-centimeter from dense video input.

Per-cell confidence: derived from depth smoothness, point density, semantic penalties, and multi-model agreement. Ranges from 0.0 (completely unreliable) to 1.0 (highly confident). Average across a typical well-lit room: 0.7-0.85.

---

## Chapter 10: Error Propagation Through the Pipeline

### 10.1 End-to-End Error Analysis

The cell position accuracy is not simply the depth estimation error — it is the depth error transformed through each processing step, with each step either amplifying or reducing the error.

Depth Pro raw output: 5-8% relative error at 3 meters produces ±15-24 centimeters of depth uncertainty per pixel.

Focal length error: ±1% multiplicative error adds approximately ±3 centimeters at 3 meters. This is a systematic error that affects all depths proportionally.

Bilateral smoothing: reduces random noise by 40-65%, leaving ±7-14 centimeters of random depth error on average (for 5-8% input). On favorable surfaces: ±5-9 centimeters.

Scale calibration: removes the systematic focal length error and Depth Pro's systematic scale bias, leaving only random error. After calibration: ±5-9 centimeters random error on favorable surfaces.

Backprojection to 3D: transforms per-pixel depth error into 3D position error. The lateral (X, Z) error is proportional to depth error divided by focal length. At typical smartphone focal lengths (4mm, ~28mm equivalent), a depth error of ±8 centimeters at 3 meters produces lateral error of ±3-5 centimeters. Vertical (Y) error is similar. Total 3D point position error: ±5-10 centimeters.

Cell binning at 5cm resolution: adds ±2.5 centimeters of quantization error (points are assigned to the nearest cell center). Combined with the 3D position error: the cell center position error is approximately ±6-11 centimeters (root-sum-of-squares of point error and quantization error).

Gradient computation via finite differences: the gradient between two adjacent cells is (value_2 - value_1) / cell_width. If each cell's value has error ε, the gradient error is approximately 2ε / cell_width. For cell position error of ±8 centimeters and cell width of 5 centimeters: gradient error is approximately ±3.2 (in units of cm/cm, or dimensionless for normalized values). This is large — the gradient error can exceed the gradient itself.

However, bilateral smoothing and the 5x5 Sobel kernel reduce the effective gradient error to approximately ±0.8-1.5 — still significant but not dominant.

Taylor expansion child prediction: the child value is predicted as parent_value + gradient × offset. The child offset is ±1.25 centimeters (half the child cell width for a 2.5cm child of a 5cm parent). The child prediction error is approximately gradient_error × offset = 1.0 × 1.25 = ±1.25 centimeters for smoothed gradients.

### 10.2 What the Error Analysis Tells Us

The Taylor expansion child prediction error of approximately ±1.25 centimeters (for smoothed, single-image input) is SMALLER than the parent cell resolution of 5 centimeters. This means subdivision does provide genuine improvement — the child's predicted value is more precise than simply assigning the parent's average value to all children.

However, the improvement is modest. The child prediction error (±1.25 cm) is comparable to the child cell size (2.5 cm). This means one level of subdivision is useful (children at 2.5 cm have meaningful gradient-predicted detail) but a second level of subdivision (grandchildren at 1.25 cm, with prediction error of approximately ±0.6 cm, also comparable to grandchild cell size) provides diminishing returns for single-image input.

For multi-image input (depth error reduced to ±3-4 cm), the gradient error drops proportionally, and multiple levels of subdivision become reliable — enabling Tier 3 (1 cm) and even Tier 4 (0.5 cm) cell resolution with meaningful Taylor-predicted detail.

The implication: for single-image input, Taylor expansion provides one reliable level of subdivision (5cm → 2.5cm). Finer detail must come from the neural texture features and AI texturing model, not from gradient prediction. For multi-image input, Taylor expansion provides two to three reliable levels (5cm → 2.5cm → 1.25cm → 0.6cm), enabling production-quality detail from gradient prediction alone.

---

## Chapter 11: The Future and Precision Ceiling

### 11.1 Convergence Toward Foundation Models for 3D

The trend in the field is unmistakable: separate models for depth estimation, 3D reconstruction, segmentation, and scene understanding are converging toward unified foundation models that perform all tasks simultaneously.

VGGT (Visual Geometry Grounded Transformer), released by Meta, is an early example: it takes multiple images as input and directly outputs camera poses, depth maps, point maps, and 3D point clouds in a single forward pass.

The trajectory suggests that within 1-2 years, a single foundation model will take photographs as input and output metric depth, semantic segmentation, 3D reconstruction, and scene graph in a single forward pass taking less than one second. When this happens, the cell grid construction will simplify (one model replaces three) and the precision will likely improve (joint optimization eliminates accumulated errors from stacking independent models).

The cell architecture is designed to be model-agnostic. When better perception models arrive, the cell grid construction simply receives more accurate point clouds. The cell structure, gradients, integrals, subdivision, collision, rendering, and animation are all unchanged. Only the quality of the input data improves.

### 11.2 The Monocular Precision Ceiling

There is a theoretical ceiling on the precision achievable from a single monocular image, imposed by the information content of the image itself. The practical ceiling, based on current and foreseeable AI capabilities, is approximately 1-2% relative depth error for well-textured scenes with known-size objects for calibration. At 3 meters distance, this corresponds to 3-6 centimeters — firmly centimeter-level but unlikely to reach millimeter-level from a single image alone.

Three developments could push precision beyond this ceiling: higher-resolution imaging (4K and 8K input), stronger learned priors (training on larger and more diverse 3D datasets), and physical reasoning (models that understand gravity and structural stability to constrain depth predictions).

Multi-image input breaks through the ceiling entirely. With sufficient views, photogrammetric precision (sub-millimeter) is achievable. The cell architecture supports this seamlessly — more images produce better cells without any architectural change.

---

## Chapter 12: Summary — What You Can Achieve Today

### 12.1 The Precision Reality

Using the best available models combined with bilateral depth smoothing and scale calibration, the achievable precision for cell construction:

Cell position accuracy (single image, favorable): 5-10 centimeters. This is the honest number after tracing error through the full pipeline — Depth Pro's 3-5% favorable error, through smoothing, calibration, backprojection, and cell binning.

Cell position accuracy (single image, unfavorable): 10-20 centimeters. Featureless walls, poor lighting, no scale reference.

Cell position accuracy (multi-image, adequate baseline): 2-5 centimeters. DUSt3R's cross-view consistency plus multi-view redundancy.

Cell position accuracy (video input): sub-2 centimeters for well-observed surfaces.

Boundary localization: 2-5 millimeters (where edges are in 2D).

Boundary depth accuracy: 5-15 millimeters (depth values at edges — worse than localization due to mixed pixels).

Sub-cell precision from gradients: approximately ±1.25 centimeters for one level of Taylor expansion from single image. Improves proportionally with multi-image input.

Taylor expansion reliability: one subdivision level reliable from single image. Two to three levels reliable from multi-image input. Beyond that, neural texture features and AI texturing provide the detail.

Per-cell confidence: 0.7-0.85 average for a well-lit room. Below 0.5 for glass, mirrors, dark regions. Above 0.9 for textured surfaces with scale reference.

### 12.2 The Eleven Factors That Most Affect Precision

Ranked from most to least impactful:

1. Number of input images. Multi-image input with adequate baseline improves precision by 2-5x over single image.

2. Presence of a known-size object for scale calibration. Eliminates systematic scale error, improving precision by 50-70% for single-image input.

3. Bilateral depth smoothing with appropriate sigma. Reduces gradient noise by 40-65%, essential for cell subdivision quality. Use spatial sigma 3-4 pixels (not larger, to preserve surface detail). Complement with 5x5 Sobel kernel during gradient computation.

4. Surface texture quality. Well-textured surfaces provide richer depth cues than featureless surfaces.

5. Camera distance to the area of interest. Closer means more pixels, which means more precision.

6. Image resolution. Higher resolution provides more detail. 12 megapixels gives approximately 40% better precision than 3 megapixels.

7. Lighting quality. Even, diffuse lighting produces better depth estimates than harsh directional lighting or dim lighting.

8. Visible scene structure. Seeing floor-wall junctions, ceiling-wall junctions, and multiple walls provides strong perspective constraints.

9. Absence of failure-mode surfaces. Scenes without large mirrors, glass tables, or highly specular surfaces produce more reliable cell grids.

10. Image sharpness. Motion blur, focus blur, and compression artifacts degrade depth precision.

11. Camera angle. A photograph from a corner showing multiple surfaces provides stronger perspective cues than one taken straight at a wall.

### 12.3 The Honest Assessment

Single-image centimeter-precision 3D reconstruction is real and achievable, but the honest precision is 5-10 centimeters under favorable conditions — not the 3-7 centimeters originally claimed, which was measured before accounting for error propagation through the full cell construction pipeline.

The system is sufficient for interior design visualization, spatial planning, game level design, film pre-visualization, and navigation. It is not sufficient for precision manufacturing, surgical robotics, or structural engineering.

Multi-image and video input breaks through the single-image ceiling, achieving 2-5 centimeter precision with adequate baseline and adding the capability of motion capture — observing how things actually move. The cell architecture handles all input types with the same data structure, seamlessly improving quality as more data becomes available.

The Taylor expansion subdivision — the cell architecture's core mechanism for revealing detail at higher resolution — provides one reliable level of subdivision from single-image input and two to three levels from multi-image input. Beyond that, neural texture features and AI texturing provide the visual detail. This is an honest characterization of what the math can and cannot do.
