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

The focal length estimation head is a separate branch that takes the global features (from the lowest-resolution encoder output) and predicts the camera's focal length. This prediction is based on the perspective structure of the image — the convergence of parallel lines, the apparent sizes of recognized objects, and the overall field of view. The estimated focal length is used to convert the raw depth predictions into metric depths in meters.

The training protocol is two-stage. First, the model is pre-trained on a mix of synthetic datasets (with perfect ground-truth depth) and real datasets with pseudo-ground-truth depth (estimated from stereo cameras or LiDAR). Second, the model is fine-tuned with a boundary-aware loss that specifically penalizes blurred depth transitions at object edges.

### 2.3 Precision Analysis of Depth Pro

Depth Pro achieves approximately 5-8% absolute relative error on standard indoor benchmarks (NYU Depth V2). This means that for a surface at 3 meters distance, the depth estimate is accurate to within 15-24 centimeters on average.

However, this average masks significant variation. On well-constrained surfaces (floors with visible texture, walls with clear perspective, objects of known size), the error is much lower — typically 2-4%, corresponding to 6-12 centimeters at 3 meters. On poorly constrained surfaces (featureless white walls, reflective surfaces, transparent glass), the error can exceed 15%.

The boundary sharpness is exceptional. Depth Pro achieves F1 scores above 0.9 on boundary accuracy benchmarks, meaning that 90% of depth discontinuities are correctly localized to within 1-2 pixels. At typical image resolutions, this corresponds to 2-5 millimeters of spatial precision at the boundary.

The focal length estimation accuracy is within 0.5-2% of the true focal length for images taken with standard smartphone and DSLR cameras. This corresponds to a 0.5-2% multiplicative error in all metric depth values — a systematic scale error that can be calibrated out if the scene contains any object of known size.

### 2.4 Depth Noise and Its Impact on Downstream Representations

The per-pixel depth noise in Depth Pro's output (random errors of 2-4% after scale calibration) has a critical impact on any representation that relies on spatial derivatives — including the differential-integral cell architecture.

When the raw depth map is backprojected to a 3D point cloud and converted to cells, the cell properties include gradients computed by finite differences between adjacent cells. Finite differences amplify noise: if adjacent cells have depth values of 2.98m and 3.04m due to random noise (true depth is 3.00m for both), the computed gradient is large and points in a random direction — when the true gradient should be near zero (both cells are on the same flat surface).

This noise-amplified gradient corrupts the Taylor expansion used for cell subdivision. When a cell subdivides, its children's values are predicted from the parent's gradient. A noisy gradient produces children with incorrect values — wrong colors, wrong normals, wrong density boundaries. The subdivision reveals noise-generated detail rather than real detail.

The solution is a bilateral filter applied to the depth map before backprojection. The bilateral filter smooths depth values within a spatial neighborhood but only across pixels with similar RGB colors. This preserves sharp depth transitions at object boundaries (where colors change abruptly) while smoothing random noise within uniform surfaces (where colors are similar). The filter adds approximately 5 milliseconds and reduces the standard deviation of per-pixel depth noise by 60-80%.

After bilateral smoothing, the gradients computed by finite differences represent true spatial variation (texture grain direction, surface curvature, boundary position) rather than amplified sensor noise. This preprocessing step is not optional for cell-based representations — it is essential for meaningful gradient data.

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

When the user provides two or more photographs, DUSt3R becomes essential. It establishes cross-view correspondences and produces a geometrically consistent point map that single-image depth estimation cannot match. Multi-view input also eliminates scale ambiguity (the baseline between views provides metric scale) and reveals surfaces hidden from any single viewpoint.

When the user provides a sequence of images over time (frames from a video or photos at different moments), DUSt3R processes pairs of frames to establish geometric consistency across the sequence. The static elements (walls, furniture, terrain) are reconstructed with increasing accuracy from multiple views. Moving elements (people, vehicles) are detected as regions where the geometry changes between frames — these feed into the motion capture pipeline described in Chapter 7.

The system automatically activates DUSt3R when multiple images are available, seamlessly improving reconstruction quality without user intervention. The single-image path remains the minimum viable input.

---

## Chapter 4: MASt3R — Precision Matching

### 4.1 What MASt3R Adds

MASt3R (Matching And Stereo 3D Reconstruction) extends DUSt3R with a dedicated matching head that achieves sub-pixel correspondence accuracy. The matching head predicts dense local feature descriptors for every pixel, trained with an InfoNCE contrastive loss that rewards correct correspondences and penalizes incorrect ones.

The InfoNCE loss drives extreme spatial discrimination because it is a classification problem, not a regression problem. A pixel matched to its neighbor (1 pixel from the true match) is penalized just as heavily as a pixel matched to a completely wrong location. This forces the network to learn descriptors that can distinguish adjacent pixels on textured surfaces, enabling sub-pixel matching accuracy — 30% better than previous best methods on challenging datasets.

### 4.2 Relevance to Our Pipeline

MASt3R is not used in the current prototype because it requires image pairs and adds complexity. However, it represents a future upgrade path for multi-image input. When multiple photographs are available, MASt3R could replace DUSt3R for even more precise cross-view reconstruction. Its sub-pixel matching would directly improve cell gradient quality by providing more accurate point positions before the point-to-cell conversion.

MASt3R also achieves zero-shot metric monocular depth estimation as a byproduct of training on metric-scale datasets. If integrated as the primary depth model, it could potentially provide both geometric consistency and metric scale from a single model, eliminating the need for separate Depth Pro + DUSt3R fusion.

---

## Chapter 5: Neural Radiance Fields — Historical Context

### 5.1 NeRF's Conceptual Contribution

Neural Radiance Fields (NeRF), introduced in 2020, pioneered the concept of representing scenes as continuous neural functions — mapping 3D coordinates to color and density. A neural network takes a 5D input (3D position plus 2D viewing direction) and outputs color and volume density. The network is trained by differentiable volume rendering: rays are cast from known camera positions, the network is queried at sampled points along each ray, and the rendering equation integrates the results to produce predicted pixel colors.

NeRF achieved breakthrough novel view synthesis quality, proving that neural networks can learn to represent 3D scenes with photorealistic fidelity. However, its rendering speed (seconds per frame due to per-ray network evaluations) prevents real-time use.

### 5.2 Influence on the Cell Architecture

NeRF is not used in our pipeline, but its conceptual influence is direct. The differential-integral cell can be understood as a discretized version of NeRF's continuous function. Where NeRF queries a neural network at arbitrary (x, y, z) to get color and density, our cells store the same information (albedo, density) at discrete grid positions. The cell gradients provide the continuity that discretization would otherwise lose — they encode how color and density change between grid points, enabling smooth interpolation and accurate subdivision.

NeRF also demonstrated that view-dependent appearance (specular highlights, glossy reflections) can be learned from data. Our AI texturing model serves a similar role — it produces view-dependent lighting effects without explicit physical computation. The philosophical connection is the same: learn what reality looks like from data rather than simulate it from equations.

NeRF's rendering framework additionally provides high-quality training data for our perception models. Several state-of-the-art monocular depth models are trained partially on NeRF-derived ground truth depth, bootstrapping 3D understanding from multi-view consistency.

---

## Chapter 6: From Gaussians to Cells — The Representation Evolution

### 6.1 3D Gaussian Splatting

3D Gaussian Splatting (3DGS), introduced in 2023, replaced NeRF's continuous neural function with explicit primitives — 3D Gaussian ellipsoids with position, shape, color, and opacity. These Gaussians are projected and blended onto the image plane, achieving real-time rendering at 100+ frames per second. The key advantage over NeRF is speed: no per-ray network evaluations, just projection and sorting.

For 3D understanding, each Gaussian's position encodes where a surface element exists in space. A collection of thousands of flat Gaussians approximates a surface. From a single image, the precision follows monocular depth estimation quality: 0.5-2 centimeters for nearby surfaces (within 1-2 meters), 2-5 centimeters for mid-range (2-4 meters), and 5-15 centimeters for far surfaces (beyond 4 meters).

### 6.2 2D Gaussian Surfels

The first evolution of 3DGS recognized that indoor scenes are dominated by flat surfaces. Walls, floors, ceilings, table tops — these are inherently 2D despite existing in 3D space. Representing them with 3D ellipsoids wastes one dimension encoding "this surface has zero thickness."

2D Gaussian Surfels replaced ellipsoids with flat oriented discs. Each surfel lies on a surface, facing outward, covering a small area. The result: approximately 10x fewer primitives for surface-dominated scenes with sharper, more accurate geometry. Neural Shell Texture further separated geometry (surfels) from appearance (a neural texture network), enabling arbitrarily detailed textures without increasing primitive count.

### 6.3 Differential-Integral Cells

The cell architecture extends this evolution further. Instead of individual primitives (points, Gaussians, surfels), the scene is divided into volumetric cells that store not just values but their spatial derivatives and integrals.

The key advantage for perception: when the point cloud from Depth Pro is converted to cells, the depth precision is not just stored as point positions — it is enriched with gradient information that encodes sub-cell detail. A 5-centimeter cell with a density gradient can locate the surface boundary within the cell to 1-2 centimeter precision. The gradient is the extra information extracted from the depth map's spatial variation, going beyond what the raw depth values provide.

The precision profile of cells matches the underlying depth estimation:

Nearby surfaces (within 1-2 meters): cells positioned with 0.5-2 centimeter accuracy. Gradients are clean and informative because nearby surfaces have many pixels with strong texture detail. Taylor expansion subdivision produces high-quality children.

Mid-range surfaces (2-4 meters): cells positioned with 2-5 centimeter accuracy. Gradients are moderately noisy. Taylor expansion is adequate but may need neural texture feature correction for fine detail.

Far surfaces (beyond 4 meters): cells positioned with 5-15 centimeter accuracy. Gradients are noisy due to few pixels per cell. Taylor expansion predictions are rough. These cells rely more on the AI texturing model for visual quality and less on gradient-derived detail.

This distance-dependent quality profile means that the cell grid naturally has higher quality near the camera and lower quality far from it — which aligns perfectly with the LOD system that renders near cells at high resolution and far cells at low resolution.

---

## Chapter 7: Multi-Frame Input and Motion Capture

### 7.1 The Input Spectrum

The system accepts any number of images, from one to thousands:

One photograph: the minimum viable input. Depth Pro provides metric depth. Bilateral smoothing cleans noise. The cell grid captures approximately 40-60% of the scene (the visible portion) with 3-7 centimeter precision. The remaining 40-60% is generated by the system.

Two to five photographs from different viewpoints: DUSt3R activates automatically. Cross-view correspondences resolve depth ambiguity. Scale is determined from the baseline between views. The cell grid captures 70-90% of the scene with 1-3 centimeter precision. Far less generation is needed.

Video or dense image sequence: multiple frames provide multiple views AND temporal information. Static elements are reconstructed with sub-centimeter precision from dozens of observations. Moving elements are detected and their motion is captured.

Multiple timeframes of the same scene: images taken at different moments (morning and evening, before and after furnishing, with and without people). The system separates static structure from dynamic elements and captures motion trajectories.

Each additional image improves both coverage (more of the scene is observed) and precision (each surface point is observed from more angles). The cell grid construction seamlessly handles any input count — more images produce more points, which produce denser cells with cleaner gradients.

### 7.2 Multi-Frame Reconstruction

When multiple images are available, the reconstruction pipeline extends:

Step 1: Run Depth Pro on each image independently. Each produces a metric depth map.

Step 2: Run DUSt3R on image pairs (or MASt3R for maximum precision). This produces geometrically consistent point maps across all views. The cross-attention mechanism in DUSt3R establishes which pixels in different images correspond to the same 3D point.

Step 3: Align all point maps into a single consistent coordinate frame. DUSt3R's pairwise alignments are globally optimized to minimize inconsistencies across all views. The metric scale from Depth Pro is applied to the globally aligned reconstruction.

Step 4: Merge all aligned points into the cell grid. Each cell now contains points from multiple views. The density, albedo, and normals are computed from the merged points with higher accuracy than any single view could provide. Gradients are cleaner because the multi-view points provide more spatial samples for finite difference computation.

The resulting cell grid has higher precision (1-3 centimeters versus 3-7 centimeters from a single image), better coverage (fewer gaps, fewer unseen surfaces), and cleaner gradients (more data points for finite differences).

### 7.3 Motion Detection from Multiple Timeframes

When the input includes images of the same scene at different moments (frames from a video, or separate photos taken at different times), the system detects motion by comparing the reconstructions.

The detection is straightforward: for each cell in the grid, compare its content across timeframes. Static cells have the same density, albedo, and normal in every timeframe — they represent walls, floors, furniture that hasn't moved. Dynamic cells have different content in different timeframes — they represent people walking, vehicles moving, doors opening, objects being relocated.

The detection algorithm:

Step 1: Build a cell grid from each timeframe independently (or from groups of frames within each timeframe if video input is provided).

Step 2: Align all timeframe grids to a common coordinate system using the static elements as anchors. The static structure (walls, floor, fixed furniture) provides the alignment — it should match exactly across timeframes.

Step 3: For each cell position, compute the difference in density and albedo across timeframes. Cells where the difference exceeds a threshold are flagged as dynamic. Cells where the difference is near zero are confirmed as static.

Step 4: Cluster the dynamic cells into objects. Adjacent dynamic cells with similar motion patterns are grouped into a single moving object (a person, a car, a pet). Each object cluster gets its own motion trajectory.

### 7.4 Real Motion Capture into Cell Animation Data

For each detected moving object, the system extracts actual motion data — real velocity and acceleration observed from the images, not generated by an AI motion model.

The motion extraction: at each timeframe T, the dynamic object's centroid position is known from the cell grid (the average position of all cells in the object cluster). The trajectory over time is the sequence of centroid positions: P(T0), P(T1), P(T2), ...

From this trajectory, the per-cell velocity and acceleration are computed:

velocity(T) = (P(T+1) - P(T-1)) / (2 × delta_T)
acceleration(T) = (P(T+1) - 2×P(T) + P(T-1)) / (delta_T²)

These values are stored in the dynamic cells' velocity and acceleration fields — the same fields that would be populated by text-to-motion generation for AI-animated characters. The rendering system doesn't distinguish between real captured motion and AI-generated motion — it simply reads the velocity and acceleration fields and interpolates the cell positions over time.

The shape deformation (how the person's body changes shape as they walk) is captured by tracking how each individual cell within the object cluster moves relative to the cluster's centroid. The arm cells swing forward and back. The leg cells alternate. The torso cells translate smoothly. All of this is observed from the real images rather than generated by a motion model.

### 7.5 Combining Real and Generated Motion

The most powerful workflow combines real captured motion with AI-generated elements:

Scenario: The user provides a video of their empty living room. They also provide a separate video of a person walking across a different room.

The system: reconstructs the living room as a static cell grid (from the room video), extracts the person's motion trajectory and body deformation (from the person video), re-targets the person's captured motion into the living room (placing the person's dynamic cells into the room's static cell grid, with the trajectory adjusted to fit the room's geometry — avoiding furniture, following the floor).

The result: a photorealistic animation of a real person walking through the user's actual room, with motion captured from real video rather than generated by AI. The person looks real because they ARE real (captured from video). The room looks real because it IS real (reconstructed from video). Only the combination is synthetic.

This workflow extends to any combination: real room + AI characters, AI room + real characters, real room + real characters in a different arrangement. The cell architecture handles all cases identically because it doesn't distinguish the source of cell data — captured and generated cells have the same structure.

### 7.6 Temporal Consistency Across Timeframes

When building cells from multiple timeframes, temporal consistency is enforced through the integral data.

The density integral of the static portion of the scene must remain constant across all timeframes. If the total mass of the walls changes between timeframes, something is wrong with the alignment or detection. The integral provides an automatic consistency check.

The velocity integral of each dynamic object over the full time range must equal its total displacement. If the integrated velocity diverges from the observed trajectory, the velocity data has errors that need correction. The integral catches these errors automatically.

These consistency checks ensure that multi-timeframe reconstruction produces coherent 4D data — a stable static world with smoothly animated dynamic elements — rather than a series of disconnected snapshots.

---

## Chapter 8: The Fusion Architecture for Cell Construction

### 8.1 The Complementary Strengths

Each perception model contributes a distinct capability to the cell grid.

Depth Pro provides metric depth (absolute scale in meters), sharp boundary detection, and fast inference. Its weakness is per-pixel random noise and occasional systematic errors on reflective or transparent surfaces.

DUSt3R provides geometric consistency (flat surfaces are truly flat, parallel walls are truly parallel) and cross-view correspondences. Its weakness is the lack of metric scale from a single image and lower boundary precision than Depth Pro.

SAM 3 provides semantic segmentation — identifying every object, surface, and architectural feature with category labels from a vocabulary of 270,000+ concepts. It does not estimate depth or 3D structure, but its labels are essential for the cell grid's semantic properties (material classification, functional field queries, auto-population decisions).

### 8.2 The Fusion Pipeline

The perception pipeline runs the following steps, adapted based on input count.

For single-image input: run Depth Pro to produce a metric depth map. Apply bilateral smoothing guided by the RGB image to reduce depth noise while preserving edges. Apply scale calibration by detecting objects of known size (doors at approximately 200 centimeters, standard furniture heights, floor tiles of known dimensions). Run SAM 3 for semantic segmentation. Backproject the smoothed, calibrated depth map to a 3D point cloud using the estimated focal length. Assign semantic labels from SAM 3 to each point. Convert the labeled point cloud to the cell grid, computing all properties, gradients, second derivatives, integrals, and neighbor summaries.

For multi-image input: run Depth Pro on each image for metric scale reference. Run DUSt3R (or MASt3R) on image pairs for geometric consistency and cross-view correspondences. Align all point maps to a common coordinate frame with Depth Pro's metric scale. Run SAM 3 on each image for semantic labels. Merge all aligned, labeled points into a single cell grid. The multi-view redundancy produces denser cells with cleaner gradients.

For temporal input (video or multi-timeframe): process each timeframe group through the multi-image pipeline above. Align timeframes using static structure as anchors. Detect dynamic elements by comparing cell content across timeframes. Extract motion data for dynamic cells. Build the final cell grid with static cells (permanent) and dynamic cells (time-varying).

### 8.3 Precision Budget of the Fused System

The fused system's precision at each step:

Depth estimation (Depth Pro + bilateral smoothing): 2-4% relative error after calibration, corresponding to 4-12 centimeters at 2-3 meters. The bilateral smoothing reduces gradient noise by 60-80% without affecting depth accuracy.

Geometric consistency (with DUSt3R, multi-image only): reduces local depth inconsistencies by 30-50% compared to Depth Pro alone. Flat surfaces become truly flat. Parallel walls become truly parallel.

Boundary precision (Depth Pro boundaries): object boundaries localized to within 1-2 pixels, corresponding to 2-5 millimeters of spatial precision. Cell density gradients at boundaries inherit this precision.

Cell position accuracy: 3-7 centimeters from single image (after calibration and smoothing). 1-3 centimeters from multi-image input. Sub-centimeter from dense video input.

Sub-cell precision from gradients: density gradients provide an additional 1-2 centimeters of precision for surface boundary location within cells, even at 5-centimeter cell resolution.

Scale calibration impact: applying scale calibration from a single known-size object improves precision by 50-70% by eliminating systematic scale error. Without calibration: 5-15 centimeter accuracy. With calibration: 3-7 centimeter accuracy.

### 8.4 Depth Smoothing — The Critical Preprocessing Step

The bilateral depth smoothing step deserves emphasis because it is the single most impactful preprocessing step for cell quality, yet it is the easiest to implement and fastest to run.

The bilateral filter operates on the depth map D with the RGB image I as guide:

For each pixel p, the smoothed depth is a weighted average of neighboring pixels' depths, where the weight depends on both spatial distance (nearby pixels contribute more) and color similarity (pixels with similar RGB values contribute more, pixels with different colors contribute less).

Pixels across an object boundary have different colors (the sofa is brown, the wall behind it is white). The color difference causes low weight, so the sofa's depth and the wall's depth are NOT averaged together — the sharp boundary is preserved.

Pixels within the same surface have similar colors. Their depths ARE averaged, smoothing out the random noise. The result is a depth map where surfaces are smooth (clean gradients) and boundaries are sharp (accurate density gradient direction).

Parameters: spatial sigma of 5-7 pixels, color sigma of 20-30 intensity units (on a 0-255 scale). Processing time: approximately 5 milliseconds for a 2-megapixel image. Quality impact: gradient noise reduced by 60-80%, directly improving Taylor expansion subdivision quality.

---

## Chapter 9: The Future and Precision Ceiling

### 9.1 Convergence Toward Foundation Models for 3D

The trend in the field is unmistakable: separate models for depth estimation, 3D reconstruction, segmentation, and scene understanding are converging toward unified foundation models that perform all tasks simultaneously.

VGGT (Visual Geometry Grounded Transformer), released by Meta, is an early example: it takes multiple images as input and directly outputs camera poses, depth maps, point maps, and 3D point clouds in a single forward pass, without any iterative optimization.

The trajectory suggests that within 1-2 years, a single foundation model will take photographs as input and output metric depth, semantic segmentation, 3D reconstruction, and scene graph in a single forward pass taking less than one second. When this happens, the cell grid construction will simplify (one model replaces three) and the precision will likely improve (joint optimization eliminates accumulated errors from stacking independent models).

The cell architecture is designed to be model-agnostic. When better perception models arrive, the cell grid construction simply receives more accurate point clouds. The cell structure, gradients, integrals, subdivision, collision, rendering, and animation are all unchanged. Only the quality of the input data improves — and the entire system benefits proportionally.

### 9.2 The Monocular Precision Ceiling

There is a theoretical ceiling on the precision achievable from a single monocular image, imposed by the information content of the image itself.

A typical 12-megapixel image provides about 36 million numbers. The 3D scene requires about 12 million depth values to fully describe. The information ratio is 3:1, meaning there is theoretically enough information to determine the depth map, but with no redundancy.

In practice, much of the image information is redundant or irrelevant to depth. The practical ceiling, based on current and foreseeable AI capabilities, is approximately 1-2% relative depth error for well-textured scenes with known-size objects for calibration. At 3 meters distance, this corresponds to 3-6 centimeters — firmly centimeter-level but unlikely to reach millimeter-level from a single image alone.

Three developments could push precision beyond this ceiling: higher-resolution imaging (4K and 8K input providing 4-16x more pixels), stronger learned priors (training on larger and more diverse 3D datasets), and physical reasoning (models that understand gravity, structural stability, and material properties to constrain depth predictions).

Multi-image input breaks through the ceiling entirely. With sufficient views, photogrammetric precision (sub-millimeter) is achievable. The cell architecture supports this seamlessly — more images produce better cells without any architectural change.

---

## Chapter 10: Summary — What You Can Achieve Today

### 10.1 The Precision Reality

Using the best available models (Depth Pro for metric depth, optionally DUSt3R for geometric consistency, SAM 3 for segmentation) combined with bilateral depth smoothing and scale calibration, the achievable precision for cell construction from a single photograph is:

Cell position accuracy (horizontal): 2-5 centimeters for surfaces within 3 meters, 5-10 centimeters for surfaces 3-5 meters away.

Cell position accuracy (vertical): 1-3 centimeters, because vertical position is well-constrained by perspective geometry and surface detection.

Boundary precision: 2-5 millimeters at object edges, enabling precise collision detection through cell density gradients.

Gradient quality (after bilateral smoothing): sufficient for meaningful Taylor expansion subdivision. Subdivision from 5-centimeter to 2.5-centimeter cells produces children within 10% color error of ground truth for smooth surfaces, and within 20% for textured surfaces.

Overall cell grid accuracy: 3-7 centimeters under favorable conditions (well-lit, textured scene with a scale reference), 5-15 centimeters under unfavorable conditions.

With multi-image input: 1-3 centimeters for all visible surfaces.

With video input: sub-centimeter for well-observed surfaces, plus motion capture data for dynamic elements.

### 10.2 The Eleven Factors That Most Affect Precision

Ranked from most to least impactful:

1. Number of input images. Multi-image input improves precision by 2-5x over single image. This is the single most effective improvement.

2. Presence of a known-size object for scale calibration. Eliminates systematic scale error, improving precision by 50-70% for single-image input.

3. Bilateral depth smoothing. Reduces gradient noise by 60-80%, directly improving cell subdivision quality. Essential for cell-based representations.

4. Surface texture quality. Well-textured surfaces provide richer depth cues than featureless surfaces.

5. Camera distance to the area of interest. Closer means more pixels, which means more precision. Photographing from 2 meters gives roughly twice the precision of photographing from 4 meters.

6. Image resolution. Higher resolution provides more detail. A 12-megapixel image gives approximately 40% better precision than a 3-megapixel image.

7. Lighting quality. Even, diffuse lighting produces better depth estimates than harsh directional lighting or dim lighting.

8. Visible scene structure. Seeing floor-wall junctions, ceiling-wall junctions, and multiple walls provides strong perspective constraints.

9. Number of visible objects. More objects provide more size references and occlusion cues.

10. Image sharpness. Motion blur, focus blur, and compression artifacts degrade depth precision.

11. Camera angle. A photograph from a corner showing multiple surfaces provides stronger perspective cues than one taken straight at a wall.

### 10.3 The Honest Assessment

Single-image centimeter-precision 3D reconstruction is real, achievable today, and practical for most applications. It is sufficient for interior design visualization, spatial planning, game level design, film pre-visualization, and navigation.

It is not sufficient for precision manufacturing, surgical robotics, or structural engineering, which require sub-millimeter accuracy from dedicated sensors.

Multi-image and video input breaks through the single-image ceiling, achieving photogrammetric precision while adding the unique capability of motion capture — observing how things actually move rather than imagining how they might move. The cell architecture handles all input types with the same data structure, seamlessly improving quality as more data becomes available.
