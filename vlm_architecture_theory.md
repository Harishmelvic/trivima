# VLM Architecture for Spatial Intelligence
## Qwen2.5-VL + 3D-RoPE with SpatialVLM Knowledge Distillation

---

# Chapter 1: The VLM's Role in the Pipeline

## 1.1 What the VLM Does

The VLM (Vision-Language Model) provides the design intelligence that pure geometry cannot. The cell grid and validation fields handle physics — collision, surfaces, boundaries, clearance. The VLM handles judgment — style, culture, function, and aesthetic quality.

The VLM is not in the per-frame rendering loop. It is invoked at specific decision points: when the system needs to identify what furniture a room is missing (auto-furnishing), when it needs to re-rank candidate placement positions by design quality (aesthetic judgment), when it needs to determine what an environment is and how to extend it (environment classification), and when it needs to plan multi-object placement sequences (lookahead planning).

The chosen model is Qwen2.5-VL-32B, modified with 3D Rotary Positional Encoding (3D-RoPE) to give it native understanding of 3D space. SpatialVLM's quantitative spatial knowledge is transferred via distillation during training rather than feature fusion at inference — a deliberate architectural decision that trades a small accuracy margin for dramatically simpler deployment.

## 1.2 Why One Model, Not Three

The original architecture considered fusing three models at the feature level: SpatialVLM for spatial measurement, Qwen2.5-VL for reasoning and aesthetics, and GLM-4.5V for 3D positional encoding. This fusion would have created a model with all three capabilities interacting at the feature level during inference.

The fusion was rejected for four reasons. First, the accuracy gain was marginal — feature-level fusion achieved 92-95% on spatial-aesthetic benchmarks versus 85-90% for the simpler distillation approach, a 2-5% improvement that did not justify the complexity. Second, running SpatialVLM at inference adds 0.5-1 second latency per query and requires an additional 13GB of GPU memory. Third, the cross-attention bridges, gated injection mechanisms, and three-point feature injection added approximately 200-300 million parameters and substantial engineering complexity. Fourth, the training cost for fusion was $15,000-30,000 versus under $5,000 for distillation.

The chosen approach: use Qwen2.5-VL as the single runtime model, enhance it with 3D-RoPE (zero cost), and distill SpatialVLM's spatial knowledge into it during training (one-time cost). At inference, only Qwen runs. The spatial measurement capability is baked into its weights, not streamed from a separate model.

---

# Chapter 2: 3D-RoPE — Native 3D Awareness at Zero Cost

## 2.1 The Problem with 2D Positional Encoding

Standard VLMs encode image positions in 2D pixel coordinates. When the model processes a patch at pixel position (100, 200), its positional encoding tells the attention mechanism "this patch is at horizontal position 100, vertical position 200." All spatial reasoning must then be inferred from visual cues — the model has no direct access to 3D coordinates.

The attention mechanism computes similarity between patches, and patches with similar positional encodings attend to each other more strongly. With 2D encoding, "nearby in the image" means "high attention." But nearby in the image does not mean nearby in 3D space.

Two patches at pixel positions (100, 200) and (110, 200) are adjacent in the image — the 2D encoding gives them high mutual attention. But one might be on a nearby table edge (1 meter from camera) and the other on a far wall (4 meters from camera). They are 3 meters apart in physical space despite being 10 pixels apart in the image. The model must learn to overcome its own positional encoding to reason spatially — fighting the built-in bias that pixel proximity equals spatial proximity.

This mismatch is why standard VLMs struggle with quantitative spatial questions. Asked "how far is the table from the wall?" a 2D-encoded model must reverse-engineer 3D distances from perspective cues, object sizes, and learned priors. It can give approximate answers but cannot achieve centimeter-level precision because its internal representations do not encode physical distances.

## 2.2 3D-RoPE: Encoding Physical Space

3D-RoPE, first demonstrated in GLM-4.5V, replaces 2D pixel coordinates with 3D physical coordinates in the Rotary Position Embedding.

For a patch at 3D position (X=1.5m, Y=0.0m, Z=-0.8m) in the room (derived from the Depth Pro point map), the positional encoding rotates the feature vector by angles proportional to X, Y, and Z — the physical metric coordinates, not the pixel coordinates.

The attention mechanism now computes similarity based on physical proximity. Two patches that are close in 3D space (a cup on a table and the table surface beneath it — 5cm apart physically) have high mutual attention, even if they are at different pixel positions in the image. Two patches that are far in 3D space (a nearby table edge and a far wall visible just behind it — 3 meters apart) have low mutual attention, even though they are adjacent pixels.

The implications for spatial reasoning are profound:

Furniture groupings become attention clusters. A sofa, coffee table, and lamp within 2 meters of each other form a natural attention group — the model "sees" them as a unit because their 3D positions are close. With 2D encoding, a sofa at the bottom of the image and a lamp at the top of the image (but physically next to each other in the room) would have low mutual attention.

Room boundaries form coherent structures. Wall cells at the same physical height attend to each other strongly, forming a continuous wall representation even if the wall spans the entire image width. Floor cells form another coherent layer. The model perceives the room's 3D structure through its attention patterns, without having to learn 3D from 2D.

Depth discontinuities are naturally handled. An object in front of a wall creates a depth jump. With 3D-RoPE, the object's patches and the wall's patches have different Z coordinates, so they attend to each other less — the model automatically separates foreground from background.

## 2.3 Implementation

We replace Qwen2.5-VL's standard 2D-RoPE with 3D-RoPE at the input stage. Before processing begins, the perception pipeline (Depth Pro + SAM 3) produces the 3D point map. Each image patch's center pixel has an associated 3D coordinate (X, Y, Z) in metric room coordinates.

The encoding dimensions are allocated equally: one-third to X, one-third to Y, one-third to Z. This gives equal representational capacity to horizontal position, vertical position, and depth.

The metric coordinates are normalized to prevent numerical issues. X and Z (horizontal) are divided by the room width/length (typically 3-6 meters). Y (vertical) is divided by the room height (typically 2.5-3 meters). After normalization, all coordinates are in the range 0 to 1.

For text tokens in the input (scene descriptions, candidate positions), which do not have associated 3D coordinates, a special non-spatial positional encoding is used. This allows the model to distinguish between tokens with spatial meaning (image patches with 3D positions) and tokens with linguistic meaning (text).

For patches where the depth estimate is unreliable (low confidence cells — glass, mirrors, dark regions), the encoding falls back to 2D pixel coordinates: (u, v, 0). This gracefully degrades to standard 2D behavior for uncertain patches rather than encoding incorrect 3D positions.

## 2.4 The Zero-Cost Property

3D-RoPE adds zero parameters and zero runtime cost to the model. It is purely a change in the values used for positional encoding — the same number of multiplications and additions are performed, just with different input values.

The only additional cost is looking up the 3D coordinates for each patch, which are already computed by the perception pipeline. The 3D-RoPE injection reads these coordinates and uses them in place of pixel coordinates — one memory read per patch.

This makes 3D-RoPE the highest-value, lowest-cost improvement in the entire architecture. It fundamentally changes how the model perceives space, at zero additional inference cost, zero additional parameters, and zero additional memory.

## 2.5 What 3D-RoPE Does NOT Do

3D-RoPE is not a substitute for spatial training data. It gives the model the ability to perceive 3D structure through attention patterns, but the model still needs to learn what to do with this perception. Without training on spatial reasoning tasks, a 3D-RoPE-enhanced model will have better spatial attention patterns but will not automatically produce better spatial answers.

3D-RoPE does not provide metric measurement capability. It encodes physical proximity (patches that are close in 3D attend more), but it does not directly teach the model to output distances in centimeters. That capability comes from the SpatialVLM distillation (Chapter 3).

3D-RoPE is only as good as the depth map it receives. If the Depth Pro output is wrong (glass, mirrors, dark regions), the 3D positions are wrong, and the attention patterns are misleading. The confidence-based fallback to 2D encoding mitigates this, but it means 3D-RoPE's benefit is concentrated on high-confidence regions.

---

# Chapter 3: SpatialVLM Knowledge Distillation

## 3.1 What SpatialVLM Knows

SpatialVLM was built specifically for quantitative spatial reasoning. Its training data includes 2 billion spatial VQA examples generated from 10 million real-world images using depth estimation and 3D reconstruction. Every training example teaches the model to associate visual patterns with real-world measurements: distances in meters, heights in centimeters, angles in degrees.

The result: SpatialVLM can answer questions like "how far is the sofa from the wall?" with answers accurate to within 10-15 centimeters. It can estimate room dimensions, furniture sizes, and spatial relationships with quantitative precision that general-purpose VLMs cannot match.

We want this capability in our model without running SpatialVLM at inference.

## 3.2 The Distillation Approach

Knowledge distillation transfers SpatialVLM's spatial reasoning capability into Qwen2.5-VL during training. The process:

Step 1 — Generate spatial labels. Run SpatialVLM on 50,000-100,000 room images. For each image, ask SpatialVLM a set of standardized spatial questions: "What is the distance between [object A] and [object B]?" "What is the height of [surface]?" "How large is the floor area?" "What are the dimensions of [object]?" Record SpatialVLM's answers as the spatial ground truth.

Step 2 — Train Qwen to match. Fine-tune Qwen2.5-VL (with 3D-RoPE and LoRA adapters) on these spatial question-answer pairs. The loss function rewards Qwen for producing answers that match SpatialVLM's spatial precision. Qwen learns to associate the same visual patterns with the same quantitative measurements — but using its own internal representations rather than SpatialVLM's features.

Step 3 — Verify knowledge transfer. Test Qwen on held-out spatial questions that SpatialVLM answered but that were not in the training set. If Qwen's answers are within 15-20% of SpatialVLM's accuracy, the knowledge transfer was successful. If Qwen achieves 85-90% of SpatialVLM's spatial precision, the distillation is effective.

The advantage over feature fusion: SpatialVLM is used only during training data generation (offline, no latency constraint). At inference, only Qwen runs — faster, simpler, cheaper. The spatial knowledge lives in Qwen's weights, not in a separate model's features.

## 3.3 What Is Lost in Distillation

Distillation is not lossless. The student (Qwen) typically achieves 85-90% of the teacher's (SpatialVLM's) accuracy on the distilled capability. The lost 10-15% comes from:

Representation compression: SpatialVLM's 13B parameters include specialized spatial features that cannot be perfectly replicated in Qwen's differently structured feature space. Some spatial nuances are lost.

Training data coverage: the distillation dataset covers 50,000-100,000 images, but SpatialVLM was trained on 10 million. The student sees a subset of scenarios and may not generalize to rare room configurations as well as the teacher.

Competing objectives: Qwen must retain its aesthetic reasoning, language quality, and general intelligence while learning spatial precision. These objectives sometimes conflict — the model may sacrifice spatial precision in exchange for better aesthetic reasoning in edge cases.

The 85-90% accuracy target is acceptable because the coordinate-validation system handles the precision-critical decisions (collision, surface detection, clearance). The VLM provides judgment, not measurement. Whether the VLM estimates a distance as 1.3m or 1.2m matters less than whether it understands that a coffee table belongs between the sofa and the TV.

## 3.4 Distillation Training Protocol

The distillation is part of the broader VLM fine-tuning, which has two phases.

Phase 1 — Spatial distillation (3-5 days on 4×A100). Qwen2.5-VL with 3D-RoPE is fine-tuned via LoRA (rank 32, alpha 64) on spatial QA data generated by SpatialVLM. The loss is L1 on numerical answers (distance predictions) plus cross-entropy on categorical answers (spatial relationship classifications like "above," "left of," "in front of"). The model learns to produce quantitative spatial measurements from visual input.

Phase 2 — Aesthetic and placement fine-tuning (3-5 days on 4×A100). The spatially-distilled model is further fine-tuned on placement judgment data: room images with candidate positions, ranked by quality from designer-arranged rooms (3D-FRONT dataset). The model learns to rank placements by aesthetic quality, explain its reasoning, and consider both spatial and aesthetic factors.

Optional Phase 3 — Human preference calibration (1-2 days on 2×A100). DPO (Direct Preference Optimization) on 5,000-10,000 human preference pairs calibrates the model's aesthetic judgments to match human expectations.

Total training: 7-12 days, $3,000-5,000 compute. Compare to the feature fusion training: 15-22 days, $15,000-30,000.

---

# Chapter 4: The VLM in the Pipeline

## 4.1 Invocation Points

The VLM is called at four specific points in the pipeline, never in the per-frame loop.

Environment classification (once per scene): the VLM examines the photograph and determines what kind of environment it is — indoor room, urban street, natural terrain, coastal scene. This classification drives the shell extension strategy (room planes vs terrain generation vs building facades), the auto-population rules (furniture for rooms, street elements for urban, natural features for terrain), and the validation field parameters.

Aesthetic re-ranking (per placement query): the coordinate-validation system identifies the top 20-50 physically valid candidate positions. The VLM examines these positions in context and re-ranks them based on style coherence, cultural appropriateness, visual relationships, and design sophistication. The VLM's top choice becomes the recommendation.

Auto-furnishing planning (once per scene, if activated): the VLM examines the cell grid's semantic content and identifies functional gaps — "this living room has a sofa but no coffee table," "this bedroom needs nightstands." It determines the placement order (anchor pieces first, dependent pieces second, accent pieces last) and provides lookahead — checking that the sofa position leaves room for the coffee table before committing.

Object style matching (per object, during auto-furnishing): when an object is retrieved from the furniture database, the VLM evaluates whether its style matches the room. A chrome floor lamp in a rustic cabin gets rejected. A mid-century side table next to a modern sofa gets approved. The VLM outputs a compatibility score and suggests alternatives if the match is poor.

## 4.2 Division of Labor

The VLM and the coordinate-validation system have clearly separated responsibilities.

The validation system handles everything spatial and physical: collision detection (cell lookup, 0.001ms), surface validation (cell type check), clearance computation (BFS distance), pathway checking (ray traversal through cells), and geometric aesthetic scoring (spatial balance, spacing rhythm, symmetry). These are precise, deterministic, and fast.

The VLM handles everything that requires understanding and judgment: room type classification, design style detection, functional gap identification, object category selection, style matching, cultural appropriateness, and qualitative aesthetic ranking. These require world knowledge and visual reasoning that geometric computation cannot provide.

Neither alone achieves what the combination provides. The validation system without the VLM produces spatially optimal but stylistically ignorant placements. The VLM without the validation system produces stylistically appropriate positions that might float in air, collide with furniture, or block doorways.

## 4.3 Latency Budget

The VLM adds latency only at decision points, not per-frame.

Environment classification: approximately 2 seconds. Called once when the scene is first loaded. Runs in parallel with cell grid construction and shell extension.

Aesthetic re-ranking of 20-50 candidates: approximately 50-100 milliseconds. Fast enough for interactive placement — the user sees the heatmap update within 100ms of selecting a position.

Auto-furnishing plan: approximately 3-5 seconds for a full room plan (5-10 objects). Called once. The actual placement (validation field evaluation) is separate and fast.

Style matching: approximately 200-500 milliseconds per object. Amortized across the auto-furnishing sequence.

None of these are in the 60+ FPS render loop. The renderer never waits for the VLM. The VLM provides decisions that the renderer executes.

---

# Chapter 5: Training the VLM

## 5.1 What Gets Trained

Qwen2.5-VL's base weights are frozen. Only LoRA adapters are trained, plus the 3D-RoPE positional encoding replacement (which is a deterministic computation, not learned).

LoRA adapters (rank 32, alpha 64) are applied to the query, key, value, and output projection matrices of all attention layers. This adds approximately 200-400 million trainable parameters — 1-2% of Qwen's total 32 billion parameters.

The 3D-RoPE modification is not trained — it is a deterministic function that converts 3D coordinates to rotational encodings using fixed trigonometric functions. The encoding scheme is the same as standard RoPE, just with different input values.

## 5.2 Training Data

The training data combines three sources.

Spatial distillation data (50,000-100,000 examples): room images paired with SpatialVLM-generated spatial measurements. Questions cover distances, heights, areas, spatial relationships, and room dimensions. This data transfers SpatialVLM's quantitative precision.

Placement judgment data (100,000+ examples): room images with candidate placement positions, ranked by quality. Derived from the 3D-FRONT dataset (18,968 professionally designed rooms) using the perturbation protocol — original designer arrangements as positive examples, displaced arrangements as negative examples with scores proportional to displacement distance.

Human preference data (5,000-10,000 pairs): room layouts where human evaluators chose between two placement options. Used for DPO calibration in the optional Phase 3.

## 5.3 Monitoring for Capability Degradation

Throughout training, canary metrics detect if Qwen's original capabilities are degrading.

Aesthetic canary: 500 held-out room layouts with known quality ratings. If correlation with human ratings drops below 0.75 (baseline is approximately 0.80), the learning rate is reduced.

Language canary: 200 held-out explanation prompts. If perplexity increases by more than 15% compared to baseline, training is paused.

Spatial canary: 500 held-out spatial measurement questions. If accuracy drops below 80% of SpatialVLM's standalone accuracy, the distillation data proportion is increased.

These canaries provide early warning of catastrophic interference — a common risk when fine-tuning large pretrained models.

## 5.4 Training Cost Summary

| Phase | Duration | Hardware | Cost |
|---|---|---|---|
| Spatial distillation | 3-5 days | 4×A100 | $1,500-2,500 |
| Placement fine-tuning | 3-5 days | 4×A100 | $1,500-2,500 |
| Human preference (optional) | 1-2 days | 2×A100 | $300-500 |
| **Total** | **7-12 days** | | **$3,000-5,000** |

Compare to the rejected feature fusion approach: 15-22 days, $15,000-30,000. The distillation approach costs 5-6x less with 85-90% of the accuracy.

---

# Chapter 6: Why Not Fusion — The Decision Record

## 6.1 What Was Considered

The feature-level fusion architecture would have run SpatialVLM alongside Qwen at inference, with cross-attention bridges injecting SpatialVLM's spatial features into Qwen's processing at three layer points. Gated injection mechanisms would control how much spatial information influences Qwen's reasoning.

This approach was fully designed (9 chapters, 396 lines of theory) and evaluated against the distillation alternative.

## 6.2 The Comparison

| Factor | Feature Fusion | Distillation (chosen) |
|---|---|---|
| Spatial accuracy | 92-95% of SpatialVLM | 85-90% of SpatialVLM |
| Inference latency | +0.5-1.0 seconds | Zero (only Qwen runs) |
| GPU memory | +13 GB (SpatialVLM) | Zero additional |
| Training cost | $15,000-30,000 | $3,000-5,000 |
| Training time | 15-22 days | 7-12 days |
| Engineering complexity | High (cross-attention, gating, multi-point injection) | Low (LoRA fine-tuning) |
| Deployment complexity | Two models, shared GPU memory | One model |
| Additional parameters | 200-300M fusion module | Zero (only LoRA) |

## 6.3 Why Distillation Won

The 2-5% accuracy improvement from fusion did not justify:
- 5-6x higher training cost
- 0.5-1 second additional latency per query
- 13 GB additional GPU memory
- Months of additional engineering for the fusion module
- Two-model deployment complexity

The coordinate-validation system handles precision-critical decisions. The VLM provides judgment. A VLM with 85-90% of SpatialVLM's precision is sufficient for judgment — it doesn't need centimeter-precision measurement because the validation system provides that.

## 6.4 When Fusion Might Be Reconsidered

Fusion becomes worthwhile if: a future unified model (VGGT-like) achieves SpatialVLM's precision AND Qwen's reasoning in one model at no extra cost (eliminating the latency and memory concerns), or the placement quality gap between 90% and 95% becomes the bottleneck for user satisfaction (suggesting the current approach has been optimized to the point where VLM quality is the limiting factor), or inference hardware becomes cheap enough that running two models is negligible cost.

Until then, distillation is the correct choice.

---

# Chapter 7: Evaluation

## 7.1 Evaluation Dimensions

The VLM is evaluated on five dimensions.

Spatial precision: does the model correctly estimate distances and spatial relationships? Measured against SpatialVLM's outputs on held-out data. Target: within 15-20% of SpatialVLM's accuracy (this is the distillation degradation budget).

Aesthetic quality: does the model rank placements in agreement with human preferences? Measured by human evaluation on held-out room images. Target: at least 80% agreement with human preferences.

Reasoning coherence: are the model's explanations logically consistent? Measured by human evaluation of explanation quality. Target: 90% rated as coherent and accurate.

3D-RoPE benefit: does 3D-RoPE improve spatial reasoning compared to standard 2D encoding? Measured by ablation — running the same model with 2D-RoPE versus 3D-RoPE. Target: at least 10% improvement in spatial accuracy from 3D-RoPE alone.

End-to-end placement quality: does the VLM + validation system produce better placements than the validation system alone? Measured by comparing placements with and without VLM aesthetic re-ranking. Target: at least 15% improvement in human preference ratings.

## 7.2 The Key Ablation

The most important ablation compares three configurations:

Configuration A — Validation only (no VLM): the validation system's geometric aesthetic score selects the placement. No VLM is involved. This is the baseline.

Configuration B — VLM with 2D-RoPE: Qwen2.5-VL with distillation and LoRA, but standard 2D positional encoding. This measures the VLM's contribution without 3D awareness.

Configuration C — VLM with 3D-RoPE (full system): Qwen2.5-VL with distillation, LoRA, and 3D-RoPE. This is the target architecture.

Expected results: C > B > A. The gap between A and B measures VLM design intelligence. The gap between B and C measures 3D-RoPE's contribution. If B ≈ C, 3D-RoPE isn't helping and the model was already inferring 3D from 2D cues. If A ≈ B, the VLM isn't adding value beyond geometric aesthetics.

---

# Chapter 8: Summary

## 8.1 The Architecture

One model: Qwen2.5-VL-32B with 3D-RoPE and LoRA adapters.

3D-RoPE: replaces pixel coordinates with metric 3D coordinates from the depth map. Zero parameters, zero cost, fundamental improvement in spatial perception.

SpatialVLM distillation: spatial measurement capability transferred via training data, not inference-time feature fusion. 85-90% of SpatialVLM's precision at zero inference cost.

Role: design intelligence (style, culture, function, aesthetics) operating on top of the validation system's physics guarantees.

## 8.2 The Numbers

| Metric | Value |
|---|---|
| Base model | Qwen2.5-VL-32B |
| Trainable parameters | 200-400M LoRA (1-2% of total) |
| 3D-RoPE cost | Zero parameters, zero latency |
| Spatial precision | 85-90% of SpatialVLM |
| Training time | 7-12 days |
| Training cost | $3,000-5,000 |
| Inference latency | 50-100ms (re-ranking), 2-5s (full planning) |
| GPU memory | ~32 GB (8-bit quantized Qwen only) |
| Aesthetic agreement with humans | Target 80%+ |

## 8.3 The Integration

The VLM sits between the validation system (which guarantees physics) and the user (who wants intelligent placement). The validation system produces the set of safe options. The VLM selects the best option. The coordinate-validation system can function without the VLM (geometric aesthetics only). The VLM cannot function without the validation system (no physics guarantees). This asymmetric dependency is intentional — physics must never depend on a neural network's judgment.
