# VLM Architecture for Spatial Intelligence
## Qwen3-VL with SpatialVLM Knowledge Distillation

---

# Chapter 1: The VLM's Role in the Pipeline

## 1.1 What the VLM Does

The VLM (Vision-Language Model) provides the design intelligence that pure geometry cannot. The cell grid and validation fields handle physics — collision, surfaces, boundaries, clearance. The VLM handles judgment — style, culture, function, and aesthetic quality.

The VLM is not in the per-frame rendering loop. It is invoked at specific decision points: when the system needs to identify what furniture a room is missing (auto-furnishing), when it needs to re-rank candidate placement positions by design quality (aesthetic judgment), when it needs to determine what an environment is and how to extend it (environment classification), and when it needs to plan multi-object placement sequences (lookahead planning).

The chosen model is Qwen3-VL, which natively provides 3D grounding, advanced spatial perception, and Interleaved-MRoPE positional encoding — capabilities that the previous Qwen2.5-VL required custom modifications (3D-RoPE injection) to approximate. SpatialVLM's quantitative spatial knowledge is still transferred via distillation during training, but less distillation is needed because Qwen3-VL starts from a stronger spatial baseline.

## 1.2 Why Qwen3-VL Over Qwen2.5-VL

Qwen3-VL provides three capabilities that Qwen2.5-VL lacked, all of which our architecture previously had to build from scratch.

Native 3D grounding: Qwen3-VL can output 3D bounding boxes for objects in indoor and outdoor scenes. Qwen2.5-VL had no 3D understanding — we had to inject 3D-RoPE positional encoding using metric depth coordinates from our perception pipeline. With Qwen3-VL, the model has its own spatial-temporal encoding (Interleaved-MRoPE) that handles 3D relationships natively.

Advanced spatial perception: Qwen3-VL judges object positions, viewpoints, and occlusion relationships out of the box. Qwen2.5-VL needed SpatialVLM distillation to achieve even basic spatial reasoning. Qwen3-VL starts at approximately 36-40% accuracy on numerical spatial questions (versus approximately 20-30% for Qwen2.5-VL), giving distillation a better starting point.

Thinking variant: Qwen3-VL comes in both Instruct (fast, direct answers) and Thinking (step-by-step reasoning) editions. The Thinking variant is valuable for auto-furnishing planning, where multi-step design reasoning benefits from explicit chain-of-thought. The Instruct variant is used for fast re-ranking where latency matters.

However, Qwen3-VL's spatial precision is not sufficient on its own. Independent benchmarks show that while Qwen3-VL improves over Qwen2.5-VL, its numerical spatial accuracy remains limited at 36-40% — meaning it gets the spatial direction right but the exact measurements wrong most of the time. Distillation from SpatialVLM remains necessary to push precision to the 60-75% range needed for placement quality.

## 1.3 Model Selection by Use Case

| Use Case | Model | Why |
|---|---|---|
| Distillation testing | Qwen3-VL-8B-Instruct | Fast, fits on A40 (16GB fp16), good baseline |
| Fast re-ranking (production) | Qwen3-VL-8B-Instruct | Under 500ms per forward pass |
| Auto-furnishing planning | Qwen3-VL-8B-Thinking | Step-by-step design reasoning |
| Maximum quality | Qwen3-VL-30B-A3B-Instruct | 30B total params, only 3B active — large model knowledge at small model speed |

The MoE variant (30B-A3B) is particularly interesting for production: it has 30 billion total parameters (capturing broad design knowledge) but activates only 3 billion per inference (keeping latency comparable to the 4B model). This is the best quality-to-latency ratio in the Qwen3-VL family.

## 1.4 Why One Model, Not Three

The original architecture considered fusing three models at the feature level: SpatialVLM for spatial measurement, Qwen2.5-VL for reasoning and aesthetics, and GLM-4.5V for 3D positional encoding. This fusion was rejected because the accuracy gain was marginal (2-5%), the inference cost was high (+0.5-1 second latency, +13GB GPU memory), and the engineering complexity was substantial (cross-attention bridges, gated injection, three-point feature injection).

The evolution from that decision: Qwen3-VL now provides the 3D encoding natively (eliminating the need for GLM-4.5V's 3D-RoPE), and its improved spatial baseline reduces the amount of SpatialVLM distillation needed. The architecture has simplified from "Qwen2.5 + 3D-RoPE hack + heavy distillation" to "Qwen3 + light distillation."

---

# Chapter 2: Native Spatial Encoding in Qwen3-VL

## 2.1 What Changed from Qwen2.5-VL

Qwen2.5-VL used standard 2D Multi-Resolution Rotary Position Embedding (MRoPE) — positional encoding based on pixel coordinates. To give it 3D awareness, our architecture injected custom 3D-RoPE: replacing pixel coordinates (u, v) with metric 3D coordinates (X, Y, Z) from the Depth Pro point map. This was a hack — modifying the model's internal positional encoding at the input stage.

Qwen3-VL replaces this with Interleaved-MRoPE, an architectural improvement that provides full-frequency allocation over time, width, and height dimensions. Combined with DeepStack (which fuses multi-level ViT features to capture fine-grained details), Qwen3-VL has significantly better spatial understanding without external positional encoding injection.

The result: we no longer need to modify the model's internal architecture. Qwen3-VL perceives spatial relationships through its native encoding, trained on datasets that include 3D grounding tasks. Our custom 3D-RoPE injection would actually conflict with Qwen3-VL's Interleaved-MRoPE and potentially degrade performance.

## 2.2 How We Provide 3D Context Instead

Instead of injecting 3D coordinates into the positional encoding (the old approach), we provide 3D information as prompt context that Qwen3-VL can reason about natively.

When invoking the VLM for any spatial task, the prompt includes the 3D scene context derived from the cell grid:

For aesthetic re-ranking: the prompt describes each candidate position with its 3D coordinates, distances to nearby objects (from BFS clearance), surface type (from surface field), and functional scores (from functional field). The VLM receives structured spatial data and reasons about it using its native 3D understanding.

For auto-furnishing: the prompt describes the room's dimensions (from shell extension planes), existing furniture positions and types (from cell grid semantic labels), and available floor area (from surface field). The VLM identifies gaps using this explicit spatial description.

For environment classification: the prompt includes the photograph plus the depth distribution and dominant surface normals from the cell grid. The VLM classifies the environment using both visual and geometric cues.

This approach is simpler (no model modification), more portable (works with any VLM, not just Qwen), and more robust (the VLM uses the spatial data through its trained reasoning path, not through a hacked positional encoding).

## 2.3 3D Bounding Box Output

Qwen3-VL can natively output 3D bounding boxes for objects — a capability Qwen2.5-VL entirely lacked. When asked "where is the sofa?" Qwen3-VL can respond with a 3D bounding box specifying the object's position and extent in 3D space.

This capability enables a new integration pattern: instead of only using the VLM for re-ranking pre-computed candidates, the VLM can directly propose 3D positions that the validation system then checks. The VLM suggests "place the coffee table at approximately (2.1, 0.0, 1.8)" and the validation system verifies collision, surface support, and clearance at that specific position. If the VLM's suggested position fails validation, the validation system finds the nearest valid position.

This hybrid (VLM proposes, validation verifies) combines the VLM's design intelligence with the validation system's physics guarantees. Neither system's errors cascade unchecked.

## 2.4 What We Lost by Dropping 3D-RoPE

The custom 3D-RoPE injection had one advantage that Qwen3-VL's native encoding does not: it used OUR depth map's metric coordinates. This meant the VLM's attention patterns directly reflected our specific 3D reconstruction — patches close in our point map attended to each other strongly, regardless of how Qwen3-VL's own spatial understanding parsed the image.

With Qwen3-VL's native encoding, the VLM uses its own learned spatial understanding, which may disagree with our depth map. If our Depth Pro reconstruction says two patches are 2 meters apart but Qwen3-VL's internal model thinks they're 1 meter apart, the VLM reasons from its own (potentially less accurate) spatial model.

This loss is mitigated in two ways. First, we provide explicit distances from the cell grid in the prompt context — the VLM doesn't need to estimate distances from the image when we've already computed them. Second, the validation system (not the VLM) makes all physics decisions, so the VLM's spatial inaccuracies don't cause collisions or floating objects.

## 2.5 The Net Assessment

Switching from custom 3D-RoPE injection to Qwen3-VL's native spatial encoding is a net positive:

Eliminated: custom model modification code, risk of encoding conflicts across model versions, the engineering complexity of injecting per-patch 3D coordinates.

Gained: simpler integration (prompt-based 3D context), model portability (can swap VLMs without rewriting positional encoding code), native 3D bounding box output, Thinking variant for complex reasoning.

Lost: direct integration of our depth map into attention patterns. Mitigated by providing explicit spatial data in prompts.

---

# Chapter 3: SpatialVLM Knowledge Distillation

## 3.1 What SpatialVLM Knows

SpatialVLM was built specifically for quantitative spatial reasoning. Its training data includes 2 billion spatial VQA examples generated from 10 million real-world images using depth estimation and 3D reconstruction. Every training example teaches the model to associate visual patterns with real-world measurements: distances in meters, heights in centimeters, angles in degrees.

The result: SpatialVLM can answer questions like "how far is the sofa from the wall?" with answers accurate to within 10-15 centimeters. It can estimate room dimensions, furniture sizes, and spatial relationships with quantitative precision that general-purpose VLMs cannot match.

We want this capability in our model without running SpatialVLM at inference.

## 3.2 The Distillation Approach

Knowledge distillation transfers SpatialVLM's spatial reasoning capability into Qwen3-VL during training. Because Qwen3-VL already has native 3D grounding (starting at approximately 36-40% numerical spatial accuracy versus 20-30% for Qwen2.5-VL), less distillation is needed — the model has a better foundation to build on.

The process:

Step 1 — Generate spatial labels. Run SpatialVLM on 50,000-100,000 room images. For each image, ask SpatialVLM standardized spatial questions. Record SpatialVLM's answers as the spatial ground truth.

Step 2 — Train Qwen3-VL to match. Fine-tune Qwen3-VL with LoRA adapters on these spatial question-answer pairs. The loss function rewards Qwen3-VL for producing answers that match SpatialVLM's spatial precision. Because Qwen3-VL starts from a stronger spatial baseline, it converges faster and may need fewer training examples than Qwen2.5-VL required.

Step 3 — Verify knowledge transfer. Test Qwen3-VL on held-out spatial questions. Target: within 25-35% of SpatialVLM's accuracy (achieving 60-75% on numerical spatial questions). This is less ambitious than the 85-90% target for Qwen2.5-VL because Qwen3-VL's native spatial ability means the gap to close is smaller.

The advantage over feature fusion remains: SpatialVLM is used only during training data generation (offline). At inference, only Qwen3-VL runs.

## 3.3 What Is Lost vs Gained in Distillation

Distillation is not lossless. The student (Qwen3-VL) typically achieves 60-75% of the teacher's (SpatialVLM's) accuracy on numerical spatial questions — but this is a significant improvement over Qwen3-VL's undistilled baseline of 36-40%.

The remaining gap comes from: representation differences (SpatialVLM's architecture was purpose-built for spatial measurement, while Qwen3-VL is a general-purpose VLM), training data coverage (the distillation dataset covers 50,000-100,000 images versus SpatialVLM's 10 million), and competing objectives (Qwen3-VL must retain aesthetic reasoning, language quality, and general intelligence alongside spatial precision).

The 60-75% accuracy target is acceptable because the coordinate-validation system handles precision-critical decisions. The VLM provides judgment, not measurement. Whether the VLM estimates a distance as 1.3m or 1.2m matters less than whether it understands that a coffee table belongs between the sofa and the TV.

## 3.4 Distillation Training Protocol

The distillation is lighter than what Qwen2.5-VL required because Qwen3-VL starts from a stronger spatial baseline.

Phase 1 — Spatial distillation (2-4 days on 4×A100). Qwen3-VL is fine-tuned via LoRA (rank 32, alpha 64) on spatial QA data generated by SpatialVLM. The model's native 3D grounding means it converges faster than Qwen2.5-VL did on the same data.

Phase 2 — Aesthetic and placement fine-tuning (3-5 days on 4×A100). The spatially-distilled model is further fine-tuned on placement judgment data from the 3D-FRONT dataset.

Optional Phase 3 — Human preference calibration (1-2 days on 2×A100). DPO on 5,000-10,000 human preference pairs.

Total training: 6-11 days, $2,500-4,500 compute. Slightly cheaper than the Qwen2.5-VL approach because Phase 1 converges faster.

---

# Chapter 4: The VLM in the Pipeline

## 4.1 Invocation Points

The VLM is called at four specific points in the pipeline, never in the per-frame loop.

Environment classification (once per scene): the VLM examines the photograph and determines what kind of environment it is — indoor room, urban street, natural terrain, coastal scene. This classification drives the shell extension strategy (room planes vs terrain generation vs building facades), the auto-population rules (furniture for rooms, street elements for urban, natural features for terrain), and the validation field parameters.

Aesthetic re-ranking of 20-50 candidates: approximately 200-500 milliseconds. Fast enough for interactive placement — the user sees the heatmap update within half a second of selecting a position. For the logit-scoring fast path (no generated explanation), latency is at the lower end. For the generative path with explanations, allow 2-5 seconds.

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

Aesthetic re-ranking of 20-50 candidates: approximately 200-500 milliseconds. Fast enough for interactive placement — the user sees the heatmap update within half a second of selecting a position.

Auto-furnishing plan: approximately 3-5 seconds for a full room plan (5-10 objects). Called once. The actual placement (validation field evaluation) is separate and fast.

Style matching: approximately 200-500 milliseconds per object. Amortized across the auto-furnishing sequence.

None of these are in the 60+ FPS render loop. The renderer never waits for the VLM. The VLM provides decisions that the renderer executes.

---

# Chapter 5: Training the VLM

## 5.1 What Gets Trained

Qwen3-VL's base weights are frozen. Only LoRA adapters are trained.

LoRA adapters (rank 32, alpha 64) are applied to the query, key, value, and output projection matrices of all attention layers. For Qwen3-VL-8B, this adds approximately 50-100 million trainable parameters — roughly 1% of the model's total. For the 30B-A3B MoE variant, LoRA is applied to the active expert layers.

No positional encoding modification is needed — Qwen3-VL's native Interleaved-MRoPE handles spatial encoding without external intervention.

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
| Spatial distillation | 2-4 days | 4×A100 | $1,000-2,000 |
| Placement fine-tuning | 3-5 days | 4×A100 | $1,500-2,500 |
| Human preference (optional) | 1-2 days | 2×A100 | $300-500 |
| **Total** | **6-11 days** | | **$2,500-4,500** |

Cheaper than the Qwen2.5-VL approach ($3,000-5,000) because Qwen3-VL's stronger spatial baseline means Phase 1 converges faster.

---

# Chapter 6: Why Not Fusion — The Decision Record

## 6.1 What Was Considered

The feature-level fusion architecture would have run SpatialVLM alongside Qwen at inference, with cross-attention bridges injecting SpatialVLM's spatial features into Qwen's processing at three layer points. Gated injection mechanisms would control how much spatial information influences Qwen's reasoning.

This approach was fully designed (9 chapters, 396 lines of theory) and evaluated against the distillation alternative.

## 6.2 The Comparison (Updated for Qwen3-VL)

| Factor | Feature Fusion (rejected) | Qwen2.5 + 3D-RoPE + Distillation (previous) | Qwen3-VL + Distillation (current) |
|---|---|---|---|
| Spatial accuracy | 92-95% of SpatialVLM | 85-90% of SpatialVLM | 60-75% of SpatialVLM (sufficient for judgment) |
| Inference latency | +0.5-1.0 seconds | Zero additional | Zero additional |
| GPU memory | +13 GB | ~32 GB (Qwen-32B) | ~16 GB (Qwen3-8B fp16) |
| Training cost | $15,000-30,000 | $3,000-5,000 | $2,500-4,500 |
| Training time | 15-22 days | 7-12 days | 6-11 days |
| Engineering complexity | Very high | Medium (3D-RoPE hack) | Low (no model modification) |
| Native 3D grounding | No | Via hack | Yes (built-in) |
| Thinking mode | No | No | Yes (Qwen3-VL-Thinking) |

## 6.3 Why the Architecture Simplified Over Time

Each iteration removed complexity while maintaining or improving capability:

Iteration 1 (Feature Fusion): three models at inference, cross-attention bridges, gated injection. Maximum accuracy but prohibitive cost and complexity.

Iteration 2 (Qwen2.5 + 3D-RoPE): one model at inference, custom positional encoding modification, heavy distillation. Good trade-off but required modifying Qwen's internals.

Iteration 3 (Qwen3-VL): one model at inference, no model modification, lighter distillation. The model's native capabilities absorbed what we previously had to build. This is the natural evolution — as foundation models improve, custom engineering becomes unnecessary.

## 6.4 When to Revisit

Revisit the VLM choice when: Qwen4-VL or a competitor achieves SpatialVLM-level spatial precision natively (eliminating the need for distillation entirely), or a unified perception+reasoning model (VGGT successor) replaces both the perception stack and the VLM in one model.

---

# Chapter 7: Evaluation

## 7.1 Evaluation Dimensions

The VLM is evaluated on five dimensions.

Spatial precision: does the model correctly estimate distances and spatial relationships? Measured against SpatialVLM's outputs on held-out data. Target: within 25-40% of SpatialVLM's accuracy (achieving 60-75% numerical accuracy, up from the 36-40% undistilled baseline).

Aesthetic quality: does the model rank placements in agreement with human preferences? Target: at least 80% agreement with human preferences.

Reasoning coherence: are the model's explanations logically consistent? Target: 90% rated as coherent and accurate.

Native 3D grounding benefit: does Qwen3-VL's native 3D grounding improve placement quality compared to Qwen2.5-VL without 3D-RoPE? Target: measurable improvement in spatial question accuracy.

End-to-end placement quality: does the VLM + validation system produce better placements than the validation system alone? Target: at least 15% improvement in human preference ratings.

## 7.2 The Key Ablation

The most important ablation compares three configurations:

Configuration A — Validation only (no VLM): the validation system's geometric aesthetic score selects the placement. No VLM involved. This is the baseline.

Configuration B — Qwen3-VL base (no distillation): Qwen3-VL-8B-Instruct with its native spatial capabilities but no SpatialVLM distillation. This measures what Qwen3-VL provides out of the box.

Configuration C — Qwen3-VL distilled (full system): Qwen3-VL-8B with LoRA distillation from SpatialVLM. This is the target architecture.

Expected results: C > B > A. The gap between A and B measures Qwen3-VL's native design intelligence. The gap between B and C measures the value of distillation. If B ≈ C, distillation isn't helping and Qwen3-VL's native spatial capability is already sufficient. If A ≈ B, the VLM isn't adding value beyond geometric aesthetics.

Optional Configuration D — Qwen3-VL-8B-Thinking: same as C but using the Thinking variant. This tests whether explicit chain-of-thought reasoning improves placement quality over direct inference.

---

# Chapter 8: Summary

## 8.1 The Architecture

One model: Qwen3-VL-8B-Instruct (or 30B-A3B for production quality).

Native spatial encoding: Qwen3-VL's Interleaved-MRoPE and DeepStack provide 3D grounding without custom positional encoding modifications. No 3D-RoPE hack needed.

SpatialVLM distillation: spatial measurement capability transferred via training data. Target: 60-75% numerical spatial accuracy (up from 36-40% undistilled baseline). Less training needed than Qwen2.5-VL because the starting point is higher.

Prompt-based 3D context: instead of injecting 3D coordinates into the model's internals, we provide explicit spatial data (distances, surface types, object positions from the cell grid) in the prompt. The VLM reasons about this data using its trained spatial understanding.

Role: design intelligence (style, culture, function, aesthetics) operating on top of the validation system's physics guarantees.

## 8.2 The Numbers

| Metric | Value |
|---|---|
| Base model | Qwen3-VL-8B-Instruct (primary) / 30B-A3B (production) |
| Trainable parameters | 50-100M LoRA (~1% of 8B model) |
| Custom model modifications | None (native spatial encoding) |
| Spatial precision (undistilled) | ~36-40% numerical accuracy |
| Spatial precision (distilled) | ~60-75% numerical accuracy |
| Training time | 6-11 days |
| Training cost | $2,500-4,500 |
| Inference latency | 200-500ms (re-ranking), 2-5s (full planning) |
| GPU memory | ~16 GB (8B fp16) / ~15-20 GB (30B-A3B MoE fp16, all params loaded, 3B active) |
| Thinking variant | Available for complex design reasoning |
| Aesthetic agreement with humans | Target 80%+ |

## 8.3 The Integration

The VLM sits between the validation system (which guarantees physics) and the user (who wants intelligent placement). The validation system produces the set of safe options. The VLM selects the best option. The coordinate-validation system can function without the VLM (geometric aesthetics only). The VLM cannot function without the validation system (no physics guarantees). This asymmetric dependency is intentional — physics must never depend on a neural network's judgment.

## 8.4 The Evolution

The VLM architecture has simplified through three iterations:

Iteration 1: Three-model feature fusion (SpatialVLM + Qwen2.5 + GLM-4.5V 3D-RoPE). Maximum capability, prohibitive complexity. Rejected.

Iteration 2: Qwen2.5-VL + custom 3D-RoPE injection + heavy SpatialVLM distillation. Good trade-off but required internal model modification. Implemented but superseded.

Iteration 3: Qwen3-VL with native 3D grounding + light distillation + prompt-based 3D context. Simplest architecture, no model modification, lower training cost. Current.

Each iteration absorbed more capability into the foundation model and removed more custom engineering. This is the intended trajectory — as foundation models improve, specialized hacks become unnecessary.
