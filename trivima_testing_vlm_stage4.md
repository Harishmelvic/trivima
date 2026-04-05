# Trivima Testing — Stage 4: VLM Intelligence
## Tests for Qwen2.5-VL + 3D-RoPE, Distillation, Aesthetic Re-Ranking, Auto-Furnishing

---

# Chapter 1: What These Tests Cover

Stage 4 adds design intelligence on top of the validated foundation (71/71 Stage 1-2 tests) and validation fields (Stage 2 unified foundation). The VLM provides judgment — style, culture, function, aesthetics — while the validation system continues to handle physics.

Items under test:

1. Qwen2.5-VL model loading and basic inference
2. 3D-RoPE positional encoding injection
3. SpatialVLM knowledge distillation quality
4. Aesthetic re-ranking of validation candidates
5. Auto-furnishing planning (gap detection + sequence)
6. Object style matching
7. End-to-end VLM + validation integration

## Dependencies

All Stage 1-2 tests must pass (71/71). Validation fields (surface, functional, BFS clearance) must be operational. The VLM module (`trivima/vlm/`) must be implemented.

## Test Data

Same 11-image test set from Stage 1-2. Additionally:

5 room images with known "good" furniture arrangements (from 3D-FRONT or manually curated) for aesthetic evaluation.

5 room images with deliberately missing furniture (sofa but no coffee table, bed but no nightstands) for auto-furnishing gap detection.

3 room images with strong style signatures (minimalist, rustic, mid-century modern) for style matching tests.

---

# Chapter 2: Model Loading and Basic Inference

## Test 2.1 — Qwen2.5-VL Loads Successfully

```
Load Qwen2.5-VL-32B with 8-bit quantization.

Verify: model loads without OOM
Verify: model responds to a basic text prompt ("Describe this image")
Verify: GPU memory usage < 35 GB (leaves room for other models)

Record: load_time, gpu_memory_peak
```

Pass: Model loads, responds coherently, fits in GPU memory.

## Test 2.2 — Sequential Model Loading

Verify that Qwen can coexist with the perception models via sequential loading.

```
Step 1: Load Depth Pro, run inference, record memory. Unload with empty_cache().
Step 2: Load SAM 3, run inference, record memory. Unload with empty_cache().
Step 3: Load Qwen2.5-VL, run inference, record memory.

Verify: no OOM at any step
Verify: Qwen has access to full GPU memory after perception models are unloaded
Record: peak memory at each step
```

Pass: All three models run sequentially without OOM on a 48GB GPU.

## Test 2.3 — Basic Room Description

Feed a room photograph to Qwen and ask it to describe what it sees.

```
Prompt: "Describe this room. What furniture is present? What style is it?"
Input: test room photograph

Verify: response mentions at least 3 objects visible in the room
Verify: response mentions a room type (living room, bedroom, etc.)
Verify: response is coherent English (not garbled tokens)
Verify: inference time < 10 seconds
```

Pass: Qwen produces a coherent room description that matches the actual room content. This is a sanity check, not a precision test.

## Test 2.4 — Batch Candidate Scoring

Test the fast re-ranking path — feeding multiple candidates in one prompt and getting logit-based scores.

```
Prompt: "Rate each placement position for a floor lamp on a scale of 1-10:
  Position A: 0.5m from sofa, 2m from window, against wall
  Position B: 3m from sofa, 0.5m from window, room center
  Position C: 0.2m from sofa, 4m from window, blocking doorway"

Verify: response provides ratings for all three positions
Verify: Position C scores lowest (blocking doorway is bad)
Verify: inference time < 3 seconds for 3 candidates
```

Pass: Qwen differentiates good from bad positions and the worst position (doorway blocker) scores lowest.

---

# Chapter 3: 3D-RoPE Integration

## Test 3.1 — 3D-RoPE Encoding Produces Valid Values

```
Create a synthetic point map: 576 patches (24×24 grid) with known 3D coordinates.
Coordinates range from (0,0,0) to (5,3,5) meters.

Compute 3D-RoPE encodings for all patches.

Verify: encoding dimensions are split equally (1/3 X, 1/3 Y, 1/3 Z)
Verify: no NaN or Inf in any encoding
Verify: encoding magnitude is bounded (not exploding)
Verify: encoding varies smoothly with position (nearby patches have similar encodings)
```

Pass: All encodings are valid, bounded, and spatially smooth.

## Test 3.2 — 3D-RoPE vs 2D-RoPE Attention Patterns

Compare attention patterns between standard 2D-RoPE and 3D-RoPE on a room image.

```
Run Qwen with 2D-RoPE on a room image. Extract attention weights from layer 16.
Run Qwen with 3D-RoPE on the same image. Extract attention weights from layer 16.

Select two patches that are:
  - Adjacent in pixels (10px apart) but far in 3D (one on table edge, one on far wall)

2D attention between them: record attention_2d
3D attention between them: record attention_3d

Verify: attention_3d < attention_2d
  (3D-RoPE should reduce attention between patches that are far in 3D space)

Select two patches that are:
  - Far in pixels (200px apart) but close in 3D (both on the same table surface)

2D attention between them: record attention_2d
3D attention between them: record attention_3d

Verify: attention_3d > attention_2d
  (3D-RoPE should increase attention between patches that are close in 3D space)
```

Pass: 3D-RoPE correctly reshapes attention patterns to reflect physical proximity rather than pixel proximity. This is the fundamental validation that 3D-RoPE works as intended.

## Test 3.3 — Confidence-Based Fallback to 2D

```
Create a point map where 80% of patches have high confidence (>0.7) and
20% have low confidence (<0.3, e.g., glass/mirror regions).

Compute 3D-RoPE with fallback: low-confidence patches use (u, v, 0) instead of (X, Y, Z).

Verify: high-confidence patches have 3D encodings (Z component varies)
Verify: low-confidence patches have 2D encodings (Z component = 0)
Verify: no discontinuity artifacts at the boundary between 3D and 2D regions
```

Pass: Fallback correctly applies 2D encoding to uncertain patches. The attention patterns for glass/mirror regions use pixel proximity (safe default) rather than incorrect 3D positions.

## Test 3.4 — 3D-RoPE Spatial Question Improvement

Test whether 3D-RoPE improves spatial reasoning compared to 2D-RoPE.

```
For 10 room images, ask Qwen spatial questions:
  "Which object is closest to the sofa?"
  "Is the table closer to the window or the door?"
  "Approximately how far is the chair from the wall?"

Run with 2D-RoPE: record correct_2d (out of 30 questions)
Run with 3D-RoPE: record correct_3d (out of 30 questions)

Verify: correct_3d >= correct_2d (3D-RoPE doesn't make spatial reasoning worse)
```

Pass: 3D-RoPE performs equal or better than 2D-RoPE on spatial questions. This is a directional test — if 3D-RoPE is somehow worse, the implementation has a bug.

---

# Chapter 4: SpatialVLM Distillation Quality

## Test 4.1 — Distillation Data Generation

Verify that SpatialVLM generates valid spatial QA pairs for training.

```
Run SpatialVLM on 100 room images.
For each, generate 10 spatial QA pairs:
  "How far is the [object A] from the [object B]?"
  "What is the height of the [surface]?"

Verify: all distances are positive and plausible (0.1m - 20m)
Verify: all heights are plausible (0m - 4m)
Verify: no NaN or error responses
Verify: at least 90% of images produce valid QA pairs (some may have too few objects)

Record: total pairs generated, time per image
```

Pass: At least 900 valid QA pairs from 100 images. SpatialVLM inference completes at ~1 second per image.

## Test 4.2 — Distilled Model Spatial Accuracy

After distillation training, test the student model (Qwen + LoRA) against SpatialVLM's answers.

```
Hold out 500 spatial QA pairs that were NOT in the training set.

Run distilled Qwen on these 500 questions.
Compare Qwen's numerical answers to SpatialVLM's answers.

Compute: mean_relative_error = mean(|qwen_answer - spatial_answer| / spatial_answer)

Verify: mean_relative_error < 0.25 (within 25% of SpatialVLM's answers)
  Target: 0.10-0.15 (85-90% of SpatialVLM's accuracy)
```

Pass: Distilled Qwen achieves within 25% of SpatialVLM's spatial precision on held-out data. The 85-90% target is aspirational — even 70-75% (25% error) is acceptable because the validation system handles measurement and the VLM handles judgment.

## Test 4.3 — Distillation Doesn't Degrade Aesthetics

Verify the canary metrics — distillation training should not harm Qwen's existing capabilities.

```
Before distillation: run Qwen on 200 aesthetic evaluation prompts.
  Record baseline_aesthetic_score.

After distillation: run Qwen on the same 200 prompts.
  Record distilled_aesthetic_score.

Verify: distilled_aesthetic_score >= 0.85 × baseline_aesthetic_score
  (less than 15% degradation in aesthetic judgment)
```

Pass: Aesthetic capability preserved within 15% of baseline. If degradation exceeds 15%, the distillation learning rate was too high or the spatial data overwhelmed the aesthetic capability.

## Test 4.4 — Distillation Doesn't Degrade Language

```
Before distillation: compute perplexity on 100 explanation prompts.
After distillation: compute perplexity on the same prompts.

Verify: perplexity increase < 15%
```

Pass: Language fluency preserved. If perplexity increases significantly, the LoRA rank may need reduction or the training data needs more diversity.

---

# Chapter 5: Aesthetic Re-Ranking

## Test 5.1 — Re-Ranking Differentiates Good from Bad

```
Build a cell grid from a test image.
Generate 20 candidate positions from the validation fields (all physically valid).

Manually classify 5 as "clearly good" (well-spaced, near relevant anchors, balanced).
Manually classify 5 as "clearly bad" (cramped, awkward, unbalanced — but physically valid).
The remaining 10 are "neutral."

Run aesthetic re-ranking on all 20.

Verify: mean rank of "clearly good" positions < mean rank of "clearly bad" positions
Verify: at least 3 of the top 5 ranked positions are from the "clearly good" set
```

Pass: The VLM correctly identifies the obviously good positions as better than the obviously bad ones. This is a coarse test — the VLM should at minimum get the easy cases right.

## Test 5.2 — Re-Ranking Consistency

Run the same re-ranking query 5 times (same image, same candidates).

```
Record the top-3 positions from each run.

Verify: the top-1 position is the same in at least 4 out of 5 runs
Verify: the top-3 set overlaps by at least 2 positions across all runs
```

Pass: Re-ranking is reasonably deterministic. Some variation is expected (temperature > 0), but the top recommendation should be stable.

## Test 5.3 — Re-Ranking Speed (Fast Path)

Test the logit-scoring fast path for interactive heatmap updates.

```
Prepare 20 candidate positions formatted as a single prompt.
Run logit-based scoring (no generation, just forward pass + read logits).

Record: inference_time

Verify: inference_time < 500ms on A40/A100
```

Pass: Fast re-ranking completes within 500ms. The VLM doc claimed 50-100ms which was optimistic — 200-500ms is realistic for Qwen-32B at 8-bit. Under 500ms is acceptable for interactive use (the heatmap updates within half a second of user action).

## Test 5.4 — Re-Ranking Speed (Full Path)

Test the generative path with explanations.

```
Prepare 5 candidate positions.
Prompt: "Rank these 5 positions for a floor lamp and explain your reasoning for the top choice."

Record: inference_time, output_length

Verify: inference_time < 5 seconds
Verify: explanation is coherent and references spatial/aesthetic factors
```

Pass: Full evaluation with explanation completes in under 5 seconds. Explanation quality is assessed by human review.

## Test 5.5 — Re-Ranking Respects Validation Scores

Verify that the VLM doesn't override physics — positions with high validation scores should generally rank higher than positions with low validation scores.

```
Generate 20 candidates with known validation scores.
10 have high composite validation score (>0.8).
10 have low composite validation score (0.3-0.5, but still physically valid).

Run aesthetic re-ranking.

Verify: at least 7 of the top 10 VLM-ranked positions come from the high-validation set
```

Pass: The VLM generally agrees with the validation system on what constitutes a good position. Occasional disagreements are acceptable (the VLM might prefer a position with slightly lower clearance because the aesthetic is better), but systematic disagreement suggests the VLM is ignoring spatial quality.

---

# Chapter 6: Auto-Furnishing

## Test 6.1 — Gap Detection in Empty Room

Provide an image of an empty room (only walls, floor, ceiling — no furniture).

```
Run auto-furnishing gap detection.

Verify: the VLM identifies at least 3 missing furniture categories
  (e.g., "this living room needs a sofa, coffee table, and floor lamp")
Verify: identified categories are appropriate for the room type
  (not suggesting a bed for a living room)
Verify: no crash or empty response
```

Pass: VLM identifies reasonable furniture gaps for the room type.

## Test 6.2 — Gap Detection in Partially Furnished Room

Provide an image of a room with some furniture (sofa present but no coffee table, no lamps).

```
Run auto-furnishing gap detection.

Verify: the VLM identifies the MISSING items (coffee table, lamp) 
Verify: the VLM does NOT suggest items that already exist (sofa)
Verify: suggestions are functionally relevant (coffee table near sofa, lamp near seating)
```

Pass: VLM correctly identifies what's missing without duplicating what's present.

## Test 6.3 — Placement Sequence Order

Test the VLM's ability to plan a placement sequence (anchor pieces first).

```
Give the VLM an empty living room and ask for a furnishing plan.

Verify: large anchor pieces are placed first (sofa before coffee table)
Verify: dependent pieces follow their anchors (coffee table after sofa)
Verify: accent pieces come last (plants, lamps after major furniture)
Verify: the sequence is at least 3 items long
```

Pass: The placement order follows the anchor → dependent → accent pattern. This ensures that each placement has the right context (the coffee table can be placed relative to the sofa because the sofa already exists).

## Test 6.4 — Sequential Placement Updates Fields

Test that the validation fields update correctly after each auto-furnished object.

```
Start with an empty room cell grid.
Auto-furnish step 1: place a sofa.

Verify: sofa cells added to the grid
Verify: collision field updated (BFS clearance now reflects sofa)
Verify: functional field updated (seating now exists for lamp queries)

Auto-furnish step 2: place a coffee table.

Verify: coffee table cells added
Verify: collision field updated (can't place anything where coffee table is)
Verify: the coffee table position is near the sofa (functional field influenced placement)
```

Pass: Each placement correctly updates the cell grid and validation fields for the next placement.

## Test 6.5 — Auto-Furnishing Doesn't Overcrowd

```
Run auto-furnishing on a small room (3m × 3m).

Verify: the VLM stops suggesting items when the room is sufficiently furnished
Verify: total number of placed items is reasonable (3-8 for a small room)
Verify: no placed item has a collision with another placed item
  (validation system prevents this, but verify)
Verify: pathways remain clear (at least 60cm between furniture for walking)
```

Pass: Auto-furnishing produces a reasonable density of furniture without overcrowding. Collision-free is guaranteed by the validation system. Pathway clearance is checked via BFS queries.

## Test 6.6 — Auto-Furnishing Performance

```
Run full auto-furnishing on a room (gap detection + 5 object placements).
Record total time.

Verify: total_time < 30 seconds for 5 objects
  (~3-5s for gap detection + ~5s per placement for validation + VLM re-ranking)
```

Pass: Full room furnishing in under 30 seconds. This is acceptable for a background process — the user can navigate while auto-furnishing runs.

---

# Chapter 7: Style Matching

## Test 7.1 — Style Compatibility Scoring

```
Present the VLM with a room image (e.g., minimalist Scandinavian living room) and 
three furniture items:
  A: Scandinavian wood side table (matching style)
  B: Victorian ornate chair (clashing style)
  C: Modern metal floor lamp (compatible style)

Prompt: "Rate the style compatibility of each item with this room (1-10)."

Verify: item A score > item B score (matching > clashing)
Verify: item C score > item B score (compatible > clashing)
Verify: item A score >= item C score (matching >= compatible)
```

Pass: VLM correctly ranks style compatibility. The obvious clash (Victorian in Scandinavian) scores lowest.

## Test 7.2 — Style Matching Across Room Types

```
Test style matching on 3 different room styles:
  Minimalist room + modern lamp → should score high
  Rustic room + chrome lamp → should score low
  Mid-century room + Eames-style chair → should score high

Verify: all 3 assessments match expected compatibility
```

Pass: VLM's style sense generalizes across different room aesthetics.

## Test 7.3 — Style Matching Returns Alternatives

```
Present a mismatched item (chrome lamp in rustic cabin).

Prompt: "This item doesn't match the room style. Suggest what style of lamp 
would be more appropriate."

Verify: VLM suggests an alternative style (e.g., "wrought iron" or "wooden base")
Verify: suggestion is coherent with the room's aesthetic
Verify: no crash or empty response
```

Pass: VLM provides meaningful style alternatives when a match is poor.

---

# Chapter 8: Integration Tests

## Test 8.1 — Full Pipeline: Photo to Furnished Room

The end-to-end test for Stage 4.

```
Input: a single room photograph (unfurnished or partially furnished)

Step 1: Perception pipeline → cell grid (Stage 1-2, already tested)
Step 2: Validation fields computed (Stage 2 unified, already tested)
Step 3: VLM environment classification → room type identified
Step 4: VLM gap detection → missing furniture list
Step 5: For each missing item:
  5a: Validation system generates 20 candidate positions
  5b: VLM re-ranks candidates
  5c: Top position selected
  5d: Object cells inserted into grid
  5e: Validation fields updated
Step 6: Final room has all placed objects

Verify: pipeline completes without crash
Verify: at least 3 objects placed
Verify: no collisions between placed objects
Verify: all placed objects are on valid surfaces
Verify: room looks reasonably furnished (human judgment)
Verify: total time < 60 seconds

Record: room_type, objects_placed, positions, total_time
```

Pass: Full auto-furnishing pipeline runs end-to-end. Room appears reasonably furnished. All physics constraints satisfied.

## Test 8.2 — VLM Adds Value Over Validation-Only

Compare placement quality with and without VLM re-ranking.

```
For 5 test rooms:
  Run placement with validation-only (geometric aesthetic score selects position)
  Run placement with VLM re-ranking (VLM selects from validation candidates)

Human evaluator rates both arrangements (blind, randomized order) on a 1-5 scale.

Verify: mean VLM-ranked rating > mean validation-only rating
Verify: improvement is at least 0.5 points on the 5-point scale
```

Pass: VLM re-ranking produces measurably better placements than geometric scoring alone. This is the key value proposition — if the VLM doesn't improve placement quality, it's not worth the complexity.

## Test 8.3 — VLM Failure Graceful Degradation

Test what happens when the VLM fails or times out.

```
Simulate VLM failure:
  1. VLM returns empty response
  2. VLM inference exceeds 30-second timeout
  3. VLM returns garbled/incoherent output

For each failure mode:
  Verify: system falls back to validation-only placement (geometric aesthetic score)
  Verify: no crash
  Verify: placement still occurs (VLM is advisory, not required)
  Verify: user is notified of degraded mode (if applicable)
```

Pass: VLM failures degrade gracefully to validation-only mode. The system never depends on VLM availability for basic functionality.

---

# Chapter 9: Test Execution

## 9.1 Test Order

```
PHASE 1 — Model Loading (must pass first, 5 minutes):
  [CRITICAL] 2.1  Qwen loads successfully
  [CRITICAL] 2.2  Sequential model loading (no OOM)
             2.3  Basic room description
             2.4  Batch candidate scoring

PHASE 2 — 3D-RoPE (must pass before distillation, 10 minutes):
  [CRITICAL] 3.1  3D-RoPE encoding valid
  [CRITICAL] 3.2  3D vs 2D attention patterns
             3.3  Confidence-based fallback
             3.4  Spatial question improvement

PHASE 3 — Distillation Quality (must pass before re-ranking, 15 minutes):
  [CRITICAL] 4.1  Distillation data generation
  [CRITICAL] 4.2  Distilled model spatial accuracy
  [CRITICAL] 4.3  Aesthetics not degraded
             4.4  Language not degraded

PHASE 4 — Aesthetic Re-Ranking (10 minutes):
  [CRITICAL] 5.1  Differentiates good from bad
             5.2  Re-ranking consistency
  [CRITICAL] 5.3  Re-ranking speed (fast path < 500ms)
             5.4  Re-ranking speed (full path < 5s)
             5.5  Respects validation scores

PHASE 5 — Auto-Furnishing (10 minutes):
  [CRITICAL] 6.1  Gap detection in empty room
  [CRITICAL] 6.2  Gap detection in partial room
             6.3  Placement sequence order
             6.4  Sequential placement updates fields
             6.5  Doesn't overcrowd
             6.6  Performance (< 30s for 5 objects)

PHASE 6 — Style Matching (5 minutes):
             7.1  Style compatibility scoring
             7.2  Style matching across room types
             7.3  Returns alternatives

PHASE 7 — Integration (15 minutes):
  [CRITICAL] 8.1  Full pipeline photo to furnished room
  [CRITICAL] 8.2  VLM adds value over validation-only
  [CRITICAL] 8.3  VLM failure graceful degradation
```

## 9.2 Summary

| Phase | Tests | Critical | Time |
|---|---|---|---|
| Model loading | 4 | 2 | 5 min |
| 3D-RoPE | 4 | 2 | 10 min |
| Distillation | 4 | 3 | 15 min |
| Re-ranking | 5 | 2 | 10 min |
| Auto-furnishing | 6 | 2 | 10 min |
| Style matching | 3 | 0 | 5 min |
| Integration | 3 | 3 | 15 min |
| **Total** | **29** | **14** | **~70 min** |

## 9.3 What Failure Means

If model loading fails (Phase 1): Qwen doesn't fit on the GPU, or quantization broke something. Try 4-bit quantization. If still OOM, need a larger GPU or a smaller model (Qwen-14B as fallback).

If 3D-RoPE fails (Phase 2): the positional encoding modification is incorrect. Check the coordinate normalization, the dimension allocation (1/3 X, 1/3 Y, 1/3 Z), and the trigonometric computation. Compare against GLM-4.5V's reference implementation.

If distillation fails (Phase 3): either the training data is poor (SpatialVLM gave bad answers), the training diverged (learning rate too high), or catastrophic interference destroyed existing capabilities. Check canary metrics. Reduce LoRA rank or learning rate. Increase data diversity.

If re-ranking fails (Phase 4): the VLM isn't using spatial or aesthetic information to differentiate positions. Check that the prompt format includes enough context (room description, position coordinates, nearby objects). The VLM needs sufficient information to make a judgment — a bare coordinate without context gives it nothing to reason about.

If auto-furnishing fails (Phase 5): gap detection might be identifying too many or too few items. The VLM needs to see the room context (what's already there) to identify what's missing. If it suggests items that already exist, the prompt isn't showing existing furniture clearly enough.

If style matching fails (Phase 6): not critical for the prototype. Style matching is a refinement — the system works without it (just uses the first matching item from the database regardless of style).

If integration fails (Phase 7): individual components work but the pipeline doesn't. Check the data flow between validation field output and VLM input — format mismatches are the most common cause. The graceful degradation test (8.3) is particularly important — the system must never crash because the VLM had a bad day.

## 9.4 Pre-Requisites

Before running these tests:

1. Qwen2.5-VL-32B checkpoint downloaded (HuggingFace, ~16GB at 8-bit)
2. SpatialVLM checkpoint downloaded (for distillation data generation)
3. LoRA training completed (Phases 1-2 of the VLM training protocol, 7-12 days)
4. 3D-RoPE implementation complete in `trivima/vlm/qwen_vlm.py`
5. All 71 Stage 1-2 tests passing
6. Validation fields operational (surface, functional, BFS clearance)

Tests 2.1-2.4 and 3.1-3.4 can run with the base Qwen model (no distillation needed). Tests 4.1-4.4 require distillation training to be complete. Tests 5.1+ require both distillation and the aesthetic re-ranking API to be implemented.
