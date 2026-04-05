# Trivima Architecture

## What It Does

Trivima turns a single photograph into a navigable, photorealistic 3D environment. Upload a room photo → walk around in it within seconds.

The system is built on a **Differential-Integral Cell Architecture** — one universal data primitive that replaces surfels, octrees, shadow maps, bump maps, LOD hierarchies, and 7+ other traditional rendering systems.

---

## The Pipeline

```
Photo ──→ Perception ──→ Cell Grid ──→ AI Texturing ──→ Render ──→ Navigate
            (1.5s)*       (0.5s)        (5-8ms/frame)   (1ms)     (60+ FPS)

* First run: ~20s (model loading). Subsequent runs with models cached: ~1.5s.
```

**Stage 1 — Perception:** Depth Pro extracts metric depth. SAM 3 (facebook/sam3, 840M params, text-prompted concept segmentation) provides semantic labels for 29 indoor concepts. Bilateral smoothing (σ=2.5) cleans depth noise. Scale calibration from detected doors corrects systematic error. Failure modes (glass, mirror, dark, sky, specular) detected and mitigated via per-pixel confidence.

**Stage 2 — Cell Grid Construction:** The point cloud is binned into 5cm cells. 5×5 Sobel gradients computed (not simple finite differences). Per-cell confidence from depth smoothness × point density × semantic penalty (multiplicative formula). Result: ~80K cells, ~40MB.

**Stage 3 — Shell Extension:** RANSAC detects floor/wall/ceiling planes. New cells generated along extended planes. Navigation starts at 1.5s with flat shading; shell completes in background at 10-15s.

**Stage 4 — AI Texturing:** Pix2PixHD-Lite GAN (25M params, real-time) or StreamDiffusion + ControlNet (production quality). Light values written back to cells via cell ID buffer with view-angle weighting. Per-cell temporal blending prevents flicker. Only dirty cells (10-30%) re-lit each frame. Requires pre-training on ~200K cell-buffer/photo pairs (~48h on 8×A100).

**Stage 5 — Rendering:** Instanced cubes via ModernGL. Adaptive LOD with subdivision cap (1 level single-image, 3 multi-image). Cell-based collision with confidence-expanded margins. Floor following with sub-cell gradient precision.

**Stage 6 — Validation:** Conservation checks (energy, mass, shadow direction) run async, one frame behind. Corrections applied gradually over 3-5 frames.

**Stage 7 — Placement (Stage 3+):** Validation fields (surface, functional, BFS clearance) produce composite scores. Placement heatmap overlaid on 3D view.

**Stage 8 — VLM Intelligence (Stage 4+):** Qwen3-VL-8B with native 3D grounding re-ranks candidates, plans auto-furnishing, matches object styles via prompt-based 3D context. Never in the render loop.

---

## The Cell — One Primitive To Rule Them All

Every cell stores 512 bytes split into two cache-aligned structs:

### CellGeo (64 bytes) — HOT, read every frame
```
density + gradient + integral     →  collision detection
normal + gradient                 →  surface orientation, bump detail
cell_type                         →  empty / surface / solid / transparent
confidence                        →  [0,1] reliability (glass/mirror → low)
collision_margin                  →  expanded margin for uncertain cells
```

### CellVisual (448 bytes) — COLD, read during rendering
```
normal curvature                  →  2nd derivative for sharp edge detection
albedo + gradient + 2nd deriv     →  IS the texture (no separate texture maps)
neural texture features [32]      →  fine detail beyond what gradients capture
light + gradient + temporal deriv →  IS the shadow map (no separate shadow maps)
material (label, roughness, refl) →  semantic classification
animation (velocity, acceleration)→  character motion
6 neighbor summaries + flux       →  local spatial queries without global search
```

### Why This Works

| Traditional System | Cell Equivalent | How |
|---|---|---|
| Bump map | Normal gradient | `∇normal` IS the surface curvature |
| Texture map | Albedo gradient | `∇albedo` IS the texture grain direction |
| Displacement map | Density gradient | `∇density` gives sub-cell surface position |
| Shadow map | Light gradient | `∇light` IS the shadow edge |
| LOD hierarchy | Taylor expansion | `child = parent + ∇ · offset + ½ ∇² · offset²` |
| Conservation check | Integrals | `∫density` = total mass, must be constant |

---

## Project Structure

```
trivima/
├── src/core/                        C++/CUDA performance-critical engine
│   ├── include/trivima/
│   │   ├── types.h                  vec3, CellKey, CellType, spatial hash
│   │   ├── cell.h                   CellGeo (64B) + CellVisual (448B)
│   │   ├── cell_grid.h             Sparse grid: hash map → parallel SoA arrays
│   │   └── serialization.h         Binary format: 32B header + 528B/cell
│   └── src/
│       ├── gradients.cu             5×5 Sobel gradient kernels
│       ├── hierarchy.cu             Subdivision (Taylor) + merge (integral avg)
│       ├── point_to_cell.cu         Point cloud → cell grid conversion
│       ├── shell_extension.cu       Room completion along detected planes
│       └── buffer_renderer.cu       Cell grid → 2D buffers + cell ID for AI model
│
├── src/bindings/                    nanobind Python wrappers
│
├── trivima/                         Python orchestration layer
│   ├── perception/
│   │   ├── depth_pro.py             Depth Pro + auto cuDNN disable
│   │   ├── sam.py                   SAM 3 (HF) → SAM 2.1 (Ultralytics) → Grounded SAM 2
│   │   ├── depth_smoothing.py       Bilateral filter (σ=2.5, joint with RGB guide)
│   │   ├── scale_calibration.py     Door/furniture detection for metric scale
│   │   ├── failure_modes.py         Mirror/glass/dark/sky/specular detection + mitigation
│   │   └── pipeline.py              Orchestrates all models with confidence propagation
│   │
│   ├── construction/
│   │   └── point_to_grid.py         5×5 Sobel gradients, multiplicative confidence, density forcing
│   │
│   ├── rendering/
│   │   ├── lod.py                   Subdivision cap: 1 (single), 3 (multi), 4 (video)
│   │   └── shaders/
│   │       ├── cell_cube.vert       Instanced cube + confidence to fragment
│   │       └── cell_cube.frag       Directional light + debug orange for low-conf cells
│   │
│   ├── navigation/
│   │   └── collision.py             Cell lookup + sub-cell precision + BFS clearance query
│   │
│   ├── texturing/
│   │   ├── buffer_renderer.py       5 buffers (albedo/depth/normal/label/cellID)
│   │   ├── cell_writeback.py        View-angle weighted + confidence-boosted AI write-back
│   │   ├── models/
│   │   │   ├── pix2pix_lite.py      Real-time GAN: 8ch→3ch, U-Net, ~25M params
│   │   │   └── controlnet_adapter.py SD Turbo + ControlNet + StreamDiffusion
│   │   ├── temporal.py              Per-cell blending (NOT screen-space EMA)
│   │   ├── inference_engine.py      Runtime coordinator with confidence-aware pipeline
│   │   └── data_pipeline.py         ScanNet/Matterport3D training pair generation
│   │
│   ├── validation/
│   │   ├── surface_field.py         Floor + elevated surface detection, confidence-weighted
│   │   ├── functional_field.py      12 object categories × semantic distance rules
│   │   ├── conservation.py          Energy, mass, shadow conservation checks
│   │   └── validator.py             Orchestrator + gradual correction + error injection
│   │
│   ├── vlm/
│   │   ├── qwen_vlm.py             Qwen3-VL-8B + SpatialContextBuilder
│   │   ├── aesthetic_ranker.py      Logit scoring (~200-500ms) + generative (~2-5s)
│   │   ├── auto_furnish.py          Gap detection + placement planning + rule-based fallback
│   │   └── training/               SpatialVLM distillation + LoRA fine-tuning
│   │
│   ├── testing/
│   │   ├── benchmark.py             FPS/SSIM/LPIPS/flicker/collision/memory
│   │   └── visual_comparison.py     Side-by-side: flat | AI | ground truth
│   │
│   └── app.py                       CLI entry point with --stats, --export-ply, --render-preview
│
├── handler.py                       RunPod serverless handler
├── distillation_test.py             Small-scale distillation validation (~$5, ~10 min)
├── Dockerfile                       Pre-built image for serverless deployment
│
├── tests/
│   ├── test_stage2.py               29 tests: cell struct + perception + grid + shell + LOD
│   ├── test_image_based.py          13 tests: 11 synthetic images + 4 GPU model tests
│   ├── test_unified_foundation.py   29 tests: confidence + surface + functional + BFS + conservation
│   ├── test_vlm_stage4.py           25 tests: model loading + spatial context + distillation + ranking
│   └── generate_test_images.py      11 synthetic rooms with GT depth/labels/failure modes
│
└── data/
    ├── test_images/                 11 synthetic test images with ground truth
    ├── sample_images/
    └── checkpoints/                 Model weights (gitignored)
```

---

## Key Design Decisions

### 1. Structure-of-Arrays (SoA) over Array-of-Structures (AoS)

Cells are split into `CellGeo` (64B, hot) and `CellVisual` (448B, cold) stored in separate contiguous arrays. During collision checks (every frame, every movement step), only the 64B geo array is touched — **8× better cache utilization** than reading 512B per cell.

### 2. Implicit Hierarchy (No Pointers in Cells)

Children of cell `(level, x, y, z)` are at `(level-1, 2x+dx, 2y+dy, 2z+dz)`. Parent is at `(level+1, x/2, y/2, z/2)`. No pointers stored — zero cost, GPU-friendly, and avoids pointer-chasing on CUDA.

### 3. Spatial Hash Map (Not Octree)

`cuco::static_map<uint64_t, uint32_t>` on GPU for O(1) parallel lookup. `std::unordered_map<CellKey, uint32_t>` on CPU. Octrees cause pointer-chasing on GPU. Flat arrays waste memory on empty space.

### 4. Per-Cell Temporal Blending (Not Screen-Space EMA)

AI texturing output is blended in 3D cell space, not 2D screen space. Cells have stable 3D positions — no parallax misalignment, no ghosting on fast camera motion. Alpha is based on view angle change: `α = clamp(1 - dot(old_view, new_view), 0.1, 1.0)`.

### 5. Dual-Path AI Texturing

- **Real-time:** Pix2PixHD-Lite GAN (25M params, <10ms, 100+ FPS)
- **Production:** StreamDiffusion + ControlNet (15-50ms, higher quality)

Fallback: if AI model is too slow, reuse previous frame's cell light values.

### 6. Cell ID Buffer for Write-Back

The renderer outputs a cell ID buffer alongside the color buffer. Each pixel stores which cell it corresponds to. After AI texturing, the output is mapped back to cells via this ID buffer, weighted by `max(dot(view_dir, cell_normal), 0.01)` — face-on pixels contribute more. Low-confidence cells get boosted AI weight (their gradient-based shading is unreliable).

### 7. Bilateral Depth Smoothing + 5×5 Sobel Gradients

Two separate smoothing concerns:
- **Depth smoothing** (bilateral, σ=2.5): mild, preserves surface detail like brick mortar
- **Gradient smoothing** (5×5 Sobel kernel): smooth during differentiation, not before

Combined, they reduce gradient noise by 40-65% without erasing fine textures.

### 8. Gradual Conservation Corrections

Corrections from validation are spread over 3-5 frames instead of applied instantly, preventing visible artifacts. Validation runs async, one frame behind rendering.

### 9. Per-Cell Confidence

`confidence = depth_smoothness × point_density × semantic_penalty`

Semantic penalties: glass (×0.2), mirror (×0.1), transparent (×0.3), specular (×0.4), dark (×0.4).

Confidence drives: collision margins (expanded for low-conf), subdivision (blocked < 0.5), AI texturing weight (boosted for low-conf), debug rendering (orange tint).

### 10. Subdivision Depth Cap

Taylor expansion child prediction error is ±1.25cm from single-image input.

| Input Type | Max Subdivisions | Finest Cell |
|---|---|---|
| Single image | 1 level | 2.5 cm |
| Multi-image | 3 levels | 0.6 cm |
| Video | 4 levels | 0.3 cm |

Low-confidence cells are never subdivided.

### 11. Failure Mode Mitigations

| Surface | Problem | Mitigation |
|---|---|---|
| Mirror | Phantom room | density=1.0, confidence=0.1, block subdivision |
| Glass | Invisible | density=1.0 (barrier), confidence=0.2 |
| Transparent | Refraction | confidence=0.3, expand collision margin |
| Dark scene | Noise | All cells low confidence, rely on AI texturing |
| Sky | Extreme depth | Exclude from grid, render as skybox |
| Specular | Oscillating | confidence=0.4, extra depth smoothing |

### 12. VLM Design Intelligence (Stage 4)

**Model:** Qwen3-VL-8B-Instruct (validated) / 30B-A3B on A100 80GB.

**Native spatial encoding:** Qwen3-VL has Interleaved-MRoPE + DeepStack for native 3D grounding. No custom 3D-RoPE injection — dropped in favor of prompt-based 3D context from the cell grid.

**Distillation results (validated):**

| Metric | Before LoRA | After LoRA (2,898 pairs) |
|---|---|---|
| Within 25% spatial accuracy | 44.4% | **88.0%** |
| Within 50% spatial accuracy | 68.9% | **92.0%** |
| Median spatial error | 31.7% | **8.3%** |
| Aesthetic canary | 1.00 | **1.00** (zero degradation) |
| Training time | — | 4.8 minutes |

88% accuracy on 2,898 synthetic pairs exceeds the 60-75% target. Full 50-100K SpatialVLM distillation will push higher.

**30B-A3B comparison:** On A40 48GB, 30B-A3B uses 40.3GB with CPU offloading, performing worse than 8B (34.7% vs 44.4% within 25%). On A100 80GB without offloading, 30B-A3B is expected to outperform 8B for aesthetic reasoning.

**Invocation points** (never in the render loop):
- Environment classification (~2s, once per scene)
- Aesthetic re-ranking: fast (~200-500ms) or full (~2-5s with explanations)
- Auto-furnishing planning (~3-5s, once per scene)
- Object style matching (~200-500ms per object)

**Training cost:** $2,500-4,500 total (spatial distillation 2-4 days + aesthetic 3-5 days + optional DPO 1-2 days).

**Architecture evolution:** Fusion (3 models) → Qwen2.5 + 3D-RoPE hack → Qwen3-VL native. Each iteration removed complexity as foundation models improved.

---

## Data Flow

```
                                ┌─────────────────────┐
  Photo ──→ Depth Pro ──→ Depth │ Bilateral Smoothing  │──→ Point Cloud
         ──→ SAM 3 ──→ Labels   │ Scale Calibration    │     ↓
                                └─────────────────────┘  Binning + 5×5 Sobel
                                                           ↓
                                                     ┌──────────┐
                                                     │ Cell Grid │
                                                     │ (sparse)  │
                                                     └─────┬────┘
                                                           │
                              ┌─── Navigation starts (1.5s) ───┐
                              │                                 │
                              ↓                                 ↓
                        Render visible cells          Shell Extension
                        (flat shading, LOD)           (RANSAC, bg 10-15s)
                              │                            │
                              ↓                            ↓
                        Buffer Renderer ←──────── Complete Grid
                        (5ch + cell ID)
                              ↓
                        AI Texturing (GAN or CN)
                              ↓
                        Cell Write-Back (view-angle weighted)
                              ↓
                        Temporal Blend (per-cell)
                              ↓
                        Conservation Validation (async)
                              ↓
                        Display @ 60+ FPS
                              │
                        ┌─────┴──────┐
                        │  Placement  │  (Stage 3+)
                        │  Heatmap    │
                        └─────┬──────┘
                              ↓
                        ┌─────────────┐
                        │  VLM Re-rank │  (Stage 4)
                        │  (Qwen3-VL)  │
                        └─────────────┘
```

---

## Hardware Requirements

### Minimum (Development + Testing)
| Component | Spec | Cost |
|---|---|---|
| GPU | NVIDIA A40 48GB | $0.39/hr RunPod |
| Container disk | 100 GB | included |
| Volume | 150 GB | $0.07/hr |
| RAM | 32 GB | included |

Runs: Depth Pro + SAM 3 + Qwen3-VL-8B (sequential), cell grid, all tests.

### Recommended (Production)
| Component | Spec | Cost |
|---|---|---|
| GPU | NVIDIA A100 80GB SXM | $1.39/hr RunPod |
| Container disk | 200 GB | included |
| Volume | 300 GB | $0.07/hr |
| RAM | 64 GB | included |

Runs: All models with headroom, 30B-A3B without CPU offloading, production AI texturing, full distillation training.

### Premium (Maximum Quality + Training)
| Component | Spec | Cost |
|---|---|---|
| GPU | NVIDIA H100 80GB SXM | $2.69/hr RunPod |
| Container disk | 200 GB | included |
| Volume | 500 GB | $0.07/hr |
| RAM | 128 GB | included |

Runs: Everything at maximum speed, 30B-A3B + perception models simultaneously, cinematic rendering, fastest training convergence.

### Multi-GPU Training
| Component | Spec | Cost |
|---|---|---|
| GPUs | 4× A100 80GB SXM | $5.56/hr RunPod |
| Full distillation training | 6-11 days | $2,500-4,500 |

### Serverless (Inference Only)
| Component | Spec | Cost |
|---|---|---|
| GPU | 48GB (A40/A6000, high supply) | $0.00034/s |
| Idle | $0.00/hr | — |
| Per request | ~$0.01-0.03 | — |
| Cold start | ~15-30s (image cached) | — |

---

## Resolution Tiers

| Tier | Cell Size | Cells/m³ | Distance | Use Case |
|------|-----------|----------|----------|----------|
| 0 | 20 cm | 125 | >15m | Distant background |
| 1 | 5 cm | 8,000 | 3-15m | Navigation |
| 2 | 2 cm | 125,000 | <3m | Close inspection |
| 3 | 1 cm | 1,000,000 | Production | High-detail render |
| 4 | 0.5 cm | 8,000,000 | Cinematic | Extreme close-ups |

---

## Memory Budget (Typical Room)

| Mode | Cells | Memory | Min GPU |
|------|-------|--------|---------|
| Real-time navigation | ~30K visible | 15-60 MB | A40 48GB |
| Production render | ~7.5M at Tier 3 | 1-4 GB | A100 80GB |
| Cinematic render | ~120M at Tier 4 | 4-15 GB (streamed) | H100 80GB |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Core engine | C++17 / CUDA 12.1 |
| GPU hash map | NVIDIA cuCollections (`cuco::static_map`) |
| Python bindings | nanobind 2.0+ |
| Build system | CMake 3.18 + scikit-build-core |
| ML framework | PyTorch 2.4+ |
| Depth estimation | Apple Depth Pro (1.9GB) |
| Segmentation | SAM 3 (facebook/sam3, 840M params) |
| Rendering | ModernGL 5.12 + GLFW |
| AI texturing (realtime) | Pix2PixHD-Lite (custom, 25M params) |
| AI texturing (production) | StreamDiffusion + ControlNet |
| VLM | Qwen3-VL-8B-Instruct + LoRA distillation |
| Training data | ScanNet, Matterport3D, 3D-FRONT |
| Deployment | RunPod serverless (A40/A100/H100) |

---

## Current Status

| Suite | Tests | Status |
|-------|-------|--------|
| Stage 2 — Cell struct + perception + grid | 29/29 | PASS |
| Image-based — 11 synthetic + GPU models | 13/13 | PASS |
| Unified foundation — validation fields | 29/29 | PASS |
| VLM Stage 4 — model loading + context + fallbacks | 11/11 | PASS |
| VLM Stage 4 — needs_training | 14 | scaffolded |
| **Total verified** | **82/82** | **PASS** |

**Distillation validated:** 44.4% → 88.0% spatial accuracy with 2,898 synthetic pairs, zero aesthetic degradation.

**Serverless endpoint:** `0gmi4sn8cc0scc` on RunPod, 48GB GPU, idle at $0/hr.

---

## Theory Documents

| Document | What It Covers |
|----------|---------------|
| `cell_architecture_theory.md` | Cell data model, pipeline stages, derivatives replace systems |
| `single_image_precision_theory.md` | Depth Pro precision, error propagation, subdivision limits |
| `unified_pipeline_theory.md` | Validation fields, interaction model, auto-furnishing |
| `vlm_architecture_theory.md` | Qwen3-VL, native spatial encoding, SpatialVLM distillation |
| `trivima_testing_stage2.md` | 28 foundation tests specification |
| `trivima_testing_unified_foundation.md` | 29 validation field tests specification |
| `trivima_testing_vlm_stage4.md` | 25 VLM tests specification |

---

## Risk Summary

| Risk | Severity | Status |
|------|----------|--------|
| Noisy depth gradients | High | MITIGATED — bilateral σ=2.5 + 5×5 Sobel |
| Temporal flickering | High | MITIGATED — per-cell blending, not screen-space EMA |
| Failure modes (glass/mirror/dark) | High | MITIGATED — SAM 3 detection + forced density + confidence |
| cuDNN driver mismatch | Medium | MITIGATED — auto-disable cuDNN on init failure |
| SAM model version | Medium | RESOLVED — SAM 3 from facebook/sam3 |
| VLM spatial accuracy | Medium | RESOLVED — 88% after distillation (target was 60-75%) |
| 30B-A3B memory on A40 | Medium | RESOLVED — use 8B on A40, 30B on A100+ |
| Dual smoothing over-smoothing | Medium | MONITORING — test brick/wood grain in Week 4 |
| GAN blurry output | Medium | DESIGNED — perceptual loss + pretrained init + ControlNet fallback |
| VLM re-ranking latency | Low | VALIDATED — 200-500ms on A40 |
