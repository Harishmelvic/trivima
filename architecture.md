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

**Stage 8 — VLM Intelligence (Stage 4+):** Qwen2.5-VL-32B + 3D-RoPE re-ranks candidates, plans auto-furnishing, matches object styles. Never in the render loop.

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
│   ├── vlm/                         (Stage 4 — planned)
│   │   ├── qwen_vlm.py             Qwen2.5-VL-32B + 3D-RoPE injection
│   │   ├── aesthetic_ranker.py      Logit scoring (~200ms) + generative (~2-5s)
│   │   ├── auto_furnish.py          Gap detection + placement planning
│   │   └── training/               SpatialVLM distillation + LoRA fine-tuning
│   │
│   ├── testing/
│   │   ├── benchmark.py             FPS/SSIM/LPIPS/flicker/collision/memory
│   │   └── visual_comparison.py     Side-by-side: flat | AI | ground truth
│   │
│   └── app.py                       CLI entry point with --stats, --export-ply, --render-preview
│
├── handler.py                       RunPod serverless handler
├── Dockerfile                       Pre-built image for serverless deployment
│
├── tests/
│   ├── test_stage2.py               29 tests: cell struct + perception + grid + shell + LOD
│   ├── test_image_based.py          13 tests: 11 synthetic images + 4 GPU model tests
│   ├── test_unified_foundation.py   29 tests: confidence + surface + functional + BFS + conservation
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

Combined, they reduce gradient noise by 40-65% without erasing fine textures. If testing shows over-smoothing on brick/wood grain, reduce bilateral σ to 2.0.

### 8. Gradual Conservation Corrections

Corrections from validation are spread over 3-5 frames instead of applied instantly, preventing visible artifacts. Validation runs async, one frame behind rendering.

### 9. Per-Cell Confidence

Each cell carries a `confidence` float [0,1] derived from multiplicative product:

`confidence = depth_smoothness × point_density × semantic_penalty`

- **Depth smoothness**: low local variance → high confidence
- **Point density**: `min(1.0, n / 10.0)` — more points per cell → higher confidence
- **Semantic penalty**: glass (×0.2), mirror (×0.1), transparent (×0.3), specular (×0.4), dark (×0.4)
- **DUSt3R agreement**: (multi-image only) model disagreement → low confidence

Confidence drives: collision margins (expanded for low-conf), subdivision (blocked < 0.5), AI texturing weight (boosted for low-conf), debug rendering (orange tint).

### 10. Subdivision Depth Cap (from error propagation analysis)

Taylor expansion child prediction error is ±1.25cm from single-image input.

| Input Type | Max Taylor Subdivisions | Finest Cell | Beyond That |
|---|---|---|---|
| Single image | 1 level | 2.5 cm | Neural texture + AI texturing |
| Multi-image | 3 levels | 0.6 cm | Gradient-predicted detail |
| Video | 4 levels | 0.3 cm | Sub-mm precision possible |

Low-confidence cells are never subdivided regardless of input type.

### 11. Failure Mode Mitigations

SAM 3 labels trigger material-specific corrections before cell construction:

| Surface | Problem | Mitigation |
|---|---|---|
| Mirror | Phantom room behind wall | Force density=1.0, confidence=0.1, block subdivision |
| Glass | Invisible to depth model | Force density=1.0 (solid barrier), confidence=0.2 |
| Transparent | Refraction distortion | Low confidence=0.3, expand collision margin |
| Dark scene | Universal noise | All cells low confidence, rely on AI texturing |
| Sky | Extreme depth (50-1000m) | Exclude from grid, render as skybox |
| Specular | Oscillating depth | Moderate confidence=0.4, extra depth smoothing |

### 12. VLM Design Intelligence (Stage 4)

**Model:** Qwen2.5-VL-32B with 3D-RoPE + SpatialVLM knowledge distillation.

**3D-RoPE** replaces 2D pixel positional encoding with metric 3D coordinates from the depth map. Encoding dimensions split equally: 1/3 X, 1/3 Y, 1/3 Z. Coordinates normalized to [0,1] by room dimensions. Patches close in physical space have high mutual attention regardless of pixel distance. Zero parameters, zero latency. Falls back to 2D for low-confidence cells.

**SpatialVLM distillation** transfers quantitative spatial reasoning into Qwen during training via 50-100K spatial QA pairs generated by running SpatialVLM offline. Achieves 85-90% of SpatialVLM's spatial precision at zero inference cost.

**Why distillation, not feature fusion:** Fusion (SpatialVLM running alongside Qwen at inference) was rejected. The 2-5% accuracy improvement did not justify: 5-6× higher training cost ($15-30K vs $3-5K), +0.5-1s latency, +13GB GPU memory, and two-model deployment complexity.

**Invocation points** (never in the render loop):
- Environment classification (~2s, once per scene, parallel with grid construction)
- Aesthetic re-ranking: fast mode (logit scoring, ~200ms) for heatmap updates; full mode (generative, ~2-5s) with explanations when user taps a position
- Auto-furnishing planning (~3-5s, once per scene)
- Object style matching (~200-500ms per object)

**Asymmetric dependency:** The validation system (physics) works without the VLM. The VLM cannot work without the validation system. Physics never depends on neural network judgment.

**Training:**
| Phase | Duration | Hardware | Cost |
|---|---|---|---|
| Spatial distillation (LoRA rank 32, alpha 64) | 3-5 days | 4×A100 | $1,500-2,500 |
| Aesthetic fine-tuning (3D-FRONT data) | 3-5 days | 4×A100 | $1,500-2,500 |
| Human preference DPO (optional) | 1-2 days | 2×A100 | $300-500 |
| **Total** | **7-12 days** | | **$3,000-5,000** |

**Canary metrics** during training to detect capability degradation:
- Aesthetic canary: 500 held-out layouts, correlation with human ratings > 0.75
- Language canary: 200 held-out prompts, perplexity increase < 15%
- Spatial canary: 500 held-out measurements, accuracy > 80% of SpatialVLM

**Evaluation dimensions:**
1. Spatial precision (within 15-20% of SpatialVLM)
2. Aesthetic quality (80%+ agreement with human preferences)
3. Reasoning coherence (90% rated coherent)
4. 3D-RoPE benefit (10%+ improvement over 2D encoding ablation)
5. End-to-end placement quality (15%+ improvement over validation-only)

**Key ablation:** Compare validation-only (A) vs VLM+2D-RoPE (B) vs VLM+3D-RoPE (C). Expected: C > B > A.

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
                        │  (Qwen+3D)   │
                        └─────────────┘
```

---

## Resolution Tiers

| Tier | Cell Size | Cells/m³ | Distance | Use Case |
|------|-----------|----------|----------|----------|
| 0 | 20 cm | 125 | >15m | Distant background |
| 1 | 5 cm | 8,000 | 3-15m | Navigation |
| 2 | 2 cm | 125,000 | <3m | Close inspection |
| 3 | 1 cm | 1,000,000 | Production | High-detail render |
| 4 | 0.5 cm | 8,000,000 | Cinematic | Extreme close-ups |

Subdivision: `child_value = parent + ∇ · offset + ½∇² · offset²`
Merge: integral-weighted average (total energy/mass/color conserved exactly)

---

## Memory Budget (Typical Room)

| Mode | Cells | Memory | GPU |
|------|-------|--------|-----|
| Real-time navigation | ~30K visible | 15-60 MB | RTX 3060+ |
| Production render | ~7.5M at Tier 3 | 1-4 GB | RTX 4090 |
| Cinematic render | ~120M at Tier 4 | 4-15 GB (streamed) | A100 |

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
| Segmentation | SAM 3 (facebook/sam3, 840M params, HF Transformers) |
| Rendering | ModernGL 5.12 + GLFW |
| AI texturing (realtime) | Pix2PixHD-Lite (custom, 25M params) |
| AI texturing (production) | StreamDiffusion + ControlNet |
| VLM (Stage 4) | Qwen2.5-VL-32B + 3D-RoPE + LoRA |
| Training data | ScanNet, Matterport3D, 3D-FRONT |
| Deployment | RunPod serverless (A40/A6000, 48GB) |

---

## Current Status

**71/71 tests pass** across 3 test suites on RunPod A40 with Depth Pro + SAM 3.

| Stage | Status | Tests |
|-------|--------|-------|
| 1-2. Foundation (cell grid, perception, confidence) | COMPLETE | 29/29 |
| Image-based (11 synthetic images, GPU models) | COMPLETE | 13/13 |
| Unified foundation (validation fields, conservation) | COMPLETE | 29/29 |
| 3. Placement (heatmaps, constraints) | NOT STARTED | — |
| 4. VLM intelligence (Qwen + 3D-RoPE) | NOT STARTED | — |

**Serverless endpoint:** `0gmi4sn8cc0scc` on RunPod, 48GB GPU (A6000/A40), idle at $0/hr.

---

## Theory Documents

| Document | What It Covers |
|----------|---------------|
| `cell_architecture_theory.md` | Cell data model, pipeline stages, derivatives replace systems |
| `single_image_precision_theory.md` | Depth Pro precision, error propagation, subdivision limits |
| `unified_pipeline_theory.md` | Validation fields, interaction model, auto-furnishing |
| `vlm_architecture_theory.md` | Qwen2.5-VL, 3D-RoPE, SpatialVLM distillation, training |
| `trivima_testing_stage2.md` | 28 foundation tests specification |
| `trivima_testing_unified_foundation.md` | 29 validation field tests specification |

---

## Risk Summary

| Risk | Severity | Status |
|------|----------|--------|
| Noisy depth gradients | High | MITIGATED — bilateral σ=2.5 + 5×5 Sobel |
| Temporal flickering | High | MITIGATED — per-cell blending, not screen-space EMA |
| Failure modes (glass/mirror/dark) | High | MITIGATED — SAM 3 detection + forced density + confidence |
| cuDNN driver mismatch on cloud | Medium | MITIGATED — auto-disable cuDNN on init failure |
| SAM model version | Medium | RESOLVED — SAM 3 from facebook/sam3 via HF Transformers |
| Dual smoothing over-smoothing | Medium | MONITORING — bilateral σ reduced to 2.5, test in Week 4 |
| GAN blurry output | Medium | DESIGNED — perceptual loss + pretrained init + ControlNet fallback |
| Training data timing | Medium | DESIGNED — start ScanNet voxelization as background job |
| VLM re-ranking latency | Low-Med | DESIGNED — two modes: logit (~200ms) + generative (~2-5s) |
