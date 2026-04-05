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

**Stage 1 — Perception:** Depth Pro extracts metric depth. SAM 3 segments objects with semantic labels. The depth map is back-projected to a 3D point cloud with per-point color, normal, and label.

**Stage 2 — Cell Grid Construction:** The point cloud is binned into 5cm cubic cells. Each cell computes density, albedo, normals, gradients, second derivatives, and integrals from its contained points. The result: ~80K cells, ~40MB.

**Stage 3 — Shell Extension:** RANSAC detects floor/wall/ceiling planes. New cells are generated along these planes to complete the room behind the camera.

**Stage 4 — AI Texturing:** A Pix2PixHD-Lite GAN takes cell buffer renders (albedo + depth + normals + labels) and produces photorealistic output. Light values are written back to cells. Only dirty cells (10-30%) are re-lit each frame. Requires pre-training on ~200K cell-buffer/photo pairs from ScanNet/Matterport3D (~48h on 8×A100).

**Stage 5 — Rendering:** Cells are drawn as instanced cubes via ModernGL. Adaptive LOD: subdivide near cells (Taylor expansion), merge distant cells. Cell-based collision and floor following for navigation.

**Stage 6 — Validation:** Conservation checks run async, one frame behind: energy (reflected ≤ incoming), mass (total density constant), shadow direction (gradient aligns with lights). Corrections applied gradually over 3-5 frames.

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
│       ├── gradients.cu             Finite-difference gradient kernels
│       ├── hierarchy.cu             Subdivision (Taylor) + merge (integral avg)
│       ├── point_to_cell.cu         Point cloud → cell grid conversion
│       ├── shell_extension.cu       Room completion along detected planes
│       └── buffer_renderer.cu       Cell grid → 2D buffers for AI model
│
├── src/bindings/                    nanobind Python wrappers
│
├── trivima/                         Python orchestration layer
│   ├── perception/
│   │   ├── depth_pro.py             Metric depth estimation
│   │   ├── sam.py                   Semantic segmentation (SAM 3 or Grounded SAM 2)
│   │   ├── depth_smoothing.py       Bilateral filter (critical for gradient quality)
│   │   ├── scale_calibration.py     Door detection for metric scale correction
│   │   └── pipeline.py              Orchestrates all perception models
│   │
│   ├── construction/
│   │   ├── point_to_grid.py         Python entry for cell grid construction
│   │   └── shell_extension.py       RANSAC plane fitting → room completion
│   │
│   ├── rendering/
│   │   ├── renderer.py              ModernGL instanced cube renderer + cell ID buffer
│   │   ├── camera.py                First-person camera (position + yaw + pitch)
│   │   └── lod.py                   Adaptive resolution: 0.5cm → 20cm per-cell
│   │
│   ├── navigation/
│   │   ├── collision.py             Cell lookup + density gradient sub-cell precision
│   │   └── floor_follow.py          Scan-down for floor surface + height smoothing
│   │
│   ├── texturing/
│   │   ├── buffer_renderer.py       Rasterize grid → 5 buffers (albedo/depth/normal/label/cellID)
│   │   ├── cell_writeback.py        AI output pixels → cell light values (view-angle weighted)
│   │   ├── models/
│   │   │   ├── pix2pix_lite.py      Real-time GAN: 8ch→3ch, U-Net, ~25M params, <10ms
│   │   │   └── controlnet_adapter.py Production path: SD Turbo + ControlNet + StreamDiffusion
│   │   ├── temporal.py              Per-cell blending (NOT screen-space EMA)
│   │   ├── inference_engine.py      Runtime coordinator: buffer→AI→writeback→blend
│   │   └── data_pipeline.py         ScanNet/Matterport3D training pair generation
│   │
│   ├── validation/
│   │   ├── conservation.py          Energy, mass, shadow conservation checks
│   │   └── validator.py             Orchestrator + gradual correction applicator
│   │
│   └── testing/
│       ├── benchmark.py             FPS/SSIM/LPIPS/flicker/collision/memory metrics
│       └── visual_comparison.py     Side-by-side output: flat | AI | ground truth
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

The renderer outputs a cell ID buffer alongside the color buffer. Each pixel stores which cell it corresponds to. After AI texturing, the output is mapped back to cells via this ID buffer, weighted by `max(dot(view_dir, cell_normal), 0.01)` — face-on pixels contribute more.

### 7. Bilateral Depth Smoothing

Single-image depth (Depth Pro) has per-pixel noise. Finite-difference gradients on noisy depth = garbage. A bilateral filter guided by the RGB image smooths depth while preserving edges, dramatically improving gradient quality and downstream Taylor expansion.

### 8. Gradual Conservation Corrections

Corrections from validation are spread over 3-5 frames instead of applied instantly, preventing visible artifacts. Validation runs async, one frame behind rendering.

### 9. Per-Cell Confidence (from error propagation theory)

Each cell carries a `confidence` float [0,1] derived from four signals:
- **Depth smoothness**: low local variance before bilateral filtering → high confidence
- **Point density**: more source points → more statistical support → higher confidence
- **Semantic penalty**: glass, mirror, transparent labels → low confidence regardless of depth quality
- **DUSt3R agreement**: (multi-image only) models agree → high, disagree → low

Confidence drives downstream behavior:
- Low-confidence cells get **expanded collision margins** (prevents walking through missed glass)
- Low-confidence cells **skip Taylor subdivision** (gradients unreliable, use neural features instead)
- Low-confidence cells get **more weight from AI texturing**, less from gradient-based shading
- Debug mode renders low-confidence cells with orange tint

### 10. Subdivision Depth Cap (from error propagation analysis)

Taylor expansion child prediction error is ±1.25cm from single-image input. This is smaller than the 5cm parent cell (subdivision helps) but comparable to the 2.5cm child cell (second subdivision has diminishing returns).

| Input Type | Max Taylor Subdivisions | Finest Cell | Beyond That |
|---|---|---|---|
| Single image | 1 level | 2.5 cm | Neural texture + AI texturing |
| Multi-image | 3 levels | 0.6 cm | Gradient-predicted detail |
| Video | 4 levels | 0.3 cm | Sub-mm precision possible |

Low-confidence cells are never subdivided regardless of input type.

### 11. Failure Mode Mitigations

SAM labels trigger material-specific corrections before cell construction:

| Surface | Problem | Mitigation |
|---|---|---|
| Mirror | Phantom room behind wall | Force density=1.0, confidence=0.1, block subdivision |
| Glass | Invisible to depth model | Force density=1.0 (solid barrier), confidence=0.2 |
| Transparent | Refraction distortion | Low confidence=0.3, expand collision margin |
| Dark scene | Universal noise | All cells low confidence, rely on AI texturing |
| Sky | Extreme depth (50-1000m) | Exclude from grid, render as skybox |
| Specular | Oscillating depth | Moderate confidence=0.4, extra depth smoothing |

---

## Data Flow

```
                                ┌─────────────────────┐
  Photo ──→ Depth Pro ──→ Depth │ Bilateral Smoothing  │──→ Point Cloud
         ──→ SAM 3 ──→ Labels   │ Scale Calibration    │     ↓
                                └─────────────────────┘  Binning
                                                           ↓
                                                     ┌──────────┐
                                                     │ Cell Grid │
                                                     │ (sparse)  │
                                                     └─────┬────┘
                                                           │
                              ┌─── Navigation starts here (1.5s) ───┐
                              │                                      │
                              ↓                                      ↓
                        Render visible cells              Shell Extension
                        (flat shading, LOD)               (RANSAC planes,
                              │                            background 10-15s)
                              │                                 │
                              ↓                                 ↓
                        Buffer Renderer ←──────────── Complete Grid
                        (5 channels + cell ID)              (merged)
                              ↓
                       ┌──────────────┐
                       │  AI Texturing │
                       │  (GAN or CN)  │
                       └──────┬───────┘
                              ↓
                        Cell Write-Back
                        (view-angle weighted)
                              ↓
                        Temporal Blend
                        (per-cell, not screen-space)
                              ↓
                        Render (ModernGL)
                        (instanced cubes + LOD)
                              ↓
                        Conservation Validation
                        (async, 1 frame behind)
                              ↓
                        Display @ 60+ FPS
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
| ML framework | PyTorch 2.5+ |
| Perception | Depth Pro, SAM 3 (or Grounded SAM 2) |
| Rendering | ModernGL 5.12 + GLFW |
| AI texturing (realtime) | Pix2PixHD-Lite (custom) |
| AI texturing (production) | StreamDiffusion + ControlNet |
| Training data | ScanNet, Matterport3D |

---

## Success Criteria

**Must-pass:**
- Photo → navigate in <5 seconds
- Looks recognizably like the actual room
- No wall clipping, no floor falling
- >20 FPS with AI texturing
- ≥15/20 test photos pass

**Nice-to-pass:**
- >30 FPS, SSIM >0.85, validation catches ≥50% of injected errors

**Stretch:**
- >45 FPS, works on outdoor/urban, indistinguishable from photo at same angle

---

## Risk Summary

| Risk | Severity | Mitigation |
|------|----------|------------|
| Noisy depth gradients | High | Bilateral smoothing + gradient quality test on synthetic scenes |
| Temporal flickering | High | Per-cell blending in 3D, not screen-space EMA |
| GAN produces blurry output | Medium | Perceptual loss + pretrained init + ControlNet fallback |
| Training data timing | Medium | Start ScanNet voxelization in Week 1 as background job |
| Shell extension seam | Medium | Known limitation — AI texture generation handles this in full system |
| SAM 2 doesn't classify | Medium | SAM 3 preferred, Grounded SAM 2 fallback |
