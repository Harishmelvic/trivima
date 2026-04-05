// buffer_renderer.cu
// CUDA kernel for rasterizing cell grid into flat buffers
// (albedo, depth, normals, labels, cell ID) for AI texturing input.

#include "trivima/cell.h"
#include "trivima/cell_grid.h"

namespace trivima {

// TODO: Implement cell grid rasterization kernel.
//       - Ray-march or splat occupied cells into screen-space buffers.
//       - Output per-pixel: albedo, depth, normal, semantic label, cell ID.
//       - Support configurable camera intrinsics and extrinsics.
//       - Buffers serve as input to the AI texturing pipeline.

} // namespace trivima
