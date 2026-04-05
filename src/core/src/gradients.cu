// gradients.cu
// CUDA kernels for finite-difference gradient and second derivative
// computation across all occupied cells.

#include "trivima/cell.h"
#include "trivima/cell_grid.h"

namespace trivima {

// TODO: Implement finite-difference gradient kernels.
//       - First-order central differences for density, albedo, etc.
//       - Second derivative (Laplacian / Hessian) computation.
//       - One thread per occupied cell, neighbor lookups via GPU grid.

} // namespace trivima
