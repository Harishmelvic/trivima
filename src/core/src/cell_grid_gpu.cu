// cell_grid_gpu.cu
// GPU sparse grid wrapping cuco::static_map for O(1) parallel cell lookup.

#include "trivima/cell.h"
#include "trivima/cell_grid.h"

namespace trivima {

// TODO: Implement GPU-backed sparse grid using cuco::static_map.
//       - Build hash map from cell Morton codes to cell indices.
//       - Provide O(1) parallel lookup of cells by 3D coordinate.
//       - Support batch insert and batch query kernels.

} // namespace trivima
