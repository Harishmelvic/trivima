// hierarchy.cu
// Subdivision (Taylor expansion) and merge (integral-weighted averaging)
// kernels for the hierarchical cell grid.

#include "trivima/cell.h"
#include "trivima/cell_grid.h"

namespace trivima {

// TODO: Implement hierarchy adjustment kernels.
//       - Subdivision: split a cell into 8 children using Taylor expansion
//         of stored polynomial fields.
//       - Merge: combine 8 sibling cells into a parent via
//         integral-weighted averaging of properties.
//       - Criteria evaluation kernel to decide which cells to split/merge.

} // namespace trivima
