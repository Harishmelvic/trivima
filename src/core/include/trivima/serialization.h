#pragma once
#include "trivima/cell_grid.h"

namespace trivima {

bool save_grid(const CellGridCPU& grid, const char* path);
bool load_grid(CellGridCPU& grid, const char* path);

} // namespace trivima
