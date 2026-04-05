set(CORE_SOURCES
    src/cell_grid.cpp
    src/serialization.cpp
)

# Add CUDA sources if available
if(CMAKE_CUDA_COMPILER)
    list(APPEND CORE_SOURCES
        src/cell_grid_gpu.cu
        src/gradients.cu
        src/hierarchy.cu
        src/point_to_cell.cu
        src/shell_extension.cu
        src/buffer_renderer.cu
    )
endif()

add_library(trivima_core STATIC ${CORE_SOURCES})

target_include_directories(trivima_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(CMAKE_CUDA_COMPILER)
    target_compile_definitions(trivima_core PUBLIC TRIVIMA_HAS_CUDA)
    # Target A100 (sm_80) and H100 (sm_90)
    set_target_properties(trivima_core PROPERTIES
        CUDA_ARCHITECTURES "80;90"
    )
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(trivima_core PRIVATE CUDA::cudart)
endif()

# Tests
if(BUILD_TESTING)
    add_executable(test_cell_sizes tests/test_cell_sizes.cpp)
    target_link_libraries(test_cell_sizes trivima_core)
endif()
