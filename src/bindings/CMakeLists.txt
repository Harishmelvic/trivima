find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)

# Try to find nanobind
execute_process(
    COMMAND ${Python_EXECUTABLE} -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

if(nanobind_DIR)
    find_package(nanobind CONFIG REQUIRED)
    message(STATUS "nanobind found at ${nanobind_DIR}")

    nanobind_add_module(trivima_native
        bind_cell.cpp
        bind_grid.cpp
    )

    target_link_libraries(trivima_native PRIVATE trivima_core)

    install(TARGETS trivima_native DESTINATION trivima)
else()
    message(WARNING "nanobind not found — Python bindings will not be built. "
                    "Install with: pip install nanobind")
endif()
