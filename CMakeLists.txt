cmake_minimum_required(VERSION 3.18)
project(trivima LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Optional CUDA support
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    message(STATUS "CUDA found: ${CMAKE_CUDA_COMPILER}")
else()
    message(STATUS "CUDA not found — building CPU-only core")
endif()

add_subdirectory(src/core)
add_subdirectory(src/bindings)
