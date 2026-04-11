cmake_minimum_required(VERSION 3.20)
project(xtpu-compiler LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Find MLIR (via LLVM)
# Usage: cmake -S compiler -B compiler/build -DCMAKE_PREFIX_PATH=/opt/homebrew/opt/llvm
# ---------------------------------------------------------------------------
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

# Our own include dirs (source + generated)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

# LLVM definitions
add_definitions(${LLVM_DEFINITIONS})

# ---------------------------------------------------------------------------
# TableGen targets — generate .inc files from .td
# ---------------------------------------------------------------------------
add_subdirectory(include/xtpu/IR)

# ---------------------------------------------------------------------------
# Dialect library
# ---------------------------------------------------------------------------
add_subdirectory(lib/IR)

# ---------------------------------------------------------------------------
# Tools (xtpu-opt)
# ---------------------------------------------------------------------------
add_subdirectory(tools)
