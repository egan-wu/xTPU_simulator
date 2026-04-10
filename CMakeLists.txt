cmake_minimum_required(VERSION 3.16)
project(xTPU_Simulator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra)

# ---------------------------------------------------------------------------
# LPDDR5-sim submodule (P3-3)
# ---------------------------------------------------------------------------
set(LPDDR5_DIR "${CMAKE_SOURCE_DIR}/submodule/lpddr5-sim")
set(LPDDR5_LIB "${LPDDR5_DIR}/build/liblpddr5_sim.a")

if(NOT EXISTS "${LPDDR5_LIB}")
    message(FATAL_ERROR
        "LPDDR5 static library not found at ${LPDDR5_LIB}.\n"
        "Build it first:\n"
        "  cd ${LPDDR5_DIR} && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel")
endif()

add_library(lpddr5_sim STATIC IMPORTED GLOBAL)
set_target_properties(lpddr5_sim PROPERTIES
    IMPORTED_LOCATION             "${LPDDR5_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${LPDDR5_DIR}/include"
)

# ---------------------------------------------------------------------------
# Core simulator library (xtpu_core)
# ---------------------------------------------------------------------------
add_library(xtpu_core STATIC
    src/engines.cpp
    src/simulator.cpp
    src/lpddr5_adapter.cpp
)
target_include_directories(xtpu_core PUBLIC include)
target_link_libraries(xtpu_core
    PUBLIC  lpddr5_sim
    PRIVATE pthread
)

# ---------------------------------------------------------------------------
# Test executable
# ---------------------------------------------------------------------------
add_executable(test_simulator
    tests/test_simulator.cpp
)
target_link_libraries(test_simulator PRIVATE xtpu_core pthread)

add_custom_target(run_tests
    COMMAND ./test_simulator
    DEPENDS test_simulator
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    COMMENT "Running xTPU Simulator tests"
)

# ---------------------------------------------------------------------------
# Sanitizer builds (P4-3)
# ---------------------------------------------------------------------------
foreach(SAN tsan asan)
    add_executable(test_simulator_${SAN} tests/test_simulator.cpp)
    target_link_libraries(test_simulator_${SAN} PRIVATE xtpu_core pthread)
    target_compile_options(test_simulator_${SAN} PRIVATE -fsanitize=${SAN} -g -O1)
    target_link_options(test_simulator_${SAN}    PRIVATE -fsanitize=${SAN})
endforeach()
