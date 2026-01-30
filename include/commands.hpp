#pragma once
#include <cstdint>
#include <vector>
#include "common_types.hpp"

enum class DMAType {
    NOP,
    MEMCPY
};

struct DMA_Command {
    DMAType type = DMAType::NOP;
    uint64_t src_addr = 0;
    uint64_t dst_addr = 0; // Relative offset
    size_t size = 0;
    uint32_t target_mask = 0; // For iDMA: TARGET_PU0 | TARGET_PU1
    int buffer_idx = 0;       // 0 or 1 for Local Memory
};

enum class ComputeType {
    NOP,
    MATMUL, // Dummy operation
    VECTOR, // Dummy
    SCALAR  // Dummy
};

struct Compute_Command {
    ComputeType type = ComputeType::NOP;
    int buffer_idx = 0; // The buffer to operate on
    uint32_t simulated_duration_ms = 0; // To simulate work
};

struct VLIWPacket {
    DMA_Command sDMA_op;
    DMA_Command iDMA_op;
    Compute_Command pu0_op;
    Compute_Command pu1_op;
    uint32_t sync_mask; // Status bits to wait for
};
