#pragma once
#include <cstdint>
#include <vector>
#include <string>

// Status Register Bit Definitions
// Refactored for Dual-Core VLIW TPU (PU0, PU1)
enum StatusBit : uint32_t {
    STATUS_IDMA_B0_BUSY = 1 << 0, // iDMA operating on Bank 0
    STATUS_IDMA_B1_BUSY = 1 << 1, // iDMA operating on Bank 1
    STATUS_IDMA_B2_BUSY = 1 << 2, // iDMA operating on Bank 2
    STATUS_IDMA_B3_BUSY = 1 << 3, // iDMA operating on Bank 3
    STATUS_SDMA_BUSY    = 1 << 4, // System DMA Busy
    
    // PU0 Status
    STATUS_PU0_MXU_BUSY = 1 << 5, 
    STATUS_PU0_VEC_BUSY = 1 << 6,
    STATUS_PU0_SCA_BUSY = 1 << 7,

    // PU1 Status
    STATUS_PU1_MXU_BUSY = 1 << 8,
    STATUS_PU1_VEC_BUSY = 1 << 9,
    STATUS_PU1_SCA_BUSY = 1 << 10
};

// DMA Command Types
enum class DMAType {
    NOP,
    MEMCPY,
    STRIDE // Added for enhanced iDMA
};

// Compute Command Types
enum class ComputeType {
    NOP,
    MATMUL,
    VECTOR_ADD,
    VECTOR_MASK
};

// DMA Command Structure
struct DMA_Command {
    DMAType type = DMAType::NOP;
    uint64_t src_addr = 0;
    uint64_t dst_addr = 0;
    size_t size = 0;

    // For iDMA: Target Bank ID (0-3)
    int bank_id = -1;

    // Cycle-accurate timing
    uint32_t duration_cycles = 0;
};

// Compute Command Structure
struct Compute_Command {
    ComputeType type = ComputeType::NOP;

    // Cycle-accurate timing
    uint32_t duration_cycles = 0;
};

// VLIW Packet Structure
struct VLIWPacket {
    DMA_Command sDMA_op;
    DMA_Command iDMA_op;
    Compute_Command pu0_op; // Operation for Processing Unit 0
    Compute_Command pu1_op; // Operation for Processing Unit 1

    uint32_t sync_mask = 0; // Bits that must be ZERO before dispatch
};
