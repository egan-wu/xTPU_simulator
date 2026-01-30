#pragma once
#include <cstdint>

// Status Register Bits
constexpr uint32_t STATUS_SDMA_BUSY    = 1 << 0;
constexpr uint32_t STATUS_PU0_DMA_BUSY = 1 << 1;
constexpr uint32_t STATUS_PU0_CMD_BUSY = 1 << 2;
constexpr uint32_t STATUS_PU1_DMA_BUSY = 1 << 3;
constexpr uint32_t STATUS_PU1_CMD_BUSY = 1 << 4;

// Target Masks for iDMA
constexpr uint32_t TARGET_PU0 = 1 << 0;
constexpr uint32_t TARGET_PU1 = 1 << 1;

// Memory Constants
constexpr size_t SCRATCHPAD_SIZE = 1024 * 1024; // 1MB
constexpr size_t LOCAL_MEM_SIZE = 64 * 1024;    // 64KB per buffer
