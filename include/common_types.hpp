#pragma once
#include <cstdint>

// Status Register Bits
constexpr uint32_t STATUS_SDMA_BUSY    = 1 << 0;
constexpr uint32_t STATUS_PU0_DMA_BUSY = 1 << 1;
constexpr uint32_t STATUS_PU0_CMD_BUSY = 1 << 2;
constexpr uint32_t STATUS_PU1_DMA_BUSY = 1 << 3;
constexpr uint32_t STATUS_PU1_CMD_BUSY = 1 << 4;
// P2-4: 任何 Engine 發生異常時設定，必須由呼叫端明確 clear_error() 後才清除（sticky）
constexpr uint32_t STATUS_ERROR        = 1 << 5;

// Target Masks for iDMA
constexpr uint32_t TARGET_PU0 = 1 << 0;
constexpr uint32_t TARGET_PU1 = 1 << 1;

// Memory Constants
constexpr size_t SYSTEM_MEMORY_SIZE = 16 * 1024 * 1024; // 16MB（P1-3: System Memory）
constexpr size_t SCRATCHPAD_SIZE    =  1 * 1024 * 1024;  // 1MB
constexpr size_t LOCAL_MEM_SIZE     =       64 * 1024;   // 64KB per buffer

// Clock Constants (P3-CR-10)
// xTPU 預設時鐘速率：1 GHz → tCK = 1000 ps，ms_to_ticks = 1,000,000
// 所有涉及 xTPU clock period 的地方都應引用此常數，確保一致性（見 P3-CR-6）。
constexpr uint32_t XTPU_DEFAULT_TCK_PS  = 1000;           // ps per xTPU tick（1 GHz）
constexpr uint64_t XTPU_DEFAULT_MS_TO_TICKS = 1'000'000ULL; // ticks per ms at 1 GHz
