#pragma once
#include <cstdint>
#include "common_types.hpp"

enum class DMAType {
    NOP,
    MEMCPY
};

// ---------------------------------------------------------------------------
// DMADirection — 傳輸方向 (P1-4)
//
// TO_DEVICE  (Load) : 外部記憶體 → 晶片內
//   - sDMA: system_mem  → scratchpad
//   - iDMA: scratchpad  → local_mem
//
// FROM_DEVICE (Store/Writeback) : 晶片內 → 外部記憶體
//   - sDMA: scratchpad  → system_mem
//   - iDMA: local_mem   → scratchpad
// ---------------------------------------------------------------------------
enum class DMADirection {
    TO_DEVICE,   // 預設：load（外部 → 晶片內）
    FROM_DEVICE  // writeback（晶片內 → 外部）
};

struct DMA_Command {
    DMAType      type        = DMAType::NOP;
    uint64_t     src_addr    = 0;
    uint64_t     dst_addr    = 0;       // Relative offset
    size_t       size        = 0;
    uint32_t     target_mask = 0;       // For iDMA: TARGET_PU0 | TARGET_PU1
    int          buffer_idx  = 0;       // 0 or 1 for Local Memory
    DMADirection direction   = DMADirection::TO_DEVICE;  // P1-4: load/writeback 方向
};

enum class ComputeType {
    NOP,
    MATMUL,
    VECTOR,
    SCALAR
};

// ---------------------------------------------------------------------------
// Compute_Command (P2-1)
//
// 欄位順序與舊 aggregate init {type, buffer_idx, simulated_duration_ms} 相容：
// 新增的 src_offset / dst_offset / length 置於末尾，預設 0。
//
// 若 length == 0：只做延遲模擬（向下相容舊測試）。
// 若 length > 0 ：從 local_mem[buffer_idx] 讀取資料、執行運算、寫回結果。
//
// 運算定義（element-wise, uint8_t，超出 0xFF 自然截斷）：
//   SCALAR : dst[i] = src[i] + 1
//            讀取: length bytes (src_offset)
//            寫入: length bytes (dst_offset)
//
//   VECTOR : dst[i] = src[i] * src[i]
//            讀取: length bytes (src_offset)
//            寫入: length bytes (dst_offset)
//
//   MATMUL : C = A × B，4×4 uint8_t 矩陣（uint32_t 累加，結果截斷至 uint8_t）
//            ⚠ length 語意與 SCALAR/VECTOR 不同：
//              length = 單一矩陣的 bytes（= N×N = 16）
//              A 位於 src_offset         （讀取 length bytes）
//              B 位於 src_offset + length（讀取 length bytes，共讀 2×length）
//              C 寫入 dst_offset          （寫入 length bytes）
//            ⚠ length 必須 == 16（N=4 固定），否則 STATUS_ERROR 被設定
// ---------------------------------------------------------------------------
struct Compute_Command {
    ComputeType type = ComputeType::NOP;
    int      buffer_idx            = 0;   // Local Memory buffer（0 or 1）
    uint32_t simulated_duration_ms = 0;   // > 0：real sleep + ms→ticks 換算（向下相容）
    uint32_t src_offset            = 0;   // P2-1: 輸入資料在 local_mem 的起始 offset
    uint32_t dst_offset            = 0;   // P2-1: 輸出資料在 local_mem 的起始 offset
    uint32_t length                = 0;   // P2-1: 操作長度（bytes）；0 = 只模擬延遲
};

struct VLIWPacket {
    DMA_Command sDMA_op;
    DMA_Command iDMA_op;
    Compute_Command pu0_op;
    Compute_Command pu1_op;
    uint32_t sync_mask = 0; // Status bits to wait for (預設 0 = 無需等待)
};
