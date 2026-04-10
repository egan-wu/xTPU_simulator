#include "simulator.hpp"
#include <iostream>

// ---------------------------------------------------------------------------
// P3-3: SimulatorConfig 建構子
//
// 初始化順序必須與 simulator.hpp 中的成員宣告順序一致：
//   timing_ → sys_mem_ → lpddr5_mem_ → scratchpad_ → local_mem_
//   → sdma_ → idma_ → pu0_ → pu1_
//
// 關鍵：sdma_ 需要 IMemoryPort& 指向 active backend。
//   - SIMPLE : 傳入 sys_mem_（SystemMemory）
//   - LPDDR5 : 傳入 *lpddr5_mem_（LPDDR5Adapter），在 lpddr5_mem_ 初始化之後
//
// lpddr5_mem_ 初始化時引用 clock_（已在 lpddr5_mem_ 之前宣告，故安全）。
// ---------------------------------------------------------------------------
Simulator::Simulator(const SimulatorConfig& cfg)
    : timing_((cfg.validate_clock_consistency(), cfg.timing)),  // P3-CR-6: validate first
      sys_mem_(cfg.backend == SimulatorConfig::MemoryBackend::LPDDR5
               ? nullptr
               : std::make_unique<SystemMemory>(cfg.mem_size)),  // P3-CR-3: 不浪費 16MB
      lpddr5_mem_(
          cfg.backend == SimulatorConfig::MemoryBackend::LPDDR5
          ? std::make_unique<LPDDR5Adapter>(
                clock_,
                LPDDR5Adapter::AdapterConfig{cfg.ddr_cfg, cfg.mem_size, cfg.xtpu_tck_ps})
          : nullptr),
      sdma_(status_reg_, clock_, timing_,
            // P3-CR-2: 依 backend 選擇 active system memory（兩者必定有一個非 null）
            lpddr5_mem_ ? static_cast<IMemoryPort&>(*lpddr5_mem_)
                        : static_cast<IMemoryPort&>(*sys_mem_),
            scratchpad_),
      idma_(status_reg_, clock_, timing_, scratchpad_,
            local_mem_[0], local_mem_[1]),
      pu0_(status_reg_, clock_, timing_, local_mem_[0], STATUS_PU0_CMD_BUSY),
      pu1_(status_reg_, clock_, timing_, local_mem_[1], STATUS_PU1_CMD_BUSY)
{ wire_perf_counters(); }

void Simulator::dispatch_packet(const VLIWPacket& packet) {
    // P4-1: 記錄 dispatch 次數（包含純 sync 的 packet）
    perf_.packets_dispatched.fetch_add(1, std::memory_order_relaxed);

    // 1. Sync / Fence: Wait for dependencies to clear
    if (packet.sync_mask != 0) {
        status_reg_.wait_on_mask(packet.sync_mask);
        // P2-CR-5: 在 sync 完成後，若 STATUS_ERROR 已被設定，主動提示呼叫端
        // 控制流不改變（不 throw、不 return），保持向下相容
        if (status_reg_.has_error()) {
            std::cerr << "[Simulator] Warning: STATUS_ERROR is set after sync. "
                      << "Call has_error() / get_error_info() to inspect, "
                      << "clear_error() to acknowledge." << std::endl;
        }
    }

    // 2. Dispatch sDMA
    if (packet.sDMA_op.type != DMAType::NOP) {
        status_reg_.set_busy(STATUS_SDMA_BUSY);
        sdma_.push_command(packet.sDMA_op);
    }

    // 3. Dispatch iDMA
    if (packet.iDMA_op.type != DMAType::NOP) {
        uint32_t mask = 0;
        if (packet.iDMA_op.target_mask & TARGET_PU0) mask |= STATUS_PU0_DMA_BUSY;
        if (packet.iDMA_op.target_mask & TARGET_PU1) mask |= STATUS_PU1_DMA_BUSY;

        if (mask != 0) {
            status_reg_.set_busy(mask);
            idma_.push_command(packet.iDMA_op);
        }
    }

    // 4. Dispatch PU0 Compute
    if (packet.pu0_op.type != ComputeType::NOP) {
        status_reg_.set_busy(STATUS_PU0_CMD_BUSY);
        pu0_.push_command(packet.pu0_op);
    }

    // 5. Dispatch PU1 Compute
    if (packet.pu1_op.type != ComputeType::NOP) {
        status_reg_.set_busy(STATUS_PU1_CMD_BUSY);
        pu1_.push_command(packet.pu1_op);
    }
}
