#pragma once
#include "sim_clock.hpp"          // TimingConfig
#include "common_types.hpp"       // SYSTEM_MEMORY_SIZE, XTPU_DEFAULT_TCK_PS
#include "lpddr5/device.h"        // lpddr5::DeviceConfig
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// SimulatorConfig — Simulator 頂層配置 (P3-3)
//
// 控制記憶體後端選擇：
//   SIMPLE  : 現有的 SystemMemory（簡單 SRAM model，固定延遲）
//   LPDDR5  : LPDDR5Adapter（接入 lpddr5-sim submodule，cycle-accurate 時序）
//
// 使用方式：
//   // SIMPLE backend（向下相容）
//   Simulator sim;
//   Simulator sim(timing_cfg);
//
//   // LPDDR5 backend
//   SimulatorConfig cfg;
//   cfg.backend = SimulatorConfig::MemoryBackend::LPDDR5;
//   Simulator sim(cfg);
// ---------------------------------------------------------------------------
struct SimulatorConfig {
    enum class MemoryBackend {
        SIMPLE,    // SystemMemory（固定延遲，向下相容）
        LPDDR5     // LPDDR5Adapter（cycle-accurate DDR 時序）
    };

    MemoryBackend        backend      = MemoryBackend::SIMPLE;
    size_t               mem_size     = SYSTEM_MEMORY_SIZE;     // backing store 大小
    uint32_t             xtpu_tck_ps  = XTPU_DEFAULT_TCK_PS;    // xTPU clock period（ps），P3-CR-10
    lpddr5::DeviceConfig ddr_cfg;               // LPDDR5 device 配置（backend==LPDDR5 時生效）
    TimingConfig         timing;                // Engine 時序參數

    // P3-CR-6: 驗證 xtpu_tck_ps 與 timing.ms_to_ticks 描述同一個 clock，互相一致。
    //   一致條件：xtpu_tck_ps × ms_to_ticks == 1e12（1 ps × 1e6 ticks/ms = 1 ms）
    //   若 ms_to_ticks == 0（純模擬模式），跳過檢查。
    // 在 Simulator(const SimulatorConfig&) ctor 中呼叫。
    void validate_clock_consistency() const {
        if (timing.ms_to_ticks == 0) return;  // 純模擬模式，ms_to_ticks 無物理意義
        constexpr uint64_t PICO_PER_MS = 1'000'000'000ULL; // 1 ms = 1e9 ps
        uint64_t implied = static_cast<uint64_t>(xtpu_tck_ps) * timing.ms_to_ticks;
        if (implied != PICO_PER_MS) {
            throw std::invalid_argument(
                "SimulatorConfig: xtpu_tck_ps=" + std::to_string(xtpu_tck_ps) +
                " and ms_to_ticks=" + std::to_string(timing.ms_to_ticks) +
                " are inconsistent (product=" + std::to_string(implied) +
                ", expected " + std::to_string(PICO_PER_MS) +
                "). Both must describe the same xTPU clock rate.");
        }
    }
};
