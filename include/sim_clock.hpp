#pragma once
#include <atomic>
#include <cstdint>
#include <cstddef>

// ---------------------------------------------------------------------------
// TimingConfig — 可配置的時序參數表 (P1-2)
//
// 所有延遲以 SimClock::Tick 為單位，而非 wallclock ms。
// 1 Tick = 1 simulated clock cycle（xTPU reference clock，預設 1GHz）。
//
// 使用情境：
//   - DMA engines 依 transfer size 計算 cache line 數 × per-line latency
//   - Compute engines 依 operation type 查表取延遲
//   - DDR Adapter（P3-2）使用 ddr_* 欄位建模 DDR4/DDR5 時序
// ---------------------------------------------------------------------------
struct TimingConfig {
    using Tick = uint64_t;

    // ── DMA 延遲 ─────────────────────────────────────────────────────────────
    Tick sdma_latency_per_cacheline = 10;  // ticks per 64B（System Mem → Scratchpad）
    Tick idma_latency_per_cacheline = 5;   // ticks per 64B（Scratchpad → LocalMem）

    // ── Compute 延遲 ─────────────────────────────────────────────────────────
    Tick matmul_latency = 100;  // MATMUL 固定延遲（tile-level，未來可改為 ops-based）
    Tick vector_latency =  20;  // VECTOR 逐元素運算
    Tick scalar_latency =   5;  // SCALAR 純量運算

    // ── 向下相容：simulated_duration_ms → ticks 的換算比率 ───────────────────
    // 1 ms 預設對應 1,000,000 ticks（假設 1GHz clock，與 XTPU_DEFAULT_TCK_PS=1000ps 一致）
    // 設定為 0 可完全停用真實 sleep（加速模擬）
    // P3-CR-6: 若 SimulatorConfig::xtpu_tck_ps 不是預設值，ms_to_ticks 必須同步調整，
    //          否則 Compute latency 與 LPDDR5 latency 會以不同的 wall-time 基準計算。
    Tick ms_to_ticks = 1'000'000ULL; // = 1e12 ps/ms / XTPU_DEFAULT_TCK_PS(1000 ps)

    // ── DDR 時序（P3-2 接入 DDR Controller 時使用）────────────────────────────
    Tick ddr_cas_latency     = 22;  // CL（Column Address Strobe Latency）
    Tick ddr_ras_to_cas      = 22;  // tRCD（RAS to CAS Delay）
    Tick ddr_row_precharge   = 22;  // tRP（Row Precharge Time）
    Tick ddr_burst_len_ticks =  4;  // BL16 burst 佔用 xTPU cycles 數

    // ── 常數 ──────────────────────────────────────────────────────────────────
    static constexpr size_t CACHELINE_SIZE = 64;  // bytes，DDR burst 基本單位
};

// ---------------------------------------------------------------------------
// SimClock — 虛擬時鐘 (P1-2)
//
// 以 atomic<Tick> 實作，允許多個 engine thread 並發 advance()。
// 語意：累積所有已模擬的運算工作量，非 wall-clock 時間。
//
// 設計說明：
//   - fetch_add 是原子 RMW，多 thread 並發 advance 不會有 lost-update
//   - 並發累積（而非取 max）是有意為之：反映「total simulated work」
//   - 若未來需要 per-engine 時間或取 max 語意，可改為 per-engine SimClock
//     後由 Simulator::get_clock() 聚合
// ---------------------------------------------------------------------------
class SimClock {
public:
    using Tick = uint64_t;

    SimClock() : current_tick_(0) {}

    // 讀取目前的 simulated tick 數
    Tick now() const {
        return current_tick_.load(std::memory_order_acquire);
    }

    // Engine 完成一筆操作後呼叫，累積對應的延遲 ticks
    void advance(Tick delta) {
        if (delta > 0) {
            current_tick_.fetch_add(delta, std::memory_order_release);
        }
    }

    // 測試輔助：重置計數器
    void reset() {
        current_tick_.store(0, std::memory_order_release);
    }

private:
    std::atomic<Tick> current_tick_;
};
