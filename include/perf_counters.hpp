#pragma once
#include <atomic>
#include <cstdint>
#include <cstddef>
#include <ostream>
#include <iomanip>

// ---------------------------------------------------------------------------
// PerfCounters — 統一效能計數器框架 (P4-1)
//
// 設計原則：
//   1. 全部以 std::atomic 實作，允許多個 engine thread 並發累積，無需額外鎖。
//   2. Engine 完成操作後即時更新（fire-and-forget），不影響 critical path。
//   3. Simulator 持有一個 PerfCounters 實例，透過 get_perf_counters() 暴露。
//   4. reset() 可在每個 workload 前呼叫，測量單一 kernel 的 perf。
//
// 使用方式：
//   auto& pc = sim.get_perf_counters();
//   // ... run workload ...
//   pc.print(std::cout);
//
// 計數器語意：
//   sdma_bytes_loaded     : SDMA TO_DEVICE 傳輸的總 bytes（system → scratchpad）
//   sdma_bytes_written    : SDMA FROM_DEVICE 傳輸的總 bytes（scratchpad → system）
//   idma_bytes_loaded     : IDMA TO_DEVICE 傳輸的總 bytes（包含 broadcast 重複計算）
//   idma_bytes_written    : IDMA FROM_DEVICE 傳輸的總 bytes
//   pu_active_ticks[2]    : 各 PU 執行 compute op 所累積的 ticks（不含 idle）
//   sdma_ops              : SDMA 完成的操作次數（含 TO_DEVICE 和 FROM_DEVICE）
//   idma_ops              : IDMA 完成的操作次數
//   compute_ops[2]        : 各 PU 完成的 compute 操作次數
//   packets_dispatched    : dispatch_packet() 被呼叫的次數（含 NOP packet）
//   sync_stall_ticks      : wait_on_mask() 所等待的 wall-clock 模擬 ticks（不易量測，TODO）
// ---------------------------------------------------------------------------
struct PerfCounters {
    // ── DMA 傳輸量 ───────────────────────────────────────────────────────────
    std::atomic<uint64_t> sdma_bytes_loaded  {0};
    std::atomic<uint64_t> sdma_bytes_written {0};
    std::atomic<uint64_t> idma_bytes_loaded  {0};
    std::atomic<uint64_t> idma_bytes_written {0};

    // ── 操作計數 ──────────────────────────────────────────────────────────────
    std::atomic<uint64_t> sdma_ops     {0};
    std::atomic<uint64_t> idma_ops     {0};
    std::atomic<uint64_t> compute_ops_pu0 {0};
    std::atomic<uint64_t> compute_ops_pu1 {0};
    std::atomic<uint64_t> packets_dispatched {0};

    // ── Compute 忙碌 ticks ────────────────────────────────────────────────────
    std::atomic<uint64_t> pu0_active_ticks {0};
    std::atomic<uint64_t> pu1_active_ticks {0};

    // ── 便利方法 ──────────────────────────────────────────────────────────────

    // 重置所有計數器（在每個 workload 前呼叫）
    void reset() {
        sdma_bytes_loaded   .store(0, std::memory_order_relaxed);
        sdma_bytes_written  .store(0, std::memory_order_relaxed);
        idma_bytes_loaded   .store(0, std::memory_order_relaxed);
        idma_bytes_written  .store(0, std::memory_order_relaxed);
        sdma_ops            .store(0, std::memory_order_relaxed);
        idma_ops            .store(0, std::memory_order_relaxed);
        compute_ops_pu0     .store(0, std::memory_order_relaxed);
        compute_ops_pu1     .store(0, std::memory_order_relaxed);
        packets_dispatched  .store(0, std::memory_order_relaxed);
        pu0_active_ticks    .store(0, std::memory_order_relaxed);
        pu1_active_ticks    .store(0, std::memory_order_relaxed);
    }

    // 輸出格式化報告
    void print(std::ostream& os, uint64_t total_ticks = 0) const {
        os << "=== PerfCounters Report ===\n"
           << "  SDMA loaded      : " << sdma_bytes_loaded.load()   << " bytes (" << sdma_ops.load()  << " ops)\n"
           << "  SDMA written     : " << sdma_bytes_written.load()  << " bytes\n"
           << "  IDMA loaded      : " << idma_bytes_loaded.load()   << " bytes (" << idma_ops.load()  << " ops)\n"
           << "  IDMA written     : " << idma_bytes_written.load()  << " bytes\n"
           << "  PU0 active ticks : " << pu0_active_ticks.load()    << " (" << compute_ops_pu0.load() << " ops)\n"
           << "  PU1 active ticks : " << pu1_active_ticks.load()    << " (" << compute_ops_pu1.load() << " ops)\n"
           << "  Packets dispatched: " << packets_dispatched.load() << "\n";
        if (total_ticks > 0) {
            uint64_t p0 = pu0_active_ticks.load();
            uint64_t p1 = pu1_active_ticks.load();
            os << std::fixed << std::setprecision(1)
               << "  PU0 utilization  : " << (p0 * 100.0 / total_ticks) << " %\n"
               << "  PU1 utilization  : " << (p1 * 100.0 / total_ticks) << " %\n";
        }
        os << "===========================\n";
    }
};
