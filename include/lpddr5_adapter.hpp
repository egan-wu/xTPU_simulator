#pragma once
#include <vector>
#include <mutex>
#include <cstring>
#include <stdexcept>
#include <ostream>
#include "memory_interface.hpp"   // IMemoryPort
#include "sim_clock.hpp"          // SimClock, TimingConfig
#include "common_types.hpp"       // SYSTEM_MEMORY_SIZE, XTPU_DEFAULT_TCK_PS
#include "lpddr5/device.h"        // lpddr5::Lpddr5Device, lpddr5::DeviceConfig

// ---------------------------------------------------------------------------
// LPDDR5Adapter — 實作 IMemoryPort (P3-2)
//
// 將 lpddr5-sim submodule 的非同步、timing-only 模型橋接為
// xTPU Engine 所期待的同步 IMemoryPort 介面。
//
// 設計：
//   1. backing_store_ 儲存真實資料（用於 read/write 實際位元組）
//   2. Lpddr5Device 模擬 LPDDR5 存取時序（cycle-accurate，不儲存資料）
//   3. read / write 為同步阻塞：先並發提交所有 cacheline 請求（pipelined），
//      再 spin-tick 等到全部完成，最後一次 SimClock::advance()。
//      (P3-CR-1: 取代舊的序列化方案，讓 LPDDR5-sim 的 scheduler 能重排)
//   4. has_own_timing() 回傳 true → SDMAEngine 不再加固定延遲（避免雙重計算）
//
// 時鐘換算：
//   xtpu_ticks = ceil(ddr_ck_cycles × tCK_ps / xtpu_tck_ps)
//   LPDDR5-6400: tCK = 1250 ps；xTPU 預設 1 GHz → tCK = 1000 ps
//   → 1 DDR CK ≈ 1.25 xTPU ticks
//
// 存取粒度：
//   每 64B（cacheline = BL16 burst size）提交一個 DDR transaction，
//   讓 LPDDR5-sim 正確追蹤 row-buffer hits/misses/conflicts。
//
// Pipelining (P3-CR-1)：
//   do_transactions() 一次提交多個 cacheline 請求，讓 LPDDR5-sim 的
//   command scheduler 可以在等待第一筆回應時就送出第二筆，
//   模擬真實 DDR controller 的 outstanding queue 行為。
//
// Timeout safety (P3-CR-4)：
//   do_transactions() 的 spin-tick 有 MAX_SPIN_CK 上限，超過後 throw，
//   SDMAEngine 的 try/catch 會把 error 紀錄到 STATUS_ERROR。
//
// Thread safety (P3-CR-9)：
//   write / read 由 mtx_ 保護，確保 current_dram_tick_ / next_req_id_ 的
//   read-modify-write 不被並發呼叫破壞。
//   ⚠ 限制：mtx_ 包覆整個 spin-tick loop，假設 LPDDR5 只有 SDMA 一個 client。
//   若未來有多個 Engine 直接存取 LPDDR5，需重構為 per-request 鎖或無鎖方案。
// ---------------------------------------------------------------------------
class LPDDR5Adapter : public IMemoryPort {
public:
    struct AdapterConfig {
        lpddr5::DeviceConfig dram_cfg;                             // LPDDR5 device 配置
        size_t               backing_size = SYSTEM_MEMORY_SIZE;    // backing store 大小
        uint32_t             xtpu_tck_ps  = XTPU_DEFAULT_TCK_PS;  // xTPU clock period (ps)，P3-CR-10
    };

    LPDDR5Adapter(SimClock& clock, const AdapterConfig& cfg);

    // ── IMemoryPort 介面 ─────────────────────────────────────────────────────

    // 寫入：先更新 backing_store，再模擬 LPDDR5 write 時序並推進 SimClock
    void write(uint64_t addr, const void* src, size_t size) override;

    // 讀取：模擬 LPDDR5 read 時序並推進 SimClock，再從 backing_store 複製資料
    void read(uint64_t addr, void* dst, size_t size) override;

    size_t capacity() const override { return backing_size_; }

    // 通知 SDMAEngine：此後端自行管理時序，不需固定延遲（避免雙重計算）
    bool has_own_timing() const override { return true; }

    // ── P3-4: DDR 效能可觀察性 ───────────────────────────────────────────────

    // 輸出 LPDDR5-sim 的 per-channel 統計報告（bandwidth、row-hit rate、latency 等）
    // P3-CR-8: 提供 ostream& 版本，可重導向到 log 檔案或在測試中捕捉。
    void print_ddr_stats(std::ostream& os, uint8_t channel = 0) const;

    // 向下相容：無 ostream 版本輸出到 std::cout
    void print_ddr_stats(uint8_t channel = 0) const;

    const lpddr5::ChannelStats& get_ddr_stats(uint8_t channel = 0) const {
        return dram_.stats(channel);
    }

    // ── 測試輔助 ─────────────────────────────────────────────────────────────

    // 直接寫入 backing_store（繞過 LPDDR5 時序模擬，用於測試資料預填）
    // 類比 SystemMemory::fill() 的角色
    void fill_direct(uint64_t addr, const void* src, size_t size) {
        std::lock_guard<std::mutex> lock(mtx_);
        bounds_check(addr, size);
        std::memcpy(backing_store_.data() + static_cast<size_t>(addr), src, size);
    }

    // 全部填充單一值（繞過時序，測試輔助）
    void fill(uint8_t value) {
        std::lock_guard<std::mutex> lock(mtx_);
        std::fill(backing_store_.begin(), backing_store_.end(), value);
    }

private:
    SimClock&            clock_;
    lpddr5::Lpddr5Device dram_;          // LPDDR5 timing simulator（不儲存資料）
    std::vector<uint8_t> backing_store_; // 真實資料儲存（SRAM 語意）
    size_t               backing_size_;
    uint32_t             xtpu_tck_ps_;   // xTPU clock period in picoseconds
    lpddr5::Tick         current_dram_tick_ = 0; // 單調遞增的 DRAM CK 計數
    uint32_t             next_req_id_        = 0; // request ID 計數器
    std::mutex           mtx_;

    // P3-CR-4: spin-tick 最大迭代次數（防止 LPDDR5-sim bug 導致無聲死迴圈）
    // 100,000 DDR CK @LPDDR5-6400 ≈ 125 µs 模擬時間，遠超任何正常 transaction。
    static constexpr lpddr5::Tick MAX_SPIN_CK = 100'000;

    // 將 DDR CK cycles 換算為 xTPU ticks（ceil，確保不低估延遲）
    SimClock::Tick ddr_to_xtpu(lpddr5::Tick ddr_ticks) const;

    // P3-CR-1: 並發提交多個 cacheline 請求，spin-tick 直到全部 CompletionEvent。
    // 取代舊的 do_transaction（序列化版本）。
    // addrs：cacheline-aligned 位址列表（每個 64B 對齊）
    // is_write：true = write，false = read
    // 回傳總消耗的 DDR CK 數（由呼叫端換算後 advance SimClock）。
    lpddr5::Tick do_transactions(const std::vector<uint64_t>& addrs, bool is_write);

    // 溢位安全邊界檢查（與 SystemMemory / Scratchpad 相同的模式）
    void bounds_check(uint64_t addr, size_t size) const;
};
