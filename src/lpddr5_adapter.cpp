#include "lpddr5_adapter.hpp"
#include <algorithm>   // std::min
#include <stdexcept>
#include <unordered_set>
#include <iostream>    // std::cout（P3-CR-8 default overload）

// ---------------------------------------------------------------------------
// LPDDR5Adapter — 實作細節
// ---------------------------------------------------------------------------

LPDDR5Adapter::LPDDR5Adapter(SimClock& clock, const AdapterConfig& cfg)
    : clock_(clock),
      dram_(cfg.dram_cfg),
      backing_store_(cfg.backing_size, 0),
      backing_size_(cfg.backing_size),
      xtpu_tck_ps_(cfg.xtpu_tck_ps)
{}

// ---------------------------------------------------------------------------
// bounds_check — 溢位安全邊界檢查（與 SystemMemory 相同的慣用模式）
// ---------------------------------------------------------------------------
void LPDDR5Adapter::bounds_check(uint64_t addr, size_t size) const {
    if (size > backing_size_ || addr > backing_size_ - size)
        throw std::out_of_range("LPDDR5Adapter: access out of bounds");
}

// ---------------------------------------------------------------------------
// ddr_to_xtpu — 將 LPDDR5 CK cycles 換算為 xTPU ticks
//
// 換算公式（ceiling division，確保不低估延遲）：
//   total_ps    = ddr_ticks × tCK_ps（ddr_ticks 個 DDR clock period）
//   xtpu_ticks  = ⌈total_ps / xtpu_tck_ps_⌉
//
// 範例（LPDDR5-6400, xTPU@1GHz）：
//   tCK = 1250 ps, xtpu_tck_ps = 1000 ps
//   35 DDR CK → 35 × 1250 = 43750 ps → ⌈43750/1000⌉ = 44 xTPU ticks
// ---------------------------------------------------------------------------
SimClock::Tick LPDDR5Adapter::ddr_to_xtpu(lpddr5::Tick ddr_ticks) const {
    uint64_t dram_tCK_ps = static_cast<uint64_t>(dram_.timing().tCK_ps);
    uint64_t total_ps    = ddr_ticks * dram_tCK_ps;
    // ceiling division
    return static_cast<SimClock::Tick>((total_ps + xtpu_tck_ps_ - 1) / xtpu_tck_ps_);
}

// ---------------------------------------------------------------------------
// do_transactions — 並發提交多個 DDR burst 請求，統一等待全部完成 (P3-CR-1)
//
// 演算法（Pipelined，取代舊的序列化 do_transaction）：
//   Phase 1 — Submission with interleaved ticking：
//     嘗試把 addrs 中的每筆位址依序送進 LPDDR5-sim。
//     若 channel queue 滿（返回 false），tick 一個 CK 後重試。
//     在等待 queue 空出時，已接受的 requests 可能提前完成（從 pending 移除）。
//     成功送出後立刻嘗試送下一筆，讓 LPDDR5-sim 的 scheduler 有機會重排。
//   Phase 2 — Completion drain：
//     所有請求都送出後，繼續 tick 直到 pending set 清空。
//
// Timeout (P3-CR-4)：
//   MAX_SPIN_CK 限制總迭代次數，超過後 throw std::runtime_error。
//   SDMAEngine 的 try/catch 會把 error 設到 STATUS_ERROR，不會無聲掛死。
//
// 回傳：total elapsed DDR CK（start → 最後一筆 completion 後的下一個 cycle），
//       由呼叫端累積後一次換算並 advance SimClock。
// ---------------------------------------------------------------------------
lpddr5::Tick LPDDR5Adapter::do_transactions(
    const std::vector<uint64_t>& addrs, bool is_write)
{
    if (addrs.empty()) return 0;

    lpddr5::Tick start = current_dram_tick_;
    std::unordered_set<uint32_t> pending;
    size_t next_to_submit = 0;

    // Phase 1 + 2 combined：送出 + 等完成
    while (next_to_submit < addrs.size() || !pending.empty()) {
        // Timeout guard (P3-CR-4)
        if (current_dram_tick_ - start > MAX_SPIN_CK) {
            throw std::runtime_error(
                "LPDDR5Adapter: transaction timeout after " +
                std::to_string(MAX_SPIN_CK) + " DDR CK; " +
                std::to_string(pending.size()) + " req(s) still pending");
        }

        // 嘗試提交下一筆（若還有未提交的）
        if (next_to_submit < addrs.size()) {
            uint32_t req_id = next_req_id_;  // peek，成功後才 commit
            bool accepted;
            if (is_write)
                accepted = dram_.write(addrs[next_to_submit], req_id, current_dram_tick_);
            else
                accepted = dram_.read(addrs[next_to_submit], req_id, current_dram_tick_);

            if (accepted) {
                ++next_req_id_;  // commit req_id
                pending.insert(req_id);
                ++next_to_submit;
                // 不立刻 tick：繼續嘗試送下一筆（實現真正的 pipelining）
                continue;
            }
            // 未接受（queue 滿）：往下 tick 一個 CK 後重試
        }

        // Tick DRAM：推進時序，收 CompletionEvent
        dram_.tick(current_dram_tick_, [&](const lpddr5::CompletionEvent& e) {
            pending.erase(e.req_id);
        });
        ++current_dram_tick_;
    }

    return current_dram_tick_ - start;
}

// ---------------------------------------------------------------------------
// write — 同步寫入
//
// 流程：
//   1. 更新 backing_store_（實際資料）
//   2. 建立 cacheline-aligned 位址列表，並發送到 do_transactions（P3-CR-1）
//   3. 將總 DDR CK 換算為 xTPU ticks，透過 SimClock::advance() 累積
//
// 資料寫入先於時序模擬，確保資料一致性（即使 read 尚未模擬完成，
// backing_store 已有正確資料，符合同步介面語意）。
// ---------------------------------------------------------------------------
void LPDDR5Adapter::write(uint64_t addr, const void* src, size_t size) {
    std::lock_guard<std::mutex> lock(mtx_);
    bounds_check(addr, size);

    // 1. 寫入真實資料到 backing store
    std::memcpy(backing_store_.data() + static_cast<size_t>(addr), src, size);

    // 2. 建立 cacheline-aligned 位址列表
    const size_t CL = TimingConfig::CACHELINE_SIZE;
    std::vector<uint64_t> addrs;
    addrs.reserve((size + CL - 1) / CL);
    for (size_t offset = 0; offset < size; offset += CL)
        addrs.push_back(addr + offset);

    // 3. 並發提交並推進 SimClock（P3-CR-1）
    lpddr5::Tick elapsed = do_transactions(addrs, /*is_write=*/true);
    clock_.advance(ddr_to_xtpu(elapsed));
}

// ---------------------------------------------------------------------------
// read — 同步讀取
//
// 流程：
//   1. 並發提交所有 cacheline read transactions（P3-CR-1）
//   2. 推進 SimClock
//   3. 從 backing_store_ 複製資料到 dst
//
// 時序模擬先於資料複製：模擬「先等待 DRAM latency，再得到資料」的語意。
// ---------------------------------------------------------------------------
void LPDDR5Adapter::read(uint64_t addr, void* dst, size_t size) {
    std::lock_guard<std::mutex> lock(mtx_);
    bounds_check(addr, size);

    // 1. 建立 cacheline-aligned 位址列表並並發提交
    const size_t CL = TimingConfig::CACHELINE_SIZE;
    std::vector<uint64_t> addrs;
    addrs.reserve((size + CL - 1) / CL);
    for (size_t offset = 0; offset < size; offset += CL)
        addrs.push_back(addr + offset);

    lpddr5::Tick elapsed = do_transactions(addrs, /*is_write=*/false);

    // 2. 推進 SimClock
    clock_.advance(ddr_to_xtpu(elapsed));

    // 3. 從 backing store 複製資料到目的地
    std::memcpy(dst, backing_store_.data() + static_cast<size_t>(addr), size);
}

// ---------------------------------------------------------------------------
// print_ddr_stats — P3-CR-8: ostream& 版本（可重導向）
// ---------------------------------------------------------------------------
void LPDDR5Adapter::print_ddr_stats(std::ostream& os, uint8_t channel) const {
    const auto& s = dram_.stats(channel);
    os << "  Channel " << static_cast<int>(channel) << ":\n"
       << "    reads_issued  : " << s.reads_issued  << "\n"
       << "    writes_issued : " << s.writes_issued << "\n"
       << "    row_hits      : " << s.row_hits      << "\n"
       << "    row_misses    : " << s.row_misses    << "\n"
       << "    row_conflicts : " << s.row_conflicts << "\n"
       << "    row_hit_rate  : " << s.row_hit_rate() * 100.0 << " %\n";
}

// 向下相容：無 ostream 版本輸出到 std::cout
void LPDDR5Adapter::print_ddr_stats(uint8_t channel) const {
    print_ddr_stats(std::cout, channel);
}
