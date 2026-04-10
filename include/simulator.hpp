#pragma once
#include <memory>
#include <stdexcept>
#include "engines.hpp"
#include "status_register.hpp"
#include "memory.hpp"
#include "sim_clock.hpp"
#include "commands.hpp"
#include "simulator_config.hpp"   // P3-3: SimulatorConfig + lpddr5::DeviceConfig
#include "lpddr5_adapter.hpp"     // P3-2: LPDDR5Adapter
#include "perf_counters.hpp"      // P4-1: PerfCounters

// ---------------------------------------------------------------------------
// Simulator — xTPU 頂層協調器
//
// 持有所有硬體資源實例，引擎透過抽象介面操作（DIP）：
//   - 記憶體：IMemoryPort / IBufferedMemory (P1-1)
//   - 時序：SimClock + TimingConfig (P1-2)
//
// P3-3: 新增 SimulatorConfig 建構子，支援兩種 system memory backend：
//   SIMPLE  : SystemMemory（固定延遲，向下相容舊測試）
//   LPDDR5  : LPDDR5Adapter（接入 lpddr5-sim，cycle-accurate DDR 時序）
//
// 成員宣告順序 = C++ 初始化順序（必須在 Engines 之前宣告所有依賴資源）：
//   status_reg_ → clock_ → timing_ → sys_mem_ → lpddr5_mem_
//   → scratchpad_ → local_mem_ → sdma_ → idma_ → pu0_ → pu1_
// ---------------------------------------------------------------------------
class Simulator {
public:
    // ── 建構子 ───────────────────────────────────────────────────────────────

    // 向下相容：使用預設/指定 TimingConfig，固定採用 SIMPLE backend
    explicit Simulator(const TimingConfig& timing = TimingConfig{})
        : timing_(timing),
          sys_mem_(std::make_unique<SystemMemory>()),
          lpddr5_mem_(nullptr),
          sdma_(status_reg_, clock_, timing_, *sys_mem_, scratchpad_),
          idma_(status_reg_, clock_, timing_, scratchpad_,
                local_mem_[0], local_mem_[1]),
          pu0_(status_reg_, clock_, timing_, local_mem_[0], STATUS_PU0_CMD_BUSY),
          pu1_(status_reg_, clock_, timing_, local_mem_[1], STATUS_PU1_CMD_BUSY)
    { wire_perf_counters(); }

    // P3-3: SimulatorConfig 建構子，支援 SIMPLE / LPDDR5 backend 切換
    explicit Simulator(const SimulatorConfig& cfg);

    // CR3-6: 單線程使用限制
    // @note NOT thread-safe. Must be called from a single thread only.
    //        dispatch_packet 中 set_busy() 和 push_command() 是非原子組合操作；
    //        若多線程並發呼叫，可能導致 busy bit 被設兩次但 guard 只清一次的 deadlock。
    //        如需多線程場景，需在上層加 external mutex 保護整個 dispatch_packet 呼叫。
    void dispatch_packet(const VLIWPacket& packet);

    // ── 測試 / 驗證用存取器 ──────────────────────────────────────────────────

    StatusRegister&     get_scoreboard()  { return status_reg_; }
    SimClock&           get_clock()       { return clock_; }
    const TimingConfig& get_timing()      { return timing_; }

    // P1-3: 回傳 SystemMemory（SIMPLE backend 測試案例預填資料使用）
    // P3-CR-2: LPDDR5 backend 下呼叫此函式會 throw，防止寫入無效的 sys_mem_。
    //           請改用 get_lpddr5_adapter() 或 get_active_system_mem()。
    SystemMemory& get_system_mem() {
        if (!sys_mem_) throw std::logic_error(
            "get_system_mem() called in LPDDR5 backend mode. "
            "Use get_lpddr5_adapter() or get_active_system_mem() instead.");
        return *sys_mem_;
    }

    // 回傳 active system memory backend（SIMPLE = SystemMemory，LPDDR5 = LPDDR5Adapter）。
    // 供需要對 active backend 做讀寫的情境使用。
    IMemoryPort& get_active_system_mem() {
        if (lpddr5_mem_) return *lpddr5_mem_;
        return *sys_mem_;
    }

    // 回傳介面引用（P1-1）
    IMemoryPort&     get_scratchpad()           { return scratchpad_; }
    IBufferedMemory& get_local_mem(int pu_idx) {
        if (pu_idx < 0 || pu_idx > 1)
            throw std::out_of_range("Simulator: invalid PU index");
        return local_mem_[pu_idx];
    }

    // ── P3-2: LPDDR5 後端存取 ────────────────────────────────────────────────

    // 回傳 LPDDR5Adapter 指標（null 表示使用 SIMPLE backend）
    LPDDR5Adapter* get_lpddr5_adapter() { return lpddr5_mem_.get(); }

    // P3-4: 輸出 LPDDR5-sim 統計報告（SIMPLE backend 時為 no-op）
    void print_ddr_stats(uint8_t channel = 0) {
        if (lpddr5_mem_) lpddr5_mem_->print_ddr_stats(channel);
    }

    // ── P4-1: PerfCounters API ────────────────────────────────────────────────
    PerfCounters&       get_perf_counters()       { return perf_; }
    const PerfCounters& get_perf_counters() const { return perf_; }
    void reset_perf_counters()                    { perf_.reset(); }

    // ── P2-4: Error Bit API（委派到 StatusRegister）─────────────────────────
    // 任何 Engine 發生異常後，STATUS_ERROR 會被設定（sticky）。
    // 呼叫端應在 dispatch 後查詢，並在處理完畢後呼叫 clear_error()。
    bool        has_error()       { return status_reg_.has_error(); }
    std::string get_error_info()  { return status_reg_.get_error_info(); }
    void        clear_error()     { status_reg_.clear_error(); }

private:
    // ── 資源宣告順序 = 初始化順序（C++ 規則） ──────────────────────────────
    // status_reg_, clock_, timing_ 必須在 Engines 之前宣告。
    // sys_mem_ 必須在 lpddr5_mem_ 之前（ternary 選擇依賴宣告順序）。
    // lpddr5_mem_ 必須在 sdma_ 之前（sdma_ ctor 引用 active backend）。
    StatusRegister status_reg_;
    SimClock       clock_;
    TimingConfig   timing_;

    // P3-3 / P3-CR-3: 雙 backend 設計
    //   sys_mem_    : SIMPLE backend 時分配；LPDDR5 backend 時為 nullptr（不浪費 16MB）
    //   lpddr5_mem_ : LPDDR5 backend 時非 null
    // 兩者必定有且僅有一個非 null。
    std::unique_ptr<SystemMemory>  sys_mem_;    // P3-CR-3: unique_ptr，LPDDR5 mode 為 nullptr
    std::unique_ptr<LPDDR5Adapter> lpddr5_mem_;

    Scratchpad   scratchpad_;
    LocalMemory  local_mem_[2];

    SDMAEngine    sdma_;
    IDMAEngine    idma_;
    ComputeEngine pu0_;
    ComputeEngine pu1_;

    PerfCounters  perf_;  // P4-1: 效能計數器（Engines 透過 non-owning pointer 更新）

    // P4-1: 在所有 engine 建構完成後，把 perf_ 的位址告知各 engine
    void wire_perf_counters() {
        sdma_.set_perf_counters(&perf_);
        idma_.set_perf_counters(&perf_);
        pu0_.set_perf_counters(&perf_);
        pu1_.set_perf_counters(&perf_);
    }
};
