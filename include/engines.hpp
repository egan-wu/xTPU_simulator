#pragma once
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <iostream>
#include "commands.hpp"
#include "status_register.hpp"
#include "memory.hpp"
#include "sim_clock.hpp"
#include "perf_counters.hpp"

// ---------------------------------------------------------------------------
// BusyClearGuard (P0-1)
// RAII guard：確保 Engine::process() 結束時（正常路徑或任何異常路徑），
// 對應的 busy bits 必定被清除，防止 wait_on_mask() 永久阻塞（死鎖）。
// ---------------------------------------------------------------------------
struct BusyClearGuard {
    StatusRegister& sr;
    uint32_t mask;

    BusyClearGuard(StatusRegister& sr, uint32_t mask) : sr(sr), mask(mask) {}
    ~BusyClearGuard() {
        if (mask != 0) {
            sr.clear_busy(mask);
        }
    }

    // 禁止拷貝，避免 double-clear
    BusyClearGuard(const BusyClearGuard&) = delete;
    BusyClearGuard& operator=(const BusyClearGuard&) = delete;
};

// ---------------------------------------------------------------------------
// Engine<CmdType> — 所有引擎的模板基底類別
//
// P1-2: 加入 SimClock& clock_ 和 const TimingConfig& timing_，
//        讓每個 engine 可以在 process() 中呼叫 clock_.advance(latency)
//        記錄虛擬延遲，取代 wallclock sleep_for。
// ---------------------------------------------------------------------------
template <typename CmdType>
class Engine {
public:
    Engine(StatusRegister& status_reg, SimClock& clock, const TimingConfig& timing)
        : status_reg(status_reg), clock_(clock), timing_(timing), perf_(nullptr), running(true) {
        worker = std::thread(&Engine::loop, this);
    }

    virtual ~Engine() {
        shutdown();
    }

    void push_command(const CmdType& cmd) {
        {
            std::lock_guard<std::mutex> lock(mtx);
            cmd_queue.push(cmd);
        }
        cv.notify_one();
    }

    // P4-1: 設定效能計數器（可選，nullptr 表示不追蹤）。
    // 由 Simulator 在建構後呼叫，Engine 持有 non-owning pointer。
    void set_perf_counters(PerfCounters* pc) { perf_ = pc; }

    /// Stop the worker thread. Safe to call multiple times.
    /// Derived class destructors MUST call this before their data members
    /// are destroyed, to prevent the worker thread from calling process()
    /// through a destroyed vtable (pure virtual function call).
    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(mtx);
            running.store(false);
        }
        cv.notify_one();
        if (worker.joinable()) worker.join();
    }

protected:
    StatusRegister&     status_reg;
    SimClock&           clock_;    // P1-2: 虛擬時鐘，process() 呼叫 advance() 累積延遲
    const TimingConfig& timing_;   // P1-2: 時序參數表，提供各操作的延遲常數
    PerfCounters*       perf_;     // P4-1: 效能計數器，nullable（nullptr = 不追蹤）

    virtual void process(const CmdType& cmd) = 0;

private:
    std::queue<CmdType> cmd_queue;
    std::thread worker;
    std::mutex mtx;
    std::condition_variable cv;
    std::atomic<bool> running;  // P0-2: atomic，消除對 mutex 保護此 flag 的隱式依賴

    void loop() {
        while (true) {
            CmdType cmd;
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [this] { return !cmd_queue.empty() || !running.load(); });

                if (!running.load() && cmd_queue.empty()) return;

                cmd = cmd_queue.front();
                cmd_queue.pop();
            }

            process(cmd);
        }
    }
};

// ---------------------------------------------------------------------------
// P1-1: Engines 依賴 IMemoryPort / IBufferedMemory 介面（非具體型別）
// P1-2: 建構子加入 SimClock& 和 TimingConfig&，在 process() 中累積 tick
// ---------------------------------------------------------------------------

class SDMAEngine : public Engine<DMA_Command> {
public:
    // P1-3: system_mem 是資料來源（SystemMemory 或未來的 DDRAdapter）
    //        scratchpad  是寫入目的地
    SDMAEngine(StatusRegister& sr, SimClock& clock, const TimingConfig& timing,
               IMemoryPort& system_mem, IMemoryPort& scratchpad);
    ~SDMAEngine() override { shutdown(); }
protected:
    void process(const DMA_Command& cmd) override;
private:
    IMemoryPort& system_mem_;   // P1-3: 真實資料來源
    IMemoryPort& scratchpad_;   // 寫入目的地
};

class IDMAEngine : public Engine<DMA_Command> {
public:
    IDMAEngine(StatusRegister& sr, SimClock& clock, const TimingConfig& timing,
               IMemoryPort& scratchpad, IBufferedMemory& lm0, IBufferedMemory& lm1);
    ~IDMAEngine() override { shutdown(); }
protected:
    void process(const DMA_Command& cmd) override;
private:
    IMemoryPort&     scratchpad_;
    IBufferedMemory& local_mem0_;
    IBufferedMemory& local_mem1_;
};

class ComputeEngine : public Engine<Compute_Command> {
public:
    ComputeEngine(StatusRegister& sr, SimClock& clock, const TimingConfig& timing,
                  IBufferedMemory& lm, uint32_t busy_bit);
    ~ComputeEngine() override { shutdown(); }
protected:
    void process(const Compute_Command& cmd) override;
private:
    IBufferedMemory& local_mem_;
    uint32_t         my_busy_bit_;
};
