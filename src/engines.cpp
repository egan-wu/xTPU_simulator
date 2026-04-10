#include "engines.hpp"
#include <chrono>
#include <vector>

// ---------------------------------------------------------------------------
// 通用輔助：依 size 計算 DMA 延遲 ticks
//   latency = ceil(size / CACHELINE_SIZE) * ticks_per_cacheline
// ---------------------------------------------------------------------------
static SimClock::Tick calc_dma_latency(size_t size,
                                       SimClock::Tick ticks_per_cacheline) {
    if (size == 0 || ticks_per_cacheline == 0) return 0;
    const size_t cls = TimingConfig::CACHELINE_SIZE;
    SimClock::Tick cachelines = (static_cast<SimClock::Tick>(size) + cls - 1) / cls;
    return cachelines * ticks_per_cacheline;
}

// ---------------------------------------------------------------------------
// SDMAEngine — System Memory ↔ Scratchpad (P1-1 + P1-2 + P1-3 + P1-4)
//
// TO_DEVICE  (P1-3): system_mem[src_addr] → scratchpad[dst_addr]
// FROM_DEVICE (P1-4): scratchpad[src_addr] → system_mem[dst_addr]  (writeback)
// ---------------------------------------------------------------------------
SDMAEngine::SDMAEngine(StatusRegister& sr, SimClock& clock,
                       const TimingConfig& timing,
                       IMemoryPort& system_mem, IMemoryPort& scratchpad)
    : Engine(sr, clock, timing), system_mem_(system_mem), scratchpad_(scratchpad) {}

void SDMAEngine::process(const DMA_Command& cmd) {
    if (cmd.type == DMAType::NOP) return;

    // P0-1: 確保任何路徑下 STATUS_SDMA_BUSY 必定被清除
    BusyClearGuard guard(status_reg, STATUS_SDMA_BUSY);

    // P1-2: 依傳輸大小計算 simulated latency（雙向相同代價）
    // P3-2: 若 system_mem_ 後端自行管理時序（LPDDR5Adapter），則跳過固定延遲，
    //        由後端的 read/write 內部呼叫 SimClock::advance()，避免雙重計算。
    SimClock::Tick latency = system_mem_.has_own_timing()
        ? 0
        : calc_dma_latency(cmd.size, timing_.sdma_latency_per_cacheline);
    clock_.advance(latency);

    std::vector<uint8_t> buffer(cmd.size);

    if (cmd.direction == DMADirection::TO_DEVICE) {
        // P1-3: system_mem → scratchpad（load 路徑）
        //        src_addr = System Memory 來源位址
        //        dst_addr = Scratchpad 目的位址
        try {
            system_mem_.read(cmd.src_addr, buffer.data(), cmd.size);
        } catch (const std::exception& e) {
            std::string msg = std::string("[SDMA] System Memory Read Error: ") + e.what();
            std::cerr << msg << " — STATUS_SDMA_BUSY will be cleared by guard" << std::endl;
            status_reg.set_error(msg);  // P2-4
            return;
        }
        try {
            scratchpad_.write(cmd.dst_addr, buffer.data(), cmd.size);
        } catch (const std::exception& e) {
            std::string msg = std::string("[SDMA] Scratchpad Write Error: ") + e.what();
            std::cerr << msg << " — STATUS_SDMA_BUSY will be cleared by guard" << std::endl;
            status_reg.set_error(msg);  // P2-4
            return;
        }
        // P4-1: 記錄傳輸量
        if (perf_) {
            perf_->sdma_bytes_loaded.fetch_add(cmd.size, std::memory_order_relaxed);
            perf_->sdma_ops.fetch_add(1, std::memory_order_relaxed);
        }
    } else {
        // P1-4: scratchpad → system_mem（writeback 路徑）
        //        src_addr = Scratchpad 來源位址
        //        dst_addr = System Memory 目的位址
        try {
            scratchpad_.read(cmd.src_addr, buffer.data(), cmd.size);
        } catch (const std::exception& e) {
            std::string msg = std::string("[SDMA] Scratchpad Read Error (writeback): ") + e.what();
            std::cerr << msg << " — STATUS_SDMA_BUSY will be cleared by guard" << std::endl;
            status_reg.set_error(msg);  // P2-4
            return;
        }
        try {
            system_mem_.write(cmd.dst_addr, buffer.data(), cmd.size);
        } catch (const std::exception& e) {
            std::string msg = std::string("[SDMA] System Memory Write Error (writeback): ") + e.what();
            std::cerr << msg << " — STATUS_SDMA_BUSY will be cleared by guard" << std::endl;
            status_reg.set_error(msg);  // P2-4
            return;
        }
        // P4-1: 記錄傳輸量
        if (perf_) {
            perf_->sdma_bytes_written.fetch_add(cmd.size, std::memory_order_relaxed);
            perf_->sdma_ops.fetch_add(1, std::memory_order_relaxed);
        }
    }
    // guard destructor 清除 STATUS_SDMA_BUSY
}

// ---------------------------------------------------------------------------
// IDMAEngine — Scratchpad ↔ LocalMemory，支援 Broadcast (P1-1 + P1-2 + P1-4)
//
// TO_DEVICE  (P1-1): scratchpad[src_addr] → local_mem[buffer_idx][dst_addr]
//                    支援 Broadcast（TARGET_PU0 | TARGET_PU1 同時寫入）
// FROM_DEVICE (P1-4): local_mem[buffer_idx][src_addr] → scratchpad[dst_addr]
//                    target_mask 指定資料來源 PU；若同時指定兩個 PU，
//                    PU0 優先（PU1 資料覆寫，一般使用單一 target）
// ---------------------------------------------------------------------------
IDMAEngine::IDMAEngine(StatusRegister& sr, SimClock& clock,
                       const TimingConfig& timing,
                       IMemoryPort& scratchpad,
                       IBufferedMemory& lm0, IBufferedMemory& lm1)
    : Engine(sr, clock, timing),
      scratchpad_(scratchpad), local_mem0_(lm0), local_mem1_(lm1) {}

void IDMAEngine::process(const DMA_Command& cmd) {
    if (cmd.type == DMAType::NOP) return;

    // P0-1: 在頂部計算 clear_mask，guard 確保任何路徑都能清除 busy bits
    bool target_pu0 = (cmd.target_mask & TARGET_PU0);
    bool target_pu1 = (cmd.target_mask & TARGET_PU1);
    uint32_t clear_mask = 0;
    if (target_pu0) clear_mask |= STATUS_PU0_DMA_BUSY;
    if (target_pu1) clear_mask |= STATUS_PU1_DMA_BUSY;

    BusyClearGuard guard(status_reg, clear_mask);

    // P1-2: IDMA 延遲（雙向相同代價）
    SimClock::Tick latency = calc_dma_latency(cmd.size,
                                               timing_.idma_latency_per_cacheline);
    clock_.advance(latency);

    std::vector<uint8_t> buffer(cmd.size);

    if (cmd.direction == DMADirection::TO_DEVICE) {
        // P1-1: scratchpad → local_mem（load / broadcast 路徑）

        // Step 1: 從 Scratchpad 讀取（broadcast 只需讀一次）
        try {
            scratchpad_.read(cmd.src_addr, buffer.data(), cmd.size);
        } catch (const std::exception& e) {
            std::string msg = std::string("[IDMA] Scratchpad Read Error: ") + e.what();
            std::cerr << msg << " — busy bits will be cleared by guard" << std::endl;
            status_reg.set_error(msg);  // P2-4
            return;
        }

        // Step 2: Broadcast 寫入目標 Local Memory
        // CR3-1: 任一目標寫入失敗後立即 return（fail-fast），
        //        與 FROM_DEVICE 錯誤路徑一致，避免 error_info 被後續失敗覆寫。
        if (target_pu0) {
            try {
                local_mem0_.write_buffer(cmd.buffer_idx, cmd.dst_addr,
                                         buffer.data(), cmd.size);
            } catch (const std::exception& e) {
                std::string msg = std::string("[IDMA] Write PU0 Error: ") + e.what();
                std::cerr << msg << " — busy bits will be cleared by guard" << std::endl;
                status_reg.set_error(msg);  // P2-4
                return;  // CR3-1: fail-fast，保留 PU0 的 error_info
            }
        }
        if (target_pu1) {
            try {
                local_mem1_.write_buffer(cmd.buffer_idx, cmd.dst_addr,
                                         buffer.data(), cmd.size);
            } catch (const std::exception& e) {
                std::string msg = std::string("[IDMA] Write PU1 Error: ") + e.what();
                std::cerr << msg << " — busy bits will be cleared by guard" << std::endl;
                status_reg.set_error(msg);  // P2-4
                return;  // CR3-1: fail-fast
            }
        }
        // P4-1: 記錄 IDMA load 傳輸量（broadcast 算一次）
        if (perf_) {
            perf_->idma_bytes_loaded.fetch_add(cmd.size, std::memory_order_relaxed);
            perf_->idma_ops.fetch_add(1, std::memory_order_relaxed);
        }
    } else {
        // P1-4: local_mem → scratchpad（writeback 路徑）
        //        src_addr   = Local Memory 來源位址
        //        dst_addr   = Scratchpad 目的位址
        //        buffer_idx = 來源 buffer（0 or 1）
        //        target_mask 指定資料來源 PU（一般為單一目標）

        // Step 1: 從來源 PU Local Memory 讀取資料
        bool data_read = false;
        if (target_pu0) {
            try {
                local_mem0_.read_buffer(cmd.buffer_idx, cmd.src_addr,
                                        buffer.data(), cmd.size);
                data_read = true;
            } catch (const std::exception& e) {
                std::string msg = std::string("[IDMA] Read PU0 Error (writeback): ") + e.what();
                std::cerr << msg << " — busy bits will be cleared by guard" << std::endl;
                status_reg.set_error(msg);  // P2-4
                return;
            }
        }
        if (target_pu1) {
            // 若 PU0 已讀取，PU1 資料覆寫（dual-source writeback 語意由呼叫端管理）
            try {
                local_mem1_.read_buffer(cmd.buffer_idx, cmd.src_addr,
                                        buffer.data(), cmd.size);
                data_read = true;
            } catch (const std::exception& e) {
                std::string msg = std::string("[IDMA] Read PU1 Error (writeback): ") + e.what();
                std::cerr << msg << " — busy bits will be cleared by guard" << std::endl;
                status_reg.set_error(msg);  // P2-4
                return;
            }
        }

        // Step 2: 寫入 Scratchpad
        if (data_read) {
            try {
                scratchpad_.write(cmd.dst_addr, buffer.data(), cmd.size);
            } catch (const std::exception& e) {
                std::string msg = std::string("[IDMA] Scratchpad Write Error (writeback): ") + e.what();
                std::cerr << msg << std::endl;
                status_reg.set_error(msg);  // P2-4
                return;
            }
            // P4-1: 記錄 IDMA writeback 傳輸量
            if (perf_) {
                perf_->idma_bytes_written.fetch_add(cmd.size, std::memory_order_relaxed);
                perf_->idma_ops.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
    // guard destructor 清除 clear_mask 中的所有 busy bits
}

// ---------------------------------------------------------------------------
// ComputeEngine — PU 計算引擎 (P1-1 + P1-2 + P2-1)
//
// P2-1: 當 cmd.length > 0 時，從 local_mem[buffer_idx] 讀取輸入、
//        執行真實運算、將結果寫回 local_mem[buffer_idx]。
//
//  SCALAR : dst[i]   = src[i] + 1              （element-wise，uint8_t 截斷）
//  VECTOR : dst[i]   = src[i] * src[i]         （element-wise 平方，uint8_t 截斷）
//  MATMUL : C = A × B，4×4 uint8_t 矩陣
//           A 在 src_offset，B 在 src_offset + length，C 寫入 dst_offset
//           length = 16 bytes（一個 4×4 矩陣）
// ---------------------------------------------------------------------------
ComputeEngine::ComputeEngine(StatusRegister& sr, SimClock& clock,
                             const TimingConfig& timing,
                             IBufferedMemory& lm, uint32_t busy_bit)
    : Engine(sr, clock, timing), local_mem_(lm), my_busy_bit_(busy_bit) {}

void ComputeEngine::process(const Compute_Command& cmd) {
    if (cmd.type == ComputeType::NOP) return;

    // P0-1: guard 確保 compute 發生任何異常時也能清除 busy bit
    BusyClearGuard guard(status_reg, my_busy_bit_);

    // P1-2: 依 operation type 從 TimingConfig 查表取延遲 ticks
    SimClock::Tick latency = 0;
    switch (cmd.type) {
        case ComputeType::MATMUL: latency = timing_.matmul_latency; break;
        case ComputeType::VECTOR: latency = timing_.vector_latency; break;
        case ComputeType::SCALAR: latency = timing_.scalar_latency; break;
        default: break;
    }

    // 向下相容：若 simulated_duration_ms > 0，用它換算 ticks（覆蓋 type-based latency）
    // 並保留真實 sleep 使並發測試的行為語意不變（test_independent_compute）
    if (cmd.simulated_duration_ms > 0) {
        latency = static_cast<SimClock::Tick>(cmd.simulated_duration_ms)
                  * timing_.ms_to_ticks;
        std::this_thread::sleep_for(
            std::chrono::milliseconds(cmd.simulated_duration_ms));
    }

    clock_.advance(latency);

    // P4-1: 記錄 compute ticks（依 busy_bit 判斷是 PU0 或 PU1）
    if (perf_ && latency > 0) {
        if (my_busy_bit_ == STATUS_PU0_CMD_BUSY)
            perf_->pu0_active_ticks.fetch_add(latency, std::memory_order_relaxed);
        else
            perf_->pu1_active_ticks.fetch_add(latency, std::memory_order_relaxed);
    }

    // P2-1: length == 0 → 只做延遲模擬（向下相容舊測試），不存取記憶體
    if (cmd.length == 0) return;

    try {
        if (cmd.type == ComputeType::SCALAR) {
            // dst[i] = src[i] + 1（逐元素 +1，uint8_t 截斷）
            std::vector<uint8_t> buf(cmd.length);
            local_mem_.read_buffer(cmd.buffer_idx,
                                   static_cast<uint64_t>(cmd.src_offset),
                                   buf.data(), cmd.length);
            for (uint32_t i = 0; i < cmd.length; ++i)
                buf[i] = static_cast<uint8_t>(buf[i] + 1u);
            local_mem_.write_buffer(cmd.buffer_idx,
                                    static_cast<uint64_t>(cmd.dst_offset),
                                    buf.data(), cmd.length);

        } else if (cmd.type == ComputeType::VECTOR) {
            // dst[i] = src[i] * src[i]（逐元素平方，uint8_t 截斷）
            std::vector<uint8_t> buf(cmd.length);
            local_mem_.read_buffer(cmd.buffer_idx,
                                   static_cast<uint64_t>(cmd.src_offset),
                                   buf.data(), cmd.length);
            for (uint32_t i = 0; i < cmd.length; ++i)
                buf[i] = static_cast<uint8_t>(
                    static_cast<uint32_t>(buf[i]) * buf[i]);
            local_mem_.write_buffer(cmd.buffer_idx,
                                    static_cast<uint64_t>(cmd.dst_offset),
                                    buf.data(), cmd.length);

        } else if (cmd.type == ComputeType::MATMUL) {
            // C = A × B，4×4 uint8_t 矩陣，累加用 uint32_t，結果截斷至 uint8_t
            // cmd.length = 16（一個矩陣的 bytes）
            // A 在 src_offset，B 在 src_offset + length，C 寫入 dst_offset
            constexpr uint32_t N = 4;

            // P2-CR-1: 驗證 length == N×N，避免靜默產生錯誤結果
            if (cmd.length != N * N) {
                std::string msg = "[Compute] MATMUL requires length == 16 (4x4 uint8_t matrix), got "
                                + std::to_string(cmd.length);
                std::cerr << msg << " — busy bit will be cleared by guard" << std::endl;
                status_reg.set_error(msg);
                return;  // guard destructor 清除 my_busy_bit_
            }
            std::vector<uint8_t> A(cmd.length), B(cmd.length), C(cmd.length, 0);
            local_mem_.read_buffer(cmd.buffer_idx,
                                   static_cast<uint64_t>(cmd.src_offset),
                                   A.data(), cmd.length);
            local_mem_.read_buffer(cmd.buffer_idx,
                                   static_cast<uint64_t>(cmd.src_offset + cmd.length),
                                   B.data(), cmd.length);
            for (uint32_t i = 0; i < N; ++i) {
                for (uint32_t j = 0; j < N; ++j) {
                    uint32_t acc = 0;
                    for (uint32_t k = 0; k < N; ++k)
                        acc += static_cast<uint32_t>(A[i * N + k])
                             * static_cast<uint32_t>(B[k * N + j]);
                    C[i * N + j] = static_cast<uint8_t>(acc & 0xFF);
                }
            }
            local_mem_.write_buffer(cmd.buffer_idx,
                                    static_cast<uint64_t>(cmd.dst_offset),
                                    C.data(), cmd.length);
        }
    } catch (const std::exception& e) {
        std::string msg = std::string("[Compute] Memory access error: ") + e.what();
        std::cerr << msg << " — busy bit will be cleared by guard" << std::endl;
        status_reg.set_error(msg);  // P2-4
        return;
    }

    // P4-1: 記錄 compute op 次數（length > 0 才算真實計算）
    if (perf_) {
        if (my_busy_bit_ == STATUS_PU0_CMD_BUSY)
            perf_->compute_ops_pu0.fetch_add(1, std::memory_order_relaxed);
        else
            perf_->compute_ops_pu1.fetch_add(1, std::memory_order_relaxed);
    }
    // guard destructor 清除 my_busy_bit_
}
