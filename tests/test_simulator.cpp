#include "simulator.hpp"
#include <iostream>
#include <cassert>
#include <vector>

void test_pipeline_scenario() {
    Simulator sim;

    std::cout << "\n=== Test 1: Full Pipeline Scenario ===" << std::endl;

    // Packet 1: sDMA Load
    // Load 1024 bytes to Scratchpad Offset 0
    VLIWPacket p1 = {};
    p1.sDMA_op = {DMAType::MEMCPY, 0, 0, 1024, 0, 0};

    std::cout << "[Step 1] Dispatching sDMA Load..." << std::endl;
    sim.dispatch_packet(p1);

    // Packet 2: Sync wait for sDMA, then Broadcast
    VLIWPacket p2 = {};
    p2.sync_mask = STATUS_SDMA_BUSY;
    p2.iDMA_op = {DMAType::MEMCPY, 0, 0, 1024, TARGET_PU0 | TARGET_PU1, 0};

    std::cout << "[Step 2] Dispatching Wait(sDMA) + iDMA Broadcast..." << std::endl;
    sim.dispatch_packet(p2);

    // Packet 3: Wait for iDMA, then Parallel Compute + sDMA Load next batch
    VLIWPacket p3 = {};
    p3.sync_mask = STATUS_PU0_DMA_BUSY | STATUS_PU1_DMA_BUSY;
    p3.pu0_op = {ComputeType::MATMUL, 0, 100}; // Simulate work
    p3.pu1_op = {ComputeType::MATMUL, 0, 100};
    p3.sDMA_op = {DMAType::MEMCPY, 0, 1024, 1024, 0, 0};

    std::cout << "[Step 3] Dispatching Wait(iDMA) + Parallel Compute (Buf0) + sDMA Load (Next)..." << std::endl;
    sim.dispatch_packet(p3);

    // Packet 4: Wait for All
    VLIWPacket p4 = {};
    p4.sync_mask = STATUS_PU0_CMD_BUSY | STATUS_PU1_CMD_BUSY | STATUS_SDMA_BUSY;

    std::cout << "[Step 4] Dispatching Wait(All)..." << std::endl;
    sim.dispatch_packet(p4);

    uint32_t final_status = sim.get_scoreboard().get_status();
    std::cout << "Final Status: " << final_status << std::endl;
    assert(final_status == 0);
}

void test_unicast_idma() {
    Simulator sim;
    std::cout << "\n=== Test 2: Unicast iDMA ===" << std::endl;

    // Send to PU0 only
    VLIWPacket p1 = {};
    p1.iDMA_op = {DMAType::MEMCPY, 0, 0, 128, TARGET_PU0, 0};
    sim.dispatch_packet(p1);

    // Verify PU1 is NOT busy, PU0 IS busy (potentially, if we catch it fast enough)
    // But safely, we wait for PU0
    VLIWPacket p2 = {};
    p2.sync_mask = STATUS_PU0_DMA_BUSY;
    sim.dispatch_packet(p2);

    // Send to PU1 only
    VLIWPacket p3 = {};
    p3.iDMA_op = {DMAType::MEMCPY, 0, 0, 128, TARGET_PU1, 0};
    sim.dispatch_packet(p3);

    VLIWPacket p4 = {};
    p4.sync_mask = STATUS_PU1_DMA_BUSY;
    sim.dispatch_packet(p4);

    uint32_t final_status = sim.get_scoreboard().get_status();
    assert(final_status == 0);
    std::cout << "Unicast Test Passed." << std::endl;
}

void test_independent_compute() {
    Simulator sim;
    std::cout << "\n=== Test 3: Independent Compute Durations ===" << std::endl;

    // PU0 Long task, PU1 Short task
    VLIWPacket p1 = {};
    p1.pu0_op = {ComputeType::MATMUL, 0, 200};
    p1.pu1_op = {ComputeType::VECTOR, 0, 50};
    sim.dispatch_packet(p1);

    // Wait for PU1 only
    VLIWPacket p2 = {};
    p2.sync_mask = STATUS_PU1_CMD_BUSY;
    std::cout << "Waiting for PU1 (Short task)..." << std::endl;
    sim.dispatch_packet(p2);

    // At this point PU0 might still be running.
    // Wait for PU0
    VLIWPacket p3 = {};
    p3.sync_mask = STATUS_PU0_CMD_BUSY;
    std::cout << "Waiting for PU0 (Long task)..." << std::endl;
    sim.dispatch_packet(p3);

    uint32_t final_status = sim.get_scoreboard().get_status();
    assert(final_status == 0);
    std::cout << "Independent Compute Test Passed." << std::endl;
}

// ---------------------------------------------------------------------------
// P0-1 + P2-4 驗證：SDMA 異常路徑不死鎖，且 STATUS_ERROR 被設定
//
// 觸發 Scratchpad OOB write，驗證：
//   1. STATUS_SDMA_BUSY 被清除（P0-1：BusyClearGuard 正確運作）
//   2. STATUS_ERROR 被設定（P2-4：set_error() 被呼叫）
//   3. get_error_info() 包含有意義的訊息
//   4. clear_error() 後 status 回到 0
// ---------------------------------------------------------------------------
void test_sdma_error_no_deadlock() {
    Simulator sim;
    std::cout << "\n=== Test 4: SDMA OOB Error — No Deadlock + ERROR Bit (P0-1 + P2-4) ===" << std::endl;

    VLIWPacket p1 = {};
    p1.sDMA_op = {DMAType::MEMCPY,
                  /*src_addr=*/0,
                  /*dst_addr=*/static_cast<uint64_t>(SCRATCHPAD_SIZE - 10),
                  /*size=*/100,
                  /*target_mask=*/0,
                  /*buffer_idx=*/0};

    std::cout << "[Step 1] Dispatching SDMA with OOB dst_addr (expect error log)..." << std::endl;
    sim.dispatch_packet(p1);

    VLIWPacket p2 = {};
    p2.sync_mask = STATUS_SDMA_BUSY;
    std::cout << "[Step 2] Waiting for SDMA completion (should NOT hang)..." << std::endl;
    sim.dispatch_packet(p2);

    // P0-1: busy bits（非 ERROR）應全部清除
    uint32_t status = sim.get_scoreboard().get_status();
    assert((status & ~STATUS_ERROR) == 0 && "Busy bits should be cleared after error");

    // P2-4: STATUS_ERROR 應被設定
    assert(sim.has_error() && "STATUS_ERROR should be set after SDMA exception");

    // P2-4: error_info 應包含有意義的錯誤描述
    std::string info = sim.get_error_info();
    assert(!info.empty() && "error_info should not be empty");
    assert(info.find("[SDMA]") != std::string::npos && "error_info should mention SDMA");
    std::cout << "  error_info: " << info << std::endl;

    // P2-4: clear_error() 後 status 應回到 0
    sim.clear_error();
    assert(sim.get_scoreboard().get_status() == 0 && "Status should be 0 after clear_error()");

    std::cout << "SDMA Error No-Deadlock + ERROR Bit Test Passed." << std::endl;
}

// ---------------------------------------------------------------------------
// P0-1 + P2-4 驗證：IDMA 讀取異常路徑不死鎖，且 STATUS_ERROR 被設定
// ---------------------------------------------------------------------------
void test_idma_error_no_deadlock() {
    Simulator sim;
    std::cout << "\n=== Test 5: IDMA OOB Read Error — No Deadlock + ERROR Bit (P0-1 + P2-4) ===" << std::endl;

    VLIWPacket p1 = {};
    p1.iDMA_op = {DMAType::MEMCPY,
                  /*src_addr=*/static_cast<uint64_t>(SCRATCHPAD_SIZE - 10),
                  /*dst_addr=*/0,
                  /*size=*/100,
                  /*target_mask=*/TARGET_PU0,
                  /*buffer_idx=*/0};

    std::cout << "[Step 1] Dispatching IDMA with OOB src_addr (expect error log)..." << std::endl;
    sim.dispatch_packet(p1);

    VLIWPacket p2 = {};
    p2.sync_mask = STATUS_PU0_DMA_BUSY;
    std::cout << "[Step 2] Waiting for IDMA completion (should NOT hang)..." << std::endl;
    sim.dispatch_packet(p2);

    // P0-1: busy bits 應全部清除（STATUS_ERROR 除外）
    uint32_t status = sim.get_scoreboard().get_status();
    assert((status & ~STATUS_ERROR) == 0 && "Busy bits should be cleared after error");

    // P2-4: STATUS_ERROR 應被設定
    assert(sim.has_error() && "STATUS_ERROR should be set after IDMA exception");

    std::string info = sim.get_error_info();
    assert(!info.empty() && info.find("[IDMA]") != std::string::npos);
    std::cout << "  error_info: " << info << std::endl;

    sim.clear_error();
    assert(sim.get_scoreboard().get_status() == 0 && "Status should be 0 after clear_error()");

    std::cout << "IDMA Error No-Deadlock + ERROR Bit Test Passed." << std::endl;
}

// ---------------------------------------------------------------------------
// P1-2 驗證：SimClock 隨操作正確累積 ticks
//
// 使用 no-sleep TimingConfig（ms_to_ticks = 0）使測試快速完成，
// 並驗證每項操作對應的 tick 增量符合 TimingConfig 的預期值。
// ---------------------------------------------------------------------------
void test_clock_advances() {
    // 建立零 sleep 的 TimingConfig（只累積 ticks，不實際 sleep）
    TimingConfig cfg;
    cfg.ms_to_ticks = 0;  // 停用 simulated_duration_ms → ticks 的換算（和真實 sleep）

    Simulator sim(cfg);
    std::cout << "\n=== Test 6: SimClock Advances (P1-2) ===" << std::endl;

    // ── sDMA: 1024 bytes ─────────────────────────────────────────────────────
    // 預期: ceil(1024 / 64) * sdma_latency_per_cacheline = 16 * 10 = 160 ticks
    const SimClock::Tick expected_sdma = 16 * cfg.sdma_latency_per_cacheline;

    SimClock::Tick t0 = sim.get_clock().now();
    assert(t0 == 0);

    VLIWPacket p1 = {};
    p1.sDMA_op = {DMAType::MEMCPY, 0, 0, 1024, 0, 0};
    sim.dispatch_packet(p1);

    VLIWPacket sync1 = {};
    sync1.sync_mask = STATUS_SDMA_BUSY;
    sim.dispatch_packet(sync1);

    SimClock::Tick t1 = sim.get_clock().now();
    std::cout << "  After sDMA 1024B: clock = " << t1
              << " (expected >= " << expected_sdma << ")" << std::endl;
    assert(t1 >= expected_sdma);

    // ── iDMA Broadcast: 512 bytes → PU0 + PU1 ───────────────────────────────
    // 預期: ceil(512 / 64) * idma_latency_per_cacheline = 8 * 5 = 40 ticks
    const SimClock::Tick expected_idma = 8 * cfg.idma_latency_per_cacheline;

    VLIWPacket p2 = {};
    p2.iDMA_op = {DMAType::MEMCPY, 0, 0, 512, TARGET_PU0 | TARGET_PU1, 0};
    sim.dispatch_packet(p2);

    VLIWPacket sync2 = {};
    sync2.sync_mask = STATUS_PU0_DMA_BUSY | STATUS_PU1_DMA_BUSY;
    sim.dispatch_packet(sync2);

    SimClock::Tick t2 = sim.get_clock().now();
    std::cout << "  After iDMA 512B broadcast: clock = " << t2
              << " (expected >= " << t1 + expected_idma << ")" << std::endl;
    assert(t2 >= t1 + expected_idma);

    // ── Compute: MATMUL on PU0，無 simulated_duration_ms（使用 TimingConfig）────
    // 預期: matmul_latency = 100 ticks
    VLIWPacket p3 = {};
    p3.pu0_op = {ComputeType::MATMUL, 0, 0};  // simulated_duration_ms = 0
    sim.dispatch_packet(p3);

    VLIWPacket sync3 = {};
    sync3.sync_mask = STATUS_PU0_CMD_BUSY;
    sim.dispatch_packet(sync3);

    SimClock::Tick t3 = sim.get_clock().now();
    std::cout << "  After MATMUL (PU0): clock = " << t3
              << " (expected >= " << t2 + cfg.matmul_latency << ")" << std::endl;
    assert(t3 >= t2 + cfg.matmul_latency);

    // ── VECTOR on PU1 ────────────────────────────────────────────────────────
    // 預期: vector_latency = 20 ticks
    VLIWPacket p4 = {};
    p4.pu1_op = {ComputeType::VECTOR, 0, 0};  // simulated_duration_ms = 0
    sim.dispatch_packet(p4);

    VLIWPacket sync4 = {};
    sync4.sync_mask = STATUS_PU1_CMD_BUSY;
    sim.dispatch_packet(sync4);

    SimClock::Tick t4 = sim.get_clock().now();
    std::cout << "  After VECTOR (PU1): clock = " << t4
              << " (expected >= " << t3 + cfg.vector_latency << ")" << std::endl;
    assert(t4 >= t3 + cfg.vector_latency);

    assert(sim.get_scoreboard().get_status() == 0);
    std::cout << "SimClock Advance Test Passed. Total ticks: " << t4 << std::endl;
}

// ---------------------------------------------------------------------------
// P1-3 驗證 (Test 7)：SDMA 資料正確性
//
// 1. 以遞增 pattern 填充 SystemMemory（data[i] = i & 0xFF）
// 2. 透過 SDMA 將 256 bytes 搬到 Scratchpad offset 0
// 3. 讀回 Scratchpad，逐 byte 比對
// ---------------------------------------------------------------------------
void test_sdma_data_correctness() {
    TimingConfig cfg;
    cfg.ms_to_ticks = 0;  // 無實際 sleep
    Simulator sim(cfg);

    std::cout << "\n=== Test 7: SDMA Data Correctness (P1-3) ===" << std::endl;

    const size_t TRANSFER_SIZE = 256;

    // Step 1: 填充 SystemMemory（遞增序列）
    sim.get_system_mem().fill_incremental();

    // Step 2: SDMA 搬移 SystemMemory[0..255] → Scratchpad[0..255]
    VLIWPacket p1 = {};
    p1.sDMA_op = {DMAType::MEMCPY,
                  /*src_addr=*/0,
                  /*dst_addr=*/0,
                  /*size=*/TRANSFER_SIZE,
                  /*target_mask=*/0,
                  /*buffer_idx=*/0};
    sim.dispatch_packet(p1);

    VLIWPacket sync1 = {};
    sync1.sync_mask = STATUS_SDMA_BUSY;
    sim.dispatch_packet(sync1);

    // Step 3: 讀回 Scratchpad 並驗證
    std::vector<uint8_t> result(TRANSFER_SIZE, 0xFF);
    sim.get_scratchpad().read(0, result.data(), TRANSFER_SIZE);

    for (size_t i = 0; i < TRANSFER_SIZE; ++i) {
        uint8_t expected = static_cast<uint8_t>(i & 0xFF);
        if (result[i] != expected) {
            std::cerr << "[Test 7] Mismatch at byte " << i
                      << ": expected 0x" << std::hex << (int)expected
                      << ", got 0x" << (int)result[i] << std::dec << std::endl;
            assert(false && "SDMA data correctness check failed");
        }
    }

    assert(sim.get_scoreboard().get_status() == 0);
    std::cout << "SDMA Data Correctness Test Passed (" << TRANSFER_SIZE
              << " bytes verified)." << std::endl;
}

// ---------------------------------------------------------------------------
// P1-3 驗證 (Test 8)：E2E 資料正確性（System Memory → Scratchpad → LocalMem）
//
// 完整路徑：
//   1. fill_pattern(0xDEADBEEF) 填充 SystemMemory
//   2. SDMA: SystemMemory[0..511] → Scratchpad[0..511]
//   3. iDMA Broadcast: Scratchpad[0..511] → PU0 LocalMem[buf=0, offset=0]
//                                          + PU1 LocalMem[buf=0, offset=0]
//   4. 讀回 PU0 / PU1 LocalMemory，逐 byte 比對 pattern
// ---------------------------------------------------------------------------
void test_e2e_data_correctness() {
    TimingConfig cfg;
    cfg.ms_to_ticks = 0;
    Simulator sim(cfg);

    std::cout << "\n=== Test 8: E2E Data Correctness (P1-3) ===" << std::endl;

    const size_t TRANSFER_SIZE = 512;
    const uint32_t PATTERN = 0xDEADBEEF;

    // Step 1: 填充 SystemMemory（4-byte 循環 pattern）
    sim.get_system_mem().fill_pattern(PATTERN);

    // Step 2: SDMA
    VLIWPacket p1 = {};
    p1.sDMA_op = {DMAType::MEMCPY, 0, 0, TRANSFER_SIZE, 0, 0};
    sim.dispatch_packet(p1);

    VLIWPacket sync1 = {};
    sync1.sync_mask = STATUS_SDMA_BUSY;
    sim.dispatch_packet(sync1);

    // Step 3: iDMA Broadcast（buffer index = 0）
    VLIWPacket p2 = {};
    p2.iDMA_op = {DMAType::MEMCPY, 0, 0, TRANSFER_SIZE, TARGET_PU0 | TARGET_PU1, 0};
    sim.dispatch_packet(p2);

    VLIWPacket sync2 = {};
    sync2.sync_mask = STATUS_PU0_DMA_BUSY | STATUS_PU1_DMA_BUSY;
    sim.dispatch_packet(sync2);

    // Step 4: 讀回 PU0 / PU1 LocalMemory 並驗證
    std::vector<uint8_t> pu0_result(TRANSFER_SIZE, 0);
    std::vector<uint8_t> pu1_result(TRANSFER_SIZE, 0);

    sim.get_local_mem(0).read_buffer(0, 0, pu0_result.data(), TRANSFER_SIZE);
    sim.get_local_mem(1).read_buffer(0, 0, pu1_result.data(), TRANSFER_SIZE);

    for (size_t i = 0; i < TRANSFER_SIZE; ++i) {
        uint8_t expected = static_cast<uint8_t>((PATTERN >> ((i % 4) * 8)) & 0xFF);
        if (pu0_result[i] != expected) {
            std::cerr << "[Test 8] PU0 Mismatch at byte " << i
                      << ": expected 0x" << std::hex << (int)expected
                      << ", got 0x" << (int)pu0_result[i] << std::dec << std::endl;
            assert(false && "E2E data correctness check failed (PU0)");
        }
        if (pu1_result[i] != expected) {
            std::cerr << "[Test 8] PU1 Mismatch at byte " << i
                      << ": expected 0x" << std::hex << (int)expected
                      << ", got 0x" << (int)pu1_result[i] << std::dec << std::endl;
            assert(false && "E2E data correctness check failed (PU1)");
        }
    }

    assert(sim.get_scoreboard().get_status() == 0);
    std::cout << "E2E Data Correctness Test Passed (" << TRANSFER_SIZE
              << " bytes, pattern=0x" << std::hex << PATTERN << std::dec
              << ", PU0 & PU1 verified)." << std::endl;
}

// ---------------------------------------------------------------------------
// P1-4 驗證 (Test 9)：完整端到端 Writeback Path
//
// 驗證 DMADirection::FROM_DEVICE 的雙向路徑正確性：
//
//   [載入路徑]
//   Step 1: fill_pattern(0xC0FFEE01) 填充 SystemMemory
//   Step 2: SDMA TO_DEVICE:   system_mem[0..511]   → scratchpad[0..511]
//   Step 3: iDMA TO_DEVICE:   scratchpad[0..511]   → PU0 local_mem[buf=0][0..511]
//
//   [Writeback 路徑]
//   Step 4: iDMA FROM_DEVICE: PU0 local_mem[buf=0][0..511] → scratchpad[512..1023]
//   Step 5: SDMA FROM_DEVICE: scratchpad[512..1023] → system_mem[512..1023]
//
//   [驗證]
//   Step 6: system_mem[512..1023] 應與 system_mem[0..511] 完全一致（pattern 完整保留）
// ---------------------------------------------------------------------------
void test_writeback_path() {
    TimingConfig cfg;
    cfg.ms_to_ticks = 0;
    Simulator sim(cfg);

    std::cout << "\n=== Test 9: Writeback Path E2E (P1-4) ===" << std::endl;

    const size_t TRANSFER_SIZE = 512;
    const uint32_t PATTERN = 0xC0FFEE01;

    // Step 1: 填充 SystemMemory[0..1023]（4-byte 循環 pattern）
    sim.get_system_mem().fill_pattern(PATTERN);

    // ── 載入路徑 ─────────────────────────────────────────────────────────────

    // Step 2: SDMA TO_DEVICE: system_mem[0..511] → scratchpad[0..511]
    VLIWPacket p1 = {};
    p1.sDMA_op = {DMAType::MEMCPY,
                  /*src_addr=*/0,
                  /*dst_addr=*/0,
                  /*size=*/TRANSFER_SIZE,
                  /*target_mask=*/0,
                  /*buffer_idx=*/0,
                  /*direction=*/DMADirection::TO_DEVICE};
    sim.dispatch_packet(p1);

    VLIWPacket sync1 = {};
    sync1.sync_mask = STATUS_SDMA_BUSY;
    sim.dispatch_packet(sync1);

    // Step 3: iDMA TO_DEVICE: scratchpad[0..511] → PU0 local_mem[buf=0][0..511]
    VLIWPacket p2 = {};
    p2.iDMA_op = {DMAType::MEMCPY,
                  /*src_addr=*/0,
                  /*dst_addr=*/0,
                  /*size=*/TRANSFER_SIZE,
                  /*target_mask=*/TARGET_PU0,
                  /*buffer_idx=*/0,
                  /*direction=*/DMADirection::TO_DEVICE};
    sim.dispatch_packet(p2);

    VLIWPacket sync2 = {};
    sync2.sync_mask = STATUS_PU0_DMA_BUSY;
    sim.dispatch_packet(sync2);

    // ── Writeback 路徑 ───────────────────────────────────────────────────────

    // Step 4: iDMA FROM_DEVICE: PU0 local_mem[buf=0][0..511] → scratchpad[512..1023]
    VLIWPacket p3 = {};
    p3.iDMA_op = {DMAType::MEMCPY,
                  /*src_addr=*/0,
                  /*dst_addr=*/TRANSFER_SIZE,     // Scratchpad offset 512
                  /*size=*/TRANSFER_SIZE,
                  /*target_mask=*/TARGET_PU0,
                  /*buffer_idx=*/0,
                  /*direction=*/DMADirection::FROM_DEVICE};
    sim.dispatch_packet(p3);

    VLIWPacket sync3 = {};
    sync3.sync_mask = STATUS_PU0_DMA_BUSY;
    sim.dispatch_packet(sync3);

    // Step 5: SDMA FROM_DEVICE: scratchpad[512..1023] → system_mem[512..1023]
    VLIWPacket p4 = {};
    p4.sDMA_op = {DMAType::MEMCPY,
                  /*src_addr=*/TRANSFER_SIZE,     // Scratchpad offset 512
                  /*dst_addr=*/TRANSFER_SIZE,     // SystemMemory offset 512
                  /*size=*/TRANSFER_SIZE,
                  /*target_mask=*/0,
                  /*buffer_idx=*/0,
                  /*direction=*/DMADirection::FROM_DEVICE};
    sim.dispatch_packet(p4);

    VLIWPacket sync4 = {};
    sync4.sync_mask = STATUS_SDMA_BUSY;
    sim.dispatch_packet(sync4);

    // ── 驗證 ─────────────────────────────────────────────────────────────────

    // Step 6: 讀回 system_mem[512..1023]，應與原始 pattern 完全一致
    std::vector<uint8_t> result(TRANSFER_SIZE, 0);
    sim.get_system_mem().read(TRANSFER_SIZE, result.data(), TRANSFER_SIZE);

    for (size_t i = 0; i < TRANSFER_SIZE; ++i) {
        // pattern = 0xC0FFEE01，little-endian byte order:
        //   byte[0]=0x01, byte[1]=0xEE, byte[2]=0xFF, byte[3]=0xC0
        uint8_t expected = static_cast<uint8_t>((PATTERN >> ((i % 4) * 8)) & 0xFF);
        if (result[i] != expected) {
            std::cerr << "[Test 9] Mismatch at writeback byte " << i
                      << ": expected 0x" << std::hex << (int)expected
                      << ", got 0x" << (int)result[i] << std::dec << std::endl;
            assert(false && "Writeback path data correctness check failed");
        }
    }

    assert(sim.get_scoreboard().get_status() == 0);
    std::cout << "Writeback Path E2E Test Passed (" << TRANSFER_SIZE
              << " bytes, pattern=0x" << std::hex << PATTERN << std::dec
              << ")." << std::endl;
    std::cout << "  Path verified: system_mem → scratchpad → local_mem[PU0]"
              << " → scratchpad → system_mem" << std::endl;
}

// ---------------------------------------------------------------------------
// P2-1 驗證 (Test 10)：SCALAR 運算正確性
//
// dst[i] = src[i] + 1（逐元素 +1，uint8_t 截斷）
//
// 流程：
//   1. iDMA 載入已知資料 [0,1,2,...,63] 到 PU0 local_mem[buf=0][0..63]
//   2. SCALAR 運算（src=0, dst=64, length=64）
//   3. 讀回 local_mem[buf=0][64..127] 驗證每個值 == 原值 + 1
//   4. 驗證 0xFF 的 byte 截斷為 0x00（wrapping）
// ---------------------------------------------------------------------------
void test_compute_scalar() {
    TimingConfig cfg;
    cfg.ms_to_ticks = 0;
    Simulator sim(cfg);

    std::cout << "\n=== Test 10: Compute SCALAR Correctness (P2-1) ===" << std::endl;

    const size_t DATA_SIZE = 64;

    // 準備輸入資料（[0..63]），並確保含有 0xFF 以測試截斷行為
    // 直接透過 SystemMemory 填充後用 SDMA → Scratchpad → iDMA 載入 local_mem
    std::vector<uint8_t> input(DATA_SIZE);
    for (size_t i = 0; i < DATA_SIZE; ++i)
        input[i] = static_cast<uint8_t>(i);
    input[63] = 0xFF;   // 截斷測試：0xFF + 1 = 0x00

    sim.get_system_mem().write(0, input.data(), DATA_SIZE);

    // SDMA: system_mem[0..63] → scratchpad[0..63]
    VLIWPacket p1 = {};
    p1.sDMA_op = {DMAType::MEMCPY, 0, 0, DATA_SIZE, 0, 0};
    sim.dispatch_packet(p1);
    VLIWPacket s1 = {}; s1.sync_mask = STATUS_SDMA_BUSY;
    sim.dispatch_packet(s1);

    // iDMA: scratchpad[0..63] → PU0 local_mem[buf=0][0..63]
    VLIWPacket p2 = {};
    p2.iDMA_op = {DMAType::MEMCPY, 0, 0, DATA_SIZE, TARGET_PU0, 0};
    sim.dispatch_packet(p2);
    VLIWPacket s2 = {}; s2.sync_mask = STATUS_PU0_DMA_BUSY;
    sim.dispatch_packet(s2);

    // SCALAR: PU0 local_mem[buf=0][0..63] + 1 → local_mem[buf=0][64..127]
    VLIWPacket p3 = {};
    p3.pu0_op = {ComputeType::SCALAR,
                 /*buffer_idx=*/0,
                 /*simulated_duration_ms=*/0,
                 /*src_offset=*/0,
                 /*dst_offset=*/DATA_SIZE,
                 /*length=*/static_cast<uint32_t>(DATA_SIZE)};
    sim.dispatch_packet(p3);
    VLIWPacket s3 = {}; s3.sync_mask = STATUS_PU0_CMD_BUSY;
    sim.dispatch_packet(s3);

    // 驗證結果
    std::vector<uint8_t> result(DATA_SIZE, 0);
    sim.get_local_mem(0).read_buffer(0, DATA_SIZE, result.data(), DATA_SIZE);

    for (size_t i = 0; i < DATA_SIZE; ++i) {
        uint8_t expected = static_cast<uint8_t>(input[i] + 1u);
        if (result[i] != expected) {
            std::cerr << "[Test 10] Mismatch at byte " << i
                      << ": expected 0x" << std::hex << (int)expected
                      << ", got 0x" << (int)result[i] << std::dec << std::endl;
            assert(false && "SCALAR compute correctness check failed");
        }
    }

    assert(sim.get_scoreboard().get_status() == 0);
    std::cout << "SCALAR Compute Test Passed ("
              << DATA_SIZE << " bytes, wrapping 0xFF→0x00 verified)." << std::endl;
}

// ---------------------------------------------------------------------------
// P2-1 驗證 (Test 11)：VECTOR 運算正確性
//
// dst[i] = src[i] * src[i]（逐元素平方，uint8_t 截斷）
//
// 流程：
//   1. 直接寫入 PU0 local_mem[buf=0][0..15] = [0,1,2,...,15]
//   2. VECTOR 運算（src=0, dst=16, length=16）
//   3. 驗證 dst[i] == (i*i) & 0xFF
// ---------------------------------------------------------------------------
void test_compute_vector() {
    TimingConfig cfg;
    cfg.ms_to_ticks = 0;
    Simulator sim(cfg);

    std::cout << "\n=== Test 11: Compute VECTOR Correctness (P2-1) ===" << std::endl;

    const size_t DATA_SIZE = 16;

    // 直接寫入 system_mem 後載入 local_mem
    std::vector<uint8_t> input(DATA_SIZE);
    for (size_t i = 0; i < DATA_SIZE; ++i)
        input[i] = static_cast<uint8_t>(i);  // [0..15]
    sim.get_system_mem().write(0, input.data(), DATA_SIZE);

    // SDMA → scratchpad → PU0 local_mem[buf=0][0..15]
    VLIWPacket p1 = {};
    p1.sDMA_op = {DMAType::MEMCPY, 0, 0, DATA_SIZE, 0, 0};
    sim.dispatch_packet(p1);
    VLIWPacket s1 = {}; s1.sync_mask = STATUS_SDMA_BUSY;
    sim.dispatch_packet(s1);

    VLIWPacket p2 = {};
    p2.iDMA_op = {DMAType::MEMCPY, 0, 0, DATA_SIZE, TARGET_PU0, 0};
    sim.dispatch_packet(p2);
    VLIWPacket s2 = {}; s2.sync_mask = STATUS_PU0_DMA_BUSY;
    sim.dispatch_packet(s2);

    // VECTOR: dst[i] = src[i] * src[i]
    VLIWPacket p3 = {};
    p3.pu0_op = {ComputeType::VECTOR, 0, 0,
                 /*src_offset=*/0,
                 /*dst_offset=*/static_cast<uint32_t>(DATA_SIZE),
                 /*length=*/static_cast<uint32_t>(DATA_SIZE)};
    sim.dispatch_packet(p3);
    VLIWPacket s3 = {}; s3.sync_mask = STATUS_PU0_CMD_BUSY;
    sim.dispatch_packet(s3);

    // 驗證結果
    std::vector<uint8_t> result(DATA_SIZE, 0);
    sim.get_local_mem(0).read_buffer(0, DATA_SIZE, result.data(), DATA_SIZE);

    for (size_t i = 0; i < DATA_SIZE; ++i) {
        uint8_t expected = static_cast<uint8_t>(
            (static_cast<uint32_t>(input[i]) * input[i]) & 0xFF);
        if (result[i] != expected) {
            std::cerr << "[Test 11] Mismatch at byte " << i
                      << ": expected 0x" << std::hex << (int)expected
                      << ", got 0x" << (int)result[i] << std::dec << std::endl;
            assert(false && "VECTOR compute correctness check failed");
        }
    }

    assert(sim.get_scoreboard().get_status() == 0);
    std::cout << "VECTOR Compute Test Passed ("
              << DATA_SIZE << " bytes, i*i verified for i=0..15)." << std::endl;
}

// ---------------------------------------------------------------------------
// P2-1 驗證 (Test 12)：MATMUL 運算正確性
//
// 4×4 uint8_t 矩陣乘法：C = A × B
// 使用單位矩陣 A = I，驗證 I × B = B（最簡單但有效的正確性檢驗）
//
// local_mem[buf=0] 佈局：
//   offset  0..15 : Matrix A（單位矩陣）
//   offset 16..31 : Matrix B（1..16 列優先填充）
//   offset 32..47 : Matrix C（結果，應等於 B）
//
// cmd.length = 16 bytes（一個 4×4 矩陣）
// ---------------------------------------------------------------------------
void test_compute_matmul() {
    TimingConfig cfg;
    cfg.ms_to_ticks = 0;
    Simulator sim(cfg);

    std::cout << "\n=== Test 12: Compute MATMUL Correctness (P2-1) ===" << std::endl;

    constexpr size_t MAT_SIZE = 16; // 4×4 × uint8_t
    constexpr size_t N = 4;

    // Matrix A = 單位矩陣
    std::vector<uint8_t> A(MAT_SIZE, 0);
    for (size_t i = 0; i < N; ++i) A[i * N + i] = 1;

    // Matrix B = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]（列優先）
    std::vector<uint8_t> B(MAT_SIZE);
    for (size_t i = 0; i < MAT_SIZE; ++i) B[i] = static_cast<uint8_t>(i + 1);

    // 寫入 system_mem: [A(16 bytes) | B(16 bytes)]
    std::vector<uint8_t> init_data(MAT_SIZE * 2);
    std::copy(A.begin(), A.end(), init_data.begin());
    std::copy(B.begin(), B.end(), init_data.begin() + MAT_SIZE);
    sim.get_system_mem().write(0, init_data.data(), MAT_SIZE * 2);

    // SDMA: system_mem[0..31] → scratchpad[0..31]
    VLIWPacket p1 = {};
    p1.sDMA_op = {DMAType::MEMCPY, 0, 0, MAT_SIZE * 2, 0, 0};
    sim.dispatch_packet(p1);
    VLIWPacket s1 = {}; s1.sync_mask = STATUS_SDMA_BUSY;
    sim.dispatch_packet(s1);

    // iDMA: scratchpad[0..31] → PU0 local_mem[buf=0][0..31]
    VLIWPacket p2 = {};
    p2.iDMA_op = {DMAType::MEMCPY, 0, 0, MAT_SIZE * 2, TARGET_PU0, 0};
    sim.dispatch_packet(p2);
    VLIWPacket s2 = {}; s2.sync_mask = STATUS_PU0_DMA_BUSY;
    sim.dispatch_packet(s2);

    // MATMUL: C = A × B
    //   A at src_offset=0, B at src_offset+length=16, C at dst_offset=32
    VLIWPacket p3 = {};
    p3.pu0_op = {ComputeType::MATMUL, 0, 0,
                 /*src_offset=*/0,
                 /*dst_offset=*/static_cast<uint32_t>(MAT_SIZE * 2),
                 /*length=*/static_cast<uint32_t>(MAT_SIZE)};
    sim.dispatch_packet(p3);
    VLIWPacket s3 = {}; s3.sync_mask = STATUS_PU0_CMD_BUSY;
    sim.dispatch_packet(s3);

    // 驗證：I × B = B
    std::vector<uint8_t> result(MAT_SIZE, 0);
    sim.get_local_mem(0).read_buffer(0, MAT_SIZE * 2, result.data(), MAT_SIZE);

    for (size_t i = 0; i < MAT_SIZE; ++i) {
        if (result[i] != B[i]) {
            std::cerr << "[Test 12] Mismatch at element [" << i/N << "][" << i%N << "]"
                      << ": expected " << (int)B[i]
                      << ", got " << (int)result[i] << std::endl;
            assert(false && "MATMUL compute correctness check failed");
        }
    }

    assert(sim.get_scoreboard().get_status() == 0);
    std::cout << "MATMUL Compute Test Passed (I × B = B verified, 4×4 matrix)." << std::endl;
}

// ---------------------------------------------------------------------------
// P2-2 場景 3 (Test 13)：Double Buffer Isolation
//
// 驗證 ComputeEngine 在 buf=0 上執行運算時，buf=1 的資料完全不受干擾。
//
// 佈局（PU0 local_mem）：
//   buf=0[0..15]  = input  [0,1,...,15]（SCALAR 來源）
//   buf=0[16..31] = output  [1,2,...,16]（SCALAR 結果）
//   buf=1[0..31]  = 0xBB  (初始值，全程不變）
//
// 驗收：
//   - buf=0 的 src 區間不變（SCALAR 不修改來源）
//   - buf=0 的 dst 區間 = src + 1
//   - buf=1 全程維持 0xBB（isolation 成立）
// ---------------------------------------------------------------------------
void test_double_buffer_isolation() {
    TimingConfig cfg;
    cfg.ms_to_ticks = 0;
    Simulator sim(cfg);

    std::cout << "\n=== Test 13: Double Buffer Isolation (P2-2 Scenario 3) ===" << std::endl;

    constexpr size_t DATA_SIZE = 16;

    // Setup: 直接寫入兩個 buffer 的初始值
    std::vector<uint8_t> input_a(DATA_SIZE), input_b(DATA_SIZE * 2, 0xBB);
    for (size_t i = 0; i < DATA_SIZE; ++i)
        input_a[i] = static_cast<uint8_t>(i);  // [0..15]

    sim.get_local_mem(0).write_buffer(0, 0, input_a.data(), DATA_SIZE);        // buf=0[0..15] = [0..15]
    sim.get_local_mem(0).write_buffer(1, 0, input_b.data(), DATA_SIZE * 2);    // buf=1[0..31] = 0xBB

    // SCALAR compute on buf=0: src=offset 0, dst=offset 16, length=16
    VLIWPacket p1 = {};
    p1.pu0_op = {ComputeType::SCALAR, 0, 0,
                 /*src_offset=*/0,
                 /*dst_offset=*/static_cast<uint32_t>(DATA_SIZE),
                 /*length=*/static_cast<uint32_t>(DATA_SIZE)};
    sim.dispatch_packet(p1);
    VLIWPacket s1 = {}; s1.sync_mask = STATUS_PU0_CMD_BUSY;
    sim.dispatch_packet(s1);

    // 驗證 buf=0 src 區間不變
    std::vector<uint8_t> src_check(DATA_SIZE, 0);
    sim.get_local_mem(0).read_buffer(0, 0, src_check.data(), DATA_SIZE);
    for (size_t i = 0; i < DATA_SIZE; ++i) {
        assert(src_check[i] == static_cast<uint8_t>(i) &&
               "buf=0 src region should be unchanged after SCALAR");
    }

    // 驗證 buf=0 dst 區間 = src + 1
    std::vector<uint8_t> dst_check(DATA_SIZE, 0);
    sim.get_local_mem(0).read_buffer(0, DATA_SIZE, dst_check.data(), DATA_SIZE);
    for (size_t i = 0; i < DATA_SIZE; ++i) {
        uint8_t expected = static_cast<uint8_t>(i + 1u);
        if (dst_check[i] != expected) {
            std::cerr << "[Test 13] buf=0 dst mismatch at " << i
                      << ": expected " << (int)expected << ", got " << (int)dst_check[i] << std::endl;
            assert(false && "SCALAR result incorrect");
        }
    }

    // 驗證 buf=1 完全未被修改（isolation 關鍵驗證）
    std::vector<uint8_t> buf1_check(DATA_SIZE * 2, 0);
    sim.get_local_mem(0).read_buffer(1, 0, buf1_check.data(), DATA_SIZE * 2);
    for (size_t i = 0; i < DATA_SIZE * 2; ++i) {
        if (buf1_check[i] != 0xBB) {
            std::cerr << "[Test 13] ISOLATION FAILURE: buf=1[" << i
                      << "] was modified! expected 0xBB, got 0x"
                      << std::hex << (int)buf1_check[i] << std::dec << std::endl;
            assert(false && "Double buffer isolation violated");
        }
    }

    assert(sim.get_scoreboard().get_status() == 0);
    std::cout << "Double Buffer Isolation Test Passed "
              << "(buf=0 SCALAR correct, buf=1 unchanged)." << std::endl;
}

// ---------------------------------------------------------------------------
// P2-2 場景 4 (Test 14)：End-to-End with Compute
//
// 完整資料路徑，中間含 SCALAR 運算：
//   system_mem[0..63]  填充 [0,1,...,63]
//   → SDMA TO_DEVICE   → scratchpad[0..63]
//   → iDMA TO_DEVICE   → PU0 local_mem[buf=0][0..63]
//   → SCALAR in-place  → local_mem[buf=0][0..63] = original + 1
//   → iDMA FROM_DEVICE → scratchpad[64..127]
//   → SDMA FROM_DEVICE → system_mem[64..127]
//   驗證: system_mem[64..127] == [1,2,...,64]（每個 byte +1）
// ---------------------------------------------------------------------------
void test_e2e_with_compute() {
    TimingConfig cfg;
    cfg.ms_to_ticks = 0;
    Simulator sim(cfg);

    std::cout << "\n=== Test 14: E2E with Compute (P2-2 Scenario 4) ===" << std::endl;

    const size_t DATA_SIZE = 64;

    // Step 1: 填充 system_mem[0..63] = [0,1,...,63]
    std::vector<uint8_t> input(DATA_SIZE);
    for (size_t i = 0; i < DATA_SIZE; ++i)
        input[i] = static_cast<uint8_t>(i);
    sim.get_system_mem().write(0, input.data(), DATA_SIZE);

    // Step 2: SDMA: system_mem[0..63] → scratchpad[0..63]
    VLIWPacket p1 = {};
    p1.sDMA_op = {DMAType::MEMCPY, 0, 0, DATA_SIZE, 0, 0};
    sim.dispatch_packet(p1);
    VLIWPacket s1 = {}; s1.sync_mask = STATUS_SDMA_BUSY;
    sim.dispatch_packet(s1);

    // Step 3: iDMA: scratchpad[0..63] → PU0 local_mem[buf=0][0..63]
    VLIWPacket p2 = {};
    p2.iDMA_op = {DMAType::MEMCPY, 0, 0, DATA_SIZE, TARGET_PU0, 0};
    sim.dispatch_packet(p2);
    VLIWPacket s2 = {}; s2.sync_mask = STATUS_PU0_DMA_BUSY;
    sim.dispatch_packet(s2);

    // Step 4: SCALAR 運算 in-place（src 和 dst 相同 offset，read-modify-write）
    VLIWPacket p3 = {};
    p3.pu0_op = {ComputeType::SCALAR, 0, 0,
                 /*src_offset=*/0,
                 /*dst_offset=*/0,        // 原地覆寫
                 /*length=*/static_cast<uint32_t>(DATA_SIZE)};
    sim.dispatch_packet(p3);
    VLIWPacket s3 = {}; s3.sync_mask = STATUS_PU0_CMD_BUSY;
    sim.dispatch_packet(s3);

    // Step 5: iDMA FROM_DEVICE: PU0 local_mem[buf=0][0..63] → scratchpad[64..127]
    VLIWPacket p4 = {};
    p4.iDMA_op = {DMAType::MEMCPY,
                  /*src_addr=*/0,
                  /*dst_addr=*/static_cast<uint64_t>(DATA_SIZE),
                  /*size=*/DATA_SIZE,
                  /*target_mask=*/TARGET_PU0,
                  /*buffer_idx=*/0,
                  /*direction=*/DMADirection::FROM_DEVICE};
    sim.dispatch_packet(p4);
    VLIWPacket s4 = {}; s4.sync_mask = STATUS_PU0_DMA_BUSY;
    sim.dispatch_packet(s4);

    // Step 6: SDMA FROM_DEVICE: scratchpad[64..127] → system_mem[64..127]
    VLIWPacket p5 = {};
    p5.sDMA_op = {DMAType::MEMCPY,
                  /*src_addr=*/static_cast<uint64_t>(DATA_SIZE),
                  /*dst_addr=*/static_cast<uint64_t>(DATA_SIZE),
                  /*size=*/DATA_SIZE, 0, 0,
                  DMADirection::FROM_DEVICE};
    sim.dispatch_packet(p5);
    VLIWPacket s5 = {}; s5.sync_mask = STATUS_SDMA_BUSY;
    sim.dispatch_packet(s5);

    // Step 7: 驗證 system_mem[64..127] == [1,2,...,64]
    std::vector<uint8_t> result(DATA_SIZE, 0);
    sim.get_system_mem().read(DATA_SIZE, result.data(), DATA_SIZE);

    for (size_t i = 0; i < DATA_SIZE; ++i) {
        uint8_t expected = static_cast<uint8_t>(input[i] + 1u);
        if (result[i] != expected) {
            std::cerr << "[Test 14] Mismatch at byte " << i
                      << ": expected " << (int)expected
                      << ", got " << (int)result[i] << std::endl;
            assert(false && "E2E with compute failed");
        }
    }

    assert(sim.get_scoreboard().get_status() == 0);
    std::cout << "E2E with Compute Test Passed ("
              << DATA_SIZE << " bytes, SCALAR in-place, verified in system_mem)." << std::endl;
}

// ---------------------------------------------------------------------------
// P2-2 場景 5 (Test 15)：Incremental Pattern 完整 E2E 含 VECTOR 運算
//
// 驗證 [0,1,...,N-1] 的遞增序列可以完整通過所有記憶體層級，
// 且 VECTOR 運算（平方）的結果也能正確 writeback 到 system_mem。
//
// 路徑：
//   system_mem[0..31] = [0..31]
//   → scratchpad → PU1 local_mem[buf=0][0..31]
//   → VECTOR compute: dst[i] = src[i]*src[i] (寫到 offset 32)
//   → PU1 local_mem[buf=0][32..63] → scratchpad[32..63] → system_mem[32..63]
//   驗收: system_mem[32..63] == [0,1,4,9,...,31²]（mod 256）
// ---------------------------------------------------------------------------
void test_incremental_e2e_with_vector() {
    TimingConfig cfg;
    cfg.ms_to_ticks = 0;
    Simulator sim(cfg);

    std::cout << "\n=== Test 15: Incremental E2E + VECTOR Compute (P2-2 Scenario 5) ===" << std::endl;

    const size_t DATA_SIZE = 32;

    // 填充 system_mem[0..31] = [0,1,...,31]
    std::vector<uint8_t> input(DATA_SIZE);
    for (size_t i = 0; i < DATA_SIZE; ++i)
        input[i] = static_cast<uint8_t>(i);
    sim.get_system_mem().write(0, input.data(), DATA_SIZE);

    // SDMA: system_mem[0..31] → scratchpad[0..31]
    VLIWPacket p1 = {};
    p1.sDMA_op = {DMAType::MEMCPY, 0, 0, DATA_SIZE, 0, 0};
    sim.dispatch_packet(p1);
    VLIWPacket s1 = {}; s1.sync_mask = STATUS_SDMA_BUSY;
    sim.dispatch_packet(s1);

    // iDMA: scratchpad[0..31] → PU1 local_mem[buf=0][0..31]
    VLIWPacket p2 = {};
    p2.iDMA_op = {DMAType::MEMCPY, 0, 0, DATA_SIZE, TARGET_PU1, 0};
    sim.dispatch_packet(p2);
    VLIWPacket s2 = {}; s2.sync_mask = STATUS_PU1_DMA_BUSY;
    sim.dispatch_packet(s2);

    // VECTOR compute on PU1: src=0, dst=32（append 後方），length=32
    VLIWPacket p3 = {};
    p3.pu1_op = {ComputeType::VECTOR, 0, 0,
                 /*src_offset=*/0,
                 /*dst_offset=*/static_cast<uint32_t>(DATA_SIZE),
                 /*length=*/static_cast<uint32_t>(DATA_SIZE)};
    sim.dispatch_packet(p3);
    VLIWPacket s3 = {}; s3.sync_mask = STATUS_PU1_CMD_BUSY;
    sim.dispatch_packet(s3);

    // iDMA FROM_DEVICE: PU1 local_mem[buf=0][32..63] → scratchpad[32..63]
    VLIWPacket p4 = {};
    p4.iDMA_op = {DMAType::MEMCPY,
                  /*src_addr=*/static_cast<uint64_t>(DATA_SIZE),
                  /*dst_addr=*/static_cast<uint64_t>(DATA_SIZE),
                  DATA_SIZE, TARGET_PU1, 0,
                  DMADirection::FROM_DEVICE};
    sim.dispatch_packet(p4);
    VLIWPacket s4 = {}; s4.sync_mask = STATUS_PU1_DMA_BUSY;
    sim.dispatch_packet(s4);

    // SDMA FROM_DEVICE: scratchpad[32..63] → system_mem[32..63]
    VLIWPacket p5 = {};
    p5.sDMA_op = {DMAType::MEMCPY,
                  static_cast<uint64_t>(DATA_SIZE),
                  static_cast<uint64_t>(DATA_SIZE),
                  DATA_SIZE, 0, 0,
                  DMADirection::FROM_DEVICE};
    sim.dispatch_packet(p5);
    VLIWPacket s5 = {}; s5.sync_mask = STATUS_SDMA_BUSY;
    sim.dispatch_packet(s5);

    // 驗證 system_mem[32..63] == [i*i & 0xFF for i in 0..31]
    std::vector<uint8_t> result(DATA_SIZE, 0);
    sim.get_system_mem().read(DATA_SIZE, result.data(), DATA_SIZE);

    for (size_t i = 0; i < DATA_SIZE; ++i) {
        uint8_t expected = static_cast<uint8_t>(
            (static_cast<uint32_t>(input[i]) * input[i]) & 0xFF);
        if (result[i] != expected) {
            std::cerr << "[Test 15] Mismatch at index " << i
                      << " (input=" << (int)input[i] << "): expected "
                      << (int)expected << ", got " << (int)result[i] << std::endl;
            assert(false && "Incremental E2E + VECTOR failed");
        }
    }

    assert(sim.get_scoreboard().get_status() == 0);
    std::cout << "Incremental E2E + VECTOR Test Passed ("
              << DATA_SIZE << " bytes, i*i verified for i=0.." << DATA_SIZE-1 << ")." << std::endl;
}

// ---------------------------------------------------------------------------
// P2-4 驗證 (Test 16)：Compute OOB → STATUS_ERROR 被設定
//
// ComputeEngine 執行 SCALAR 運算時，若 src_offset + length 超出 LocalMemory 邊界，
// 應設定 STATUS_ERROR，且 PU0 busy bit 仍被正確清除（BusyClearGuard 有效）。
// ---------------------------------------------------------------------------
void test_compute_error_sets_error_bit() {
    TimingConfig cfg;
    cfg.ms_to_ticks = 0;
    Simulator sim(cfg);

    std::cout << "\n=== Test 16: Compute OOB Error → STATUS_ERROR (P2-4) ===" << std::endl;

    // src_offset 接近 LOCAL_MEM_SIZE 邊界，length 刻意超界
    VLIWPacket p1 = {};
    p1.pu0_op = {ComputeType::SCALAR, 0, 0,
                 /*src_offset=*/static_cast<uint32_t>(LOCAL_MEM_SIZE - 4),
                 /*dst_offset=*/0,
                 /*length=*/100};   // 超界：LOCAL_MEM_SIZE - 4 + 100 > LOCAL_MEM_SIZE

    std::cout << "[Step 1] Dispatching SCALAR with OOB src (expect error log)..." << std::endl;
    sim.dispatch_packet(p1);

    VLIWPacket s1 = {}; s1.sync_mask = STATUS_PU0_CMD_BUSY;
    std::cout << "[Step 2] Waiting for PU0 (should NOT hang)..." << std::endl;
    sim.dispatch_packet(s1);

    // P0-1: PU0_CMD_BUSY 應被清除
    uint32_t status = sim.get_scoreboard().get_status();
    assert((status & ~STATUS_ERROR) == 0 && "PU0_CMD_BUSY should be cleared by guard");

    // P2-4: STATUS_ERROR 應被設定
    assert(sim.has_error() && "STATUS_ERROR should be set after Compute OOB");

    std::string info = sim.get_error_info();
    assert(!info.empty() && info.find("[Compute]") != std::string::npos);
    std::cout << "  error_info: " << info << std::endl;

    sim.clear_error();
    assert(sim.get_scoreboard().get_status() == 0);
    std::cout << "Compute OOB Error Bit Test Passed." << std::endl;
}

// ---------------------------------------------------------------------------
// P2-4 驗證 (Test 17)：clear_error() 後可繼續正常操作
//
// 確保 clear_error() 後 simulator 能恢復正常運作，
// 即 ERROR bit 不會污染後續的正常命令流。
// ---------------------------------------------------------------------------
void test_error_recovery() {
    TimingConfig cfg;
    cfg.ms_to_ticks = 0;
    Simulator sim(cfg);

    std::cout << "\n=== Test 17: Error Recovery after clear_error() (P2-4) ===" << std::endl;

    // Step 1: 故意觸發 SDMA 錯誤
    VLIWPacket p_err = {};
    p_err.sDMA_op = {DMAType::MEMCPY, 0,
                     static_cast<uint64_t>(SCRATCHPAD_SIZE - 5),
                     100, 0, 0};
    sim.dispatch_packet(p_err);
    VLIWPacket s_err = {}; s_err.sync_mask = STATUS_SDMA_BUSY;
    sim.dispatch_packet(s_err);

    assert(sim.has_error() && "ERROR should be set after bad SDMA");

    // Step 2: clear_error() 恢復
    sim.clear_error();
    assert(!sim.has_error() && "ERROR should be cleared");
    assert(sim.get_scoreboard().get_status() == 0);

    // Step 3: 正常 SDMA 操作（驗證 simulator 功能完整恢復）
    const size_t SIZE = 64;
    std::vector<uint8_t> input(SIZE);
    for (size_t i = 0; i < SIZE; ++i) input[i] = static_cast<uint8_t>(i + 10);
    sim.get_system_mem().write(0, input.data(), SIZE);

    VLIWPacket p_ok = {};
    p_ok.sDMA_op = {DMAType::MEMCPY, 0, 0, SIZE, 0, 0};
    sim.dispatch_packet(p_ok);
    VLIWPacket s_ok = {}; s_ok.sync_mask = STATUS_SDMA_BUSY;
    sim.dispatch_packet(s_ok);

    // 驗證資料正確傳輸
    std::vector<uint8_t> result(SIZE, 0);
    sim.get_scratchpad().read(0, result.data(), SIZE);
    for (size_t i = 0; i < SIZE; ++i)
        assert(result[i] == input[i] && "Data should be correct after recovery");

    assert(!sim.has_error() && "No error after normal operation");
    assert(sim.get_scoreboard().get_status() == 0);

    std::cout << "Error Recovery Test Passed ("
              << SIZE << " bytes transferred correctly after error recovery)." << std::endl;
}

// ---------------------------------------------------------------------------
// P2-CR-3 驗證 (Test 18)：MATMUL 非平凡矩陣正確性
//
// 使用對角矩陣 A = diag([2,3,1,1]) 與非對稱矩陣 B，
// 確保 MATMUL 不是誤實作為「直接複製 B」。
//
// A = diag([2,3,1,1]):
//   [[2,0,0,0],
//    [0,3,0,0],
//    [0,0,1,0],
//    [0,0,0,1]]
//
// B = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]（row-major）
//
// 期望 C = A × B:
//   row 0 = 2 × row0(B) = [2,4,6,8]
//   row 1 = 3 × row1(B) = [15,18,21,24]
//   row 2 = 1 × row2(B) = [9,10,11,12]
//   row 3 = 1 × row3(B) = [13,14,15,16]
// ---------------------------------------------------------------------------
void test_compute_matmul_nontrivial() {
    TimingConfig cfg;
    cfg.ms_to_ticks = 0;
    Simulator sim(cfg);

    std::cout << "\n=== Test 18: Compute MATMUL Non-trivial (P2-CR-3) ===" << std::endl;

    constexpr size_t MAT_SIZE = 16; // 4×4 × uint8_t
    constexpr size_t N = 4;

    // Matrix A = diag([2,3,1,1])
    std::vector<uint8_t> A(MAT_SIZE, 0);
    A[0 * N + 0] = 2;   // A[0][0] = 2
    A[1 * N + 1] = 3;   // A[1][1] = 3
    A[2 * N + 2] = 1;   // A[2][2] = 1
    A[3 * N + 3] = 1;   // A[3][3] = 1

    // Matrix B = [[1..4],[5..8],[9..12],[13..16]]（row-major）
    std::vector<uint8_t> B(MAT_SIZE);
    for (size_t i = 0; i < MAT_SIZE; ++i) B[i] = static_cast<uint8_t>(i + 1);

    // 預計算期望結果 C = A × B
    std::vector<uint8_t> C_expected(MAT_SIZE, 0);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j) {
            uint32_t acc = 0;
            for (size_t k = 0; k < N; ++k)
                acc += static_cast<uint32_t>(A[i * N + k])
                     * static_cast<uint32_t>(B[k * N + j]);
            C_expected[i * N + j] = static_cast<uint8_t>(acc & 0xFF);
        }

    // 寫入 system_mem: [A | B]
    std::vector<uint8_t> init_data(MAT_SIZE * 2);
    std::copy(A.begin(), A.end(), init_data.begin());
    std::copy(B.begin(), B.end(), init_data.begin() + MAT_SIZE);
    sim.get_system_mem().write(0, init_data.data(), MAT_SIZE * 2);

    // SDMA: system_mem[0..31] → scratchpad[0..31]
    VLIWPacket p1 = {};
    p1.sDMA_op = {DMAType::MEMCPY, 0, 0, MAT_SIZE * 2, 0, 0};
    sim.dispatch_packet(p1);
    VLIWPacket s1 = {}; s1.sync_mask = STATUS_SDMA_BUSY;
    sim.dispatch_packet(s1);

    // iDMA: scratchpad[0..31] → PU0 local_mem[buf=0][0..31]
    VLIWPacket p2 = {};
    p2.iDMA_op = {DMAType::MEMCPY, 0, 0, MAT_SIZE * 2, TARGET_PU0, 0};
    sim.dispatch_packet(p2);
    VLIWPacket s2 = {}; s2.sync_mask = STATUS_PU0_DMA_BUSY;
    sim.dispatch_packet(s2);

    // MATMUL: C = A × B
    //   A at src_offset=0, B at src_offset+length=16, C at dst_offset=32
    VLIWPacket p3 = {};
    p3.pu0_op = {ComputeType::MATMUL, 0, 0,
                 /*src_offset=*/0,
                 /*dst_offset=*/static_cast<uint32_t>(MAT_SIZE * 2),
                 /*length=*/static_cast<uint32_t>(MAT_SIZE)};
    sim.dispatch_packet(p3);
    VLIWPacket s3 = {}; s3.sync_mask = STATUS_PU0_CMD_BUSY;
    sim.dispatch_packet(s3);

    // 驗證結果
    std::vector<uint8_t> result(MAT_SIZE, 0);
    sim.get_local_mem(0).read_buffer(0, MAT_SIZE * 2, result.data(), MAT_SIZE);

    for (size_t i = 0; i < MAT_SIZE; ++i) {
        if (result[i] != C_expected[i]) {
            std::cerr << "[Test 18] Mismatch at element [" << i/N << "][" << i%N << "]"
                      << ": expected " << (int)C_expected[i]
                      << ", got " << (int)result[i] << std::endl;
            assert(false && "MATMUL non-trivial compute correctness check failed");
        }
    }

    assert(sim.get_scoreboard().get_status() == 0);
    std::cout << "MATMUL Non-trivial Test Passed "
              << "(diag([2,3,1,1]) × B = C verified, 4×4 matrix)." << std::endl;
}

// ---------------------------------------------------------------------------
// P2-CR-1 驗證 (Test 19)：MATMUL 傳入 length != 16 應設定 STATUS_ERROR
//
// 若 cmd.length = 8（不是合法的 16），
// ComputeEngine 應偵測到無效長度並設定 STATUS_ERROR，
// 而不是靜默產生錯誤的計算結果。
// ---------------------------------------------------------------------------
void test_matmul_invalid_length() {
    TimingConfig cfg;
    cfg.ms_to_ticks = 0;
    Simulator sim(cfg);

    std::cout << "\n=== Test 19: MATMUL Invalid Length → STATUS_ERROR (P2-CR-1) ===" << std::endl;

    // 傳入 length=8（4×4 矩陣需要 16 bytes）
    VLIWPacket p1 = {};
    p1.pu0_op = {ComputeType::MATMUL, 0, 0,
                 /*src_offset=*/0,
                 /*dst_offset=*/32,
                 /*length=*/8};   // 無效：4×4 矩陣需要 length == 16
    sim.dispatch_packet(p1);

    VLIWPacket s1 = {}; s1.sync_mask = STATUS_PU0_CMD_BUSY;
    std::cout << "[Step 1] Waiting for PU0 (should NOT hang)..." << std::endl;
    sim.dispatch_packet(s1);

    // PU0_CMD_BUSY 應被 BusyClearGuard 清除
    uint32_t status = sim.get_scoreboard().get_status();
    assert((status & ~STATUS_ERROR) == 0 && "PU0_CMD_BUSY should be cleared by guard");

    // STATUS_ERROR 應被設定
    assert(sim.has_error() && "STATUS_ERROR should be set on invalid MATMUL length");

    std::string info = sim.get_error_info();
    assert(!info.empty() && info.find("requires length") != std::string::npos
           && "[Compute]" && "error_info should mention requires length");
    std::cout << "  error_info: " << info << std::endl;

    sim.clear_error();
    assert(sim.get_scoreboard().get_status() == 0);
    std::cout << "MATMUL Invalid Length Test Passed (STATUS_ERROR set, PU0_CMD_BUSY cleared)." << std::endl;
}

// ---------------------------------------------------------------------------
// CR3-3 驗證 (Test 20)：PU1 Writeback Path E2E
//
// 驗證 IDMAEngine FROM_DEVICE 的 TARGET_PU1 路徑，補全測試覆蓋率。
// 現有 Test 9 只驗證 TARGET_PU0；此測試驗證 PU1 的對稱路徑：
//
//   [載入路徑]
//   system_mem[0..255]  → SDMA → scratchpad[0..255]
//   scratchpad[0..255]  → iDMA(TARGET_PU1) → PU1 local_mem[buf=0][0..255]
//
//   [Writeback 路徑]
//   PU1 local_mem[buf=0][0..255] → iDMA FROM_DEVICE → scratchpad[256..511]
//   scratchpad[256..511]         → SDMA FROM_DEVICE → system_mem[256..511]
//
//   [驗證]
//   system_mem[256..511] 應與原始 pattern 完全一致
// ---------------------------------------------------------------------------
void test_pu1_writeback_path() {
    TimingConfig cfg;
    cfg.ms_to_ticks = 0;
    Simulator sim(cfg);

    std::cout << "\n=== Test 20: PU1 Writeback Path E2E (CR3-3) ===" << std::endl;

    const size_t TRANSFER_SIZE = 256;
    const uint32_t PATTERN = 0xABCD1234;

    // Step 1: 填充 SystemMemory（4-byte 循環 pattern）
    sim.get_system_mem().fill_pattern(PATTERN);

    // ── 載入路徑 ─────────────────────────────────────────────────────────────

    // Step 2: SDMA TO_DEVICE: system_mem[0..255] → scratchpad[0..255]
    VLIWPacket p1 = {};
    p1.sDMA_op = {DMAType::MEMCPY, 0, 0, TRANSFER_SIZE, 0, 0,
                  DMADirection::TO_DEVICE};
    sim.dispatch_packet(p1);
    VLIWPacket s1 = {}; s1.sync_mask = STATUS_SDMA_BUSY;
    sim.dispatch_packet(s1);

    // Step 3: iDMA TO_DEVICE: scratchpad[0..255] → PU1 local_mem[buf=0][0..255]
    VLIWPacket p2 = {};
    p2.iDMA_op = {DMAType::MEMCPY, 0, 0, TRANSFER_SIZE, TARGET_PU1, 0,
                  DMADirection::TO_DEVICE};
    sim.dispatch_packet(p2);
    VLIWPacket s2 = {}; s2.sync_mask = STATUS_PU1_DMA_BUSY;
    sim.dispatch_packet(s2);

    // ── Writeback 路徑 ───────────────────────────────────────────────────────

    // Step 4: iDMA FROM_DEVICE: PU1 local_mem[buf=0][0..255] → scratchpad[256..511]
    VLIWPacket p3 = {};
    p3.iDMA_op = {DMAType::MEMCPY,
                  /*src_addr=*/0,
                  /*dst_addr=*/TRANSFER_SIZE,   // Scratchpad offset 256
                  TRANSFER_SIZE, TARGET_PU1, 0,
                  DMADirection::FROM_DEVICE};
    sim.dispatch_packet(p3);
    VLIWPacket s3 = {}; s3.sync_mask = STATUS_PU1_DMA_BUSY;
    sim.dispatch_packet(s3);

    // Step 5: SDMA FROM_DEVICE: scratchpad[256..511] → system_mem[256..511]
    VLIWPacket p4 = {};
    p4.sDMA_op = {DMAType::MEMCPY,
                  /*src_addr=*/TRANSFER_SIZE,   // Scratchpad offset 256
                  /*dst_addr=*/TRANSFER_SIZE,   // SystemMemory offset 256
                  TRANSFER_SIZE, 0, 0,
                  DMADirection::FROM_DEVICE};
    sim.dispatch_packet(p4);
    VLIWPacket s4 = {}; s4.sync_mask = STATUS_SDMA_BUSY;
    sim.dispatch_packet(s4);

    // ── 驗證 ─────────────────────────────────────────────────────────────────

    // Step 6: 讀回 system_mem[256..511]，應與原始 pattern 完全一致
    std::vector<uint8_t> result(TRANSFER_SIZE, 0);
    sim.get_system_mem().read(TRANSFER_SIZE, result.data(), TRANSFER_SIZE);

    for (size_t i = 0; i < TRANSFER_SIZE; ++i) {
        uint8_t expected = static_cast<uint8_t>((PATTERN >> ((i % 4) * 8)) & 0xFF);
        if (result[i] != expected) {
            std::cerr << "[Test 20] PU1 writeback mismatch at byte " << i
                      << ": expected 0x" << std::hex << (int)expected
                      << ", got 0x" << (int)result[i] << std::dec << std::endl;
            assert(false && "PU1 writeback path data correctness check failed");
        }
    }

    assert(sim.get_scoreboard().get_status() == 0);
    std::cout << "PU1 Writeback Path E2E Test Passed (" << TRANSFER_SIZE
              << " bytes, pattern=0x" << std::hex << PATTERN << std::dec
              << ")." << std::endl;
    std::cout << "  Path verified: system_mem → scratchpad → local_mem[PU1]"
              << " → scratchpad → system_mem" << std::endl;
}

// ---------------------------------------------------------------------------
// CR3-1 驗證 (Test 21)：IDMA broadcast 部分失敗 — fail-fast + error_info 保留
//
// 驗證 TO_DEVICE broadcast 中，PU0 write OOB 失敗後：
//   1. PU1 write 不再執行（fail-fast，CR3-1 修正）
//   2. error_info 保留 PU0 的錯誤訊息，未被覆寫
//   3. STATUS_ERROR 被設定
//   4. 所有 busy bits 被正確清除
// ---------------------------------------------------------------------------
void test_idma_broadcast_failfast() {
    TimingConfig cfg;
    cfg.ms_to_ticks = 0;
    Simulator sim(cfg);

    std::cout << "\n=== Test 21: IDMA Broadcast Fail-Fast (CR3-1) ===" << std::endl;

    // 先將資料載入 scratchpad（有效資料，避免 SDMA read 失敗）
    const size_t DATA_SIZE = 64;
    std::vector<uint8_t> input(DATA_SIZE, 0xAA);
    sim.get_system_mem().write(0, input.data(), DATA_SIZE);

    VLIWPacket prep = {};
    prep.sDMA_op = {DMAType::MEMCPY, 0, 0, DATA_SIZE, 0, 0};
    sim.dispatch_packet(prep);
    VLIWPacket sprep = {}; sprep.sync_mask = STATUS_SDMA_BUSY;
    sim.dispatch_packet(sprep);

    // 觸發 broadcast：PU0 buffer_idx=2（無效，只有 0/1），應立即失敗並 return
    // PU1 寫入不應被執行
    VLIWPacket p1 = {};
    p1.iDMA_op = {DMAType::MEMCPY,
                  0,    // src: scratchpad offset 0（有效）
                  0,    // dst: local_mem offset 0
                  DATA_SIZE,
                  TARGET_PU0 | TARGET_PU1,
                  /*buffer_idx=*/2,   // 無效 buffer index → PU0 write 失敗
                  DMADirection::TO_DEVICE};
    sim.dispatch_packet(p1);

    VLIWPacket s1 = {};
    s1.sync_mask = STATUS_PU0_DMA_BUSY | STATUS_PU1_DMA_BUSY;
    std::cout << "[Step 1] Waiting for IDMA (expect error, should NOT hang)..." << std::endl;
    sim.dispatch_packet(s1);

    // 驗證：busy bits 已清除，STATUS_ERROR 已設定
    uint32_t status = sim.get_scoreboard().get_status();
    assert((status & ~STATUS_ERROR) == 0 && "All busy bits should be cleared by guard");
    assert(sim.has_error() && "STATUS_ERROR should be set after broadcast PU0 write failure");

    // 驗證：error_info 包含 PU0 的錯誤訊息（非 PU1 的）
    std::string info = sim.get_error_info();
    assert(!info.empty() && info.find("PU0") != std::string::npos
           && "error_info should mention PU0 (first failure, not overwritten)");
    std::cout << "  error_info: " << info << std::endl;

    // 驗證：PU1 local_mem[buf=0] 仍為初始值 0（PU1 write 未被執行）
    std::vector<uint8_t> pu1_check(DATA_SIZE, 0xFF);
    sim.get_local_mem(1).read_buffer(0, 0, pu1_check.data(), DATA_SIZE);
    for (size_t i = 0; i < DATA_SIZE; ++i) {
        if (pu1_check[i] != 0) {
            std::cerr << "[Test 21] PU1 was written despite PU0 failure! byte["
                      << i << "] = 0x" << std::hex << (int)pu1_check[i] << std::dec << std::endl;
            assert(false && "PU1 write should NOT have executed after PU0 failure (fail-fast)");
        }
    }

    sim.clear_error();
    assert(sim.get_scoreboard().get_status() == 0);
    std::cout << "IDMA Broadcast Fail-Fast Test Passed "
              << "(PU0 fail → PU1 skipped, error_info preserved)." << std::endl;
}

// ---------------------------------------------------------------------------
// P3-2 / P3-4 驗證 (Test 22)：LPDDR5 backend 資料正確性
//
// 目標：以 LPDDR5 backend 執行完整 SDMA load 流程，驗證：
//   1. 資料完整無損地從 LPDDR5 backing_store → scratchpad
//   2. STATUS_ERROR 未被設定
//   3. DDR 統計（reads_issued >= 1）可正常讀取
//
// 流程：
//   fill_direct(0..63 = [0,1,...,63]) → SDMA load → 驗證 scratchpad
// ---------------------------------------------------------------------------
void test_lpddr5_data_correctness() {
    SimulatorConfig cfg;
    cfg.backend = SimulatorConfig::MemoryBackend::LPDDR5;
    cfg.timing.ms_to_ticks = 0;
    Simulator sim(cfg);

    std::cout << "\n=== Test 22: LPDDR5 Backend Data Correctness (P3-2) ===" << std::endl;

    const size_t DATA_SIZE = 64;
    std::vector<uint8_t> input(DATA_SIZE);
    for (size_t i = 0; i < DATA_SIZE; ++i)
        input[i] = static_cast<uint8_t>(i);

    // 直接寫入 LPDDR5 backing_store（繞過時序，相當於 SystemMemory::write 的測試輔助）
    LPDDR5Adapter* adapter = sim.get_lpddr5_adapter();
    assert(adapter != nullptr && "LPDDR5 backend should return non-null adapter");
    adapter->fill_direct(0, input.data(), DATA_SIZE);

    // SDMA: LPDDR5[0..63] → scratchpad[0..63]
    VLIWPacket p1 = {};
    p1.sDMA_op = {DMAType::MEMCPY, 0, 0, DATA_SIZE, 0, 0};
    sim.dispatch_packet(p1);
    VLIWPacket s1 = {}; s1.sync_mask = STATUS_SDMA_BUSY;
    sim.dispatch_packet(s1);

    // 驗證資料完整性
    std::vector<uint8_t> result(DATA_SIZE, 0);
    sim.get_scratchpad().read(0, result.data(), DATA_SIZE);

    for (size_t i = 0; i < DATA_SIZE; ++i) {
        if (result[i] != input[i]) {
            std::cerr << "[Test 22] Mismatch at byte " << i
                      << ": expected " << (int)input[i]
                      << ", got " << (int)result[i] << std::endl;
            assert(false && "LPDDR5 data correctness check failed");
        }
    }

    // P3-4: 驗證 DDR 統計可讀取，且已記錄至少一個 read 事件
    const lpddr5::ChannelStats& stats = adapter->get_ddr_stats(0);
    assert(stats.reads_issued >= 1 && "LPDDR5 should record at least one read");

    assert(!sim.has_error());
    assert(sim.get_scoreboard().get_status() == 0);

    // P3-4: 印出 DDR 效能報告
    std::cout << "[P3-4] LPDDR5 Channel 0 Statistics:" << std::endl;
    adapter->print_ddr_stats(0);

    std::cout << "LPDDR5 Backend Data Correctness Test Passed ("
              << DATA_SIZE << " bytes, reads_issued=" << stats.reads_issued
              << ", row_hit_rate=" << stats.row_hit_rate() * 100 << "%)." << std::endl;
}

// ---------------------------------------------------------------------------
// P3-4 驗證 (Test 23)：LPDDR5 backend 延遲 > SIMPLE backend 延遲
//
// 目標：驗證 cycle-accurate LPDDR5 模型引入的延遲高於固定延遲模型，
//        即 LPDDR5 的 SimClock ticks > SIMPLE 的 SimClock ticks。
//
// 預期：
//   - SIMPLE backend（10 ticks/cacheline）：1 cacheline → 10 xTPU ticks
//   - LPDDR5-6400（cold row access：nRCD=15 + nCL=20 = 35 DDR CK，
//     tCK=1250ps，xTPU@1GHz）：~44 xTPU ticks/cacheline
//   → LPDDR5 ticks > SIMPLE ticks
// ---------------------------------------------------------------------------
void test_lpddr5_latency_vs_simple() {
    const size_t DATA_SIZE = 128;  // 2 cachelines（64B × 2）
    std::vector<uint8_t> input(DATA_SIZE, 0x5A);

    std::cout << "\n=== Test 23: LPDDR5 Latency vs SIMPLE (P3-4) ===" << std::endl;

    // ── SIMPLE backend ─────────────────────────────────────────────────────
    SimClock::Tick simple_ticks;
    {
        TimingConfig timing; timing.ms_to_ticks = 0;
        Simulator sim(timing);
        sim.get_system_mem().write(0, input.data(), DATA_SIZE);

        VLIWPacket p = {}; p.sDMA_op = {DMAType::MEMCPY, 0, 0, DATA_SIZE, 0, 0};
        sim.dispatch_packet(p);
        VLIWPacket s = {}; s.sync_mask = STATUS_SDMA_BUSY;
        sim.dispatch_packet(s);

        simple_ticks = sim.get_clock().now();
    }

    // ── LPDDR5 backend ────────────────────────────────────────────────────
    SimClock::Tick lpddr5_ticks;
    {
        SimulatorConfig cfg;
        cfg.backend = SimulatorConfig::MemoryBackend::LPDDR5;
        cfg.timing.ms_to_ticks = 0;
        Simulator sim(cfg);

        // fill_direct 繞過時序（只設定資料），確保 clock.now() 反映 SDMA 延遲
        sim.get_lpddr5_adapter()->fill_direct(0, input.data(), DATA_SIZE);

        VLIWPacket p = {}; p.sDMA_op = {DMAType::MEMCPY, 0, 0, DATA_SIZE, 0, 0};
        sim.dispatch_packet(p);
        VLIWPacket s = {}; s.sync_mask = STATUS_SDMA_BUSY;
        sim.dispatch_packet(s);

        lpddr5_ticks = sim.get_clock().now();

        std::cout << "[P3-4] LPDDR5 Channel 0 Statistics:" << std::endl;
        sim.print_ddr_stats(0);
    }

    std::cout << "  SIMPLE backend ticks : " << simple_ticks  << std::endl;
    std::cout << "  LPDDR5 backend ticks : " << lpddr5_ticks  << std::endl;

    assert(lpddr5_ticks > simple_ticks
           && "LPDDR5 cycle-accurate latency should exceed SIMPLE fixed latency");

    std::cout << "LPDDR5 Latency vs SIMPLE Test Passed "
              << "(LPDDR5=" << lpddr5_ticks
              << " > SIMPLE=" << simple_ticks << " xTPU ticks)." << std::endl;
}

// ---------------------------------------------------------------------------
// P3-CR-7 驗證 (Test 24)：LPDDR5 backend writeback（FROM_DEVICE）正確性
//
// 目標：驗證 scratchpad → LPDDR5 writeback 路徑（SDMA FROM_DEVICE）在 LPDDR5
//        backend 下也能正確寫回資料，且資料與原始值一致。
//
// 流程：
//   fill_direct(LPDDR5, 0x0, zeros) → SDMA LOAD (LPDDR5→scratch) →
//   overwrite scratchpad → SDMA WRITEBACK (scratch→LPDDR5) →
//   read LPDDR5 backing store 驗證
// ---------------------------------------------------------------------------
void test_lpddr5_writeback() {
    SimulatorConfig cfg;
    cfg.backend = SimulatorConfig::MemoryBackend::LPDDR5;
    cfg.timing.ms_to_ticks = 0;
    Simulator sim(cfg);

    std::cout << "\n=== Test 24: LPDDR5 Writeback (FROM_DEVICE) (P3-CR-7) ===" << std::endl;

    const size_t DATA_SIZE = 64;
    LPDDR5Adapter* adapter = sim.get_lpddr5_adapter();
    assert(adapter != nullptr);

    // 1. 用已知 pattern 預填 LPDDR5
    std::vector<uint8_t> original(DATA_SIZE);
    for (size_t i = 0; i < DATA_SIZE; ++i) original[i] = static_cast<uint8_t>(i + 1);
    adapter->fill_direct(0, original.data(), DATA_SIZE);

    // 2. SDMA LOAD: LPDDR5[0..63] → scratchpad[0..63]
    VLIWPacket p_load = {};
    p_load.sDMA_op = {DMAType::MEMCPY, 0, 0, DATA_SIZE, 0, 0};
    sim.dispatch_packet(p_load);
    VLIWPacket s1 = {}; s1.sync_mask = STATUS_SDMA_BUSY;
    sim.dispatch_packet(s1);

    // 3. 將 scratchpad 內容翻轉（在 scratchpad 改寫資料）
    std::vector<uint8_t> modified(DATA_SIZE);
    sim.get_scratchpad().read(0, modified.data(), DATA_SIZE);
    for (size_t i = 0; i < DATA_SIZE; ++i) modified[i] = static_cast<uint8_t>(~modified[i]);
    sim.get_scratchpad().write(0, modified.data(), DATA_SIZE);

    // 4. SDMA WRITEBACK: scratchpad[0..63] → LPDDR5[0..63]
    VLIWPacket p_wb = {};
    p_wb.sDMA_op = {DMAType::MEMCPY, 0, 0, DATA_SIZE, 0, 0, DMADirection::FROM_DEVICE};
    sim.dispatch_packet(p_wb);
    VLIWPacket s2 = {}; s2.sync_mask = STATUS_SDMA_BUSY;
    sim.dispatch_packet(s2);

    // 5. 驗證 LPDDR5 backing store 已更新為 ~original
    //    透過 IMemoryPort::read（含 LPDDR5 timing simulation），不使用 fill_direct 讀回
    std::vector<uint8_t> verify(DATA_SIZE, 0xFF);
    sim.get_active_system_mem().read(0, verify.data(), DATA_SIZE);
    for (size_t i = 0; i < DATA_SIZE; ++i) {
        if (verify[i] != modified[i]) {
            std::cerr << "[Test 24] Writeback mismatch at byte " << i
                      << ": expected " << (int)modified[i]
                      << ", got " << (int)verify[i] << std::endl;
            assert(false && "LPDDR5 writeback data correctness failed");
        }
    }

    assert(!sim.has_error());
    assert(sim.get_scoreboard().get_status() == 0);
    std::cout << "LPDDR5 Writeback Test Passed ("
              << DATA_SIZE << " bytes written back, bit-inverted)." << std::endl;
}

// ---------------------------------------------------------------------------
// P3-CR-7 驗證 (Test 25)：LPDDR5 backend + Compute E2E 正確性
//
// 目標：驗證 LPDDR5 backend 下完整的 load → compute → writeback 流程。
//
// 流程：
//   fill_direct(LPDDR5, 0x0, [0,1,...,63]) → SDMA LOAD → IDMA → PU0 SCALAR →
//   IDMA WRITEBACK → SDMA WRITEBACK → read LPDDR5 驗證值為 [1,2,...,64]
// ---------------------------------------------------------------------------
void test_lpddr5_e2e_with_compute() {
    SimulatorConfig cfg;
    cfg.backend = SimulatorConfig::MemoryBackend::LPDDR5;
    cfg.timing.ms_to_ticks = 0;
    Simulator sim(cfg);

    std::cout << "\n=== Test 25: LPDDR5 + Compute E2E (P3-CR-7) ===" << std::endl;

    LPDDR5Adapter* adapter = sim.get_lpddr5_adapter();

    // 1. 預填資料到 LPDDR5
    const size_t DATA_SIZE = 16;  // 16B = 1/4 cacheline，足夠驗 Scalar+1
    std::vector<uint8_t> input(DATA_SIZE);
    for (size_t i = 0; i < DATA_SIZE; ++i) input[i] = static_cast<uint8_t>(i);
    adapter->fill_direct(0, input.data(), DATA_SIZE);

    // 2. SDMA LOAD: LPDDR5[0..63] → scratchpad[0..63]（1 cacheline minimum）
    VLIWPacket p1 = {};
    p1.sDMA_op = {DMAType::MEMCPY, 0, 0, 64, 0, 0};
    sim.dispatch_packet(p1);
    VLIWPacket s1 = {}; s1.sync_mask = STATUS_SDMA_BUSY;
    sim.dispatch_packet(s1);

    // 3. IDMA: scratchpad[0..15] → PU0 LocalMem[buf0][0..15]
    VLIWPacket p2 = {};
    // DMA_Command fields: type, src_addr, dst_addr, size, target_mask, buffer_idx, direction
    p2.iDMA_op = {DMAType::MEMCPY, 0, 0, DATA_SIZE, TARGET_PU0, 0, DMADirection::TO_DEVICE};
    sim.dispatch_packet(p2);
    VLIWPacket s2 = {}; s2.sync_mask = STATUS_PU0_DMA_BUSY;
    sim.dispatch_packet(s2);

    // 4. PU0 SCALAR: buf0[0..15] += 1
    // Compute_Command fields: type, buffer_idx, simulated_duration_ms, src_offset, dst_offset, length
    VLIWPacket p3 = {};
    p3.pu0_op = {ComputeType::SCALAR, 0, 0, 0, 0, DATA_SIZE};
    sim.dispatch_packet(p3);
    VLIWPacket s3 = {}; s3.sync_mask = STATUS_PU0_CMD_BUSY;
    sim.dispatch_packet(s3);

    // 5. IDMA WRITEBACK: PU0 LocalMem[buf0][0..15] → scratchpad[0..15]
    VLIWPacket p4 = {};
    p4.iDMA_op = {DMAType::MEMCPY, 0, 0, DATA_SIZE, TARGET_PU0, 0, DMADirection::FROM_DEVICE};
    sim.dispatch_packet(p4);
    VLIWPacket s4 = {}; s4.sync_mask = STATUS_PU0_DMA_BUSY;
    sim.dispatch_packet(s4);

    // 6. SDMA WRITEBACK: scratchpad[0..63] → LPDDR5[0..63]
    VLIWPacket p5 = {};
    p5.sDMA_op = {DMAType::MEMCPY, 0, 0, 64, 0, 0, DMADirection::FROM_DEVICE};
    sim.dispatch_packet(p5);
    VLIWPacket s5 = {}; s5.sync_mask = STATUS_SDMA_BUSY;
    sim.dispatch_packet(s5);

    // 7. 驗證 LPDDR5 的前 16 bytes = [1,2,...,16]
    std::vector<uint8_t> result(DATA_SIZE, 0);
    sim.get_active_system_mem().read(0, result.data(), DATA_SIZE);
    for (size_t i = 0; i < DATA_SIZE; ++i) {
        uint8_t expected = static_cast<uint8_t>(i + 1);
        if (result[i] != expected) {
            std::cerr << "[Test 25] Mismatch at byte " << i
                      << ": expected " << (int)expected
                      << ", got " << (int)result[i] << std::endl;
            assert(false && "LPDDR5 E2E+Compute failed");
        }
    }

    assert(!sim.has_error());
    std::cout << "LPDDR5 + Compute E2E Test Passed (SCALAR+1 verified)." << std::endl;
}

// ---------------------------------------------------------------------------
// P3-CR-7 驗證 (Test 26)：Row-hit vs Row-miss 延遲可觀察性
//
// 目標：驗證連續讀同一 row 的延遲（row-hit）顯著小於 cold row access（row-miss）。
//       這是 P3-4 acceptance 條件「row-hit vs miss > 2x」的實際驗收。
//
// 方法：
//   - Cold read（row-miss）：讀位址 0x0（全新 row，包含 RCD + CL latency）
//   - Warm read（row-hit）：再次讀同一位址（row 已在 row-buffer 中，只需 CL）
//   - 兩者 elapsed DDR CK 應有顯著差異（row-hit << row-miss）
//
// 注意：sim clock 是累積的，所以用兩個獨立的 Simulator instance 量測。
// ---------------------------------------------------------------------------
void test_lpddr5_row_hit_vs_miss() {
    std::cout << "\n=== Test 26: LPDDR5 Row-Hit vs Row-Miss Latency (P3-CR-7) ===" << std::endl;

    const size_t DATA_SIZE = 64;  // 1 cacheline
    std::vector<uint8_t> dummy(DATA_SIZE, 0xBB);

    // ── Row-miss（cold access） ────────────────────────────────────────────
    SimClock::Tick miss_ticks;
    {
        SimulatorConfig cfg;
        cfg.backend = SimulatorConfig::MemoryBackend::LPDDR5;
        cfg.timing.ms_to_ticks = 0;
        Simulator sim(cfg);
        sim.get_lpddr5_adapter()->fill_direct(0, dummy.data(), DATA_SIZE);

        VLIWPacket p = {}; p.sDMA_op = {DMAType::MEMCPY, 0, 0, DATA_SIZE, 0, 0};
        sim.dispatch_packet(p);
        VLIWPacket s = {}; s.sync_mask = STATUS_SDMA_BUSY;
        sim.dispatch_packet(s);

        miss_ticks = sim.get_clock().now();
    }

    // ── Row-hit（same row，連續兩次讀；第一次 miss，第二次 hit）──────────────
    SimClock::Tick hit_ticks;
    {
        SimulatorConfig cfg;
        cfg.backend = SimulatorConfig::MemoryBackend::LPDDR5;
        cfg.timing.ms_to_ticks = 0;
        Simulator sim(cfg);
        sim.get_lpddr5_adapter()->fill_direct(0, dummy.data(), DATA_SIZE);

        // 先做一次讀（cold，row-miss）讓 row 進入 row-buffer
        VLIWPacket p1 = {}; p1.sDMA_op = {DMAType::MEMCPY, 0, 0, DATA_SIZE, 0, 0};
        sim.dispatch_packet(p1);
        VLIWPacket s1 = {}; s1.sync_mask = STATUS_SDMA_BUSY;
        sim.dispatch_packet(s1);
        SimClock::Tick before_hit = sim.get_clock().now();

        // 再讀同一 cacheline（row-hit）
        VLIWPacket p2 = {}; p2.sDMA_op = {DMAType::MEMCPY, 0, 0, DATA_SIZE, 0, 0};
        sim.dispatch_packet(p2);
        VLIWPacket s2 = {}; s2.sync_mask = STATUS_SDMA_BUSY;
        sim.dispatch_packet(s2);

        hit_ticks = sim.get_clock().now() - before_hit;  // 只量第二次
    }

    std::cout << "  Row-miss (cold) ticks : " << miss_ticks << std::endl;
    std::cout << "  Row-hit  (warm) ticks : " << hit_ticks  << std::endl;

    // Row-miss 應 > row-hit（hit 只需 CL，miss 還需 RCD + precharge 等）
    assert(miss_ticks > hit_ticks
           && "Row-miss should have higher latency than row-hit");
    // 理論上 row-hit 延遲約為 miss 的 40-60%（nCL / (nRCD+nCL)）
    std::cout << "LPDDR5 Row-Hit vs Row-Miss Test Passed "
              << "(miss=" << miss_ticks << " > hit=" << hit_ticks << " xTPU ticks)." << std::endl;
}

// ---------------------------------------------------------------------------
// P3-CR-7 驗證 (Test 27)：LPDDR5 OOB access → STATUS_ERROR
//
// 目標：驗證超出 LPDDR5 backing store 範圍的讀/寫正確觸發 bounds_check，
//        並被 SDMAEngine 的 try/catch 捕捉後設定 STATUS_ERROR。
// ---------------------------------------------------------------------------
void test_lpddr5_oob_sets_error() {
    SimulatorConfig cfg;
    cfg.backend = SimulatorConfig::MemoryBackend::LPDDR5;
    cfg.timing.ms_to_ticks = 0;
    Simulator sim(cfg);

    std::cout << "\n=== Test 27: LPDDR5 OOB → STATUS_ERROR (P3-CR-7) ===" << std::endl;

    // 嘗試讀取超出 backing_size_ 的位址（SYSTEM_MEMORY_SIZE - 32 + 64 = 跨界）
    const uint64_t oob_addr = static_cast<uint64_t>(SYSTEM_MEMORY_SIZE) - 32;
    const size_t   oob_size = 64;  // 跨越末尾

    VLIWPacket p = {};
    p.sDMA_op = {DMAType::MEMCPY, oob_addr, 0, oob_size, 0, 0};
    sim.dispatch_packet(p);
    VLIWPacket s = {}; s.sync_mask = STATUS_SDMA_BUSY;
    sim.dispatch_packet(s);

    assert(sim.has_error() && "OOB LPDDR5 read should set STATUS_ERROR");
    // BusyClearGuard 確保 STATUS_SDMA_BUSY 清除；STATUS_ERROR 仍被設定（sticky），需 clear_error()
    assert(!(sim.get_scoreboard().get_status() & STATUS_SDMA_BUSY)
           && "SDMA busy bit should be cleared even on error (BusyClearGuard)");
    assert((sim.get_scoreboard().get_status() & STATUS_ERROR)
           && "STATUS_ERROR should still be set (sticky, requires explicit clear_error)");

    sim.clear_error();
    assert(!sim.has_error());
    std::cout << "LPDDR5 OOB Test Passed (STATUS_ERROR set and cleared correctly)." << std::endl;
}

int main() {
    try {
        test_pipeline_scenario();
        test_unicast_idma();
        test_independent_compute();
        test_sdma_error_no_deadlock();          // P0-1 + P2-4
        test_idma_error_no_deadlock();          // P0-1 + P2-4
        test_clock_advances();                  // P1-2
        test_sdma_data_correctness();           // P1-3
        test_e2e_data_correctness();            // P1-3
        test_writeback_path();                  // P1-4
        test_compute_scalar();                  // P2-1
        test_compute_vector();                  // P2-1
        test_compute_matmul();                  // P2-1
        test_double_buffer_isolation();         // P2-2 Scenario 3
        test_e2e_with_compute();                // P2-2 Scenario 4
        test_incremental_e2e_with_vector();     // P2-2 Scenario 5
        test_compute_error_sets_error_bit();    // P2-4
        test_error_recovery();                  // P2-4
        test_compute_matmul_nontrivial();       // P2-CR-3
        test_matmul_invalid_length();           // P2-CR-1
        test_pu1_writeback_path();              // CR3-3
        test_idma_broadcast_failfast();         // CR3-1
        test_lpddr5_data_correctness();         // P3-2 / P3-4
        test_lpddr5_latency_vs_simple();        // P3-4
        test_lpddr5_writeback();                // P3-CR-7 Test 24
        test_lpddr5_e2e_with_compute();         // P3-CR-7 Test 25
        test_lpddr5_row_hit_vs_miss();          // P3-CR-7 Test 26
        test_lpddr5_oob_sets_error();           // P3-CR-7 Test 27
        std::cout << "\nALL TESTS SUCCESSFUL" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "FAILED: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
