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

int main() {
    try {
        test_pipeline_scenario();
        test_unicast_idma();
        test_independent_compute();
        std::cout << "\nALL TESTS SUCCESSFUL" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "FAILED: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
