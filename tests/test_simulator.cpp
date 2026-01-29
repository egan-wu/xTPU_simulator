#include "../src/simulator.hpp"
#include <iostream>
#include <cassert>
#include <vector>

void test_sequence() {
    Simulator sim;

    std::cout << "Starting Test Sequence..." << std::endl;

    // Packet 1: sDMA Load
    // Load 1024 bytes to Scratchpad Offset 0
    VLIWPacket p1 = {};
    p1.sDMA_op = {DMAType::MEMCPY, 0, 0, 1024, 0, 0};

    std::cout << "\n[Step 1] Dispatching sDMA Load..." << std::endl;
    sim.dispatch_packet(p1);

    // Packet 2: Sync wait for sDMA, then Broadcast
    // We can combine sync and dispatch in one packet
    // Wait for sDMA to finish, then iDMA broadcast from Scratchpad(0) to LocalMem(Buf0)
    VLIWPacket p2 = {};
    p2.sync_mask = STATUS_SDMA_BUSY;
    p2.iDMA_op = {DMAType::MEMCPY, 0, 0, 1024, TARGET_PU0 | TARGET_PU1, 0};

    std::cout << "\n[Step 2] Dispatching Wait(sDMA) + iDMA Broadcast..." << std::endl;
    sim.dispatch_packet(p2);

    // Packet 3: Wait for iDMA, then Parallel Compute + sDMA Load next batch
    // This demonstrates Double Buffering:
    // - Compute on Buf0 (just loaded)
    // - sDMA Load to Scratchpad Offset 1024 (for next batch)
    VLIWPacket p3 = {};
    p3.sync_mask = STATUS_PU0_DMA_BUSY | STATUS_PU1_DMA_BUSY;
    p3.pu0_op = {ComputeType::MATMUL, 0, 200}; // Simulate 200ms work
    p3.pu1_op = {ComputeType::MATMUL, 0, 200};
    p3.sDMA_op = {DMAType::MEMCPY, 0, 1024, 1024, 0, 0};

    std::cout << "\n[Step 3] Dispatching Wait(iDMA) + Parallel Compute (Buf0) + sDMA Load (Next)..." << std::endl;
    sim.dispatch_packet(p3);

    // Packet 4: Wait for All
    // Barrier to ensure everything finishes
    VLIWPacket p4 = {};
    p4.sync_mask = STATUS_PU0_CMD_BUSY | STATUS_PU1_CMD_BUSY | STATUS_SDMA_BUSY;

    std::cout << "\n[Step 4] Dispatching Wait(All)..." << std::endl;
    sim.dispatch_packet(p4);

    std::cout << "\nTest Sequence Complete." << std::endl;

    // Verify State (Scoreboard should be 0)
    uint32_t final_status = sim.get_scoreboard().get_status();
    std::cout << "Final Status: " << final_status << std::endl;
    assert(final_status == 0);
}

int main() {
    try {
        test_sequence();
        std::cout << "SUCCESS" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "FAILED: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
