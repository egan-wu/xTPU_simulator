#include "engines.hpp"
#include <chrono>
#include <vector>

// SDMA Implementation
SDMAEngine::SDMAEngine(StatusRegister& sr, Scratchpad& sp)
    : Engine(sr), scratchpad(sp) {}

void SDMAEngine::process(const DMA_Command& cmd) {
    if (cmd.type == DMAType::NOP) return; // Should not happen if filtered, but safe

    // Simulate transfer time
    // 128GB/s is fast, but for simulation let's simulate 1ms per 1KB or something visible
    // std::this_thread::sleep_for(std::chrono::milliseconds(1));

    // Perform transfer
    // src is system memory (fake), dst is scratchpad
    // We'll just create a dummy buffer of 'size' and write it
    std::vector<uint8_t> dummy_data(cmd.size, 0xAA); // 0xAA pattern

    try {
        scratchpad.write(cmd.dst_addr, dummy_data.data(), cmd.size);
    } catch (const std::exception& e) {
        std::cerr << "SDMA Error: " << e.what() << std::endl;
    }

    // Done signal
    status_reg.clear_busy(STATUS_SDMA_BUSY);
}

// IDMA Implementation
IDMAEngine::IDMAEngine(StatusRegister& sr, Scratchpad& sp, LocalMemory& lm0, LocalMemory& lm1)
    : Engine(sr), scratchpad(sp), local_mem0(lm0), local_mem1(lm1) {}

void IDMAEngine::process(const DMA_Command& cmd) {
    if (cmd.type == DMAType::NOP) return;

    // Read from Scratchpad
    std::vector<uint8_t> buffer(cmd.size);
    try {
        scratchpad.read(cmd.src_addr, buffer.data(), cmd.size);
    } catch (const std::exception& e) {
        std::cerr << "IDMA Read Error: " << e.what() << std::endl;
        // Even on error, we should probably clear busy to avoid hang?
        // Or not, to signal failure? For this sim, let's clear.
    }

    // Write to targets
    // target_mask determines destinations
    bool target_pu0 = (cmd.target_mask & TARGET_PU0);
    bool target_pu1 = (cmd.target_mask & TARGET_PU1);

    if (target_pu0) {
        try {
            local_mem0.write(cmd.buffer_idx, cmd.dst_addr, buffer.data(), cmd.size);
        } catch (const std::exception& e) {
            std::cerr << "IDMA Write PU0 Error: " << e.what() << std::endl;
        }
    }

    if (target_pu1) {
        try {
            local_mem1.write(cmd.buffer_idx, cmd.dst_addr, buffer.data(), cmd.size);
        } catch (const std::exception& e) {
            std::cerr << "IDMA Write PU1 Error: " << e.what() << std::endl;
        }
    }

    // Simulate latency
    // std::this_thread::sleep_for(std::chrono::milliseconds(1));

    // Clear Status Bits
    uint32_t clear_mask = 0;
    if (target_pu0) clear_mask |= STATUS_PU0_DMA_BUSY;
    if (target_pu1) clear_mask |= STATUS_PU1_DMA_BUSY;

    status_reg.clear_busy(clear_mask);
}

// Compute Implementation
ComputeEngine::ComputeEngine(StatusRegister& sr, LocalMemory& lm, uint32_t busy_bit)
    : Engine(sr), local_mem(lm), my_busy_bit(busy_bit) {}

void ComputeEngine::process(const Compute_Command& cmd) {
    if (cmd.type == ComputeType::NOP) return;

    // Simulate Compute Work
    if (cmd.simulated_duration_ms > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(cmd.simulated_duration_ms));
    }

    // We might "read" or "write" local memory here to simulate usage
    // Accessing buffer_idx to assert we can access it
    // Implementation not strictly required by prompt ("Focus is on software-hardware contract")
    // but good for completeness.

    status_reg.clear_busy(my_busy_bit);
}
