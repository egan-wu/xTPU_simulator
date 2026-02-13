#include <iostream>
#include <vector>
#include "control_unit.hpp"
#include "common.hpp"

int main() {
    std::cout << "=== VLIW TPU Cycle-Accurate Simulator ===" << std::endl;

    // 1. Define Test Program based on requirements
    std::vector<VLIWPacket> program;

    // Instruction 0: iDMA move to Bank 0 (Duration = 5)
    // No Sync needed.
    VLIWPacket p0;
    p0.iDMA_op.type = DMAType::MEMCPY;
    p0.iDMA_op.bank_id = 0; // Target Bank 0
    p0.iDMA_op.duration_cycles = 5;
    p0.sync_mask = 0; // Run immediately
    program.push_back(p0);

    // Instruction 1: MXU Execute on PU0 (Duration = 10)
    // Must SYNC on Bank 0 (Wait for iDMA to finish)
    VLIWPacket p1;
    p1.pu0_op.type = ComputeType::MATMUL;
    p1.pu0_op.duration_cycles = 10;
    p1.sync_mask = STATUS_IDMA_B0_BUSY; // Wait for Bank 0
    program.push_back(p1);

    // Instruction 2: Vector Execute on PU0 (Duration = 3)
    // Depend on MXU
    VLIWPacket p2;
    p2.pu0_op.type = ComputeType::VECTOR_ADD;
    p2.pu0_op.duration_cycles = 3;
    p2.sync_mask = STATUS_PU0_MXU_BUSY;
    program.push_back(p2);

    // 2. Initialize Simulator
    ControlUnit cu;
    cu.load_program(program);

    // 3. Run for enough cycles to see the behavior
    // iDMA (5) + MXU (10) + Vector (3) + overheads approx 20 cycles
    cu.run(30);

    return 0;
}
