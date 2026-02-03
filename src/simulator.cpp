#include "simulator.hpp"
#include <iomanip>

Simulator::Simulator()
    : scratchpad(1024*1024),
      local_memory(65536),
      sdma(status_reg, scratchpad),
      idma(status_reg, scratchpad, local_memory),
      mxu(status_reg),
      vector_unit(status_reg),
      pc(0),
      cycle_count(0) {}

void Simulator::load_program(const std::vector<VLIWPacket>& program) {
    instruction_memory = program;
    pc = 0;
    cycle_count = 0;
}

void Simulator::run(int max_cycles) {
    std::cout << "Starting Simulation..." << std::endl;
    std::cout << "Cycle | PC | Status | Action" << std::endl;
    std::cout << "------+----+--------+-------" << std::endl;

    // Continue running until program ends AND all engines are idle
    while (((size_t)pc < instruction_memory.size() || is_busy()) && cycle_count < max_cycles) {
        step();
    }

    if ((size_t)pc >= instruction_memory.size() && !is_busy()) {
        std::cout << "Simulation Finished: All instructions executed." << std::endl;
    } else {
         std::cout << "Simulation Stopped: Max cycles reached." << std::endl;
    }
}

void Simulator::step() {
    // 1. Tick all engines (Simulate hardware progress)
    // IMPORTANT: Engines update their status (clear busy bits) here if done.
    sdma.tick();
    idma.tick();
    mxu.tick();
    vector_unit.tick();

    // 2. Fetch & Decode
    if ((size_t)pc >= instruction_memory.size()) {
        cycle_count++;
        log_state("IDLE");
        return;
    }

    const VLIWPacket& packet = instruction_memory[pc];

    // 3. Sync Check (The core "Software-Hardware Contract")
    // If the SyncMask requires bits that are currently BUSY, we STALL.
    bool sync_stall = status_reg.check_stall(packet.sync_mask);

    // 3.1 Structural Hazard Check
    // If the target engine is already busy, we STALL.
    bool struct_stall = check_structural_hazard(packet);

    if (sync_stall || struct_stall) {
        log_state(sync_stall ? "STALL (SYNC)" : "STALL (STRUCT)");
        // PC does NOT increment.
    } else {
        // 4. Dispatch
        // If no stall, we dispatch commands to engines and set their Busy bits.
        dispatch_packet(packet);
        log_state("DISPATCH");
        pc++; // Move to next instruction
    }

    cycle_count++;
}

bool Simulator::check_structural_hazard(const VLIWPacket& packet) const {
    if (packet.sDMA_op.type != DMAType::NOP && sdma.is_busy()) return true;
    if (packet.iDMA_op.type != DMAType::NOP && idma.is_busy()) return true;
    if (packet.mxu_op.type != ComputeType::NOP && mxu.is_busy()) return true;
    if (packet.vector_op.type != ComputeType::NOP && vector_unit.is_busy()) return true;
    return false;
}

void Simulator::dispatch_packet(const VLIWPacket& packet) {
    // Dispatch sDMA
    if (packet.sDMA_op.type != DMAType::NOP) {
        status_reg.set_busy(STATUS_SDMA_BUSY);
        sdma.process(packet.sDMA_op);
    }

    // Dispatch iDMA
    if (packet.iDMA_op.type != DMAType::NOP) {
        // Calculate which bit to set based on Bank ID
        uint32_t mask = 0;
        switch(packet.iDMA_op.bank_id) {
            case 0: mask = STATUS_IDMA_B0_BUSY; break;
            case 1: mask = STATUS_IDMA_B1_BUSY; break;
            case 2: mask = STATUS_IDMA_B2_BUSY; break;
            case 3: mask = STATUS_IDMA_B3_BUSY; break;
        }
        if (mask != 0) {
            status_reg.set_busy(mask);
            idma.process(packet.iDMA_op);
        }
    }

    // Dispatch MXU
    if (packet.mxu_op.type != ComputeType::NOP) {
        status_reg.set_busy(STATUS_MXU_BUSY);
        mxu.process(packet.mxu_op);
    }

    // Dispatch Vector
    if (packet.vector_op.type != ComputeType::NOP) {
        status_reg.set_busy(STATUS_VEC_BUSY);
        vector_unit.process(packet.vector_op);
    }
}

void Simulator::log_state(const std::string& action) {
    std::cout << std::setw(5) << cycle_count << " | "
              << std::setw(2) << pc << " | "
              << "0x" << std::hex << std::setw(2) << std::setfill('0') << status_reg.get_status() << std::dec << std::setfill(' ') << " | "
              << action << std::endl;
}
