#include "control_unit.hpp"
#include <iomanip>

#include "control_unit.hpp"
#include <iomanip>

ControlUnit::ControlUnit()
    : scratchpad(1024*1024),
      sdma(scratchpad),
      idma(scratchpad, pu0.get_local_memory(), pu1.get_local_memory()),
      pu0(0),
      pu1(1),
      pc(0),
      cycle_count(0) {}

void ControlUnit::load_program(const std::vector<VLIWPacket>& program) {
    instruction_memory = program;
    pc = 0;
    cycle_count = 0;
}

void ControlUnit::run(int max_cycles) {
    std::cout << "Starting Simulation..." << std::endl;
    std::cout << "Cycle | PC | Status | Action" << std::endl;
    std::cout << "------+----+--------+-------" << std::endl;

    while (((size_t)pc < instruction_memory.size() || is_busy()) && cycle_count < max_cycles) {
        step();
    }

    if ((size_t)pc >= instruction_memory.size() && !is_busy()) {
        std::cout << "Simulation Finished: All instructions executed." << std::endl;
    } else {
         std::cout << "Simulation Stopped: Max cycles reached." << std::endl;
    }
}

void ControlUnit::step() {
    // 1. Tick all engines and update Status Register on completion
    if (sdma.tick()) {
        status_reg.clear_busy(STATUS_SDMA_BUSY);
    }

    if (idma.tick()) {
        status_reg.clear_busy(idma.get_completed_mask());
    }

    auto pu0_status = pu0.tick();
    if (pu0_status.mxu_done) status_reg.clear_busy(STATUS_PU0_MXU_BUSY);
    if (pu0_status.vector_done) status_reg.clear_busy(STATUS_PU0_VEC_BUSY);
    if (pu0_status.scalar_done) status_reg.clear_busy(STATUS_PU0_SCA_BUSY);

    auto pu1_status = pu1.tick();
    if (pu1_status.mxu_done) status_reg.clear_busy(STATUS_PU1_MXU_BUSY);
    if (pu1_status.vector_done) status_reg.clear_busy(STATUS_PU1_VEC_BUSY);
    if (pu1_status.scalar_done) status_reg.clear_busy(STATUS_PU1_SCA_BUSY);

    // 2. Fetch & Decode
    if ((size_t)pc >= instruction_memory.size()) {
        cycle_count++;
        log_state("IDLE");
        return;
    }

    const VLIWPacket& packet = instruction_memory[pc];

    // 3. Sync Check
    bool sync_stall = status_reg.check_stall(packet.sync_mask);

    // 3.1 Structural Hazard Check
    bool struct_stall = check_structural_hazard(packet);

    if (sync_stall || struct_stall) {
        log_state(sync_stall ? "STALL (SYNC)" : "STALL (STRUCT)");
    } else {
        // 4. Dispatch
        dispatch_packet(packet);
        log_state("DISPATCH");
        pc++;
    }

    cycle_count++;
}

bool ControlUnit::check_structural_hazard(const VLIWPacket& packet) const {
    if (packet.sDMA_op.type != DMAType::NOP && sdma.is_busy()) return true;
    if (packet.iDMA_op.type != DMAType::NOP && idma.is_busy()) return true;
    if (packet.pu0_op.type != ComputeType::NOP && pu0.is_busy()) return true;
    if (packet.pu1_op.type != ComputeType::NOP && pu1.is_busy()) return true;
    return false;
}

void ControlUnit::dispatch_packet(const VLIWPacket& packet) {
    // Dispatch sDMA
    if (packet.sDMA_op.type != DMAType::NOP) {
        status_reg.set_busy(STATUS_SDMA_BUSY);
        sdma.process(packet.sDMA_op);
    }

    // Dispatch iDMA
    if (packet.iDMA_op.type != DMAType::NOP) {
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

    // Dispatch PU0
    if (packet.pu0_op.type != ComputeType::NOP) {
        if (packet.pu0_op.type == ComputeType::MATMUL) {
            status_reg.set_busy(STATUS_PU0_MXU_BUSY);
            pu0.dispatch_mxu(packet.pu0_op);
        } else if (packet.pu0_op.type == ComputeType::VECTOR_ADD || packet.pu0_op.type == ComputeType::VECTOR_MASK) {
            status_reg.set_busy(STATUS_PU0_VEC_BUSY);
            pu0.dispatch_vector(packet.pu0_op);
        } else {
            // Scalar assumption for others or default
             status_reg.set_busy(STATUS_PU0_SCA_BUSY);
             pu0.dispatch_scalar(packet.pu0_op);
        }
    }

    // Dispatch PU1
    if (packet.pu1_op.type != ComputeType::NOP) {
        if (packet.pu1_op.type == ComputeType::MATMUL) {
             status_reg.set_busy(STATUS_PU1_MXU_BUSY);
             pu1.dispatch_mxu(packet.pu1_op);
        } else if (packet.pu1_op.type == ComputeType::VECTOR_ADD || packet.pu1_op.type == ComputeType::VECTOR_MASK) {
             status_reg.set_busy(STATUS_PU1_VEC_BUSY);
             pu1.dispatch_vector(packet.pu1_op);
        } else {
             status_reg.set_busy(STATUS_PU1_SCA_BUSY);
             pu1.dispatch_scalar(packet.pu1_op);
        }
    }
}

void ControlUnit::log_state(const std::string& action) {
    std::cout << std::setw(5) << cycle_count << " | "
              << std::setw(2) << pc << " | "
              << "0x" << std::hex << std::setw(3) << std::setfill('0') << status_reg.get_status() << std::dec << std::setfill(' ') << " | "
              << action << std::endl;
}
