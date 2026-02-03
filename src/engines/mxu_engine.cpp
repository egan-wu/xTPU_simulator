#include "engines/mxu_engine.hpp"

MXUEngine::MXUEngine(StatusRegister& sr) : Engine(sr) {}

void MXUEngine::process(const Compute_Command& cmd) {
    if (cmd.type == ComputeType::NOP) return;
    remaining_cycles = cmd.duration_cycles;
}

void MXUEngine::on_complete() {
    status_reg.clear_busy(STATUS_MXU_BUSY);
}
