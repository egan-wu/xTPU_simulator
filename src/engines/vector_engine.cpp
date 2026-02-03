#include "engines/vector_engine.hpp"

VectorEngine::VectorEngine(StatusRegister& sr) : Engine(sr) {}

void VectorEngine::process(const Compute_Command& cmd) {
    if (cmd.type == ComputeType::NOP) return;
    remaining_cycles = cmd.duration_cycles;
}

void VectorEngine::on_complete() {
    status_reg.clear_busy(STATUS_VEC_BUSY);
}
