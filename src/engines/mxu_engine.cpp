#include "engines/mxu_engine.hpp"

MXUEngine::MXUEngine() : Engine() {}

void MXUEngine::process(const Compute_Command& cmd) {
    if (cmd.type == ComputeType::NOP) return;
    remaining_cycles = cmd.duration_cycles;
}
