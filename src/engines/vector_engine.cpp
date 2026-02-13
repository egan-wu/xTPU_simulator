#include "engines/vector_engine.hpp"

VectorEngine::VectorEngine() : Engine() {}

void VectorEngine::process(const Compute_Command& cmd) {
    if (cmd.type == ComputeType::NOP) return;
    remaining_cycles = cmd.duration_cycles;
}
