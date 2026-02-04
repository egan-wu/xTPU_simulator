#include "engines/sdma_engine.hpp"

SDMAEngine::SDMAEngine(Scratchpad& sp)
    : Engine(), scratchpad(sp) {}

void SDMAEngine::process(const DMA_Command& cmd) {
    if (cmd.type == DMAType::NOP) return;

    remaining_cycles = cmd.duration_cycles;

    // Simulate data movement (stub)
    // scratchpad.write(cmd.dst_addr, dummy_data, cmd.size);
    // In a cycle-accurate model, actual data movement might happen at the end or incrementally.
    // We stick to control logic behavior here.
}
