#include "engines/idma_engine.hpp"

IDMAEngine::IDMAEngine(StatusRegister& sr, Scratchpad& sp, LocalMemory& lm)
    : Engine(sr), scratchpad(sp), local_memory(lm), current_busy_mask(0) {}

void IDMAEngine::process(const DMA_Command& cmd) {
    if (cmd.type == DMAType::NOP) return;

    remaining_cycles = cmd.duration_cycles;

    // Determine which bit to clear upon completion
    // The busy bit was SET by the Dispatcher (Simulator), but we need to know which one to CLEAR.
    // Based on bank_id:
    switch (cmd.bank_id) {
        case 0: current_busy_mask = STATUS_IDMA_B0_BUSY; break;
        case 1: current_busy_mask = STATUS_IDMA_B1_BUSY; break;
        case 2: current_busy_mask = STATUS_IDMA_B2_BUSY; break;
        case 3: current_busy_mask = STATUS_IDMA_B3_BUSY; break;
        default: current_busy_mask = 0; break;
    }
}

void IDMAEngine::on_complete() {
    if (current_busy_mask != 0) {
        status_reg.clear_busy(current_busy_mask);
        current_busy_mask = 0;
    }
}
