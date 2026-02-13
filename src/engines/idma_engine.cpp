#include "engines/idma_engine.hpp"

IDMAEngine::IDMAEngine(Scratchpad& sp, LocalMemory& lm0, LocalMemory& lm1)
    : Engine(), scratchpad(sp), local_memory0(lm0), local_memory1(lm1), current_busy_mask(0) {}

void IDMAEngine::process(const DMA_Command& cmd) {
    if (cmd.type == DMAType::NOP) return;

    remaining_cycles = cmd.duration_cycles;

    // Determine which bit to clear upon completion
    switch (cmd.bank_id) {
        case 0: current_busy_mask = STATUS_IDMA_B0_BUSY; break;
        case 1: current_busy_mask = STATUS_IDMA_B1_BUSY; break;
        case 2: current_busy_mask = STATUS_IDMA_B2_BUSY; break;
        case 3: current_busy_mask = STATUS_IDMA_B3_BUSY; break;
        default: current_busy_mask = 0; break;
    }
    
    // In a real simulator, we would perform the data transfer here
    // For now, we just route to correct memory bank for validity checks
    if (cmd.bank_id == 0 || cmd.bank_id == 1) {
        // Target PU0
        // local_memory0.write(..., cmd.bank_id, ...);
    } else if (cmd.bank_id == 2 || cmd.bank_id == 3) {
        // Target PU1
        // local_memory1.write(..., cmd.bank_id - 2, ...);
    }
}
