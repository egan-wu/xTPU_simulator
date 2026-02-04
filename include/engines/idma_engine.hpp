#pragma once
#include "engine_base.hpp"
#include "memory/scratchpad.hpp"
#include "memory/local_memory.hpp"

class IDMAEngine : public Engine {
public:
    IDMAEngine(Scratchpad& sp, LocalMemory& lm0, LocalMemory& lm1);

    void process(const DMA_Command& cmd);

    // Helper to get the cleared mask *after* tick() returns true
    uint32_t get_completed_mask() const { return current_busy_mask; }

private:
    Scratchpad& scratchpad;
    LocalMemory& local_memory0;
    LocalMemory& local_memory1;

    uint32_t current_busy_mask;
};
