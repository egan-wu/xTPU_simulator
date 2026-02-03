#pragma once
#include "engine_base.hpp"
#include "memory/scratchpad.hpp"
#include "memory/local_memory.hpp"

class IDMAEngine : public Engine {
public:
    IDMAEngine(StatusRegister& sr, Scratchpad& sp, LocalMemory& lm);

    void process(const DMA_Command& cmd);

protected:
    void on_complete() override;

private:
    Scratchpad& scratchpad;
    LocalMemory& local_memory;

    // We need to know WHICH busy bit to clear when done
    // Since iDMA can target different banks, we store the current mask
    uint32_t current_busy_mask;
};
