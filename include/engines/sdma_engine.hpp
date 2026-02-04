#pragma once
#include "engine_base.hpp"
#include "memory/scratchpad.hpp"

class SDMAEngine : public Engine {
public:
    SDMAEngine(Scratchpad& sp);

    void process(const DMA_Command& cmd);

private:
    Scratchpad& scratchpad;
};
