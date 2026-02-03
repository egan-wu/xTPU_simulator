#pragma once
#include "engine_base.hpp"
#include "memory/scratchpad.hpp"

class SDMAEngine : public Engine {
public:
    SDMAEngine(StatusRegister& sr, Scratchpad& sp);

    void process(const DMA_Command& cmd);

protected:
    void on_complete() override;

private:
    Scratchpad& scratchpad;
};
