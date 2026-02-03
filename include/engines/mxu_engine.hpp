#pragma once
#include "engine_base.hpp"

class MXUEngine : public Engine {
public:
    MXUEngine(StatusRegister& sr);

    void process(const Compute_Command& cmd);

protected:
    void on_complete() override;
};
