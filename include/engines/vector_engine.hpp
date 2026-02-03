#pragma once
#include "engine_base.hpp"

class VectorEngine : public Engine {
public:
    VectorEngine(StatusRegister& sr);

    void process(const Compute_Command& cmd);

protected:
    void on_complete() override;
};
