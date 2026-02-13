#pragma once
#include "engine_base.hpp"

class MXUEngine : public Engine {
public:
    MXUEngine();

    void process(const Compute_Command& cmd);
};
