#pragma once
#include "engine_base.hpp"

class VectorEngine : public Engine {
public:
    VectorEngine();

    void process(const Compute_Command& cmd);
};
