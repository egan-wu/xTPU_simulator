#pragma once
#include "common.hpp"
#include "status_register.hpp"

class Engine {
public:
    Engine() : remaining_cycles(0) {}
    virtual ~Engine() = default;

    // Returns true if the operation JUST completed in this tick
    virtual bool tick() {
        if (remaining_cycles > 0) {
            remaining_cycles--;
            if (remaining_cycles == 0) {
                return true; // Just finished
            }
        }
        return false;
    }

    bool is_busy() const {
        return remaining_cycles > 0;
    }

protected:
    uint32_t remaining_cycles;
};
