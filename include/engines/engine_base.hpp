#pragma once
#include "common.hpp"
#include "status_register.hpp"

class Engine {
public:
    Engine(StatusRegister& sr) : status_reg(sr), remaining_cycles(0) {}
    virtual ~Engine() = default;

    virtual void tick() {
        if (remaining_cycles > 0) {
            remaining_cycles--;
            if (remaining_cycles == 0) {
                on_complete();
            }
        }
    }

    bool is_busy() const {
        return remaining_cycles > 0;
    }

protected:
    StatusRegister& status_reg;
    uint32_t remaining_cycles;

    virtual void on_complete() = 0;
};
