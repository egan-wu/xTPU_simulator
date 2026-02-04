#pragma once
#include <vector>
#include <iostream>
#include "common.hpp"
#include "engines/engine_base.hpp" // Ensure base is visible
#include "status_register.hpp"
#include "memory/scratchpad.hpp"
#include "engines/sdma_engine.hpp"
#include "engines/idma_engine.hpp"
#include "engines/processing_unit.hpp"

class ControlUnit {
public:
    ControlUnit();

    void load_program(const std::vector<VLIWPacket>& program);
    void run(int max_cycles);

    // Single step of the cycle-accurate simulation
    void step();

    bool is_busy() const {
        return sdma.is_busy() || idma.is_busy() || pu0.is_busy() || pu1.is_busy();
    }

private:
    StatusRegister status_reg;
    Scratchpad scratchpad;
    // LocalMemory removed from here, it is now inside PUs

    SDMAEngine sdma;
    IDMAEngine idma;
    
    ProcessingUnit pu0;
    ProcessingUnit pu1;

    std::vector<VLIWPacket> instruction_memory;
    int pc;
    long cycle_count;

    void dispatch_packet(const VLIWPacket& packet);
    bool check_structural_hazard(const VLIWPacket& packet) const;
    void log_state(const std::string& action);
};
