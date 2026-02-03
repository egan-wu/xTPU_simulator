#pragma once
#include <vector>
#include <iostream>
#include "common.hpp"
#include "status_register.hpp"
#include "memory/scratchpad.hpp"
#include "memory/local_memory.hpp"
#include "engines/sdma_engine.hpp"
#include "engines/idma_engine.hpp"
#include "engines/mxu_engine.hpp"
#include "engines/vector_engine.hpp"

class Simulator {
public:
    Simulator();

    void load_program(const std::vector<VLIWPacket>& program);
    void run(int max_cycles);

    // Single step of the cycle-accurate simulation
    void step();

    bool is_busy() const {
        return sdma.is_busy() || idma.is_busy() || mxu.is_busy() || vector_unit.is_busy();
    }

private:
    StatusRegister status_reg;
    Scratchpad scratchpad;
    LocalMemory local_memory;

    SDMAEngine sdma;
    IDMAEngine idma;
    MXUEngine mxu;
    VectorEngine vector_unit;

    std::vector<VLIWPacket> instruction_memory;
    int pc;
    long cycle_count;

    void dispatch_packet(const VLIWPacket& packet);
    bool check_structural_hazard(const VLIWPacket& packet) const;
    void log_state(const std::string& action);
};
