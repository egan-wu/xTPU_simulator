#pragma once
#include "engines.hpp"
#include "status_register.hpp"
#include "memory.hpp"
#include "commands.hpp"

class Simulator {
public:
    Simulator()
        : sdma(status_reg, scratchpad),
          idma(status_reg, scratchpad, local_mem[0], local_mem[1]),
          pu0(status_reg, local_mem[0], STATUS_PU0_CMD_BUSY),
          pu1(status_reg, local_mem[1], STATUS_PU1_CMD_BUSY)
    {}

    void dispatch_packet(const VLIWPacket& packet);

    // Direct access for testing verification
    StatusRegister& get_scoreboard() { return status_reg; }
    Scratchpad& get_scratchpad() { return scratchpad; }
    LocalMemory& get_local_mem(int pu_idx) {
        if (pu_idx < 0 || pu_idx > 1) throw std::out_of_range("Invalid PU index");
        return local_mem[pu_idx];
    }

private:
    StatusRegister status_reg;
    Scratchpad scratchpad;
    LocalMemory local_mem[2]; // 0 for PU0, 1 for PU1

    SDMAEngine sdma;
    IDMAEngine idma;
    ComputeEngine pu0;
    ComputeEngine pu1;
};
