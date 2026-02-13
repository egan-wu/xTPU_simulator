#pragma once
#include <cstdint>
#include <iostream>
#include <bitset>
#include <iomanip>

class StatusRegister {
public:
    StatusRegister() : busy_mask(0) {}

    // Non-locking because we are single-threaded cycle-accurate now
    void set_busy(uint32_t mask) {
        uint32_t old = busy_mask;
        busy_mask |= mask;
        log_transition(old, busy_mask);
    }

    void clear_busy(uint32_t mask) {
        uint32_t old = busy_mask;
        busy_mask &= ~mask;
        log_transition(old, busy_mask);
    }

    uint32_t get_status() const {
        return busy_mask;
    }

    bool check_stall(uint32_t sync_mask) const {
        return (busy_mask & sync_mask) != 0;
    }

private:
    uint32_t busy_mask;

    void log_transition(uint32_t old_val, uint32_t new_val) {
        // Only log if something actually changed (optimization)
        if (old_val == new_val) return;

        // Log in a compact format showing which bits cleared
        std::cout << "[Scoreboard] [0b" << std::bitset<7>(old_val)
                  << "] -> [0b" << std::bitset<7>(new_val) << "]";

        // Print readable names of cleared bits
        uint32_t cleared = old_val & ~new_val;
        if (cleared) {
            std::cout << " Cleared: ";
            if (cleared & (1<<0)) std::cout << "IDMA_B0 ";
            if (cleared & (1<<1)) std::cout << "IDMA_B1 ";
            if (cleared & (1<<2)) std::cout << "IDMA_B2 ";
            if (cleared & (1<<3)) std::cout << "IDMA_B3 ";
            if (cleared & (1<<4)) std::cout << "SDMA ";
            if (cleared & (1<<5)) std::cout << "MXU ";
            if (cleared & (1<<6)) std::cout << "VEC ";
        }
        std::cout << std::endl;
    }
};
