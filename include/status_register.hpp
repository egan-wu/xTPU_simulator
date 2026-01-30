#pragma once
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <iostream>
#include <bitset>
#include <iomanip>
#include "common_types.hpp"

class StatusRegister {
public:
    StatusRegister() : busy_mask(0) {}

    void set_busy(uint32_t mask) {
        std::lock_guard<std::mutex> lock(mtx);
        uint32_t old = busy_mask.load();
        busy_mask.fetch_or(mask);
        log_transition(old, busy_mask.load());
    }

    void clear_busy(uint32_t mask) {
        std::lock_guard<std::mutex> lock(mtx);
        uint32_t old = busy_mask.load();
        busy_mask.fetch_and(~mask);
        log_transition(old, busy_mask.load());
        cv.notify_all();
    }

    void wait_on_mask(uint32_t mask) {
        std::unique_lock<std::mutex> lock(mtx);
        if ((busy_mask.load() & mask) != 0) {
             // Optional: Log waiting
             // std::cout << "[Sync] Waiting for mask: 0b" << std::bitset<5>(mask) << std::endl;
        }
        cv.wait(lock, [this, mask] {
            return (busy_mask.load() & mask) == 0;
        });
    }

    uint32_t get_status() const {
        return busy_mask.load();
    }

private:
    std::atomic<uint32_t> busy_mask;
    std::mutex mtx;
    std::condition_variable cv;

    void log_transition(uint32_t old_val, uint32_t new_val) {
        std::cout << "[Scoreboard] [0b" << std::bitset<5>(old_val)
                  << "] -> [0b" << std::bitset<5>(new_val) << "]" << std::endl;
    }
};
