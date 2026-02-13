#pragma once
#include <vector>
#include <cstdint>
#include <iostream>
#include <cstring>

class LocalMemory {
public:
    // 4 Banks (0-3), each with some size (e.g. 64KB)
    LocalMemory(size_t bank_size = 65536) {
        for(int i=0; i<4; ++i) {
            banks[i].resize(bank_size, 0);
        }
    }

    void write(int bank_idx, uint64_t offset, const void* data, size_t size) {
        if (bank_idx < 0 || bank_idx >= 4) {
             std::cerr << "[LocalMemory] Invalid Bank ID: " << bank_idx << std::endl;
             return;
        }
        if (offset + size > banks[bank_idx].size()) {
            std::cerr << "[LocalMemory] Write out of bounds on Bank " << bank_idx << std::endl;
            return;
        }
        std::memcpy(banks[bank_idx].data() + offset, data, size);
    }

    // In a real TPU, Compute Units read directly from Local Memory
    // We just provide a stub for completeness
private:
    std::vector<uint8_t> banks[4];
};
