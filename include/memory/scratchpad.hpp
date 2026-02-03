#pragma once
#include <vector>
#include <cstdint>
#include <iostream>
#include <cstring>

class Scratchpad {
public:
    Scratchpad(size_t size_bytes = 1024 * 1024) : memory(size_bytes, 0) {}

    void write(uint64_t offset, const void* data, size_t size) {
        if (offset + size > memory.size()) {
            std::cerr << "[Scratchpad] Write out of bounds!" << std::endl;
            return;
        }
        std::memcpy(memory.data() + offset, data, size);
    }

    void read(uint64_t offset, void* buffer, size_t size) const {
        if (offset + size > memory.size()) {
            std::cerr << "[Scratchpad] Read out of bounds!" << std::endl;
            return;
        }
        std::memcpy(buffer, memory.data() + offset, size);
    }

private:
    std::vector<uint8_t> memory;
};
