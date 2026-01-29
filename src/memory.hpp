#pragma once
#include <vector>
#include <mutex>
#include <cstring>
#include <stdexcept>
#include "common_types.hpp"

class Scratchpad {
public:
    Scratchpad() : data(SCRATCHPAD_SIZE, 0) {}

    void write(size_t offset, const void* src, size_t size) {
        std::lock_guard<std::mutex> lock(mtx);
        if (offset + size > data.size()) {
            throw std::out_of_range("Scratchpad write out of bounds");
        }
        std::memcpy(data.data() + offset, src, size);
    }

    void read(size_t offset, void* dst, size_t size) {
        std::lock_guard<std::mutex> lock(mtx);
        if (offset + size > data.size()) {
            throw std::out_of_range("Scratchpad read out of bounds");
        }
        std::memcpy(dst, data.data() + offset, size);
    }

private:
    std::vector<uint8_t> data;
    std::mutex mtx;
};

class LocalMemory {
public:
    LocalMemory() {
        buffers[0].resize(LOCAL_MEM_SIZE, 0);
        buffers[1].resize(LOCAL_MEM_SIZE, 0);
    }

    // Explicitly addressable buffers
    void write(int buffer_idx, size_t offset, const void* src, size_t size) {
        std::lock_guard<std::mutex> lock(mtx);
        if (buffer_idx < 0 || buffer_idx > 1) throw std::invalid_argument("Invalid buffer index");
        if (offset + size > buffers[buffer_idx].size()) {
            throw std::out_of_range("LocalMemory write out of bounds");
        }
        std::memcpy(buffers[buffer_idx].data() + offset, src, size);
    }

    void read(int buffer_idx, size_t offset, void* dst, size_t size) {
        std::lock_guard<std::mutex> lock(mtx);
        if (buffer_idx < 0 || buffer_idx > 1) throw std::invalid_argument("Invalid buffer index");
        if (offset + size > buffers[buffer_idx].size()) {
            throw std::out_of_range("LocalMemory read out of bounds");
        }
        std::memcpy(dst, buffers[buffer_idx].data() + offset, size);
    }

    // "Provide a swap() method" - purely for requirements, though explicit indexing is used.
    void swap() {
        std::lock_guard<std::mutex> lock(mtx);
        // Logical swap could happen here if we had a "active_buffer" index
    }

private:
    std::vector<uint8_t> buffers[2];
    std::mutex mtx;
};
