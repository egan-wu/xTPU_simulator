#pragma once
#include <vector>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include "common_types.hpp"
#include "memory_interface.hpp"

// ---------------------------------------------------------------------------
// SystemMemory — 實作 IMemoryPort (P1-3)
//
// 模擬主記憶體（DRAM / HBM）。SDMAEngine 的資料來源。
// 預設 16MB，可在 Simulator 建構時指定大小。
//
// 提供測試輔助方法讓測試案例快速填充已知 pattern，
// 使 P2-2 的資料正確性驗證得以端對端追蹤每個 byte。
//
// DDR 整合路徑（P3-2）：
//   DDRAdapter 也實作 IMemoryPort，Simulator 建構時可改傳 DDRAdapter 給 SDMAEngine，
//   SystemMemory 此時成為 DDRAdapter 的內部後端儲存，不影響 Engine 程式碼。
// ---------------------------------------------------------------------------
class SystemMemory : public IMemoryPort {
public:
    explicit SystemMemory(size_t size = SYSTEM_MEMORY_SIZE)
        : data_(size, 0) {}

    void write(uint64_t addr, const void* src, size_t size) override {
        std::lock_guard<std::mutex> lock(mtx_);
        // CR3-2: 溢位安全邊界檢查：避免 addr + size 整數溢位繞過上界
        if (size > data_.size() || addr > data_.size() - size)
            throw std::out_of_range("SystemMemory write out of bounds");
        std::memcpy(data_.data() + static_cast<size_t>(addr), src, size);
    }

    void read(uint64_t addr, void* dst, size_t size) override {
        std::lock_guard<std::mutex> lock(mtx_);
        // CR3-2: 溢位安全邊界檢查
        if (size > data_.size() || addr > data_.size() - size)
            throw std::out_of_range("SystemMemory read out of bounds");
        std::memcpy(dst, data_.data() + static_cast<size_t>(addr), size);
    }

    size_t capacity() const override { return data_.size(); }

    // ── 測試輔助（P2-2 資料正確性驗證使用）────────────────────────────────────

    // 全部填充單一值
    void fill(uint8_t value) {
        std::lock_guard<std::mutex> lock(mtx_);
        std::fill(data_.begin(), data_.end(), value);
    }

    // 遞增序列：data[i] = i & 0xFF（容易驗證傳輸是否無損）
    void fill_incremental() {
        std::lock_guard<std::mutex> lock(mtx_);
        for (size_t i = 0; i < data_.size(); ++i)
            data_[i] = static_cast<uint8_t>(i & 0xFF);
    }

    // 四字節循環 pattern：data[i] = (pattern >> ((i%4)*8)) & 0xFF
    void fill_pattern(uint32_t pattern) {
        std::lock_guard<std::mutex> lock(mtx_);
        for (size_t i = 0; i < data_.size(); ++i)
            data_[i] = static_cast<uint8_t>((pattern >> ((i % 4) * 8)) & 0xFF);
    }

private:
    std::vector<uint8_t> data_;
    std::mutex mtx_;
};

// ---------------------------------------------------------------------------
// Scratchpad — 實作 IMemoryPort (P1-1)
//
// 1MB 片上共用 SRAM。SDMAEngine 寫入，IDMAEngine 讀取。
// 透過 IMemoryPort 介面操作，未來可替換為 DDRAdapter 等其他後端。
// ---------------------------------------------------------------------------
class Scratchpad : public IMemoryPort {
public:
    Scratchpad() : data_(SCRATCHPAD_SIZE, 0) {}

    void write(uint64_t addr, const void* src, size_t size) override {
        std::lock_guard<std::mutex> lock(mtx_);
        // CR3-2: 溢位安全邊界檢查（統一三個記憶體類別的模式）
        if (size > data_.size() || addr > data_.size() - size)
            throw std::out_of_range("Scratchpad write out of bounds");
        std::memcpy(data_.data() + static_cast<size_t>(addr), src, size);
    }

    void read(uint64_t addr, void* dst, size_t size) override {
        std::lock_guard<std::mutex> lock(mtx_);
        // CR3-2: 溢位安全邊界檢查
        if (size > data_.size() || addr > data_.size() - size)
            throw std::out_of_range("Scratchpad read out of bounds");
        std::memcpy(dst, data_.data() + static_cast<size_t>(addr), size);
    }

    size_t capacity() const override { return data_.size(); }

private:
    std::vector<uint8_t> data_;
    std::mutex mtx_;
};

// ---------------------------------------------------------------------------
// LocalMemory — 實作 IBufferedMemory (P1-1 + P2-3)
//
// 每個 PU 的私有記憶體，64KB × 2 buffer。
// - write_buffer / read_buffer：IDMA 明確指定 buffer index
// - swap_buffers：Compute 完成後切換 active buffer（Double Buffering）
// - IMemoryPort::write/read：ComputeEngine 透明存取 active buffer
//
// CR3-4: capacity() 語意說明
//   IMemoryPort 語意：capacity() 回傳透過 write/read 可定址的最大範圍。
//   write/read 委派到 active buffer（單一 buffer = LOCAL_MEM_SIZE）。
//   因此 capacity() 回傳 LOCAL_MEM_SIZE，與 IMemoryPort 語意一致。
//   如需取得雙 buffer 總容量，請使用 total_capacity()。
// ---------------------------------------------------------------------------
class LocalMemory : public IBufferedMemory {
public:
    LocalMemory() : active_buf_(0) {
        buffers_[0].resize(LOCAL_MEM_SIZE, 0);
        buffers_[1].resize(LOCAL_MEM_SIZE, 0);
    }

    // --- IBufferedMemory 實作 ---

    void write_buffer(int buf_idx, uint64_t offset, const void* src, size_t size) override {
        std::lock_guard<std::mutex> lock(mtx_);
        validate(buf_idx, offset, size);
        std::memcpy(buffers_[buf_idx].data() + static_cast<size_t>(offset), src, size);
    }

    void read_buffer(int buf_idx, uint64_t offset, void* dst, size_t size) override {
        std::lock_guard<std::mutex> lock(mtx_);
        validate(buf_idx, offset, size);
        std::memcpy(dst, buffers_[buf_idx].data() + static_cast<size_t>(offset), size);
    }

    // 切換 active buffer（P2-3：Double Buffer 語意完整實作）
    // 使用 mutex 保護 read-modify-write，確保併發 swap 不會遺失更新。
    void swap_buffers() override {
        std::lock_guard<std::mutex> lock(mtx_);
        active_buf_.store(1 - active_buf_.load(std::memory_order_relaxed),
                          std::memory_order_release);
    }

    // active_buf_ 是 atomic，不需要 lock 即可讀取
    int active_buffer() const override {
        return active_buf_.load(std::memory_order_acquire);
    }

    // CR3-4: capacity() 回傳 per-buffer 大小（= IMemoryPort::write/read 可定址範圍）
    // 與 SystemMemory / Scratchpad 的語意一致（皆為單一定址空間的最大範圍）
    size_t capacity() const override { return LOCAL_MEM_SIZE; }

    // 雙 buffer 的總儲存空間（供需要知道硬體總容量的呼叫端使用）
    size_t total_capacity() const { return LOCAL_MEM_SIZE * 2; }

private:
    std::vector<uint8_t> buffers_[2];
    std::mutex mtx_;
    std::atomic<int> active_buf_;  // 目前 active 的 buffer index（0 或 1）

    void validate(int buf_idx, uint64_t offset, size_t size) const {
        if (buf_idx < 0 || buf_idx > 1)
            throw std::invalid_argument("LocalMemory: invalid buffer index");
        // CR3-2: 溢位安全邊界檢查（統一三個記憶體類別的模式）
        const size_t buf_size = buffers_[buf_idx].size();
        if (size > buf_size || offset > buf_size - size)
            throw std::out_of_range("LocalMemory: access out of bounds");
    }
};
