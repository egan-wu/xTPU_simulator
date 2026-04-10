#pragma once
#include <cstdint>
#include <cstddef>

// ---------------------------------------------------------------------------
// IMemoryPort — 任何平坦定址記憶體的通用介面 (P1-1)
//
// 設計原則：依賴反轉 (Dependency Inversion Principle)
//   高層模組 (Engine) 依賴此抽象，不直接依賴具體實作。
//   允許在不修改 Engine 原始碼的情況下替換記憶體後端：
//     Scratchpad  → 簡單 SRAM model
//     SystemMemory → 平坦 DRAM model (P1-3)
//     DDRAdapter  → 接入外部 DDR Controller project (P3-2)
// ---------------------------------------------------------------------------
class IMemoryPort {
public:
    virtual ~IMemoryPort() = default;

    virtual void write(uint64_t addr, const void* src, size_t size) = 0;
    virtual void read(uint64_t addr, void* dst, size_t size) = 0;

    // 容量查詢（供 DDRAdapter 做位址範圍驗證）
    virtual size_t capacity() const = 0;

    // P3-2: 此後端是否自行管理 SimClock 時序。
    // 回傳 true 時，Engine 在呼叫 write/read 前不加固定延遲，
    // 避免 SDMAEngine 固定延遲 + LPDDR5Adapter 內部延遲雙重計算。
    // 預設 false（SystemMemory / Scratchpad 不自行推進時鐘）。
    virtual bool has_own_timing() const { return false; }
};

// ---------------------------------------------------------------------------
// IBufferedMemory — Double Buffer 語意的記憶體介面 (P1-1 + P2-3)
//
// 繼承 IMemoryPort，額外提供：
//   1. write_buffer / read_buffer：明確指定 buffer index（IDMA 使用）
//   2. swap_buffers：原子切換 active buffer（Compute 完成後呼叫）
//   3. active_buffer：查詢當前 active buffer index
//
// IMemoryPort 的 write / read（無 buffer 參數版本）預設操作 active buffer，
// 讓 ComputeEngine 可以透過 IMemoryPort 介面透明存取 local memory。
// ---------------------------------------------------------------------------
class IBufferedMemory : public IMemoryPort {
public:
    // 明確指定 buffer 的讀寫（供 IDMAEngine 按 cmd.buffer_idx 寫入）
    virtual void write_buffer(int buf_idx, uint64_t offset, const void* src, size_t size) = 0;
    virtual void read_buffer(int buf_idx, uint64_t offset, void* dst, size_t size) = 0;

    // Double buffer 管理
    virtual void swap_buffers() = 0;
    virtual int  active_buffer() const = 0;

    // IMemoryPort::write/read 的預設實作：委派到 active buffer
    // ComputeEngine 透過此介面讀寫，不需要知道 buffer index 細節
    void write(uint64_t addr, const void* src, size_t size) override {
        write_buffer(active_buffer(), addr, src, size);
    }
    void read(uint64_t addr, void* dst, size_t size) override {
        read_buffer(active_buffer(), addr, dst, size);
    }
};
