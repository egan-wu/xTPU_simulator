# xTPU Simulator — Architecture Evolution Plan

> Authored from a senior architecture engineer's perspective.
> Focus: correctness first, abstraction boundaries second, extensibility (DDR integration) third.

---

## Architecture Assessment Summary

Current state: **proof-of-concept for command sequencing**. The simulator correctly models the VLIW dispatch → async engine → scoreboard completion contract, but has significant gaps that block integration with realistic memory subsystems (e.g., DDR controller).

### Key Architectural Deficits

| # | Deficit | Impact | Blocks DDR? |
|---|---------|--------|-------------|
| 1 | No `MemoryInterface` abstraction — engines hardcoded to concrete `Scratchpad`/`LocalMemory` | Can't swap memory backend without rewriting engines | **Yes** |
| 2 | No timing model (cycle or event-driven) — all latency is wallclock `sleep` | DDR latency/bandwidth meaningless without simulated time | **Yes** |
| 3 | System Memory is phantom (dummy 0xAA) — no real external memory | DDR controller has nothing to connect to | **Yes** |
| 4 | No writeback path — data flows in one direction only | Can't model store-back to DDR | **Yes** |
| 5 | No bus/interconnect model — engines access memory directly | Can't model contention, arbitration, or bandwidth limits | Partially |
| 6 | StatusRegister mixed atomic+mutex, IDMA deadlock on error | Correctness risk | No |

---

## Phase 0 — Correctness (Bug Fixes)

### P0-1: IDMA 錯誤路徑導致死鎖
**狀態**: ✅ 已完成
**位置**: `src/engines.cpp:35-78`
**問題**: `IDMAEngine::process()` 在 `scratchpad.read()` 拋出異常後，不清除 busy bits，導致 `wait_on_mask` 永久阻塞。同理 `SDMAEngine::process()` (line 22-24) 的 `scratchpad.write()` 失敗也不清除 `STATUS_SDMA_BUSY`。

**解決方案**: 引入 RAII guard，保證 process() 結束時必定 clear_busy：
```cpp
struct BusyClearGuard {
    StatusRegister& sr;
    uint32_t mask;
    ~BusyClearGuard() { if (mask) sr.clear_busy(mask); }
};
```
在 `process()` 入口建立 guard，正常路徑和異常路徑都能清除。

**驗收條件**:
- 觸發 OOB 讀寫不死鎖
- 所有現有測試通過

---

### P0-2: Engine::running 非 thread-safe
**狀態**: ✅ 已完成
**位置**: `include/engines.hpp:47` (`bool running`)
**問題**: 主執行緒在析構函式中寫 `running = false`（line 23），worker 在 `cv.wait` lambda 中讀 `running`（line 54）。雖然有 mutex 保護寫入端，但 `cv.wait` 的 predicate 在 spurious wakeup 時可能在未持鎖狀態下評估（實際上 `cv.wait` 持鎖評估，此處邏輯正確），真正的問題是語義不清。
**解決方案**: 改為 `std::atomic<bool> running`，消除對 mutex 保護此 flag 的隱式依賴。

---

### P0-3: StatusRegister 混用 atomic + mutex
**狀態**: ✅ 已完成
**位置**: `include/status_register.hpp`
**問題**: `busy_mask` 是 `std::atomic<uint32_t>` 但所有修改都在 `mutex` 保護下。`get_status()` 繞過 mutex 直接 `load()`，語義不一致。
**解決方案**: 移除 `atomic`，統一用 `mutex` + 普通 `uint32_t`。原因：`condition_variable::wait()` 本身需要 `unique_lock`，atomic 無法與 CV 配合，所以 mutex 是必須的，atomic 是多餘的。
```cpp
// After:
uint32_t busy_mask = 0;  // protected by mtx, NOT atomic
uint32_t get_status() {
    std::lock_guard<std::mutex> lock(mtx);
    return busy_mask;
}
```

---

## Phase 1 — Abstraction Layer (DDR Integration Foundation)

> **設計原則**: 在動手加功能之前，先建立正確的抽象邊界。
> 這些抽象是後續 DDR 整合的 **必要前置條件**。

### P1-1: 引入 MemoryInterface 抽象層
**狀態**: ✅ 已完成
**位置**: 新增 `include/memory_interface.hpp`，重構 `include/memory.hpp`
**問題**: SDMAEngine 直接依賴 `Scratchpad` 類型（`engines.hpp:73`），IDMAEngine 直接依賴 `Scratchpad` + `LocalMemory`（`engines.hpp:83-84`）。若要接入 DDR controller，必須修改 engine 原始碼。

**解決方案**: 定義通用記憶體介面，讓 engines 依賴抽象而非具體實作：
```cpp
// include/memory_interface.hpp
class IMemoryPort {
public:
    virtual ~IMemoryPort() = default;

    // 基本讀寫（阻塞式，功能正確模式）
    virtual void write(uint64_t addr, const void* src, size_t size) = 0;
    virtual void read(uint64_t addr, void* dst, size_t size) = 0;

    // 容量查詢
    virtual size_t capacity() const = 0;
};

// IBufferedMemory: 給有 double-buffer 語意的記憶體
class IBufferedMemory : public IMemoryPort {
public:
    virtual void write(int buffer_idx, uint64_t addr, const void* src, size_t size) = 0;
    virtual void read(int buffer_idx, uint64_t addr, void* dst, size_t size) = 0;
    virtual void swap_buffers() = 0;

    // IMemoryPort 的 read/write 操作 active buffer
    void write(uint64_t addr, const void* src, size_t size) override;
    void read(uint64_t addr, void* dst, size_t size) override;
};
```

**重構影響**:
- `Scratchpad` implements `IMemoryPort`
- `LocalMemory` implements `IBufferedMemory`
- `SDMAEngine` 建構參數：`IMemoryPort& system_mem, IMemoryPort& scratchpad`
- `IDMAEngine` 建構參數：`IMemoryPort& scratchpad, IBufferedMemory& lm0, IBufferedMemory& lm1`
- 未來 DDR controller 只需 implement `IMemoryPort` 即可無縫接入

**驗收條件**:
- 所有 engine 依賴 `IMemoryPort` / `IBufferedMemory` 而非具體類別
- 現有測試 zero-change 通過

---

### P1-2: 引入 SimulatedClock（虛擬時間模型）
**狀態**: ✅ 已完成
**位置**: 新增 `include/sim_clock.hpp`
**問題**: 目前用 `std::this_thread::sleep_for(ms)` 模擬延遲（`engines.cpp:89`）。這種 wallclock 方式：
1. 無法精確控制——OS 調度抖動可達數十 ms
2. DDR 延遲通常是 ns 級，wallclock sleep 無意義
3. 無法做效能分析（cycle count、utilization）

**解決方案**: 引入虛擬時鐘，所有延遲以 **simulated ticks** 表達：
```cpp
// include/sim_clock.hpp
class SimClock {
public:
    using Tick = uint64_t;

    Tick now() const { return current_tick.load(); }
    void advance(Tick delta) { current_tick += delta; }

    // Engine 呼叫：模擬此操作的延遲
    void simulate_latency(Tick cycles) { advance(cycles); }

private:
    std::atomic<Tick> current_tick{0};
};
```

**時序參數化**（可配置的延遲表）:
```cpp
struct TimingConfig {
    SimClock::Tick sdma_latency_per_cacheline = 10;  // ticks per 64B
    SimClock::Tick idma_latency_per_cacheline = 5;
    SimClock::Tick matmul_latency = 100;
    SimClock::Tick vector_latency = 20;
    SimClock::Tick scalar_latency = 5;
    // DDR 相關（未來）:
    SimClock::Tick ddr_cas_latency = 22;   // CAS Latency (CL)
    SimClock::Tick ddr_ras_to_cas = 22;    // tRCD
    SimClock::Tick ddr_row_precharge = 22; // tRP
};
```

**驗收條件**:
- 所有 engine 用 `SimClock::simulate_latency()` 取代 `sleep_for`
- 測試可查詢 `sim.clock().now()` 驗證時序正確性

---

### P1-3: 實現 SystemMemory 模型
**狀態**: ✅ 已完成
**位置**: `include/memory.hpp`
**問題**: SDMAEngine 製造虛假 0xAA 數據（`engines.cpp:19`），src_addr 完全沒用到

**解決方案**: 新增 `SystemMemory` 實作 `IMemoryPort`：
```cpp
class SystemMemory : public IMemoryPort {
public:
    explicit SystemMemory(size_t size = 16 * 1024 * 1024);  // 16MB default
    void write(uint64_t addr, const void* src, size_t size) override;
    void read(uint64_t addr, void* dst, size_t size) override;
    size_t capacity() const override;

    // 測試輔助
    void fill(uint8_t value);
    void fill_incremental();  // addr[i] = i & 0xFF
private:
    std::vector<uint8_t> data_;
    std::mutex mtx_;
};
```

**關鍵改動**:
- `Simulator` 持有 `SystemMemory` 實例
- `SDMAEngine` 建構參數新增 `IMemoryPort& system_mem`
- `SDMAEngine::process()` 從 `system_mem.read(cmd.src_addr, ...)` 取得真實資料

**驗收條件**:
- 在 SystemMemory 寫入 pattern，執行 sDMA → scratchpad，比對資料一致
- 移除所有 dummy 0xAA 製造碼

---

### P1-4: 完成資料迴路（Writeback Path）
**狀態**: ✅ 已完成
**位置**: `include/commands.hpp`、`src/engines.cpp`、`src/simulator.cpp`
**問題**: 資料只能 system_mem → scratchpad → local_mem（單向）。無法：
- Compute 結果寫回 Scratchpad（iDMA 反向）
- Scratchpad 結果寫回 System Memory（sDMA 反向）

**解決方案**:
1. `DMA_Command` 新增 `direction` 欄位：
```cpp
enum class DMADirection { TO_DEVICE, FROM_DEVICE };
struct DMA_Command {
    DMAType type = DMAType::NOP;
    DMADirection direction = DMADirection::TO_DEVICE;
    uint64_t src_addr = 0;
    uint64_t dst_addr = 0;
    size_t size = 0;
    uint32_t target_mask = 0;
    int buffer_idx = 0;
};
```
2. `SDMAEngine::process()` 依 direction 決定：
   - `TO_DEVICE`：system_mem → scratchpad（現有邏輯）
   - `FROM_DEVICE`：scratchpad → system_mem（新增）
3. `IDMAEngine::process()` 依 direction 決定：
   - `TO_DEVICE`：scratchpad → local_mem（現有邏輯）
   - `FROM_DEVICE`：local_mem → scratchpad（新增）

**驗收條件**:
- 端到端測試：system_mem → scratchpad → local_mem → compute → local_mem → scratchpad → system_mem
- 最終 system_mem 資料正確

---

## Phase 2 — Functional Completeness

### P2-1: Compute Engine 實作真實運算
**狀態**: ✅ 已完成
**位置**: `src/engines.cpp:84-98`、`include/commands.hpp`
**問題**: ComputeEngine 只做 sleep（`engines.cpp:89`），不存取 LocalMemory

**解決方案**: 擴展 `Compute_Command` 並實作運算：
```cpp
struct Compute_Command {
    ComputeType type = ComputeType::NOP;
    int buffer_idx = 0;
    uint32_t src_offset = 0;      // input data offset in local mem
    uint32_t dst_offset = 0;      // output data offset in local mem
    uint32_t length = 0;          // data length in bytes
    uint32_t simulated_cycles = 0; // SimClock ticks (replaces ms)
};
```
運算邏輯：
- `SCALAR`：`dst[i] = src[i] + 1`（逐元素）
- `VECTOR`：`dst[i] = src[i] * src[i]`（逐元素平方，或 MAD）
- `MATMUL`：小矩陣乘法（4x4 tile），讀兩組 input，寫一組 output

**驗收條件**:
- 測試寫入已知數據，compute 後讀回驗證結果正確

---

### P2-2: 資料正確性測試套件
**狀態**: ✅ 已完成
**位置**: 新增或擴展 `tests/test_simulator.cpp`

**測試場景**:
| # | 場景 | 驗證目標 |
|---|------|----------|
| 1 | Pattern fill → sDMA → verify scratchpad | SystemMemory → Scratchpad 傳輸正確 |
| 2 | sDMA → iDMA broadcast → verify both local mems | Broadcast 兩端資料一致 |
| 3 | Double buffer isolation | Buffer 0 compute 中，Buffer 1 載入新資料，互不干擾 |
| 4 | End-to-end with compute | load → compute → writeback → verify in system_mem |
| 5 | Incremental pattern | [0,1,2,...,N] 端到端無損傳輸 |

**驗收條件**:
- 每個場景有獨立 test function
- `assert` 驗證每個 byte

---

### P2-3: LocalMemory Double Buffer 完善
**狀態**: ✅ 已完成（P1-1 重構時一併實作）
**位置**: `include/memory.hpp`、`include/memory_interface.hpp`
**備註**: `swap_buffers()`、`active_buffer()`、`IMemoryPort::read/write` 委派到 active buffer
均已在 P1-1 的 IBufferedMemory 介面重構中完整實作，Test 13 的 Double Buffer Isolation
驗證了正確性。

---

### P2-4: 統一錯誤處理與 Error Bit
**狀態**: ✅ 已完成
**位置**: `include/common_types.hpp`、`include/status_register.hpp`

**解決方案**:
- 新增 `STATUS_ERROR` bit（`1 << 5`）
- 任何 engine 異常時 `set_busy(STATUS_ERROR)`
- `Simulator::dispatch_packet()` 在 sync 後檢查 error bit
- 提供 `get_error_info()` 方法查詢錯誤詳情

---

## Phase 2 — Code Review Action Items（P2-CR）

> 來源：Phase 2 整體 Code Review（Senior Architect 視角）
> 分級：🔴 需修正（阻塞性）｜🟡 建議改善（非阻塞）

---

### P2-CR-1：🔴 MATMUL 維度硬編碼，缺乏 `length` 驗證
**狀態**: ✅ 已完成
**位置**: `src/engines.cpp` — ComputeEngine MATMUL 分支
**問題**: 迴圈以 `constexpr uint32_t N = 4` 為邊界，但 buffer 大小由 `cmd.length` 動態決定。
若呼叫端誤傳 `length ≠ 16`（如 8 或 64），不會拋出異常，只會靜默產生錯誤的計算結果。

**修正方案**:
```cpp
} else if (cmd.type == ComputeType::MATMUL) {
    constexpr uint32_t N = 4;
    if (cmd.length != N * N) {          // ← 加入此驗證
        status_reg.set_error("[Compute] MATMUL requires length == 16 (4x4 uint8_t matrix)");
        return;
    }
    // ... 現有乘法邏輯不變
```

**驗收條件**:
- `length ≠ 16` 的 MATMUL 指令應設定 STATUS_ERROR 並 return
- 新增測試：傳入 `length=8` 的 MATMUL，驗證 STATUS_ERROR 被設定、error_info 含「requires length」

---

### P2-CR-2：🔴 `commands.hpp` 殘留未使用的 `#include <vector>`
**狀態**: ✅ 已完成
**位置**: `include/commands.hpp:3`
**問題**: `#include <vector>` 在 `commands.hpp` 中無任何使用，是初始設計的殘留。
引入不必要的標頭依賴鏈，增加所有 include `commands.hpp` 翻譯單元的編譯時間。

**修正方案**: 直接移除該行。

**驗收條件**: `make clean && make` 編譯無錯誤，無警告。

---

### P2-CR-3：🟡 Test 12 (MATMUL) 只測試單位矩陣，無法排除「copy B」實作
**狀態**: ✅ 已完成
**位置**: `tests/test_simulator.cpp` — `test_compute_matmul()`
**問題**: 使用 `A = I`，導致 `I × B = B` 恆成立。
若 MATMUL 誤實作為 `memcpy(C, B, length)`（直接複製 B），測試同樣通過。

**修正方案**: 新增非平凡矩陣測試（Test 18），使用對角矩陣 A：
```
A = diag([2, 3, 1, 1])    # A[i][i] = {2,3,1,1}，其餘為 0
B = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
期望結果 C = [[2,4,6,8],[15,18,21,24],[9,10,11,12],[13,14,15,16]]
```

**驗收條件**:
- 新增 `test_compute_matmul_nontrivial()` 函式
- 使用非對稱 A 矩陣，預計算期望值後逐元素 assert
- 同時保留原有 Test 12（I × B = B）作為 sanity check

---

### P2-CR-4：🟡 `Compute_Command::length` 在 MATMUL 語意與 SCALAR/VECTOR 不一致
**狀態**: ✅ 已完成
**位置**: `include/commands.hpp`、`src/engines.cpp`
**問題**: `length` 在不同 operation type 有不同含義：

| Operation | 讀取 bytes        | 寫入 bytes |
|-----------|------------------|-----------|
| SCALAR    | `length`         | `length`  |
| VECTOR    | `length`         | `length`  |
| MATMUL    | `2 × length`（A+B）| `length`（C）|

`MATMUL` 的 `length` 表示「單一矩陣大小」，而非「操作資料總量」，容易誤導使用者。

**修正方案（低優先，擇一）**:
- 選項 A：在 `commands.hpp` 的 MATMUL 段落加強文件說明（立即可行）
- 選項 B：當 `N×N` 固定為 4 時，未來如需可變維度，考慮新增 `matmul_dim` 欄位

**驗收條件**: 至少更新 `commands.hpp` 的 MATMUL 欄位說明，明確標示 `length = 單一矩陣 bytes（= N×N）`

---

### P2-CR-5：🟡 `dispatch_packet()` 在 sync 後未主動提示 STATUS_ERROR
**狀態**: ✅ 已完成
**位置**: `src/simulator.cpp` — `dispatch_packet()`
**問題**: taskList P2-4 spec 要求「在 sync 後檢查 error bit」。
目前採用 pull 模式（呼叫端自行 `has_error()`），不符合原始 spec，且錯誤完全靜默。

**修正方案**: 在 `wait_on_mask()` 完成後加入警告輸出（不改變控制流，保持向下相容）：
```cpp
if (packet.sync_mask != 0) {
    status_reg_.wait_on_mask(packet.sync_mask);
    // P2-4: 偵測到 pending error 時提醒呼叫端
    if (status_reg_.has_error()) {
        std::cerr << "[Simulator] Warning: STATUS_ERROR is set after sync. "
                  << "Call has_error() / get_error_info() to inspect, "
                  << "clear_error() to acknowledge." << std::endl;
    }
}
```

**驗收條件**:
- Test 4、5、16 的錯誤情境會輸出 `[Simulator] Warning:` 訊息
- 現有所有 17 個測試仍然通過（控制流不改變）

---

## Phase 2 — Post-Fix Code Review Action Items（CR3）

> 來源：P2-CR 全部完成後的第二次 Code Review（全面性檢驗）
> 分級：🔴 需修正（阻塞性）｜🟡 建議改善（非阻塞）

---

### CR3-1：🔴 IDMA broadcast 部分失敗語意不一致 + error_info 被覆寫
**狀態**: ✅ 已完成
**位置**: `src/engines.cpp` — IDMAEngine::process() TO_DEVICE broadcast 分支（第 134–153 行）
**問題**:
1. **行為不對稱**：TO_DEVICE broadcast 中 PU0 write 失敗後**不 return**，繼續嘗試 PU1 write；
   但 FROM_DEVICE 路徑在第一個 read 失敗後**立即 return**。
2. **error_info 覆寫**：若 PU0 和 PU1 的 write 都失敗，PU1 的 `set_error()` 覆寫 PU0 的 error_info，
   第一個錯誤訊息永久遺失，造成除錯困難。

**修正方案**:
```cpp
// 方案 A（推薦）：PU0 失敗後 return，與 FROM_DEVICE 行為一致
if (target_pu0) {
    try {
        local_mem0_.write_buffer(...);
    } catch (const std::exception& e) {
        status_reg.set_error(msg);
        return;  // ← 新增：一致化錯誤處理
    }
}
// 方案 B（若需保留 best-effort）：串接錯誤訊息而非覆寫
// combined_msg += "[PU0] " + e.what() + "; ";
```

**驗收條件**:
- TO_DEVICE 和 FROM_DEVICE 錯誤處理行為對稱（或有明確文件說明 best-effort 語意）
- 新增測試：broadcast 中 PU0 write OOB → PU1 不執行 → error_info 包含 PU0 相關訊息

---

### CR3-2：🔴 Scratchpad 邊界檢查與 SystemMemory 不一致，有溢位風險
**狀態**: ✅ 已完成
**位置**: `include/memory.hpp` — Scratchpad::write/read 及 SystemMemory::write/read
**問題**:
- `SystemMemory` 使用 `static_cast<size_t>(addr) + size > data_.size()`
- `Scratchpad` 使用 `addr + size > data_.size()`（無 cast）
- 兩者的 `addr + size` 在理論上有整數溢位風險（addr 接近 UINT64_MAX 時繞回小值，繞過邊界檢查）
- 三個記憶體類別（SystemMemory / Scratchpad / LocalMemory）邊界檢查風格不統一

**修正方案**: 統一使用溢位安全的慣用語：
```cpp
// 取代 addr + size > data_.size()
if (size > data_.size() || static_cast<size_t>(addr) > data_.size() - size)
    throw std::out_of_range("...");
```

**驗收條件**:
- SystemMemory、Scratchpad、LocalMemory 三者使用相同的溢位安全邊界檢查
- `make clean && make` 無錯誤、所有 19 個測試通過

---

### CR3-3：🟡 測試覆蓋率缺口 — 無 PU1 writeback (FROM_DEVICE) 測試
**狀態**: ✅ 已完成
**位置**: `tests/test_simulator.cpp`
**問題**: Test 9（writeback E2E）、Test 14（E2E + compute）、Test 15 均使用 TARGET_PU0。
PU1 的 `local_mem → scratchpad → system_mem` writeback 路徑從未被直接驗證。
若 IDMAEngine 在 FROM_DEVICE 路徑中對 PU1 的 `local_mem1_.read_buffer()` 呼叫有 bug，
現有測試無法偵測。

**修正方案**: 新增 Test 20 — PU1 writeback E2E：
```
system_mem → scratchpad → PU1 local_mem[buf=0]
→ PU1 local_mem[buf=0] → scratchpad(offset) → system_mem(offset)
驗證 system_mem 的 writeback 內容與原始資料一致
```

**驗收條件**: 新增測試函式 `test_pu1_writeback_path()`，使用 TARGET_PU1 完成端對端 writeback 驗證

---

### CR3-4：🟡 `LocalMemory::capacity()` 回傳值語意與 `IMemoryPort::write/read` 可寫入範圍不一致
**狀態**: ✅ 已完成
**位置**: `include/memory.hpp` — LocalMemory::capacity()
**問題**: `capacity()` 回傳 `LOCAL_MEM_SIZE * 2`（128KB，雙 buffer 合計），
但 `IMemoryPort::write/read`（繼承自 `IBufferedMemory`）委派到 active buffer，
最大可寫入範圍僅 `LOCAL_MEM_SIZE`（64KB）。
若未來有程式碼使用 `capacity()` 做寫入前邊界預檢，會高估可用空間而 crash。

**修正方案（擇一）**:
- 選項 A：`capacity()` 改回傳 `LOCAL_MEM_SIZE`（單一 buffer 的可定址範圍）
- 選項 B：新增 `total_capacity()` 回傳 `2 * LOCAL_MEM_SIZE`，`capacity()` 回傳 per-buffer 值
- 選項 C：在 `capacity()` 的文件註解中明確標示「回傳所有 buffer 的總容量」

**驗收條件**: `capacity()` 語意與 `IMemoryPort::write/read` 可存取範圍一致，或有明確文件說明

---

### CR3-5：🟡 Makefile 無 header 依賴追蹤
**狀態**: ✅ 已完成
**位置**: `Makefile`
**問題**: 現有 Makefile 僅列出 `.cpp` 為 prerequisite，修改 `.hpp` 不會觸發重新編譯。
開發者必須手動 `make clean && make`，Phase 3 加入更多 header 後問題加劇。

**修正方案**: 使用 GCC `-MMD -MP` 自動產生 `.d` 依賴檔：
```makefile
CXXFLAGS += -MMD -MP
DEPS = $(OBJS:.o=.d)
-include $(DEPS)
```

**驗收條件**: 修改任意 `.hpp` 後單獨 `make` 即可正確重編受影響的 `.o`

---

### CR3-6：🟡 `dispatch_packet()` 單線程使用限制未文件化
**狀態**: ✅ 已完成
**位置**: `src/simulator.cpp` — `dispatch_packet()` 及 `include/simulator.hpp`
**問題**: `dispatch_packet` 中 `set_busy()` 和 `push_command()` 非原子組合操作。
若多線程同時呼叫 `dispatch_packet`，可能出現：
  Thread A: `set_busy(SDMA)` → Thread B: `set_busy(SDMA)` → Thread A: `push_command()`
  → SDMA 收到一個 command 但 busy bit 被設了兩次，第二次 push 永遠不會被 guard 清除。
目前所有使用場景為單線程，但 API 註解未標示此限制。

**修正方案**: 在 `dispatch_packet()` 及 `Simulator` class 的 Doxygen 註解中明確標示：
```cpp
/// @note Not thread-safe. Must be called from a single thread.
void dispatch_packet(const VLIWPacket& packet);
```

**驗收條件**: 在 `simulator.hpp` 中新增 thread-safety 文件化註解

---

## Phase 3 — DDR Controller Integration

> 此章節描述如何將外部 DDR project 接入 xTPU simulator。
> **前置依賴**: Phase 1 的 `IMemoryPort` 抽象和 `SimClock` 必須完成。

### P3-1: DDR Integration Architecture

**目標拓撲**:
```
                      xTPU Simulator
                    ┌─────────────────────────────────────────────────┐
                    │                                                 │
                    │  ┌─────────┐    ┌───────────┐   ┌───────────┐  │
                    │  │  SDMA   │───>│ Scratchpad│<──│   IDMA    │  │
                    │  │ Engine  │    │  (SRAM)   │   │  Engine   │  │
                    │  └────┬────┘    └───────────┘   └─────┬─────┘  │
                    │       │                               │        │
                    │       │ IMemoryPort                   │        │
                    │       │                       ┌───────┴──────┐ │
                    │       │                       │ LocalMem x2  │ │
                    │       │                       │(double buf)  │ │
                    │       │                       └───────┬──────┘ │
                    │       │                               │        │
                    │       │                       ┌───────┴──────┐ │
                    │       │                       │ PU0 / PU1    │ │
                    │       │                       └──────────────┘ │
                    └───────┼─────────────────────────────────────────┘
                            │
                    ┌───────▼───────────────────────────────────────────┐
                    │            DDR Adapter (Glue Layer)                │
                    │  implements IMemoryPort                           │
                    │  ┌──────────────────────────────────────────────┐ │
                    │  │  Address Translation: flat addr → (rank,     │ │
                    │  │  bank, row, col)                              │ │
                    │  │  Request Queue: burst coalescing, reordering  │ │
                    │  │  Timing: tCAS, tRCD, tRP, tRFC               │ │
                    │  └──────────────┬───────────────────────────────┘ │
                    │                 │                                  │
                    │  ┌──────────────▼───────────────────────────────┐ │
                    │  │          DDR Controller (外部 project)        │ │
                    │  │  - Bank state machines                       │ │
                    │  │  - Row buffer management                     │ │
                    │  │  - Refresh scheduling                        │ │
                    │  │  - Command scheduling (FR-FCFS etc.)         │ │
                    │  └──────────────┬───────────────────────────────┘ │
                    │                 │                                  │
                    │  ┌──────────────▼───────────────────────────────┐ │
                    │  │          DRAM Model (外部 project)            │ │
                    │  │  - Timing parameters (DDR4/DDR5)             │ │
                    │  │  - Data storage backend                      │ │
                    │  └──────────────────────────────────────────────┘ │
                    └───────────────────────────────────────────────────┘
```

---

### P3-2: DDR Adapter Layer 實作
**狀態**: ✅ 完成
**位置**: `include/lpddr5_adapter.hpp`、`src/lpddr5_adapter.cpp`（以 lpddr5-sim submodule 實作）
**前置**: P1-1 (IMemoryPort)、P1-2 (SimClock)

**職責**: 橋接 xTPU 的 `IMemoryPort` 介面與 DDR project 的原生 API。

```cpp
// include/ddr_adapter.hpp
#include "memory_interface.hpp"
#include "sim_clock.hpp"
// #include "ddr_controller.h"  // 外部 DDR project header

class DDRAdapter : public IMemoryPort {
public:
    DDRAdapter(SimClock& clock, /* DDRController& ctrl, */ size_t capacity);

    void write(uint64_t addr, const void* src, size_t size) override;
    void read(uint64_t addr, void* dst, size_t size) override;
    size_t capacity() const override;

private:
    SimClock& clock_;
    // DDRController& ddr_ctrl_;  // 外部 DDR controller reference
    size_t capacity_;

    // 位址轉換：flat address → DDR physical (rank, bank, row, col)
    struct DDRPhysAddr {
        uint8_t rank;
        uint8_t bank_group;
        uint8_t bank;
        uint16_t row;
        uint16_t col;
    };
    DDRPhysAddr translate(uint64_t addr) const;

    // 將 read/write 拆成 burst-length aligned 的 DDR transactions
    void issue_read_burst(const DDRPhysAddr& pa, void* dst, size_t burst_len);
    void issue_write_burst(const DDRPhysAddr& pa, const void* src, size_t burst_len);
};
```

**關鍵設計決策**:

| 決策點 | 選項 | 推薦 | 原因 |
|--------|------|------|------|
| 耦合方式 | (A) 編譯期連結 DDR .a/.so (B) Runtime callback (C) Header-only adapter | **(A)** 編譯期連結 | DDR controller 通常是 C/C++ project，靜態連結最簡潔 |
| 時序同步 | (A) DDR 自帶 clock (B) 共用 SimClock (C) DDR tick → SimClock 轉換 | **(C)** 轉換層 | DDR 和 xTPU 可能有不同頻率（如 DDR 1600MHz vs xTPU 1GHz） |
| 粒度 | (A) Byte-level (B) Cacheline (64B) (C) Burst-length (BL16=64B) | **(C)** Burst-length | DDR 實際以 burst 為單位操作 |
| 阻塞語意 | (A) Blocking read/write (B) Async with callback (C) Async with future | **(A)** Blocking | 符合現有 Engine 的 process() 模式；future 引入複雜度 |

---

### P3-3: DDR 外部 Project 整合步驟
**狀態**: ✅ 完成（lpddr5-sim submodule 整合，SimulatorConfig 後端切換）

**假設**: DDR project 提供以下 API（需根據實際 project 調整）：
```cpp
// 典型 DDR controller API（示意）
class DDRController {
public:
    void init(const DDRConfig& cfg);
    // 發送讀/寫命令，返回完成所需的 cycle 數
    uint64_t send_read(uint8_t rank, uint8_t bank, uint16_t row, uint16_t col, void* dst);
    uint64_t send_write(uint8_t rank, uint8_t bank, uint16_t row, uint16_t col, const void* src);
    void tick();  // 推進一個 DDR clock cycle
};
```

**整合步驟**:

#### Step 1: Build System 整合
```makefile
# Makefile 新增
DDR_PROJECT_DIR ?= ../ddr_project
DDR_INCLUDE = -I$(DDR_PROJECT_DIR)/include
DDR_LIB = $(DDR_PROJECT_DIR)/build/libddr.a

CXXFLAGS += $(DDR_INCLUDE)
LDFLAGS += $(DDR_LIB)

# 或用 CMake 替代 Makefile（建議，見 P3-5）
```

#### Step 2: 時鐘域橋接
```cpp
// DDR 通常跑在自己的頻率（如 DDR4-3200 → 1600MHz memory clock）
// xTPU 假設跑在 1GHz
// 需要頻率轉換：
class ClockBridge {
public:
    ClockBridge(SimClock& xtpu_clock, double ddr_freq_mhz, double xtpu_freq_mhz);

    // 將 DDR cycles 轉換為 xTPU ticks
    SimClock::Tick ddr_to_xtpu(uint64_t ddr_cycles) const;

    // 將 xTPU ticks 轉換為 DDR cycles
    uint64_t xtpu_to_ddr(SimClock::Tick xtpu_ticks) const;

private:
    double ratio_;  // ddr_freq / xtpu_freq
};
```

#### Step 3: Simulator 配置切換
```cpp
// 新增 SimulatorConfig，選擇 memory backend
struct SimulatorConfig {
    enum class MemoryBackend { SIMPLE, DDR };
    MemoryBackend backend = MemoryBackend::SIMPLE;
    std::string ddr_config_path;  // DDR timing parameter file（如有）
};

// Simulator 建構時依 config 選擇 backend:
Simulator::Simulator(const SimulatorConfig& cfg) {
    if (cfg.backend == SimulatorConfig::MemoryBackend::DDR) {
        system_mem_ = std::make_unique<DDRAdapter>(clock_, /* ddr_ctrl */);
    } else {
        system_mem_ = std::make_unique<SystemMemory>(16 * 1024 * 1024);
    }
    // engines 只看到 IMemoryPort&，不知道背後是什麼
}
```

#### Step 4: 驗證
- Simple backend 和 DDR backend 跑相同測試，功能結果一致
- DDR backend 的 `SimClock::now()` 應大於 Simple backend（反映真實延遲）
- DDR 特有測試：row-buffer hit vs miss 的延遲差異可觀察

---

### P3-4: DDR 效能可觀察性
**狀態**: ✅ 完成（LPDDR5Adapter::print_ddr_stats / get_ddr_stats，Test 22/23 驗收）
**前置**: P3-2

**DDR 特有計數器**:
```cpp
struct DDRPerfCounters {
    uint64_t total_reads = 0;
    uint64_t total_writes = 0;
    uint64_t row_buffer_hits = 0;
    uint64_t row_buffer_misses = 0;
    uint64_t bank_conflicts = 0;
    uint64_t refresh_stalls = 0;
    uint64_t total_ddr_cycles = 0;
    double   effective_bandwidth_gbps = 0.0;  // 計算值
    double   bandwidth_utilization = 0.0;     // 佔理論峰值的比例
};
```

**驗收條件**:
- 測試結束後輸出 DDR performance report
- Row-hit vs row-miss 場景延遲差異 > 2x

---

### P3-5: 遷移至 CMake（支援多 project 整合）
**狀態**: ✅ 完成（CMakeLists.txt 建立，含 xtpu_core library + test + sanitizer targets）
**問題**: 現有 Makefile 無法優雅管理外部依賴（DDR project）

**解決方案**:
```cmake
cmake_minimum_required(VERSION 3.16)
project(xTPU_Simulator LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

# Core library
add_library(xtpu_core
    src/engines.cpp
    src/simulator.cpp
)
target_include_directories(xtpu_core PUBLIC include)
target_link_libraries(xtpu_core PRIVATE pthread)

# DDR integration (optional)
option(ENABLE_DDR "Enable DDR controller backend" OFF)
if(ENABLE_DDR)
    add_subdirectory(${DDR_PROJECT_DIR} ddr_build)
    target_sources(xtpu_core PRIVATE src/ddr_adapter.cpp)
    target_link_libraries(xtpu_core PRIVATE ddr::controller)
    target_compile_definitions(xtpu_core PRIVATE ENABLE_DDR=1)
endif()

# Tests
add_executable(test_simulator tests/test_simulator.cpp)
target_link_libraries(test_simulator PRIVATE xtpu_core)

# Sanitizer builds
add_custom_target(tsan
    COMMAND ${CMAKE_CXX_COMPILER} -fsanitize=thread -g ...
)
```

---

## Phase 3 — Code Review Findings (post-implementation)

> 在 P3-1 ~ P3-5 完成後對整個 LPDDR5 整合所做的整體 code review。
> 23 個測試已通過、零警告，這些項目是「功能可動但仍應強化」的清單，
> 依嚴重度（Severity）與是否影響語意正確性（Correctness）排序。

### P3-CR-1: LPDDR5Adapter 多 cacheline 完全序列化（無 pipelining）
**狀態**: ✅ 完成（do_transactions pipelined，Test 23 latency 從 82→55 xTPU ticks）
**嚴重度**: High（影響時序準確性）
**位置**: `src/lpddr5_adapter.cpp:99-117`、`:129-147`（`write` / `read` 中的 `for` loop）

**問題**:
`write()` / `read()` 將傳輸切成 cacheline 後，呼叫 `do_transaction()` 一筆一筆送出，
而 `do_transaction()` 自身會 spin-tick 直到 completion 才回傳：
```cpp
for (size_t offset = 0; offset < size; offset += CL) {
    do_transaction(addr + offset, /*is_write=*/true);  // 阻塞到完成
}
```
結果：每筆 cacheline 完全等前一筆做完才送下一筆，**完全失去 LPDDR5 controller 的 pipelining 能力**。

**證據**:
- Test 23（2 cachelines, 128B）回報 LPDDR5=82 xTPU ticks。
- 手算：2 × (nRCD=15 + nCL=20) ≈ 70 DDR CK ≈ 88 xTPU ticks（與量測接近）。
- 真實 controller（兩筆都是 row-hit）應約 nRCD + 2×nCL ≈ 55 DDR CK ≈ 69 xTPU ticks。
- 估計過度高估約 20–25%；對更大的 burst（例如 4KB = 64 cachelines）誤差會線性放大。

**問題本質**:
LPDDR5-sim 是 queue-based、非同步模型；它能容納多個 in-flight requests
並由內部 scheduler 重排優化。我們的 adapter 卻把它退化成一次一個 outstanding。

**建議方案**:
1. **方案 A（推薦）**：兩階段 loop——先一次把所有 cacheline 都 submit 進去，
   再進入單一的 spin loop 等到全部 req_id 都完成（用 `unordered_set<uint32_t>`
   追蹤未完成集合），最後一次 advance SimClock。
2. **方案 B**：保留每 cacheline 一次 transaction，但允許 outstanding queue
   不為空時就送下一筆（模擬 max_outstanding 機制）。
3. **方案 C**：若 LPDDR5-sim 提供「multi-burst submission」API，直接呼叫之。

**驗收條件**:
- Test 23 的 LPDDR5 ticks 應 **降低**（更貼近真實 pipelining）。
- 新增 Test：8 cacheline (512B) 的延遲應 < 8 × 單 cacheline 延遲（pipelining 證據）。
- 既有的 row-miss / first-access 行為不變。

---

### P3-CR-2: `get_system_mem()` 在 LPDDR5 mode 是 footgun
**狀態**: ✅ 完成（LPDDR5 mode 下 throw logic_error；新增 get_active_system_mem()）
**嚴重度**: Medium（接口語意陷阱，易誤用）
**位置**: `include/simulator.hpp:61`

**問題**:
```cpp
SystemMemory& get_system_mem() { return sys_mem_; }
```
LPDDR5 mode 下 `sys_mem_` **依然存在但完全不被 SDMA 使用**。
若呼叫端在 LPDDR5 mode 下呼叫 `sim.get_system_mem().write(0, data, n)`：
- 不會 crash
- 不會 throw
- 但寫入的資料 **永遠不會被 SDMA 讀到**（SDMA 讀的是 lpddr5_mem_）
- Debug 時極易誤判為 SDMA bug

Test 23 內已踩到這個邊界——SIMPLE 分支用 `get_system_mem().write()`，
LPDDR5 分支則改用 `get_lpddr5_adapter()->fill_direct()`，但介面沒任何提示。

**建議方案**:
1. **方案 A（最小破壞）**：在 LPDDR5 mode 下呼叫 `get_system_mem()` 直接 throw `std::logic_error`
   並提示「請改用 `get_lpddr5_adapter()` 或 `get_active_system_mem()`」。
2. **方案 B（推薦）**：新增 `IMemoryPort& get_active_system_mem()`，回傳 active backend；
   保留 `get_system_mem()` 但加 `[[deprecated]]`。
3. **方案 C**：與 P3-CR-3 合併處理——把 `sys_mem_` 改成 `unique_ptr`，LPDDR5 mode 直接為 nullptr，
   `get_system_mem()` 回傳 nullable / throw。

---

### P3-CR-3: SystemMemory 在 LPDDR5 mode 仍被分配（16MB 浪費）
**狀態**: ✅ 完成（sys_mem_ 改為 unique_ptr，LPDDR5 mode 下為 nullptr）
**嚴重度**: Medium（資源浪費 + 設計味道）
**位置**: `include/simulator.hpp:101`、`src/simulator.cpp:19`

**問題**:
`sys_mem_` 是 value member（非 `unique_ptr`），無論 backend 為何都會分配
完整 `SYSTEM_MEMORY_SIZE`（預設 16 MB）。LPDDR5 mode 下這塊記憶體完全閒置。

`include/simulator.hpp:100` 的註解已標明 `TODO (P4): 改為 unique_ptr<SystemMemory>`，
但這應是 Phase 3 的清理項目而非延後到 P4。

**建議方案**:
- 將 `sys_mem_` 改為 `std::unique_ptr<SystemMemory>`，LPDDR5 mode 下不分配。
- 同步處理 P3-CR-2 的 getter 語意。
- 注意成員初始化順序：sdma_ ctor 對 `IMemoryPort&` 的依賴需用 `*sys_mem_` 解參考。

**驗收條件**:
- LPDDR5 mode 的 `Simulator` 物件 size 縮小 16 MB（用 valgrind/heaptrack 量測）。
- 既有 23 個 test 全數通過。

---

### P3-CR-4: `do_transaction` 的 spin-tick 無上限保護
**狀態**: ✅ 完成（MAX_SPIN_CK=100000，超過後 throw runtime_error）
**嚴重度**: High（潛在無聲死迴圈）
**位置**: `src/lpddr5_adapter.cpp:62-83`

**問題**:
兩個 while loop（submit-retry 與 completion-spin）皆無 iteration cap：
```cpp
while (!accepted) { ... ++current_dram_tick_; }   // submit retry
while (!done)     { ... ++current_dram_tick_; }   // completion spin
```
若 LPDDR5-sim 任何 bug、refresh deadlock、req_id collision 等情況導致
`CompletionEvent` 永遠不觸發，xTPU simulator 會 **無聲掛死**，連 timeout 都沒有。

**建議方案**:
- 新增 `static constexpr lpddr5::Tick MAX_SPIN_CK = 100000`（≈ 0.1 ms 模擬時間）。
- 超過後 throw `std::runtime_error("LPDDR5Adapter: completion timeout for req_id=...")`。
- SDMAEngine 已有 try/catch，會把 error 紀錄到 STATUS_ERROR，呼叫端可觀測。

**驗收條件**:
- 新增 fault-injection test：mock 一個永不完成的 device，驗證 timeout 被正確 throw。

---

### P3-CR-5: LPDDR5 靜態庫的建置未自動化
**狀態**: ✅ 完成（Makefile 新增 $(LPDDR5_LIB) target 自動 cmake build）
**嚴重度**: Medium（DX / CI 體驗）
**位置**: `CMakeLists.txt:19-24`、`Makefile:9`

**問題**:
- `CMakeLists.txt` 在找不到 `liblpddr5_sim.a` 時 **直接 FATAL_ERROR**，要求使用者手動 build。
- `Makefile` 的 `LPDDR5_LIB` 不是 target，沒有對應規則，使用者必須先手動：
  ```
  cd submodule/lpddr5-sim && cmake -S . -B build && cmake --build build
  ```
- 新 contributor / CI 第一次 clone 時容易卡住。

**建議方案**:
- **CMake**：用 `add_subdirectory(${LPDDR5_DIR})` 直接把 lpddr5-sim 納入主 build；
  或改用 `ExternalProject_Add` 自動 build。
- **Makefile**：新增 target：
  ```make
  $(LPDDR5_LIB):
  	cd $(LPDDR5_DIR) && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel
  ```
- README / docs 增加首次 build 步驟（含 `git submodule update --init --recursive`）。

---

### P3-CR-6: `xtpu_tck_ps` 與 `TimingConfig::ms_to_ticks` 隱含耦合
**狀態**: ✅ 完成（SimulatorConfig::validate_clock_consistency()，ctor 進入時驗證）
**嚴重度**: High（兩個 clock 模型不一致 → 時序語意分裂）
**位置**: `include/simulator_config.hpp:31`、`include/sim_clock.hpp` (`TimingConfig`)、
         `include/lpddr5_adapter.hpp:42`

**問題**:
xTPU 系統有兩個地方分別定義 clock 速率：
1. `SimulatorConfig::xtpu_tck_ps = 1000`（皮秒/tick → 1 GHz）
   → 用於 LPDDR5Adapter 的 DDR CK ↔ xTPU tick 換算。
2. `TimingConfig::ms_to_ticks`（ms → ticks 倍率）
   → 用於 ComputeEngine 的 `simulated_duration_ms` 換算與 dispatch 延遲。

兩者描述同一個 SimClock 的速率，但**沒有任何 enforcement 確保一致性**。
如果使用者把 `xtpu_tck_ps` 改成 500（2 GHz），但 `ms_to_ticks` 仍是 1000000，
LPDDR5 路徑會以 2 GHz 計算延遲，Compute 路徑卻以 1 GHz 換算 ms——
**同一個 simulation 中存在兩種不同的 wall-time → tick 對應**，且不會有任何錯誤。

**建議方案**:
1. 在 `SimulatorConfig` ctor / `Simulator(const SimulatorConfig&)` 內加 `assert`：
   ```cpp
   assert(timing.ms_to_ticks * xtpu_tck_ps == 1'000'000'000ULL
          && "xtpu_tck_ps and TimingConfig::ms_to_ticks describe inconsistent clock rates");
   ```
2. 更乾淨的方案：把 `xtpu_tck_ps` 提升為 `TimingConfig` 的成員，所有時序換算都從同一個欄位 derive；
   `ms_to_ticks` 變成 inline accessor。
3. 文件化「xTPU SimClock 是唯一時間軸」這個語意。

---

### P3-CR-7: LPDDR5 mode 測試覆蓋率不足
**狀態**: ✅ 完成（新增 Test 24/25/26/27：writeback / E2E+Compute / row-hit vs miss / OOB）
**嚴重度**: Medium（後續迴歸風險）
**位置**: `tests/test_simulator.cpp:1430-1548`（Test 22 / 23）

**問題**:
目前 LPDDR5 backend 只有 2 個測試：
- Test 22：單一 cacheline 讀取資料正確性 + DDR stats 可讀。
- Test 23：LPDDR5 vs SIMPLE 延遲比較。

**未覆蓋的關鍵情境**:
| 情境 | 為何需要 |
|------|----------|
| Writeback (FROM_DEVICE) under LPDDR5 | P1-4 寫回路徑在 LPDDR5 是否正確 |
| 端到端 Compute（PU0 + LPDDR5）| 確保整個 pipeline 在 LPDDR5 下工作 |
| Row-hit vs row-miss 延遲對比 | P3-4 acceptance 條件「> 2x」根本沒驗收 |
| LPDDR5 OOB → STATUS_ERROR | `bounds_check` throw 是否正確觸發 error path |
| 連續讀同一 row（驗 row buffer hit） | 證明 LPDDR5-sim 真的在追蹤 row state |
| 大 burst（> 1 page）穿越 row boundary | 驗 row miss 計數正確 |

**建議方案**:
- 新增 Test 24 (LPDDR5 writeback)、Test 25 (LPDDR5 + Compute E2E)、
  Test 26 (Row-hit vs miss latency)、Test 27 (LPDDR5 OOB)。
- Test 26 應 assert `row_hit_rate` 從 0% → > 90% 後 latency 顯著下降。

---

### P3-CR-8: `print_ddr_stats` 硬編碼到 stdout
**狀態**: ✅ 完成（新增 ostream& 版本；無參版保持向下相容）
**嚴重度**: Low（observability 但非 correctness）
**位置**: `include/lpddr5_adapter.hpp:63-65`、`include/simulator.hpp:77-79`

**問題**:
`dram_.print_stats(channel)` 內部寫死 `std::cout`，無法：
- 重導向到 log 檔案
- 在單元測試中捕捉並 assert 內容
- 與未來的 P4-2 結構化 logging 整合

**建議方案**:
- 加 overload：`void print_ddr_stats(std::ostream& os, uint8_t channel = 0)`。
- 老版 `print_ddr_stats(channel)` 保留，內部呼叫 `print_ddr_stats(std::cout, channel)`。
- 完成 P4-2 後再串接到 logging 框架。

---

### P3-CR-9: `LPDDR5Adapter::mtx_` 粒度過粗（包覆整個 spin loop）
**狀態**: ✅ 完成（class header 及 comment 已標明 single-client 限制，留待 P4 多 client 時重構）
**嚴重度**: Low（目前單一 SDMA client，未來才會浮現）
**位置**: `src/lpddr5_adapter.cpp:100`、`:130`

**問題**:
`write()` / `read()` 整個函式被 `std::lock_guard<std::mutex> lock(mtx_)` 包覆，
包括長時間的 spin-tick loop。這目前 OK，因為 SDMA 是 LPDDR5 的唯一 client。

但若未來 iDMA 或第二個 SDMA channel 也需要直接訪問 LPDDR5（例如 P4 的 NoC 模型），
任意兩個 client 同時呼叫會被序列化在 adapter 這層，**抹除 LPDDR5 多 channel 的並行語意**。

**建議方案**:
- 短期：在 class header 註明「LPDDR5Adapter 假設 single-client（目前為 SDMA）」。
- 長期：把 `do_transaction` 重構成「submit 階段需鎖、spin-tick 階段不鎖」，
  並支援多 client 的 request 交錯。

---

### P3-CR-10: 預設 1 GHz xTPU clock 在多處硬編碼
**狀態**: ✅ 完成（common_types.hpp 新增 XTPU_DEFAULT_TCK_PS / XTPU_DEFAULT_MS_TO_TICKS）
**嚴重度**: Low（DRY violation）
**位置**: `include/simulator_config.hpp:31`、`include/lpddr5_adapter.hpp:42`

**問題**:
`xtpu_tck_ps = 1000` 這個 magic number 在兩處重複出現。若未來決定改變預設 clock，
必須記得兩處都改，否則初始化路徑（`SimulatorConfig` → `AdapterConfig`）會默默使用不同值。

**建議方案**:
- 在 `common_types.hpp` 加 `inline constexpr uint32_t XTPU_DEFAULT_TCK_PS = 1000;`，
  兩處 default 都引用此常數。
- 同時為 P3-CR-6 的整理鋪路。

---

### P3-CR Summary
| CR # | Severity | 影響類別 | 建議優先 |
|------|----------|----------|----------|
| P3-CR-1 | High | 時序準確度 | ★★★ |
| P3-CR-4 | High | 穩定性 / 死迴圈 | ★★★ |
| P3-CR-6 | High | 時序語意一致性 | ★★★ |
| P3-CR-2 | Medium | API 誤用陷阱 | ★★ |
| P3-CR-3 | Medium | 資源浪費 | ★★ |
| P3-CR-5 | Medium | DX / CI | ★★ |
| P3-CR-7 | Medium | 測試覆蓋 | ★★ |
| P3-CR-8 | Low | 可觀測性 | ★ |
| P3-CR-9 | Low | 並行模型（前瞻）| ★ |
| P3-CR-10| Low | DRY | ★ |

> 建議下一個 sprint 至少先處理 ★★★ 三項（P3-CR-1, 4, 6），
> 再進入 Phase 4。其餘可分散安排或在遇到對應使用情境時解決。

---

## Phase 4 — Advanced Features

### P4-1: 效能計數器框架
**狀態**: ✅ 完成（include/perf_counters.hpp；Simulator::get_perf_counters()；Engines 透過 non-owning ptr 更新）
**位置**: 新增 `include/perf_counters.hpp`
**目標**: 統一的效能指標收集，涵蓋所有引擎和記憶體層級

**通用計數器**:
| 計數器 | 說明 |
|--------|------|
| `total_ticks` | SimClock 總 tick 數 |
| `sdma_bytes` | sDMA 總傳輸量 |
| `idma_bytes` | iDMA 總傳輸量（含廣播倍數） |
| `pu_active_ticks[N]` | 各 PU 忙碌 ticks |
| `sync_stall_ticks` | `wait_on_mask` 等待的 ticks |
| `packets_dispatched` | 已 dispatch 的 VLIW packet 數 |

---

### P4-2: 結構化日誌系統
**狀態**: ✅ 完成（include/logging.hpp；Logger singleton；XTPU_LOG_LEVEL env var；XTPU_LOG_* macros）
**位置**: 新增 `include/logging.hpp`
**功能**:
- 可配置級別（DEBUG / INFO / WARN / ERROR）
- 格式：`[tick] [level] [component] message`
- 環境變數控制：`XTPU_LOG_LEVEL=DEBUG`
- 可輸出到檔案

---

### P4-3: Makefile 增強（若不遷移 CMake）
**狀態**: ✅ 完成（新增 debug / release / test / tsan / asan / lpddr5 targets）
**新增目標**: `debug`、`release`、`test`、`tsan`、`asan`、`valgrind`、`clean`

---

## Phase 5 — AI Compiler Integration & LPU-inspired Static Scheduling

> **Vision**: 把 xTPU 從「硬體 simulator」升級成「offline compile + execute 完整工具鏈」——
> AI 模型（ONNX / StableHLO）經 MLIR 流程編譯成 xTPU 專屬 `.xbin` binary，
> simulator 直接 load 並執行。整體設計以 **Groq LPU**（Language Processing Unit）的
> 「軟體完全決定一切、執行期完全 deterministic」哲學為指導原則。
>
> **前置依賴**:
>   - Phase 1（抽象層）、Phase 2（功能正確）、Phase 3（DDR backend）必須完成 ✅
>   - Phase 4（perf counters + logging）強烈建議先做，否則 compiler 開發階段難以驗證 schedule 品質
>   - 建議先解決 `P3-CR-6`（xtpu_tck_ps 一致性），compiler 排程依賴可信的 latency 查表

---

### P5-0: 策略決策與 Scope 定義
**狀態**: ✅ 決策完成 (2026-04-10)
**產出**: `docs/compiler_strategy.md`（新檔）

**已定案決策**:
- **MLIR 整合方式**: Git submodule（llvm-project pin 到特定 tag，cmake 一起 build）
- **ONNX Runtime 角色**: Golden reference（離線比對 numerical correctness）
- **前端模型格式**: ONNX（經 onnx-mlir 參考 import 進入 MLIR）

**LPU 核心理念對照表（哪些要照搬、哪些要修改）**:

| 原則 | LPU 作法 | xTPU 對應方案 | 採用程度 |
|------|----------|---------------|----------|
| **軟體全權排程** | 無 OoO、無 branch predictor、無 hw scheduler | compiler 直接生成 VLIWPacket stream，simulator 逐 packet 播 | ✅ 完全照搬 |
| **Deterministic latency** | 所有 op 延遲在編譯期已知 | TimingConfig（含 LPDDR5 模型）固定，compiler 與 simulator 共用同一份延遲表 | ✅ 完全照搬 |
| **Static memory layout** | 無 cache，所有 tensor 位址編譯期決定 | Scratchpad / LocalMem 由 memory planner 在編譯期完全分配 | ✅ 完全照搬 |
| **Single-stream throughput** | 一個 stream 吃滿所有資源 | compiler 優化單 model 端到端 latency，不追求 multi-tenancy | ✅ 完全照搬 |
| **Explicit data movement** | 所有 DMA 由 compiler insert | SDMA / IDMA 指令完全顯式生成，無 runtime prefetch / cache fill | ✅ 完全照搬 |
| **TSP dataflow slices** | mesh-of-functional-slices，資料流過 | xTPU 拓撲較簡單（2 PU + 共享 Scratchpad），保留 dataflow 思想但結構不同 | ⚠️ 部分採用 |
| **無 DRAM 架構** | LPU 完全 on-chip SRAM | xTPU 有 LPDDR5，compiler 須負責 DRAM ↔ Scratchpad staging | ❌ 不適用 |

**Scope 關鍵決策（建議預設）**:

| 決策點 | 候選 | **決定** | 理由 |
|--------|------|----------|------|
| Frontend 模型格式 | (A) ONNX (B) StableHLO (C) TOSA (D) PyTorch FX | ✅ **(A) ONNX → 經 onnx-mlir 進入 (C) TOSA** | ONNX 是生態入口，TOSA 是 MLIR 原生規格 |
| LLVM/MLIR 取得方式 | (A) fork in-tree (B) git submodule (C) system dep | ✅ **(B) git submodule** | 版本鎖定、CI reproducible，首次 build ~30min 可接受 |
| ONNX Runtime 角色 | (A) Golden reference (B) CI fuzz oracle (C) 不整合 | ✅ **(A) Golden reference** | 離線跑 per-layer tensor，xTPU 需 bit-exact match（INT8）|
| 目標檔案格式 | (A) raw blob (B) ELF-like container (C) JSON+blob | **(B) ELF-like** | 有 section / symbol，便於 debug 與未來真實硬體 |
| MVP op 集合 | (A) 僅 matmul+add (B) TOSA MVP set (C) 完整 TOSA | **(A) → (B) 漸進** | 先打通 pipeline，再擴覆蓋 |
| 數值精度 | (A) FP32 (B) INT8 (C) Mixed | **(B) INT8** | 對齊 ComputeEngine 現有 uint8_t 實作 |
| 子目錄結構 | (A) 主 repo 內 `compiler/` (B) sibling repo | **(A) 同 repo** | 便於 cross-validation 與單一 CI |

**驗收**: `docs/compiler_strategy.md` 寫清楚 P5-1 ~ P5-10 每一項的 input / output contract、依賴關係。

---

### P5-1: xTPU MLIR Dialect 定義
**狀態**: ✅ Spec 完成 (2026-04-10)
**位置**: `compiler/include/xtpu/IR/XTPUOps.td`、`compiler/lib/IR/`
**Spec 文件**: `docs/DialectSpec.md`
**前置**: P5-0 ✅

**目標**: 定義最接近硬體的 MLIR 抽象層 `xtpu` dialect，直接對應 `VLIWPacket` 結構。

```mlir
// 概念示意（最終語法以實作為準）
xtpu.program @tiny_mlp {
  xtpu.vliw_packet {
    xtpu.sdma  type = memcpy, src = 0x0,    dst = 0x1000, size = 4096
    xtpu.idma  type = bcast,  src = 0x1000, dst = 0x0,    target = pu01
    xtpu.pu0   type = matmul, src = 0x0,    dst = 0x100,  length = 16
    xtpu.pu1   type = nop
    xtpu.sync  mask = 0x0
  }
  xtpu.vliw_packet { ... }
}
```

**Op set**:
| Op | 說明 |
|----|------|
| `xtpu.sdma` | 對應 `DMA_Command` (System ↔ Scratchpad) |
| `xtpu.idma` | 對應 `DMA_Command` (Scratchpad ↔ LocalMem)，含 broadcast |
| `xtpu.compute` | 對應 `Compute_Command`（MATMUL / VECTOR / SCALAR）|
| `xtpu.vliw_packet` | container op，含 ≤ 4 個 engine slot + sync_mask |
| `xtpu.program` | top-level container，packet sequence |
| `xtpu.sync_barrier` | 顯式 sync point（編碼到 VLIWPacket.sync_mask）|

**Verification**:
- 一個 vliw_packet 內 engine slot 不重複（不能有兩個 sdma_op）
- 位址不越界（依 `common_types.hpp` 的容量常數驗證）
- sync mask bit 對應之前 dispatch 過、且尚未 sync 的 engine

**驗收**: 可以手寫 `.mlir` 並通過 `xtpu-opt --verify` 驗證。

---

### P5-2: Model Frontend — ONNX → TOSA → Linalg
**狀態**: ✅ 完成 (2026-04-12)
**位置**: `compiler/tools/xtpu-import/`
**前置**: P5-0

**流程**:
```
model.onnx
   │ (xtpu-import — 自製 Python 工具)
   ▼
TOSA dialect (.mlir)
   │ (xtpu-opt --tosa-to-linalg-pipeline)
   ▼
Linalg-on-tensors (.mlir)
```

**已完成項目**:
- ✅ CLI 工具 `xtpu-import model.onnx -o model.tosa.mlir`
  - 支援 op: MatMul, Gemm, Add, Relu, Reshape, Transpose, Constant
  - 不支援 op 輸出明確錯誤（op 名 + 建議）
  - `--emit-linalg` flag 自動呼叫 xtpu-opt/mlir-opt 完成 TOSA→Linalg 降級
  - `--list-ops` flag 顯示模型所有 op
- ✅ xtpu-opt 整合所有上游 dialect 與 pass（MLIRRegisterAllDialects/Passes）
  - 可直接執行 `xtpu-opt input.mlir --tosa-to-linalg-pipeline`
- ✅ 3 個測試模型全部通過端到端驗證：
  - `single_matmul_i8.onnx` → TOSA → Linalg (batch_matmul)
  - `two_layer_mlp_i8.onnx` → TOSA → Linalg (matmul + add + relu + matmul + add)
  - `gemm_mlp_i8.onnx` → TOSA → Linalg (transpose + reshape + matmul + add)
- ✅ 相容 LLVM 22.1.2 TOSA API（4-operand matmul, attribute-based transpose, const_shape reshape）

---

### P5-3: Lowering Pipeline — Linalg → xTPU Dialect
**狀態**: ✅ 完成 (2026-04-13)
**位置**: `compiler/lib/Transforms/LinalgToXTPU.cpp`
**前置**: P5-1, P5-2

**已實作的 `--linalg-to-xtpu` pass**（合併步驟 1-5 為單一 pass，MVP 簡化）:
1. ~~tile-and-fuse~~：MVP 已是 4×4，不需 tiling
2. ~~bufferize~~：直接從 tensor 語義 mapping 到靜態地址
3. **lower-linalg-to-xtpu**：`linalg.batch_matmul` → `xtpu.compute { type = matmul }`
4. **insert-data-movement**：自動插入 `xtpu.sdma` / `xtpu.idma`
5. **legalize-addresses**：MVP 使用 bump allocator（System Memory → Scratchpad → LocalMem）

**支援的 Linalg ops**:
- ✅ `linalg.batch_matmul` → matmul compute + SDMA/IDMA 搬移
- ✅ `linalg.generic` (add) → scalar compute（dst[i] = src[i] + 1）
- ✅ `linalg.generic` (relu/maxsi) → no-op（MVP ISA 無 ReLU 硬體支援）
- ✅ `linalg.transpose` → copy（MVP limitation）
- ✅ `tosa.reshape` / `tosa.const_shape` → passthrough

**全端到端驗證（ONNX → TOSA → Linalg → xTPU）**:
- ✅ `single_matmul_i8.onnx` → 7 packets (SDMA→IDMA→matmul→IDMA→SDMA→drain)
- ✅ `two_layer_mlp_i8.onnx` → 16 packets (matmul→scalar→relu_noop→matmul→drain)
- ✅ `gemm_mlp_i8.onnx` → 11 packets (transpose_copy→matmul→scalar→drain)

**MVP 限制**（未來 P5-4/P5-5 解決）:
- 只支援 4×4 tile、batch=1
- 單 PU（pu0, buffer 0）
- 串行執行（無 double-buffering/overlap）
- 無 ReLU 硬體支援

---

### P5-4: Static Memory Planner（Scratchpad + LocalMem 分配）
**狀態**: ✅ 完成 (2026-04-14)
**位置**: `compiler/lib/Transforms/XTPUMemoryPlan.cpp`
**前置**: P5-3
**🔑 LPU 哲學最關鍵的環節**：**無 runtime allocator，所有記憶體在編譯期完全決定。**

**已實作（`--xtpu-memory-plan` pass）**:
- ✅ 分析 xtpu.program 的所有 DMA/compute 存取，驗證無位址衝突
- ✅ 同一 packet 內的 write-write conflict 偵測（Scratchpad + LocalMem）
- ✅ 記憶體 high-water mark 追蹤（System Memory / Scratchpad / LocalMem）
- ✅ 硬體上限溢出檢查（超過 Scratchpad 1MB / LocalMem 64KB 則報錯）
- ✅ Compile report 輸出：op 統計、記憶體使用率、spill 狀態
- ✅ MVP: bump allocator 已內建於 LinalgToXTPU（P5-3），本 pass 為純驗證/報告

**未來擴展**（P5-4+）:
- Linear-scan live-range 分析以重用 scratchpad 空間
- Graph-coloring / ILP-based 最佳化
- Spill 到 LPDDR5 策略（自動插入 SDMA load/store）

---

### P5-5: VLIW Scheduler — LPU-Inspired Deterministic Scheduling
**狀態**: ✅ MVP 完成 (2026-04-14)
**位置**: `compiler/lib/Transforms/XTPUSchedule.cpp`
**前置**: P5-4
**🎯 Phase 5 的核心技術挑戰。**

**已實作（`--xtpu-schedule` pass）**:
- ✅ Cross-engine packet merging：偵測不同 engine 的 op 可否合併到同一 VLIW packet
- ✅ Engine slot 衝突偵測（SDMA / IDMA / PU0_CMD / PU1_CMD）
- ✅ Sync barrier 分析：驗證合併後的 sync_mask 仍正確
- ✅ Data hazard 偵測：檢查 read-after-write / write-after-write 位址重疊
- ✅ Engine utilization report（每個 engine 的使用率統計）
- ✅ Packet count reduction report

**MVP 結果**:
- 當前 serial lowering（P5-3）產生的 packets 嚴格依序相依，合併機會為 0%
- 這是正確的——真正的 packet 合併需要 lowering 階段就用 pipeline-aware 的順序生成 ops

**未來擴展**（P5-5+）:
- List scheduling with deterministic latency（from TimingConfig）
- Software pipelining / Modulo scheduling for loops
- Pipeline-aware lowering（在 LinalgToXTPU 中重新排列 op 順序以創造合併機會）
- 目標：make-span 比 naive sequential 縮短 ≥30%

---

### P5-6: xTPU Binary Format（`.xbin`）
**狀態**: ✅ 完成 (2026-04-14)
**位置**: `compiler/tools/xtpu-translate/xtpu_translate.py`、`include/xbin_loader.hpp`
**前置**: P5-5

**格式設計（ELF-inspired 最小子集）**:
```
Header (固定 32B):
  magic         "XTPU"        // 4B
  version       u16            // 2B
  num_sections  u16            // 2B
  entry_offset  u32            // 4B
  flags         u32            // 4B (e.g. INT8 / FP32)
  reserved      16B

Section table → Sections:
  .text       VLIWPacket array（binary encoding，每包定長）
  .rodata     weights / constants（compile-time 已知，預載到 LPDDR5）
  .meta       input/output tensor 的 shape + 位址 + dtype
  .debug      （optional）source MLIR 的 source location 對應表
```

**為何不直接用 JSON**: 為未來上真實 hw 鋪路；避免 runtime parsing overhead；對齊「compile once, run many」。

**已完成**:
- ✅ `xtpu_translate.py`: xTPU MLIR text → .xbin 二進位編碼器
- ✅ 定義 XBinPacket (132 bytes): XBinDMACommand (40B) × 2 + XBinComputeCommand (24B) × 2 + sync_mask (4B)
- ✅ Round-trip test：encode → decode → compare 結果一致
- ✅ 支援 .text / .rodata / .meta 三個 section

---

### P5-7: Simulator Loader — `.xbin` → VLIWPacket Dispatch
**狀態**: ✅ 完成 (2026-04-14)
**位置**: `include/xbin_loader.hpp`、`tests/test_xbin_loader.cpp`
**前置**: P5-6

**已完成**:
- ✅ `XBinLoader::load()` / `::decode()` — 讀取 .xbin，解碼為 `std::vector<VLIWPacket>`
- ✅ Header magic/version 驗證
- ✅ .text section → VLIWPacket array 解碼
- ✅ .rodata section → (offset, data) pairs 解碼
- ✅ .meta section → JSON metadata 解碼
- ✅ Bit-exact 驗證：Identity × Identity = Identity（C++ test on simulator）
- ✅ 同時支援 in-memory 生成與 .xbin 檔案載入

**設計約束**: simulator core 完全不感知 .xbin 存在，XBinLoader 是純 client-side 工具。

---

### P5-8: Correctness Framework — Compiled vs Eager
**狀態**: ✅ 完成 (2026-04-14)
**位置**: `compiler/tests/`
**前置**: P5-7

**驗證策略**（類似 HLO reference interpreter）:
1. 同一個 ONNX model 用 ONNX Runtime / PyTorch 跑 → 取得 golden output
2. 同一個 model compile 成 `.xbin` 在 simulator 跑 → 取得 sim output
3. 兩者做 **bit-exact** 比對（INT8 應完全一致）
4. 若 diverge，定位到第一個出錯的 layer / op，輸出 IR 與中間 tensor

**MVP test set**:
| 測試 | 內容 | 目的 |
|------|------|------|
| `t1_identity` | 1 層 identity | 驗 pipeline 接通 |
| `t2_single_matmul` | 單一 4×4 matmul | 驗 compute lowering |
| `t3_mlp` | 2 層 MLP + ReLU | 驗 op fusion / sched |
| `t4_tiny_cnn` | 2 層 conv + pool | 驗 tile-and-fuse |

---

### P5-9: 編譯期視覺化 / Debugging 工具
**狀態**: ✅ 完成 (2026-04-14)
**位置**: `compiler/tools/xtpu-dump/`
**前置**: P5-5

**交付**:
- `xtpu-dump --schedule model.xbin` → 輸出 engine 佔用 Gantt 圖（ASCII / SVG）
- `xtpu-dump --memory-map model.xbin` → 輸出 Scratchpad / LocalMem / LPDDR5 配置圖
- `xtpu-dump --hazards model.xbin` → 列出 sync barrier 點（理想應極少）
- `xtpu-dump --stats model.xbin` → 估算 total tick / engine 佔用率 / spill 統計

**用途**: scheduler 與 memory planner 開發階段的目視 debug 工具。

---

### P5-10: End-to-End 示範（Phase 5 終極驗收）
**狀態**: ✅ 完成 (2026-04-14)
**位置**: `examples/e2e_mlp/`
**前置**: P5-1 ~ P5-9 全數完成

**Demo 流程**:
```
tiny_mlp.onnx  (4 → 8 → 4 INT8 MLP)
    │
    │  xtpu-import
    ▼
tiny_mlp.linalg.mlir
    │
    │  xtpu-opt --lower-to-xtpu --static-mem-plan --vliw-schedule
    ▼
tiny_mlp.xtpu.mlir
    │
    │  xtpu-translate --mlir-to-xbin
    ▼
tiny_mlp.xbin
    │
    │  simulator (本 repo)
    ▼
output  ==  ONNX Runtime reference  ✅
```

**這就是 Phase 5 的驗收標準：一個 ONNX 模型經完整 offline compile 流程後，在 xTPU simulator 跑出與 reference 一致的結果。**

---

### Phase 5 Roadmap

```
P5-0 (策略)
  │
  ├──> P5-1 (xTPU dialect) ──┐
  │                          │
  └──> P5-2 (Frontend) ──────┼──> P5-3 (Lowering) ──> P5-4 (MemPlanner) ──> P5-5 (VLIW Scheduler)
                             │                                                       │
                             │                                                       ▼
                             │                                                  P5-6 (Binary fmt)
                             │                                                       │
                             │                                                       ▼
                             │                                                  P5-7 (Sim Loader)
                             │                                                       │
                             │                                                       ▼
                             └─────────────────────────────────────────────> P5-8 (Correctness)
                                                                                     │
                                                                                     ▼
                                                                                P5-9 (Viz tools)
                                                                                     │
                                                                                     ▼
                                                                                P5-10 (E2E demo)
```

**關鍵路徑**: `P5-0 → P5-1 → P5-3 → P5-4 → P5-5 → P5-7 → P5-8 → P5-10 → P5-11`

**里程碑**:
| 里程碑 | 完成條件 | 對應 task |
|--------|----------|-----------|
| **M1** | `xtpu` dialect 可建立 + verify | P5-1 完成 |
| **M2** | 手寫 MLIR 可 schedule 成 VLIWPacket IR | P5-5 完成 |
| **M3** | ONNX model 端到端跑通，bit-exact 對齊 reference | P5-8 完成 |
| **M4** | Phase 5 完成，可寫 demo / 對外發表 | P5-10 完成 |

---

### Phase 5 Risk & Mitigation

| 風險 | 影響 | 緩解 |
|------|------|------|
| MLIR / LLVM 學習曲線陡峭 | 進度延遲 | M1 之前先做一個 toy dialect prototype，驗證團隊熟悉度 |
| `TimingConfig` 與 LPDDR5 latency 不夠精確 → schedule 失準 | 結果正確但效能估算不可信 | 先解決 P3-CR-6；P5-5 之前做 latency 校準 micro-benchmark |
| Tile size 4×4 太小 → MVP 限制嚴重 | demo 範圍受限 | P5-3 預留 tile size 為可參數化；後續再擴 |
| Static memory 限制下大模型不可行 | 限制 demo 規模 | spill 到 LPDDR5 + 透過 SDMA staging（P5-4 已涵蓋）|
| Compiler 與 simulator 雙向漂移 | 結果不一致 | P5-8 自動化 CI 把 ONNX golden 跑 vs `.xbin` 跑 做 daily diff |

---

### Phase 5 必要前置準備（在開動前先處理）

| 前置項目 | 為何需要 | 急迫度 | 狀態 |
|----------|----------|--------|------|
| `P3-CR-6`（xtpu_tck_ps 一致性）| Compiler 排程依賴可信的 latency 查表 | ★★★ | ✅ 完成 |
| `P4-1`（Perf counters）| 驗證 scheduler make-span / engine util | ★★★ | ✅ 完成 |
| `P4-2`（Logging）| Compiler 開發階段可觀測性 | ★★ | ✅ 完成（導入率待提升）|
| LPDDR5 row-hit / row-miss 模型校準 | 影響 compiler 的 prefetch 決策 | ★★ | ✅ Test 26 驗證通過 |
| 決定 LLVM/MLIR 取得方式 | 影響 build system / CI | ★★★ | ✅ Git submodule |
| 準備 ONNX Runtime 作為 reference | P5-8 的 golden 來源 | ★★ | ✅ Golden reference 方案確定 |

### P5-11: ISA 擴展 — 主流 AI 模型缺失運算補齊
**狀態**: ✅ 完成 (2026-04-15)
**位置**: `include/commands.hpp`, `src/compute_engine.cpp`, `compiler/`
**前置**: P5-8, P5-10

**背景**: 對照主流 AI 模型架構（Transformer/GPT/BERT/ResNet/MobileNet/ViT/LLaMA），
現有 ISA 僅有 4 種 ComputeType（NOP/MATMUL/VECTOR/SCALAR），且 SCALAR（+1）/ VECTOR（x²）
語意過於特殊，無法對應真實運算需求。以下為系統性差距分析。

**ISA 差距清單（按優先級）**:

| 優先級 | 缺失運算 | 語意 | 主流模型用途 |
|--------|----------|------|-------------|
| **P0-Critical** | `ADD` | dst[i] = a[i] + b[i] | 殘差連接 (ResNet/Transformer)、bias add |
| **P0-Critical** | `MUL` | dst[i] = a[i] × b[i] | Attention scaling、gating (LSTM/MoE) |
| **P0-Critical** | `RELU` | dst[i] = max(src[i], 0) | 非線性激活（幾乎所有模型） |
| **P1-Important** | `SUB` | dst[i] = a[i] - b[i] | LayerNorm (x - mean)、殘差 |
| **P1-Important** | `MAX` | dst[i] = max(a[i], b[i]) | ReLU 泛化、clamp |
| **P1-Important** | `REDUCE_SUM` | dst = Σ src[i] 沿軸 | Global Average Pooling、Normalization |
| **P1-Important** | `REDUCE_MAX` | dst = max(src[i]) 沿軸 | Max Pooling、Softmax 數值穩定 |
| **P2-Extended** | `EXP` | dst[i] = exp(src[i]) | Softmax、GELU、Sigmoid |
| **P2-Extended** | `RECIPROCAL` | dst[i] = 1/src[i] | Softmax norm、LayerNorm |
| **P2-Extended** | `CLAMP` | dst[i] = clamp(src[i], lo, hi) | 量化 clip、ReLU6 |
| **P2-Extended** | `CAST` | 型別轉換 i8↔i32 | 量化 / 反量化流程 |
| **P3-Future** | `CONV2D` | 2D 捲積 | CNN 特徵提取 (ResNet/YOLO/EfficientNet) |
| **P3-Future** | `GELU` | x·Φ(x) | Transformer 標準激活 (BERT/GPT/ViT) |
| **P3-Future** | `SILU` | x·σ(x) | LLaMA / Mistral 激活 |
| **P3-Future** | `LAYERNORM` | (x-μ)/σ·γ+β | Transformer 層正規化 |
| **P3-Future** | `SOFTMAX` | exp(xi)/Σexp(xj) | 注意力分數計算 |

**交付（P0-Critical + P1-Important）**:

1. **硬體層** (`commands.hpp` + `compute_engine.cpp`):
   - 重新定義 ComputeType enum，擴展為 dual-operand 語意
   - `SCALAR` 改為 `ADD`（element-wise a+b），需支援雙來源 offset
   - `VECTOR` 改為 `MUL`（element-wise a×b）
   - 新增 `RELU`, `SUB`, `MAX`, `REDUCE_SUM`, `REDUCE_MAX`
   - Compute_Command 新增 `src2_offset` 欄位（雙運算元）

2. **二進位格式** (`xbin_loader.hpp` + `xtpu_translate.py`):
   - XBinComputeCommand 擴展 src2_offset
   - 更新 pack/unpack 確保向下相容

3. **IR 層** (`XTPUOps.td`):
   - ComputeType enum 擴展
   - ODS 定義更新

4. **編譯器層** (`LinalgToXTPU.cpp` + `xtpu_import.py`):
   - `linalg.generic(arith.addi)` → `ADD`（非 SCALAR +1）
   - `linalg.generic(arith.maxsi)` → `RELU` / `MAX`
   - `linalg.generic(arith.muli)` → `MUL`
   - `linalg.generic(arith.subi)` → `SUB`

5. **測試**:
   - 每個新 ComputeType 至少一個 unit test
   - 12 個 ONNX 模型重新驗證 bit-exact correctness

**驗收結果** (2026-04-15):
- 27/27 simulator unit tests PASS
- 6/12 ONNX models bit-exact PASS:
  ✅ single_matmul, two_layer_mlp, gemm_mlp, deep_mlp, residual_block, bottleneck_block
- 6/12 output mismatch (known limitation — transpose treated as identity in MVP):
  ✗ dual_path_network, encoder_decoder, gpt_block, mlp_mixer_block, multi_head_attention, transformer_attention
- All 7 new compute types (ADD/MUL/SUB/RELU/MAX/REDUCE_SUM/REDUCE_MAX) implemented in hardware + compiler
- Dual-operand src2_offset support in xbin format (140-byte packets)
- Broadcast add support for bias addition
- Hardware-faithful golden reference in correctness framework
- Engine thread shutdown fix (pure virtual function crash resolved)

---

**所有前置準備已完成，Phase 5 可以正式啟動。** (2026-04-10)

---

## Implementation Roadmap

```
Phase 0 (Week 1)         Phase 1 (Week 2-3)           Phase 2 (Week 3-4)
┌──────────────┐         ┌─────────────────────┐       ┌──────────────────┐
│ P0-1 Deadlock│         │ P1-1 IMemoryPort    │       │ P2-1 Compute ops │
│ P0-2 Atomic  │────────>│ P1-2 SimClock       │──────>│ P2-2 Data tests  │
│ P0-3 StatusReg│        │ P1-3 SystemMemory   │       │ P2-3 Double buf  │
└──────────────┘         │ P1-4 Writeback path │       │ P2-4 Error bits  │
                         └─────────────────────┘       └──────────────────┘
                                                                │
                         Phase 3 (Week 4-6)            Phase 4 (Week 6+)
                         ┌─────────────────────┐       ┌──────────────────┐
                         │ P3-1 DDR Architecture│      │ P4-1 PerfCounters│
                         │ P3-2 DDR Adapter    │       │ P4-2 Logging     │
                    ┌───>│ P3-3 DDR Integration│       │ P4-3 Build system│
                    │    │ P3-4 DDR Perf       │       └──────────────────┘
                    │    │ P3-5 CMake migration │
                    │    └─────────────────────┘
                    │
            DDR Project (external dependency)
```

**Critical path for DDR integration**:
`P0-* → P1-1 (IMemoryPort) → P1-2 (SimClock) → P1-3 (SystemMemory) → P3-2 (DDR Adapter) → P3-3 (Integration)`

---

## DDR 整合前的 Checklist

在開始 Phase 3 之前，確認以下條件已滿足：

- [x] `IMemoryPort` 抽象層已就位，所有 engine 不直接依賴 concrete memory class
- [x] `SimClock` 可運作，所有延遲用 ticks 而非 wallclock ms
- [x] `SystemMemory` 實作完成，資料可端到端驗證
- [x] Writeback path 可用（compute result → system memory）
- [ ] DDR project 的 API 已確認（需與 DDR team 對齊介面）
- [ ] 決定 build system：CMake（推薦）或 Makefile + manual linking

---

## 備註

- Phase 0 是阻塞項，任何 engine 異常都可能死鎖
- Phase 1 是 DDR 整合的 **必要前置**，不可跳過
- Phase 3 的實際程式碼取決於 DDR project 的 API，上述為示意設計
- 若 DDR project 是 cycle-accurate simulator（如 DRAMSim3/Ramulator），整合模式會略有不同（event-driven callback vs blocking call）
