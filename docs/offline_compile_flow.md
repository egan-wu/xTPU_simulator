# xTPU Offline Compiler 架構與實作教學指南

這份文件旨在幫助初學者與新進開發者了解 xTPU 專案的 Offline Compile 流程。我們將從整體架構的高階視角切入，接著深入探討各個編譯階段的演算法原理與程式碼實作，最後提供「如何擴充與修改編譯器」的實戰指引。

---

## 1. 高階流程概述 (High-Level Overview)

xTPU 編譯器的主要任務是將大家熟知的機器學習模型（ONNX 格式）轉換成 xTPU 模擬器可以執行的專屬二進位格式（`.xbin`）。整個轉換過程被分為三個主要階段，每個階段都由特定的工具負責：

```text
  [ ONNX Model ] (.onnx)
        │
        ▼ 1. Frontend: xtpu-import (Python)
  [ TOSA MLIR ] (.tosa.mlir)
        │
        ▼ 2. Middle-end: xtpu-opt (C++ / MLIR)
  [ xTPU MLIR ] (.xtpu.mlir)
        │
        ▼ 3. Backend: xtpu-translate (Python)
  [ Executable ] (.xbin)
```

1. **Frontend (`xtpu-import`)**：
   - 負責讀取 ONNX 模型，進行型別與形狀推導（Shape Inference）。
   - 將 ONNX 算子（如 `MatMul`, `Add`, `Relu`）映射到 MLIR 的高階張量方言（TOSA Dialect）。
2. **Middle-end (`xtpu-opt`)**：
   - 使用 MLIR 框架的核心能力。首先將 TOSA 降級（Lowering）到 `Linalg` 方言。
   - 接著執行客製化的 Pass（`LinalgToXTPU.cpp`），將無關硬體的張量運算轉換為 **包含 DMA 搬移與硬體同步指令** 的 `xTPU` 方言。
3. **Backend (`xtpu-translate`)**：
   - 解析 `xTPU` 方言的文字檔，並根據 xTPU 模擬器定義的封包格式（VLIW Packet），將指令打包並序列化成 `.xbin` 二進位檔案。

---

## 2. 深入探討與實作原理

### 2.1 Frontend: `xtpu-import`
檔案位置：`compiler/tools/xtpu-import/xtpu_import.py`

**實作原理**：
這是一個 Python 腳本，使用 `onnx` 套件來解析模型。
- **型別對應**：將 ONNX 的資料型別（如 `INT8`, `FLOAT`）對應到 MLIR 的型別（如 `i8`, `f32`）。我們 xTPU 主要支援 INT8 量化流程。
- **常數處理 (Initializers)**：模型中的權重會被提取並轉換為 `tosa.const` 節點。
- **節點轉換**：遍歷 ONNX Graph 中的每一個 Node，並呼叫對應的轉換函式（例如 `convert_matmul`、`convert_relu`）。每個函式會生成對應的 TOSA MLIR 語法字串。
- **處理 Broadcasting**：在加法 (`Add`) 等運算中，若兩個輸入張量形狀不同，Frontend 會主動安插 `tosa.reshape` 或是利用 TOSA 支援的廣播語義來處理。

### 2.2 Middle-end: `LinalgToXTPU.cpp`
檔案位置：`compiler/lib/Transforms/LinalgToXTPU.cpp`

這是整個編譯器**最核心且最需要硬體知識**的模組。它的任務是將運算降級成 xTPU 指令（封裝成封包 VLIW Packet）。

**演算法與核心元件**：
1. **MemoryPlanner (記憶體規劃器)**：
   - **原理**：採用最簡單的 Bump Allocator（單向遞增分配器）。
   - **實作**：它追蹤 System Memory (LPDDR), Scratchpad Memory (1MB SRAM) 以及 Local Memory (PU內部的暫存) 的 offset（位移量）。
   - 當需要分配空間時，直接將當前的 offset 加上所需的大小（Size），然後更新 offset。這確保了不同張量在記憶體中不會發生碰撞。
   - 硬體規定所有運算均使用 `uint8` (1 byte/element)，因此 `MemoryPlanner` 會計算每個 Tensor 在硬體中的實際位元組大小。

2. **PacketEmitter (封包生成器)**：
   - **原理**：xTPU 是 VLIW (Very Long Instruction Word) 架構，每個封包可以同時包含 sDMA, iDMA 和兩個 PU 的計算指令。
   - **同步機制 (`sync_mask`)**：這非常重要！當發出一個指令（例如載入記憶體），後續的指令若依賴該資料，必須等待它完成。`PacketEmitter` 會記錄哪些引擎目前是 "Busy"（忙碌中）。在生成下一個封包時，如果需要等待，它會產生一個對應的 `sync_mask`（例如等待 `sdma`, `pu0_dma`）。
   - **實作**：提供 `emitSDMALoad`, `emitIDMALoad`, `emitCompute` 等方法，將這些操作寫成 `xtpu` MLIR 方言。

3. **操作降級邏輯 (Lowering Logic)**：
   - 遍歷 `linalg` 運算（如 `linalg.batch_matmul`, `linalg.generic`）。
   - **以 `Generic Add` 為例**：
     1. 解析 operands，確認輸入在哪裡。
     2. 透過 `emitSDMALoad` 將資料從 System Memory 搬到 Scratchpad。
     3. 透過 `emitIDMALoad` 將資料從 Scratchpad 搬到 PU Local Memory。
     4. 發出 `emitCompute(xtpu::ComputeType::add, ...)` 進行計算。
     5. 透過 `emitIDMAStore` 將結果搬回 Scratchpad。
   - **常數處理 (.rodata)**：將權重資料轉換為 byte array 並以十六進位字串的形式存入 MLIR 的 Module Attribute (`xtpu.rodata`)。

### 2.3 Backend: `xtpu-translate`
檔案位置：`compiler/tools/xtpu-translate/xtpu_translate.py`

**實作原理**：
這個 Python 腳本將包含 `xtpu.packet` 的 MLIR 轉為 `.xbin`。
- **解析 MLIR (Regex Parsing)**：使用正則表達式掃描每一行的 `xtpu.sdma`, `xtpu.idma`, `xtpu.compute` 和 `sync_mask`。
- **建構 Data Class**：將掃描到的指令建立為 Python 的 `DMACommand` 和 `ComputeCommand` 物件。
- **序列化 (Serialization)**：使用 Python 的 `struct.pack`，將指令封裝成硬體規定的固定位元組（例如 sDMA 40 bytes, Compute 28 bytes）。一個完整的 `VLIWPacket` 剛好是 140 bytes。
- **二進位檔案結構**：
  - **Header**：包含 Magic Number `XTPU`、版本號、Entry Offset。
  - **.text**：所有的 VLIW 封包指令。
  - **.rodata**：唯讀常數資料（如權重）。
  - **.meta**：JSON 格式的中繼資料，描述模型輸入輸出的形狀。

---

## 3. 實戰教學：如何擴充編譯器

假設今天硬體新增了一個指令：**Sigmoid (`xtpu::ComputeType::sigmoid`)**，身為編譯器開發者，你需要進行以下三個步驟來打通全線路：

### 步驟 1：修改 Frontend (`xtpu-import.py`)
1. 將 `Sigmoid` 加入支援的 OP 列表 `_SUPPORTED_OPS`。
2. 實作 `convert_sigmoid(self, node: onnx.NodeProto)` 函式。
3. 在 `convert_sigmoid` 中，產生對應的 `tosa.sigmoid` MLIR 節點。
   *(註：若 TOSA 不支援，可轉為其他等效 Linalg 表示或直接定義客製化的 TOSA/Linalg Op)*。

### 步驟 2：修改 Middle-end (`LinalgToXTPU.cpp`)
1. 找到 `classifyGeneric` 函式，教導編譯器如何認出 Sigmoid 運算（例如檢查內部是否包含計算 Sigmoid 的 `math` 或 `arith` op），並回傳如 `GenericPattern::Sigmoid`。
2. 在主邏輯中新增判斷分支：
   ```cpp
   if (pattern == GenericPattern::Sigmoid) {
       // 1. 確認輸入在 Scratchpad 或 System Memory，準備搬移
       // 2. emitIDMALoad 將資料搬進 Local Memory
       // 3. emitCompute(xtpu::ComputeType::sigmoid, ...) 呼叫硬體指令
       // 4. emitIDMAStore 把結果存回 Scratchpad
   }
   ```
3. 確認 C++ MLIR Dialect (`XTPUEnums.td` 等) 中有定義 `sigmoid` 這個 `ComputeType`。

### 步驟 3：修改 Backend (`xtpu_translate.py`)
1. 在 Python 腳本的最上方常數區塊，新增對應硬體 ISA 的 Enum 值：
   ```python
   COMPUTE_SIGMOID = 11  # 假設硬體定義為 11
   COMPUTE_TYPE_MAP = {
       ...
       "sigmoid": COMPUTE_SIGMOID,
   }
   ```
2. 這樣當腳本解析到 `type = sigmoid` 的文字時，就會自動包裝成正確的硬體整數代碼，寫入 `.xbin` 中。

---

## 4. 總結

xTPU 的 Offline Compile 流程設計得非常模組化：
- **Python 腳本 (Frontend/Backend)**：適合快速處理文字解析、檔案格式轉換與高階 Graph 解析。
- **C++ MLIR Pass (Middle-end)**：提供強大的編譯器基礎架構，讓我們能以嚴謹的方式處理記憶體配置（MemoryPlanner）與硬體指令排程（PacketEmitter）。

對於初學者，建議先從 `xtpu-translate.py` 去觀察封包的二進位結構，接著看 `xtpu-import.py` 了解 ONNX 怎麼變成文字，最後再深入魔王關卡 `LinalgToXTPU.cpp` 研究如何生成 DMA 搬移與同步鎖（Sync Mask）。

祝你在 xTPU 的開發旅程中順利！
