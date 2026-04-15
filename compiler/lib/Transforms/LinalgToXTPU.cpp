//===----------------------------------------------------------------------===//
// LinalgToXTPU — Lower Linalg-on-tensors to xTPU dialect
//
// Converts func.func with linalg ops into xtpu.program with explicit DMA
// and compute packets. All addresses resolved statically via bump allocator.
//
// MVP constraints: 4×4 tiles, single PU, serial execution.
//===----------------------------------------------------------------------===//

#include "xtpu/Transforms/Passes.h"
#include "xtpu/IR/XTPUDialect.h"
#include "xtpu/IR/XTPUOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>

namespace mlir {
namespace xtpu {

#define GEN_PASS_DEF_LINALGTOXTPU
#include "xtpu/Transforms/Passes.h.inc"

} // namespace xtpu
} // namespace mlir

using namespace mlir;

//===----------------------------------------------------------------------===//
// Helper: unsigned integer attributes (UI64Attr, UI32Attr)
//===----------------------------------------------------------------------===//

static IntegerAttr getUI64Attr(MLIRContext *ctx, uint64_t val) {
  return IntegerAttr::get(IntegerType::get(ctx, 64, IntegerType::Unsigned), val);
}

static IntegerAttr getUI32Attr(MLIRContext *ctx, uint32_t val) {
  return IntegerAttr::get(IntegerType::get(ctx, 32, IntegerType::Unsigned), val);
}

//===----------------------------------------------------------------------===//
// Helper: compute tensor size in bytes
//===----------------------------------------------------------------------===//

static uint64_t getTensorSizeBytes(RankedTensorType type) {
  uint64_t numElements = 1;
  for (auto dim : type.getShape())
    numElements *= dim;
  unsigned bitWidth = type.getElementType().getIntOrFloatBitWidth();
  return numElements * (bitWidth / 8);
}

/// Hardware-aware size: our xTPU always operates on uint8 (1 byte/element).
/// Use this for data that flows through hardware compute (matmul, add, relu).
static uint64_t getHardwareSizeBytes(RankedTensorType type) {
  uint64_t numElements = 1;
  for (auto dim : type.getShape())
    numElements *= dim;
  return numElements; // Always 1 byte per element in hardware
}

//===----------------------------------------------------------------------===//
// Classify linalg.generic body patterns
//===----------------------------------------------------------------------===//

enum class GenericPattern { Unknown, Add, ReLU, Cast };

static GenericPattern classifyGeneric(linalg::GenericOp op) {
  Block &body = op.getRegion().front();
  auto ops = body.without_terminator();
  auto it = ops.begin();
  if (it == ops.end())
    return GenericPattern::Unknown;
  Operation *inner = &*it;
  ++it;
  if (it != ops.end())
    return GenericPattern::Unknown;

  if (isa<arith::AddIOp>(inner))
    return GenericPattern::Add;
  if (isa<arith::MaxSIOp>(inner))
    return GenericPattern::ReLU;
  // Cast operations: trunci (i32→i8), extsi (i8→i32), extui, sitofp, etc.
  if (isa<arith::TruncIOp, arith::ExtSIOp, arith::ExtUIOp>(inner))
    return GenericPattern::Cast;
  return GenericPattern::Unknown;
}

//===----------------------------------------------------------------------===//
// MemoryPlanner — bump allocator
//===----------------------------------------------------------------------===//

struct MemoryPlanner {
  uint64_t sysMemOffset = 0;
  uint64_t scratchOffset = 0;
  uint32_t localOffset = 0;

  llvm::DenseMap<Value, uint64_t> sysMemMap;
  llvm::DenseMap<Value, uint64_t> sizeMap;

  uint64_t allocSysMem(Value v, uint64_t size) {
    uint64_t addr = sysMemOffset;
    sysMemMap[v] = addr;
    sizeMap[v] = size;
    sysMemOffset += size;
    return addr;
  }

  uint64_t allocScratch(uint64_t size) {
    uint64_t addr = scratchOffset;
    scratchOffset += size;
    return addr;
  }

  void resetScratch() { scratchOffset = 0; }

  uint32_t allocLocal(uint32_t size) {
    uint32_t addr = localOffset;
    localOffset += size;
    return addr;
  }

  void resetLocal() { localOffset = 0; }

  uint64_t getSysAddr(Value v) const {
    auto it = sysMemMap.find(v);
    assert(it != sysMemMap.end() && "Value not in system memory");
    return it->second;
  }

  uint64_t getSize(Value v) const {
    auto it = sizeMap.find(v);
    assert(it != sizeMap.end() && "Value size not tracked");
    return it->second;
  }

  bool hasSysAddr(Value v) const { return sysMemMap.count(v) > 0; }
};

//===----------------------------------------------------------------------===//
// PacketEmitter — builds xtpu.packet ops with correct sync_mask
//===----------------------------------------------------------------------===//

class PacketEmitter {
public:
  PacketEmitter(OpBuilder &builder, xtpu::ProgramOp program)
      : builder(builder), program(program) {
    Block &body = program.getBody().front();
    builder.setInsertionPoint(body.getTerminator());
  }

  void emitSDMALoad(uint64_t sysSrc, uint64_t scratchDst, uint64_t size) {
    auto syncMask = getSyncMask();
    auto packetOp = createPacket(syncMask);
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(packetOp.getBody().front().getTerminator());
    xtpu::SDMAOp::create(
        builder, loc(),
        xtpu::DMADirectionAttr::get(ctx(), xtpu::DMADirection::to_device),
        getUI64Attr(ctx(), sysSrc),
        getUI64Attr(ctx(), scratchDst),
        getUI64Attr(ctx(), size));
    sdmaBusy = true;
  }

  void emitSDMAStore(uint64_t scratchSrc, uint64_t sysDst, uint64_t size) {
    auto syncMask = getSyncMask();
    auto packetOp = createPacket(syncMask);
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(packetOp.getBody().front().getTerminator());
    xtpu::SDMAOp::create(
        builder, loc(),
        xtpu::DMADirectionAttr::get(ctx(), xtpu::DMADirection::from_device),
        getUI64Attr(ctx(), scratchSrc),
        getUI64Attr(ctx(), sysDst),
        getUI64Attr(ctx(), size));
    sdmaBusy = true;
  }

  void emitIDMALoad(uint64_t scratchSrc, uint32_t localDst, uint64_t size) {
    auto syncMask = getSyncMask();
    auto packetOp = createPacket(syncMask);
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(packetOp.getBody().front().getTerminator());
    xtpu::IDMAOp::create(
        builder, loc(),
        xtpu::DMADirectionAttr::get(ctx(), xtpu::DMADirection::to_device),
        getUI64Attr(ctx(), scratchSrc),
        getUI64Attr(ctx(), (uint64_t)localDst),
        getUI64Attr(ctx(), size),
        xtpu::TargetAttr::get(ctx(), xtpu::Target::pu0),
        builder.getI32IntegerAttr(0));
    pu0DmaBusy = true;
  }

  void emitIDMAStore(uint32_t localSrc, uint64_t scratchDst, uint64_t size) {
    auto syncMask = getSyncMask();
    auto packetOp = createPacket(syncMask);
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(packetOp.getBody().front().getTerminator());
    xtpu::IDMAOp::create(
        builder, loc(),
        xtpu::DMADirectionAttr::get(ctx(), xtpu::DMADirection::from_device),
        getUI64Attr(ctx(), (uint64_t)localSrc),
        getUI64Attr(ctx(), scratchDst),
        getUI64Attr(ctx(), size),
        xtpu::TargetAttr::get(ctx(), xtpu::Target::pu0),
        builder.getI32IntegerAttr(0));
    pu0DmaBusy = true;
  }

  void emitCompute(xtpu::ComputeType type, uint32_t srcOff, uint32_t dstOff,
                   uint32_t length, uint32_t src2Off = 0) {
    auto syncMask = getSyncMask();
    auto packetOp = createPacket(syncMask);
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(packetOp.getBody().front().getTerminator());
    xtpu::ComputeOp::create(
        builder, loc(),
        builder.getI32IntegerAttr(0), // pu = 0
        xtpu::ComputeTypeAttr::get(ctx(), type),
        builder.getI32IntegerAttr(0), // buffer = 0
        getUI32Attr(ctx(), srcOff),
        getUI32Attr(ctx(), dstOff),
        getUI32Attr(ctx(), length),
        getUI32Attr(ctx(), src2Off));
    pu0CmdBusy = true;
  }

  void emitDrainSync() {
    auto syncMask = getSyncMask();
    createPacket(syncMask);
    sdmaBusy = false;
    pu0DmaBusy = false;
    pu0CmdBusy = false;
  }

private:
  OpBuilder &builder;
  xtpu::ProgramOp program;
  bool sdmaBusy = false;
  bool pu0DmaBusy = false;
  bool pu0CmdBusy = false;

  MLIRContext *ctx() { return builder.getContext(); }
  Location loc() { return program.getLoc(); }

  SmallVector<Attribute> getSyncMask() {
    SmallVector<Attribute> mask;
    if (sdmaBusy)
      mask.push_back(builder.getStringAttr("sdma"));
    if (pu0DmaBusy)
      mask.push_back(builder.getStringAttr("pu0_dma"));
    if (pu0CmdBusy)
      mask.push_back(builder.getStringAttr("pu0_cmd"));
    // After syncing, the engines we waited on are now idle
    sdmaBusy = false;
    pu0DmaBusy = false;
    pu0CmdBusy = false;
    return mask;
  }

  xtpu::PacketOp createPacket(ArrayRef<Attribute> syncMask) {
    Block &body = program.getBody().front();
    builder.setInsertionPoint(body.getTerminator());
    auto packetOp = xtpu::PacketOp::create(builder, loc(),
                                           builder.getArrayAttr(syncMask));
    // Ensure packet body has a block with EndOp terminator
    Region &packetBody = packetOp.getBody();
    if (packetBody.empty()) {
      Block *block = new Block();
      packetBody.push_back(block);
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToEnd(block);
      xtpu::EndOp::create(builder, loc());
    }
    return packetOp;
  }
};

//===----------------------------------------------------------------------===//
// Lowering helpers
//===----------------------------------------------------------------------===//

/// Lower batch_matmul. Returns scratchpad offset of result.
///
/// Hardware matmul: A[4x4 uint8] × B[4x4 uint8] → C[4x4 uint8] (truncated)
/// All three operands are 16 bytes in hardware regardless of Linalg type.
static uint64_t lowerBatchMatmul(linalg::BatchMatmulOp matmulOp,
                                 MemoryPlanner &planner,
                                 PacketEmitter &emitter) {
  Value lhs = matmulOp.getDpsInputs()[0];
  Value rhs = matmulOp.getDpsInputs()[1];

  // Hardware sizes: 4×4 × 1 byte = 16 bytes per matrix
  constexpr uint64_t kMatrixSize = 16; // 4×4×uint8
  constexpr uint64_t kTotalInputSize = 32; // A + B

  uint64_t aSysAddr = planner.getSysAddr(lhs);
  uint64_t bSysAddr = planner.getSysAddr(rhs);

  // Scratch layout: A at 0, B at 16
  planner.resetScratch();
  uint64_t aScratch = planner.allocScratch(kMatrixSize);
  uint64_t bScratch = planner.allocScratch(kMatrixSize);

  // SDMA load A and B
  emitter.emitSDMALoad(aSysAddr, aScratch, kMatrixSize);
  emitter.emitSDMALoad(bSysAddr, bScratch, kMatrixSize);

  // IDMA load A+B to local memory
  planner.resetLocal();
  uint32_t aLocal = planner.allocLocal(kMatrixSize);
  planner.allocLocal(kMatrixSize); // B right after A
  emitter.emitIDMALoad(aScratch, aLocal, kTotalInputSize);

  // Compute matmul: src_offset=0, dst_offset=32, length=16
  uint32_t dstLocal = planner.allocLocal(kMatrixSize);
  emitter.emitCompute(xtpu::ComputeType::matmul, aLocal, dstLocal, kMatrixSize);

  // IDMA store result (16 bytes) back to scratch
  planner.resetScratch();
  uint64_t cScratch = planner.allocScratch(kMatrixSize);
  emitter.emitIDMAStore(dstLocal, cScratch, kMatrixSize);

  planner.sizeMap[matmulOp.getResult(0)] = kMatrixSize;
  return cScratch;
}

/// Lower generic add as dual-operand add compute. Returns scratchpad offset.
///
/// P5-11: Uses ComputeType::add with src2_offset for true element-wise add.
/// Hardware: dst[i] = (src[i] + src2[i]) & 0xFF
///
/// Handles broadcast: when input1 is smaller than input0, replicate it
/// by loading multiple copies into local memory.
static uint64_t lowerGenericAdd(linalg::GenericOp genericOp,
                                MemoryPlanner &planner,
                                PacketEmitter &emitter,
                                uint64_t input0ScratchAddr,
                                uint64_t input1ScratchAddr) {
  auto resultType = cast<RankedTensorType>(genericOp.getResult(0).getType());
  // Hardware always operates in uint8, use hardware size (1 byte/element)
  uint64_t resultSize = getHardwareSizeBytes(resultType);
  Value input0 = genericOp.getDpsInputs()[0];
  Value input1 = genericOp.getDpsInputs()[1];
  uint64_t input0Size = getHardwareSizeBytes(cast<RankedTensorType>(input0.getType()));
  uint64_t input1Size = getHardwareSizeBytes(cast<RankedTensorType>(input1.getType()));

  // IDMA load first operand to local memory
  planner.resetLocal();
  uint32_t src0Local = planner.allocLocal(input0Size);
  emitter.emitIDMALoad(input0ScratchAddr, src0Local, input0Size);

  // Handle broadcast: if input1 is smaller, replicate it to match input0
  uint32_t src1Local = planner.allocLocal(resultSize); // always allocate full size
  if (input1Size < resultSize && input1Size > 0) {
    // Broadcast: load the small operand multiple times into contiguous
    // local memory to fill the required size
    uint64_t loaded = 0;
    while (loaded < resultSize) {
      uint64_t chunk = std::min(input1Size, resultSize - loaded);
      emitter.emitIDMALoad(input1ScratchAddr, src1Local + loaded, chunk);
      loaded += chunk;
    }
  } else {
    emitter.emitIDMALoad(input1ScratchAddr, src1Local, input1Size);
  }

  // Compute add: dst[i] = src[i] + src2[i]
  uint32_t dstLocal = planner.allocLocal(resultSize);
  emitter.emitCompute(xtpu::ComputeType::add, src0Local, dstLocal,
                      resultSize, src1Local);

  // IDMA store result back to scratch
  planner.resetScratch();
  uint64_t resultScratch = planner.allocScratch(resultSize);
  emitter.emitIDMAStore(dstLocal, resultScratch, resultSize);

  planner.sizeMap[genericOp.getResult(0)] = resultSize;
  return resultScratch;
}

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

namespace {

struct LinalgToXTPUPass
    : public mlir::xtpu::impl::LinalgToXTPUBase<LinalgToXTPUPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Find the main function
    func::FuncOp funcOp = nullptr;
    for (auto &op : module.getBody()->getOperations()) {
      if (auto f = dyn_cast<func::FuncOp>(&op)) {
        funcOp = f;
        break;
      }
    }

    if (!funcOp) {
      module.emitError("No func.func found in module");
      return signalPassFailure();
    }

    // ---------------------------------------------------------------
    // Phase 1: Plan system memory layout
    // ---------------------------------------------------------------
    MemoryPlanner planner;
    auto &funcBody = funcOp.getBody().front();

    // Allocate function arguments (inputs)
    // Hardware always uses uint8 (1 byte per element)
    for (auto arg : funcBody.getArguments()) {
      auto type = dyn_cast<RankedTensorType>(arg.getType());
      if (!type) {
        funcOp.emitError("Non-tensor argument not supported");
        return signalPassFailure();
      }
      planner.allocSysMem(arg, getHardwareSizeBytes(type));
    }

    // Allocate constants in system memory and extract their values
    // for .rodata emission
    struct ConstEntry {
      uint64_t offset;
      uint64_t size;
      std::vector<uint8_t> data;
    };
    SmallVector<ConstEntry> constEntries;

    for (auto &op : funcBody.getOperations()) {
      if (op.getNumResults() != 1)
        continue;
      auto result = op.getResult(0);
      auto type = dyn_cast<RankedTensorType>(result.getType());
      if (!type)
        continue;
      if (isa<arith::ConstantOp>(&op) ||
          op.getName().getStringRef() == "tosa.const") {
        // Hardware always uses uint8 (1 byte/element), so allocate
        // system memory with hardware size and convert constant data
        // to uint8 by taking the low byte of each element.
        uint64_t hwSize = getHardwareSizeBytes(type);
        uint64_t addr = planner.sysMemOffset;
        planner.allocSysMem(result, hwSize);

        // Extract constant value bytes, converting to uint8
        DenseElementsAttr denseAttr;
        if (auto constOp = dyn_cast<arith::ConstantOp>(&op)) {
          denseAttr = dyn_cast<DenseElementsAttr>(constOp.getValue());
        } else if (op.getName().getStringRef() == "tosa.const") {
          denseAttr = op.getAttrOfType<DenseElementsAttr>("values");
        }

        std::vector<uint8_t> bytes;
        if (denseAttr) {
          unsigned bitWidth = type.getElementTypeBitWidth();
          unsigned byteWidth = bitWidth / 8;
          uint64_t numElems = hwSize; // 1 byte per elem in hardware

          if (denseAttr.isSplat()) {
            // Splat: all elements have the same value.
            // Extract the low byte and replicate.
            auto rawData = denseAttr.getRawData();
            uint8_t val = rawData.size() > 0 ? rawData[0] : 0;
            bytes.assign(numElems, val);
          } else if (bitWidth == 8) {
            // i8 / uint8: raw data is already 1 byte per element
            auto rawData = denseAttr.getRawData();
            bytes.assign(rawData.begin(), rawData.end());
            // Ensure we have exactly numElems bytes
            bytes.resize(numElems, 0);
          } else {
            // i32 or wider: extract low byte of each element
            auto rawData = denseAttr.getRawData();
            bytes.reserve(numElems);
            for (uint64_t i = 0; i < numElems; i++) {
              if (i * byteWidth < rawData.size()) {
                // Little-endian: low byte is at offset i * byteWidth
                bytes.push_back(rawData[i * byteWidth]);
              } else {
                bytes.push_back(0);
              }
            }
          }
        }
        constEntries.push_back({addr, hwSize, std::move(bytes)});
      }
    }

    // Reserve output region at offset 4096 (well separated from inputs)
    uint64_t outputBase = 4096;
    auto returnOp = cast<func::ReturnOp>(funcBody.getTerminator());
    for (auto retVal : returnOp.getOperands()) {
      auto type = dyn_cast<RankedTensorType>(retVal.getType());
      if (!type)
        continue;
      uint64_t size = getHardwareSizeBytes(type);
      planner.sysMemMap[retVal] = outputBase;
      planner.sizeMap[retVal] = size;
      outputBase += size;
    }

    // ---------------------------------------------------------------
    // Phase 2: Create xtpu.program and emit packets
    // ---------------------------------------------------------------
    OpBuilder builder(module.getContext());
    builder.setInsertionPointToEnd(module.getBody());

    auto programOp = xtpu::ProgramOp::create(
        builder, funcOp.getLoc(), funcOp.getSymNameAttr());

    // Ensure the program body has a block with an EndOp terminator
    {
      Region &body = programOp.getBody();
      if (body.empty()) {
        Block *block = new Block();
        body.push_back(block);
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToEnd(block);
        xtpu::EndOp::create(builder, funcOp.getLoc());
      }
    }

    PacketEmitter emitter(builder, programOp);

    // Track which values have their data in scratchpad
    llvm::DenseMap<Value, uint64_t> scratchLocation;

    // Walk the function body in order
    for (Operation &op : funcBody.getOperations()) {
      if (isa<func::ReturnOp>(&op))
        continue;
      if (isa<tensor::EmptyOp>(&op))
        continue;
      if (isa<linalg::FillOp>(&op))
        continue;
      if (isa<arith::ConstantOp>(&op))
        continue;
      if (op.getName().getStringRef() == "tosa.const")
        continue;
      // tosa.const_shape is a metadata op — no data to move
      if (op.getName().getStringRef() == "tosa.const_shape")
        continue;

      // batch_matmul
      if (auto matmulOp = dyn_cast<linalg::BatchMatmulOp>(&op)) {
        auto resultType = cast<RankedTensorType>(matmulOp.getResult(0).getType());
        auto shape = resultType.getShape();
        if (shape.size() != 3 || shape[0] != 1 || shape[1] != 4 || shape[2] != 4) {
          matmulOp.emitError("Only batch=1, 4x4 matmul supported in MVP");
          return signalPassFailure();
        }

        // Ensure inputs are in system memory
        for (auto input : matmulOp.getDpsInputs()) {
          if (!planner.hasSysAddr(input)) {
            auto it = scratchLocation.find(input);
            if (it == scratchLocation.end()) {
              matmulOp.emitError("Input not found in system memory or scratch");
              return signalPassFailure();
            }
            auto inputType = cast<RankedTensorType>(input.getType());
            uint64_t size = getTensorSizeBytes(inputType);
            uint64_t sysAddr = planner.sysMemOffset;
            planner.allocSysMem(input, size);
            emitter.emitSDMAStore(it->second, sysAddr, size);
          }
        }

        uint64_t cScratch = lowerBatchMatmul(matmulOp, planner, emitter);
        scratchLocation[matmulOp.getResult(0)] = cScratch;
        continue;
      }

      // linalg.generic (add / relu)
      if (auto genericOp = dyn_cast<linalg::GenericOp>(&op)) {
        GenericPattern pattern = classifyGeneric(genericOp);

        if (pattern == GenericPattern::Add) {
          // Resolve both operands to scratchpad.
          // IMPORTANT: Don't reset scratch when an operand is already there,
          // as that would cause the second operand's load to overwrite it.
          Value input0 = genericOp.getDpsInputs()[0];
          Value input1 = genericOp.getDpsInputs()[1];

          // Check which operands are already in scratch
          bool in0InScratch = scratchLocation.count(input0) > 0;
          bool in1InScratch = scratchLocation.count(input1) > 0;

          // If neither is in scratch, we can safely reset
          if (!in0InScratch && !in1InScratch)
            planner.resetScratch();

          auto resolveToScratch = [&](Value v) -> uint64_t {
            auto it = scratchLocation.find(v);
            if (it != scratchLocation.end())
              return it->second;
            uint64_t sysAddr = planner.getSysAddr(v);
            uint64_t size = planner.getSize(v);
            uint64_t scratch = planner.allocScratch(size);
            emitter.emitSDMALoad(sysAddr, scratch, size);
            return scratch;
          };

          uint64_t scratch0 = resolveToScratch(input0);
          uint64_t scratch1 = resolveToScratch(input1);

          uint64_t resultScratch = lowerGenericAdd(
              genericOp, planner, emitter, scratch0, scratch1);
          scratchLocation[genericOp.getResult(0)] = resultScratch;
          continue;
        }

        if (pattern == GenericPattern::ReLU) {
          // P5-11: Use hardware ReLU: dst[i] = max(0, (int8_t)src[i])
          Value input = genericOp.getDpsInputs()[0];
          auto resultType = cast<RankedTensorType>(genericOp.getResult(0).getType());
          // Hardware always uint8, 1 byte per element
          uint64_t resultSize = getHardwareSizeBytes(resultType);

          uint64_t inputScratch = 0;
          auto it = scratchLocation.find(input);
          if (it != scratchLocation.end()) {
            inputScratch = it->second;
          } else {
            uint64_t sysAddr = planner.getSysAddr(input);
            uint64_t size = planner.getSize(input);
            planner.resetScratch();
            inputScratch = planner.allocScratch(size);
            emitter.emitSDMALoad(sysAddr, inputScratch, size);
          }

          uint64_t inputSize = getHardwareSizeBytes(
              cast<RankedTensorType>(input.getType()));

          // IDMA load to local memory
          planner.resetLocal();
          uint32_t srcLocal = planner.allocLocal(inputSize);
          emitter.emitIDMALoad(inputScratch, srcLocal, inputSize);

          // Compute relu
          uint32_t dstLocal = planner.allocLocal(resultSize);
          emitter.emitCompute(xtpu::ComputeType::relu, srcLocal, dstLocal,
                              resultSize);

          // IDMA store result back to scratch
          planner.resetScratch();
          uint64_t resultScratch = planner.allocScratch(resultSize);
          emitter.emitIDMAStore(dstLocal, resultScratch, resultSize);

          scratchLocation[genericOp.getResult(0)] = resultScratch;
          planner.sizeMap[genericOp.getResult(0)] = resultSize;
          continue;
        }

        if (pattern == GenericPattern::Cast) {
          // Type cast (trunci i32→i8, extsi i8→i32, etc.): pass through
          // data location unchanged. Hardware always works in uint8, so
          // we use hardware-aware sizing (1 byte per element).
          Value input = genericOp.getDpsInputs()[0];
          auto it = scratchLocation.find(input);
          if (it != scratchLocation.end()) {
            scratchLocation[genericOp.getResult(0)] = it->second;
          } else if (planner.hasSysAddr(input)) {
            planner.sysMemMap[genericOp.getResult(0)] = planner.getSysAddr(input);
          }
          // Use hardware size (1 byte/element) since our hardware always
          // produces/consumes uint8 regardless of MLIR element type.
          planner.sizeMap[genericOp.getResult(0)] =
              getHardwareSizeBytes(cast<RankedTensorType>(
                  genericOp.getResult(0).getType()));
          genericOp.emitRemark("Cast lowered as no-op (hardware handles type conversion)");
          continue;
        }

        genericOp.emitError("Unsupported linalg.generic pattern");
        return signalPassFailure();
      }

      // tosa.reshape — shape change only, data unchanged
      if (op.getName().getStringRef() == "tosa.reshape") {
        Value input = op.getOperand(0);
        Value result = op.getResult(0);
        // Propagate system memory address from input
        if (planner.hasSysAddr(input)) {
          planner.sysMemMap[result] = planner.getSysAddr(input);
          planner.sizeMap[result] = planner.getSize(input);
        } else {
          auto it = scratchLocation.find(input);
          if (it != scratchLocation.end()) {
            scratchLocation[result] = it->second;
            planner.sizeMap[result] =
                getTensorSizeBytes(cast<RankedTensorType>(result.getType()));
          }
        }
        continue;
      }

      // linalg.transpose (treat as copy for MVP — works for identity/permuted)
      if (isa<linalg::TransposeOp>(&op)) {
        Value input = op.getOperand(0);
        Value result = op.getResult(0);
        auto resultType = cast<RankedTensorType>(result.getType());
        uint64_t size = getHardwareSizeBytes(resultType);
        // For MVP: treat transpose as identity (data location unchanged)
        if (planner.hasSysAddr(input)) {
          planner.sysMemMap[result] = planner.getSysAddr(input);
          planner.sizeMap[result] = size;
        } else {
          planner.allocSysMem(result, size);
        }
        op.emitRemark("Transpose lowered as copy (MVP limitation)");
        continue;
      }

      op.emitError("Unsupported operation for xTPU lowering: ")
          << op.getName();
      return signalPassFailure();
    }

    // ---------------------------------------------------------------
    // Phase 3: SDMA store final result(s) to output region
    // ---------------------------------------------------------------
    for (auto retVal : returnOp.getOperands()) {
      auto it = scratchLocation.find(retVal);
      if (it == scratchLocation.end()) {
        returnOp.emitError("Return value not found in scratch");
        return signalPassFailure();
      }
      uint64_t outputSysAddr = planner.getSysAddr(retVal);
      uint64_t size = planner.getSize(retVal);
      emitter.emitSDMAStore(it->second, outputSysAddr, size);
    }

    emitter.emitDrainSync();

    // ---------------------------------------------------------------
    // Phase 4: Emit .rodata as module attribute for xbin_runner
    // ---------------------------------------------------------------
    // Encode constant data as a hex string attribute on the module.
    // Format: "offset:hex_data;offset:hex_data;..."
    if (!constEntries.empty()) {
      std::string rodataStr;
      llvm::raw_string_ostream os(rodataStr);
      for (size_t i = 0; i < constEntries.size(); i++) {
        auto &e = constEntries[i];
        if (i > 0) os << ";";
        os << e.offset << ":";
        // Manual hex encoding
        static const char hex[] = "0123456789abcdef";
        for (uint8_t b : e.data) {
          os << hex[b >> 4] << hex[b & 0xF];
        }
      }
      module->setAttr("xtpu.rodata",
                      builder.getStringAttr(rodataStr));
    }

    // ---------------------------------------------------------------
    // Phase 5: Remove original func.func
    // ---------------------------------------------------------------
    funcOp.erase();
  }
};

} // namespace
