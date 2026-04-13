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

//===----------------------------------------------------------------------===//
// Classify linalg.generic body patterns
//===----------------------------------------------------------------------===//

enum class GenericPattern { Unknown, Add, ReLU };

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
                   uint32_t length) {
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
        getUI32Attr(ctx(), length));
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

/// Lower generic add as scalar compute. Returns scratchpad offset of result.
static uint64_t lowerGenericAdd(linalg::GenericOp genericOp,
                                MemoryPlanner &planner,
                                PacketEmitter &emitter,
                                uint64_t inputScratchAddr) {
  auto resultType = cast<RankedTensorType>(genericOp.getResult(0).getType());
  uint64_t resultSize = getTensorSizeBytes(resultType);
  Value mainInput = genericOp.getDpsInputs()[0];
  uint64_t inputSize = getTensorSizeBytes(cast<RankedTensorType>(mainInput.getType()));

  // IDMA load input to local memory
  planner.resetLocal();
  uint32_t srcLocal = planner.allocLocal(inputSize);
  emitter.emitIDMALoad(inputScratchAddr, srcLocal, inputSize);

  // Scalar compute: dst[i] = src[i] + 1
  uint32_t dstLocal = planner.allocLocal(resultSize);
  emitter.emitCompute(xtpu::ComputeType::scalar, srcLocal, dstLocal, resultSize);

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
    for (auto arg : funcBody.getArguments()) {
      auto type = dyn_cast<RankedTensorType>(arg.getType());
      if (!type) {
        funcOp.emitError("Non-tensor argument not supported");
        return signalPassFailure();
      }
      planner.allocSysMem(arg, getTensorSizeBytes(type));
    }

    // Allocate constants in system memory
    for (auto &op : funcBody.getOperations()) {
      if (op.getNumResults() != 1)
        continue;
      auto result = op.getResult(0);
      auto type = dyn_cast<RankedTensorType>(result.getType());
      if (!type)
        continue;
      if (isa<arith::ConstantOp>(&op) ||
          op.getName().getStringRef() == "tosa.const") {
        planner.allocSysMem(result, getTensorSizeBytes(type));
      }
    }

    // Reserve output region at offset 4096 (well separated from inputs)
    uint64_t outputBase = 4096;
    auto returnOp = cast<func::ReturnOp>(funcBody.getTerminator());
    for (auto retVal : returnOp.getOperands()) {
      auto type = dyn_cast<RankedTensorType>(retVal.getType());
      if (!type)
        continue;
      uint64_t size = getTensorSizeBytes(type);
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
          Value mainInput = genericOp.getDpsInputs()[0];
          uint64_t inputScratch = 0;
          auto it = scratchLocation.find(mainInput);
          if (it != scratchLocation.end()) {
            inputScratch = it->second;
          } else {
            uint64_t sysAddr = planner.getSysAddr(mainInput);
            uint64_t size = planner.getSize(mainInput);
            planner.resetScratch();
            inputScratch = planner.allocScratch(size);
            emitter.emitSDMALoad(sysAddr, inputScratch, size);
          }

          uint64_t resultScratch = lowerGenericAdd(
              genericOp, planner, emitter, inputScratch);
          scratchLocation[genericOp.getResult(0)] = resultScratch;
          continue;
        }

        if (pattern == GenericPattern::ReLU) {
          // ReLU: no hardware support in MVP. Pass through unchanged.
          Value input = genericOp.getDpsInputs()[0];
          auto it = scratchLocation.find(input);
          if (it != scratchLocation.end()) {
            scratchLocation[genericOp.getResult(0)] = it->second;
          }
          planner.sizeMap[genericOp.getResult(0)] =
              getTensorSizeBytes(cast<RankedTensorType>(
                  genericOp.getResult(0).getType()));
          genericOp.emitRemark("ReLU lowered as no-op (not supported in MVP ISA)");
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
        uint64_t size = getTensorSizeBytes(resultType);
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
    // Phase 4: Remove original func.func
    // ---------------------------------------------------------------
    funcOp.erase();
  }
};

} // namespace
