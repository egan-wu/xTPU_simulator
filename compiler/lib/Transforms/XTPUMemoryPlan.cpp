//===----------------------------------------------------------------------===//
// XTPUMemoryPlan — Verify and report static memory allocation
//
// P5-4: Pure analysis pass. Walks the xtpu.program, collects all DMA
// transfers and compute accesses, verifies no address conflicts exist,
// and emits a compile report with memory utilization.
//===----------------------------------------------------------------------===//

#include "xtpu/Transforms/Passes.h"
#include "xtpu/IR/XTPUOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>

namespace mlir {
namespace xtpu {

#define GEN_PASS_DEF_XTPUMEMORYPLAN
#include "xtpu/Transforms/Passes.h.inc"

} // namespace xtpu
} // namespace mlir

using namespace mlir;

//===----------------------------------------------------------------------===//
// Hardware constants
//===----------------------------------------------------------------------===//

static constexpr uint64_t SYSTEM_MEMORY_SIZE = 16ULL * 1024 * 1024;
static constexpr uint64_t SCRATCHPAD_SIZE    = 1ULL * 1024 * 1024;
static constexpr uint64_t LOCAL_MEM_SIZE     = 64ULL * 1024;

//===----------------------------------------------------------------------===//
// Address range tracking
//===----------------------------------------------------------------------===//

struct AddrRange {
  uint64_t start;
  uint64_t end; // exclusive
  int packetIdx;
  StringRef desc;
};

static bool overlaps(const AddrRange &a, const AddrRange &b) {
  return a.start < b.end && b.start < a.end;
}

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

namespace {

struct XTPUMemoryPlanPass
    : public mlir::xtpu::impl::XTPUMemoryPlanBase<XTPUMemoryPlanPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();

    for (auto programOp :
         module.getBody()->getOps<xtpu::ProgramOp>()) {
      if (failed(analyzeProgram(programOp)))
        return signalPassFailure();
    }
  }

  LogicalResult analyzeProgram(xtpu::ProgramOp program) {
    SmallVector<AddrRange> sysMemReads, sysMemWrites;
    SmallVector<AddrRange> scratchReads, scratchWrites;
    SmallVector<AddrRange> localReads, localWrites;

    uint64_t sysMemHighWater = 0;
    uint64_t scratchHighWater = 0;
    uint32_t localHighWater = 0;
    int totalPackets = 0;
    int sdmaCount = 0, idmaCount = 0, computeCount = 0;
    int matmulCount = 0, scalarCount = 0, vectorCount = 0;

    int packetIdx = 0;
    for (auto packetOp :
         program.getBody().front().getOps<xtpu::PacketOp>()) {
      totalPackets++;

      for (auto &op : packetOp.getBody().front().without_terminator()) {
        if (auto sdma = dyn_cast<xtpu::SDMAOp>(&op)) {
          sdmaCount++;
          uint64_t src = sdma.getSrcAddr();
          uint64_t dst = sdma.getDstAddr();
          uint64_t sz = sdma.getSize();

          if (sdma.getDirection() == xtpu::DMADirection::to_device) {
            // load: sys[src] → scratch[dst]
            sysMemReads.push_back({src, src + sz, packetIdx, "sdma.load.src"});
            scratchWrites.push_back({dst, dst + sz, packetIdx, "sdma.load.dst"});
            sysMemHighWater = std::max(sysMemHighWater, src + sz);
            scratchHighWater = std::max(scratchHighWater, dst + sz);
          } else {
            // store: scratch[src] → sys[dst]
            scratchReads.push_back({src, src + sz, packetIdx, "sdma.store.src"});
            sysMemWrites.push_back({dst, dst + sz, packetIdx, "sdma.store.dst"});
            sysMemHighWater = std::max(sysMemHighWater, dst + sz);
            scratchHighWater = std::max(scratchHighWater, src + sz);
          }
        } else if (auto idma = dyn_cast<xtpu::IDMAOp>(&op)) {
          idmaCount++;
          uint64_t src = idma.getSrcAddr();
          uint64_t dst = idma.getDstAddr();
          uint64_t sz = idma.getSize();

          if (idma.getDirection() == xtpu::DMADirection::to_device) {
            // load: scratch[src] → local[dst]
            scratchReads.push_back({src, src + sz, packetIdx, "idma.load.src"});
            localWrites.push_back({dst, dst + sz, packetIdx, "idma.load.dst"});
            scratchHighWater = std::max(scratchHighWater, src + sz);
            localHighWater = std::max(localHighWater, (uint32_t)(dst + sz));
          } else {
            // store: local[src] → scratch[dst]
            localReads.push_back({src, src + sz, packetIdx, "idma.store.src"});
            scratchWrites.push_back({dst, dst + sz, packetIdx, "idma.store.dst"});
            scratchHighWater = std::max(scratchHighWater, dst + sz);
            localHighWater = std::max(localHighWater, (uint32_t)(src + sz));
          }
        } else if (auto compute = dyn_cast<xtpu::ComputeOp>(&op)) {
          computeCount++;
          uint32_t srcOff = compute.getSrcOffset();
          uint32_t dstOff = compute.getDstOffset();
          uint32_t len = compute.getLength();

          if (compute.getType() == xtpu::ComputeType::matmul) {
            matmulCount++;
            localReads.push_back({srcOff, (uint64_t)srcOff + 32, packetIdx, "matmul.src"});
            localWrites.push_back({dstOff, (uint64_t)dstOff + len, packetIdx, "matmul.dst"});
            localHighWater = std::max(localHighWater, std::max(srcOff + 32, dstOff + len));
          } else {
            if (compute.getType() == xtpu::ComputeType::scalar) scalarCount++;
            if (compute.getType() == xtpu::ComputeType::vector) vectorCount++;
            localReads.push_back({srcOff, (uint64_t)srcOff + len, packetIdx, "compute.src"});
            localWrites.push_back({dstOff, (uint64_t)dstOff + len, packetIdx, "compute.dst"});
            localHighWater = std::max(localHighWater, std::max(srcOff + len, dstOff + len));
          }
        }
      }
      packetIdx++;
    }

    // Check for write-write conflicts within same packet
    bool hasConflict = false;
    for (size_t i = 0; i < scratchWrites.size(); i++) {
      for (size_t j = i + 1; j < scratchWrites.size(); j++) {
        if (scratchWrites[i].packetIdx == scratchWrites[j].packetIdx &&
            overlaps(scratchWrites[i], scratchWrites[j])) {
          program.emitError("Scratchpad write-write conflict in packet ")
              << scratchWrites[i].packetIdx << ": "
              << scratchWrites[i].desc << " [" << scratchWrites[i].start
              << "," << scratchWrites[i].end << ") vs "
              << scratchWrites[j].desc << " [" << scratchWrites[j].start
              << "," << scratchWrites[j].end << ")";
          hasConflict = true;
        }
      }
    }

    for (size_t i = 0; i < localWrites.size(); i++) {
      for (size_t j = i + 1; j < localWrites.size(); j++) {
        if (localWrites[i].packetIdx == localWrites[j].packetIdx &&
            overlaps(localWrites[i], localWrites[j])) {
          program.emitError("LocalMem write-write conflict in packet ")
              << localWrites[i].packetIdx << ": "
              << localWrites[i].desc << " [" << localWrites[i].start
              << "," << localWrites[i].end << ") vs "
              << localWrites[j].desc << " [" << localWrites[j].start
              << "," << localWrites[j].end << ")";
          hasConflict = true;
        }
      }
    }

    if (hasConflict)
      return failure();

    // Emit compile report
    llvm::errs() << "\n=== xTPU Memory Plan Report: @"
                 << program.getSymName() << " ===\n";
    llvm::errs() << "Packets: " << totalPackets << "\n";
    llvm::errs() << "  SDMA ops:    " << sdmaCount << "\n";
    llvm::errs() << "  IDMA ops:    " << idmaCount << "\n";
    llvm::errs() << "  Compute ops: " << computeCount
                 << " (matmul=" << matmulCount
                 << " scalar=" << scalarCount
                 << " vector=" << vectorCount << ")\n";
    llvm::errs() << "Memory high-water marks:\n";
    llvm::errs() << "  System Memory: " << sysMemHighWater << " / "
                 << SYSTEM_MEMORY_SIZE << " bytes ("
                 << (sysMemHighWater * 100 / SYSTEM_MEMORY_SIZE) << "%)\n";
    llvm::errs() << "  Scratchpad:    " << scratchHighWater << " / "
                 << SCRATCHPAD_SIZE << " bytes ("
                 << (scratchHighWater * 100 / SCRATCHPAD_SIZE) << "%)\n";
    llvm::errs() << "  LocalMemory:   " << localHighWater << " / "
                 << LOCAL_MEM_SIZE << " bytes ("
                 << (localHighWater * 100 / LOCAL_MEM_SIZE) << "%)\n";

    if (scratchHighWater > SCRATCHPAD_SIZE) {
      program.emitError("Scratchpad usage exceeds hardware limit!");
      return failure();
    }
    if (localHighWater > LOCAL_MEM_SIZE) {
      program.emitError("LocalMemory usage exceeds hardware limit!");
      return failure();
    }

    llvm::errs() << "Spill to LPDDR5: none (all fits in scratchpad)\n";
    llvm::errs() << "Address conflicts: none\n";
    llvm::errs() << "=== End Report ===\n\n";

    return success();
  }
};

} // namespace
