//===----------------------------------------------------------------------===//
// XTPUSchedule — VLIW packet scheduling: merge compatible packets
//
// P5-5: Optimizes the serial packet stream from LinalgToXTPU by merging
// consecutive packets that use non-conflicting engines.
//
// Key optimizations:
//   1. Cross-engine merging: SDMA + IDMA or SDMA + Compute in same packet
//   2. Sync barrier reduction: remove unnecessary waits
//   3. Engine utilization reporting
//
// Correctness: merged schedule produces same memory-visible behavior
// as the serial schedule. We only merge when there are no data dependencies
// between the ops (they access disjoint memory regions or the sync was
// only for engine completion, not data readiness).
//===----------------------------------------------------------------------===//

#include "xtpu/Transforms/Passes.h"
#include "xtpu/IR/XTPUOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>

namespace mlir {
namespace xtpu {

#define GEN_PASS_DEF_XTPUSCHEDULE
#include "xtpu/Transforms/Passes.h.inc"

} // namespace xtpu
} // namespace mlir

using namespace mlir;

//===----------------------------------------------------------------------===//
// Engine slot classification
//===----------------------------------------------------------------------===//

enum EngineSlot : uint8_t {
  SLOT_NONE    = 0,
  SLOT_SDMA    = 1 << 0,
  SLOT_IDMA    = 1 << 1,
  SLOT_PU0_CMD = 1 << 2,
  SLOT_PU1_CMD = 1 << 3,
};

/// Get the engine slots used by ops inside a packet.
static uint8_t getUsedSlots(xtpu::PacketOp packet) {
  uint8_t slots = SLOT_NONE;
  for (auto &op : packet.getBody().front().without_terminator()) {
    if (isa<xtpu::SDMAOp>(&op))
      slots |= SLOT_SDMA;
    else if (isa<xtpu::IDMAOp>(&op))
      slots |= SLOT_IDMA;
    else if (auto compute = dyn_cast<xtpu::ComputeOp>(&op)) {
      if (compute.getPu() == 0)
        slots |= SLOT_PU0_CMD;
      else
        slots |= SLOT_PU1_CMD;
    }
  }
  return slots;
}

/// Get the engine slots that a sync_mask waits on.
static uint8_t getSyncSlots(xtpu::PacketOp packet) {
  uint8_t slots = SLOT_NONE;
  for (auto attr : packet.getSyncMask()) {
    auto str = cast<StringAttr>(attr).getValue();
    if (str == "sdma") slots |= SLOT_SDMA;
    else if (str == "pu0_dma") slots |= SLOT_IDMA;
    else if (str == "pu0_cmd") slots |= SLOT_PU0_CMD;
    else if (str == "pu1_dma") slots |= SLOT_IDMA;
    else if (str == "pu1_cmd") slots |= SLOT_PU1_CMD;
  }
  return slots;
}

/// Check if a packet is empty (only has terminator).
static bool isEmptyPacket(xtpu::PacketOp packet) {
  return packet.getBody().front().without_terminator().empty();
}

//===----------------------------------------------------------------------===//
// Address range overlap checking for safety
//===----------------------------------------------------------------------===//

struct MemAccess {
  uint64_t addr;
  uint64_t size;
  bool isWrite;
  int space; // 0=sysmem, 1=scratch, 2=local
};

static SmallVector<MemAccess> getMemAccesses(xtpu::PacketOp packet) {
  SmallVector<MemAccess> accesses;
  for (auto &op : packet.getBody().front().without_terminator()) {
    if (auto sdma = dyn_cast<xtpu::SDMAOp>(&op)) {
      if (sdma.getDirection() == xtpu::DMADirection::to_device) {
        accesses.push_back({sdma.getSrcAddr(), sdma.getSize(), false, 0}); // read sys
        accesses.push_back({sdma.getDstAddr(), sdma.getSize(), true, 1});  // write scratch
      } else {
        accesses.push_back({sdma.getSrcAddr(), sdma.getSize(), false, 1}); // read scratch
        accesses.push_back({sdma.getDstAddr(), sdma.getSize(), true, 0});  // write sys
      }
    } else if (auto idma = dyn_cast<xtpu::IDMAOp>(&op)) {
      if (idma.getDirection() == xtpu::DMADirection::to_device) {
        accesses.push_back({idma.getSrcAddr(), idma.getSize(), false, 1}); // read scratch
        accesses.push_back({idma.getDstAddr(), idma.getSize(), true, 2});  // write local
      } else {
        accesses.push_back({idma.getSrcAddr(), idma.getSize(), false, 2}); // read local
        accesses.push_back({idma.getDstAddr(), idma.getSize(), true, 1});  // write scratch
      }
    } else if (auto compute = dyn_cast<xtpu::ComputeOp>(&op)) {
      uint32_t srcOff = compute.getSrcOffset();
      uint32_t dstOff = compute.getDstOffset();
      uint32_t len = compute.getLength();
      uint32_t srcLen = (compute.getType() == xtpu::ComputeType::matmul) ? 32 : len;
      accesses.push_back({srcOff, srcLen, false, 2}); // read local
      accesses.push_back({dstOff, len, true, 2});     // write local
    }
  }
  return accesses;
}

/// Check if merging two packets would cause a data hazard.
/// A hazard exists if one packet writes to a region that the other reads/writes
/// in the same memory space.
static bool hasDataHazard(xtpu::PacketOp a, xtpu::PacketOp b) {
  auto accessA = getMemAccesses(a);
  auto accessB = getMemAccesses(b);

  for (auto &aa : accessA) {
    for (auto &bb : accessB) {
      if (aa.space != bb.space) continue;
      // Check overlap
      uint64_t aEnd = aa.addr + aa.size;
      uint64_t bEnd = bb.addr + bb.size;
      if (aa.addr < bEnd && bb.addr < aEnd) {
        // Overlapping — hazard if either is a write
        if (aa.isWrite || bb.isWrite)
          return true;
      }
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

namespace {

struct XTPUSchedulePass
    : public mlir::xtpu::impl::XTPUScheduleBase<XTPUSchedulePass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();

    for (auto programOp :
         module.getBody()->getOps<xtpu::ProgramOp>()) {
      scheduleProgram(programOp);
    }
  }

  void scheduleProgram(xtpu::ProgramOp program) {
    int originalCount = 0;

    // Collect all packets
    SmallVector<xtpu::PacketOp> packets;
    for (auto packetOp :
         program.getBody().front().getOps<xtpu::PacketOp>()) {
      packets.push_back(packetOp);
      originalCount++;
    }

    if (packets.size() <= 1)
      return;

    // Greedy merge: try to merge packet[i+1] into packet[i]
    // Conditions for merge:
    // 1. No engine slot conflict
    // 2. packet[i+1]'s sync_mask only waits on engines used by packet[i]
    //    (i.e., the sync is just waiting for the previous packet to finish,
    //     which is implicit if they're in the same packet)
    // 3. No data hazard between the two packets' ops
    SmallVector<xtpu::PacketOp> mergedPackets;
    size_t i = 0;

    while (i < packets.size()) {
      xtpu::PacketOp current = packets[i];

      // Try to merge subsequent packets into current
      while (i + 1 < packets.size()) {
        xtpu::PacketOp next = packets[i + 1];

        // Skip empty sync-only packets at end (drain) — don't merge
        if (isEmptyPacket(next))
          break;

        uint8_t curSlots = getUsedSlots(current);
        uint8_t nextSlots = getUsedSlots(next);

        // Engine conflict?
        if (curSlots & nextSlots)
          break;

        // The next packet's sync_mask must only wait on engines that
        // the current packet uses. This means the sync is resolved
        // by being in the same VLIW word.
        uint8_t nextSync = getSyncSlots(next);
        if (nextSync & ~curSlots)
          break; // waits on something not in current → can't merge

        // Check data hazards
        if (hasDataHazard(current, next))
          break;

        // Safe to merge! Move ops from next into current.
        Block &curBody = current.getBody().front();
        Block &nextBody = next.getBody().front();

        // Collect ops to move (skip terminator)
        SmallVector<Operation *> opsToMove;
        for (auto &op : nextBody.without_terminator())
          opsToMove.push_back(&op);

        // Move ops before current's terminator
        for (auto *op : opsToMove)
          op->moveBefore(curBody.getTerminator());

        // Merge sync masks: keep current's sync_mask (it already has
        // the necessary waits for everything before it)
        // The next packet's sync was for current, which is now implicit.

        // Remove the empty next packet
        next.erase();
        packets.erase(packets.begin() + i + 1);
      }

      i++;
    }

    int finalCount = 0;
    for (auto _ : program.getBody().front().getOps<xtpu::PacketOp>()) {
      (void)_;
      finalCount++;
    }

    // Emit statistics
    int saved = originalCount - finalCount;
    double reduction = originalCount > 0
        ? (saved * 100.0 / originalCount) : 0.0;

    llvm::errs() << "\n=== xTPU Schedule Report: @"
                 << program.getSymName() << " ===\n";
    llvm::errs() << "Packets before: " << originalCount << "\n";
    llvm::errs() << "Packets after:  " << finalCount << "\n";
    llvm::errs() << "Reduction:      " << saved << " packets ("
                 << (int)reduction << "%)\n";

    // Engine utilization
    int sdmaSlots = 0, idmaSlots = 0, pu0Slots = 0, pu1Slots = 0;
    int emptySlots = 0;
    for (auto packetOp :
         program.getBody().front().getOps<xtpu::PacketOp>()) {
      uint8_t slots = getUsedSlots(packetOp);
      if (slots & SLOT_SDMA) sdmaSlots++;
      if (slots & SLOT_IDMA) idmaSlots++;
      if (slots & SLOT_PU0_CMD) pu0Slots++;
      if (slots & SLOT_PU1_CMD) pu1Slots++;
      if (slots == SLOT_NONE) emptySlots++;
    }

    llvm::errs() << "Engine utilization (slots used / total packets):\n";
    if (finalCount > 0) {
      llvm::errs() << "  SDMA:     " << sdmaSlots << "/" << finalCount
                   << " (" << (sdmaSlots * 100 / finalCount) << "%)\n";
      llvm::errs() << "  IDMA:     " << idmaSlots << "/" << finalCount
                   << " (" << (idmaSlots * 100 / finalCount) << "%)\n";
      llvm::errs() << "  PU0 Cmd:  " << pu0Slots << "/" << finalCount
                   << " (" << (pu0Slots * 100 / finalCount) << "%)\n";
      llvm::errs() << "  PU1 Cmd:  " << pu1Slots << "/" << finalCount
                   << " (" << (pu1Slots * 100 / finalCount) << "%)\n";
      llvm::errs() << "  Empty:    " << emptySlots << "/" << finalCount
                   << " (drain syncs)\n";
    }
    llvm::errs() << "=== End Report ===\n\n";
  }
};

} // namespace
