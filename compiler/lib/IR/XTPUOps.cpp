//===----------------------------------------------------------------------===//
// xTPU Op Verifiers
//
// Implements V-rules from docs/DialectSpec.md.
// Each verify method returns success() or emitOpError().
//===----------------------------------------------------------------------===//

#include "xtpu/IR/XTPUOps.h"
#include "mlir/IR/OpImplementation.h"
#include <unordered_set>

using namespace mlir;
using namespace mlir::xtpu;

// ---------------------------------------------------------------------------
// Hardware constants (must match simulator's common_types.hpp)
// ---------------------------------------------------------------------------
static constexpr uint64_t SYSTEM_MEMORY_SIZE = 16ULL * 1024 * 1024;  // 16 MB
static constexpr uint64_t SCRATCHPAD_SIZE    = 1ULL * 1024 * 1024;   // 1 MB
static constexpr uint64_t LOCAL_MEM_SIZE     = 64ULL * 1024;         // 64 KB

// ---------------------------------------------------------------------------
// Valid sync_mask string values
// ---------------------------------------------------------------------------
static const std::unordered_set<std::string> kValidSyncMasks = {
    "sdma", "pu0_dma", "pu0_cmd", "pu1_dma", "pu1_cmd"
};

// ---------------------------------------------------------------------------
// Custom parsers/printers for DMA direction keyword (load/store)
// Must be in mlir::xtpu namespace — the generated code calls them unqualified.
// ---------------------------------------------------------------------------
namespace mlir {
namespace xtpu {

static ParseResult parseDMAKeyword(OpAsmParser &parser,
                                   DMADirectionAttr &direction) {
  StringRef keyword;
  auto loc = parser.getCurrentLocation();
  if (parser.parseKeyword(&keyword))
    return failure();

  if (keyword == "load") {
    direction = DMADirectionAttr::get(parser.getContext(),
                                       DMADirection::to_device);
    return success();
  }
  if (keyword == "store") {
    direction = DMADirectionAttr::get(parser.getContext(),
                                       DMADirection::from_device);
    return success();
  }
  return parser.emitError(loc, "expected 'load' or 'store'");
}

static void printDMAKeyword(OpAsmPrinter &printer, Operation *,
                            DMADirectionAttr direction) {
  if (direction.getValue() == DMADirection::to_device)
    printer << "load";
  else
    printer << "store";
}

} // namespace xtpu
} // namespace mlir

// TableGen-generated enum definitions
#include "xtpu/IR/XTPUEnums.cpp.inc"

// TableGen-generated op definitions
#define GET_OP_CLASSES
#include "xtpu/IR/XTPUOps.cpp.inc"

// ---------------------------------------------------------------------------
// xtpu.program verifier
// ---------------------------------------------------------------------------
LogicalResult ProgramOp::verify() {
  // Body must contain at least one packet
  auto &block = getBody().front();
  bool hasPacket = false;
  for (auto &op : block.without_terminator()) {
    if (!isa<PacketOp>(op))
      return emitOpError("body must contain only 'xtpu.packet' ops, found '")
             << op.getName() << "'";
    hasPacket = true;
  }
  if (!hasPacket)
    return emitOpError("must contain at least one 'xtpu.packet'");
  return success();
}

// ---------------------------------------------------------------------------
// xtpu.packet verifier (V1, V2, V19)
// ---------------------------------------------------------------------------
LogicalResult PacketOp::verify() {
  // V2: Validate sync_mask entries
  for (auto attr : getSyncMask()) {
    auto str = dyn_cast<StringAttr>(attr);
    if (!str)
      return emitOpError("sync_mask entries must be strings");
    if (kValidSyncMasks.find(str.getValue().str()) == kValidSyncMasks.end())
      return emitOpError("invalid sync_mask value '")
             << str.getValue()
             << "', valid values: sdma, pu0_dma, pu0_cmd, pu1_dma, pu1_cmd";
  }

  // V1: No duplicate engine slots; V19: unique PU per packet
  bool hasSDMA = false;
  bool hasIDMA = false;
  std::unordered_set<int> puSeen;

  for (auto &op : getBody().front().without_terminator()) {
    if (isa<SDMAOp>(op)) {
      if (hasSDMA)
        return emitOpError("packet contains duplicate 'xtpu.sdma' ops (V1)");
      hasSDMA = true;
    } else if (isa<IDMAOp>(op)) {
      if (hasIDMA)
        return emitOpError("packet contains duplicate 'xtpu.idma' ops (V1)");
      hasIDMA = true;
    } else if (auto compute = dyn_cast<ComputeOp>(op)) {
      int pu = compute.getPu();
      if (!puSeen.insert(pu).second)
        return emitOpError("packet contains duplicate PU ")
               << pu << " compute ops (V19)";
    } else if (!isa<EndOp>(op)) {
      return emitOpError("packet body contains unexpected op '")
             << op.getName() << "'";
    }
  }

  return success();
}

// ---------------------------------------------------------------------------
// xtpu.sdma verifier (V4, V5, V6)
// ---------------------------------------------------------------------------
LogicalResult SDMAOp::verify() {
  uint64_t sz = getSize();
  uint64_t src = getSrcAddr();
  uint64_t dst = getDstAddr();

  // V4: size > 0
  if (sz == 0)
    return emitOpError("size must be > 0 (V4)");

  if (getDirection() == DMADirection::to_device) {
    // V5: load bounds
    if (src > SYSTEM_MEMORY_SIZE || sz > SYSTEM_MEMORY_SIZE - src)
      return emitOpError("load src_addr + size exceeds SYSTEM_MEMORY_SIZE (")
             << SYSTEM_MEMORY_SIZE << ") (V5)";
    if (dst > SCRATCHPAD_SIZE || sz > SCRATCHPAD_SIZE - dst)
      return emitOpError("load dst_addr + size exceeds SCRATCHPAD_SIZE (")
             << SCRATCHPAD_SIZE << ") (V5)";
  } else {
    // V6: store bounds
    if (src > SCRATCHPAD_SIZE || sz > SCRATCHPAD_SIZE - src)
      return emitOpError("store src_addr + size exceeds SCRATCHPAD_SIZE (")
             << SCRATCHPAD_SIZE << ") (V6)";
    if (dst > SYSTEM_MEMORY_SIZE || sz > SYSTEM_MEMORY_SIZE - dst)
      return emitOpError("store dst_addr + size exceeds SYSTEM_MEMORY_SIZE (")
             << SYSTEM_MEMORY_SIZE << ") (V6)";
  }

  return success();
}

// ---------------------------------------------------------------------------
// xtpu.idma verifier (V7, V8, V9, V10, V11)
// ---------------------------------------------------------------------------
LogicalResult IDMAOp::verify() {
  uint64_t sz = getSize();
  uint64_t src = getSrcAddr();
  uint64_t dst = getDstAddr();
  int buf = getBuffer();

  // V7: size > 0
  if (sz == 0)
    return emitOpError("size must be > 0 (V7)");

  // V8: buffer in {0, 1}
  if (buf < 0 || buf > 1)
    return emitOpError("buffer must be 0 or 1, got ") << buf << " (V8)";

  // V11: broadcast store is illegal
  if (getTarget() == Target::pu01 &&
      getDirection() == DMADirection::from_device)
    return emitOpError("broadcast (pu01) is only valid for load, "
                       "not store (V11)");

  if (getDirection() == DMADirection::to_device) {
    // V9: load bounds
    if (src > SCRATCHPAD_SIZE || sz > SCRATCHPAD_SIZE - src)
      return emitOpError("load src_addr + size exceeds SCRATCHPAD_SIZE (V9)");
    if (dst > LOCAL_MEM_SIZE || sz > LOCAL_MEM_SIZE - dst)
      return emitOpError("load dst_addr + size exceeds LOCAL_MEM_SIZE (V9)");
  } else {
    // V10: store bounds
    if (src > LOCAL_MEM_SIZE || sz > LOCAL_MEM_SIZE - src)
      return emitOpError("store src_addr + size exceeds LOCAL_MEM_SIZE (V10)");
    if (dst > SCRATCHPAD_SIZE || sz > SCRATCHPAD_SIZE - dst)
      return emitOpError("store dst_addr + size exceeds SCRATCHPAD_SIZE (V10)");
  }

  return success();
}

// ---------------------------------------------------------------------------
// xtpu.compute verifier (V12-V18)
// ---------------------------------------------------------------------------
LogicalResult ComputeOp::verify() {
  int pu = getPu();
  int buf = getBuffer();
  uint32_t srcOff = getSrcOffset();
  uint32_t dstOff = getDstOffset();
  uint32_t len = getLength();

  // V12: pu in {0, 1}
  if (pu < 0 || pu > 1)
    return emitOpError("pu must be 0 or 1, got ") << pu << " (V12)";

  // V13: buffer in {0, 1}
  if (buf < 0 || buf > 1)
    return emitOpError("buffer must be 0 or 1, got ") << buf << " (V13)";

  // V14: length > 0
  if (len == 0)
    return emitOpError("length must be > 0 (V14)");

  if (getType() == ComputeType::matmul) {
    // V15: MATMUL requires length == 16
    if (len != 16)
      return emitOpError("matmul requires length == 16, got ")
             << len << " (V15)";

    // V16: src_offset + 32 <= LOCAL_MEM_SIZE (reads A + B)
    if (static_cast<uint64_t>(srcOff) + 32 > LOCAL_MEM_SIZE)
      return emitOpError("matmul src_offset + 32 exceeds LOCAL_MEM_SIZE (V16)");
  } else {
    // V17: scalar/vector: src_offset + length <= LOCAL_MEM_SIZE
    if (static_cast<uint64_t>(srcOff) + len > LOCAL_MEM_SIZE)
      return emitOpError("src_offset + length exceeds LOCAL_MEM_SIZE (V17)");
  }

  // V18: dst_offset + length <= LOCAL_MEM_SIZE
  if (static_cast<uint64_t>(dstOff) + len > LOCAL_MEM_SIZE)
    return emitOpError("dst_offset + length exceeds LOCAL_MEM_SIZE (V18)");

  return success();
}
