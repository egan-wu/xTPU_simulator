#!/usr/bin/env python3
"""
xtpu-dump — P5-9: Compilation visualization and debugging tool.

Reads a .xbin file and produces human-readable diagnostic reports:

  --schedule   Engine occupancy Gantt chart (ASCII)
  --memory     Scratchpad / LocalMem / SystemMem allocation map
  --hazards    Synchronization barrier analysis
  --stats      Total ticks, engine utilization, spill statistics
  --all        All of the above (default)

Usage:
  python3 xtpu_dump.py model.xbin
  python3 xtpu_dump.py model.xbin --schedule
  python3 xtpu_dump.py model.xbin --stats
"""

import argparse
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# .xbin format constants (matching xbin_loader.hpp)
# ---------------------------------------------------------------------------
XBIN_MAGIC = b'XTPU'
DMA_NOP = 0
DMA_MEMCPY = 1
DIR_TO_DEVICE = 0
DIR_FROM_DEVICE = 1

COMPUTE_NOP = 0
COMPUTE_MATMUL = 1
COMPUTE_VECTOR = 2
COMPUTE_SCALAR = 3

COMPUTE_NAMES = {
    0: "NOP", 1: "MATMUL", 2: "VECTOR", 3: "SCALAR",
    4: "ADD", 5: "MUL", 6: "SUB", 7: "RELU",
    8: "MAX", 9: "REDUCE_SUM", 10: "REDUCE_MAX",
}

# Sync mask bits
STATUS_SDMA_BUSY    = 0x01
STATUS_PU0_DMA_BUSY = 0x02
STATUS_PU0_CMD_BUSY = 0x04
STATUS_PU1_DMA_BUSY = 0x08
STATUS_PU1_CMD_BUSY = 0x10

SYNC_NAMES = {
    STATUS_SDMA_BUSY:    "SDMA",
    STATUS_PU0_DMA_BUSY: "PU0_DMA",
    STATUS_PU0_CMD_BUSY: "PU0_CMD",
    STATUS_PU1_DMA_BUSY: "PU1_DMA",
    STATUS_PU1_CMD_BUSY: "PU1_CMD",
}

# Timing estimates (cycles per operation — simplified model)
SDMA_LATENCY = 10     # System DMA cycles per transfer
IDMA_LATENCY = 5      # Internal DMA cycles per transfer
MATMUL_LATENCY = 20   # 4x4 matmul cycles
SCALAR_LATENCY = 4    # Element-wise scalar op cycles
VECTOR_LATENCY = 4    # Element-wise vector op cycles


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class DMACmd:
    type: int = 0
    direction: int = 0
    src_addr: int = 0
    dst_addr: int = 0
    size: int = 0
    target_mask: int = 0
    buffer_idx: int = 0

    @property
    def is_active(self) -> bool:
        return self.type != DMA_NOP

    @property
    def dir_str(self) -> str:
        return "LOAD" if self.direction == DIR_TO_DEVICE else "STORE"


@dataclass
class ComputeCmd:
    type: int = 0
    buffer_idx: int = 0
    simulated_duration: int = 0
    src_offset: int = 0
    dst_offset: int = 0
    length: int = 0

    @property
    def is_active(self) -> bool:
        return self.type != COMPUTE_NOP

    @property
    def type_name(self) -> str:
        return COMPUTE_NAMES.get(self.type, f"UNK({self.type})")


@dataclass
class Packet:
    idx: int
    sdma: DMACmd
    idma: DMACmd
    pu0: ComputeCmd
    pu1: ComputeCmd
    sync_mask: int


def sync_mask_str(mask: int) -> str:
    parts = []
    for bit, name in SYNC_NAMES.items():
        if mask & bit:
            parts.append(name)
    return "+".join(parts) if parts else "NONE"


# ---------------------------------------------------------------------------
# .xbin parser
# ---------------------------------------------------------------------------
def load_xbin(path: str) -> Tuple[List[Packet], dict]:
    """Load .xbin and return packets + metadata."""
    with open(path, 'rb') as f:
        data = f.read()

    assert data[:4] == XBIN_MAGIC, f"Bad magic: {data[:4]}"
    version, num_sections = struct.unpack_from("<HH", data, 4)
    entry_offset, flags = struct.unpack_from("<II", data, 8)

    # Find .text section
    text_offset = None
    meta_offset = None
    rodata_offset = None

    for i in range(num_sections):
        off = 32 + i * 8
        sec_type, sec_flags, sec_off = struct.unpack_from("<HHI", data, off)
        if sec_type == 0:
            text_offset = sec_off
        elif sec_type == 1:
            rodata_offset = sec_off
        elif sec_type == 2:
            meta_offset = sec_off

    packets = []
    if text_offset is not None:
        off = text_offset
        num_pkts = struct.unpack_from("<I", data, off)[0]
        off += 4

        for i in range(num_pkts):
            # DMA: 40 bytes each
            sdma = DMACmd(*struct.unpack_from("<II QQQ Ii", data, off))
            off += 40
            idma = DMACmd(*struct.unpack_from("<II QQQ Ii", data, off))
            off += 40
            # Compute: 28 bytes each (P5-11: +src2_offset)
            vals = struct.unpack_from("<iI IIIII", data, off)
            pu0 = ComputeCmd(*vals[:6])
            off += 28
            vals = struct.unpack_from("<iI IIIII", data, off)
            pu1 = ComputeCmd(*vals[:6])
            off += 28
            # sync_mask: 4 bytes
            sync_mask = struct.unpack_from("<I", data, off)[0]
            off += 4

            packets.append(Packet(i, sdma, idma, pu0, pu1, sync_mask))

    meta = {"version": version, "flags": flags, "file_size": len(data),
            "num_packets": len(packets)}
    if rodata_offset:
        num_entries = struct.unpack_from("<I", data, rodata_offset)[0]
        meta["rodata_entries"] = num_entries

    return packets, meta


# ---------------------------------------------------------------------------
# --schedule: Engine Gantt chart
# ---------------------------------------------------------------------------
def dump_schedule(packets: List[Packet]):
    """ASCII Gantt chart showing engine utilization per packet."""
    print("\n=== Engine Schedule (Gantt) ===")
    print()

    # Header
    w = 10  # column width
    print(f"{'PKT':>4} | {'SYNC':>12} | {'SDMA':^{w}} | {'IDMA':^{w}} | {'PU0':^{w}} | {'PU1':^{w}} |")
    print(f"{'----':>4}-+-{'------------':>12}-+-{'-'*w}-+-{'-'*w}-+-{'-'*w}-+-{'-'*w}-+")

    for pkt in packets:
        sync = sync_mask_str(pkt.sync_mask)

        # SDMA
        if pkt.sdma.is_active:
            sdma_str = f"{pkt.sdma.dir_str} {pkt.sdma.size}B"
        else:
            sdma_str = "."

        # IDMA
        if pkt.idma.is_active:
            idma_str = f"{pkt.idma.dir_str} {pkt.idma.size}B"
        else:
            idma_str = "."

        # PU0
        if pkt.pu0.is_active:
            pu0_str = f"{pkt.pu0.type_name} {pkt.pu0.length}B"
        else:
            pu0_str = "."

        # PU1
        if pkt.pu1.is_active:
            pu1_str = f"{pkt.pu1.type_name} {pkt.pu1.length}B"
        else:
            pu1_str = "."

        print(f"{pkt.idx:>4} | {sync:>12} | {sdma_str:^{w}} | {idma_str:^{w}} | {pu0_str:^{w}} | {pu1_str:^{w}} |")

    # ASCII gantt bars
    print()
    print("Engine timeline (each char = 1 packet slot):")
    engines = {"SDMA": [], "IDMA": [], "PU0 ": [], "PU1 ": []}
    for pkt in packets:
        engines["SDMA"].append('#' if pkt.sdma.is_active else '-')
        engines["IDMA"].append('#' if pkt.idma.is_active else '-')
        engines["PU0 "].append('#' if pkt.pu0.is_active else '-')
        engines["PU1 "].append('#' if pkt.pu1.is_active else '-')

    for name, timeline in engines.items():
        bar = ''.join(timeline)
        active = sum(1 for c in timeline if c == '#')
        pct = active / len(timeline) * 100 if timeline else 0
        print(f"  {name}: [{bar}] {pct:.0f}%")


# ---------------------------------------------------------------------------
# --memory: Memory allocation map
# ---------------------------------------------------------------------------
def dump_memory(packets: List[Packet]):
    """Show memory access patterns across all packets."""
    print("\n=== Memory Access Map ===")
    print()

    sys_reads = []   # (addr, size)
    sys_writes = []
    scratch_reads = []
    scratch_writes = []
    local_reads = []
    local_writes = []

    for pkt in packets:
        if pkt.sdma.is_active:
            if pkt.sdma.direction == DIR_TO_DEVICE:
                sys_reads.append((pkt.sdma.src_addr, pkt.sdma.size))
                scratch_writes.append((pkt.sdma.dst_addr, pkt.sdma.size))
            else:
                scratch_reads.append((pkt.sdma.src_addr, pkt.sdma.size))
                sys_writes.append((pkt.sdma.dst_addr, pkt.sdma.size))

        if pkt.idma.is_active:
            if pkt.idma.direction == DIR_TO_DEVICE:
                scratch_reads.append((pkt.idma.src_addr, pkt.idma.size))
                local_writes.append((pkt.idma.dst_addr, pkt.idma.size))
            else:
                local_reads.append((pkt.idma.src_addr, pkt.idma.size))
                scratch_writes.append((pkt.idma.dst_addr, pkt.idma.size))

    def show_region(name: str, reads, writes, capacity: int):
        all_addrs = reads + writes
        if not all_addrs:
            print(f"  {name}: (unused)")
            return
        min_addr = min(a for a, _ in all_addrs)
        max_end = max(a + s for a, s in all_addrs)
        hwm = max_end
        print(f"  {name}:")
        print(f"    Range: 0x{min_addr:04x} - 0x{max_end:04x} ({max_end} bytes)")
        print(f"    High-water mark: {hwm} / {capacity} bytes ({hwm/capacity*100:.1f}%)")
        print(f"    Reads:  {len(reads)} transfers, {sum(s for _, s in reads)} bytes total")
        print(f"    Writes: {len(writes)} transfers, {sum(s for _, s in writes)} bytes total")

    show_region("System Memory (16 MB)", sys_reads, sys_writes, 16 * 1024 * 1024)
    show_region("Scratchpad (1 MB)", scratch_reads, scratch_writes, 1024 * 1024)
    show_region("LocalMem (64 KB)", local_reads, local_writes, 64 * 1024)

    # Output region detection
    output_writes = [(a, s) for a, s in sys_writes if a >= 4096]
    if output_writes:
        out_addr = min(a for a, _ in output_writes)
        out_size = sum(s for _, s in output_writes)
        print(f"\n  Output region: 0x{out_addr:04x}, {out_size} bytes")


# ---------------------------------------------------------------------------
# --hazards: Synchronization analysis
# ---------------------------------------------------------------------------
def dump_hazards(packets: List[Packet]):
    """Analyze synchronization barriers and potential hazards."""
    print("\n=== Synchronization Analysis ===")
    print()

    sync_count = 0
    full_sync_count = 0
    sync_types = {}

    for pkt in packets:
        if pkt.sync_mask != 0:
            sync_count += 1
            mask_str = sync_mask_str(pkt.sync_mask)
            sync_types[mask_str] = sync_types.get(mask_str, 0) + 1

            # Full sync = waiting on all engines
            bits = bin(pkt.sync_mask).count('1')
            if bits >= 3:
                full_sync_count += 1

    total = len(packets)
    print(f"  Total packets:      {total}")
    print(f"  Sync barriers:      {sync_count} ({sync_count/total*100:.0f}% of packets)")
    print(f"  Full syncs (3+ engines): {full_sync_count}")
    print()

    if sync_types:
        print("  Sync type breakdown:")
        for mask_str, count in sorted(sync_types.items(), key=lambda x: -x[1]):
            print(f"    {mask_str:30s}: {count:3d} times")

    # Detect back-to-back syncs (inefficient — could be merged)
    b2b = 0
    for i in range(1, len(packets)):
        if packets[i].sync_mask != 0 and packets[i-1].sync_mask != 0:
            # Check if current packet is empty (pure sync)
            p = packets[i]
            if not p.sdma.is_active and not p.idma.is_active and \
               not p.pu0.is_active and not p.pu1.is_active:
                b2b += 1

    if b2b > 0:
        print(f"\n  Empty sync packets (optimization opportunity): {b2b}")


# ---------------------------------------------------------------------------
# --stats: Summary statistics
# ---------------------------------------------------------------------------
def dump_stats(packets: List[Packet], meta: dict):
    """Performance statistics and estimates."""
    print("\n=== Compilation Statistics ===")
    print()

    # Basic counts
    total = len(packets)
    sdma_count = sum(1 for p in packets if p.sdma.is_active)
    idma_count = sum(1 for p in packets if p.idma.is_active)
    pu0_count = sum(1 for p in packets if p.pu0.is_active)
    pu1_count = sum(1 for p in packets if p.pu1.is_active)

    # Compute type breakdown
    compute_types = {}
    for p in packets:
        if p.pu0.is_active:
            name = p.pu0.type_name
            compute_types[name] = compute_types.get(name, 0) + 1
        if p.pu1.is_active:
            name = p.pu1.type_name
            compute_types[name] = compute_types.get(name, 0) + 1

    print(f"  File size:    {meta.get('file_size', 0):,} bytes")
    print(f"  Total packets: {total}")
    print(f"  .rodata entries: {meta.get('rodata_entries', 0)}")
    print()

    print("  Engine usage:")
    print(f"    SDMA:  {sdma_count:3d} packets ({sdma_count/total*100:.0f}%)")
    print(f"    IDMA:  {idma_count:3d} packets ({idma_count/total*100:.0f}%)")
    print(f"    PU0:   {pu0_count:3d} packets ({pu0_count/total*100:.0f}%)")
    print(f"    PU1:   {pu1_count:3d} packets ({pu1_count/total*100:.0f}%)")

    if compute_types:
        print()
        print("  Compute operations:")
        for name, count in sorted(compute_types.items()):
            print(f"    {name:10s}: {count}")

    # DMA transfer stats
    total_sdma_bytes = sum(p.sdma.size for p in packets if p.sdma.is_active)
    total_idma_bytes = sum(p.idma.size for p in packets if p.idma.is_active)

    print()
    print("  Data movement:")
    print(f"    SDMA total: {total_sdma_bytes:,} bytes ({total_sdma_bytes/1024:.1f} KB)")
    print(f"    IDMA total: {total_idma_bytes:,} bytes ({total_idma_bytes/1024:.1f} KB)")

    # Estimated cycle count (serial execution)
    est_cycles = 0
    for p in packets:
        pkt_cycles = 0
        if p.sdma.is_active:
            pkt_cycles = max(pkt_cycles, SDMA_LATENCY)
        if p.idma.is_active:
            pkt_cycles = max(pkt_cycles, IDMA_LATENCY)
        if p.pu0.is_active:
            lat = {COMPUTE_MATMUL: MATMUL_LATENCY,
                   COMPUTE_SCALAR: SCALAR_LATENCY,
                   COMPUTE_VECTOR: VECTOR_LATENCY}.get(p.pu0.type, 1)
            pkt_cycles = max(pkt_cycles, lat)
        est_cycles += pkt_cycles

    print()
    print("  Estimated execution:")
    print(f"    Serial cycles:  ~{est_cycles} (assuming no overlap)")
    print(f"    Compute/total:  {pu0_count + pu1_count}/{total} packets = "
          f"{(pu0_count + pu1_count)/total*100:.0f}% compute density")

    # Parallelism score (how many engines are active per packet on average)
    active_sum = sum(
        int(p.sdma.is_active) + int(p.idma.is_active) +
        int(p.pu0.is_active) + int(p.pu1.is_active)
        for p in packets
    )
    print(f"    Avg engines/pkt: {active_sum/total:.2f} (max 4)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="xtpu-dump: Compilation visualization & debug (P5-9)")
    parser.add_argument("input", help=".xbin file to analyze")
    parser.add_argument("--schedule", action="store_true",
                        help="Show engine Gantt chart")
    parser.add_argument("--memory", action="store_true",
                        help="Show memory allocation map")
    parser.add_argument("--hazards", action="store_true",
                        help="Show synchronization analysis")
    parser.add_argument("--stats", action="store_true",
                        help="Show compilation statistics")
    args = parser.parse_args()

    # Default: show all
    show_all = not (args.schedule or args.memory or args.hazards or args.stats)

    path = args.input
    if not Path(path).exists():
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(1)

    packets, meta = load_xbin(path)
    print(f"=== xtpu-dump: {Path(path).name} ===")
    print(f"    {meta['num_packets']} packets, {meta['file_size']:,} bytes")

    if show_all or args.stats:
        dump_stats(packets, meta)
    if show_all or args.schedule:
        dump_schedule(packets)
    if show_all or args.memory:
        dump_memory(packets)
    if show_all or args.hazards:
        dump_hazards(packets)


if __name__ == "__main__":
    main()
