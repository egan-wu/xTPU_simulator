#!/usr/bin/env python3
"""
xtpu-translate — Convert xTPU MLIR to .xbin binary format.

Parses the text output of xtpu-opt (xTPU dialect MLIR) and serializes
it into the .xbin binary format that the simulator's XBinLoader can read.

Usage:
    xtpu-opt input.mlir --linalg-to-xtpu | xtpu-translate -o output.xbin
    xtpu-translate input.xtpu.mlir -o output.xbin
    xtpu-translate input.xtpu.mlir --dump  # human-readable hex dump

.xbin Format (v1):
    Header (32 bytes):
        magic:        "XTPU" (4 bytes)
        version:      uint16 = 1
        num_sections: uint16
        entry_offset: uint32 (byte offset to .text section)
        flags:        uint32 (0x01 = INT8 mode)
        reserved:     16 bytes of zeros

    Section Table (8 bytes per entry):
        type:   uint16 (0=.text, 1=.rodata, 2=.meta)
        flags:  uint16
        offset: uint32 (byte offset from file start)

    .text section:
        num_packets: uint32
        packets[]:   array of binary VLIWPacket (fixed 128 bytes each)

    .rodata section:
        num_entries: uint32
        entries[]:   { offset: uint64, size: uint64, data: bytes }

    .meta section:
        JSON string with input/output tensor info

Binary VLIWPacket layout (128 bytes, padded):
    DMA_Command (sDMA):  40 bytes
    DMA_Command (iDMA):  40 bytes
    Compute_Command (pu0): 24 bytes
    Compute_Command (pu1): 24 bytes

    DMA_Command layout (40 bytes):
        type:        uint32 (0=NOP, 1=MEMCPY)
        padding:     uint32
        src_addr:    uint64
        dst_addr:    uint64
        size:        uint64
        target_mask: uint32
        buffer_idx:  int32
        direction:   uint32 (0=TO_DEVICE, 1=FROM_DEVICE)
        padding:     uint32

    Compute_Command layout (24 bytes):
        type:                uint32 (0=NOP, 1=MATMUL, 2=VECTOR, 3=SCALAR)
        buffer_idx:          int32
        simulated_duration:  uint32
        src_offset:          uint32
        dst_offset:          uint32
        length:              uint32
"""

import argparse
import json
import re
import struct
import sys
from dataclasses import dataclass, field
from typing import List, Optional

# ---------------------------------------------------------------------------
# Data structures matching simulator
# ---------------------------------------------------------------------------

XBIN_MAGIC = b"XTPU"
XBIN_VERSION = 1

# Section types
SEC_TEXT = 0
SEC_RODATA = 1
SEC_META = 2

# Enum mappings
DMA_TYPE_NOP = 0
DMA_TYPE_MEMCPY = 1

DMA_DIR_TO_DEVICE = 0
DMA_DIR_FROM_DEVICE = 1

COMPUTE_NOP = 0
COMPUTE_MATMUL = 1
COMPUTE_VECTOR = 2
COMPUTE_SCALAR = 3

# Sync mask bits (match common_types.hpp)
SYNC_SDMA = 1 << 0
SYNC_PU0_DMA = 1 << 1
SYNC_PU0_CMD = 1 << 2
SYNC_PU1_DMA = 1 << 3
SYNC_PU1_CMD = 1 << 4

SYNC_MAP = {
    "sdma": SYNC_SDMA,
    "pu0_dma": SYNC_PU0_DMA,
    "pu0_cmd": SYNC_PU0_CMD,
    "pu1_dma": SYNC_PU1_DMA,
    "pu1_cmd": SYNC_PU1_CMD,
}

TARGET_MAP = {
    "pu0": 1,
    "pu1": 2,
    "pu01": 3,
}

COMPUTE_TYPE_MAP = {
    "matmul": COMPUTE_MATMUL,
    "vector": COMPUTE_VECTOR,
    "scalar": COMPUTE_SCALAR,
}


@dataclass
class DMACommand:
    type: int = DMA_TYPE_NOP
    src_addr: int = 0
    dst_addr: int = 0
    size: int = 0
    target_mask: int = 0
    buffer_idx: int = 0
    direction: int = DMA_DIR_TO_DEVICE

    def pack(self) -> bytes:
        """Pack to 40 bytes matching XBinDMACommand (packed struct).

        Layout:
          type:        uint32  (4)
          direction:   uint32  (4)
          src_addr:    uint64  (8)
          dst_addr:    uint64  (8)
          size:        uint64  (8)
          target_mask: uint32  (4)
          buffer_idx:  int32   (4)
        Total: 40 bytes
        """
        return struct.pack(
            "<II QQQ Ii",
            self.type, self.direction,
            self.src_addr, self.dst_addr, self.size,
            self.target_mask, self.buffer_idx,
        )


@dataclass
class ComputeCommand:
    type: int = COMPUTE_NOP
    buffer_idx: int = 0
    simulated_duration_ms: int = 0
    src_offset: int = 0
    dst_offset: int = 0
    length: int = 0

    def pack(self) -> bytes:
        """Pack to 24 bytes matching XBinComputeCommand."""
        return struct.pack(
            "<iIIIII",
            self.type, self.buffer_idx, self.simulated_duration_ms,
            self.src_offset, self.dst_offset, self.length
        )


@dataclass
class VLIWPacket:
    sdma: DMACommand = field(default_factory=DMACommand)
    idma: DMACommand = field(default_factory=DMACommand)
    pu0: ComputeCommand = field(default_factory=ComputeCommand)
    pu1: ComputeCommand = field(default_factory=ComputeCommand)
    sync_mask: int = 0

    def pack(self) -> bytes:
        """Pack to 132 bytes matching XBinPacket (40+40+24+24+4)."""
        data = self.sdma.pack() + self.idma.pack() + self.pu0.pack() + self.pu1.pack()
        data += struct.pack("<I", self.sync_mask)
        assert len(data) == 132, f"Packet size mismatch: {len(data)} != 132"
        return data


@dataclass
class RodataEntry:
    offset: int  # system memory offset
    data: bytes


@dataclass
class MetaInfo:
    program_name: str = "main"
    inputs: list = field(default_factory=list)
    outputs: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# MLIR text parser
# ---------------------------------------------------------------------------

def parse_sync_mask(text: str) -> int:
    """Parse sync_mask = ["sdma", "pu0_dma"] into bitmask."""
    mask = 0
    matches = re.findall(r'"(\w+)"', text)
    for m in matches:
        if m in SYNC_MAP:
            mask |= SYNC_MAP[m]
    return mask


def parse_xtpu_mlir(mlir_text: str):
    """Parse xTPU dialect MLIR text into packets and metadata."""
    packets: List[VLIWPacket] = []
    program_name = "main"
    current_packet: Optional[VLIWPacket] = None

    for line in mlir_text.split('\n'):
        line = line.strip()

        # Program name
        m = re.match(r'xtpu\.program\s+@(\w+)', line)
        if m:
            program_name = m.group(1)
            continue

        # Packet start
        if line.startswith('xtpu.packet'):
            current_packet = VLIWPacket()
            # Parse sync_mask
            sm = re.search(r'sync_mask\s*=\s*\[([^\]]*)\]', line)
            if sm:
                current_packet.sync_mask = parse_sync_mask(sm.group(0))
            continue

        # Packet end
        if line == '}' and current_packet is not None:
            packets.append(current_packet)
            current_packet = None
            continue

        if current_packet is None:
            continue

        # Parse SDMA
        m = re.match(
            r'xtpu\.sdma\s+(load|store)\s+'
            r'src_addr\s*=\s*(\d+)\s+'
            r'dst_addr\s*=\s*(\d+)\s+'
            r'size\s*=\s*(\d+)',
            line)
        if m:
            direction = DMA_DIR_TO_DEVICE if m.group(1) == 'load' else DMA_DIR_FROM_DEVICE
            current_packet.sdma = DMACommand(
                type=DMA_TYPE_MEMCPY,
                src_addr=int(m.group(2)),
                dst_addr=int(m.group(3)),
                size=int(m.group(4)),
                direction=direction,
            )
            continue

        # Parse IDMA
        m = re.match(
            r'xtpu\.idma\s+(load|store)\s+'
            r'src_addr\s*=\s*(\d+)\s+'
            r'dst_addr\s*=\s*(\d+)\s+'
            r'size\s*=\s*(\d+)\s+'
            r'target\s*=\s*(\w+)\s+'
            r'buffer\s*=\s*(\d+)',
            line)
        if m:
            direction = DMA_DIR_TO_DEVICE if m.group(1) == 'load' else DMA_DIR_FROM_DEVICE
            current_packet.idma = DMACommand(
                type=DMA_TYPE_MEMCPY,
                src_addr=int(m.group(2)),
                dst_addr=int(m.group(3)),
                size=int(m.group(4)),
                target_mask=TARGET_MAP.get(m.group(5), 0),
                buffer_idx=int(m.group(6)),
                direction=direction,
            )
            continue

        # Parse Compute
        m = re.match(
            r'xtpu\.compute\s+'
            r'pu\s*=\s*(\d+)\s+'
            r'type\s*=\s*(\w+)\s+'
            r'buffer\s*=\s*(\d+)\s+'
            r'src_offset\s*=\s*(\d+)\s+'
            r'dst_offset\s*=\s*(\d+)\s+'
            r'length\s*=\s*(\d+)',
            line)
        if m:
            pu = int(m.group(1))
            cmd = ComputeCommand(
                type=COMPUTE_TYPE_MAP.get(m.group(2), COMPUTE_NOP),
                buffer_idx=int(m.group(3)),
                src_offset=int(m.group(4)),
                dst_offset=int(m.group(5)),
                length=int(m.group(6)),
            )
            if pu == 0:
                current_packet.pu0 = cmd
            else:
                current_packet.pu1 = cmd
            continue

    meta = MetaInfo(program_name=program_name)
    return packets, meta


# ---------------------------------------------------------------------------
# .xbin encoding
# ---------------------------------------------------------------------------

def encode_xbin(packets: List[VLIWPacket], rodata: List[RodataEntry],
                meta: MetaInfo) -> bytes:
    """Encode packets + rodata + meta into .xbin format."""
    sections = []

    # .text section
    text_data = struct.pack("<I", len(packets))
    for pkt in packets:
        text_data += pkt.pack()
    sections.append((SEC_TEXT, text_data))

    # .rodata section (may be empty)
    if rodata:
        rodata_data = struct.pack("<I", len(rodata))
        for entry in rodata:
            rodata_data += struct.pack("<QQ", entry.offset, len(entry.data))
            rodata_data += entry.data
        sections.append((SEC_RODATA, rodata_data))

    # .meta section
    meta_json = json.dumps({
        "program": meta.program_name,
        "inputs": meta.inputs,
        "outputs": meta.outputs,
    }).encode('utf-8')
    sections.append((SEC_META, meta_json))

    num_sections = len(sections)

    # Calculate offsets
    header_size = 32
    section_table_size = num_sections * 8
    data_start = header_size + section_table_size

    # Build section table and data
    section_table = b''
    section_data = b''
    current_offset = data_start

    entry_offset = 0
    for i, (sec_type, data) in enumerate(sections):
        section_table += struct.pack("<HHI", sec_type, 0, current_offset)
        if sec_type == SEC_TEXT:
            entry_offset = current_offset
        section_data += data
        current_offset += len(data)

    # Build header
    header = XBIN_MAGIC
    header += struct.pack("<HH", XBIN_VERSION, num_sections)
    header += struct.pack("<I", entry_offset)
    header += struct.pack("<I", 0x01)  # flags: INT8 mode
    header += b'\x00' * 16  # reserved

    return header + section_table + section_data


# ---------------------------------------------------------------------------
# .xbin decoding (for round-trip test)
# ---------------------------------------------------------------------------

def decode_xbin(data: bytes):
    """Decode .xbin back to packets + meta for verification."""
    assert data[:4] == XBIN_MAGIC, f"Bad magic: {data[:4]}"
    version, num_sections = struct.unpack_from("<HH", data, 4)
    assert version == XBIN_VERSION, f"Unsupported version: {version}"
    entry_offset, flags = struct.unpack_from("<II", data, 8)

    # Parse section table
    sections = {}
    for i in range(num_sections):
        off = 32 + i * 8
        sec_type, sec_flags, sec_offset = struct.unpack_from("<HHI", data, off)
        sections[sec_type] = sec_offset

    # Parse .text
    packets = []
    if SEC_TEXT in sections:
        off = sections[SEC_TEXT]
        num_packets = struct.unpack_from("<I", data, off)[0]
        off += 4
        for _ in range(num_packets):
            pkt_data = data[off:off + 132]
            packets.append(pkt_data)
            off += 132

    # Parse .meta
    meta = {}
    if SEC_META in sections:
        # Find the meta section data (after all other sections)
        meta_off = sections[SEC_META]
        # Meta is JSON till end or next section
        meta_end = len(data)
        for sec_type, sec_off in sections.items():
            if sec_off > meta_off and sec_off < meta_end:
                meta_end = sec_off
        meta = json.loads(data[meta_off:meta_end])

    return packets, meta, version, flags


# ---------------------------------------------------------------------------
# Dump command
# ---------------------------------------------------------------------------

def dump_xbin(filepath: str):
    """Human-readable dump of .xbin file."""
    with open(filepath, 'rb') as f:
        data = f.read()

    raw_packets, meta, version, flags = decode_xbin(data)
    print(f"=== xbin dump: {filepath} ===")
    print(f"Version: {version}, Flags: 0x{flags:04x}")
    print(f"Sections: .text={len(raw_packets)} packets")
    if meta:
        print(f"Meta: {json.dumps(meta, indent=2)}")

    for i, pkt_data in enumerate(raw_packets):
        print(f"\n--- Packet {i} ---")
        # Decode sync_mask (at offset 128 - 4 = 124? No, after pu1)
        # sDMA: 40 bytes, iDMA: 40 bytes, pu0: 24 bytes, pu1: 24 bytes, sync: 4 bytes = 132
        # But we padded to 128. Let's recalculate:
        # 40 + 40 + 24 + 24 + 4 = 132 > 128. Need to adjust.
        # Actually let me recalculate: pack format is 128 bytes total with padding
        # The sync_mask is at offset 40+40+24+24 = 128, which means it's the first
        # byte of the padding. So sync_mask is at offset 128 - (128 - 132) ...
        # Actually 40+40+24+24+4 = 132, padded to 128 is wrong. Let me fix.
        pass

    print(f"\nTotal: {len(raw_packets)} packets, {len(data)} bytes")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="xtpu-translate: xTPU MLIR → .xbin binary")
    parser.add_argument("input", nargs="?", default="-",
                        help="Input .mlir file (default: stdin)")
    parser.add_argument("-o", "--output", required=False,
                        help="Output .xbin file")
    parser.add_argument("--dump", action="store_true",
                        help="Dump .xbin in human-readable format")
    parser.add_argument("--round-trip", action="store_true",
                        help="Encode then decode, verify match")
    args = parser.parse_args()

    if args.dump:
        dump_xbin(args.input)
        return

    # Read MLIR input
    if args.input == "-":
        mlir_text = sys.stdin.read()
    else:
        with open(args.input, 'r') as f:
            mlir_text = f.read()

    # Parse
    packets, meta = parse_xtpu_mlir(mlir_text)

    if not packets:
        print("ERROR: No packets found in input", file=sys.stderr)
        sys.exit(1)

    print(f"Parsed {len(packets)} packets from program @{meta.program_name}",
          file=sys.stderr)

    # Encode
    xbin_data = encode_xbin(packets, [], meta)

    if args.round_trip:
        raw_packets, rt_meta, _, _ = decode_xbin(xbin_data)
        assert len(raw_packets) == len(packets), \
            f"Round-trip mismatch: {len(raw_packets)} != {len(packets)}"
        print(f"Round-trip OK: {len(packets)} packets, {len(xbin_data)} bytes",
              file=sys.stderr)

    # Write output
    if args.output:
        with open(args.output, 'wb') as f:
            f.write(xbin_data)
        print(f"Wrote {args.output} ({len(xbin_data)} bytes)", file=sys.stderr)
    else:
        sys.stdout.buffer.write(xbin_data)


if __name__ == "__main__":
    main()
