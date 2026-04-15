#pragma once
//===----------------------------------------------------------------------===//
// XBinLoader — Load .xbin compiled programs into the simulator (P5-7)
//
// Reads a .xbin binary (produced by xtpu-translate), decodes the packet
// stream and constant data, and provides a high-level API to run programs
// on the simulator.
//
// The .xbin format is a simple container with 3 sections:
//   .text   — encoded VLIWPacket stream
//   .rodata — constant weights to pre-load into system memory
//   .meta   — JSON metadata (program name, input/output info)
//
// Design: The simulator core does not know about .xbin. XBinLoader is
// a pure client-side tool that produces std::vector<VLIWPacket>.
//===----------------------------------------------------------------------===//

#include <cstddef>
#include "commands.hpp"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

//===----------------------------------------------------------------------===//
// .xbin binary format definitions
//===----------------------------------------------------------------------===//

#pragma pack(push, 1)

struct XBinHeader {
    char     magic[4];       // "XTPU"
    uint16_t version;        // 1
    uint16_t num_sections;
    uint32_t entry_offset;   // byte offset to .text section
    uint32_t flags;          // 0x01 = INT8 mode
    uint8_t  reserved[16];
};
static_assert(sizeof(XBinHeader) == 32, "XBinHeader must be 32 bytes");

struct XBinSectionEntry {
    uint16_t type;   // 0=.text, 1=.rodata, 2=.meta
    uint16_t flags;
    uint32_t offset; // byte offset from file start
};
static_assert(sizeof(XBinSectionEntry) == 8, "Section entry must be 8 bytes");

// Binary-encoded DMA command (fixed 40 bytes)
struct XBinDMACommand {
    uint32_t type;         // 0=NOP, 1=MEMCPY
    uint32_t direction;    // 0=TO_DEVICE, 1=FROM_DEVICE
    uint64_t src_addr;
    uint64_t dst_addr;
    uint64_t size;
    uint32_t target_mask;
    int32_t  buffer_idx;
};
static_assert(sizeof(XBinDMACommand) == 40, "XBinDMACommand must be 40 bytes");

// Binary-encoded Compute command (fixed 28 bytes)
// P5-11: added src2_offset for dual-operand ops
struct XBinComputeCommand {
    uint32_t type;                // 0=NOP,1=MATMUL,2=VECTOR,3=SCALAR,4=ADD,...
    int32_t  buffer_idx;
    uint32_t simulated_duration;
    uint32_t src_offset;
    uint32_t dst_offset;
    uint32_t length;
    uint32_t src2_offset;         // P5-11: second operand (for ADD/MUL/SUB/MAX)
};
static_assert(sizeof(XBinComputeCommand) == 28, "XBinComputeCommand must be 28 bytes");

// Binary-encoded VLIWPacket (fixed 140 bytes)
// P5-11: grew from 132 to 140 bytes (2 × 4 bytes for src2_offset)
struct XBinPacket {
    XBinDMACommand     sdma;       // 40 bytes
    XBinDMACommand     idma;       // 40 bytes
    XBinComputeCommand pu0;        // 28 bytes
    XBinComputeCommand pu1;        // 28 bytes
    uint32_t           sync_mask;  //  4 bytes
    // Total: 140 bytes
};
static_assert(sizeof(XBinPacket) == 140, "XBinPacket must be 140 bytes");

#pragma pack(pop)

//===----------------------------------------------------------------------===//
// XBinLoader
//===----------------------------------------------------------------------===//

class XBinLoader {
public:
    struct Program {
        std::string name;
        std::vector<VLIWPacket> packets;
        std::vector<std::pair<uint64_t, std::vector<uint8_t>>> rodata; // (offset, data)
        std::string meta_json;
    };

    /// Load a .xbin file from disk.
    static Program load(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open .xbin file: " + path);
        }

        std::vector<uint8_t> data(
            (std::istreambuf_iterator<char>(file)),
            std::istreambuf_iterator<char>());

        return decode(data);
    }

    /// Decode .xbin from a memory buffer.
    static Program decode(const std::vector<uint8_t>& data) {
        if (data.size() < sizeof(XBinHeader)) {
            throw std::runtime_error(".xbin too small for header");
        }

        Program prog;

        // Parse header
        XBinHeader header;
        std::memcpy(&header, data.data(), sizeof(header));

        if (std::memcmp(header.magic, "XTPU", 4) != 0) {
            throw std::runtime_error("Bad .xbin magic");
        }
        if (header.version != 1) {
            throw std::runtime_error("Unsupported .xbin version: " +
                                     std::to_string(header.version));
        }

        // Parse section table
        size_t sec_table_off = sizeof(XBinHeader);
        for (int i = 0; i < header.num_sections; i++) {
            XBinSectionEntry entry;
            std::memcpy(&entry, data.data() + sec_table_off + i * sizeof(entry),
                        sizeof(entry));

            switch (entry.type) {
            case 0: // .text
                decodeText(data, entry.offset, prog);
                break;
            case 1: // .rodata
                decodeRodata(data, entry.offset, prog);
                break;
            case 2: // .meta
                decodeMeta(data, entry.offset, data.size(), prog);
                break;
            }
        }

        return prog;
    }

private:
    static void decodeText(const std::vector<uint8_t>& data,
                           uint32_t offset, Program& prog) {
        uint32_t num_packets;
        std::memcpy(&num_packets, data.data() + offset, 4);
        offset += 4;

        for (uint32_t i = 0; i < num_packets; i++) {
            XBinPacket bin_pkt;
            std::memcpy(&bin_pkt, data.data() + offset, sizeof(bin_pkt));
            offset += sizeof(bin_pkt);

            VLIWPacket pkt;

            // Decode sDMA
            pkt.sDMA_op.type = (bin_pkt.sdma.type == 1) ? DMAType::MEMCPY : DMAType::NOP;
            pkt.sDMA_op.direction = (bin_pkt.sdma.direction == 1)
                ? DMADirection::FROM_DEVICE : DMADirection::TO_DEVICE;
            pkt.sDMA_op.src_addr = bin_pkt.sdma.src_addr;
            pkt.sDMA_op.dst_addr = bin_pkt.sdma.dst_addr;
            pkt.sDMA_op.size = bin_pkt.sdma.size;
            pkt.sDMA_op.target_mask = bin_pkt.sdma.target_mask;
            pkt.sDMA_op.buffer_idx = bin_pkt.sdma.buffer_idx;

            // Decode iDMA
            pkt.iDMA_op.type = (bin_pkt.idma.type == 1) ? DMAType::MEMCPY : DMAType::NOP;
            pkt.iDMA_op.direction = (bin_pkt.idma.direction == 1)
                ? DMADirection::FROM_DEVICE : DMADirection::TO_DEVICE;
            pkt.iDMA_op.src_addr = bin_pkt.idma.src_addr;
            pkt.iDMA_op.dst_addr = bin_pkt.idma.dst_addr;
            pkt.iDMA_op.size = bin_pkt.idma.size;
            pkt.iDMA_op.target_mask = bin_pkt.idma.target_mask;
            pkt.iDMA_op.buffer_idx = bin_pkt.idma.buffer_idx;

            // Decode PU0
            pkt.pu0_op.type = static_cast<ComputeType>(bin_pkt.pu0.type);
            pkt.pu0_op.buffer_idx = bin_pkt.pu0.buffer_idx;
            pkt.pu0_op.simulated_duration_ms = bin_pkt.pu0.simulated_duration;
            pkt.pu0_op.src_offset = bin_pkt.pu0.src_offset;
            pkt.pu0_op.dst_offset = bin_pkt.pu0.dst_offset;
            pkt.pu0_op.length = bin_pkt.pu0.length;
            pkt.pu0_op.src2_offset = bin_pkt.pu0.src2_offset;

            // Decode PU1
            pkt.pu1_op.type = static_cast<ComputeType>(bin_pkt.pu1.type);
            pkt.pu1_op.buffer_idx = bin_pkt.pu1.buffer_idx;
            pkt.pu1_op.simulated_duration_ms = bin_pkt.pu1.simulated_duration;
            pkt.pu1_op.src_offset = bin_pkt.pu1.src_offset;
            pkt.pu1_op.dst_offset = bin_pkt.pu1.dst_offset;
            pkt.pu1_op.length = bin_pkt.pu1.length;
            pkt.pu1_op.src2_offset = bin_pkt.pu1.src2_offset;

            // Decode sync_mask
            pkt.sync_mask = bin_pkt.sync_mask;

            prog.packets.push_back(pkt);
        }
    }

    static void decodeRodata(const std::vector<uint8_t>& data,
                             uint32_t offset, Program& prog) {
        uint32_t num_entries;
        std::memcpy(&num_entries, data.data() + offset, 4);
        offset += 4;

        for (uint32_t i = 0; i < num_entries; i++) {
            uint64_t addr, size;
            std::memcpy(&addr, data.data() + offset, 8);
            std::memcpy(&size, data.data() + offset + 8, 8);
            offset += 16;

            std::vector<uint8_t> entry_data(data.begin() + offset,
                                            data.begin() + offset + size);
            offset += size;

            prog.rodata.push_back({addr, std::move(entry_data)});
        }
    }

    static void decodeMeta(const std::vector<uint8_t>& data,
                           uint32_t offset, size_t end, Program& prog) {
        prog.meta_json = std::string(data.begin() + offset, data.begin() + end);
    }
};
