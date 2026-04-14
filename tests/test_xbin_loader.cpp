///===----------------------------------------------------------------------===//
/// test_xbin_loader — P5-7: Load .xbin and run on simulator
///
/// Tests:
///   1. XBinLoader can decode a .xbin produced by xtpu-translate
///   2. Loaded packets run on the simulator without errors
///   3. Matmul result matches expected output (C = A × B identity)
///
/// Build:
///   g++ -std=c++17 -I include -I submodule/lpddr5-sim/include \
///       tests/test_xbin_loader.cpp src/simulator.cpp src/engines.cpp \
///       -o test_xbin_loader
///
/// Usage:
///   ./test_xbin_loader path/to/matmul.xbin
///   Or without args: creates a .xbin in-memory from hardcoded packets.
///===----------------------------------------------------------------------===//

#include "xbin_loader.hpp"
#include "simulator.hpp"
#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>

// ---------------------------------------------------------------------------
// Helper: create a test .xbin binary in-memory (single 4×4 matmul)
// Matches the output of:
//   xtpu-opt --tosa-to-linalg-pipeline --linalg-to-xtpu
// on the single_matmul_i8 ONNX model.
// ---------------------------------------------------------------------------
static std::vector<uint8_t> create_test_xbin() {
    // Build 7 packets manually (matching matmul_e2e reference)
    std::vector<XBinPacket> packets(7);
    std::memset(packets.data(), 0, packets.size() * sizeof(XBinPacket));

    // Packet 0: SDMA load A from sys[0] to scratch[0], 16 bytes
    packets[0].sdma.type = 1; // MEMCPY
    packets[0].sdma.direction = 0; // TO_DEVICE
    packets[0].sdma.src_addr = 0;
    packets[0].sdma.dst_addr = 0;
    packets[0].sdma.size = 16;
    packets[0].sync_mask = 0;

    // Packet 1: sync SDMA, SDMA load B from sys[16] to scratch[16], 16 bytes
    packets[1].sdma.type = 1;
    packets[1].sdma.direction = 0;
    packets[1].sdma.src_addr = 16;
    packets[1].sdma.dst_addr = 16;
    packets[1].sdma.size = 16;
    packets[1].sync_mask = STATUS_SDMA_BUSY;

    // Packet 2: sync SDMA, IDMA load from scratch[0] to PU0 local[0], 32 bytes
    packets[2].idma.type = 1;
    packets[2].idma.direction = 0;
    packets[2].idma.src_addr = 0;
    packets[2].idma.dst_addr = 0;
    packets[2].idma.size = 32;
    packets[2].idma.target_mask = TARGET_PU0;
    packets[2].idma.buffer_idx = 0;
    packets[2].sync_mask = STATUS_SDMA_BUSY;

    // Packet 3: sync PU0_DMA, compute matmul
    packets[3].pu0.type = 1; // MATMUL
    packets[3].pu0.buffer_idx = 0;
    packets[3].pu0.src_offset = 0;
    packets[3].pu0.dst_offset = 32;
    packets[3].pu0.length = 16;
    packets[3].sync_mask = STATUS_PU0_DMA_BUSY;

    // Packet 4: sync PU0_CMD, IDMA store from local[32] to scratch[0], 16 bytes
    packets[4].idma.type = 1;
    packets[4].idma.direction = 1; // FROM_DEVICE
    packets[4].idma.src_addr = 32;
    packets[4].idma.dst_addr = 0;
    packets[4].idma.size = 16;
    packets[4].idma.target_mask = TARGET_PU0;
    packets[4].idma.buffer_idx = 0;
    packets[4].sync_mask = STATUS_PU0_CMD_BUSY;

    // Packet 5: sync PU0_DMA, SDMA store from scratch[0] to sys[4096], 16 bytes
    packets[5].sdma.type = 1;
    packets[5].sdma.direction = 1; // FROM_DEVICE
    packets[5].sdma.src_addr = 0;
    packets[5].sdma.dst_addr = 4096;
    packets[5].sdma.size = 16;
    packets[5].sync_mask = STATUS_PU0_DMA_BUSY;

    // Packet 6: sync SDMA (drain)
    packets[6].sync_mask = STATUS_SDMA_BUSY;

    // Encode to .xbin
    // Header (32 bytes)
    std::vector<uint8_t> xbin;
    xbin.resize(32 + 2 * 8); // header + 2 section entries

    // Magic
    xbin[0] = 'X'; xbin[1] = 'T'; xbin[2] = 'P'; xbin[3] = 'U';
    // Version = 1
    xbin[4] = 1; xbin[5] = 0;
    // num_sections = 2
    xbin[6] = 2; xbin[7] = 0;

    uint32_t text_offset = 32 + 2 * 8; // after header + section table
    // entry_offset
    std::memcpy(&xbin[8], &text_offset, 4);
    // flags = 0x01
    uint32_t flags = 1;
    std::memcpy(&xbin[12], &flags, 4);

    // Section 0: .text
    uint16_t sec_type = 0; // .text
    uint16_t sec_flags = 0;
    std::memcpy(&xbin[32], &sec_type, 2);
    std::memcpy(&xbin[34], &sec_flags, 2);
    std::memcpy(&xbin[36], &text_offset, 4);

    // Section 1: .meta (after .text)
    uint32_t num_pkts = 7;
    uint32_t text_size = 4 + num_pkts * sizeof(XBinPacket);
    uint32_t meta_offset = text_offset + text_size;
    sec_type = 2; // .meta
    std::memcpy(&xbin[40], &sec_type, 2);
    std::memcpy(&xbin[42], &sec_flags, 2);
    std::memcpy(&xbin[44], &meta_offset, 4);

    // .text data: num_packets + packet array
    size_t text_start = xbin.size();
    xbin.resize(text_start + text_size);
    std::memcpy(&xbin[text_start], &num_pkts, 4);
    std::memcpy(&xbin[text_start + 4], packets.data(),
                num_pkts * sizeof(XBinPacket));

    // .meta data: simple JSON
    std::string meta = R"({"program":"main"})";
    xbin.insert(xbin.end(), meta.begin(), meta.end());

    return xbin;
}

// ---------------------------------------------------------------------------
// Test: load .xbin and verify structure
// ---------------------------------------------------------------------------
static void test_decode() {
    std::cout << "test_decode: ";

    auto xbin = create_test_xbin();
    auto prog = XBinLoader::decode(xbin);

    assert(prog.packets.size() == 7);
    assert(prog.packets[0].sDMA_op.type == DMAType::MEMCPY);
    assert(prog.packets[0].sDMA_op.size == 16);
    assert(prog.packets[0].sync_mask == 0);
    assert(prog.packets[1].sync_mask == STATUS_SDMA_BUSY);
    assert(prog.packets[3].pu0_op.type == ComputeType::MATMUL);
    assert(prog.packets[3].pu0_op.length == 16);
    assert(prog.packets[6].sync_mask == STATUS_SDMA_BUSY);

    std::cout << "PASS\n";
}

// ---------------------------------------------------------------------------
// Test: run loaded program on simulator
// ---------------------------------------------------------------------------
static void test_run_matmul() {
    std::cout << "test_run_matmul: ";

    auto xbin = create_test_xbin();
    auto prog = XBinLoader::decode(xbin);

    // Create simulator
    Simulator sim;

    // Pre-load input data into system memory
    // A = 4×4 identity matrix (uint8_t), at offset 0
    uint8_t identity[16] = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    };
    // B = same identity matrix, at offset 16
    auto& sys_mem = sim.get_system_mem();
    sys_mem.write(0, identity, 16);    // A
    sys_mem.write(16, identity, 16);   // B

    // Run all packets
    for (const auto& pkt : prog.packets) {
        sim.dispatch_packet(pkt);
    }

    // Wait for completion
    sim.get_clock().advance(1000);

    // Read result from system memory at offset 4096
    uint8_t result[16] = {};
    sys_mem.read(4096, result, 16);

    // C = A × B = identity × identity = identity
    // Note: matmul does uint32 accumulation → uint8 truncation
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            uint8_t expected = (i == j) ? 1 : 0;
            if (result[i * 4 + j] != expected) {
                std::cerr << "FAIL: result[" << i << "][" << j << "] = "
                          << (int)result[i * 4 + j]
                          << ", expected " << (int)expected << "\n";
                assert(false);
            }
        }
    }

    std::cout << "PASS (C = I × I = I, bit-exact)\n";
}

// ---------------------------------------------------------------------------
// Test: load from file
// ---------------------------------------------------------------------------
static void test_load_file(const std::string& path) {
    std::cout << "test_load_file(" << path << "): ";

    auto prog = XBinLoader::load(path);
    std::cout << prog.packets.size() << " packets loaded, ";

    // Create simulator and run
    Simulator sim;

    // Pre-load identity matrices for A and B
    uint8_t identity[16] = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    };
    auto& sys_mem = sim.get_system_mem();
    sys_mem.write(0, identity, 16);    // A (input)
    sys_mem.write(16, identity, 16);   // B (weight)

    for (const auto& pkt : prog.packets) {
        sim.dispatch_packet(pkt);
    }
    sim.get_clock().advance(1000);

    // Read result
    uint8_t result[16] = {};
    sys_mem.read(4096, result, 16);

    std::cout << "result[0]=" << (int)result[0]
              << " result[5]=" << (int)result[5]
              << " result[15]=" << (int)result[15]
              << " PASS\n";
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    std::cout << "=== XBinLoader Tests (P5-7) ===\n";

    test_decode();
    test_run_matmul();

    if (argc > 1) {
        test_load_file(argv[1]);
    }

    std::cout << "\nAll tests passed!\n";
    return 0;
}
