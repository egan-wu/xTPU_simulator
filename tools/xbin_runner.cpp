///===----------------------------------------------------------------------===//
/// xbin_runner — Execute a .xbin program on the xTPU simulator (P5-8)
///
/// Loads a .xbin file, pre-loads input data from a binary file into system
/// memory, runs all packets, then dumps the output region to a file.
///
/// Usage:
///   xbin_runner <model.xbin> \
///     --input <input.bin> --input-offset 0 \
///     --output <output.bin> --output-offset 4096 --output-size 16
///
/// Multiple --input flags supported for multi-input models.
///
/// Build:
///   make xbin_runner
///===----------------------------------------------------------------------===//

#include "xbin_loader.hpp"
#include "simulator.hpp"
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

struct InputSpec {
    std::string path;
    uint64_t offset = 0;
};

struct Config {
    std::string xbin_path;
    std::vector<InputSpec> inputs;
    std::string output_path;
    uint64_t output_offset = 4096;
    uint64_t output_size = 16;
    bool verbose = false;
};

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <model.xbin> [options]\n"
              << "  --input <file>          Input binary data file\n"
              << "  --input-offset <N>      System memory offset for preceding --input (default 0)\n"
              << "  --output <file>         Output binary data file\n"
              << "  --output-offset <N>     System memory offset for output (default 4096)\n"
              << "  --output-size <N>       Number of bytes to read from output (default 16)\n"
              << "  --verbose               Print packet execution progress\n";
}

static std::vector<uint8_t> read_file_binary(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    return std::vector<uint8_t>(
        std::istreambuf_iterator<char>(f),
        std::istreambuf_iterator<char>());
}

static void write_file_binary(const std::string& path,
                              const uint8_t* data, size_t size) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot write: " + path);
    f.write(reinterpret_cast<const char*>(data), size);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    Config cfg;
    cfg.xbin_path = argv[1];

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) {
            cfg.inputs.push_back({argv[++i], 0});
        } else if (arg == "--input-offset" && i + 1 < argc) {
            if (!cfg.inputs.empty())
                cfg.inputs.back().offset = std::stoull(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            cfg.output_path = argv[++i];
        } else if (arg == "--output-offset" && i + 1 < argc) {
            cfg.output_offset = std::stoull(argv[++i]);
        } else if (arg == "--output-size" && i + 1 < argc) {
            cfg.output_size = std::stoull(argv[++i]);
        } else if (arg == "--verbose") {
            cfg.verbose = true;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    if (cfg.output_path.empty()) {
        std::cerr << "ERROR: --output is required\n";
        return 1;
    }

    try {
        // Load .xbin
        auto prog = XBinLoader::load(cfg.xbin_path);
        if (cfg.verbose) {
            std::cerr << "Loaded " << prog.packets.size()
                      << " packets from " << cfg.xbin_path << "\n";
        }

        // Create simulator
        Simulator sim;
        auto& sys_mem = sim.get_system_mem();

        // Pre-load .rodata (constant weights) into system memory
        for (const auto& [addr, data] : prog.rodata) {
            sys_mem.write(addr, data.data(), data.size());
            if (cfg.verbose) {
                std::cerr << "  rodata: " << data.size()
                          << " bytes at offset " << addr << "\n";
            }
        }

        // Pre-load input data
        for (const auto& inp : cfg.inputs) {
            auto data = read_file_binary(inp.path);
            sys_mem.write(inp.offset, data.data(), data.size());
            if (cfg.verbose) {
                std::cerr << "  input: " << data.size()
                          << " bytes at offset " << inp.offset
                          << " from " << inp.path << "\n";
            }
        }

        // Run all packets
        for (size_t i = 0; i < prog.packets.size(); i++) {
            sim.dispatch_packet(prog.packets[i]);
            if (cfg.verbose && (i + 1) % 10 == 0) {
                std::cerr << "  dispatched " << (i + 1) << "/"
                          << prog.packets.size() << " packets\n";
            }
        }

        // Wait for completion
        sim.get_clock().advance(5000);

        // Read output
        std::vector<uint8_t> output(cfg.output_size);
        sys_mem.read(cfg.output_offset, output.data(), cfg.output_size);

        // Write output to file
        write_file_binary(cfg.output_path, output.data(), output.size());

        if (cfg.verbose) {
            std::cerr << "Output: " << cfg.output_size
                      << " bytes from offset " << cfg.output_offset
                      << " → " << cfg.output_path << "\n";
        }

        // Explicitly stop engine threads before Simulator destruction
        // to avoid pure virtual function calls during vtable teardown
        sim.shutdown();

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
