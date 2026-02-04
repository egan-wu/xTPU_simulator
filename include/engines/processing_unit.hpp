#pragma once
#include <vector>
#include <string>
#include "status_register.hpp" // Keep for future use if needed, but not member
#include "common.hpp"
#include "memory/local_memory.hpp"
#include "engines/mxu_engine.hpp"
#include "engines/vector_engine.hpp"

// Simple Scalar Engine (Placeholder)
class ScalarEngine : public Engine {
public:
    ScalarEngine() : Engine() {}
    void process(const Compute_Command& cmd) {
        if (cmd.type == ComputeType::NOP) return;
        remaining_cycles = cmd.duration_cycles;
    }
};

class ProcessingUnit {
public:
    ProcessingUnit(int id);

    struct CompletionStatus {
        bool mxu_done = false;
        bool vector_done = false;
        bool scalar_done = false;
    };

    // Hardware lifecycle
    // Ticks all internal components and returns completion signals
    CompletionStatus tick();

    // Check if any unit in this PU is busy
    bool is_busy() const;

    // Command Dispatch
    void dispatch_mxu(const Compute_Command& cmd);
    void dispatch_vector(const Compute_Command& cmd);
    void dispatch_scalar(const Compute_Command& cmd);

    // Access Internal Memory
    LocalMemory& get_local_memory();

    int get_id() const { return id; }

private:
    int id;

    // Architecture Components
    // "Local Memory (quad buffer)"
    LocalMemory local_memory; 

    // "MXU 1024x1024 MAC"
    MXUEngine mxu;

    // "Vector Unit"
    VectorEngine vector_unit;

    // "Scalar Unit"
    ScalarEngine scalar_unit;
};
