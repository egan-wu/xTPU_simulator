#include "engines/processing_unit.hpp"

ProcessingUnit::ProcessingUnit(int id) 
    : id(id), local_memory(65536) {} // 64KB per PU

ProcessingUnit::CompletionStatus ProcessingUnit::tick() {
    CompletionStatus status;
    
    if (mxu.tick()) {
        status.mxu_done = true;
    }
    
    if (vector_unit.tick()) {
        status.vector_done = true;
    }

    if (scalar_unit.tick()) {
        status.scalar_done = true;
    }

    return status;
}

bool ProcessingUnit::is_busy() const {
    return mxu.is_busy() || vector_unit.is_busy() || scalar_unit.is_busy();
}

void ProcessingUnit::dispatch_mxu(const Compute_Command& cmd) {
    mxu.process(cmd);
}

void ProcessingUnit::dispatch_vector(const Compute_Command& cmd) {
    vector_unit.process(cmd);
}

void ProcessingUnit::dispatch_scalar(const Compute_Command& cmd) {
    scalar_unit.process(cmd);
}

LocalMemory& ProcessingUnit::get_local_memory() {
    return local_memory;
}
