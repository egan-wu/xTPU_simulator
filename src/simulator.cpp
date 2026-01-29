#include "simulator.hpp"

void Simulator::dispatch_packet(const VLIWPacket& packet) {
    // 1. Sync / Fence: Wait for dependencies to clear
    if (packet.sync_mask != 0) {
        status_reg.wait_on_mask(packet.sync_mask);
    }

    // 2. Dispatch sDMA
    if (packet.sDMA_op.type != DMAType::NOP) {
        status_reg.set_busy(STATUS_SDMA_BUSY);
        sdma.push_command(packet.sDMA_op);
    }

    // 3. Dispatch iDMA
    if (packet.iDMA_op.type != DMAType::NOP) {
        uint32_t mask = 0;
        if (packet.iDMA_op.target_mask & TARGET_PU0) mask |= STATUS_PU0_DMA_BUSY;
        if (packet.iDMA_op.target_mask & TARGET_PU1) mask |= STATUS_PU1_DMA_BUSY;

        // Only dispatch if there is a target
        if (mask != 0) {
            status_reg.set_busy(mask);
            idma.push_command(packet.iDMA_op);
        }
    }

    // 4. Dispatch PU0 Compute
    if (packet.pu0_op.type != ComputeType::NOP) {
        status_reg.set_busy(STATUS_PU0_CMD_BUSY);
        pu0.push_command(packet.pu0_op);
    }

    // 5. Dispatch PU1 Compute
    if (packet.pu1_op.type != ComputeType::NOP) {
        status_reg.set_busy(STATUS_PU1_CMD_BUSY);
        pu1.push_command(packet.pu1_op);
    }
}
