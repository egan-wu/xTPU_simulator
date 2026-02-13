# VLIW TPU Simulator Verification

This document details the verification of the Cycle-Accurate VLIW TPU Simulator.

## Architecture

The simulator models a VLIW architecture with the following components:
*   **System DMA (sDMA)**: Handles external data movement.
*   **Internal DMA (iDMA)**: Handles data movement to Local Memory (4 Banks), with Broadcast support.
*   **MXU (Matrix Unit)**: Handles matrix operations.
*   **Vector Unit**: Handles vector operations.

## Test Scenario

The verification scenario in `src/main.cpp` executes the following sequence:

1.  **Instruction 0**: `iDMA` Move to Bank 0 (Duration: 5 cycles).
    *   No Sync dependency.
2.  **Instruction 1**: `MXU` MatMul (Duration: 10 cycles).
    *   **Dependency**: Sync on Bank 0 (Must wait for Inst 0 to finish).
3.  **Instruction 2**: `Vector` Add (Duration: 3 cycles).
    *   **Dependency**: Sync on MXU (Must wait for Inst 1 to finish).

## Expected Behavior

1.  **Cycle 0**: Inst 0 Dispatched. iDMA becomes BUSY (Bank 0).
2.  **Cycle 1-4**: Inst 1 attempts to Fetch. Sync Check fails (iDMA Busy). **STALL**.
3.  **Cycle 5**: iDMA completes (5 cycles elapsed). Busy bit cleared. Inst 1 Sync Check passes. Inst 1 Dispatched. MXU becomes BUSY.
4.  **Cycle 6-14**: Inst 2 attempts to Fetch. Sync Check fails (MXU Busy). **STALL**.
5.  **Cycle 15**: MXU completes (10 cycles elapsed). Busy bit cleared. Inst 2 Dispatched. Vector becomes BUSY.
6.  **Cycle 16-18**: Vector executes.
7.  **Cycle 19**: Vector completes. Simulation ends.

## Verification Log

```text
Cycle | PC | Status | Action
------+----+--------+-------
[Scoreboard] [0b0000000] -> [0b0000001]
    0 |  0 | 0x01 | DISPATCH  <-- Inst 0 Dispatched
    1 |  1 | 0x01 | STALL (SYNC)
    2 |  1 | 0x01 | STALL (SYNC)
    3 |  1 | 0x01 | STALL (SYNC)
    4 |  1 | 0x01 | STALL (SYNC)
[Scoreboard] [0b0000001] -> [0b0000000] Cleared: IDMA_B0
[Scoreboard] [0b0000000] -> [0b0100000]
    5 |  1 | 0x20 | DISPATCH  <-- Inst 1 Dispatched (MXU Busy)
    6 |  2 | 0x20 | STALL (SYNC)
    ...
   14 |  2 | 0x20 | STALL (SYNC)
[Scoreboard] [0b0100000] -> [0b0000000] Cleared: MXU
[Scoreboard] [0b0000000] -> [0b1000000]
   15 |  2 | 0x40 | DISPATCH  <-- Inst 2 Dispatched (VEC Busy)
   ...
[Scoreboard] [0b1000000] -> [0b0000000] Cleared: VEC
   19 |  3 | 0x00 | IDLE
```
