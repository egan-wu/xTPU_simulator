// RUN: xtpu-opt --verify %s
//
// P5-1 acceptance test: double-buffered compute with overlap
//
// Demonstrates the LPU-inspired pipelining pattern:
//   - While PU0 computes on buffer 0, IDMA loads next tile into buffer 1
//   - While PU0 computes on buffer 1, IDMA writes back buffer 0 result
//
// This is the fundamental scheduling pattern that the VLIW scheduler (P5-5)
// must learn to generate automatically.

xtpu.program @double_buffer_matmul {
  // === Prologue: Load tile 0 ===

  // Load tile 0 from LPDDR5 to Scratchpad
  xtpu.packet {
    xtpu.sdma load src_addr = 0 dst_addr = 0 size = 64
  }

  // IDMA: Scratchpad → PU0 buf0 (A0 + B0)
  xtpu.packet sync_mask = ["sdma"] {
    xtpu.idma load src_addr = 0 dst_addr = 0 size = 32 target = pu0 buffer = 0
  }

  // === Steady state: Overlap compute(buf0) + load(buf1) ===

  // PU0 computes on buf0; simultaneously SDMA loads tile 1
  xtpu.packet sync_mask = ["pu0_dma"] {
    xtpu.sdma load src_addr = 64 dst_addr = 64 size = 64
    xtpu.compute pu = 0 type = matmul buffer = 0 src_offset = 0 dst_offset = 32 length = 16
  }

  // Sync both SDMA and PU0, then IDMA loads tile 1 into buf1
  xtpu.packet sync_mask = ["sdma", "pu0_cmd"] {
    xtpu.idma load src_addr = 64 dst_addr = 0 size = 32 target = pu0 buffer = 1
  }

  // PU0 computes on buf1; simultaneously IDMA writes back buf0 result
  xtpu.packet sync_mask = ["pu0_dma"] {
    xtpu.compute pu = 0 type = matmul buffer = 1 src_offset = 0 dst_offset = 32 length = 16
    xtpu.idma store src_addr = 32 dst_addr = 0 size = 16 target = pu0 buffer = 0
  }

  // === Epilogue: Write back tile 1 result ===

  // Wait for both PU0 (buf1 compute) and IDMA (buf0 writeback)
  xtpu.packet sync_mask = ["pu0_cmd", "pu0_dma"] {
    xtpu.idma store src_addr = 32 dst_addr = 16 size = 16 target = pu0 buffer = 1
  }

  // SDMA stores both results back to LPDDR5
  xtpu.packet sync_mask = ["pu0_dma"] {
    xtpu.sdma store src_addr = 0 dst_addr = 8192 size = 32
  }

  // Final drain
  xtpu.packet sync_mask = ["sdma"] {
  }
}
