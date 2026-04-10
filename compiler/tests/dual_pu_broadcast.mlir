// RUN: xtpu-opt --verify %s
//
// P5-1 acceptance test: dual-PU parallel compute with broadcast
//
// Demonstrates:
//   - IDMA broadcast (target = pu01): same weight matrix to both PUs
//   - Dual PU parallel compute: PU0 and PU1 in the same VLIW packet
//   - Sequential IDMA for per-PU activations (IDMA is a single engine)

xtpu.program @dual_pu_broadcast {
  // 1. Load weights from LPDDR5 → Scratchpad
  xtpu.packet {
    xtpu.sdma load src_addr = 0 dst_addr = 0 size = 64
  }

  // 2. Broadcast weights to both PUs (same data, buf0[0..31])
  xtpu.packet sync_mask = ["sdma"] {
    xtpu.idma load src_addr = 0 dst_addr = 0 size = 32 target = pu01 buffer = 0
  }

  // 3. Load activations from LPDDR5
  xtpu.packet sync_mask = ["pu0_dma", "pu1_dma"] {
    xtpu.sdma load src_addr = 64 dst_addr = 64 size = 64
  }

  // 4. IDMA PU0 activation (IDMA is single engine, must be separate packets)
  xtpu.packet sync_mask = ["sdma"] {
    xtpu.idma load src_addr = 64 dst_addr = 32 size = 16 target = pu0 buffer = 0
  }

  // 5. IDMA PU1 activation
  xtpu.packet sync_mask = ["pu0_dma"] {
    xtpu.idma load src_addr = 80 dst_addr = 32 size = 16 target = pu1 buffer = 0
  }

  // 6. Both PUs compute MATMUL in parallel
  xtpu.packet sync_mask = ["pu1_dma"] {
    xtpu.compute pu = 0 type = matmul buffer = 0 src_offset = 0 dst_offset = 48 length = 16
    xtpu.compute pu = 1 type = matmul buffer = 0 src_offset = 0 dst_offset = 48 length = 16
  }

  // 7. Writeback PU0 result
  xtpu.packet sync_mask = ["pu0_cmd", "pu1_cmd"] {
    xtpu.idma store src_addr = 48 dst_addr = 0 size = 16 target = pu0 buffer = 0
  }

  // 8. Writeback PU1 result
  xtpu.packet sync_mask = ["pu0_dma"] {
    xtpu.idma store src_addr = 48 dst_addr = 16 size = 16 target = pu1 buffer = 0
  }

  // 9. SDMA store both results to LPDDR5
  xtpu.packet sync_mask = ["pu1_dma"] {
    xtpu.sdma store src_addr = 0 dst_addr = 4096 size = 32
  }

  // 10. Final drain
  xtpu.packet sync_mask = ["sdma"] {
  }
}
