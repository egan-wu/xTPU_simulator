// RUN: xtpu-opt --verify %s
//
// P5-1 acceptance test: hand-written xtpu dialect program
// Full pipeline: SDMA load → IDMA → MATMUL → IDMA writeback → SDMA store
//
// This program computes C = A × B where A and B are 4×4 uint8_t matrices:
//   A is at System Memory[0x0000..0x000F]  (16 bytes)
//   B is at System Memory[0x0010..0x001F]  (16 bytes)
//   C is written to System Memory[0x1000..0x100F] (16 bytes)

xtpu.program @matmul_4x4 {
  // Packet 0: SDMA load A+B from LPDDR5 to Scratchpad
  //   System Memory[0x0000..0x003F] → Scratchpad[0x0000..0x003F]
  //   (load 64 bytes = 1 cacheline, contains both A and B)
  xtpu.packet {
    xtpu.sdma load src_addr = 0 dst_addr = 0 size = 64
  }

  // Packet 1: Sync SDMA, then IDMA load A+B to PU0 LocalMem buffer 0
  //   Scratchpad[0x0000..0x001F] → PU0 LocalMem[buf0][0x0000..0x001F]
  //   (32 bytes = A(16) + B(16))
  xtpu.packet sync_mask = ["sdma"] {
    xtpu.idma load src_addr = 0 dst_addr = 0 size = 32 target = pu0 buffer = 0
  }

  // Packet 2: Sync IDMA→PU0, then PU0 MATMUL
  //   A at buf0[0..15], B at buf0[16..31], C written to buf0[32..47]
  xtpu.packet sync_mask = ["pu0_dma"] {
    xtpu.compute pu = 0 type = matmul buffer = 0 src_offset = 0 dst_offset = 32 length = 16
  }

  // Packet 3: Sync PU0, then IDMA writeback C to Scratchpad
  //   PU0 LocalMem[buf0][32..47] → Scratchpad[0x0000..0x000F]
  xtpu.packet sync_mask = ["pu0_cmd"] {
    xtpu.idma store src_addr = 32 dst_addr = 0 size = 16 target = pu0 buffer = 0
  }

  // Packet 4: Sync IDMA, then SDMA store C to LPDDR5
  //   Scratchpad[0x0000..0x000F] → System Memory[0x1000..0x100F]
  xtpu.packet sync_mask = ["pu0_dma"] {
    xtpu.sdma store src_addr = 0 dst_addr = 4096 size = 16
  }

  // Packet 5: Final drain — wait for SDMA to complete
  xtpu.packet sync_mask = ["sdma"] {
  }
}
