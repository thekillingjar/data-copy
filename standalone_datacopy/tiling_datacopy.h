#ifndef DATACOPY_TILING_H_
#define DATACOPY_TILING_H_
#include <cstdint>

// A[rows, cols] -> B[cols, rows], int32. Tiles cover the complete rows axis.
// rows % 8 == 0 ensures every output row is 32B aligned without padding.
struct DataCopyTiling {
  uint32_t rows;
  uint32_t cols;
  uint32_t tile_cols;
  uint32_t ub_bytes;
  uint32_t store;  // Diagnostic output, disabled during timing.
};
#endif
