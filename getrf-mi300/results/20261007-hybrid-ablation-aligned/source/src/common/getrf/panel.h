// SPDX-License-Identifier: MIT AND BSD-2-Clause
// Inner LU panels adapted from ck_getrf and rocSOLVER; see
// third_party/ck_getrf_notices for the retained upstream licenses.
#pragma once

#include "portability.h"

namespace cdls::detail::getrf {

// Pivot publication is backend-specific. Once those pivots are factored,
// write back L11/L21, solve U12, and update only the remaining square.
// Tile is an oriented (row,column) view; it may be a CuTe tensor or padded LDS.
template <int Extent, int P, int Threads, class Tile>
CDLS_GETRF_DEVICE void finish_lu_panel(int tid, int panel, int bound, Tile& c,
                                       double (&x)[P]) {
  if (tid < Extent) {
#pragma unroll
    for (int j = 0; j < P; ++j) c(tid, panel + j) = x[j];
  }
  __syncthreads();
  if (tid >= panel + P && tid < bound) {
#pragma unroll
    for (int k = 0; k < P; ++k) {
      double value = c(panel + k, tid);
#pragma unroll
      for (int j = 0; j < k; ++j) value -= c(panel + k, panel + j) * x[j];
      x[k] = value;
    }
#pragma unroll
    for (int k = 0; k < P; ++k) c(panel + k, tid) = x[k];
  }
  __syncthreads();
  const int start = panel + P;
  const int remaining = bound > start ? bound - start : 0;
  static_assert(Extent == 64 && Threads % 64 == 0);
  const int shift = remaining <= 8    ? 3
                    : remaining <= 16 ? 4
                    : remaining <= 32 ? 5
                                      : 6;
  const int span = 1 << shift;
  const int row = start + (tid & (span - 1));
  if (row < bound && tid < span * remaining) {
    double lower[P];
#pragma unroll
    for (int k = 0; k < P; ++k) lower[k] = c(row, panel + k);
    for (int idx = tid; idx < span * remaining; idx += Threads) {
      const int col = start + (idx >> shift);
      double value = c(row, col);
#pragma unroll
      for (int k = 0; k < P; ++k) value -= lower[k] * c(panel + k, col);
      c(row, col) = value;
    }
  }
  __syncthreads();
}

}  // namespace cdls::detail::getrf
