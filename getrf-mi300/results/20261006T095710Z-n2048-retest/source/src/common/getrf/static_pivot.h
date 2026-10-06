#pragma once
#include <cmath>

#include "portability.h"

namespace cdls::detail::getrf {
template <int Extent, int RegisterAxis, int Threads, class TensorSmem,
          class Counter>
CDLS_GETRF_DEVICE void factor_with_static_pivot(const unsigned thread_idx,
                                                int bound, TensorSmem& smem,
                                                Counter counter,
                                                double criterion) {
  constexpr int RegisterValues = Extent / RegisterAxis;
  using T = double;
  static_assert(Extent == 64 && RegisterAxis == 16 &&
                Threads == RegisterAxis * RegisterAxis);

  const auto thread_row = thread_idx % RegisterAxis;
  const auto thread_col = thread_idx / RegisterAxis;

  for (unsigned diag_idx = 0; diag_idx < bound; ++diag_idx) {
    T l[RegisterValues], u[RegisterValues];
    bool pred_row[RegisterValues] = {false, false, false, false};
    bool pred_col[RegisterValues] = {false, false, false, false};

    if (1 + diag_idx + thread_idx < bound) {
      smem(1 + diag_idx + thread_idx, diag_idx) /= smem(diag_idx, diag_idx);
    }
    __syncthreads();

#pragma unroll
    for (unsigned i = 0; i < RegisterValues; ++i) {
      if (1 + diag_idx + i * RegisterAxis + thread_row < bound) {
        l[i] = smem(1 + diag_idx + i * RegisterAxis + thread_row, diag_idx);
        pred_row[i] = true;
      }
      if (1 + diag_idx + i * RegisterAxis + thread_col < bound) {
        u[i] = smem(diag_idx, 1 + diag_idx + i * RegisterAxis + thread_col);
        pred_col[i] = true;
      }
    }

#pragma unroll
    for (unsigned row = 0; row < RegisterValues; ++row) {
#pragma unroll
      for (unsigned col = 0; col < RegisterValues; ++col) {
        if (pred_row[row] && pred_col[col]) {
          auto value = smem(1 + diag_idx + row * RegisterAxis + thread_row,
                            1 + diag_idx + col * RegisterAxis + thread_col);
          value -= l[row] * u[col];
          if (thread_idx == 0 && row == 0 && col == 0) {
            if (std::abs(value) < criterion) {
              if (value < 0) {
                value = -criterion;
              } else {
                value = criterion;
              }
              counter.increment();
            }
          }
          smem(1 + diag_idx + row * RegisterAxis + thread_row,
               1 + diag_idx + col * RegisterAxis + thread_col) = value;
        }
      }
    }

    __syncthreads();
  }
}

}  // namespace cdls::detail::getrf
