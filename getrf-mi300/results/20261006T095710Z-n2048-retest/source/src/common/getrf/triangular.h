#pragma once

#include "numeric.h"

namespace cdls::detail::getrf {

// The same forward recurrence solves Lx=B and U^T x^T=B^T. Oriented views
// describe storage; Ops owns subgroup communication and singular-pivot policy.
template <int Extent, int Lanes, TriangularDiagonal Mode, class Ops,
          class Coefficients, class Rhs, class Output>
CDLS_GETRF_DEVICE void forward_solve(int tid, int bound,
                                     const Coefficients& coefficients, Rhs& rhs,
                                     Output& output, Ops ops) {
  constexpr int Values = Extent / Lanes;
  static_assert(Extent % Lanes == 0);
  const int lane = tid % Lanes, line = tid / Lanes;
  double x[Values];
#pragma unroll
  for (int group = 0; group < Values; ++group)
    x[group] = rhs(lane + group * Lanes, line);
#pragma unroll
  for (int group = 0; group < Values; ++group) {
#pragma unroll
    for (int source = 0; source < Lanes; ++source) {
      const int k = group * Lanes + source;
      if (lane == source) {
        if constexpr (Mode != TriangularDiagonal::unit) {
          const double diagonal = coefficients(k, k);
          if constexpr (Mode == TriangularDiagonal::safe_reciprocals)
            x[group] *= diagonal;
          else if constexpr (Mode == TriangularDiagonal::mixed_reciprocals) {
            if (reciprocal_safe(diagonal))
              x[group] *= diagonal;
            else
              x[group] = ops.divide(x[group], diagonal);
          } else
            x[group] = ops.divide(x[group], diagonal);
        }
      }
      const double pivot =
          ops.template broadcast<Lanes>(x[group], source, k, line, rhs);
      if (lane > source)
        x[group] -= pivot * coefficients(lane + group * Lanes, k);
#pragma unroll
      for (int next = group + 1; next < Values; ++next)
        x[next] -= pivot * coefficients(lane + next * Lanes, k);
    }
    ops.finish_group();
    if (line < bound) output(lane + group * Lanes, line) = x[group];
  }
  ops.finish();
}

}  // namespace cdls::detail::getrf
