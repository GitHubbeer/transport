#pragma once

#include <cmath>

#include "portability.h"

namespace cdls::detail::getrf {

// The interval is closed under reciprocation: mixed raw/reciprocal diagonal
// storage can distinguish extreme pivots without a separate tag array.
CDLS_GETRF_HD bool reciprocal_safe(double value) {
  const double magnitude = std::fabs(value);
  return magnitude >= 0x1p-500 && magnitude <= 0x1p500;
}

enum class TriangularDiagonal {
  unit,
  raw,
  mixed_reciprocals,
  safe_reciprocals
};

}  // namespace cdls::detail::getrf
