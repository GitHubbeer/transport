#pragma once
#include "../../common/getrf/static_pivot.h"
namespace cdls::rocm_detail::getrf {
struct RocmPivotCount {
  int* value;
  __device__ void increment() const { atomicAdd(value, 1); }
};
}  // namespace cdls::rocm_detail::getrf
