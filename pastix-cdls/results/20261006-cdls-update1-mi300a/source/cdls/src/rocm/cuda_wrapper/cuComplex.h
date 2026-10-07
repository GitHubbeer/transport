#pragma once

#include <hip/hip_complex.h>

typedef hipFloatComplex cuFloatComplex;
typedef hipDoubleComplex cuDoubleComplex;

__HOST_DEVICE__ inline hipFloatComplex make_cuFloatComplex(const float x,
                                                           const float y) {
  return make_hipFloatComplex(x, y);
}

__HOST_DEVICE__ inline hipDoubleComplex make_cuDoubleComplex(const double x,
                                                             const double y) {
  return make_hipDoubleComplex(x, y);
}

__HOST_DEVICE__ inline float cuCrealf(const cuFloatComplex z) {
  return hipCrealf(z);
}

__HOST_DEVICE__ inline float cuCimagf(const cuFloatComplex z) {
  return hipCimagf(z);
}

__HOST_DEVICE__ inline double cuCreal(const cuDoubleComplex z) {
  return hipCreal(z);
}

__HOST_DEVICE__ inline double cuCimag(const cuDoubleComplex z) {
  return hipCimag(z);
}
