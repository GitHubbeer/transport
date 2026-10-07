#pragma once

#include <cdls/common.h>

#if defined(__cplusplus)
#include <cstdint>
#include <cstdio>
#else
#include <stdint.h>
#include <stdio.h>
#endif

#pragma once

#if defined(__cplusplus)
extern "C" {
#endif

// Helper macro for HIP errors
#ifndef CDLS_CHECK_HIP_ERROR
#define CDLS_CHECK_HIP_ERROR(expression)                            \
  if (auto status = (expression); status != hipSuccess) {           \
    fprintf(stderr, "hip error: '%s'(%d) at %s:%d\n",               \
            hipGetErrorString(status), status, __FILE__, __LINE__); \
    exit(EXIT_FAILURE);                                             \
  }
#endif

#ifndef CDLS_CHECK_HIPRTC_ERROR
#define CDLS_CHECK_HIPRTC_ERROR(expression)                            \
  if (auto status = (expression); status != HIPRTC_SUCCESS) {          \
    fprintf(stderr, "hipRTC error: '%s'(%d) at %s:%d\n",               \
            hiprtcGetErrorString(status), status, __FILE__, __LINE__); \
    exit(EXIT_FAILURE);                                                \
  }
#endif

#ifndef CDLS_CHECK_ROCBLAS_ERROR
#define CDLS_CHECK_ROCBLAS_ERROR(expression)                               \
  if (auto status = (expression); status != rocblas_status_success) {      \
    fprintf(stderr, "rocBLAS error: %d at %s:%d\n", int(status), __FILE__, \
            __LINE__);                                                     \
    exit(EXIT_FAILURE);                                                    \
  }
#endif

// Helper macro for CUDA errors
#ifndef CDLS_CHECK_CUDA_ERROR
#define CDLS_CHECK_CUDA_ERROR(expression)                            \
  if (auto status = (expression); status != cudaSuccess) {           \
    fprintf(stderr, "cuda error: '%s'(%d) at %s:%d\n",               \
            cudaGetErrorString(status), status, __FILE__, __LINE__); \
    exit(EXIT_FAILURE);                                              \
  }
#endif

#ifndef CDLS_CHECK_CUSOLVER_ERROR
#define CDLS_CHECK_CUSOLVER_ERROR(expression)                               \
  if (auto status = (expression); status != CUSOLVER_STATUS_SUCCESS) {      \
    fprintf(stderr, "cuSOLVER error: %d at %s:%d\n", int(status), __FILE__, \
            __LINE__);                                                      \
    exit(EXIT_FAILURE);                                                     \
  }
#endif

#ifndef CDLS_CHECK_CDLS_ERROR
#define CDLS_CHECK_CDLS_ERROR(expression)                               \
  if (auto status = (expression); status != CDLS_STATUS_SUCCESS) {      \
    fprintf(stderr, "CDLS error: %d at %s:%d\n", int(status), __FILE__, \
            __LINE__);                                                  \
    exit(EXIT_FAILURE);                                                 \
  }
#endif

// HIP Host function to retrieve the warp size
enum hipWarpSize_t : uint32_t {
  Wave32 = 32,
  Wave64 = 64,
  UNSUPPORTED_WARP_SIZE,
};

struct cdlsHandle {
  void *stream_work;
  void *stream_memset;
};

#if defined(__cplusplus)
}
#endif
