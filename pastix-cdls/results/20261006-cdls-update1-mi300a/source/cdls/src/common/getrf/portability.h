#pragma once

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#elif defined(__CUDACC__)
#include <cuda_runtime.h>
#endif

#if defined(__CUDACC__) || defined(__HIPCC__)
#define CDLS_GETRF_HD __host__ __device__ inline
#define CDLS_GETRF_DEVICE __device__ __forceinline__
#else
#define CDLS_GETRF_HD inline
#define CDLS_GETRF_DEVICE inline
#endif
