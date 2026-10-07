#pragma once
#include <stddef.h>
#include <cdls/common.h>

#if defined(__cplusplus)
extern "C" {
#endif

cdlsStatus_t cdlsDgetrf_bufferSize(long m, long n, size_t* buffer_size);

cdlsStatus_t cdlsDgetrf(cdlsHandle_t handle, long m, long n, double* C,
                        long ldc, void* workspace, int* pivot_count);

cdlsStatus_t cdlsDgetrf_static_pivoting(cdlsHandle_t handle, long m, long n,
                                        double* C, long ldc, void* workspace,
                                        int* pivot_count, double criterion);

#if defined(__cplusplus)
}
#endif
