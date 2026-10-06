#pragma once
#include <cdls/internal/common.h>
#include <rocblas/rocblas.h>

struct RocmHandle {
  cdlsHandle common;
  rocblas_handle blas;
};
