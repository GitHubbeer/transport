#include <cdls/getrf.h>

#include <climits>

#include "../common/getrf/hybrid.h"
#include "getrf/tiled.h"
#include "handle.h"

using namespace cdls::rocm_detail::getrf;

namespace {
bool supported_square(long m, long n) {
  return m == n && m > 0 && m <= INT_MAX &&
         workspace_size(static_cast<int>(m)) != 0;
}

// CK's zero-pivot report has its own workspace slot. The public pivot_count
// retains its existing meaning as a static-pivot replacement counter.
std::size_t buffer_bytes(int n) { return workspace_size(n) + sizeof(int); }
int* internal_info(void* workspace, int n) {
  return reinterpret_cast<int*>(static_cast<char*>(workspace) +
                                workspace_size(n));
}

bool valid_factorization(cdlsHandle_t handle, long m, long n,
                         const double* matrix, long lda,
                         const void* workspace) {
  return supported_square(m, n) && handle && matrix && workspace && lda >= m &&
         reinterpret_cast<std::uintptr_t>(workspace) % alignof(int) == 0;
}

cdlsStatus_t factorize_no_pivot(cdlsHandle_t handle, int n, double* matrix,
                                long lda, void* workspace) {
  const auto stream = static_cast<hipStream_t>(handle->stream_work);
  int* info = internal_info(workspace, n);
  if (n <= tile_size)
    return launch_small<true>(n, matrix, lda, info, stream, 0) == hipSuccess
               ? CDLS_STATUS_SUCCESS
               : CDLS_STATUS_EXECUTION_FAILED;
  if (hipMemsetAsync(info, 0, sizeof(int), stream) != hipSuccess)
    return CDLS_STATUS_EXECUTION_FAILED;
  // CK production policy: 512-wide outer panels, fused tail <= 4096.
  if (n <= 4096 || lda > INT_MAX)
    return launch_prefix(n, n, matrix, lda, info, workspace, stream, 0) ==
                   hipSuccess
               ? CDLS_STATUS_SUCCESS
               : CDLS_STATUS_EXECUTION_FAILED;
  auto blas = reinterpret_cast<RocmHandle*>(handle)->blas;
  if (rocblas_set_stream(blas, stream) != rocblas_status_success)
    return CDLS_STATUS_EXECUTION_FAILED;
  const double minus = -1.0, plus = 1.0;
  return cdls::detail::getrf::factorize_hybrid(
      static_cast<long>(n), matrix, lda, 4096, tile_size,
      [](long remaining) { return std::min<long>(512, remaining); },
      [&](long remaining, double* base, int panels) {
        const auto offset = (base - matrix) / (static_cast<long>(lda) + 1);
        return launch_prefix(remaining, panels ? panels * tile_size : remaining,
                             base, lda, info, workspace, stream,
                             static_cast<int>(offset)) == hipSuccess
                   ? CDLS_STATUS_SUCCESS
                   : CDLS_STATUS_EXECUTION_FAILED;
      },
      [&](double* base, int trailing, int width) {
        return rocblas_dgemm(
                   blas, rocblas_operation_none, rocblas_operation_none,
                   trailing, trailing, width, &minus, base + width, lda,
                   base + static_cast<std::size_t>(width) * lda, lda, &plus,
                   base + width + static_cast<std::size_t>(width) * lda,
                   lda) == rocblas_status_success
                   ? CDLS_STATUS_SUCCESS
                   : CDLS_STATUS_EXECUTION_FAILED;
      });
}
}  // namespace

extern "C" cdlsStatus_t cdlsDgetrf_bufferSize(long m, long n, size_t* bytes) {
  if (!bytes || !supported_square(m, n)) return CDLS_STATUS_INVALID_VALUE;
  *bytes = buffer_bytes(static_cast<int>(m));
  return CDLS_STATUS_SUCCESS;
}

extern "C" cdlsStatus_t cdlsDgetrf(cdlsHandle_t handle, long m, long n,
                                   double* matrix, long lda, void* workspace,
                                   int* pivot_count) {
  if (!valid_factorization(handle, m, n, matrix, lda, workspace))
    return CDLS_STATUS_INVALID_VALUE;
  return factorize_no_pivot(handle, static_cast<int>(m), matrix, lda,
                            workspace);
}

extern "C" cdlsStatus_t cdlsDgetrf_static_pivoting(cdlsHandle_t handle, long m,
                                                   long n, double* matrix,
                                                   long lda, void* workspace,
                                                   int* pivot_count,
                                                   double criterion) {
  if (!valid_factorization(handle, m, n, matrix, lda, workspace) ||
      !pivot_count)
    return CDLS_STATUS_INVALID_VALUE;
  const auto stream = static_cast<hipStream_t>(handle->stream_work);
  const int extent = static_cast<int>(m);
  if (hipMemsetAsync(workspace, 0, buffer_bytes(extent), stream) != hipSuccess)
    return CDLS_STATUS_EXECUTION_FAILED;
  const int tiles = (extent - 1) / tile_size + 1;
  return launch_tiles<true>(extent, matrix, lda, tiles, tiles,
                            static_cast<int*>(workspace),
                            internal_info(workspace, extent), stream, 0,
                            pivot_count, criterion) == hipSuccess
             ? CDLS_STATUS_SUCCESS
             : CDLS_STATUS_EXECUTION_FAILED;
}
