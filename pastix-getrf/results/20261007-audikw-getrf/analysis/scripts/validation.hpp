// Benchmark matrix generation and residual checks adapted from
// ck_getrf/src/benchmark.cpp. Retained upstream notices:
// third_party/ck_getrf_notices.
#include <cdls/getrf.h>
#include <cdls/internal/common.h>
#include <rocsolver/rocsolver.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(hipError_t e) {
  if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e));
}
void check(rocblas_status e) {
  if (e != rocblas_status_success)
    throw std::runtime_error("rocBLAS/rocSOLVER status " + std::to_string(e));
}
template <class T>
struct Buffer {
  T* p = nullptr;
  explicit Buffer(std::size_t count) {
    check(hipMalloc(reinterpret_cast<void**>(&p),
                    std::max<std::size_t>(count, 1) * sizeof(T)));
  }
  ~Buffer() {
    if (p) (void)hipFree(p);
  }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
};
void check(cdlsStatus_t e) {
  if (e != CDLS_STATUS_SUCCESS)
    throw std::runtime_error("CDLS status " + std::to_string(e));
}
struct Context {
  cdlsHandle_t cdls{};
  hipStream_t stream{};
  rocblas_handle blas{};
  hipEvent_t begin{}, end{};
  Context() {
    check(cdlsCreate(&cdls));
    stream = static_cast<hipStream_t>(cdls->stream_work);
    check(rocblas_create_handle(&blas));
    check(rocblas_set_stream(blas, stream));
    check(rocblas_set_pointer_mode(blas, rocblas_pointer_mode_host));
    check(hipEventCreate(&begin));
    check(hipEventCreate(&end));
  }
  ~Context() {
    (void)hipEventDestroy(begin);
    (void)hipEventDestroy(end);
    (void)rocblas_destroy_handle(blas);
    (void)cdlsDestroy(cdls);
  }
};
constexpr double sentinel = -987654321.25;
constexpr double eps = std::numeric_limits<double>::epsilon();

__global__ void unpack(int n, int lda, const double* lu, double* l, double* u) {
  const std::size_t total = static_cast<std::size_t>(n) * n;
  for (std::size_t idx =
           blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const int i = idx % n, j = idx / n;
    const double v = lu[i + static_cast<std::size_t>(j) * lda];
    l[idx] = i == j ? 1 : (i > j ? v : 0);
    u[idx] = i <= j ? v : 0;
  }
}

double residual(Context& ctx, int n, int lda, const double* lu,
                const std::vector<double>& original) {
  if (n == 0) return 0;
  const std::size_t count = static_cast<std::size_t>(n) * n;
  Buffer<double> l(count), u(count), r(count);
  hipLaunchKernelGGL(unpack,
                     dim3(std::min<std::size_t>((count + 255) / 256, 65535)),
                     dim3(256), 0, ctx.stream, n, lda, lu, l.p, u.p);
  check(hipGetLastError());
  check(hipMemcpy2DAsync(r.p, n * sizeof(double), original.data(),
                         lda * sizeof(double), n * sizeof(double), n,
                         hipMemcpyHostToDevice, ctx.stream));
  const double minus = -1, plus = 1;
  // Full reconstruction for EVERY size, independently of the factorization
  // path.
  check(rocblas_dgemm(ctx.blas, rocblas_operation_none, rocblas_operation_none,
                      n, n, n, &minus, l.p, n, u.p, n, &plus, r.p, n));
  std::vector<double> host(count);
  check(hipMemcpyAsync(host.data(), r.p, count * sizeof(double),
                       hipMemcpyDeviceToHost, ctx.stream));
  check(hipStreamSynchronize(ctx.stream));
  long double error = 0, norm = 0;
  for (std::size_t idx = 0; idx < count; ++idx) {
    if (!std::isfinite(host[idx]))
      return std::numeric_limits<double>::infinity();
    error += static_cast<long double>(host[idx]) * host[idx];
    const double a = original[idx % n + (idx / n) * lda];
    norm += static_cast<long double>(a) * a;
  }
  return norm == 0 ? (error == 0 ? 0 : std::numeric_limits<double>::infinity())
                   : std::sqrt(error / norm) / (n * eps);
}

double cpu_residual(int n, int lda, const std::vector<double>& a,
                    const std::vector<double>& lu) {
  long double error = 0, norm = 0;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      long double product = 0;
      for (int k = 0; k <= std::min(i, j); ++k)
        product += (i == k ? 1.L
                           : static_cast<long double>(
                                 lu[i + static_cast<std::size_t>(k) * lda])) *
                   lu[k + static_cast<std::size_t>(j) * lda];
      const long double v = a[i + static_cast<std::size_t>(j) * lda];
      error += (v - product) * (v - product);
      norm += v * v;
    }
  return norm == 0 ? (error == 0 ? 0 : std::numeric_limits<double>::infinity())
                   : std::sqrt(error / norm) / (std::max(n, 1) * eps);
}


} // namespace
