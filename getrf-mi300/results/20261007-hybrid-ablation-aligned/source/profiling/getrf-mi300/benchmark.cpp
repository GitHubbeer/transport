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

std::vector<double> matrix(int n, int lda, const std::string& kind, int zero) {
  std::vector<double> a(static_cast<std::size_t>(lda) * n, sentinel);
  std::mt19937_64 gen(20261001);
  std::uniform_real_distribution<double> dist(-1, 1);
  if (kind == "reciprocal_safe_edges" || kind == "reciprocal_mixed_edges") {
    // A=L*diag(U) keeps the pivot values exact, including both ends of the
    // reciprocal-safe interval. Nontrivial L21 checks normalization and
    // padding.
    const double low = 0x1p-500, high = 0x1p500;
    std::vector<double> pivots{low, high, std::nextafter(low, high),
                               std::nextafter(high, low), 3.0};
    if (kind == "reciprocal_mixed_edges") {
      pivots.push_back(std::nextafter(low, 0.0));
      pivots.push_back(
          std::nextafter(high, std::numeric_limits<double>::infinity()));
    }
    for (int j = 0; j < n; ++j) {
      const double pivot = (j % 2 ? -1.0 : 1.0) * pivots[j % pivots.size()];
      for (int i = 0; i < n; ++i)
        a[i + static_cast<std::size_t>(j) * lda] =
            i == j ? pivot : (i > j ? 0.125 * dist(gen) * pivot : 0.0);
    }
    return a;
  }
  if (kind == "known_lu") {
    // Well-conditioned constructed factors, but A need not be diagonally
    // dominant.
    std::vector<double> l(static_cast<std::size_t>(n) * n, 0), u(l.size(), 0);
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i) {
        if (i > j) l[i + static_cast<std::size_t>(j) * n] = 0.05 * dist(gen);
        if (i == j) l[i + static_cast<std::size_t>(j) * n] = 1;
        if (i < j) u[i + static_cast<std::size_t>(j) * n] = 0.1 * dist(gen);
        if (i == j) u[i + static_cast<std::size_t>(j) * n] = i % 2 ? -0.5 : 0.5;
      }
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i) {
        long double v = 0;
        for (int k = 0; k <= std::min(i, j); ++k)
          v +=
              static_cast<long double>(l[i + static_cast<std::size_t>(k) * n]) *
              u[k + static_cast<std::size_t>(j) * n];
        a[i + static_cast<std::size_t>(j) * lda] = static_cast<double>(v);
      }
    return a;
  }
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      double v = 0;
      if (kind == "dense" || kind == "column_scaled" || kind == "scale_low" ||
          kind == "scale_high" || kind == "random")
        v = kind == "random" ? dist(gen) : i == j ? n + 1.0 : dist(gen);
      else if (kind == "signed")
        v = i == j ? (i % 2 ? -n - 1.0 : n + 1.0) : dist(gen);
      else if (kind == "identity" || kind == "singular")
        v = i == j ? 1.0 : 0;
      if (kind == "column_scaled") v *= j % 2 ? 1e200 : 1e-200;
      if (kind == "scale_low") v = std::ldexp(v, -600);
      if (kind == "scale_high") v = std::ldexp(v, 600);
      a[i + static_cast<std::size_t>(j) * lda] = v;
    }
  if (kind == "singular") a[zero + static_cast<std::size_t>(zero) * lda] = 0;
  return a;
}

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

struct Options {
  std::string method = "auto", matrix = "dense", reference = "npvt";
  int n = 1024, lda = 0, repeats = 30, warmup = 5, device = 0, offset = 0;
  bool validate_only = false, samples = false;
};
struct Timing {
  double gpu_ms, submit_ms, wall_ms;
};

bool run_case(Context& ctx, const Options& opt, int n, int lda,
              const std::string& kind = "dense", int zero = -1) {
  const auto host = matrix(n, lda, kind, zero);
  const std::size_t bytes = host.size() * sizeof(double);
  Buffer<double> original(host.size()), ours(host.size() + opt.offset),
      ref(host.size() + opt.offset);
  auto* ours_matrix = ours.p + opt.offset;
  auto* ref_matrix = ref.p + opt.offset;
  Buffer<int> pivots(n);
  Buffer<int> ours_info(1), ref_info(1);
  std::size_t workspace_bytes = 0;
  check(cdlsDgetrf_bufferSize(n, n, &workspace_bytes));
  Buffer<unsigned char> workspace(workspace_bytes);
  if (bytes)
    check(hipMemcpyAsync(original.p, host.data(), bytes, hipMemcpyHostToDevice,
                         ctx.stream));
  // Preallocate rocSOLVER's internal workspace outside the timed interval.
  check(rocblas_start_device_memory_size_query(ctx.blas));
  auto query =
      opt.reference == "pivot"
          ? rocsolver_dgetrf(ctx.blas, n, n, ref_matrix, lda, pivots.p,
                             ref_info.p)
          : rocsolver_dgetrf_npvt(ctx.blas, n, n, ref_matrix, lda, ref_info.p);
  if (query != rocblas_status_success &&
      query != rocblas_status_size_increased &&
      query != rocblas_status_size_unchanged)
    check(query);
  size_t roc_workspace_bytes = 0;
  check(rocblas_stop_device_memory_size_query(ctx.blas, &roc_workspace_bytes));
  Buffer<unsigned char> roc_workspace(roc_workspace_bytes);
  if (roc_workspace_bytes)
    check(
        rocblas_set_workspace(ctx.blas, roc_workspace.p, roc_workspace_bytes));
  const int initial_counter = opt.method == "static" ? 0 : 7;
  check(hipMemcpyAsync(ours_info.p, &initial_counter, sizeof(int),
                       hipMemcpyHostToDevice, ctx.stream));
  auto invoke = [&](bool cdls, bool timed) {
    double* a = cdls ? ours_matrix : ref_matrix;
    check(hipMemcpyAsync(a, original.p, bytes, hipMemcpyDeviceToDevice,
                         ctx.stream));
    check(hipStreamSynchronize(
        ctx.stream));  // input restoration excluded from all clocks
    const auto wall_start = std::chrono::steady_clock::now();
    if (timed) check(hipEventRecord(ctx.begin, ctx.stream));
    const auto submit_start = std::chrono::steady_clock::now();
    if (cdls && opt.method == "static") {
      check(hipMemsetAsync(ours_info.p, 0, sizeof(int), ctx.stream));
      check(cdlsDgetrf_static_pivoting(ctx.cdls, n, n, a, lda, workspace.p,
                                       ours_info.p, 0.0));
    } else if (cdls)
      check(cdlsDgetrf(ctx.cdls, n, n, a, lda, workspace.p, ours_info.p));
    else if (opt.reference == "pivot")
      check(rocsolver_dgetrf(ctx.blas, n, n, a, lda, pivots.p, ref_info.p));
    else
      check(rocsolver_dgetrf_npvt(ctx.blas, n, n, a, lda, ref_info.p));
    const auto submit_end = std::chrono::steady_clock::now();
    if (timed) {
      check(hipEventRecord(ctx.end, ctx.stream));
      check(hipEventSynchronize(ctx.end));
      const auto wall_end = std::chrono::steady_clock::now();
      float ms = 0;
      check(hipEventElapsedTime(&ms, ctx.begin, ctx.end));
      return Timing{
          ms,
          std::chrono::duration<double, std::milli>(submit_end - submit_start)
              .count(),
          std::chrono::duration<double, std::milli>(wall_end - wall_start)
              .count()};
    }
    check(hipStreamSynchronize(ctx.stream));
    return Timing{};
  };

  // Warm both implementations and let rocSOLVER allocate its cached workspace.
  for (int i = 0; i < opt.warmup; ++i) {
    invoke(true, false);
    invoke(false, false);
  }
  std::vector<double> cdls_ms, roc_ms, cdls_submit, roc_submit, cdls_wall,
      roc_wall;
  auto sample = [&](bool ours) {
    const auto t = invoke(ours, true);
    (ours ? cdls_ms : roc_ms).push_back(t.gpu_ms);
    (ours ? cdls_submit : roc_submit).push_back(t.submit_ms);
    (ours ? cdls_wall : roc_wall).push_back(t.wall_ms);
  };
  if (!opt.validate_only)
    for (int i = 0; i < opt.repeats; ++i) {
      // Alternate order to reduce systematic thermal/clock bias.
      if (i % 2) {
        sample(false);
        sample(true);
      } else {
        sample(true);
        sample(false);
      }
    }

  int ci = -1, ri = -1;
  check(hipMemcpyAsync(&ci, ours_info.p, sizeof(int), hipMemcpyDeviceToHost,
                       ctx.stream));
  check(hipMemcpyAsync(&ri, ref_info.p, sizeof(int), hipMemcpyDeviceToHost,
                       ctx.stream));
  std::vector<double> c(host.size()), r(host.size());
  if (bytes) {
    check(hipMemcpyAsync(c.data(), ours_matrix, bytes, hipMemcpyDeviceToHost,
                         ctx.stream));
    check(hipMemcpyAsync(r.data(), ref_matrix, bytes, hipMemcpyDeviceToHost,
                         ctx.stream));
  }
  check(hipStreamSynchronize(ctx.stream));
  const int expected =
      n == 0 ? 0 : (kind == "zero" ? 1 : (kind == "singular" ? zero + 1 : 0));
  bool pass = ci == initial_counter && ri == expected;
  std::vector<double> reference_input;
  if (opt.reference == "pivot") reference_input = host;
  if (opt.reference == "pivot") {
    std::vector<int> ipiv(n);
    check(hipMemcpy(ipiv.data(), pivots.p, n * sizeof(int),
                    hipMemcpyDeviceToHost));
    for (int k = 0; k < n; ++k) {
      if (ipiv[k] < 1 || ipiv[k] > n)
        throw std::runtime_error("Invalid pivot index");
      for (int col = 0; col < n; ++col)
        std::swap(
            reference_input[k + static_cast<size_t>(col) * lda],
            reference_input[ipiv[k] - 1 + static_cast<size_t>(col) * lda]);
    }
  }
  long double diff[2]{}, norm[2]{};
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < lda; ++i) {
      const std::size_t idx = i + static_cast<std::size_t>(j) * lda;
      if (i >= n) {
        pass = pass && c[idx] == sentinel && r[idx] == sentinel;
        continue;
      }
      pass = pass && std::isfinite(c[idx]) && std::isfinite(r[idx]);
      const int tri = i > j ? 0 : 1;
      const long double d = static_cast<long double>(c[idx]) - r[idx];
      diff[tri] += d * d;
      norm[tri] += static_cast<long double>(r[idx]) * r[idx];
    }
  const double l_error = std::sqrt(diff[0] / std::max(norm[0], 1.L));
  const double u_error = std::sqrt(diff[1] / std::max(norm[1], 1.L));
  const double tolerance = 64 * std::max(n, 1) * eps;
  const double c_res = residual(ctx, n, lda, ours_matrix, host);
  const double r_res =
      residual(ctx, n, lda, ref_matrix,
               opt.reference == "pivot" ? reference_input : host);
  pass = pass &&
         (opt.reference == "pivot" || kind == "random" ||
          (l_error <= tolerance && u_error <= tolerance)) &&
         c_res <= 100 && r_res <= 100;
  double cpu_c = 0, cpu_r = 0;
  if (n <= 128) {
    cpu_c = cpu_residual(n, lda, host, c);
    cpu_r = cpu_residual(n, lda,
                         opt.reference == "pivot" ? reference_input : host, r);
    pass = pass && cpu_c <= 100 && cpu_r <= 100;
  }
  std::cout << std::scientific << std::setprecision(4) << "validate n=" << n
            << " lda=" << lda << " kind=" << kind << " info=" << ci << '/' << ri
            << " L_rel=" << l_error << " U_rel=" << u_error
            << " residual_scaled=" << c_res << '/' << r_res;
  if (n <= 128) std::cout << " cpu_residual_scaled=" << cpu_c << '/' << cpu_r;
  std::cout << ' ' << (pass ? "PASS" : "FAIL") << '\n';

  if (!cdls_ms.empty()) {
    if (opt.samples)
      for (std::size_t i = 0; i < cdls_ms.size(); ++i)
        std::cout << std::fixed << std::setprecision(9) << "sample n=" << n
                  << " lda=" << lda << " iteration=" << i
                  << " first=" << (i % 2 ? "rocsolver" : "cdls")
                  << " cdls_ms=" << cdls_ms[i] << " rocsolver_ms=" << roc_ms[i]
                  << " cdls_submit_ms=" << cdls_submit[i]
                  << " roc_submit_ms=" << roc_submit[i]
                  << " cdls_wall_ms=" << cdls_wall[i]
                  << " roc_wall_ms=" << roc_wall[i] << '\n';
    auto median = [](std::vector<double> v) {
      std::sort(v.begin(), v.end());
      const auto mid = v.size() / 2;
      return v.size() % 2 ? v[mid] : (v[mid - 1] + v[mid]) / 2;
    };
    const double ct = median(cdls_ms), rt = median(roc_ms);
    const double flops = static_cast<double>(n) * (n - 1.0) * (4.0 * n + 1) / 6;
    std::cout << std::fixed << std::setprecision(6) << "bench n=" << n
              << " lda=" << lda << " repeats=" << opt.repeats
              << " cdls_median_ms=" << ct << " rocsolver_median_ms=" << rt
              << " cdls_mean_ms="
              << std::accumulate(cdls_ms.begin(), cdls_ms.end(), 0.) /
                     cdls_ms.size()
              << " rocsolver_mean_ms="
              << std::accumulate(roc_ms.begin(), roc_ms.end(), 0.) /
                     roc_ms.size()
              << " cdls_gflops=" << flops / (ct * 1e6)
              << " workspace_bytes=" << workspace_bytes
              << " roc_workspace_bytes=" << roc_workspace_bytes
              << " cdls_submit_ms=" << median(cdls_submit)
              << " roc_submit_ms=" << median(roc_submit)
              << " cdls_wall_ms=" << median(cdls_wall)
              << " roc_wall_ms=" << median(roc_wall)
              << " rocsolver_gflops=" << flops / (rt * 1e6)
              << " speedup=" << rt / ct << '\n';
  }
  return pass;
}

}  // namespace

int main(int argc, char** argv) try {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--validate-only")
      opt.validate_only = true;
    else if (arg == "--method") {
      if (++i >= argc) throw std::runtime_error("Missing method");
      opt.method = argv[i];
      if (opt.method != "auto" && opt.method != "static")
        throw std::runtime_error("method: auto or static");
    } else if (arg == "--matrix") {
      if (++i >= argc) throw std::runtime_error("Missing matrix kind");
      opt.matrix = argv[i];
      if (opt.matrix != "dense" && opt.matrix != "signed" &&
          opt.matrix != "column_scaled" && opt.matrix != "known_lu" &&
          opt.matrix != "identity" && opt.matrix != "random" &&
          opt.matrix != "scale_low" && opt.matrix != "scale_high")
        throw std::runtime_error(
            "matrix: dense, signed, column_scaled, known_lu or identity");
    } else if (arg == "--reference") {
      if (++i >= argc) throw std::runtime_error("Missing reference");
      opt.reference = argv[i];
      if (opt.reference != "npvt" && opt.reference != "pivot")
        throw std::runtime_error("reference: npvt or pivot");
    } else if (arg == "--samples")
      opt.samples = true;
    else if (arg == "--help") {
      std::cout << "cdls_getrf_bench [--n N] [--lda LDA] [--offset 0|1] "
                   "[--method auto|static] [--reference npvt|pivot] [--matrix "
                   "dense|signed|column_scaled|known_lu|identity|random|scale_"
                   "low|scale_high] [--warmup W] [--repeats R] [--samples] "
                   "[--validate-only]\n";
      return 0;
    } else {
      if (++i >= argc) throw std::runtime_error("Missing value for " + arg);
      std::size_t consumed = 0;
      const int value = std::stoi(argv[i], &consumed);
      if (consumed != std::string(argv[i]).size())
        throw std::runtime_error("Invalid integer");
      if (arg == "--n")
        opt.n = value;
      else if (arg == "--lda")
        opt.lda = value;
      else if (arg == "--repeats")
        opt.repeats = value;
      else if (arg == "--warmup")
        opt.warmup = value;
      else if (arg == "--device")
        opt.device = value;
      else if (arg == "--offset")
        opt.offset = value;
      else
        throw std::runtime_error("Unknown argument " + arg);
    }
  }
  if (opt.n <= 0 || opt.n > 65536 || opt.lda < 0 ||
      (opt.lda && opt.lda < std::max(1, opt.n)) || opt.repeats < 1 ||
      opt.warmup < 1 || opt.device < 0 || opt.offset < 0 || opt.offset > 1)
    throw std::runtime_error(
        "Require 0 <= n <= 65536, lda >= max(1,n), repeats/warmup >= 1, device "
        ">= 0");
  check(hipSetDevice(opt.device));
  hipDeviceProp_t prop{};
  check(hipGetDeviceProperties(&prop, opt.device));
  const std::string arch =
      std::string(prop.gcnArchName)
          .substr(0, std::string(prop.gcnArchName).find(':'));
  if (arch != "gfx90a" && arch != "gfx940" && arch != "gfx941" &&
      arch != "gfx942")
    throw std::runtime_error("FP64 MFMA requires gfx90a/gfx94x for this build");
  int runtime = 0;
  check(hipRuntimeGetVersion(&runtime));
  char solver_version[256]{}, blas_version[256]{};
  check(rocsolver_get_version_string(solver_version, sizeof(solver_version)));
  check(rocblas_get_version_string(blas_version, sizeof(blas_version)));
  std::cout << "device=" << prop.name << " arch=" << prop.gcnArchName
            << " HIP=" << runtime << " CUs=" << prop.multiProcessorCount
            << " memory_bytes=" << prop.totalGlobalMem
            << " rocSOLVER=" << solver_version << " rocBLAS=" << blas_version
            << " baseline="
            << (opt.reference == "pivot" ? "rocsolver_dgetrf"
                                         : "rocsolver_dgetrf_npvt")
            << " fp64 seed=20261001 method=" << opt.method
            << " matrix=" << opt.matrix << " n=" << opt.n << " lda=" << opt.lda
            << " offset=" << opt.offset << " warmup=" << opt.warmup
            << " repeats=" << opt.repeats << '\n';
  Context ctx;
  const bool pass =
      run_case(ctx, opt, opt.n, opt.lda ? opt.lda : opt.n, opt.matrix);
  return pass ? 0 : 1;
} catch (const std::exception& e) {
  std::cerr << "ERROR: " << e.what() << '\n';
  return 1;
}
