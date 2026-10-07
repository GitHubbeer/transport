#include <cdls/getrf.h>
#include <cdls/internal/common.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
#ifdef CDLS_TEST_ROCM
#include <hip/hip_runtime_api.h>
#define GPU(name) hip##name
#define GPU_SUCCESS hipSuccess
#define GPU_H2D hipMemcpyHostToDevice
#define GPU_D2H hipMemcpyDeviceToHost
#else
#include <cuda_runtime_api.h>
#define GPU(name) cuda##name
#define GPU_SUCCESS cudaSuccess
#define GPU_H2D cudaMemcpyHostToDevice
#define GPU_D2H cudaMemcpyDeviceToHost
#endif

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
template <class Status>
void gpu_check(Status status) {
  require(status == GPU_SUCCESS, "GPU runtime failure");
}
struct DeviceBuffer {
  void* data = nullptr;
  explicit DeviceBuffer(std::size_t bytes) {
    gpu_check(GPU(Malloc)(&data, bytes));
  }
  ~DeviceBuffer() { (void)GPU(Free)(data); }
};
struct Handle {
  cdlsHandle_t data = nullptr;
  Handle() {
    require(cdlsCreate(&data) == CDLS_STATUS_SUCCESS, "create handle");
  }
  ~Handle() { (void)cdlsDestroy(data); }
};

// Full reconstruction for small tiles; large cases check three independent
// signed projections A*v == L*(U*v), exercising every factor entry in O(n^2).
void check_residual(const std::vector<double>& a, const std::vector<double>& lu,
                    int n, int lda) {
  double error = 0, scale = 0;
  if (n <= 257) {
    for (int col = 0; col < n; ++col)
      for (int row = 0; row < n; ++row) {
        long double value = row <= col ? lu[row + std::size_t(col) * lda] : 0;
        for (int k = 0; k < std::min(row, col + 1); ++k)
          value += static_cast<long double>(lu[row + std::size_t(k) * lda]) *
                   lu[k + std::size_t(col) * lda];
        const double expected = a[row + std::size_t(col) * lda];
        require(std::isfinite(static_cast<double>(value)),
                "nonfinite reconstruction");
        error =
            std::max(error, std::abs(static_cast<double>(value) - expected));
        scale = std::max(scale, std::abs(expected));
      }
  } else {
    std::mt19937 gen(773);
    for (int trial = 0; trial < 3; ++trial) {
      std::vector<double> v(n), av(n, 0), uv(n, 0), luv(n, 0);
      for (auto& x : v) x = (gen() & 1) ? 1.0 : -1.0;
      for (int col = 0; col < n; ++col) {
        for (int row = 0; row < n; ++row)
          av[row] += a[row + std::size_t(col) * lda] * v[col];
        for (int row = 0; row <= col; ++row)
          uv[row] += lu[row + std::size_t(col) * lda] * v[col];
      }
      luv = uv;
      for (int col = 0; col < n; ++col)
        for (int row = col + 1; row < n; ++row)
          luv[row] += lu[row + std::size_t(col) * lda] * uv[col];
      for (int row = 0; row < n; ++row) {
        require(std::isfinite(luv[row]), "nonfinite projection");
        error = std::max(error, std::abs(luv[row] - av[row]));
        scale = std::max(scale, std::abs(av[row]));
      }
    }
  }
  require(error / std::max(scale, 1e-300) < 2e-11,
          "LU residual exceeded tolerance");
}

void numerical_case(cdlsHandle_t handle, int n, int padding, bool offset,
                    bool static_pivot, int exponent = 0) {
  const int lda = n + padding;
  const std::size_t cells = std::size_t(lda) * n;
  std::vector<double> a(cells, -1234.5), factors(cells);
  std::mt19937 gen(2026 + n);
  std::uniform_real_distribution<double> random(-0.5, 0.5);
  for (int col = 0; col < n; ++col)
    for (int row = 0; row < n; ++row)
      a[row + std::size_t(col) * lda] =
          std::ldexp(row == col ? n + 1.0 : random(gen), exponent);
  size_t bytes = 0;
  require(cdlsDgetrf_bufferSize(n, n, &bytes) == CDLS_STATUS_SUCCESS,
          "buffer size");
  DeviceBuffer matrix((cells + offset) * sizeof(double)), workspace(bytes),
      count(sizeof(int));
  auto* base = static_cast<double*>(matrix.data) + offset;
  const int sentinel = 7;
  for (int repeat = 0; repeat < 2; ++repeat) {
    gpu_check(GPU(Memcpy)(base, a.data(), cells * sizeof(double), GPU_H2D));
    gpu_check(GPU(Memcpy)(count.data, &sentinel, sizeof(int), GPU_H2D));
    auto status = static_pivot
                      ? cdlsDgetrf_static_pivoting(
                            handle, n, n, base, lda, workspace.data,
                            static_cast<int*>(count.data), 0.0)
                      : cdlsDgetrf(handle, n, n, base, lda, workspace.data,
                                   static_cast<int*>(count.data));
    require(status == CDLS_STATUS_SUCCESS, "factorization status");
    gpu_check(GPU(DeviceSynchronize)());
    gpu_check(
        GPU(Memcpy)(factors.data(), base, cells * sizeof(double), GPU_D2H));
    int actual = 0;
    gpu_check(GPU(Memcpy)(&actual, count.data, sizeof(int), GPU_D2H));
    require(actual == sentinel, "unexpected pivot_count modification");
    for (int col = 0; col < n; ++col)
      for (int row = n; row < lda; ++row)
        require(factors[row + std::size_t(col) * lda] == -1234.5,
                "padding overwritten");
    check_residual(a, factors, n, lda);
  }
  std::cout << "PASS n=" << n << " lda=" << lda << " offset=" << offset
            << " static=" << static_pivot << " scale=2^" << exponent << '\n';
}

void static_counter_case(cdlsHandle_t handle, int n) {
  const int lda = n + 3;
  std::vector<double> a(std::size_t(lda) * n, 0);
  for (int k = 0; k < n; ++k)
    a[k + std::size_t(k) * lda] = k % 64 == 0 ? 1.0 : (k % 2 ? -1e-6 : 1e-6);
  size_t bytes = 0;
  require(cdlsDgetrf_bufferSize(n, n, &bytes) == CDLS_STATUS_SUCCESS,
          "static buffer size");
  DeviceBuffer matrix(a.size() * sizeof(double)), workspace(bytes),
      count(sizeof(int));
  gpu_check(
      GPU(Memcpy)(matrix.data, a.data(), a.size() * sizeof(double), GPU_H2D));
  const int initial = 7;
  gpu_check(GPU(Memcpy)(count.data, &initial, sizeof(int), GPU_H2D));
  require(
      cdlsDgetrf_static_pivoting(
          handle, n, n, static_cast<double*>(matrix.data), lda, workspace.data,
          static_cast<int*>(count.data), 0.01) == CDLS_STATUS_SUCCESS,
      "static factorization status");
  gpu_check(GPU(DeviceSynchronize)());
  int actual = 0;
  gpu_check(GPU(Memcpy)(&actual, count.data, sizeof(int), GPU_D2H));
  require(actual == initial + n - (n + 63) / 64, "static replacement count");
  gpu_check(
      GPU(Memcpy)(a.data(), matrix.data, a.size() * sizeof(double), GPU_D2H));
  for (int col = 0; col < n; ++col)
    for (int row = 0; row < n; ++row) {
      const double expected = row != col      ? 0
                              : row % 64 == 0 ? 1.0
                              : row % 2       ? -0.01
                                              : 0.01;
      require(a[row + std::size_t(col) * lda] == expected,
              "static factor/sign mismatch");
    }
}

int main() try {
  Handle handle;
  size_t bytes = 0;
  require(cdlsDgetrf_bufferSize(0, 0, &bytes) == CDLS_STATUS_INVALID_VALUE,
          "zero extent validation");
  require(cdlsDgetrf_bufferSize(64, 65, &bytes) == CDLS_STATUS_INVALID_VALUE,
          "square validation");
  require(cdlsDgetrf_bufferSize(64, 64, nullptr) == CDLS_STATUS_INVALID_VALUE,
          "null size validation");
  require(cdlsDgetrf(nullptr, 64, 64, nullptr, 64, nullptr, nullptr) ==
              CDLS_STATUS_INVALID_VALUE,
          "null factorization validation");
  for (int n : {1,    2,    7,    8,    15,   16,   17,   31,  32,  33,
                63,   64,   65,   127,  128,  129,  257,  513, 895, 896,
                1025, 2559, 2560, 2561, 4096, 4097, 4225, 8193})
    numerical_case(handle.data, n, n % 2 ? 3 : 0, n % 3 == 0, false);
  for (int n : {2, 33, 64, 65, 129, 513}) {
    numerical_case(handle.data, n, 5, true, true);
    static_counter_case(handle.data, n);
  }
  for (int exponent : {-600, 600})
    for (int n : {31, 63, 65, 129})
      numerical_case(handle.data, n, 3, true, false, exponent);
  std::cout << "PASS: GETRF numerics, static pivots, padding, reuse and "
               "extreme scales\n";
} catch (const std::exception& error) {
  std::cerr << "FAIL: " << error.what() << '\n';
  return 1;
}
