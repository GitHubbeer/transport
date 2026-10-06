// SPDX-License-Identifier: MIT AND BSD-2-Clause
// Copyright (c) 2018-2025 Advanced Micro Devices, Inc. All rights reserved.
// CK primitives and accumulator transfer follow AMD's MIT-licensed examples:
// example/01_gemm/gemm_xdl_fp64.cpp and gridwise_gemm_xdlops_v2r3.hpp.
// Blocked left-looking organization follows TheCoreTeam/cdls
// src/rocm/getrf.cpp. Inner LU panel organization also follows official
// rocSOLVER rocm-6.4.2 roclapack_getrf.hpp /
// roclapack_getf2_specialized_kernels.hpp (BSD-2-Clause). See
// third_party/ck_getrf_notices for the retained upstream licenses.
#include "../../common/getrf/panel.h"
#include "../../common/getrf/schedule.h"
#include "../../common/getrf/triangular.h"
#include "config.h"
#include "static_pivot.h"
#define CK_EXPERIMENTAL_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM 1
// The pinned CK 6.4 headers use warpSize in constexpr expressions. HIP 7
// exposes it through a runtime conversion. This library supports only the
// wave64 gfx90a/gfx94x targets checked by CMake; scope the constant to CK's
// headers, after HIP headers have been included, without editing upstream CK.
#if HIP_VERSION_MAJOR >= 7
#define warpSize 64
#endif
#include "ck/tensor_operation/gpu/block/blockwise_gemm_xdlops.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/utility/synchronization.hpp"
#include "ck_tile/core.hpp"
#if HIP_VERSION_MAJOR >= 7
#undef warpSize
#endif
#include <algorithm>
#include <climits>
#include <cstdint>
#include <type_traits>

namespace cdls::rocm_detail::getrf {
namespace {
using namespace cdls::detail::getrf;
constexpr int B = tile_size, K = B / 4, Threads = (B / 32) * (B / 32) * 64;
constexpr int CStride = B + (padded_lds ? 1 : 0);
constexpr int CSpace = B * CStride;
constexpr int DiagSpace = padded_lds ? B * (B + 1) / 2 : B * B;
__device__ constexpr int diag_index(int row, int col) {
  return row + col * B - (padded_lds ? col * (col + 1) / 2 : 0);
}
constexpr int OperandStride = B + 1;
constexpr int OperandSpace = K * OperandStride;
template <int N>
using I = ck::Number<N>;
using ck::make_tuple;

struct Negate {
  __device__ void operator()(double& output, const double& input) const {
    output = -input;
  }
};

// As in CK wrapper_optimized_gemm.cpp, pad the LDS M/N stride by one.
// Store B transposed to the same K0,N,K1 layout to avoid stride-16 bank
// conflicts. Explicit double input AND accumulation. The 6.4.2 wrapper API
// accumulates non-integer inputs in float, so use the underlying CK block
// primitive.
constexpr auto ADesc = ck::make_naive_tensor_descriptor(
    make_tuple(I<K>{}, I<B>{}, I<1>{}),
    make_tuple(I<OperandStride>{}, I<1>{}, I<1>{}));
constexpr auto BDesc = ck::make_naive_tensor_descriptor(
    make_tuple(I<K>{}, I<B>{}, I<1>{}),
    make_tuple(I<OperandStride>{}, I<1>{}, I<1>{}));
template <int T>
using GemmFor = ck::BlockwiseGemmXdlops_k0mk1_k0nk1_m0n0m1n1m2m3m4n2_v1<
    T, double, double, double, decltype(ADesc), decltype(BDesc), 16, 16, 2,
    2 * Threads / T, 1>;

// Task broadcast aliases storage before the first use of LDS. A packed
// triangle leaves room for bank-conflict-free padded C columns below 64 KiB.
union Shared {
  int task;
  double ab[2][2][OperandSpace];
  struct {
    double c[CSpace], diag[DiagSpace];
  } lu;
};
static_assert(sizeof(Shared) <= 2 * B * B * sizeof(double));

__device__ void wait_ready(const int* flag, const int* second = nullptr) {
  if (threadIdx.x == 0) {
    // Avoid repeatedly invalidating caches while the producer is still running.
    // The flag only transitions 0->1 during this launch; acquire once
    // published.
    while (!__hip_atomic_load(flag, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT))
      __builtin_amdgcn_s_sleep(1);
    (void)__hip_atomic_load(flag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
    if (second) {
      while (!__hip_atomic_load(second, __ATOMIC_RELAXED,
                                __HIP_MEMORY_SCOPE_AGENT))
        __builtin_amdgcn_s_sleep(1);
      (void)__hip_atomic_load(second, __ATOMIC_ACQUIRE,
                              __HIP_MEMORY_SCOPE_AGENT);
    }
  }
  __syncthreads();
}

template <int T>
__device__ void fetch(const double* a, int n, std::int64_t lda, int row,
                      int col, int k, double (&ra)[B * K / T],
                      double (&rb)[B * K / T]) {
#pragma unroll
  for (int t = 0; t < B * K / T; ++t) {
    const int idx = threadIdx.x + t * T;
    const int ar = row + idx % B, ak = k + idx / B;
    const int bk = k + idx % K, bc = col + idx / K;
    ra[t] = ar < n ? a[ar + static_cast<std::size_t>(ak) * lda] : 0;
    rb[t] = bc < n ? a[bk + static_cast<std::size_t>(bc) * lda] : 0;
  }
}
template <int T>
__device__ void stage(Shared& s, int buf, const double (&ra)[B * K / T],
                      const double (&rb)[B * K / T]) {
#pragma unroll
  for (int t = 0; t < B * K / T; ++t) {
    const int idx = threadIdx.x + t * T;
    s.ab[buf][0][(idx / B) * OperandStride + idx % B] = ra[t];
    s.ab[buf][1][(idx % K) * OperandStride + idx / K] = rb[t];
  }
}

// The LU frontier has one source lane for the entire wave. Reading both
// FP64 halves into SGPRs avoids the LDS permute used by a general shuffle.
__device__ __forceinline__ double panel_broadcast(double value,
                                                  unsigned source) {
  const auto bits = ck_tile::bit_cast<std::uint64_t>(value);
  const std::uint64_t lo = static_cast<unsigned>(
      __builtin_amdgcn_readlane(static_cast<unsigned>(bits), source));
  const std::uint64_t hi = static_cast<unsigned>(
      __builtin_amdgcn_readlane(static_cast<unsigned>(bits >> 32), source));
  return ck_tile::bit_cast<double>(lo | (hi << 32));
}

template <bool Transposed = false>
struct TileView {
  double* data;
  __device__ double& operator()(int row, int col) const {
    return data[Transposed ? col + row * CStride : row + col * CStride];
  }
};
template <bool LowerTile, bool Transpose>
struct DiagonalView {
  const double* data;
  __device__ double operator()(int row, int col) const {
    return data[(LowerTile && !Transpose && !padded_lds)
                    ? col + row * B
                    : diag_index(row, col)];
  }
};

// Small inner panels avoid a full 64-double row live range. rocSOLVER uses
// width 16; width 8 measured faster with our fused single-barrier kernel.
template <int T>
__device__ __forceinline__ void diagonal(double* c, double* inverse, int size,
                                         int offset, int* info) {
  constexpr int P = inner_size;
  const int row = threadIdx.x;
  int first_zero = 0;
  for (int p = 0; p < size; p += P) {
    double x[P];
#pragma unroll
    for (int j = 0; j < P; ++j) x[j] = row < B ? c[row + (p + j) * CStride] : 0;
    if constexpr (lu_wave_panel && !lu_reciprocal) {
      // All panel rows reside in wave zero. Keep every lane in that wave
      // active during each broadcast, including rows outside a partial tile.
      if (row < 64) {
        unsigned panel_zeros = 0;
        ck::static_for<0, P, 1>{}([&](auto pivot_index) {
          constexpr int k = pivot_index;
          if (p + k < size) {
            const double pivot = panel_broadcast(x[k], p + k);
            if constexpr (lu_defer_info) {
              // Pivot broadcasts are uniform in wave zero. Accumulate without
              // entering a leader-only reporting branch at every pivot.
              panel_zeros |= static_cast<unsigned>(pivot == 0) << k;
            } else if (row == 0 && pivot == 0) {
              atomicCAS(info, 0, offset + p + k + 1);
            }
            if (row > p + k && row < size && pivot != 0) x[k] /= pivot;
#pragma unroll
            for (int j = k + 1; j < P; ++j) {
              const double upper = panel_broadcast(x[j], p + k);
              if (row > p + k && row < size)
                x[j] = __builtin_fma(-x[k], upper, x[j]);
            }
          }
        });
        if constexpr (lu_defer_info) {
          if (panel_zeros && first_zero == 0)
            first_zero = offset + p + __builtin_ctz(panel_zeros) + 1;
        }
      }
    } else {
      ck::static_for<0, P, 1>{}([&](auto pivot_index) {
        constexpr int k = pivot_index;
        if (p + k < size) {
          if (row == p + k) {
#pragma unroll
            for (int j = k; j < P; ++j) c[row + (p + j) * CStride] = x[j];
            if constexpr (lu_reciprocal) {
              // Unused diagonal LDS storage; distinct address for every pivot.
              // Keep extreme pivots on the original division path to avoid
              // reciprocal overflow/underflow. First publication barrier orders
              // it.
              inverse[p + k] = reciprocal_safe(x[k]) ? 1.0 / x[k] : 0.0;
            }
            if (x[k] == 0) atomicCAS(info, 0, offset + p + k + 1);
          }
          __syncthreads();
          if (row > p + k && row < size) {
            const double pivot = c[p + k + (p + k) * CStride];
            if (pivot != 0) {
              if constexpr (lu_reciprocal) {
                const double inv = inverse[p + k];
                if (inv != 0)
                  x[k] *= inv;
                else
                  x[k] /= pivot;
              } else
                x[k] /= pivot;
            }
#pragma unroll
            for (int j = k + 1; j < P; ++j)
              x[j] = __builtin_fma(-x[k], c[p + k + (p + j) * CStride], x[j]);
          }
          // Each pivot publishes a distinct LDS row. Retain the first full
          // barrier (also orders global info CAS), then wait before panel
          // stores.
          if constexpr (!lu_single_barrier) __syncthreads();
        }
      });
      if constexpr (lu_single_barrier) __syncthreads();
    }
    auto tile = TileView<>{c};
    cdls::detail::getrf::finish_lu_panel<B, P, T>(row, p, size, tile, x);
  }
  if constexpr (lu_defer_info && lu_wave_panel && !lu_reciprocal) {
    // Report before the leader reaches the tile's completion publication.
    // Later diagonal tasks depend on that publication; CAS retains earlier
    // tiles' or hybrid panels' first zero. The LDS pivot path reports inline.
    if (row == 0 && first_zero) atomicCAS(info, 0, first_zero);
  }
}

// A single tile needs neither MFMA updates nor a task/ready workspace.
// Reuse the small tiled path's thread count and LU panel routine; avoid
// the generic tile conversion and dependency machinery.
template <bool ResetInfo>
__global__ __launch_bounds__(2 * Threads) void getrf_small_kernel(
    int n, double* a, std::int64_t lda, int* info, int info_offset) {
  __shared__ double c[CSpace], inverse[B];
  if constexpr (ResetInfo) {
    if (threadIdx.x == 0) *info = 0;
  }
  for (int idx = threadIdx.x; idx < B * B; idx += 2 * Threads) {
    const int row = idx % B, col = idx / B;
    c[idx % B + (idx / B) * CStride] =
        row < n && col < n ? a[row + static_cast<std::size_t>(col) * lda] : 0;
  }
  __syncthreads();
  diagonal<2 * Threads>(c, inverse, n, info_offset, info);
  for (int idx = threadIdx.x; idx < B * B; idx += 2 * Threads) {
    const int row = idx % B, col = idx / B;
    if (row < n && col < n)
      a[row + static_cast<std::size_t>(col) * lda] =
          c[idx % B + (idx / B) * CStride];
  }
}

template <bool ResetInfo = false>
hipError_t launch_small(int n, double* a, std::int64_t lda, int* info,
                        hipStream_t stream, int info_offset) {
  hipLaunchKernelGGL((getrf_small_kernel<ResetInfo>), dim3(1),
                     dim3(2 * Threads), 0, stream, n, a, lda, info,
                     info_offset);
  return hipGetLastError();
}

// DPP16's quad permutation broadcasts within each four-lane RHS. For
// eight lanes, shift a quad's broadcast into its partner in the same row16.
// The row boundary is never crossed by an eight-lane RHS group.
template <int Lanes, int Source>
__device__ __forceinline__ double rhs_dpp(double value) {
  static_assert(Lanes == 4 || Lanes == 8);
  static_assert(Source >= 0 && Source < Lanes);
  const auto bits = ck_tile::bit_cast<std::uint64_t>(value);
  unsigned lo = __builtin_amdgcn_mov_dpp(static_cast<unsigned>(bits),
                                         (Source % 4) * 0x55, 0xf, 0xf, false);
  unsigned hi = __builtin_amdgcn_mov_dpp(static_cast<unsigned>(bits >> 32),
                                         (Source % 4) * 0x55, 0xf, 0xf, false);
  if constexpr (Lanes == 8) {
    const unsigned other_lo = __builtin_amdgcn_mov_dpp(
        lo, Source < 4 ? 0x114 : 0x104, 0xf, 0xf, false);
    const unsigned other_hi = __builtin_amdgcn_mov_dpp(
        hi, Source < 4 ? 0x114 : 0x104, 0xf, 0xf, false);
    const bool partner = (threadIdx.x % 8 < 4) != (Source < 4);
    lo = partner ? other_lo : lo;
    hi = partner ? other_hi : hi;
  }
  return ck_tile::bit_cast<double>(static_cast<std::uint64_t>(lo) |
                                   (static_cast<std::uint64_t>(hi) << 32));
}
template <int Lanes>
__device__ __forceinline__ double rhs_dpp(double value, int source) {
  switch (source) {
    case 0:
      return rhs_dpp<Lanes, 0>(value);
    case 1:
      return rhs_dpp<Lanes, 1>(value);
    case 2:
      return rhs_dpp<Lanes, 2>(value);
    case 3:
      return rhs_dpp<Lanes, 3>(value);
    default:
      if constexpr (Lanes == 8) {
        switch (source) {
          case 4:
            return rhs_dpp<Lanes, 4>(value);
          case 5:
            return rhs_dpp<Lanes, 5>(value);
          case 6:
            return rhs_dpp<Lanes, 6>(value);
          case 7:
            return rhs_dpp<Lanes, 7>(value);
        }
      }
      __builtin_unreachable();
  }
}

struct RocmSolveOps {
  __device__ __forceinline__ double divide(double value, double pivot) const {
    return pivot != 0 ? value / pivot : value;
  }
  template <int Lanes, class Rhs>
  __device__ __forceinline__ double broadcast(double value, int source, int k,
                                              int line, Rhs& rhs) const {
    if constexpr (trsm_wave_broadcast && B == 64) {
      if constexpr (trsm_dpp == 2 || (trsm_dpp == 1 && Lanes == 4))
        return rhs_dpp<Lanes>(value, source);
      return ck_tile::warp_shuffle(value,
                                   (threadIdx.x % 64 / Lanes) * Lanes + source);
    } else {
      if (threadIdx.x % Lanes == source) rhs(k, line) = value;
      __syncthreads();
      return rhs(k, line);
    }
  }
  // Output aliases the RHS tile; finish every reader before overwriting it.
  __device__ void finish_group() const {
    if constexpr (!(trsm_wave_broadcast && B == 64)) __syncthreads();
  }
  __device__ void finish() const { __syncthreads(); }
};

template <bool LowerTile, int T, bool Transpose, int DiagonalMode = 0>
__device__ __forceinline__ void solve(double* c, const double* d, int rows,
                                      int cols) {
  constexpr auto mode =
      !LowerTile          ? TriangularDiagonal::unit
      : DiagonalMode == 2 ? TriangularDiagonal::safe_reciprocals
      : DiagonalMode == 1 ? TriangularDiagonal::mixed_reciprocals
                          : TriangularDiagonal::raw;
  auto rhs = TileView<LowerTile>{c};
  const auto coefficients = DiagonalView<LowerTile, Transpose>{d};
  cdls::detail::getrf::forward_solve<B, T / B, mode>(
      threadIdx.x, LowerTile ? rows : cols, coefficients, rhs, rhs,
      RocmSolveOps{});
}

// Like cdls CUDA solve_right_upper, each CTA caches one exact reciprocal
// per safe U pivot in its private LDS copy. Global factors stay raw. A zero
// or extreme pivot retains guarded division; no approximate reciprocal is used.
template <bool Transpose>
__device__ __forceinline__ bool prepare_upper_diagonal(double* d, double* c) {
  int unsafe = 0;
  if (threadIdx.x < B) {
    const int index = diag_index(threadIdx.x, threadIdx.x);
    const double pivot = d[index];
    unsafe = !reciprocal_safe(pivot);
    if (!unsafe) d[index] = 1.0 / pivot;
  }
  // Only wave zero owns diagonal pivots. Its full-wave vote also covers
  // tile32: lanes outside the diagonal retain unsafe=0. One CTA barrier
  // publishes both the reciprocal values and the vote to every RHS subgroup.
  // Use a slot no solve consumes, including the dense path that already
  // fills 64 KiB: padded C's extra row, or the unused half of dense U.
  int* const published_unsafe =
      reinterpret_cast<int*>(padded_lds ? c + B : d + (Transpose ? B : 1));
  if (threadIdx.x < 64) {
    const unsigned long long votes = __ballot(unsafe);
    if (threadIdx.x == 0) *published_unsafe = votes != 0;
  }
  __syncthreads();
  return __builtin_amdgcn_readfirstlane(*published_unsafe) == 0;
}

template <int T, bool Transpose, bool Banded, bool StaticPivot = false>
__global__ __launch_bounds__(T, 1) void getrf_kernel(
    int n, double* a, std::int64_t lda, int tiles, int panels, int* next,
    int* ready, int* info, int info_offset, int* pivot_count,
    double criterion) {
  __shared__ Shared s;
  if (threadIdx.x == 0) s.task = atomicAdd(next, 1);
  __syncthreads();
  // Like CK optimized GEMM loop bounds, keep workgroup-uniform control in
  // SGPRs.
  const int task = __builtin_amdgcn_readfirstlane(s.task);
  const Tile tile = diagonal_first && panels < tiles
                        ? task_tile_diagonal_first(task, tiles, panels)
                    : Banded ? task_tile_banded(task, tiles)
                             : task_tile(task, tiles);
  const int bi = tile.row, bj = tile.col, panel = min(bi, bj);
  const int row = bi * B, col = bj * B;
  const int rows = min(B, n - row), cols = min(B, n - col);
  __syncthreads();

  constexpr bool UseDeferredWait =
      deferred_wait || (small_pipeline && T != Threads);
  constexpr bool UseLdsSync = lds_only_sync || (small_pipeline && T != Threads);
  using Gemm = GemmFor<T>;
  Gemm gemm;
  auto acc = gemm.GetCThreadBuffer();
  constexpr auto ct = Gemm::GetCThreadDescriptor_M0_N0_M1_N1_M2_M3_M4_N2();
  constexpr auto cmn = ck::make_naive_tensor_descriptor(
      make_tuple(I<B>{}, I<B>{}), make_tuple(I<1>{}, I<B>{}));
  constexpr auto cd = Gemm::MakeCGridDescriptor_M0_N0_M1_N1_M2_M3_M4_N2(cmn);
  constexpr auto lds_mn = ck::make_naive_tensor_descriptor(
      make_tuple(I<B>{}, I<B>{}), make_tuple(I<1>{}, I<CStride>{}));
  constexpr auto lds_cd =
      Gemm::MakeCGridDescriptor_M0_N0_M1_N1_M2_M3_M4_N2(lds_mn);
  constexpr auto lengths = ck::generate_sequence_v2(
      [](auto i) {
        return Gemm::GetCThreadDescriptor_M0_N0_M1_N1_M2_M3_M4_N2().GetLength(
            i);
      },
      I<8>{});
  using Pass =
      std::conditional_t<fused_update, Negate,
                         ck::tensor_operation::element_wise::PassThrough>;
  const auto origin =
      Gemm::CalculateCThreadOriginDataIndex8D(I<0>{}, I<0>{}, I<0>{}, I<0>{});
  const auto dst_origin = ck::generate_tuple(
      [&](auto i) { return static_cast<ck::index_t>(origin[i]); }, I<8>{});
  if constexpr (fused_update) {
    // CUDA's update organization: accumulate -A + L*U directly, then negate
    // during the one accumulator-to-LDS transfer. This changes summation order
    // and removes the extra LDS read/modify/write pass and its barrier.
    using Curve = ck::SpaceFillingCurve<decltype(lengths),
                                        ck::Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
                                        ck::Sequence<1, 1, 1, 1, 1, 1, 1, 1>>;
    ck::static_for<0, Curve::GetNumOfAccess(), 1>{}([&](auto i) {
      constexpr auto index = Curve::GetIndex(i);
      constexpr int local = ct.CalculateOffset(index);
      const auto coordinate = ck::generate_tuple(
          [&](auto dim) { return dst_origin[dim] + index[dim]; }, I<8>{});
      const int flat = cd.CalculateOffset(coordinate), r = flat % B,
                c = flat / B;
      acc(I<local>{}) =
          r < rows && c < cols
              ? -a[row + r + static_cast<std::size_t>(col + c) * lda]
              : 0;
    });
  } else
    acc.Clear();
  if (panel > 0) {
    double ra[B * K / T], rb[B * K / T];
    wait_ready(ready + bi, ready + bj * tiles);
    fetch<T>(a, n, lda, row, col, 0, ra, rb);
    stage<T>(s, 0, ra, rb);
    __syncthreads();
    int buf = 0;
    for (int k = 0; k < panel * B; k += K) {
      const bool more = k + K < panel * B;
      const bool boundary = more && (k + K) % B == 0;
      if (more && !(UseDeferredWait && boundary)) {
        if ((k + K) % B == 0) {
          wait_ready(ready + bi + ((k + K) / B) * tiles,
                     ready + (k + K) / B + bj * tiles);
        }
        fetch<T>(a, n, lda, row, col, k + K, ra, rb);
      }
      const auto ab = ck::make_dynamic_buffer<ck::AddressSpaceEnum::Lds>(
          s.ab[buf][0], OperandSpace);
      const auto bb = ck::make_dynamic_buffer<ck::AddressSpaceEnum::Lds>(
          s.ab[buf][1], OperandSpace);
      gemm.Run(ab, bb, acc);
      if (UseDeferredWait && boundary) {
        wait_ready(ready + bi + ((k + K) / B) * tiles,
                   ready + (k + K) / B + bj * tiles);
        fetch<T>(a, n, lda, row, col, k + K, ra, rb);
      }
      if (more) stage<T>(s, 1 - buf, ra, rb);
      // Operands are in LDS here; CK's optional sync avoids waiting for
      // unrelated VMEM. Global publication and producer-consumer waits retain
      // full barriers.
      if constexpr (UseLdsSync)
        ck::block_sync_lds();
      else
        __syncthreads();
      buf = 1 - buf;
    }
  }
  // Switch LDS from double-buffered GEMM operands to C + diagonal storage.
  __syncthreads();
  auto cb = ck::make_dynamic_buffer<ck::AddressSpaceEnum::Lds>(s.lu.c, CSpace);
  ck::ThreadwiseTensorSliceTransfer_v1r3<
      double, double, decltype(ct), decltype(lds_cd), Pass, decltype(lengths),
      ck::Sequence<0, 1, 2, 3, 4, 5, 6, 7>, 7, 1,
      ck::InMemoryDataOperationEnum::Set, 1, true>
      copy(lds_cd, dst_origin, Pass{});
  copy.Run(ct,
           make_tuple(I<0>{}, I<0>{}, I<0>{}, I<0>{}, I<0>{}, I<0>{}, I<0>{},
                      I<0>{}),
           acc, lds_cd, cb);
  __syncthreads();
  if constexpr (!fused_update) {
    for (int idx = threadIdx.x; idx < B * B; idx += T) {
      const int r = idx % B, c = idx / B;
      s.lu.c[r + c * CStride] =
          r < rows && c < cols
              ? a[row + r + static_cast<std::size_t>(col + c) * lda] -
                    s.lu.c[r + c * CStride]
              : 0;
    }
    __syncthreads();
  }
  if (bi == bj) {
    if constexpr (StaticPivot) {
      auto tile = TileView<>{s.lu.c};
      cdls::detail::getrf::factor_with_static_pivot<B, 16, T>(
          threadIdx.x, rows, tile, RocmPivotCount{pivot_count}, criterion);
    } else
      diagonal<T>(s.lu.c, s.lu.diag, rows, info_offset + row, info);
  } else {
    wait_ready(ready + panel + panel * tiles);
    if constexpr (padded_lds) {
      // Both solves consume a lower-oriented triangle: U is transposed for
      // L21, while L already has this orientation for U12. Only this half
      // is live, freeing LDS for the padded C tile.
      for (int idx = threadIdx.x; idx < B * B; idx += T) {
        const int r = idx % B, c = idx / B;
        if (bi > bj ? r >= c : r > c)
          s.lu.diag[diag_index(r, c)] =
              a[panel * B + (bi > bj ? c : r) +
                static_cast<std::size_t>(panel * B + (bi > bj ? r : c)) * lda];
      }
    } else if constexpr (Transpose) {
      for (int idx = threadIdx.x; idx < B * B; idx += T) {
        const int r = idx % B, c = idx / B;
        if (bi > bj ? r <= c : r > c)
          s.lu.diag[bi > bj ? c + r * B : idx] =
              a[panel * B + r + static_cast<std::size_t>(panel * B + c) * lda];
      }
    } else {
      for (int idx = threadIdx.x; idx < B * B; idx += T)
        s.lu.diag[idx] = a[panel * B + idx % B +
                           static_cast<std::size_t>(panel * B + idx / B) * lda];
    }
    __syncthreads();
    if (bi > bj) {
      if constexpr (trsm_diag_reciprocal && !StaticPivot) {
        const bool safe = prepare_upper_diagonal<Transpose>(s.lu.diag, s.lu.c);
        // Select once per tile; the common safe loop has no per-pivot tests
        // or divisions. The mixed loop preserves zero/scale guards.
        if (safe)
          solve<true, T, Transpose, 2>(s.lu.c, s.lu.diag, rows, cols);
        else
          solve<true, T, Transpose, 1>(s.lu.c, s.lu.diag, rows, cols);
      } else
        solve<true, T, Transpose>(s.lu.c, s.lu.diag, rows, cols);
    } else
      solve<false, T, Transpose>(s.lu.c, s.lu.diag, rows, cols);
  }
  for (int idx = threadIdx.x; idx < B * B; idx += T) {
    const int r = idx % B, c = idx / B;
    if (r < rows && c < cols)
      a[row + r + static_cast<std::size_t>(col + c) * lda] =
          s.lu.c[r + c * CStride];
  }
  // Fence every writer before the leader publishes the completed tile.
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0)
    __hip_atomic_store(ready + bi + bj * tiles, 1, __ATOMIC_RELEASE,
                       __HIP_MEMORY_SCOPE_AGENT);
}
// Matrix size dispatch keeps the eight-wave, transposed TRSM variant out of
// the large matrices where extra MFMA operand traffic outweighed its benefit.
template <bool StaticPivot = false>
hipError_t launch_tiles(int n, double* a, std::int64_t lda, int tiles,
                        int panels, int* next, int* info, hipStream_t stream,
                        int info_offset, int* pivot_count = nullptr,
                        double criterion = 0) {
  const int tasks = panels * (2 * tiles - panels);
  if constexpr (!StaticPivot && small_switch > 0) {
    if (n <= small_switch) {
      if constexpr (banded_order) {
        if (n >= 896 && n <= 2560 && (panels == tiles || panels % 4 == 0)) {
          hipLaunchKernelGGL(
              (getrf_kernel<2 * Threads, true, true, StaticPivot>), dim3(tasks),
              dim3(2 * Threads), 0, stream, n, a, lda, tiles, panels, next,
              next + 1, info, info_offset, pivot_count, criterion);
          return hipGetLastError();
        }
      }
      hipLaunchKernelGGL((getrf_kernel<2 * Threads, true, false, StaticPivot>),
                         dim3(tasks), dim3(2 * Threads), 0, stream, n, a, lda,
                         tiles, panels, next, next + 1, info, info_offset,
                         pivot_count, criterion);
      return hipGetLastError();
    }
  }
  hipLaunchKernelGGL(
      (getrf_kernel<Threads, transpose_large_trsm, false, StaticPivot>),
      dim3(tasks), dim3(Threads), 0, stream, n, a, lda, tiles, panels, next,
      next + 1, info, info_offset, pivot_count, criterion);
  return hipGetLastError();
}
hipError_t launch_prefix(int n, int prefix, double* a, std::int64_t lda,
                         int* info, void* workspace, hipStream_t stream,
                         int info_offset) {
  if constexpr (small_kernel) {
    if (n <= B && prefix >= n)
      return launch_small(n, a, lda, info, stream, info_offset);
  }
  const auto status = hipMemsetAsync(workspace, 0, workspace_size(n), stream);
  if (status != hipSuccess) return status;
  const int tiles = (n - 1) / B + 1;
  const int panels = (prefix - 1) / B + 1;
  auto* next = static_cast<int*>(workspace);
  return launch_tiles(n, a, lda, tiles, panels, next, info, stream,
                      info_offset);
}
}  // namespace

std::size_t workspace_size(int n) {
  if (n <= 0) return 0;
  const std::size_t tiles = (static_cast<std::size_t>(n) + B - 1) / B;
  if (tiles * tiles > INT_MAX) return 0;
  return (1 + tiles * tiles) * sizeof(int);
}

}  // namespace cdls::rocm_detail::getrf
