#pragma once
#include <hip/hip_runtime.h>

#include <cstddef>

namespace cdls::rocm_detail::getrf {
// Production defaults from ck_getrf. Geometry and MI300 dispatch remain
// independent of CUDA's measured settings and hardware warp width.
inline constexpr int tile_size = 64;
inline constexpr int inner_size = 8;
inline constexpr bool diagonal_first = false;
inline constexpr bool banded_order = true;
inline constexpr bool fused_update = true;
inline constexpr bool deferred_wait = false;
inline constexpr bool lds_only_sync = false;
inline constexpr bool small_pipeline = true;
inline constexpr bool trsm_diag_reciprocal = true;
inline constexpr bool padded_lds = true;
inline constexpr bool lu_defer_info = true;
inline constexpr bool lu_wave_panel = true;
inline constexpr bool lu_reciprocal = false;
inline constexpr bool small_kernel = true;
inline constexpr bool lu_single_barrier = true;
inline constexpr int trsm_dpp = 2;
inline constexpr bool trsm_wave_broadcast = true;
inline constexpr bool transpose_large_trsm = true;
inline constexpr int small_switch = 2560;
std::size_t workspace_size(int n);
}  // namespace cdls::rocm_detail::getrf
