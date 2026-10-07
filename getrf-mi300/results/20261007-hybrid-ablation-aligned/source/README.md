# cdls

The core dense linear solvers. GETRF computes square, FP64, column-major,
in-place LU factors with unit diagonal L. Both backends retain the existing
no-pivot and static-pivot entry points in `include/cdls/getrf.h`.

CUDA and ROCm share algorithm code in `src/common/getrf`:

| File | Shared work |
| --- | --- |
| `schedule.h` | Dependency-ordered panel borders and bounded diagonal lookahead |
| `panel.h` | LU panel writeback, U12 solve, and active trailing-corner update |
| `triangular.h` | Forward substitution for Lx=B and xU=B through oriented views |
| `numeric.h` | Guarded exact reciprocals and triangular diagonal modes |
| `static_pivot.h` | Next-pivot replacement and replacement counting |
| `hybrid.h` | Prefix LU/TRSM followed by BLAS updates and a fused tail |

CUDA retains its CuTe MMA, asynchronous copies, register-LU recovery and
measured A100 dispatch settings. ROCm uses the optimized CK MFMA implementation,
padded LDS, packed triangular coefficients, wave64/DPP broadcasts and MI300
dispatch settings. Backend atomics, publication barriers and matrix updates
remain local. Floating-point evaluation order can differ between backends.

ROCm requires wave64 `gfx90a`, `gfx940`, `gfx941` or `gfx942` (the default).
CK headers are fetched at a pinned ROCm 6.4.2 revision, or supplied through
`CDLS_CK_SOURCE_DIR`; rocWMMA and CUDA compatibility wrappers are no longer
needed for ROCm GETRF. Matrix addresses and strides remain 64-bit; task counts
must fit a signed int. `cdlsDgetrf_bufferSize` returns the backend's workspace
requirement. The no-pivot call leaves `pivot_count` untouched; the static-pivot
call adds its replacements to the caller's device counter.

For a ROCm-only build and validation:

```bash
cmake -S . -B build/getrf \
  -DCMAKE_PREFIX_PATH="$ROCM_PATH" \
  -DCMAKE_HIP_COMPILER="$ROCM_PATH/llvm/bin/clang++" \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 \
  -DCMAKE_DISABLE_FIND_PACKAGE_CUDAToolkit=ON \
  -DCDLS_ENABLE_TESTS=ON
cmake --build build/getrf -j4
ctest --test-dir build/getrf --output-on-failure
```

Add `-DCDLS_CK_SOURCE_DIR=/path/to/composable_kernel` to use an existing checkout.
`CDLS_GETRF_ENABLE_HYBRID` defaults to `ON`. Configure a separate build with
`-DCDLS_GETRF_ENABLE_HYBRID=OFF` for a no-pivot GETRF ablation: large matrices
use the existing full fused tiled path instead of outer panels and BLAS
trailing updates. Small-size dispatch, public API, workspace and static-pivot
behavior are preserved. The option applies to both CUDA and ROCm; MI300
measurement instructions are in `profiling/getrf-mi300/HYBRID_ABLATION.md`.
For CUDA, configure with the CUDA toolkit and `CMAKE_CUDA_ARCHITECTURES=80`
or a supported newer architecture. `CDLS_ENABLE_TESTS` builds the same GPU
numerical test for each available backend, plus a portable task-order test.
Validation covers tile and dispatch boundaries, hybrid sizes, padded/misaligned
matrix origins, workspace reuse, static-pivot counters and extreme scales.
Small matrices use full LU reconstruction; large matrices use three independent
signed projections with a relative residual tolerance of `2e-11`.
