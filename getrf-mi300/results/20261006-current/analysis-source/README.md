# Current CDLS GETRF versus rocSOLVER on MI300A

`benchmark.cpp` calls the current public CDLS entry points. The primary baseline
is `rocsolver_dgetrf_npvt`, with the same FP64 square, column-major, no-pivot
operation. The supplementary `pivot` phase calls `rocsolver_dgetrf`; it performs
additional pivot search and row swaps. The `static` phase uses criterion zero
and includes resetting its replacement counter in the timed interval.

The primary size/layout input has diagonal n+1 and uniform[-1,1] off-diagonals,
with fixed seed=20261001. Other input families are reported separately. The trace
driver uses a deterministic integer-pattern off-diagonal on the same diagonally
dominant family; its measurements only supply kernel attribution.

The sweep includes 116 configurations, three separate runs each, and 9,240
alternating paired samples. It covers n=1..32768, tile/dispatch boundaries,
lda=n+1/n+17/n+128/2n, an 8-byte origin offset, signed diagonals, column scaling,
known LU factors, identity, uniform extreme scales, and unconstrained random
inputs. Large n>=24576 use 10 pairs/run; ordinary cases use 30 and additional
inputs/static/pivoted references use 20. Small cases warm both implementations
20 times; most sizes use 5 warmups and the largest use 3.

Every case uses the CDLS handle's nonblocking stream for both implementations.
rocSOLVER's workspace is queried and allocated before timing. GPU events bracket
factorization including workspace/info initialization and gaps while the host
submits kernels. Input restoration, allocation and validation are excluded.
Host API submission and synchronous wall latency are also recorded. Thus the
measurement represents public factorization latency under normal host submission.

Both final factors undergo full GPU LU reconstruction for every tested size,
plus long-double CPU reconstruction for n<=128. With pivoting, the reconstruction
uses PA. Padding, finite factors, rocSOLVER info and CDLS counter semantics are
checked. The residual criterion is ||A-LU||_F/(||A||_F*n*epsilon)<=100 for each
implementation. Factor differences are additional checks for regular input
families; arbitrary random no-pivot inputs use their reconstruction residuals.
Failed numerical cases are retained in logs and excluded from valid speedup
claims; they are useful stability stress results.

Build and run with ROCm 6.4.2:

```bash
cmake -S . -B build/getrf-shared \
  -DCMAKE_PREFIX_PATH="$ROCM_PATH" \
  -DCMAKE_HIP_COMPILER="$ROCM_PATH/llvm/bin/clang++" \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 \
  -DCMAKE_DISABLE_FIND_PACKAGE_CUDAToolkit=ON -DCDLS_ENABLE_TESTS=ON
cmake --build build/getrf-shared -j4
hipcc -O3 -std=c++17 -Iinclude profiling/getrf-mi300/benchmark.cpp \
  -Lbuild/getrf-shared/src/rocm -Wl,-rpath,"$PWD/build/getrf-shared/src/rocm" \
  -lcdls_rocm -lrocblas -lrocsolver -o build/cdls-getrf-bench
python3 profiling/getrf-mi300/run.py profiling/getrf-mi300/results/new-run
python3 profiling/getrf-mi300/summarize.py profiling/getrf-mi300/results/new-run
python3 profiling/getrf-mi300/plot.py \
  profiling/getrf-mi300/results/new-run/summary.csv \
  profiling/getrf-mi300/results/new-run/figures
```

Only after the timing sweep ends, run the independent diagnostic passes and
generate the report and plots (the plotting environment needs NumPy/Matplotlib):

```bash
python3 profiling/getrf-mi300/stress.py profiling/getrf-mi300/results/new-run
python3 profiling/getrf-mi300/profile.py profiling/getrf-mi300/results/new-run
python3 profiling/getrf-mi300/finalize.py profiling/getrf-mi300/results/new-run
```

The stress pass compares the same arbitrary random matrices to pivoted rocSOLVER
using PA=LU residuals, without adding its results to timing statistics. The trace
pass uses rocprofv3 and ROCTX pause/resume to isolate one warmed factorization at
each of six sizes, plus the static-pivot path at n=4096/8192. Trace kernel busy times are intrusive diagnostic measurements,
not replacements for the ordinary paired timings. `finalize.py` uses the plotting
environment at `build/getrf-perf-env/bin/python` and writes a SHA-256 artifact
manifest.

Use a new output directory. `--plan-only` prints the case plan. `--phases`
selects phases; `--resume` only accepts unchanged source and binary identities.
The result directory preserves executable/library copies, kernel source snapshots,
source hashes, exact commands, environment/library identities, raw logs, parsed
samples, validation results and summaries. A completed sweep with numerical
stress failures has metadata status FAIL and lists every failing case; completion
and numerical validity are separate in the report.
