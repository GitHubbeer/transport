# Hybrid 消融评测

`ablate_hybrid.py` 使用同一 `benchmark.cpp`、同一输入分别运行
`CDLS_GETRF_ENABLE_HYBRID=ON/OFF` 构建。OFF 通过 `launch_prefix(n,n,...)`
执行完整 fused tiled LU；ON 保持生产 hybrid 策略。这是 GPU kernel 与
BLAS 组合的消融，输入均为 FP64 方阵、列主序、无主元 LU。

在已分配的单卡 MI300 节点上，设置 ROCm 路径并构建两个版本：

```bash
export ROCM_PATH=/nfsapps/ubuntu-24.04/opt/rocm-6.4.2
export LD_LIBRARY_PATH="$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}"
for variant in hybrid-ablation nohybrid; do
  hybrid=ON
  if [ "$variant" = nohybrid ]; then hybrid=OFF; fi
  cmake -S . -B "build/getrf-$variant" \
    -DCMAKE_PREFIX_PATH="$ROCM_PATH" \
    -DCMAKE_HIP_COMPILER="$ROCM_PATH/llvm/bin/clang++" \
    -DCMAKE_HIP_ARCHITECTURES=gfx942 \
    -DCMAKE_DISABLE_FIND_PACKAGE_CUDAToolkit=ON \
    -DCDLS_ENABLE_EXAMPLES=OFF -DCDLS_ENABLE_TESTS=ON \
    -DCDLS_GETRF_ENABLE_HYBRID="$hybrid"
  cmake --build "build/getrf-$variant" -j4
  ctest --test-dir "build/getrf-$variant" --output-on-failure
done
"$ROCM_PATH/bin/hipcc" -O3 -std=c++17 -Iinclude \
  profiling/getrf-mi300/benchmark.cpp \
  -Lbuild/getrf-hybrid-ablation/src/rocm \
  -Wl,-rpath,"$PWD/build/getrf-hybrid-ablation/src/rocm" \
  -lcdls_rocm -lrocblas -lrocsolver -o build/cdls-getrf-ablation-bench
python3 profiling/getrf-mi300/ablate_hybrid.py \
  profiling/getrf-mi300/results/new-hybrid-ablation-aligned \
  --binary build/cdls-getrf-ablation-bench \
  --reference-run profiling/getrf-mi300/results/20261006-current --max-n 16384
build/getrf-perf-env/bin/python profiling/getrf-mi300/ablate_hybrid.py \
  profiling/getrf-mi300/results/new-hybrid-ablation-aligned --plot-only
build/getrf-perf-env/bin/python profiling/getrf-mi300/report_hybrid_inputs.py \
  profiling/getrf-mi300/results/new-hybrid-ablation-aligned --plot
```

若已有固定 CK checkout，可在两个配置命令中添加相同的
`-DCDLS_CK_SOURCE_DIR=/path/to/composable_kernel`。
绘图需要 Matplotlib；计时脚本仅使用 Python 标准库，绘图在测量结束后单独运行。
输出目录必须是新目录，避免覆盖历史结果。

论文的主评测使用 `--reference-run` 读取之前评测的 `metadata.json`，
逐项复用原 116 配置的尺寸、矩阵类型、lda/offset、method/reference、
预热和计时重复次数。按用户要求，`--max-n 16384` 保留其中 113 个配置，
每种构建 3 轮；包含原主尺寸中 n≤16384 的全部点、
边界、padding/stride/offset、dense/signed/column_scaled/identity/known_lu/
random/scale_low/scale_high 输入，以及 static/pivot 阶段。
轮次交替构建顺序和尺寸顺序，`--plan-only` 可查看计划。
`--resume` 会校验冻结的 benchmark/library 与数值源码，复用已完成运行，
只执行计划内缺失的轮次。若收缩上限，超出范围的已完成记录保留在
`excluded-commands.json`，不计入主报告或主图。计时 runner 的版本变更
保存在 `source-versions/`；数值源码和 benchmark 二进制必须保持不变。

未指定 `--reference-run` 时，可用 `--sizes`、`--trials`、`--repeats`、
`--warmup` 和 `--no-layout` 做探索性评测；其默认值是大矩阵子集，
不作为与之前完整评测对齐的结果。

HIP event 计时包含公共 API 中的 workspace/info 初始化与主机提交间隙，
排除输入恢复、内存分配和校验。每个进程也测量 rocSOLVER no-pivot
作为参考。OFF/ON 比值仅使用 CDLS 的对应轮次中位数。
每个进程对最终因子执行完整 GPU LU 重构，检查 scaled Frobenius
residual≤100、有限值、padding、counter 和 rocSOLVER info。
失败配置保留日志，排除在有效性能结论之外。
原计划包含已知的 random/n=2048 无主元数值失败，完整评测可能返回
exit code 1；这时 `execution_complete=true` 表示计划已完整执行，
`status=FAIL` 表示其中有数值校验失败，已有的报告和有效配置结果仍然保留。

产物包括 `REPORT.md`、`INPUTS_LAYOUT.md`、`summary.csv`、`samples.csv`、PNG/PDF/SVG 图、
执行命令、原始日志、设备状态、源码、构建缓存、二进制及 SHA-256 manifest。
论文应表述为当前 fused 实现和硬件上的消融收益，而非所有非 hybrid
算法的普遍下界。
`hybrid-ablation-large.*` 聚焦原尺寸点中的 4096..16384，
`hybrid-ablation.*` 保留 n≤16384 的完整主尺寸曲线，`PAPER_CAPTION.txt`
提供与实测数字一致的英文图注。
