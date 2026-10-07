# CDLS GETRF 与 PaStiX 的 MI300A 测试资料

本仓库归档基于 CK FP64 XDLops/MFMA 的 CDLS ROCm GETRF 测试，以及与 rocSOLVER 的比较。
设备为 AMD Instinct MI300A（gfx942）。原有 dense GETRF 测试使用 ROCm 6.4.2，
新增 PaStiX CDLS + 1D update1 测试使用 ROCm 7.2.4。原始 CDLS 仓库为
[TheCoreTeam/cdls](https://github.com/TheCoreTeam/cdls)，测试基于 `dev-nvi-getrf` 分支
`8be4f73` 及测试时的未提交修改；准确实现见各结果目录的源码快照和 `git-diff.txt`。
新增 PaStiX 批次固定 CDLS 提交 `44067f1e654e5c4a593cc82c34cde4f4c48a62b2`，
并归档对应 ROCm 源码、运行二进制和 PaStiX 工作区补丁。

## 测试结果

| 测试批次 | 覆盖范围 | 验证 |
|---|---|---|
| [完整测试](getrf-mi300/results/20261006-current/REPORT.md) | 116 组配置、348 次运行、9,240 对样本 | 345 次通过；任意随机输入 n=2048 的 3 次运行失败 |
| [n=2048 重测](getrf-mi300/results/20261006T095710Z-n2048-retest/REPORT.md) | 对角占优输入、3 轮、90 对样本 | 全部通过 |
| [PaStiX CDLS + 1D update1](pastix-cdls/results/20261006-cdls-update1-mi300a/README.md) | 10 个稀疏矩阵，每个 3 次独立运行 | 30/30 残差通过且 refinement=0；其中 21 次前向误差检查未通过，按指定条件纳入 |

n=2048 重测的 CDLS 中位延迟为 **1.716088 ms**，rocSOLVER 无 pivot 路径为
**4.269848 ms**，加速比 **2.488×**。与上次 CDLS 1.722758 ms 的结果基本一致。
此次重测未重复任意随机输入的数值压力测试。

普通 GETRF 在 n≤4096 使用融合 CK kernel；大尺寸结合 CK 面板与 rocBLAS DGEMM。
主比较双方均为 FP64、方阵、列主序、无 pivot；带 pivot 基线及静态 pivot 结果单列。
计时使用 HIP events，包括初始化及主机提交 kernel 的间隙，排除矩阵恢复、分配和验证。
结果取三轮各自中位数的中位数，详细口径见报告。

## 资料位置

- `getrf-mi300/`：benchmark、测试计划、分析、绘图、压力测试及 trace 脚本。
- `getrf-mi300/results/`：原始日志、CSV 样本与汇总、数值验证、环境和命令、报告及性能图。
- `pastix-cdls/results/`：PaStiX 数值分解性能，包含逐次样本、汇总、原始日志、环境、固定二进制和 CDLS ROCm 源码快照。
- 每个结果目录中的 `source/`、`benchmark`、`libcdls_rocm.so`：测试时的源码快照与二进制。
- `MANIFEST.json`：文件 SHA-256 校验值；完整测试沿用原始清单，2048 重测补充归档清单。
- `licenses/ck_getrf_notices/`：CK、rocBLAS、rocSOLVER 的上游许可和来源说明。

原始记录保留测试机器的路径和环境信息；报告中的历史路径供追溯，仓库内浏览请使用上述相对链接。

## 验证已有结果

下面的命令仅从保留日志重算统计和核验源码、二进制哈希，无需 GPU：

```bash
python3 getrf-mi300/summarize.py getrf-mi300/results/20261006-current
python3 getrf-mi300/summarize.py getrf-mi300/results/20261006T095710Z-n2048-retest
```

完整测试的验证状态为 `FAIL`，原因是报告中保留的三次数值压力测试失败。
它们不纳入有效性能结论；所有测量记录均保留。

## 用归档二进制重测 n=2048

在安装 ROCm 6.4.2、可访问 MI300A 的 Linux 节点，从仓库根目录运行：

```bash
export ROCM_PATH=/path/to/rocm-6.4.2
archive="$PWD/getrf-mi300/results/20261006T095710Z-n2048-retest"
export LD_LIBRARY_PATH="$archive:$ROCM_PATH/lib:$ROCM_PATH/lib64:${LD_LIBRARY_PATH:-}"
output="n2048-rerun-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir "$output"
for trial in 1 2 3; do
  "$archive/benchmark" --samples --n 2048 --lda 2048 --offset 0 \
    --method auto --reference npvt --matrix dense --repeats 30 --warmup 5 \
    > "$output/trial-$trial.log"
done
```

benchmark 使用固定 seed=20261001，输出每对样本、数值验证及单轮中位数。
归档二进制需要匹配的 ROCm 运行库；完整源码编译和全尺寸测试流程见
[getrf-mi300/README.md](getrf-mi300/README.md)，其命令按原始 CDLS 仓库布局执行。
本仓库中的脚本和源码快照保留原样，仓库本身是测试归档。
