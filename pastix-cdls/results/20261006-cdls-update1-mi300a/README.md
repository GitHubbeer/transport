# PaStiX CDLS + 1D update1：MI300A 性能

2026-10-06 测试。AMD Instinct MI300A（gfx942），ROCm 7.2.4，2 个 CPU 线程
（固定 CPU 0、1），OpenBLAS/OMP 各 1 线程，32 GiB 设备内存池，FP64 LU。
CDLS 来自 `dev-nvi-getrf`，固定提交 `44067f1e654e5c4a593cc82c34cde4f4c48a62b2`。
每个矩阵执行 3 次独立的完整求解器进程。下表为 PaStiX 数值分解耗时中位数，
计时取 `PASTIX_TIMING,pastix_numfact`；GFlop/s 取带吞吐量的实际 `Time to factorize` 行。

| 矩阵 | 中位耗时（秒） | 范围（秒） | 中位 GFlop/s | 前向误差检查 |
|---|---:|---:|---:|---|
| CoupCons3D | 3.6443 | 3.6387–3.6874 | 245.56 | 通过 |
| dielFilterV3real | 11.5260 | 11.5241–13.2448 | 167.79 | 未通过 |
| inline_1 | 5.4952 | 5.4803–5.5204 | 46.49 | 未通过 |
| audikw_1 | 8.8438 | 8.8091–8.9191 | 1290 | 未通过 |
| nd6k | 0.4445 | 0.4423–0.4451 | 517.59 | 未通过 |
| xenon2 | 2.4393 | 2.4234–2.4459 | 41.94 | 通过 |
| radiation | 2.6493 | 2.6417–2.6505 | 214.43 | 未通过 |
| dgreen | 13.7961 | 13.6567–13.8256 | 207.56 | 未通过 |
| pwtk | 2.3069 | 2.3048–2.3165 | 18.54 | 未通过 |
| cage12 | 2.2608 | 2.2521–2.2612 | 1510 | 通过 |

30/30 个样本均满足用户确认的判定条件：最终相对残差 ≤ 1e-12，后向误差检查
SUCCESS，refinement 迭代次数为 0；即使前向误差检查失败也纳入性能统计。
最大最终相对残差为 6.131891e-15。21 次运行的前向误差检查失败，保留其实际
退出码 255 和原始日志，不将其标为完整数值检查通过。

`audikw_1` 使用现有完整修复版（943695 阶，文件存储 39297771 项）；原始输入
文件被截断。输入路径、大小和 SHA-256 在 `manifest.json` 中记录；矩阵文件未归档。

## 归档内容

- [accepted_statistics.csv](accepted_statistics.csv)：按上述条件筛选的中位数、范围、残差和前向检查状态。
- [samples.csv](samples.csv)：30 个样本的计时、检查结果、退出码和日志索引。
- `logs/`：原始日志，以及每次运行的命令、环境、时间与内存记录。
- `manifest.json`、`hardware.txt`、`libraries.txt`：版本、输入/二进制哈希、设备和实际库解析路径。
- `binaries/`：冻结的求解器、PaStiX CDLS 插件及实际加载的 `libcdls_rocm.so`。
- `source/cdls/`：固定提交中的 ROCm GETRF、共享算法、头文件、CMake 和上游许可快照。
- `runner.py`：执行时的测量脚本快照；`pastix-working-tree.patch`：测量时的 PaStiX 工作区补丁。
- `qualify.py`、`acceptance.json`：独立复核原始日志并应用零 refinement 条件的脚本和结果。
- `SHA256SUMS`：整个归档的文件校验值。

## 本地检查

在本目录执行以下命令，无需 GPU。它们核验归档并从原始日志重算接受条件与汇总：

```bash
sha256sum -c SHA256SUMS
python3 qualify.py
```

原始环境记录保留了测试节点的绝对路径。`qualify.py` 使用本目录内的文件，
可直接在本地执行；`runner.py` 保留原测量环境的路径，重跑需要 MI300A、对应运行库
及矩阵输入，并需将其源仓库路径配置为实际位置。
