# MI300A audikw_1 实际 GETRF 执行粒度

本批次补齐普通 1D、无预取稀疏分解中的对角块证据，输入是前驱 Schur 更新及
diagonal GEADD 完成后、GETRF 执行前捕获的实际内容，完整保留 `lda*n` 个 FP64 值。
软件统一为 ROCm 7.2.4、rocSOLVER 3.32.0.dabb6df2b9、rocBLAS 5.2.0.dabb6df2b9，
当前 CDLS 提交为 `44067f1e654e5c4a593cc82c34cde4f4c48a62b2`。两个入口分别为
`cdlsDgetrf` 和 `rocsolver_dgetrf_npvt`，均无行交换。

kernel 数写作“总 / 初始化 / 计算”；耗时为无 profiler 完整调用中位数，单位 µs。
每个输入 5 次预热、3 个 trial，每 trial 30 次交替配对。

| n, lda | 真实频数 | rocSOLVER kernel | CDLS kernel | rocSOLVER → CDLS 耗时 | 加速比 |
|---|---:|---:|---:|---:|---:|
| 27, 201 | 44 | 2 / 1 / 1 | 1 / 0 / 1 | 18.580 → 13.960 | 1.331× |
| 189, 837 | 1 | 36 / 2 / 34 | 4 / 3 / 1 | 241.802 → 126.161 | 1.917× |
| 351, 1914 | 1 | 66 / 2 / 64 | 4 / 3 / 1 | 467.265 → 250.882 | 1.862× |

完整分布有 10,898 次调用、5,385 种 `(n,lda)`，最大 n 为 351。三个选中 pair
共覆盖 0.4221% 的调用；有计算拆分的两个 pair 共覆盖 0.01835%。每个 pair 捕获
一份具体内容，频数不表示捕获了全部出现的内容。本配置没有触发 hybrid。

189/351 的 rocSOLVER 路径由 panel GETF2、内部三角求解和 GEMM 组成；CDLS 统计
包含三个初始化 dispatch 和一个 fused 计算 kernel。27 的计算 kernel 数未减少，
减少的是独立 info-reset 启动。这些结果支持启动数量及完整调用时长的并列描述，
不支持将全部收益归因于 launch overhead；没有 GPU 利用率不足的硬件指标证据。

18 份代表输入验证均通过 LU 重构、有限值、padding 和 info 检查；CDLS ordinary API
没有 public 数值 info，使用明确绑定版本的内部 workspace 诊断。完整 solver 的
backward residual 很小，但严格 forward-error 检查失败、退出 255，原日志保留。
旧 ROCm 6.4.2 的 dense n=32768 上 CDLS 较慢，亦保留在独立历史库存中。

- [完整报告 HTML](results/20261007-audikw-getrf/REPORT.html)
- [六行汇总 CSV，含软件版本、输入来源、波动和数值状态](results/20261007-audikw-getrf/analysis/summary.csv)
- [完整 `(n,lda)` 频数](results/20261007-audikw-getrf/workload/frequency.csv)
- [逐调用信息](results/20261007-audikw-getrf/workload/calls.csv)
- [全部计时样本](results/20261007-audikw-getrf/analysis/samples.csv)
- [逐 kernel 提交关联与执行顺序](results/20261007-audikw-getrf/analysis/kernel-attribution.csv)
- [真实输入 manifest](results/20261007-audikw-getrf/inputs/manifest.json)
- [环境、代码与二进制版本](results/20261007-audikw-getrf/environment/manifest.json)
- [历史 hybrid 库存](results/20261007-audikw-getrf/historical/inventory.json)
- [复算及重跑说明](results/20261007-audikw-getrf/analysis/reproduce.txt)
- [完整归档包](results/20261007-audikw-getrf.tar.gz)及其 [SHA-256](results/20261007-audikw-getrf.tar.gz.sha256)

从仓库根目录校验拷贝和复算（无需 GPU）：

```bash
cd pastix-getrf/results/20261007-audikw-getrf
sha256sum --quiet -c SHA256SUMS
python3 analysis/scripts/analyze.py "$PWD"
python3 analysis/scripts/report.py "$PWD"
```

原始环境、命令中的机器绝对路径保留不变。复算从本目录读取原始日志和 trace，
`report.py` 的输出路径会变化，其余数据无需 GPU 即可验证。GitHub 可浏览本 README
和 CSV；HTML 报告需下载后在浏览器中打开。
