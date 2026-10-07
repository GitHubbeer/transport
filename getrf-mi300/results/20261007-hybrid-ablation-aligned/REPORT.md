# Hybrid 消融实验：MI300A FP64 no-pivot LU

执行完成；数值校验状态：FAIL；节点：`sh5-pl1-s12-33`；开始：2026-10-07T09:58:20.799910+00:00。

计划已完整执行：n≤16384，113 个配置、678 个运行、3 轮；672 个运行通过校验。数值失败记录保留并排除在有效加速比中。

对照为同一源码的 `CDLS_GETRF_ENABLE_HYBRID=ON/OFF` 两个 Release 构建。OFF 将完整矩阵交给已有的 fused tiled 路径，保持 tile、MFMA、调度与数值算法配置。ON 在 n>4096 时使用 512 列外层 panel、rocBLAS DGEMM 更新及 ≤4096 的 fused 尾部。这里的 hybrid 指 GPU 内部的 kernel/BLAS 组合，不是 CPU/GPU 分工。


尺寸、输入、布局、method/reference 和每个配置的预热/重复次数逐项复用之前的评测计划：`/shared/prerelease/home/yuxi_hong_research_lws/jie/tingxuan/cdls/profiling/getrf-mi300/results/20261006-current`，共 113 个配置。除 ON/OFF 开关外，数值实现和 benchmark 源码与此前一致。小尺寸每轮预热 20 次，通常尺寸 5 次；主尺寸/边界/布局通常每轮 30 个样本，输入/static/pivot 通常 20 个。每个配置的确切参数保存在 metadata.json。static 路径不使用 hybrid，可作为开关无影响的对照；pivot phase 的 rocSOLVER 参考包含动态主元。

主实验固定 seed=20261001、lda=n、列主序 FP64，非对角元素 uniform[-1,1]、对角 n+1。每种实现每个配置 3 轮，预热及重复次数逐项按原计划执行。轮次交替 ON/OFF 顺序，并反转尺寸顺序；同一 GPU 串行执行。每个进程也交替测量 rocSOLVER no-pivot 作为参考；hybrid 消融加速比只由 CDLS 两个构建计算。

HIP event 测量公共 factorization 调用，包含 workspace/info 初始化和主机提交间隙；输入恢复、分配与校验排除在计时外。结果为各轮中位数的中位数；加速比为逐轮 OFF/ON 比值的中位数，范围表示轮次最小/最大值，不是置信区间。每个进程对最终 LU 做完整 GPU 重构，检查有限值、padding、counter 和 rocSOLVER info；scaled residual=||A-LU||F/(||A||F*n*epsilon)≤100。

| n | Hybrid (ms) | 无 hybrid (ms) | 加速比 OFF/ON | 轮次范围 |
|---:|---:|---:|---:|---:|
| 1 | 0.0085 | 0.0078 | 1.08× | 0.92–1.14× |
| 2 | 0.0077 | 0.0069 | 0.90× | 0.89–1.08× |
| 4 | 0.0079 | 0.0083 | 1.06× | 0.99–1.07× |
| 8 | 0.0082 | 0.0074 | 0.90× | 0.89–0.93× |
| 16 | 0.0097 | 0.0096 | 0.99× | 0.98–1.07× |
| 32 | 0.0141 | 0.0142 | 1.00× | 0.98–1.01× |
| 64 | 0.0268 | 0.0269 | 1.00× | 1.00–1.00× |
| 128 | 0.0813 | 0.0810 | 1.00× | 0.95–1.02× |
| 256 | 0.1719 | 0.1710 | 0.99× | 0.99–1.00× |
| 512 | 0.3658 | 0.3655 | 1.00× | 0.99–1.00× |
| 768 | 0.6221 | 0.6255 | 1.01× | 1.00–1.02× |
| 1024 | 0.7523 | 0.7536 | 1.00× | 1.00–1.02× |
| 1536 | 1.2260 | 1.2290 | 1.01× | 0.99–1.01× |
| 2048 | 1.7289 | 1.7182 | 0.99× | 0.99–1.01× |
| 2560 | 2.3509 | 2.3375 | 1.00× | 0.99–1.01× |
| 3072 | 3.1004 | 3.0783 | 0.99× | 0.99–1.00× |
| 4096 | 4.6933 | 4.7345 | 1.01× | 1.00–1.01× |
| 5120 | 7.2491 | 7.8857 | 1.09× | 1.08–1.09× |
| 6144 | 10.2781 | 12.2343 | 1.19× | 1.19–1.19× |
| 8192 | 18.6114 | 27.6341 | 1.48× | 1.48–1.49× |
| 10240 | 31.1729 | 55.4008 | 1.78× | 1.78–1.79× |
| 12288 | 48.6309 | 95.8700 | 1.97× | 1.96–1.97× |
| 16384 | 103.9872 | 225.4500 | 2.17× | 2.16–2.17× |

在本次有效的大矩阵配置（n>4096）中，hybrid 加速比为 1.09–2.17×。该实验支持关于此硬件、FP64 和当前 fused 实现扩展性的结论；不能据此推断所有非 hybrid 算法或其他 GPU 都需要相同设计。

失败运行（保留日志，不计入有效性能结论）：

- `t1-input-auto-npvt-random-n2048-lda2048-off0-hybrid.log`，returncode=1
- `t1-input-auto-npvt-random-n2048-lda2048-off0-nohybrid.log`，returncode=1
- `t2-input-auto-npvt-random-n2048-lda2048-off0-nohybrid.log`，returncode=1
- `t2-input-auto-npvt-random-n2048-lda2048-off0-hybrid.log`，returncode=1
- `t3-input-auto-npvt-random-n2048-lda2048-off0-hybrid.log`，returncode=1
- `t3-input-auto-npvt-random-n2048-lda2048-off0-nohybrid.log`，returncode=1

布局补充实验见 `summary.csv`。原始逐次数据见 `samples.csv`、完整日志及 `commands.json`；源码/二进制/构建缓存及 SHA-256 见 `source/`、各 variant 目录、`metadata.json` 和 `MANIFEST.json`。

详细输入/布局/边界表见 [INPUTS_LAYOUT.md](INPUTS_LAYOUT.md)。
论文大矩阵图见 [hybrid-ablation-large.pdf](hybrid-ablation-large.pdf)，英文图注见 [PAPER_CAPTION.txt](PAPER_CAPTION.txt)。
