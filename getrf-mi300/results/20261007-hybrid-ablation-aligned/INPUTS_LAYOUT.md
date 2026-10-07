# 与原评测对齐的输入和布局结果

按用户要求，尺寸上限为 n=16384；在该范围内逐项复用原计划的全部输入和布局。

完整计划已执行：113 个配置、678 个运行；672 个运行通过全部校验。

下表耗时为各轮中位数的中位数，加速比为对应轮次 OFF/ON 比值的中位数。失败配置保留原始耗时，但不列入有效加速比结论。

## 输入类型

| 阶段 | n | 输入 | lda / offset | 方法 / 参考 | Hybrid ms | 无 hybrid ms | OFF/ON |
|---|---:|---|---|---|---:|---:|---:|
| input | 64 | signed | 64 / 0 | auto / npvt | 0.02680 | 0.02686 | 1.00× |
| input | 512 | signed | 512 / 0 | auto / npvt | 0.36784 | 0.36554 | 0.98× |
| input | 2048 | signed | 2048 / 0 | auto / npvt | 1.72773 | 1.71112 | 0.99× |
| input | 8192 | signed | 8192 / 0 | auto / npvt | 18.57393 | 27.58020 | 1.48× |
| input | 16384 | signed | 16384 / 0 | auto / npvt | 103.07790 | 224.06175 | 2.17× |
| input | 64 | column_scaled | 64 / 0 | auto / npvt | 0.02680 | 0.02684 | 1.00× |
| input | 512 | column_scaled | 512 / 0 | auto / npvt | 0.42239 | 0.42443 | 1.00× |
| input | 2048 | column_scaled | 2048 / 0 | auto / npvt | 1.95399 | 1.98388 | 1.01× |
| input | 8192 | column_scaled | 8192 / 0 | auto / npvt | 18.98753 | 27.81399 | 1.46× |
| input | 16384 | column_scaled | 16384 / 0 | auto / npvt | 105.25983 | 224.35294 | 2.13× |
| input | 64 | identity | 64 / 0 | auto / npvt | 0.02692 | 0.02690 | 0.99× |
| input | 512 | identity | 512 / 0 | auto / npvt | 0.36570 | 0.36298 | 1.00× |
| input | 2048 | identity | 2048 / 0 | auto / npvt | 1.70633 | 1.72417 | 1.01× |
| input | 8192 | identity | 8192 / 0 | auto / npvt | 14.58778 | 25.14296 | 1.72× |
| input | 16384 | identity | 16384 / 0 | auto / npvt | 65.99299 | 188.88532 | 2.87× |
| input | 64 | known_lu | 64 / 0 | auto / npvt | 0.02674 | 0.02686 | 1.01× |
| input | 512 | known_lu | 512 / 0 | auto / npvt | 0.36606 | 0.36368 | 1.00× |
| input | 64 | random | 64 / 0 | auto / npvt | 0.02682 | 0.02674 | 1.00× |
| input | 512 | random | 512 / 0 | auto / npvt | 0.36736 | 0.36578 | 0.99× |
| input | 2048 | random | 2048 / 0 | auto / npvt | — | — | 校验失败 |
| input | 65 | scale_low | 65 / 0 | auto / npvt | 0.07389 | 0.07379 | 1.00× |
| input | 2048 | scale_low | 2048 / 0 | auto / npvt | 1.95794 | 1.95807 | 1.00× |
| input | 65 | scale_high | 65 / 0 | auto / npvt | 0.07273 | 0.07353 | 1.01× |
| input | 2048 | scale_high | 2048 / 0 | auto / npvt | 1.96694 | 1.94414 | 1.00× |

## 布局

| 阶段 | n | 输入 | lda / offset | 方法 / 参考 | Hybrid ms | 无 hybrid ms | OFF/ON |
|---|---:|---|---|---|---:|---:|---:|
| padding | 64 | dense | 65 / 0 | auto / npvt | 0.02688 | 0.02690 | 1.00× |
| padding | 64 | dense | 81 / 0 | auto / npvt | 0.02698 | 0.02698 | 1.00× |
| padding | 64 | dense | 192 / 0 | auto / npvt | 0.02688 | 0.02694 | 1.00× |
| padding | 512 | dense | 513 / 0 | auto / npvt | 0.36863 | 0.37103 | 1.01× |
| padding | 512 | dense | 529 / 0 | auto / npvt | 0.36618 | 0.36410 | 0.99× |
| padding | 512 | dense | 640 / 0 | auto / npvt | 0.36163 | 0.36029 | 1.00× |
| padding | 2048 | dense | 2049 / 0 | auto / npvt | 1.75038 | 1.75040 | 1.00× |
| padding | 2048 | dense | 2065 / 0 | auto / npvt | 1.77388 | 1.76783 | 1.00× |
| padding | 2048 | dense | 2176 / 0 | auto / npvt | 1.74311 | 1.73566 | 1.00× |
| padding | 4096 | dense | 4097 / 0 | auto / npvt | 4.76705 | 4.79533 | 1.01× |
| padding | 4096 | dense | 4113 / 0 | auto / npvt | 4.78481 | 4.79445 | 1.00× |
| padding | 4096 | dense | 4224 / 0 | auto / npvt | 4.75758 | 4.75809 | 1.00× |
| padding | 8192 | dense | 8193 / 0 | auto / npvt | 19.30302 | 27.51204 | 1.43× |
| padding | 8192 | dense | 8209 / 0 | auto / npvt | 19.25192 | 27.66871 | 1.44× |
| padding | 8192 | dense | 8320 / 0 | auto / npvt | 18.75817 | 28.02995 | 1.49× |
| padding | 16384 | dense | 16385 / 0 | auto / npvt | 108.40718 | 224.05988 | 2.07× |
| padding | 16384 | dense | 16401 / 0 | auto / npvt | 108.66130 | 224.35400 | 2.06× |
| padding | 16384 | dense | 16512 / 0 | auto / npvt | 105.32555 | 224.17553 | 2.13× |
| stride | 64 | dense | 128 / 0 | auto / npvt | 0.02690 | 0.02692 | 1.00× |
| stride | 512 | dense | 1024 / 0 | auto / npvt | 0.35973 | 0.36251 | 1.01× |
| stride | 2048 | dense | 4096 / 0 | auto / npvt | 1.72945 | 1.71815 | 1.00× |
| stride | 8192 | dense | 16384 / 0 | auto / npvt | 18.74903 | 27.81309 | 1.48× |
| offset | 64 | dense | 64 / 1 | auto / npvt | 0.02688 | 0.02686 | 1.00× |
| offset | 512 | dense | 512 / 1 | auto / npvt | 0.37233 | 0.37149 | 1.00× |
| offset | 2048 | dense | 2048 / 1 | auto / npvt | 1.76875 | 1.76288 | 1.01× |
| offset | 8192 | dense | 8192 / 1 | auto / npvt | 19.33432 | 28.59436 | 1.48× |

## 边界

| 阶段 | n | 输入 | lda / offset | 方法 / 参考 | Hybrid ms | 无 hybrid ms | OFF/ON |
|---|---:|---|---|---|---:|---:|---:|
| boundary | 7 | dense | 7 / 0 | auto / npvt | 0.00775 | 0.00823 | 1.12× |
| boundary | 15 | dense | 15 / 0 | auto / npvt | 0.00989 | 0.00992 | 0.95× |
| boundary | 17 | dense | 17 / 0 | auto / npvt | 0.01110 | 0.01104 | 1.00× |
| boundary | 31 | dense | 31 / 0 | auto / npvt | 0.01428 | 0.01438 | 1.01× |
| boundary | 33 | dense | 33 / 0 | auto / npvt | 0.01595 | 0.01595 | 1.00× |
| boundary | 63 | dense | 63 / 0 | auto / npvt | 0.02678 | 0.02678 | 1.00× |
| boundary | 65 | dense | 65 / 0 | auto / npvt | 0.06284 | 0.06320 | 1.01× |
| boundary | 127 | dense | 127 / 0 | auto / npvt | 0.08173 | 0.08138 | 1.00× |
| boundary | 129 | dense | 129 / 0 | auto / npvt | 0.10743 | 0.10732 | 1.00× |
| boundary | 255 | dense | 255 / 0 | auto / npvt | 0.17244 | 0.17443 | 1.00× |
| boundary | 257 | dense | 257 / 0 | auto / npvt | 0.19846 | 0.19736 | 1.00× |
| boundary | 511 | dense | 511 / 0 | auto / npvt | 0.37299 | 0.36806 | 0.99× |
| boundary | 513 | dense | 513 / 0 | auto / npvt | 0.40672 | 0.40975 | 1.01× |
| boundary | 767 | dense | 767 / 0 | auto / npvt | 0.61987 | 0.62454 | 1.02× |
| boundary | 769 | dense | 769 / 0 | auto / npvt | 0.65336 | 0.65953 | 1.01× |
| boundary | 895 | dense | 895 / 0 | auto / npvt | 0.74058 | 0.73667 | 0.99× |
| boundary | 896 | dense | 896 / 0 | auto / npvt | 0.65198 | 0.65675 | 1.00× |
| boundary | 897 | dense | 897 / 0 | auto / npvt | 0.69661 | 0.69260 | 0.99× |
| boundary | 1023 | dense | 1023 / 0 | auto / npvt | 0.77130 | 0.76954 | 1.00× |
| boundary | 1025 | dense | 1025 / 0 | auto / npvt | 0.80291 | 0.81154 | 1.00× |
| boundary | 2047 | dense | 2047 / 0 | auto / npvt | 1.74574 | 1.74576 | 1.00× |
| boundary | 2049 | dense | 2049 / 0 | auto / npvt | 1.79509 | 1.79908 | 1.00× |
| boundary | 2559 | dense | 2559 / 0 | auto / npvt | 2.37850 | 2.38201 | 1.00× |
| boundary | 2561 | dense | 2561 / 0 | auto / npvt | 2.51850 | 2.51113 | 0.99× |
| boundary | 3071 | dense | 3071 / 0 | auto / npvt | 3.13026 | 3.12619 | 1.00× |
| boundary | 3073 | dense | 3073 / 0 | auto / npvt | 3.19498 | 3.22841 | 1.01× |
| boundary | 4095 | dense | 4095 / 0 | auto / npvt | 4.77470 | 4.77556 | 1.00× |
| boundary | 4097 | dense | 4097 / 0 | auto / npvt | 4.98301 | 4.84217 | 0.97× |
| boundary | 8191 | dense | 8191 / 0 | auto / npvt | 19.40229 | 27.66249 | 1.43× |
| boundary | 8193 | dense | 8193 / 0 | auto / npvt | 19.94706 | 27.98398 | 1.40× |

## Static 与动态主元参考补充实验

| 阶段 | n | 输入 | lda / offset | 方法 / 参考 | Hybrid ms | 无 hybrid ms | OFF/ON |
|---|---:|---|---|---|---:|---:|---:|
| static | 64 | dense | 64 / 0 | static / npvt | 0.07694 | 0.07634 | 1.00× |
| static | 512 | dense | 512 / 0 | static / npvt | 0.69729 | 0.69685 | 1.00× |
| static | 2048 | dense | 2048 / 0 | static / npvt | 3.00485 | 3.00735 | 1.00× |
| static | 4096 | dense | 4096 / 0 | static / npvt | 6.67575 | 6.67400 | 1.00× |
| static | 8192 | dense | 8192 / 0 | static / npvt | 28.57524 | 28.45818 | 0.99× |
| pivot | 64 | dense | 64 / 0 | auto / pivot | 0.02690 | 0.02678 | 1.00× |
| pivot | 512 | dense | 512 / 0 | auto / pivot | 0.36430 | 0.36702 | 1.01× |
| pivot | 2048 | dense | 2048 / 0 | auto / pivot | 1.70682 | 1.72108 | 1.00× |
| pivot | 8192 | dense | 8192 / 0 | auto / pivot | 18.55788 | 27.39719 | 1.48× |
| pivot | 16384 | dense | 16384 / 0 | auto / pivot | 102.28837 | 222.74813 | 2.18× |

## 数值失败记录

原评测中失败的配置：('input', 2048, 2048, 0, 'auto', 'npvt', 'random')

- `t1-input-auto-npvt-random-n2048-lda2048-off0-hybrid.log`：returncode=1，scaled residual=1.4289e+02/3.1391e+07。
- `t1-input-auto-npvt-random-n2048-lda2048-off0-nohybrid.log`：returncode=1，scaled residual=1.4289e+02/3.1391e+07。
- `t2-input-auto-npvt-random-n2048-lda2048-off0-nohybrid.log`：returncode=1，scaled residual=1.4289e+02/3.1391e+07。
- `t2-input-auto-npvt-random-n2048-lda2048-off0-hybrid.log`：returncode=1，scaled residual=1.4289e+02/3.1391e+07。
- `t3-input-auto-npvt-random-n2048-lda2048-off0-hybrid.log`：returncode=1，scaled residual=1.4289e+02/3.1391e+07。
- `t3-input-auto-npvt-random-n2048-lda2048-off0-nohybrid.log`：returncode=1，scaled residual=1.4289e+02/3.1391e+07。
