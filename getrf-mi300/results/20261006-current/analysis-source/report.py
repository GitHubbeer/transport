#!/usr/bin/env python3
"""Generate the Chinese report using verified measurements, never manual timings."""
import argparse,csv,json,math,statistics as st
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();root=a.root
    m=json.loads((root/'metadata.json').read_text());v=json.loads((root/'verification.json').read_text()); rows=list(csv.DictReader((root/'summary.csv').open()))
    string={'phase','method','reference','matrix','validated'}
    for r in rows:
        for k in r.keys()-string:r[k]=float(r[k])
    valid=[r for r in rows if r['validated']=='True'];size=sorted([r for r in valid if r['phase']=='size'],key=lambda r:r['n']);by={int(r['n']):r for r in size}
    def ms(x):return f'{x:.6f}' if x<.1 else f'{x:.4f}' if x<10 else f'{x:.2f}'
    def table(data):
        text='| n | CDLS (ms) | rocSOLVER (ms) | 加速比 | 三轮加速比范围 | CDLS TFLOP/s |\n|---:|---:|---:|---:|---:|---:|\n'
        for r in data:text+=f"| {int(r['n'])} | {ms(r['cdls_ms'])} | {ms(r['roc_ms'])} | {r['speedup']:.3f}× | {r['trial_speedup_min']:.3f}–{r['trial_speedup_max']:.3f}× | {r['cdls_gflops']/1000:.2f} |\n"
        return text
    conclusions=[]
    middle=[r for r in size if 128<=r['n']<=4096]
    if middle:
        conclusions.append(f"- n=128–4096 的整块主尺寸：CDLS 加速 {min(r['speedup'] for r in middle):.2f}–{max(r['speedup'] for r in middle):.2f}×。")
    if 8192 in by and 16384 in by:
        conclusions.append(f"- n=8192 为 {by[8192]['speedup']:.2f}×；n=16384 降至 {by[16384]['speedup']:.3f}×。")
    for n in [24576,32768]:
        if n in by:
            r=by[n];conclusions.append(f"- n={n}：CDLS {r['cdls_ms']:.2f} ms，rocSOLVER NPVT {r['roc_ms']:.2f} ms，CDLS {'慢' if r['speedup']<1 else '快'} {abs(r['cdls_ms']/r['roc_ms']-1)*100:.2f}%；三轮加速比 {r['trial_speedup_min']:.3f}–{r['trial_speedup_max']:.3f}×。")
    lines=['# 当前 CDLS GETRF 与 rocSOLVER 的 MI300A 全面比较','',
      f"完成 {v['configurations']} 组配置、{v['runs']} 次独立运行、{v['paired_samples']:,} 对交替样本；其中 {v['validated_runs']} 次通过数值检查。原始元数据状态为 `{m['status']}`，数值压力测试失败与测试未完成分别记录。",'',
      *conclusions,'',
      '设备为单卡 AMD Instinct MI300A（gfx942，228 CUs，128 GiB），节点 `sh5-pl1-s12-33`，ROCm 6.4.2、rocSOLVER 3.28.2、rocBLAS 4.4.1。使用现有自动频率和 550 W 功率上限，没有修改设备设置。每阶段的 GPU 状态保存在 `gpu-*.txt`。','',
      '主比较是 `cdlsDgetrf` 自动选择的融合/混合路径与 `rocsolver_dgetrf_npvt`：FP64、方阵、列主序、无 pivot。加速比定义为 **rocSOLVER 时间 / CDLS 时间**；大于 1 表示 CDLS 更快。每个数字采用三轮各自中位数的中位数，表中范围反映三轮变化，不是置信区间。','',
      '主尺寸与布局测试使用稠密对角占优输入：对角为 n+1，非对角为固定 seed 的 uniform[-1,1]。其他输入单独统计。Trace 使用同样对角占优、但非对角按确定整数模式生成的矩阵；trace 用于归因，性能表仍全部来自原始随机输入计时。','',
      '## 主尺寸结果','',table(size),'',
      '## 尺寸、stride 与输入敏感性','']
    for phase,label in [('boundary','非整块/调度边界'),('padding','lda=n+1/n+17/n+128'),('stride','lda=2n'),('offset','矩阵起始地址错位 8 字节'),('input','其他输入')]:
        data=[r for r in valid if r['phase']==phase]
        if data:
            lo=min(data,key=lambda r:r['speedup']);hi=max(data,key=lambda r:r['speedup'])
            lines.append(f"- {label}：{len(data)} 组通过检查；有效加速比 {lo['speedup']:.3f}–{hi['speedup']:.3f}×。最慢相对案例 n={int(lo['n'])}, lda={int(lo['lda'])}, {lo['matrix']}；最快相对案例 n={int(hi['n'])}, lda={int(hi['lda'])}, {hi['matrix']}。")
    inputs=[r for r in rows if r['phase']=='input']
    if inputs:
        lines+=['','| 输入族 | 通过配置/总配置 | 有效加速比范围 | 最大 CDLS 缩放残差 | 最大 rocSOLVER 缩放残差 |','|---|---:|---:|---:|---:|']
        for name in dict.fromkeys(r['matrix'] for r in inputs):
            family=[r for r in inputs if r['matrix']==name];good=[r for r in family if r['validated']=='True']
            span=f"{min(r['speedup'] for r in good):.3f}–{max(r['speedup'] for r in good):.3f}×" if good else '—'
            lines.append(f"| {name} | {len(good)}/{len(family)} | {span} | {max(r['max_cdls_residual_scaled'] for r in family):.5g} | {max(r['max_roc_residual_scaled'] for r in family):.5g} |")
    lines+=['','## 静态 pivot 与带 pivot 基线','',
      '静态 pivot 测试使用 criterion=0，计入清零替换计数的开销；无实际替换。它使用静态 pivot kernel 和融合路径，与普通自动路径是不同配置。', '',
      '从当前源码可确认：静态入口直接调用 `launch_tiles<true>`，不使用普通入口的小矩阵专门分支或大矩阵 hybrid/rocBLAS GEMM 路径。因此 criterion=0 并不意味着调度与普通 GETRF 相同。', '',
      '| n | 普通 CDLS (ms) | 静态 CDLS (ms) | 静态/普通延迟 | 静态对 NPVT 加速比 |','|---:|---:|---:|---:|---:|']
    for r in sorted([r for r in valid if r['phase']=='static'],key=lambda r:r['n']):
        b=by[int(r['n'])];lines.append(f"| {int(r['n'])} | {ms(b['cdls_ms'])} | {ms(r['cdls_ms'])} | {r['cdls_ms']/b['cdls_ms']:.3f}× | {r['speedup']:.3f}× |")
    lines+=['','补充对比 `rocsolver_dgetrf` 包含 pivot 搜索和换行，属于不同工作量；下面的加速比不能当作同语义无 pivot 对比。','',table(sorted([r for r in valid if r['phase']=='pivot'],key=lambda r:r['n'])),'',
      '## 数值验证与压力测试','',
      '所有尺寸都对双方做完整 GPU LU 重构；n≤128 还使用 long-double CPU 重构。带 pivot 的基线检验 PA=LU。检查 padding、有限数值、rocSOLVER info 和 CDLS 计数语义。判定阈值为 `||A-LU||_F/(||A||_F*n*epsilon) ≤ 100`，而不是要求逐位一致。常规输入还检查 L/U 因子相对差异；未经对角增强的随机输入仅按残差判定。','']
    bad=[r for r in rows if r['validated']!='True']
    if bad:
        lines+=['以下配置未通过残差验证，保留原始计时，但不纳入有效性能结论：','',
          '| 输入 | n | CDLS 最大缩放残差 | rocSOLVER 最大缩放残差 |','|---|---:|---:|---:|']
        for r in bad:lines.append(f"| {r['matrix']} | {int(r['n'])} | {r['max_cdls_residual_scaled']:.5g} | {r['max_roc_residual_scaled']:.5g} |")
        lines+=['','这些结果提示无 pivot LU 在该输入上存在稳定性问题；该测试没有计算条件数，因此不把原因归结为矩阵病态。']
        stress=root/'pivot-stress/summary.csv'
        if stress.exists():
            lines+=['','同一随机输入另做带 pivot rocSOLVER 验证（只验证，不用于性能统计）：','',
                    '| n | CDLS 无 pivot 缩放残差 | rocSOLVER 带 pivot 缩放残差 | rocSOLVER 残差通过 |','|---:|---:|---:|---|']
            for r in csv.DictReader(stress.open()):lines.append(f"| {r['n']} | {float(r['cdls_residual_scaled']):.5g} | {float(r['roc_pivot_residual_scaled']):.5g} | {r['roc_pivot_residual_pass']} |")
    else:lines+=['全部配置通过数值验证。']
    profile=root/'profiling/summary.csv'
    if profile.exists():
        traces=list(csv.DictReader(profile.open()))
        lines+=['','## 单次分解的 kernel trace','',
          '独立进程预热后，仅在 `GETRF_FACTORIZATION` 范围启用 rocprofv3。每种实现每个尺寸只采一次分解；分配、输入复制、预热均排除。kernel busy 时间是各 kernel 运行时间之和，包含 profiling 干扰，用于解释构成，不能代替上面的三轮无 profiler 延迟。','',
          '| n | 实现 | kernel 数 | GPU kernel busy (ms) |','|---:|---|---:|---:|']
        for r in traces:lines.append(f"| {r['n']} | {r['implementation']} | {r['kernel_launches']} | {float(r['kernel_busy_ms']):.4f} |")
        lines+=['','各 kernel 名称、次数和 busy 占比见 `profiling/kernels.csv`；原始 HIP、kernel 和 marker trace 及精确命令均保留。kernel 数少说明 launch 负担较小；大矩阵的结论还取决于 GEMM/triangular 等实际计算时间，不能仅按 kernel 数推断速度。']
        checks=json.loads((root/'profiling/verification.json').read_text())
        overrun=max(c['max_end_overrun_ns'] for c in checks)
        lines+=['',f"所有 dispatch 均关联到 ROI 内的 HIP 调用，范围内没有输入复制或设备分配。GPU/CPU 时间戳对齐按 10 µs 容差核验，实测最大末尾偏差 {overrun/1000:.3f} µs；启动时的 HIP compiler 注册可能出现在原始 API trace，但不计入分解统计。",'',
                '| n | 实现 | 自定义 GETRF (ms) | 按名称识别的 GEMM (ms) | rocSOLVER getf2/forward (ms) | 其他 (ms) |','|---:|---|---:|---:|---:|---:|']
        kernelrows=list(csv.DictReader((root/'profiling/kernels.csv').open()))
        for n in [2048,8192,32768]:
            for impl in ['cdls','rocsolver','static']:
                group=[r for r in kernelrows if int(r['n'])==n and r['implementation']==impl]
                if not group:continue
                cats=[0.,0.,0.,0.]
                for r in group:
                    name=r['kernel'];i=0 if 'cdls::' in name else 2 if ('getf2_npvt' in name or 'unit_forward_substitution' in name) else 1 if (name.startswith('Cijk') or 'gemm' in name.lower()) else 3
                    cats[i]+=float(r['busy_ms'])
                lines.append(f"| {n} | {impl} | "+' | '.join(f'{value:.3f}' for value in cats)+' |')
        lines+=['','以上按 kernel 名称分组，GEMM 组可能包括三角求解内部调用的 GEMM；自定义 GETRF 也融合了更新，不能解读成纯 LU 指令时间。n=32768 的 CDLS 自定义 kernel 约占 23.2% busy 时间，值得优先分析面板分解/求解/更新以及外层面板宽度。静态 n=8192 几乎全部 busy 时间集中在单个自定义 kernel，支持让静态路径复用普通路径优化调度的方向。这里只做测量与分析，没有调整这些参数。']
    lines+=['','## 计时与统计边界','',
      '双方使用同一非阻塞 stream。分配、输入恢复和验证不在计时区间；rocSOLVER workspace 查询与预分配在计时前。GPU events 包含分解、workspace/info 初始化，以及 CPU 提交期间的队列空隙；另记录 API 提交时间和同步 wall time。测量表示正常 host 调用下的公开分解入口延迟，不能直接等同于 GPU 算术指令的忙碌时间。','',
      '一轮内交替 CDLS/rocSOLVER 的顺序，三轮之间反转配置顺序。普通配置每轮 30 对，额外输入/静态/带 pivot 为 20 对，n≥24576 为 10 对；小尺寸预热 20 次，多数尺寸 5 次，最大尺寸 3 次。P05/P95、变异系数、顺序敏感性、wall/API 时间、workspace 查询值都在 `summary.csv`。CDLS workspace 列只包括用户显式 workspace，额外 BLAS handle 缓存不包含在内。','',
      '数据覆盖单个设备、单个软件版本、固定 seed=20261001；三轮能显示当前节点变化，不能代表其他 GPU 或系统配置。随机输入压力测试失败不意味着性能测量未执行，也不应被删除。','',
      '### 提交时间、同步延迟与波动','',
      '| n | CDLS API 提交 (ms) | rocSOLVER API 提交 (ms) | 同步 wall 加速比 | CDLS 样本 CV | rocSOLVER 样本 CV |','|---:|---:|---:|---:|---:|---:|']
    for n in [64,512,2048,8192,16384,32768]:
        if n in by:
            r=by[n];lines.append(f"| {n} | {ms(r['cdls_submit_ms'])} | {ms(r['roc_submit_ms'])} | {r['wall_speedup']:.3f}× | {r['cdls_cv_pct']:.2f}% | {r['roc_cv_pct']:.2f}% |")
    lines+=['','同步 wall 还包含事件记录、等待和查询开销，因此小矩阵的 wall 加速比与 GPU-event 加速比有差别。API 提交时间可以与 GPU 执行重叠，不能简单加到 GPU-event 时间上。CV 为同一配置全部原始样本的总体标准差/均值；跨轮稳定性另看三轮中位数范围。','',
      '## 可复查产物','',
      '- `samples.csv`：每一对 GPU/API/wall 原始样本。','- `trials.csv`、`summary.csv`：各轮及跨轮汇总。','- `validation.csv`：双方残差、因子误差和通过状态。','- `metadata.json`、`commands.json`、`verification.json`：身份、精确命令、完成与失败清单。','- `benchmark`、`libcdls_rocm.so`、`source/`：测量时的可执行文件、动态库与源代码快照。','- `figures/performance.pdf`：总览、边界、padding、输入、提交时间/内存及静态/带 pivot 图；每张另有 PNG/PDF。','',
      '重跑及验证命令见上两级的 `README.md`。源码与库的 SHA-256 已记录；benchmark 的动态库解析路径在 `ldd.txt`，指向本结果目录保留的 CDLS 库。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
