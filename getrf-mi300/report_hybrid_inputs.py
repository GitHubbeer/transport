#!/usr/bin/env python3
"""Report all aligned input/layout phases after ablate_hybrid.py completes."""
import argparse
import csv
import json
from pathlib import Path
import shutil


def plot_large(out, primary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 7,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7})
    rows = [primary[n] for n in sorted(primary) if n >= 4096]
    ns = [int(r['n']) for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
    for variant, label in [('hybrid', 'Hybrid'), ('nohybrid', 'Fused only')]:
        axes[0].plot(ns, [float(r[variant + '_ms']) for r in rows], 'o-', label=label)
        axes[1].plot(ns, [float(r[variant + '_gflops']) / 1000 for r in rows], 'o-', label=label)
    axes[0].set_ylabel('Factorization latency (ms)')
    axes[1].set_ylabel('FP64 throughput (TFLOP/s)')
    axes[2].plot(ns, [float(r['speedup']) for r in rows], 'o-')
    axes[2].fill_between(ns, [float(r['speedup_min']) for r in rows],
                         [float(r['speedup_max']) for r in rows], alpha=.2)
    axes[2].axhline(1, color='gray', linestyle='--', linewidth=1)
    axes[2].set_ylabel('Speedup: fused only / hybrid')
    for ax in axes:
        ax.set_xlabel('Matrix dimension n')
        ax.set_xticks([n for n in (4096, 8192, 12288, 16384) if min(ns) <= n <= max(ns)])
        ax.tick_params(axis='x', labelsize=7)
        ax.grid(alpha=.2)
    axes[0].legend()
    axes[1].legend()
    fig.suptitle(f'MI300A: FP64 no-pivot LU hybrid ablation (n ≤ {max(ns)})\n'
                 'Same inputs as the previous evaluation; 3 trials; medians and trial range', fontsize=9)
    fig.tight_layout()
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(out / ('hybrid-ablation-large.' + suffix), dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('results', type=Path)
    p.add_argument('--plot', action='store_true', help='Also draw the large-matrix figure (requires Matplotlib)')
    args = p.parse_args()
    out = args.results
    meta = json.loads((out / 'metadata.json').read_text())
    if not meta.get('execution_complete'):
        p.error('Wait until the complete measurement plan has finished')
    rows = list(csv.DictReader((out / 'summary.csv').open()))
    commands = json.loads((out / 'commands.json').read_text())
    reference_commands = out / 'reference-commands.json'
    if not reference_commands.exists():
        reference_commands = Path(meta['reference_run']) / 'commands.json'
    previous = json.loads(reference_commands.read_text())
    alignment = json.loads((out / 'PLAN_ALIGNMENT.json').read_text())
    assert alignment['exact_plan_match']
    lines = ['# 与原评测对齐的输入和布局结果', '',
             f"完整计划已执行：{len(rows)} 个配置、{len(commands)} 个运行；"
             f"{sum(e['validated'] for e in commands)} 个运行通过全部校验。", '',
             '下表耗时为各轮中位数的中位数，加速比为对应轮次 OFF/ON 比值的中位数。'
             '失败配置保留原始耗时，但不列入有效加速比结论。', '']
    if meta.get('max_n'):
        lines[2:2] = [f"按用户要求，尺寸上限为 n={meta['max_n']}；在该范围内逐项复用原计划的全部输入和布局。", '']
    for phases, title in [(('input',), '输入类型'),
                          (('padding', 'stride', 'offset'), '布局'),
                          (('boundary',), '边界'),
                          (('static', 'pivot'), 'Static 与动态主元参考补充实验')]:
        lines += [f'## {title}', '',
                  '| 阶段 | n | 输入 | lda / offset | 方法 / 参考 | Hybrid ms | 无 hybrid ms | OFF/ON |',
                  '|---|---:|---|---|---|---:|---:|---:|']
        for row in rows:
            if row['phase'] not in phases:
                continue
            valid = row['validated'] == 'True'
            timing = (f"{float(row['hybrid_ms']):.5f} | {float(row['nohybrid_ms']):.5f} | "
                      f"{float(row['speedup']):.2f}×" if valid else '— | — | 校验失败')
            lines.append(f"| {row['phase']} | {row['n']} | {row['matrix']} | "
                         f"{row['lda']} / {row['offset']} | {row['method']} / {row['reference']} | {timing} |")
        lines += ['']
    old_failures = {(e['phase'], e['n'], e['lda'], e['offset'], e['method'], e['reference'], e['matrix'])
                    for e in previous if not e['validated']}
    failures = [e for e in commands if not e['validated']]
    lines += ['## 数值失败记录', '',
              '原评测中失败的配置：' + (', '.join(map(str, sorted(old_failures))) or '无'), '']
    for e in failures:
        lines.append(f"- `{e['log']}`：returncode={e['returncode']}，"
                     f"scaled residual={e['validation'].get('residual_scaled', 'missing')}。")
    if not failures:
        lines.append('本次所有配置均通过校验。')
    (out / 'INPUTS_LAYOUT.md').write_text('\n'.join(lines) + '\n')
    report = out / 'REPORT.md'
    text = report.read_text()
    scope = (f"计划已完整执行：n≤{meta.get('max_n') or max(int(r['n']) for r in rows)}，"
             f"{len(rows)} 个配置、{len(commands)} 个运行、3 轮；"
             f"{sum(e['validated'] for e in commands)} 个运行通过校验。"
             '数值失败记录保留并排除在有效加速比中。\n\n')
    text = text.replace('\n状态：FAIL；', '\n执行完成；数值校验状态：FAIL；')
    if scope not in text:
        text = text.replace('\n\n对照为', '\n\n' + scope + '对照为', 1)
    if meta.get('max_n') and meta['max_n'] < 24576:
        text = text.replace('，n≥24576 为 3 次', '').replace('，n≥24576 为 10 个', '')
    text = text.replace('参考包含动态主元。\n主实验', '参考包含动态主元。\n\n主实验')
    text = text.replace('有效性能结论）：\n- ', '有效性能结论）：\n\n- ')
    if '详细输入/布局/边界表见 [INPUTS_LAYOUT.md]' not in text:
        text += '\n详细输入/布局/边界表见 [INPUTS_LAYOUT.md](INPUTS_LAYOUT.md)。'
        text += '\n论文大矩阵图见 [hybrid-ablation-large.pdf](hybrid-ablation-large.pdf)，英文图注见 [PAPER_CAPTION.txt](PAPER_CAPTION.txt)。\n'
    report.write_text(text)
    primary = {int(r['n']): r for r in rows if r['phase'] == 'size' and r['validated'] == 'True'}
    if 8192 in primary and max(primary) >= 8192:
        largest = max(primary)
        small, large = primary[8192], primary[largest]
        caption = ('Hybrid execution ablation for FP64 no-pivot LU on AMD MI300A. '
                   'The fused-only baseline uses the same numerical kernels and scheduling settings, '
                   'with all BLAS trailing-matrix updates disabled. The size sweep uses the same '
                   'diagonally dominant matrices as the previous evaluation (seed 20261001, lda=n). '
                   'Points show medians across three trial medians; the shaded speedup range shows '
                   'the minimum and maximum paired trial ratios, rather than a confidence interval. '
                   f"Hybrid execution achieves {float(small['speedup']):.2f}x speedup at n=8192 "
                   f"and {float(large['speedup']):.2f}x at n={largest}. At n={largest}, latency falls from "
                   f"{float(large['nohybrid_ms']):.2f} ms to {float(large['hybrid_ms']):.2f} ms. "
                   'The increasing benefit supports hybrid execution as a way to improve large-matrix '
                   'scaling of this fused implementation on this hardware. Input restoration, '
                   'allocation and validation are excluded from factorization timing.\n')
        (out / 'PAPER_CAPTION.txt').write_text(caption)
    if args.plot:
        plot_large(out, primary)
    dest = out / 'analysis-source'
    dest.mkdir(exist_ok=True)
    shutil.copy2(__file__, dest)


if __name__ == '__main__':
    main()
