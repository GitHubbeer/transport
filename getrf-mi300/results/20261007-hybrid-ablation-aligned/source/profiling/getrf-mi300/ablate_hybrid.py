#!/usr/bin/env python3
"""Alternate ON/OFF builds of the same public GETRF, using benchmark.cpp.

GPU runs are serialized. Every invocation also measures rocSOLVER npvt as a
drift reference and validates both factors with full GPU LU reconstruction.
"""
import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def fields(line):
    return dict(re.findall(r'(\w+)=([^\s]+)', line))


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def write_csv(path, rows):
    if rows:
        with path.open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def summarize(out, meta, commands):
    raw, summaries = [], []
    for entry in commands:
        for line in (out / entry['log']).read_text().splitlines():
            if line.startswith('sample '):
                raw.append(dict(case=entry['case'], trial=entry['trial'],
                                variant=entry['variant'], validated=entry['validated'],
                                **fields(line)))
    write_csv(out / 'samples.csv', raw)
    for case in meta['cases']:
        runs = [e for e in commands if e['case'] == case['name']]
        valid = len(runs) == 2 * meta['trials'] and all(e['validated'] for e in runs)
        row = dict(case=case['name'], n=case['n'], lda=case['lda'], offset=case['offset'],
                   phase=case['phase'], matrix=case.get('matrix', 'dense'),
                   method=case.get('method', 'auto'), reference=case.get('reference', 'npvt'),
                   repeats=case.get('repeats', meta['repeats']), warmup=case.get('warmup', meta['warmup']),
                   validated=valid, hybrid_ms='', nohybrid_ms='',
                   speedup='', speedup_min='', speedup_max='', hybrid_gflops='',
                   nohybrid_gflops='', hybrid_scaled_residual='', nohybrid_scaled_residual='',
                   roc_reference_ms='')
        if valid:
            medians = {v: [float(e['bench']['cdls_median_ms']) for e in runs
                            if e['variant'] == v] for v in ('hybrid', 'nohybrid')}
            h, f = (statistics.median(medians[v]) for v in ('hybrid', 'nohybrid'))
            ratios = []
            for trial in range(1, meta['trials'] + 1):
                pair = {e['variant']: float(e['bench']['cdls_median_ms'])
                        for e in runs if e['trial'] == trial}
                ratios.append(pair['nohybrid'] / pair['hybrid'])
            flop = case['n'] * (case['n'] - 1.0) * (4.0 * case['n'] + 1) / 6
            row.update(hybrid_ms=h, nohybrid_ms=f, speedup=statistics.median(ratios),
                       speedup_min=min(ratios), speedup_max=max(ratios),
                       hybrid_gflops=flop / (h * 1e6), nohybrid_gflops=flop / (f * 1e6),
                       roc_reference_ms=statistics.median(float(e['bench']['rocsolver_median_ms']) for e in runs))
            for v in ('hybrid', 'nohybrid'):
                row[v + '_scaled_residual'] = max(float(e['validation']['residual_scaled'].split('/')[0])
                                                 for e in runs if e['variant'] == v)
        summaries.append(row)
    write_csv(out / 'summary.csv', summaries)
    primary = [r for r in summaries if r['phase'] == 'size' and r['validated']]
    lines = ['# Hybrid 消融实验：MI300A FP64 no-pivot LU', '',
             f"状态：{meta['status']}；节点：`{meta['hostname']}`；开始：{meta['start_utc']}。", '',
             '对照为同一源码的 `CDLS_GETRF_ENABLE_HYBRID=ON/OFF` 两个 Release 构建。'
             'OFF 将完整矩阵交给已有的 fused tiled 路径，保持 tile、MFMA、调度与数值算法配置。'
             'ON 在 n>4096 时使用 512 列外层 panel、rocBLAS DGEMM 更新及 ≤4096 的 fused 尾部。'
             '这里的 hybrid 指 GPU 内部的 kernel/BLAS 组合，不是 CPU/GPU 分工。', '',
             f"主实验固定 seed=20261001、lda=n、列主序 FP64，非对角元素 uniform[-1,1]、对角 n+1。"
             f"每种实现每个配置 {meta['trials']} 轮，每轮 {meta['repeats']} 次计时、{meta['warmup']} 次预热。"
             '轮次交替 ON/OFF 顺序，并反转尺寸顺序；同一 GPU 串行执行。'
             '每个进程也交替测量 rocSOLVER no-pivot 作为参考；hybrid 消融加速比只由 CDLS 两个构建计算。', '',
             'HIP event 测量公共 factorization 调用，包含 workspace/info 初始化和主机提交间隙；'
             '输入恢复、分配与校验排除在计时外。结果为各轮中位数的中位数；加速比为逐轮 OFF/ON 比值的中位数，'
             '范围表示轮次最小/最大值，不是置信区间。每个进程对最终 LU 做完整 GPU 重构，检查有限值、padding、'
             'counter 和 rocSOLVER info；scaled residual=||A-LU||F/(||A||F*n*epsilon)≤100。', '',
             '| n | Hybrid (ms) | 无 hybrid (ms) | 加速比 OFF/ON | 轮次范围 |',
             '|---:|---:|---:|---:|---:|']
    for row in primary:
        lines.append(f"| {row['n']} | {row['hybrid_ms']:.4f} | {row['nohybrid_ms']:.4f} | "
                     f"{row['speedup']:.2f}× | {row['speedup_min']:.2f}–{row['speedup_max']:.2f}× |")
    large = [r for r in primary if r['n'] > 4096]
    if large:
        lines += ['', f"在本次有效的大矩阵配置（n>4096）中，hybrid 加速比为 "
                  f"{min(r['speedup'] for r in large):.2f}–{max(r['speedup'] for r in large):.2f}×。"
                  '该实验支持关于此硬件、FP64 和当前 fused 实现扩展性的结论；'
                  '不能据此推断所有非 hybrid 算法或其他 GPU 都需要相同设计。']
    failed = [e for e in commands if not e['validated']]
    if failed:
        lines += ['', '失败运行（保留日志，不计入有效性能结论）：']
        lines += [f"- `{e['log']}`，returncode={e['returncode']}" for e in failed]
    lines += ['', '布局补充实验见 `summary.csv`。原始逐次数据见 `samples.csv`、完整日志及 '
              '`commands.json`；源码/二进制/构建缓存及 SHA-256 见 `source/`、各 variant 目录、'
             '`metadata.json` 和 `MANIFEST.json`。']
    if meta.get('reference_run'):
        lines[6:6] = ['', f"尺寸、输入、布局、method/reference 和每个配置的预热/重复次数逐项复用之前的评测计划："
                       f"`{meta['reference_run']}`，共 {len(meta['cases'])} 个配置。"
                       '除 ON/OFF 开关外，数值实现和 benchmark 源码与此前一致。'
                       '小尺寸每轮预热 20 次，通常尺寸 5 次，n≥24576 为 3 次；'
                       '主尺寸/边界/布局通常每轮 30 个样本，输入/static/pivot 通常 20 个，n≥24576 为 10 个。'
                       '每个配置的确切参数保存在 metadata.json。'
                       'static 路径不使用 hybrid，可作为开关无影响的对照；pivot phase 的 rocSOLVER 参考包含动态主元。']
        lines = [line.replace(f"每种实现每个配置 {meta['trials']} 轮，每轮 {meta['repeats']} 次计时、{meta['warmup']} 次预热。",
                              f"每种实现每个配置 {meta['trials']} 轮，预热及重复次数逐项按原计划执行。") for line in lines]
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    return primary


def plot(out, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    ns = [r['n'] for r in rows]
    for variant, label in [('hybrid', 'Hybrid'), ('nohybrid', 'Fused only')]:
        axes[0].plot(ns, [r[variant + '_ms'] for r in rows], 'o-', label=label)
        axes[1].plot(ns, [r[variant + '_gflops'] / 1000 for r in rows], 'o-', label=label)
    axes[0].set_yscale('log')
    axes[0].set_ylabel('Factorization latency (ms)')
    axes[1].set_ylabel('FP64 throughput (TFLOP/s)')
    axes[2].plot(ns, [r['speedup'] for r in rows], 'o-')
    axes[2].fill_between(ns, [r['speedup_min'] for r in rows],
                         [r['speedup_max'] for r in rows], alpha=.2)
    axes[2].axhline(1, color='gray', linewidth=1)
    axes[2].set_ylabel('Speedup: fused only / hybrid')
    for ax in axes:
        ax.set_xscale('log', base=2)
        ax.set_xlabel('Matrix dimension n')
        ax.axvline(4096, color='gray', linestyle='--', linewidth=1)
        ax.grid(alpha=.2)
    axes[0].legend()
    axes[1].legend()
    fig.suptitle('MI300A: FP64 no-pivot LU hybrid ablation\nMedians; shaded range = trial range')
    fig.tight_layout()
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(out / ('hybrid-ablation.' + suffix), dpi=180)
    plt.close(fig)


def execute(out, meta, cases, commands):
    completed = {(e['trial'], e['case'], e['variant']) for e in commands}
    rocm = Path(os.getenv('ROCM_PATH', '/nfsapps/ubuntu-24.04/opt/rocm-6.4.2'))
    for trial in range(1, meta['trials'] + 1):
        ran = False
        for case in (cases if trial % 2 else list(reversed(cases))):
            for variant in (('hybrid', 'nohybrid') if trial % 2 else ('nohybrid', 'hybrid')):
                if (trial, case['name'], variant) in completed:
                    continue
                ran = True
                env = os.environ.copy()
                env['LD_LIBRARY_PATH'] = str(out / variant) + ':' + env.get('LD_LIBRARY_PATH', '')
                name = f"t{trial}-{case['name']}-{variant}"
                repeats = case.get('repeats', meta['repeats'])
                warmup = case.get('warmup', meta['warmup'])
                argv = [str(out / 'benchmark'), '--samples', '--n', str(case['n']), '--lda', str(case['lda']),
                        '--offset', str(case['offset']), '--repeats', str(repeats), '--warmup', str(warmup),
                        '--matrix', case.get('matrix', 'dense'), '--method', case.get('method', 'auto'),
                        '--reference', case.get('reference', 'npvt')]
                entry = dict(case=case['name'], trial=trial, variant=variant, argv=argv,
                             library_sha256=meta['library_sha256'][variant], log=name + '.log', start_utc=utc())
                with (out / entry['log']).open('w') as stream:
                    try:
                        entry['returncode'] = subprocess.run(argv, env=env, stdout=stream,
                                                             stderr=subprocess.STDOUT, timeout=1800).returncode
                    except subprocess.TimeoutExpired:
                        entry['returncode'] = 124
                lines = (out / entry['log']).read_text().splitlines()
                validations = [x for x in lines if x.startswith('validate ')]
                benches = [x for x in lines if x.startswith('bench ')]
                entry.update(end_utc=utc(), validation=fields(validations[0]) if len(validations) == 1 else {},
                             bench=fields(benches[0]) if len(benches) == 1 else {})
                entry['validated'] = (entry['returncode'] == 0 and len(validations) == 1
                                      and validations[0].endswith(' PASS') and len(benches) == 1
                                      and sum(x.startswith('sample ') for x in lines) == repeats)
                commands.append(entry)
                save(out / 'commands.json', commands)
                print(f"{utc()} {name} {'PASS' if entry['validated'] else 'FAIL'} "
                      f"{entry['bench'].get('cdls_median_ms', '?')} ms", flush=True)
        if ran:
            with (out / f'gpu-after-t{trial}.txt').open('w') as stream:
                subprocess.run([str(rocm / 'bin/rocm-smi')], stdout=stream, stderr=subprocess.STDOUT)
    assert meta['source_sha256'] == {f: sha(ROOT / f) for f in meta['source_sha256']}, 'Source changed during run'
    meta.update(status='PASS' if all(e['validated'] for e in commands) else 'FAIL', end_utc=utc(),
                execution_complete=True, total_runs=len(commands), valid_runs=sum(e['validated'] for e in commands))
    save(out / 'metadata.json', meta)
    summarize(out, meta, commands)
    save(out / 'MANIFEST.json', {str(f.relative_to(out)): sha(f)
                                for f in out.rglob('*') if f.is_file() and f.name != 'MANIFEST.json'})
    if meta['status'] != 'PASS':
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    p.add_argument('--binary', type=Path, default=ROOT / 'build/cdls-getrf-bench')
    p.add_argument('--hybrid-build', type=Path, default=ROOT / 'build/getrf-hybrid-ablation')
    p.add_argument('--nohybrid-build', type=Path, default=ROOT / 'build/getrf-nohybrid')
    p.add_argument('--reference-run', type=Path, help='Reuse the exact plan saved in a previous run metadata.json')
    p.add_argument('--max-n', type=int, help='Keep previous-plan configurations with n at or below this limit')
    p.add_argument('--resume', action='store_true', help='Reuse frozen binaries and completed runs; only shrink the plan')
    p.add_argument('--sizes', nargs='+', type=int)
    p.add_argument('--trials', type=int, default=3)
    p.add_argument('--repeats', type=int, default=30)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--no-layout', action='store_true')
    p.add_argument('--plan-only', action='store_true')
    p.add_argument('--plot-only', action='store_true')
    args = p.parse_args()
    out = args.output.resolve()
    if min(args.trials, args.repeats, args.warmup, *(args.sizes or [1])) <= 0:
        p.error('sizes, trials, repeats, warmup must be positive')
    if args.max_n is not None and args.max_n <= 0:
        p.error('--max-n must be positive')
    reference_meta = None
    if args.reference_run:
        if args.sizes or args.no_layout:
            p.error('--reference-run uses the full previous plan; do not combine it with --sizes or --no-layout')
        reference_meta = json.loads((args.reference_run / 'metadata.json').read_text())
        cases = []
        for c in reference_meta['plan']:
            if c['trials'] != args.trials:
                p.error('--trials must match the previous plan')
            cases.append(dict(c, name=f"{c['phase']}-{c['method']}-{c['reference']}-{c['matrix']}-n{c['n']}-lda{c['lda']}-off{c['offset']}"))
    else:
        sizes = args.sizes or [2048, 4096, 4097, 5120, 6144, 8192, 10240, 12288, 16384, 24576, 32768]
        cases = [dict(name=f'n{n}', n=n, lda=n, offset=0, phase='size') for n in sizes]
        if not args.no_layout:
            cases += [dict(name='n8192-pad17-off1', n=8192, lda=8209, offset=1, phase='layout'),
                      dict(name='n16384-pad128', n=16384, lda=16512, offset=0, phase='layout')]
    if args.max_n is not None:
        cases = [c for c in cases if c['n'] <= args.max_n]
    if not cases:
        p.error('No configurations remain within the requested limit')
    if args.plan_only:
        print(json.dumps(cases, indent=2))
        return
    if args.resume:
        meta = json.loads((out / 'metadata.json').read_text())
        if args.trials != meta['trials']:
            p.error('Resume must retain the original trial count')
        old_cases = {c['name']: c for c in meta['cases']}
        if not all(old_cases.get(c['name']) == c for c in cases):
            p.error('Resume may only retain unchanged configurations from the original plan')
        if sha(out / 'benchmark') != meta['binary_sha256']:
            p.error('Frozen benchmark binary changed')
        for variant, expected in meta['library_sha256'].items():
            if sha(out / variant / 'libcdls_rocm.so') != expected:
                p.error(f'Frozen {variant} library changed')
        runner = str(Path(__file__).resolve().relative_to(ROOT))
        for filename, expected in meta['source_sha256'].items():
            if filename != runner and sha(ROOT / filename) != expected:
                p.error(f'Numerical/benchmark source changed: {filename}')
        if meta['source_sha256'][runner] != sha(Path(__file__)):
            history = out / 'source-versions'
            history.mkdir(exist_ok=True)
            old_sha = meta['source_sha256'][runner]
            shutil.copy2(out / 'source' / runner, history / f'ablate_hybrid-{old_sha}.py')
            meta.setdefault('runner_history', []).append(dict(sha256=old_sha, retained_snapshot=str(history.name),
                                                              reason='Add max-size filtering/resume; benchmark and libraries unchanged'))
            shutil.copy2(__file__, out / 'source' / runner)
            meta['source_sha256'][runner] = sha(Path(__file__))
        commands = json.loads((out / 'commands.json').read_text())
        selected = {c['name'] for c in cases}
        excluded = [e for e in commands if e['case'] not in selected]
        if excluded:
            archived = json.loads((out / 'excluded-commands.json').read_text()) if (out / 'excluded-commands.json').exists() else []
            save(out / 'excluded-commands.json', archived + excluded)
        commands = [e for e in commands if e['case'] in selected]
        meta.update(cases=cases, status='RUNNING', max_n=args.max_n, execution_complete=False,
                    resumed_utc=utc(), resume_arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
        save(out / 'metadata.json', meta)
        save(out / 'commands.json', commands)
        save(out / 'PLAN_ALIGNMENT.json', dict(exact_plan_match=True, reference_case_filter=f'n <= {args.max_n}' if args.max_n else 'all',
                                              previous_full_cases=len(reference_meta['plan']) if reference_meta else len(old_cases), cases=len(cases),
                                              runs=sum(c.get('trials', meta['trials']) * 2 for c in cases),
                                              paired_samples=sum(c.get('trials', meta['trials']) * c.get('repeats', meta['repeats']) * 2 for c in cases)))
        execute(out, meta, cases, commands)
        return
    if args.plot_only:
        meta = json.loads((out / 'metadata.json').read_text())
        commands = json.loads((out / 'commands.json').read_text())
        plot(out, summarize(out, meta, commands))
        save(out / 'MANIFEST.json', {str(f.relative_to(out)): sha(f)
                                    for f in out.rglob('*') if f.is_file() and f.name != 'MANIFEST.json'})
        return
    builds = dict(hybrid=args.hybrid_build.resolve(), nohybrid=args.nohybrid_build.resolve())
    settings = {}
    for variant, build in builds.items():
        cache = (build / 'CMakeCache.txt').read_text()
        expected = 'ON' if variant == 'hybrid' else 'OFF'
        if f'CDLS_GETRF_ENABLE_HYBRID:BOOL={expected}\n' not in cache:
            p.error(f'{build} must have CDLS_GETRF_ENABLE_HYBRID={expected}')
        settings[variant] = {line.split(':', 1)[0]: line.split('=', 1)[1]
                             for line in cache.splitlines()
                             if re.match(r'(CMAKE_BUILD_TYPE|CMAKE_HIP_ARCHITECTURES|CMAKE_HIP_COMPILER|CMAKE_HIP_FLAGS\w*|CDLS_CK_SOURCE_DIR):', line)}
    if settings['hybrid'] != settings['nohybrid']:
        p.error('Both builds must use the same compiler, architecture, flags and CK source')
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.binary, out / 'benchmark')
    sources = [ROOT / 'CMakeLists.txt', HERE / 'benchmark.cpp', Path(__file__), ROOT / 'include/cdls/getrf.h']
    sources += [f for directory in ('src/common/getrf', 'src/rocm', 'include/cdls')
                for f in (ROOT / directory).rglob('*') if f.is_file() and 'cuda_wrapper' not in f.parts]
    sources = sorted(set(sources))
    for f in sources:
        dest = out / 'source' / f.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest)
    libraries = {}
    for variant, build in builds.items():
        dest = out / variant
        dest.mkdir()
        shutil.copy2(build / 'src/rocm/libcdls_rocm.so', dest)
        shutil.copy2(build / 'CMakeCache.txt', dest)
        for filename in ('flags.make', 'link.txt'):
            shutil.copy2(build / 'src/rocm/CMakeFiles/cdls_rocm.dir' / filename, dest)
        libraries[variant] = sha(dest / 'libcdls_rocm.so')
    meta = dict(status='RUNNING', start_utc=utc(), hostname=platform.node(), cases=cases, max_n=args.max_n,
                trials=args.trials, repeats=args.repeats, warmup=args.warmup,
                arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                source_sha256={str(f.relative_to(ROOT)): sha(f) for f in sources},
                binary_sha256=sha(out / 'benchmark'), library_sha256=libraries, build_settings=settings,
                environment={k: v for k, v in os.environ.items()
                             if k.startswith(('SLURM_', 'ROCR_', 'HIP_', 'HSA_'))
                             or k in ('ROCM_PATH', 'LD_LIBRARY_PATH', 'LOADEDMODULES')})
    if reference_meta:
        shutil.copy2(args.reference_run / 'metadata.json', out / 'reference-metadata.json')
        meta.update(reference_run=str(args.reference_run.resolve()),
                    reference_metadata_sha256=sha(out / 'reference-metadata.json'),
                    reference_source_changes=[f for f, old in reference_meta['source_sha256'].items()
                                              if (ROOT / f).is_file() and sha(ROOT / f) != old])
    rocm = Path(os.getenv('ROCM_PATH', '/nfsapps/ubuntu-24.04/opt/rocm-6.4.2'))
    for name, argv in [('git-head', ['git', 'rev-parse', 'HEAD']), ('git-diff', ['git', 'diff']),
                       ('compiler', [str(rocm / 'bin/hipcc'), '--version']),
                       ('gpu-before', [str(rocm / 'bin/rocm-smi')]), ('ldd', ['ldd', str(out / 'benchmark')])]:
        with (out / (name + '.txt')).open('w') as stream:
            subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, check=False)
    ck_source = settings['hybrid'].get('CDLS_CK_SOURCE_DIR')
    if ck_source:
        for name, argv in [('ck-head', ['git', '-C', ck_source, 'rev-parse', 'HEAD']),
                           ('ck-diff', ['git', '-C', ck_source, 'diff']),
                           ('ck-status', ['git', '-C', ck_source, 'status', '--short'])]:
            with (out / (name + '.txt')).open('w') as stream:
                subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, check=False)
    for variant in builds:
        env = os.environ.copy()
        env['LD_LIBRARY_PATH'] = str(out / variant) + ':' + env.get('LD_LIBRARY_PATH', '')
        with (out / variant / 'ldd.txt').open('w') as stream:
            subprocess.run(['ldd', str(out / 'benchmark')], env=env,
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
    save(out / 'metadata.json', meta)
    execute(out, meta, cases, [])


if __name__ == '__main__':
    main()
