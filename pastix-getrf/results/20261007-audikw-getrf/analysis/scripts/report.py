#!/usr/bin/env python3
"""Generate a standalone report from the auditable summary table."""
import csv
import html
import json
from pathlib import Path
import sys

base=Path(sys.argv[1]).resolve()
rows=list(csv.DictReader((base/'analysis/summary.csv').open()))
selection=json.loads((base/'workload/selection.json').read_text())
env=json.loads((base/'environment/manifest.json').read_text())
history=json.loads((base/'historical/inventory.json').read_text())
affinity=json.loads((base/'workload/distribution.command.json').read_text())['argv'][2]
def table(headers,records):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join('<td>'+html.escape(str(c))+'</td>' for c in r)+'</tr>' for r in records)+'</tbody></table>'
def link(path,text):return f'<a href="{html.escape(path)}">{html.escape(text)}</a>'
summary=[]
for r in rows:
    summary.append([r['backend'],'ROCm 7.2.4; CDLS '+r['implementation_revision'][:10] if r['backend']=='cdls' else 'ROCm 7.2.4; rocSOLVER 3.32.0.dabb6df2b9',
        f'audikw_1 Schur 后 cblk={r["cblk"]} / call={r["sparse_call_id"]}',r['n'],r['lda'],r['real_occurrences'],
        f'{r["total_kernels"]} / {r["initialization_kernels"]} / {r["compute_kernels"]}',
        f'{float(r["median_us"]):.3f} [{float(r["p10_us"]):.3f}, {float(r["p90_us"]):.3f}]',
        f'{float(r["wall_median_us"]):.3f}',f'PASS; info=0; residual≤{float(r["residual_relative_max"]):.3e}; padding 不变'])
parts=['''<!doctype html><html lang="zh"><meta charset="utf-8"><title>MI300A audikw_1 GETRF 执行粒度证据</title>
<style>body{font:16px/1.65 system-ui,sans-serif;max-width:1450px;margin:32px auto;padding:0 24px;color:#20232a}table{border-collapse:collapse;width:100%;font-size:14px;margin:24px 0}th,td{border:1px solid #cdd2d8;padding:8px;text-align:left;vertical-align:top}th{background:#edf1f5}code{background:#edf1f5;padding:2px 4px}a{color:#1556a5}p{max-width:1200px}</style>
<h1>MI300A audikw_1 GETRF：第一轮最小证据</h1>
<p>本轮真实对角输入中，rocSOLVER 在 (n,lda)=(189,837) 和 (351,1914) 上将 LU 计算拆为多个 kernel；CDLS 以一个 fused 计算 kernel 完成，另含三个初始化 kernel。完整无 profiler 调用分别快 1.917×、1.862×。最小输入 (27,201) 的两者均只有一个计算 kernel：CDLS 将 info 清零并入计算，完整调用快 1.331×。这些数据支持启动数量和端到端调用耗时的并列描述，不支持把全部收益归因于 launch overhead。</p>
<p>表中时长为微秒；“总 / 初始化 / 计算”是每次完整调用的 GPU dispatch 数，包含 HIP memset 实际产生的 kernel。无 profiler 耗时为每个后端 90 个样本的中位数，方括号为 P10–P90；另给同步 wall 中位数作为参考。</p>''',
table(['backend','软件版本','真实输入来源','n','lda','真实频数','kernel 总/初始化/计算','无 profiler 完整调用 µs [P10,P90]','同步 wall µs','数值验证'],summary)]
parts += [f'<p>节点 ppac-pl1-s24-16，Slurm 22565，PPAC_MI300A_SPX 单 GPU，MI300A gfx942，228 CU；ROCm 7.2.4，HIP runtime 70253211，rocSOLVER 3.32.0.dabb6df2b9，rocBLAS 5.2.0.dabb6df2b9。PaStiX HEAD {env["pastix_head"]}，CDLS HEAD {env["cdls_head"]}。代码带既有工作区改动和本轮审计补丁；完整 SHA-256、动态链接路径和源码包见 '+link('environment/manifest.json','环境 manifest')+'。运行中的插件是 hip-hipmm-cdls-getrf / cdlsDgetrf；基线是 rocsolver_dgetrf_npvt；均为 FP64、列主序、无行交换、无 threshold pivot correction。本轮没有使用 CK standalone 耗时替代 CDLS 测量。</p>',
f'''<p>配置固定为普通 1D、update1（rocBLAS GEMM + workspace update，EnablePrefetch=false），Metis seed=3452，min/max blocksize=160/320，tasks2d_level=0，LU，static pivoting=0，2 个 CPU 线程，CPU affinity={affinity}，单 GPU，32 GiB pool，8 个 queue，GETRF queue=0。源码显示 update0 的 fused-scatter 路径有预取，故它不满足本轮约束。两次采集使用相同 ordering 参数，全部 10,898 个 (call_id,cblk,n,lda,backend,queue) 逐项一致；同一进程中所有 GETRF 的实际 HIP stream 相同，地址记录在逐调用 CSV。</p>
<p>输入使用 matrix-repaired/audikw_1.mtx，943,695 阶、39,297,771 个存储条目；修复文件 SHA-256=b2239f90db6a9a5c74e7bc4c6e365876a85752c6a87830e7abcd05d646bd33bb。它与旧读取失败文件明确区分。solver 在 numfact 前进行了正常的矩阵归一化，本轮捕获的是该实际流程中的内容，不是原 Matrix Market 子矩阵。</p>''',
f'<p>完整工作负载共 {selection["total_calls"]:,} 次 GETRF、{selection["unique_pairs"]:,} 种 (n,lda)、163 种 n；n 范围为 3–351。n≤128 有 9,389 次（86.153%），n&gt;128 有 1,509 次（13.847%），n&gt;4096 为零。本配置未触发 hybrid。最常见的 n 是 60（433 次），最常见的精确 (n,lda) 则是 (27,201)（44 次），二者含义不同。</p>',
'''<p>选择规则在任何对比测量前写入脚本并运行：选最高频精确 pair；选最接近 modal n 与最大 n 的区间中点的观测 pair；选最大 n；同 n 时按 pair 频数和 lda 破同分。中等输入是尺寸区间中点附近，不是调用加权中位数。三个 pair 共出现 46 次，占全部调用 0.4221%；其中有 LU 计算拆分的两个 pair 共出现 2 次，占 0.01835%。若只看 n，三个尺寸共 228 次（2.0921%），189/351 共 15 次（0.1376%）；其余 lda 尚未 trace，不能当作精确覆盖。</p>
<p>每个 pair 只捕获第一次出现的实际内容，共三个具体对角块。44 次频数不表示捕获了 44 份内容。所有 lda*n 个原始 FP64 值均保存，包含 n..lda−1 的稀疏非对角行；这里“padding”是 GETRF 视角的非参与行，本身可能含有有效稀疏系数。捕获 D2H 排入 GETRF 的同一个 stream，并在拷贝后同步，位于前驱 Schur 更新和 diagonal GEADD 之后、cdlsDgetrf 之前。两个实现每次计时前均恢复完全相同的 byte image。</p>''',
table(['输入 pair','原 sparse call_id / cblk','频数','覆盖','原始字节数'],[[f'({x["n"]},{x["lda"]})',
 f'{x["first_call_id"]} / '+next(r['cblk'] for r in rows if int(r['n'])==x['n']),x['count'],f'{100*x["call_fraction"]:.4f}%',8*x['n']*x['lda']] for x in selection['selected']]),
'''<p>执行结构按提交关联 ID 归属：每个 backend / pair 在 5 次预热后独立 profile 3 次，每次用带 backend、局部 call_id、n、lda 的 ROCTX 范围包住完整调用及同步。分析程序将 Kernel Correlation_Id 连到唯一 HIP API，再要求 API 在线程一致的唯一 GETRF 范围内，GPU 开始/结束也位于该范围内。所有 339 个 kernel dispatch 均唯一归属；3 次结构完全一致。输入恢复、分配、预热和校验在 profiler 暂停区；HIP 注册日志虽仍存在，也未因时间窗口或总进程计数混入。profile call_id=0/1/2 为重放局部编号；分析 CSV 同时保留原 sparse_call_id/cblk/input SHA。</p>
<p>(27,201)：rocSOLVER 为 reset_info → getf2_npvt_small_kernel&lt;27&gt;；CDLS 为 getrf_small_kernel&lt;true&gt;，info 清零融合在其中。这里减少的是单独 info-reset 启动，计算 kernel 数未减少。</p>
<p>(189,837)：rocSOLVER 为 iota_n → reset_info → 11 轮 [GETF2&lt;16&gt; → unit_forward_substitution → Tensile GEMM] → GETF2&lt;13&gt;，共 2 个初始化、34 个计算 kernel。CDLS 为 info memset（1 个 fill kernel）→ task workspace memset（同一次 HIP API 产生 2 个 fill kernel）→ getrf_kernel&lt;512,true,false,false&gt;，共 3+1 个。减少 32 个总启动，计算从 34 个变为 1 个，同时初始化数增加 1。</p>
<p>(351,1914)：rocSOLVER 为 iota_n → reset_info → 21 轮 [GETF2&lt;16&gt; → unit_forward_substitution → Tensile GEMM] → GETF2&lt;15&gt;，共 2+64 个。CDLS 为同样的 3 次初始化 dispatch 和 1 个 fused kernel。减少 62 个总启动，计算从 64 个变为 1 个。unit_forward_substitution 是内部三角求解计算，计入 internal_TRSM 类；Cijk_* 是内部 GEMM。这些都是 GETRF 内部工作，外层稀疏 GEMM/TRSM 未包含。完整 kernel 名、GPU 执行开始顺序、提交顺序、API 和归属保存在 kernel-attribution.csv；简化序列见 kernel-sequences.json。</p>
<p>计时沿用已有 runner 的方法：5 次预热，3 个独立进程 trial，每 trial 30 次交替配对；trial2 反转尺寸顺序。先完成无 profiler 的全部 270 对/540 次调用，再采 trace。rocSOLVER workspace 预查询与分配、CDLS handle/BLAS 状态与输入备份都在计时外；同一非阻塞 stream 上用 HIP events 包住完整 public GETRF，包括正常 info/workspace 初始化与主机供给 kernel 的间隙，排除恢复、分配、验证。另记录 host submit 与同步 wall（后者含 event/sync 管理开销），保留全部原始样本。三 trial 中位数、MAD、IQR、P10/P90、极值和标准差可复算。</p>
<p>验证共 18 份结果：每 trial、每后端完整 GPU LU 重构，要求 ||A−LU||F/(||A||F·n·ε)≤100，所有因子有限，padding 逐 bit 不变。rocSOLVER public info 使用与 solver 相同的 mapped pinned storage；当前 CDLS ordinary entry 没有 public singular-pivot info，插件写 host_info=0 只是占位。因此通过版本绑定的 internal workspace 最后一个 int 读取 CDLS 真实 zero-pivot 状态，所有预热和计时调用均为 0，且读取在计时外。CDLS 的“info=0”应注明 internal diagnostic，不能表述为已验证其 public info 契约。相对 LU 残差分别至多 1.295e−16、4.554e−16、6.880e−16。</p>
<p>全稀疏运行的 backward residual 为 distribution 5.046826e−16、capture 5.186442e−16，数值分解和 solve 均完成。不过例程的更严格 forward-error 检查打印 FAILED，退出码均为 255；原日志和 exit status 保留。本报告对具体 GETRF 输入的 PASS 不改写全 solver 的退出状态，也不将这两个有审计/捕获扰动的 numfact 时间用于稀疏性能比较。</p>
<p>论文可支持：“在 MI300A 的 audikw_1 实际对角块样本中，rocSOLVER 在两个中等/较大样本上将无主元 LU 分为 panel、三角求解和 GEMM 的多 kernel 序列；当前 fused 实现显著减少完整调用的 GPU 启动数，并在同版本、同输入、无 profiler 测量中降低完整调用延迟。”也可单独说明小块将 info 初始化并入计算。不能声称所有对角块都存在计算拆分、不能把两个 pair 的证据扩展成所有 1,509 个 n&gt;128 调用、不能据此估计整个 audikw 分解的加速贡献。</p>
<p>尚需补测：若论文需要广泛覆盖率，按调用频数补齐其余 (n,lda) 或建立实测验证的 dispatch 分组；若要定量归因 launch overhead，需要额外受控实验区分提交间隙、计算路径、访存与同步变化。没有硬件性能指标，不宣称已经测得 GPU 利用率不足。hybrid 在本 audikw 配置中没有实际输入，不能用大 dense 消融说明其稀疏贡献。</p>
<p>历史库存独立归档，未将旧 ROCm 6.4.2 dense 耗时与 7.2.4 kernel 数拼接。6.4.2 的 rocSOLVER 为 3.28.2、rocBLAS 为 4.4.1；旧 CDLS binary/source hash 与本轮也分别保留。可复用范围如下：</p>''',
table(['库存','ROCm','状态（核验快照）','可复用项及限制'],[[x['label'],x['ROCm'],x['status'],x['note']] for x in history]),
'''<p>不支持强假设的历史结果同样保留：ROCm 6.4.2 的 dense (32768,32768) 测量中 CDLS 中位数 629.460 ms，rocSOLVER 607.862 ms，CDLS 较慢（speedup=0.9657）。原 116 配置中 random n2048 三轮数值失败，115 个配置通过；这些失败不能被后续 retest 覆盖。旧 trace 使用同版本但不同的合成内容，只能独立用于结构研究；不会与本轮真实输入构造混合数据行。aligned hybrid ON/OFF 库存在本次核验时仍运行，本轮未追加 hybrid 消融任务。</p>
<p>复算入口：'''+link('analysis/summary.csv','汇总 CSV')+'；'+link('analysis/samples.csv','全部原始计时样本')+'；'+link('workload/frequency.csv','完整 (n,lda) 频数')+'；'+link('analysis/kernel-attribution.csv','逐 kernel 归属')+'；'+link('analysis/verification.json','分析检查结果')+'；'+link('inputs/manifest.json','真实输入和 SHA')+'；'+link('historical/inventory.json','历史库存')+'。原始日志/命令在 workload、timing；原始 trace 在 traces；代码和重跑说明在 analysis/reproduce.txt 与 analysis/scripts。</p></html>']
(base/'REPORT.html').write_text('\n'.join(parts))
print(base/'REPORT.html')
