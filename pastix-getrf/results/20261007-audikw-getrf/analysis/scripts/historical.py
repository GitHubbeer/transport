#!/usr/bin/env python3
"""Inventory existing data without merging versions or unfinished experiments."""
import collections
import csv
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys

root=Path(__file__).resolve().parents[2]
base=Path(sys.argv[1]).resolve();dest=base/'historical';dest.mkdir(exist_ok=False)
sources=[
 ('cdls-dense-20261006',root.parent/'cdls/profiling/getrf-mi300/results/20261006-current','6.4.2',
  '可复用已通过配置的 dense 计时与同版本 trace；不是 audikw 实际输入；random n2048 三轮失败必须保留。'),
 ('cdls-random2048-retest',root.parent/'cdls/profiling/getrf-mi300/results/20261006T095710Z-n2048-retest','6.4.2',
  '单独的稳定性补测；不能覆盖或替换原失败记录。'),
 ('hybrid-pilot',root.parent/'cdls/profiling/getrf-mi300/results/20261007-hybrid-ablation','6.4.2',
  'SUPERSEDED_PILOT：范围较小的 ON/OFF 消融，仅探索性复用，不当作完整对齐评测。'),
 ('hybrid-aligned',root.parent/'cdls/profiling/getrf-mi300/results/20261007-hybrid-ablation-aligned','6.4.2',
  '匹配原 116 配置的 ON/OFF 实验；库存核验时仍 RUNNING，不纳入本轮已完成结果。'),
 ('ck-dense-20261002',root/'third_party/ck_getrf/docs/performance/20261002','6.4.2',
  '旧 CK dense 曲线、原始 samples 与 source/binary hash，可单独作历史结果；不能与本轮 7.2.4 计数配对。'),
 ('pastix-ck-20261002',root/'results/20261002-ck-vs-rocsolver','7.2.4',
  '稀疏全分解计时和固定二进制；可复用旧 rocSOLVER plugin 二进制身份，本轮未用其耗时推断 GETRF。'),
 ('pastix-cdls-20261006',root/'results/20261006-cdls-update1-mi300a','7.2.4',
  'audikw repaired 输入、1D update1 的环境/资源/命令与 backend 身份可复用；没有逐 GETRF shape/trace，不能替代新采集。'),
]
inventory=[]
for label,p,version,note in sources:
    assert p.is_dir()
    d=dest/label;d.mkdir()
    meta_name=next((name for name in ['metadata.json','manifest.json','EXPERIMENT.json'] if (p/name).exists()),None)
    meta=json.loads((p/meta_name).read_text()) if meta_name else {}
    files={};copied=[]
    for f in p.rglob('*'):
        if not f.is_file():continue
        # Pin raw observations and their analysis inputs; no profiler timings
        # are copied into the new performance table.
        if f.suffix not in ('.csv','.json','.txt','.log','.py','.cpp','.h','.hpp','.md'):continue
        h=hashlib.sha256(f.read_bytes()).hexdigest();files[str(f.relative_to(p))]=h
    for name in [meta_name,'verification.json','summary.csv','statistics.csv','samples.csv','compiler.txt','ldd.txt','git-head.txt','git-diff.txt','commands.json']:
        if name and (p/name).is_file():shutil.copyfile(p/name,d/name);copied.append(name)
    if (p/'profiling/summary.csv').exists():shutil.copyfile(p/'profiling/summary.csv',d/'profile-summary.csv');copied.append('profile-summary.csv')
    (d/'source-files-sha256.json').write_text(json.dumps(files,indent=2)+'\n')
    summary=[]
    if (p/'summary.csv').exists():summary=list(csv.DictReader((p/'summary.csv').open()))
    valid=dict(collections.Counter(r.get('validated','unspecified') for r in summary))
    inventory.append(dict(label=label,path=str(p),ROCm=version,status=meta.get('status','unspecified'),
         recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
         note=note,summary_rows=len(summary),validation_flags=valid,copied=copied,hashed_source_records=len(files)))
(dest/'inventory.json').write_text(json.dumps(inventory,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(inventory,indent=2,ensure_ascii=False))
