#!/usr/bin/env python3
"""Re-verify preserved logs and derive tables/figures from paired samples."""
import argparse, csv, hashlib, json, math, re, statistics as st
from collections import defaultdict
from pathlib import Path

def fields(line): return dict(re.findall(r'(\w+)=([^\s]+)',line))
def write_csv(path,rows):
    if rows:
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def pct(values,p):
    values=sorted(values); x=(len(values)-1)*p; lo=int(x); hi=min(lo+1,len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(x-lo)

def main():
    p=argparse.ArgumentParser(); p.add_argument('output',type=Path); p.add_argument('--allow-partial',action='store_true'); a=p.parse_args(); root=a.output
    meta=json.loads((root/'metadata.json').read_text()); cmds=json.loads((root/'commands.json').read_text())
    assert meta['status'] in ('PASS','FAIL') or a.allow_partial
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(root/'benchmark')==meta['binary_sha256'] and sha(root/'libcdls_rocm.so')==meta['library_sha256']
    for path,digest in meta['source_sha256'].items(): assert sha(root/'source'/path)==digest
    columns=['phase','method','reference','matrix','n','lda','offset']; key=lambda x:tuple(x[k] for k in columns)
    samples=[]; trials=[]; validations=[]; groups=defaultdict(list); configs={key(c):c for c in meta['plan']}
    for c in cmds:
        lines=(root/c['log']).read_text().splitlines(); base={k:c[k] for k in columns}|dict(trial=c['trial'],log=c['log'],validated=c['validated'])
        local=[]
        for line in lines:
            if line.startswith('sample '):
                f=fields(line); assert int(f['n'])==c['n'] and int(f['lda'])==c['lda']
                assert f['first']==('rocsolver' if int(f['iteration'])%2 else 'cdls')
                row=base|{k:float(f[k]) for k in ['cdls_ms','rocsolver_ms','cdls_submit_ms','roc_submit_ms','cdls_wall_ms','roc_wall_ms']}|dict(iteration=int(f['iteration']),first=f['first'])
                assert all(math.isfinite(row[k]) and row[k]>0 for k in ['cdls_ms','rocsolver_ms'])
                local.append(row); samples.append(row)
            elif line.startswith('validate '):
                f=fields(line); cr,rr=map(float,f['residual_scaled'].split('/'))
                validations.append(base|dict(info=f['info'],L_rel=float(f['L_rel']),U_rel=float(f['U_rel']),cdls_residual_scaled=cr,roc_residual_scaled=rr,pass_validation=line.endswith(' PASS')))
        if not local: continue
        assert len(local)==c['repeats'] and sorted(r['iteration'] for r in local)==list(range(c['repeats']))
        logged=fields(next(line for line in lines if line.startswith('bench ')))
        ct=st.median(r['cdls_ms'] for r in local); rt=st.median(r['rocsolver_ms'] for r in local)
        assert abs(ct-float(logged['cdls_median_ms']))<6e-7 and abs(rt-float(logged['rocsolver_median_ms']))<6e-7
        row=base|dict(samples=len(local),warmup=c['warmup'],cdls_ms=ct,roc_ms=rt,speedup=rt/ct,workspace_bytes=int(logged['workspace_bytes']),roc_workspace_bytes=int(logged['roc_workspace_bytes']))
        trials.append(row); groups[key(row)].append(row)
    if not a.allow_partial:
        assert len(cmds)==sum(c['trials'] for c in meta['plan']) and set(groups)==set(configs)
    samplegroups=defaultdict(list); valg=defaultdict(list)
    for row in samples: samplegroups[key(row)].append(row)
    for row in validations: valg[key(row)].append(row)
    summary=[]
    for k,ts in sorted(groups.items()):
        cfg=configs[k]; ss=samplegroups[k]; vv=valg[k]
        if not a.allow_partial: assert len(ts)==cfg['trials']
        ct=st.median(r['cdls_ms'] for r in ts); rt=st.median(r['roc_ms'] for r in ts)
        flop=cfg['n']*(cfg['n']-1)*(4*cfg['n']+1)/6
        row=dict(zip(columns,k))|dict(validated=all(t['validated'] for t in ts),trials=len(ts),samples_per_impl=len(ss),cdls_ms=ct,roc_ms=rt,speedup=rt/ct,time_saved_pct=100*(1-ct/rt),trial_speedup_min=min(t['speedup'] for t in ts),trial_speedup_max=max(t['speedup'] for t in ts),cdls_gflops=flop/(ct*1e6),roc_gflops=flop/(rt*1e6),workspace_bytes=ts[0]['workspace_bytes'],roc_workspace_bytes=ts[0]['roc_workspace_bytes'])
        for prefix,field in [('cdls','cdls_ms'),('roc','rocsolver_ms')]:
            vs=[s[field] for s in ss]; row.update({prefix+'_p05_ms':pct(vs,.05),prefix+'_p95_ms':pct(vs,.95),prefix+'_cv_pct':100*st.pstdev(vs)/st.mean(vs)})
        for field in ['cdls_submit_ms','roc_submit_ms','cdls_wall_ms','roc_wall_ms']: row[field]=st.median(s[field] for s in ss)
        row['wall_speedup']=row['roc_wall_ms']/row['cdls_wall_ms']
        row['paired_speedup_p50']=st.median(s['rocsolver_ms']/s['cdls_ms'] for s in ss)
        for first in ['cdls','rocsolver']:
            selected=[s for s in ss if s['first']==first]
            row[first+'_first_cdls_ms']=st.median(s['cdls_ms'] for s in selected)
            row[first+'_first_roc_ms']=st.median(s['rocsolver_ms'] for s in selected)
        row.update(max_L_rel=max(v['L_rel'] for v in vv),max_U_rel=max(v['U_rel'] for v in vv),max_cdls_residual_scaled=max(v['cdls_residual_scaled'] for v in vv),max_roc_residual_scaled=max(v['roc_residual_scaled'] for v in vv))
        summary.append(row)
    for name,rows in [('samples',samples),('trials',trials),('validation',validations),('summary',summary)]: write_csv(root/(name+'.csv'),rows)
    report=dict(status=meta['status'],configurations=len(summary),runs=len(cmds),validated_runs=sum(c['validated'] for c in cmds),paired_samples=len(samples),failed_logs=[c['log'] for c in cmds if not c['validated']])
    (root/'verification.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report))
    for row in summary:
        if row['phase']=='size': print(f"n={row['n']:5} CDLS={row['cdls_ms']:.6f} ms ROC={row['roc_ms']:.6f} ms speedup={row['speedup']:.3f}x trials={row['trial_speedup_min']:.3f}..{row['trial_speedup_max']:.3f}")
if __name__=='__main__': main()
