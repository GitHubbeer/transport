#!/usr/bin/env python3
"""Serialized paired CDLS / rocSOLVER measurements of the current public API."""
import argparse, csv, datetime, hashlib, json, os, platform, re, shutil, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

def plan():
    cases = []
    def add(phase, n, matrix='dense', lda=None, offset=0, method='auto', reference='npvt'):
        repeats = 10 if n >= 24576 else 20 if phase in ('input','static','pivot') else 30
        cases.append(dict(phase=phase, n=n, lda=lda or n, offset=offset, method=method,
                          reference=reference, matrix=matrix, trials=3, repeats=repeats,
                          warmup=20 if n <= 128 else 3 if n >= 24576 else 5))
    for n in [1,2,4,8,16,32,64,128,256,512,768,1024,1536,2048,2560,3072,4096,
              5120,6144,8192,10240,12288,16384,24576,32768]: add('size',n)
    for n in [7,15,17,31,33,63,65,127,129,255,257,511,513,767,769,895,896,897,
              1023,1025,2047,2049,2559,2561,3071,3073,4095,4097,8191,8193]: add('boundary',n)
    for n in [64,512,2048,4096,8192,16384]:
        for pad in [1,17,128]: add('padding',n,lda=n+pad)
    for n in [64,512,2048,8192]: add('stride',n,lda=2*n)
    for n in [64,512,2048,8192]: add('offset',n,offset=1)
    for matrix in ['signed','column_scaled','identity']:
        for n in [64,512,2048,8192,16384]: add('input',n,matrix=matrix)
    for n in [64,512]: add('input',n,matrix='known_lu')
    for n in [64,512,2048]: add('input',n,matrix='random')
    for matrix in ['scale_low','scale_high']:
        for n in [65,2048]: add('input',n,matrix=matrix)
    for n in [64,512,2048,4096,8192]: add('static',n,method='static')
    for n in [64,512,2048,8192,16384,32768]: add('pivot',n,reference='pivot')
    return cases

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def parse(line): return dict(re.findall(r'(\w+)=([^\s]+)',line))
def save_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp'); temp.write_text(json.dumps(value,indent=2)+'\n'); temp.replace(path)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path); p.add_argument('--binary',type=Path,default=ROOT/'build/cdls-getrf-bench')
    p.add_argument('--library',type=Path,default=ROOT/'build/getrf-shared/src/rocm/libcdls_rocm.so')
    p.add_argument('--phases',nargs='+'); p.add_argument('--plan-only',action='store_true'); p.add_argument('--resume',action='store_true')
    args=p.parse_args(); cases=[c for c in plan() if not args.phases or c['phase'] in args.phases]
    if args.plan_only: print(json.dumps(cases,indent=2)); return
    out=args.output.resolve(); sources=[ROOT/'CMakeLists.txt', ROOT/'include/cdls/getrf.h', HERE/'benchmark.cpp',HERE/'run.py']
    sources+=list((ROOT/'src/common/getrf').glob('*'))+list((ROOT/'src/rocm').glob('*.cpp'))+list((ROOT/'src/rocm').glob('*.h'))+list((ROOT/'src/rocm/getrf').glob('*'))
    identities={str(f.relative_to(ROOT)):sha(f) for f in sources}
    env=os.environ.copy(); env['LD_LIBRARY_PATH']=str(out)+':'+env.get('LD_LIBRARY_PATH','')
    if args.resume:
        meta=json.loads((out/'metadata.json').read_text()); commands=json.loads((out/'commands.json').read_text())
        assert meta['plan']==cases and meta['source_sha256']==identities
        assert sha(out/'benchmark')==meta['binary_sha256'] and sha(out/'libcdls_rocm.so')==meta['library_sha256']
    else:
        out.mkdir(parents=True,exist_ok=False); shutil.copy2(args.binary,out/'benchmark'); shutil.copy2(args.library,out/'libcdls_rocm.so')
        for f in sources:
            target=out/'source'/f.relative_to(ROOT); target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(f,target)
        meta=dict(status='RUNNING',start_utc=utc(),hostname=platform.node(),platform=platform.platform(),plan=cases,
                  source_sha256=identities,binary_sha256=sha(out/'benchmark'),library_sha256=sha(out/'libcdls_rocm.so'),
                  precision='FP64',baseline='rocsolver_dgetrf_npvt; pivoted GETRF is a separate supplementary phase',
                  seed=20261001,timing='Same nonblocking stream; HIP events include initialization and host-fed launch gaps; input restore/allocations/validation excluded; explicit rocSOLVER workspace query and preallocation; alternating paired order',
                  environment={k:v for k,v in os.environ.items() if k.startswith(('SLURM_','ROCR_','HIP_','HSA_')) or k in ('ROCM_PATH','LD_LIBRARY_PATH','LOADEDMODULES')})
        commands=[]
        for name,argv in [('git-head',['git','rev-parse','HEAD']),('git-diff',['git','diff']),('ldd',['ldd',str(out/'benchmark')]),('gpu-before',['/nfsapps/ubuntu-24.04/opt/rocm-6.4.2/bin/rocm-smi']),('compiler',['/nfsapps/ubuntu-24.04/opt/rocm-6.4.2/bin/hipcc','--version'])]:
            with (out/(name+'.txt')).open('w') as f: subprocess.run(argv,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=False)
    def save():
        save_json(out/'metadata.json',meta); save_json(out/'commands.json',commands)
    completed={c['name'] for c in commands}; save()
    for phase in dict.fromkeys(c['phase'] for c in cases):
        subset=[c for c in cases if c['phase']==phase]
        for trial in range(1,4):
            for c in (subset if trial%2 else list(reversed(subset))):
                name=f"{phase}-t{trial}-{c['method']}-{c['reference']}-{c['matrix']}-n{c['n']}-lda{c['lda']}-off{c['offset']}"
                if name in completed: continue
                argv=[str(out/'benchmark'),'--samples']
                for key in ['n','lda','offset','method','reference','matrix','repeats','warmup']: argv+=['--'+key,str(c[key])]
                entry=dict(c,trial=trial,name=name,argv=argv,log=name+'.log',start_utc=utc())
                with (out/entry['log']).open('w') as f:
                    try: entry['returncode']=subprocess.run(argv,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1800).returncode
                    except subprocess.TimeoutExpired: entry['returncode']=124
                lines=(out/entry['log']).read_text().splitlines()
                entry.update(end_utc=utc(),validation_passes=sum(x.startswith('validate ') and x.endswith(' PASS') for x in lines),samples=sum(x.startswith('sample ') for x in lines))
                entry['validated']=entry['returncode']==0 and entry['validation_passes']==1 and entry['samples']==c['repeats']
                benches=[x for x in lines if x.startswith('bench ')]
                entry['bench']=parse(benches[0]) if len(benches)==1 else {}
                commands.append(entry); save()
                print(f"{utc()} {len(commands)}/{sum(c['trials'] for c in cases)} {name} {'PASS' if entry['validated'] else 'FAIL'} "+(benches[0] if benches else '\n'.join(lines[-3:])),flush=True)
        with (out/('gpu-after-'+phase+'.txt')).open('w') as f: subprocess.run(['/nfsapps/ubuntu-24.04/opt/rocm-6.4.2/bin/rocm-smi'],stdout=f,stderr=subprocess.STDOUT)
        assert identities=={str(f.relative_to(ROOT)):sha(f) for f in sources}, 'Source changed during measurement'
    meta.update(status='PASS' if all(c['validated'] for c in commands) else 'FAIL',end_utc=utc()); save()
    print(f"Complete: {len(commands)} runs, {sum(c['validated'] for c in commands)} validated",flush=True)
    if meta['status']!='PASS': raise SystemExit(1)
if __name__=='__main__': main()
