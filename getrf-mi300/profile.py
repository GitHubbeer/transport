#!/usr/bin/env python3
"""Run after the timing sweep; intrusive traces never supply benchmark timings."""
import argparse,csv,json,os,subprocess
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
ROCM=Path('/nfsapps/ubuntu-24.04/opt/rocm-6.4.2')
def main():
    p=argparse.ArgumentParser();p.add_argument('results',type=Path);a=p.parse_args();out=a.results.resolve();meta=json.loads((out/'metadata.json').read_text())
    assert meta['status'] in ('PASS','FAIL'), 'Wait for the serial timing sweep to finish'
    directory=out/'profiling';directory.mkdir(exist_ok=False)
    binary=directory/'trace';argv=[str(ROCM/'bin/hipcc'),'-O3','-std=c++17','-I'+str(ROOT/'include'),str(ROOT/'profiling/getrf-mi300/trace.cpp'),'-L'+str(out),'-Wl,-rpath,'+str(out),'-lcdls_rocm','-lrocblas','-lrocsolver','-lrocprofiler-sdk-roctx','-o',str(binary)]
    with (directory/'build.log').open('w') as f:subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT,check=True)
    env=os.environ.copy();env['LD_LIBRARY_PATH']=str(out)+':'+str(ROCM/'lib')+':'+env.get('LD_LIBRARY_PATH','')
    commands=[];summary=[];kernels=[]
    for n in [64,512,2048,4096,8192,32768]:
        for implementation in ['cdls','rocsolver']+(['static'] if n in [4096,8192] else []):
            dest=directory/f'{implementation}-n{n}';dest.mkdir()
            argv=[str(ROCM/'bin/rocprofv3'),'--kernel-trace','--hip-trace','--marker-trace','--stats','--output-format','csv','--output-directory',str(dest),'--output-file','trace','--',str(binary),str(n),implementation]
            with (dest/'run.log').open('w') as f:r=subprocess.run(argv,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=600)
            commands.append(dict(n=n,implementation=implementation,argv=argv,returncode=r.returncode))
            (directory/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
            if r.returncode:print(f'FAIL profile {implementation} n={n}',flush=True);continue
            paths=list(dest.rglob('*kernel_trace.csv'))
            if len(paths)!=1: print(f'No unique kernel trace for {dest}',flush=True);continue
            rows=list(csv.DictReader(paths[0].open()));counts=defaultdict(lambda:[0,0])
            for row in rows:
                name=row['Kernel_Name'];duration=(float(row['End_Timestamp'])-float(row['Start_Timestamp']))/1e6
                counts[name][0]+=1;counts[name][1]+=duration
            total=sum(s[1] for s in counts.values());count=sum(s[0] for s in counts.values())
            summary.append(dict(n=n,implementation=implementation,kernel_launches=count,kernel_busy_ms=total,unique_kernels=len(counts)))
            for name,(cnt,ms) in sorted(counts.items(),key=lambda item:-item[1][1]):kernels.append(dict(n=n,implementation=implementation,kernel=name,launches=cnt,busy_ms=ms,busy_fraction=ms/total if total else 0))
            print(f'profile {implementation} n={n}: {count} kernels, {total:.6f} ms GPU busy',flush=True)
    for name,rows in [('summary',summary),('kernels',kernels)]:
        if rows:
            with (directory/(name+'.csv')).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
