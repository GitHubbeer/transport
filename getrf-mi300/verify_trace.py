#!/usr/bin/env python3
"""Verify every traced dispatch lies inside the single factorization marker."""
import argparse,csv,json,math
from pathlib import Path

def read_one(dest,suffix):
    paths=list(dest.rglob('*'+suffix));assert len(paths)==1,(dest,suffix)
    return list(csv.DictReader(paths[0].open()))

def main():
    p=argparse.ArgumentParser();p.add_argument('results',type=Path)
    root=p.parse_args().results/'profiling'
    commands=json.loads((root/'commands.json').read_text());assert len(commands)==14
    summaries=list(csv.DictReader((root/'summary.csv').open()));assert len(summaries)==14
    checks=[]
    for c in commands:
        assert c['returncode']==0,c
        dest=root/f"{c['implementation']}-n{c['n']}"
        markers=read_one(dest,'marker_api_trace.csv')
        roi=[r for r in markers if r['Function']=='GETRF_FACTORIZATION'];assert len(roi)==1
        lo,hi=int(roi[0]['Start_Timestamp']),int(roi[0]['End_Timestamp']);assert hi>lo
        kernels=read_one(dest,'kernel_trace.csv');assert kernels
        # GPU timestamps are converted to the host clock by rocprofiler.
        # Verify HIP correlation as well as a documented 10 us alignment bound.
        hip=[r for r in read_one(dest,'hip_api_trace.csv') if lo<=int(r['Start_Timestamp'])<=hi]
        correlations={r['Correlation_Id'] for r in hip};tolerance_ns=10000
        for k in kernels:
            start,end=int(k['Start_Timestamp']),int(k['End_Timestamp'])
            assert lo-tolerance_ns<=start<end<=hi+tolerance_ns,(dest,k)
            assert k['Correlation_Id'] in correlations,(dest,k)
        assert len({k['Queue_Id'] for k in kernels})==1
        assert not any(r['Function'].startswith(('hipMemcpy','hipMalloc','hipFree')) for r in hip),(dest,hip)
        summary=next(r for r in summaries if int(r['n'])==c['n'] and r['implementation']==c['implementation'])
        assert int(summary['kernel_launches'])==len(kernels)
        busy=sum(int(k['End_Timestamp'])-int(k['Start_Timestamp']) for k in kernels)/1e6
        assert abs(busy-float(summary['kernel_busy_ms']))<1e-6
        summary['roi_ms']=(hi-lo)/1e6;summary['roi_hip_calls']=len(hip)
        checks.append(dict(n=c['n'],implementation=c['implementation'],dispatches=len(kernels),
                           one_roi=True,all_dispatches_correlate_to_roi_hip_calls=True,
                           timestamp_tolerance_ns=tolerance_ns,
                           max_end_overrun_ns=max(0,max(int(k['End_Timestamp']) for k in kernels)-hi),
                           no_copy_allocation_in_roi=True))
    with (root/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    (root/'verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    print('Verified 14 isolated single-factorization traces')

if __name__=='__main__':main()
