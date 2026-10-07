#!/usr/bin/env python3
"""Recompute attribution and performance from raw CSV traces and text samples."""
import collections
import csv
import json
import math
from pathlib import Path
import re
import statistics
import sys
from select_inputs import write_csv

def read_csv(path): return list(csv.DictReader(Path(path).open()))
def quantile(v,p):
    v=sorted(v); pos=(len(v)-1)*p;lo=math.floor(pos);hi=math.ceil(pos)
    return v[lo]+(v[hi]-v[lo])*(pos-lo)
def stats(v):
    med=statistics.median(v)
    return dict(samples=len(v),median_us=med,p10_us=quantile(v,.1),p90_us=quantile(v,.9),
                q25_us=quantile(v,.25),q75_us=quantile(v,.75),min_us=min(v),max_us=max(v),
                stdev_us=statistics.stdev(v),mad_us=statistics.median(abs(x-med) for x in v))
def fields(line): return dict(re.findall(r'(\w+)=([^ ]+)',line))

def main(base):
    base=Path(base);out=base/"analysis"
    selection=json.loads((base/"workload/selection.json").read_text())
    inputs=json.loads((base/"inputs/manifest.json").read_text())
    environment=json.loads((base/'environment/manifest.json').read_text())
    workload=read_csv(base/"workload/calls.csv")
    assert len({r['cblk'] for r in workload})==len(workload)
    assert sum(int(r['n']) for r in workload)==943695
    freq=collections.Counter(int(r['n']) for r in workload)
    write_csv(out/"n-frequency.csv",[dict(n=n,count=c,call_fraction=c/len(workload)) for n,c in sorted(freq.items())])
    samples=[];validation=[];versions=set()
    for path in sorted((base/"timing").glob('*.log')):
        trial=int(re.search(r'trial(\d+)',path.name)[1]);text=path.read_text().splitlines()
        command=json.loads(path.with_suffix('.command.json').read_text())
        assert command['exit_code']==0
        assert 'LD_PRELOAD' not in command['environment']
        assert not any(k.startswith('ROCPROFILER_') for k in command['environment'])
        assert 'replay-profile' not in ' '.join(command['argv'])
        versions.add(next(l for l in text if l.startswith('device=')).split(' n=')[0])
        for line in text:
            if line.startswith('sample '):
                d=fields(line);samples.append(dict(trial=trial,**d))
            if line.startswith('validation '):
                d=fields(line);assert d['status']=='PASS';validation.append(dict(trial=trial,**d))
    assert len(versions)==1 and len(samples)==270 and len(validation)==18
    write_csv(out/'samples.csv',samples);write_csv(out/'validation.csv',validation)
    counts=[];attribution=[];categories=[];sequences={}
    for directory in sorted((base/'traces').iterdir()):
        command=json.loads((directory/'run.command.json').read_text());assert command['exit_code']==0
        markers=[r for r in read_csv(directory/'trace_marker_api_trace.csv') if r['Function'].startswith('GETRF|')]
        assert len(markers)==3
        apis=collections.defaultdict(list)
        for r in read_csv(directory/'trace_hip_api_trace.csv'): apis[r['Correlation_Id']].append(r)
        kernels=read_csv(directory/'trace_kernel_trace.csv');groups=collections.defaultdict(list)
        for k in kernels:
            related=apis[k['Correlation_Id']]
            assert len(related)==1
            a=related[0]
            owners=[m for m in markers if a['Thread_Id']==m['Thread_Id'] and
                    int(m['Start_Timestamp'])<=int(a['Start_Timestamp'])<=int(a['End_Timestamp'])<=int(m['End_Timestamp'])]
            assert len(owners)==1
            m=owners[0];assert int(m['Start_Timestamp'])<=int(k['Start_Timestamp'])<=int(k['End_Timestamp'])<=int(m['End_Timestamp'])
            meta=dict(s.split('=',1) for s in m['Function'].split('|')[1:])
            source=next(x for x in inputs if x['n']==int(meta['n']) and x['lda']==int(meta['lda']))
            k=dict(k,backend=meta['backend'],call_id=int(meta['call_id']),n=int(meta['n']),lda=int(meta['lda']),
                   sparse_call_id=source['call_id'],cblk=source['cblk'],input_file=source['filename'],input_sha256=source['sha256'],
                   submitting_api=a['Function'],api_start_ns=a['Start_Timestamp'],api_end_ns=a['End_Timestamp'],
                   marker=m['Function'],trace=str((directory/'trace_kernel_trace.csv').relative_to(base)))
            groups[k['call_id']].append(k)
        assert set(groups)=={0,1,2}
        signatures=[]
        for call,ks in sorted(groups.items()):
            ks.sort(key=lambda k:int(k['Dispatch_Id']))
            execution={id(k):rank+1 for rank,k in enumerate(sorted(ks,key=lambda k:int(k['Start_Timestamp'])))}
            fills=[k['Correlation_Id'] for k in ks if k['Kernel_Name'].startswith('__amd_rocclr_fillBuffer')]
            for position,k in enumerate(ks,1):
                name=k['Kernel_Name'];backend=k['backend']
                if '::reset_info<' in name:kind,role='initialization','reset_info'
                elif '::iota_n<' in name:kind,role='initialization','initialize_scalars'
                elif name.startswith('__amd_rocclr_fillBuffer'):
                    assert backend=='cdls' and k['submitting_api']=='hipMemsetAsync'
                    kind='initialization';role='reset_internal_info' if k['Correlation_Id']==fills[0] else 'reset_task_workspace'
                elif 'getrf_small_kernel<' in name:kind,role='compute','fused_small_with_info_reset'
                elif '::getrf_kernel<' in name:kind,role='compute','fused_tiled_LU'
                elif 'getf2_npvt_small_kernel<' in name:kind,role='compute','panel_GETF2'
                elif 'unit_forward_substitution_kernel<' in name:kind,role='compute','internal_TRSM'
                elif name.startswith('Cijk_'):kind,role='compute','internal_GEMM'
                else:raise AssertionError('Unclassified kernel '+name)
                k.update(submission_order=position,execution_start_order=execution[id(k)],kind=kind,role=role)
                attribution.append(k)
            signature=[(k['Kernel_Name'],k['kind'],k['role']) for k in ks];signatures.append(signature)
            kinds=collections.Counter(k['kind'] for k in ks);roles=collections.Counter(k['role'] for k in ks)
            counts.append(dict(backend=ks[0]['backend'],n=ks[0]['n'],lda=ks[0]['lda'],call_id=call,
                               total=len(ks),initialization=kinds['initialization'],compute=kinds['compute'],
                               roi_marker=ks[0]['marker']))
            for role,c in roles.items():categories.append(dict(backend=ks[0]['backend'],n=ks[0]['n'],lda=ks[0]['lda'],call_id=call,role=role,count=c))
        assert signatures[0]==signatures[1]==signatures[2]
        sequences[directory.name]=[s[2] for s in signatures[0]]
    write_csv(out/'kernel-attribution.csv',attribution);write_csv(out/'kernel-counts.csv',counts)
    write_csv(out/'kernel-categories.csv',categories)
    (out/'kernel-sequences.json').write_text(json.dumps(sequences,indent=2)+'\n')
    trial_stats=[];summaries=[]
    for inp in inputs:
        n,lda=inp['n'],inp['lda'];s=next(x for x in selection['selected'] if x['n']==n and x['lda']==lda)
        subset=[x for x in samples if int(x['n'])==n and int(x['lda'])==lda]
        for backend in ('cdls','rocsolver'):
            key='cdls' if backend=='cdls' else 'rocsolver';wallkey='cdls_wall_ms' if backend=='cdls' else 'roc_wall_ms'
            v=[float(x[key+'_ms'])*1000 for x in subset];assert len(v)==90
            trace_rows=[x for x in counts if x['backend']==backend and x['n']==n and x['lda']==lda]
            assert len(trace_rows)==3 and len({(x['total'],x['initialization'],x['compute']) for x in trace_rows})==1
            for trial in range(1,4):
                tv=[float(x[key+'_ms'])*1000 for x in subset if x['trial']==trial];assert len(tv)==30
                trial_stats.append(dict(backend=backend,n=n,lda=lda,trial=trial,**stats(tv)))
            vr=[x for x in validation if x['backend']==backend and int(x['n'])==n]
            assert len(vr)==3
            summaries.append(dict(backend=backend,ROCm='7.2.4',rocSOLVER='3.32.0.dabb6df2b9',rocBLAS='5.2.0.dabb6df2b9',
                entry='cdlsDgetrf' if backend=='cdls' else 'rocsolver_dgetrf_npvt',precision='FP64',pivot_mode='no pivot',
                implementation_revision=environment['cdls_head'] if backend=='cdls' else 'dabb6df2b9',
                runner_sha256=environment['binary_sha256']['replay'],
                library_sha256=environment['binary_sha256']['libcdls_rocm.so'] if backend=='cdls' else next(v for k,v in environment['resolved_library_sha256'].items() if '/librocsolver.so' in k),
                pastix_revision=environment['pastix_head']+' plus archived working tree and audit patch',
                input_source=inp['filename'],input_sha256=inp['sha256'],cblk=inp['cblk'],sparse_call_id=inp['call_id'],
                role=s['role'],n=n,lda=lda,real_occurrences=s['count'],exact_pair_call_fraction=s['call_fraction'],
                total_kernels=trace_rows[0]['total'],initialization_kernels=trace_rows[0]['initialization'],compute_kernels=trace_rows[0]['compute'],
                **stats(v),wall_median_us=statistics.median(float(x[wallkey])*1000 for x in subset),validation='PASS',
                info_source=vr[0]['info_source'],residual_relative_max=max(float(x['residual_relative']) for x in vr),padding='bitwise unchanged'))
    write_csv(out/'summary.csv',summaries);write_csv(out/'trial-statistics.csv',trial_stats)
    (out/'verification.json').write_text(json.dumps(dict(status='PASS',paired_samples=270,timed_calls=540,
        validation_records=18,profiled_calls=len(counts),attributed_kernels=len(attribution),
        all_kernels_correlated_to_unique_ROI=True,three_repeats_same_structure=True,
        distinct_cblks=len(workload),sum_diagonal_columns=sum(int(r['n']) for r in workload),
        identical_workload_geometry_on_distribution_and_capture=True,device_versions=list(versions)),indent=2)+'\n')
    for n in [x['n'] for x in inputs]:
        r={x['backend']:x for x in summaries if x['n']==n}
        print(n,{b:(x['total_kernels'],x['initialization_kernels'],x['compute_kernels'],x['median_us']) for b,x in r.items()},'speedup',r['rocsolver']['median_us']/r['cdls']['median_us'])

if __name__=='__main__':main(sys.argv[1])
