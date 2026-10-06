#!/usr/bin/env python3
"""Standalone PDF/PNG figures derived only from verified summary CSV."""
import argparse,csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('summary',type=Path);p.add_argument('output',type=Path);a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
    rows=list(csv.DictReader(a.summary.open()))
    strings={'phase','method','reference','matrix','validated'}
    for r in rows:
        for k in r.keys()-strings:r[k]=float(r[k])
    valid=[r for r in rows if r['validated']=='True']; group=lambda phase:sorted([r for r in valid if r['phase']==phase],key=lambda r:r['n'])
    blue='#0072B2';orange='#D55E00';gray='#444444'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':200,'pdf.fonttype':42})
    pdf=PdfPages(a.output/'performance.pdf')
    def save(fig,name):
        fig.savefig(a.output/(name+'.png'),bbox_inches='tight');fig.savefig(a.output/(name+'.pdf'),bbox_inches='tight');pdf.savefig(fig,bbox_inches='tight');plt.close(fig)
    def speed(ax,data,color=blue,label='rocSOLVER NPVT / CDLS latency'):
        x=[r['n'] for r in data];ax.plot(x,[r['speedup'] for r in data],'o-',color=color,label=label,markersize=3)
        ax.fill_between(x,[r['trial_speedup_min'] for r in data],[r['trial_speedup_max'] for r in data],color=color,alpha=.15)
        ax.axhline(1,color=gray,ls=':');ax.grid(alpha=.2)
    size=group('size');base={int(r['n']):r for r in size};x=[r['n'] for r in size]
    fig,axs=plt.subplots(3,1,figsize=(9,10),sharex=True)
    axs[0].loglog(x,[r['cdls_ms'] for r in size],'o-',color=blue,label='CDLS automatic fused/hybrid')
    axs[0].loglog(x,[r['roc_ms'] for r in size],'s--',color=orange,label='rocSOLVER dgetrf_npvt')
    axs[1].semilogx(x,[r['cdls_gflops']/1000 for r in size],'o-',color=blue,label='CDLS')
    axs[1].semilogx(x,[r['roc_gflops']/1000 for r in size],'s--',color=orange,label='rocSOLVER NPVT')
    speed(axs[2],size);axs[2].set_xscale('log')
    for ax in axs:ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=9)
    axs[0].set_ylabel('Factorization latency (ms)');axs[1].set_ylabel('Nominal FP64 throughput (TFLOP/s)');axs[2].set_ylabel('rocSOLVER latency / CDLS latency');axs[2].set_xlabel('Square dimension n')
    axs[2].set_xticks([1,16,64,256,1024,4096,16384,32768],['1','16','64','256','1024','4096','16384','32768'])
    fig.suptitle('MI300A / ROCm 6.4.2: FP64 GETRF, diagonally dominant dense input\nNo pivoting; 3 trials; restoration excluded; shaded band = trial range',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.94));save(fig,'overview')
    fig,axs=plt.subplots(2,3,figsize=(12,7))
    for ax,(lo,hi,title) in zip(axs.flat,[(25,70,'Single-tile transition'),(120,265,'Tile-count transitions'),(740,1050,'Lookahead activation'),(2000,2600,'Wave-count transition'),(3050,4150,'Fused / hybrid transition'),(8100,8250,'Large odd dimensions')]):
        data=sorted([r for r in valid if r['phase'] in ('boundary','size') and lo<=r['n']<=hi],key=lambda r:r['n']);speed(ax,data);ax.set_title(title);ax.set_xlabel('n');ax.set_ylabel('Speedup')
    fig.suptitle('Tile / scheduling / dispatch boundaries');fig.tight_layout(rect=(0,0,1,.94));save(fig,'boundaries')
    fig,axs=plt.subplots(1,2,figsize=(11,4.8))
    for pad,color in [(1,blue),(17,orange),(128,'#009E73')]:
        data=[r for r in group('padding') if r['lda']-r['n']==pad];xx=[r['n'] for r in data]
        axs[0].semilogx(xx,[r['cdls_ms']/base[int(r['n'])]['cdls_ms'] for r in data],'o-',color=color,label=f'CDLS lda=n+{pad}')
        axs[0].semilogx(xx,[r['roc_ms']/base[int(r['n'])]['roc_ms'] for r in data],'s--',color=color,alpha=.55,label=f'rocSOLVER +{pad}')
        speed(axs[1],data,color,label=f'lda=n+{pad}')
    for ax in axs:ax.set_xscale('log');ax.grid(alpha=.2);ax.legend(fontsize=8,frameon=False);ax.set_xlabel('n')
    axs[0].axhline(1,color=gray,ls=':');axs[0].set_ylabel('Padded / packed latency');axs[1].set_ylabel('Speedup over NPVT')
    fig.suptitle('Leading-dimension sensitivity');fig.tight_layout();save(fig,'padding')
    fig,axs=plt.subplots(1,2,figsize=(11,4.8))
    for matrix,mark in [('signed','s'),('column_scaled','^'),('identity','D'),('known_lu','x')]:
        data=[r for r in group('input') if r['matrix']==matrix];axs[0].semilogx([r['n'] for r in data],[r['speedup'] for r in data],mark+'-',label=matrix)
    speed(axs[1],group('offset'),blue,'8-byte origin offset');speed(axs[1],group('stride'),orange,'lda=2n')
    for ax in axs:ax.set_xscale('log');ax.axhline(1,color=gray,ls=':');ax.grid(alpha=.2);ax.legend(fontsize=8,frameon=False);ax.set_xlabel('n');ax.set_ylabel('Speedup over NPVT')
    axs[0].set_title('Input families with passing residuals');axs[1].set_title('Origin / stride sensitivity');fig.tight_layout();save(fig,'inputs_alignment')
    fig,axs=plt.subplots(1,2,figsize=(11,4.8))
    for field,color,label in [('cdls_submit_ms',blue,'CDLS submission'),('roc_submit_ms',orange,'rocSOLVER submission')]:axs[0].loglog(x,[r[field] for r in size],'o-',color=color,label=label)
    axs[0].loglog(x,[r['cdls_wall_ms'] for r in size],':',color=blue,label='CDLS wall');axs[0].loglog(x,[r['roc_wall_ms'] for r in size],':',color=orange,label='rocSOLVER wall')
    axs[1].loglog(x,[r['workspace_bytes']/1024 for r in size],'o-',color=blue,label='CDLS explicit workspace')
    axs[1].loglog([r['n'] for r in size if r['roc_workspace_bytes']>0],[r['roc_workspace_bytes']/1024 for r in size if r['roc_workspace_bytes']>0],'s--',color=orange,label='rocSOLVER queried workspace')
    for ax in axs:ax.grid(alpha=.2);ax.legend(fontsize=8,frameon=False);ax.set_xlabel('n')
    axs[0].set_ylabel('Time (ms)');axs[1].set_ylabel('Workspace (KiB)');fig.suptitle('Host submission and workspace requirements\nCDLS BLAS-handle allocations are additional to its explicit workspace')
    fig.tight_layout(rect=(0,0,1,.90));save(fig,'submission_memory')
    fig,axs=plt.subplots(1,2,figsize=(11,4.8))
    static=group('static');xx=[r['n'] for r in static];axs[0].semilogx(xx,[r['cdls_ms']/base[int(r['n'])]['cdls_ms'] for r in static],'o-',color=blue,label='Static criterion=0 / ordinary CDLS')
    axs[0].axhline(1,color=gray,ls=':');axs[0].set_ylabel('CDLS latency ratio');axs[0].set_title('Static-pivot path overhead')
    speed(axs[1],group('pivot'),orange,'CDLS no-pivot / rocSOLVER with pivoting');axs[1].set_xscale('log');axs[1].set_ylabel('rocSOLVER pivoted latency / CDLS latency');axs[1].set_title('Supplementary: different pivoting semantics')
    for ax in axs:ax.grid(alpha=.2);ax.legend(fontsize=8,frameon=False);ax.set_xlabel('n')
    fig.tight_layout();save(fig,'static_pivoted_reference');pdf.close();print(a.output/'performance.pdf')
if __name__=='__main__':main()
