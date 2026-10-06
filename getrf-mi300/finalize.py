#!/usr/bin/env python3
"""Generate final tables, figures, report and a content manifest after the sweep."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('results',type=Path);a=p.parse_args();out=a.results.resolve()
    meta=json.loads((out/'metadata.json').read_text());assert meta['status']!='RUNNING'
    subprocess.run([sys.executable,str(HERE/'summarize.py'),str(out)],check=True)
    subprocess.run([sys.executable,str(HERE/'verify_trace.py'),str(out)],check=True)
    subprocess.run([sys.executable,str(HERE/'report.py'),str(out)],check=True)
    python=ROOT/'build/getrf-perf-env/bin/python'
    subprocess.run([str(python),str(HERE/'plot.py'),str(out/'summary.csv'),str(out/'figures')],check=True)
    analysis=out/'analysis-source';analysis.mkdir(exist_ok=True)
    for pth in [HERE/'summarize.py',HERE/'plot.py',HERE/'report.py',HERE/'profile.py',HERE/'trace.cpp',HERE/'stress.py',HERE/'verify_trace.py',HERE/'README.md',HERE/'finalize.py']:
        (analysis/pth.name).write_bytes(pth.read_bytes())
    versions=subprocess.check_output([str(python),'-c','import numpy,matplotlib; print("numpy="+numpy.__version__+" matplotlib="+matplotlib.__version__)'],text=True)
    (out/'plot-environment.txt').write_text(versions)
    manifest={str(f.relative_to(out)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(out.rglob('*')) if f.is_file() and f.name!='MANIFEST.json'}
    (out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Finalized',out,'with',len(manifest),'artifacts')
if __name__=='__main__':main()
