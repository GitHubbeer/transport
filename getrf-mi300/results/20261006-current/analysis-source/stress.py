#!/usr/bin/env python3
"""After the timing sweep, diagnose random-input residuals with pivoted LU."""
import argparse, csv, json, math, os, re, subprocess
from pathlib import Path

def main():
    p = argparse.ArgumentParser(); p.add_argument('results', type=Path)
    out = p.parse_args().results.resolve()
    assert json.loads((out/'metadata.json').read_text())['status'] in ('PASS','FAIL')
    dest = out/'pivot-stress'; dest.mkdir(exist_ok=False)
    env = os.environ.copy(); env['LD_LIBRARY_PATH'] = str(out)+':'+env.get('LD_LIBRARY_PATH','')
    rows, commands = [], []
    for n in [64,512,2048]:
        argv = [str(out/'benchmark'),'--n',str(n),'--matrix','random',
                '--reference','pivot','--validate-only']
        log = dest/f'n{n}.log'
        with log.open('w') as f:
            result = subprocess.run(argv,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=300)
        commands.append(dict(n=n,argv=argv,returncode=result.returncode,log=log.name))
        line = next(x for x in log.read_text().splitlines() if x.startswith('validate '))
        fields = dict(re.findall(r'(\w+)=([^\s]+)',line))
        cr, rr = map(float,fields['residual_scaled'].split('/'))
        rows.append(dict(n=n,cdls_residual_scaled=cr,roc_pivot_residual_scaled=rr,
                         cdls_residual_pass=math.isfinite(cr) and cr<=100,
                         roc_pivot_residual_pass=math.isfinite(rr) and rr<=100,
                         info=fields['info'],joint_validation=line.endswith(' PASS')))
        print(line,flush=True)
    (dest/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    with (dest/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

if __name__=='__main__':main()
