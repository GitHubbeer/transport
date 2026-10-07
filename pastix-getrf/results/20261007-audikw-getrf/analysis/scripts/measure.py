#!/usr/bin/env python3
"""Run all unprofiled pairs first, then independent, repeated ROI traces."""
import hashlib
import json
import os
from pathlib import Path
import sys
from record import run
from select_inputs import join, write_csv

base = Path(sys.argv[1]).resolve()
phase = sys.argv[2]
assert phase in ("timing", "profile")
selection = json.loads((base/"workload/selection.json").read_text())
rows = join(base/"workload/capture-core.csv",base/"workload/capture-streams.csv")
write_csv(base/"workload/capture-calls.csv",rows)
original = join(base/"workload/distribution-core.csv",base/"workload/distribution-streams.csv")
assert [[r[k] for k in ("call_id","cblk","n","lda","backend","queue")] for r in rows] == [
    [r[k] for k in ("call_id","cblk","n","lda","backend","queue")] for r in original]
inputs = []
for s in selection["selected"]:
    paths = list((base/"inputs").glob(f'*-n{s["n"]}-lda{s["lda"]}.bin'))
    assert len(paths)==1
    path = paths[0]
    meta = json.loads(path.with_suffix(".json").read_text())
    assert meta["call_id"]==s["first_call_id"]
    source = rows[meta["call_id"]]
    meta.update(cblk=int(source["cblk"]),backend=source["backend"],stream=source["stream"],
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),filename=path.name)
    inputs.append(meta)
(base/"inputs/manifest.json").write_text(json.dumps(inputs,indent=2)+"\n")
env = dict(os.environ)
for k in list(env):
    if k.startswith(("GETRF_","PASTIX_","ROCPROFILER_")) or k=="LD_PRELOAD": env.pop(k)
rocm = Path("/nfsapps/ubuntu-24.04/opt/rocm-7.2.4")
env.update(LD_LIBRARY_PATH=f"{base}/binaries:{rocm}/lib:"+env.get("LD_LIBRARY_PATH",""),
           OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
cpus = sorted(os.sched_getaffinity(0))[:2]
affinity = ["taskset","-c",",".join(map(str,cpus))]
failed = []
if phase=="timing":
    for trial in range(1,4):
        for inp in inputs[::(-1 if trial%2==0 else 1)]:
            name=f'trial{trial}-n{inp["n"]}-lda{inp["lda"]}'
            argv=affinity+[str(base/"binaries/replay"),str(base/"inputs"/inp["filename"]),str(inp["n"]),str(inp["lda"])]
            if run(base/f"timing/{name}.log",argv,env): failed.append(name)
else:
    # Never use the trace's timestamps for a speedup claim.
    for inp in inputs:
        for backend in ("cdls","rocsolver"):
            name=f'{backend}-n{inp["n"]}-lda{inp["lda"]}'
            dest=base/"traces"/name;dest.mkdir(exist_ok=False)
            argv=affinity+[str(rocm/"bin/rocprofv3"),"--kernel-trace","--hip-trace","--marker-trace",
                  "--output-format","csv","--output-directory",str(dest),"--output-file","trace","--",
                  str(base/"binaries/replay-profile"),str(base/"inputs"/inp["filename"]),str(inp["n"]),str(inp["lda"]),backend]
            if run(dest/"run.log",argv,env):failed.append(name)
print("failed",failed)
sys.exit(bool(failed))
