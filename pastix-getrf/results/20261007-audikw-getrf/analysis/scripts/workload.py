#!/usr/bin/env python3
import json
import os
from pathlib import Path
import shutil
import sys
from record import run
base = Path(sys.argv[1]).resolve()
phase = sys.argv[2]
assert phase in ("distribution", "capture")
root = Path(__file__).resolve().parents[2]
rocm = Path("/nfsapps/ubuntu-24.04/opt/rocm-7.2.4")
env = dict(os.environ)
env.update(LD_LIBRARY_PATH=f"{base}/binaries:{rocm}/lib:/nfsapps/ubuntu-24.04/opt/rocmplus-7.2.4/openblas-v0.3.33/lib:"+env.get("LD_LIBRARY_PATH",""),
           LD_PRELOAD=str(base/"binaries/capture.so"), GETRF_REAL_CDLS=str(base/"binaries/libcdls_rocm.so"),
           GETRF_STREAM_CSV=str(base/f"workload/{phase}-streams.csv"),
           PASTIX_GETRF_WORKLOAD_CSV=str(base/f"workload/{phase}-core.csv"),
           PASTIX_GETRF_PARAMETERS_CSV=str(base/f"workload/{phase}-parameters.csv"),
           PASTIX_DEVICE_BACKEND=str(base/"binaries/cdls.so"),
           PASTIX_DEVICE_POOL_BYTES="34359738368", PASTIX_DEVICE_LAYOUT="1d",
           PASTIX_GPU_UPDATE_STRATEGY="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
if phase == "capture":
    env.update(GETRF_CAPTURE_SELECTION=str(base/"workload/selection.txt"), GETRF_CAPTURE_DIR=str(base/"inputs"))
cpus = sorted(os.sched_getaffinity(0))
# Two distinct physical cores within the allocation (same choice in both runs).
cpus = [cpus[0], cpus[1]]
argv = ["taskset","-c",",".join(map(str,cpus)), "stdbuf","-oL","-eL", str(base/"binaries/solver"),
        "--mm", str(root.parent/"matrix-repaired/audikw_1.mtx"), "-g","1","-t","2","-s","1","-c","2",
        "-i","iparm_tasks2d_level","0","-i","iparm_factorization","2",
        "-i","iparm_min_blocksize","160","-i","iparm_max_blocksize","320",
        "-i","iparm_ordering","0","-i","iparm_gpu_update_strategy","1"]
existing = set(root.glob("pastix-*/idparam_*.csv"))
rc = run(base/f"workload/{phase}.log",argv,env)
for p in set(root.glob("pastix-*/idparam_*.csv"))-existing:
    shutil.copyfile(p,base/f"workload/{phase}-parameters.csv")
# Forward solution checks can return 255 even with a successful backward residual;
# retain the raw status and never silently turn it into a PASS.
sys.exit(rc)
