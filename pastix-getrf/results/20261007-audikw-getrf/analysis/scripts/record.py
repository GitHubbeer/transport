#!/usr/bin/env python3
"""Archive a command, a restricted environment, exit status and raw output."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

def run(log, argv, env=None):
    log = Path(log)
    env = dict(os.environ if env is None else env)
    metadata = {"argv": list(map(str, argv)), "cwd": os.getcwd(),
                "start_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "environment": {k:v for k,v in env.items() if k.startswith(
                    ("SLURM_", "ROCR_", "HIP_", "HSA_", "PASTIX_", "GETRF_", "ROCPROFILER_"))
                    or k in ("LD_LIBRARY_PATH", "LD_PRELOAD", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "PATH")}}
    start = time.monotonic()
    with log.open("x") as f:
        proc = subprocess.run(list(map(str, argv)), env=env, stdout=f, stderr=subprocess.STDOUT)
    metadata.update(exit_code=proc.returncode, wall_s=time.monotonic()-start,
                    end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    log.with_suffix(".command.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(log, "exit", proc.returncode, flush=True)
    return proc.returncode

if __name__ == "__main__":
    sys.exit(run(sys.argv[1], sys.argv[2:]))
