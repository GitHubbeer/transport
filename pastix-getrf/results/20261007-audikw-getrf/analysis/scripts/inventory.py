#!/usr/bin/env python3
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from record import run

root = Path(__file__).resolve().parents[2]
base = Path(sys.argv[1]).resolve()
dest = base/"environment"
rocm = Path("/nfsapps/ubuntu-24.04/opt/rocm-7.2.4")
env = dict(os.environ)
env["LD_LIBRARY_PATH"] = f'{base}/binaries:{rocm}/lib:'+env.get("LD_LIBRARY_PATH","")
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk:=f.read(8*1024*1024):h.update(chunk)
    return h.hexdigest()
for name,argv in [
    ("compiler",[rocm/"bin/hipcc","--version"]),
    ("gpu-before",[rocm/"bin/rocm-smi","--showallinfo"]),
    ("rocminfo",[rocm/"bin/rocminfo"]),
    ("kernel",["uname","-a"]),
    ("slurm",["scontrol","show","job",os.environ["SLURM_JOB_ID"]]),
    ("solver-libraries",["ldd",base/"binaries/solver"]),
    ("plugin-libraries",["ldd",base/"binaries/cdls.so"]),
    ("reference-plugin-libraries",["ldd",base/"binaries/rocsolver.so"]),
    ("replay-libraries",["ldd",base/"binaries/replay"]),
    ("profile-libraries",["ldd",base/"binaries/replay-profile"]),
    ("cpu",["lscpu","-e=CPU,CORE,SOCKET,NODE"]),
]: run(dest/f"{name}.txt",argv,env)
for filename in ("rocsolver/rocsolver-version.h","rocblas/internal/rocblas-version.h"):
    shutil.copyfile(rocm/"include"/filename,dest/Path(filename).name)
for build in ("build-mi300a-core","build-mi300a-roc","build-mi300a-ck"):
    shutil.copyfile(root/build/"CMakeCache.txt",dest/f'{build}-CMakeCache.txt')
sources={}
for subdir in ("src","include"):
    for path in (root/"third_party/cdls"/subdir).rglob("*"):
        if path.is_file():sources[str(path.relative_to(root))]=sha(path)
manifest = dict(rocm_root=str(rocm),rocm_version=(rocm/".info/version").read_text().strip(),
                pastix_head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                cdls_head=subprocess.check_output(["git","-C","third_party/cdls","rev-parse","HEAD"],text=True).strip(),
                ck_head=subprocess.check_output(["git","-C","third_party/ck_getrf","rev-parse","HEAD"],text=True).strip(),
                binary_sha256={p.name:sha(p) for p in (base/"binaries").iterdir() if p.is_file()},
                cdls_source_sha256=sources, cpu_affinity=sorted(os.sched_getaffinity(0)),
                matrix=dict(path=str(root.parent/"matrix-repaired/audikw_1.mtx"),
                            sha256=sha(root.parent/"matrix-repaired/audikw_1.mtx")))
libraries={}
for path in dest.glob('*libraries.txt'):
    for line in path.read_text().splitlines():
        fields=line.split()
        for field in fields:
            if field.startswith('/') and Path(field).is_file():libraries[field]=sha(field)
manifest['resolved_library_sha256']=libraries
(dest/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
paths = subprocess.check_output(["git","ls-files"],text=True).splitlines()
with (dest/"source-files.txt").open("w") as f:
    f.write(''.join(p+'\n' for p in paths if (root/p).is_file()))
subprocess.run(["tar","-czf",str(dest/"pastix-source.tar.gz"),"-T",str(dest/"source-files.txt")],check=True)
subprocess.run(["tar","-czf",str(dest/"cdls-source.tar.gz"),"third_party/cdls/src","third_party/cdls/include","third_party/cdls/CMakeLists.txt"],check=True)
(dest/"working-tree.patch").write_bytes(subprocess.check_output(["git","diff","HEAD"]))
