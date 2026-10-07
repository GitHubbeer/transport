#!/usr/bin/env python3
"""Freeze scripts, write checksums and create a self-contained evidence archive."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

base=Path(sys.argv[1]).resolve()
replace=len(sys.argv)>2 and sys.argv[2]=='--replace'
root=Path(__file__).resolve().parents[2]
source=Path(__file__).resolve().parent
dest=base/'analysis/scripts';dest.mkdir(exist_ok=replace)
for p in source.rglob('*'):
    if p.is_file() and '__pycache__' not in p.parts:
        target=dest/p.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
shutil.copyfile(source/'reproduce.txt',base/'analysis/reproduce.txt')
shutil.copyfile(root/'device/getrf_workload_audit.hpp',base/'environment/getrf_workload_audit.hpp')
(base/'environment/final-working-tree.patch').write_bytes(subprocess.check_output(['git','diff','HEAD'],cwd=root))
verification=json.loads((base/'analysis/verification.json').read_text());assert verification['status']=='PASS'
checksums=[]
for p in sorted(base.rglob('*')):
    if p.is_file() and p.name!='SHA256SUMS':
        digest=hashlib.sha256(p.read_bytes()).hexdigest();checksums.append(f'{digest}  {p.relative_to(base)}\n')
(base/'SHA256SUMS').write_text(''.join(checksums))
archive=base.with_suffix('.tar.gz')
assert replace or not archive.exists()
with tarfile.open(archive,'w:gz') as tar:tar.add(base,arcname=base.name)
digest=hashlib.sha256(archive.read_bytes()).hexdigest()
archive.with_name(archive.name+'.sha256').write_text(f'{digest}  {archive.name}\n')
print(archive,archive.stat().st_size,digest)
