#!/usr/bin/env python3
"""Join exact calls and select sizes before measuring either backend.

Rules: modal (n,lda); closest observed pair to midpoint of modal/max n;
maximum n. Within a tied n choose the most frequent lda, then smallest lda.
These three exact pairs are the direct coverage, not the whole size interval.
"""
import collections
import csv
import json
from pathlib import Path
import sys

def join(core, streams):
    core = list(csv.DictReader(Path(core).open()))
    streams = list(csv.DictReader(Path(streams).open()))
    assert len(core) == len(streams) and core
    for c,s in zip(core,streams):
        assert all(c[k] == s[k] for k in ("call_id", "n", "lda"))
        c["stream"] = s["stream"]
    return core

def write_csv(path, rows):
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def main(base):
    base = Path(base)
    rows = join(base/"workload/distribution-core.csv", base/"workload/distribution-streams.csv")
    write_csv(base/"workload/calls.csv", rows)
    count = collections.Counter((int(r["n"]),int(r["lda"])) for r in rows)
    frequency = [dict(n=n, lda=ld, count=c, call_fraction=c/len(rows)) for (n,ld),c in sorted(count.items())]
    write_csv(base/"workload/frequency.csv", frequency)
    modal = min(count, key=lambda p:(-count[p],p))
    largest = min(count, key=lambda p:(-p[0],-count[p],p[1]))
    target = (modal[0]+largest[0])/2
    medium = min((p for p in count if p not in (modal,largest)), key=lambda p:(abs(p[0]-target),-count[p],p[1]))
    selection = [dict(role=role,n=p[0],lda=p[1],count=count[p],call_fraction=count[p]/len(rows),
                      first_call_id=int(next(r["call_id"] for r in rows if (int(r["n"]),int(r["lda"]))==p)))
                 for role,p in zip(("high_frequency","medium","large"),(modal,medium,largest))]
    document = dict(rule=__doc__, total_calls=len(rows),unique_pairs=len(count),selected=selection,
                    exact_pair_coverage=sum(x["count"] for x in selection)/len(rows),
                    unweighted_call_coverage_only=True)
    (base/"workload/selection.json").write_text(json.dumps(document,indent=2)+"\n")
    (base/"workload/selection.txt").write_text("".join(f'{x["n"]} {x["lda"]}\n' for x in selection))
    print(json.dumps(document,indent=2))

if __name__ == "__main__": main(sys.argv[1])
