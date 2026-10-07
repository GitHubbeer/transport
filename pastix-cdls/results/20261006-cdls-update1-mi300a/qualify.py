"""Apply the user's residual-pass + zero-refinement acceptance rule to raw logs."""
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

out = Path(__file__).resolve().parent
meta = json.loads((out / "manifest.json").read_text())
assert meta["status"] == "COMPLETE", "Finish the campaign before qualification"
rows = list(csv.DictReader((out / "samples.csv").open()))
assert len(rows) == len(meta["matrices"]) * meta["repeats"]
assert hashlib.sha256((out / "runner.py").read_bytes()).hexdigest() == meta["runner_sha256"]
for name, data in meta["binaries"].items():
    assert hashlib.sha256((out / "binaries" / name).read_bytes()).hexdigest() == data["sha256"]

accepted = []
for row in rows:
    log = out / row["log"]
    assert hashlib.sha256(log.read_bytes()).hexdigest() == row["log_sha256"]
    text = log.read_text()
    job = json.loads(log.with_suffix(".json").read_text())
    assert job["environment"]["PASTIX_DEVICE_LAYOUT"] == "1d"
    assert job["environment"]["PASTIX_GPU_UPDATE_STRATEGY"] == "1"
    assert job["environment"]["PASTIX_DEVICE_POOL_BYTES"] == str(meta["pool_bytes"])
    assert job["exit_code"] == int(row["exit_code"])
    assert f"PaStiX FP64 Numfact device backend: {meta['backend']}\n" in text
    factor = float(re.findall(r"(?m)^PASTIX_TIMING,pastix_numfact,(\S+)$", text)[-1])
    measured = re.findall(r"(?m)^\s*Time to factorize\s+(\S+)\s+s\s+\(\s*(\S+)\s+([KMGT]?)Flop/s\s*\)", text)[-1]
    assert math.isclose(factor, float(measured[0]), rel_tol=1e-6)
    assert factor == float(row["factorize_s"])
    residual = float(re.findall(r"max\(\|\| b_i - A x_i \|\|_2 / \|\| b_i \|\|_2\)\s+(\S+)", text)[-1])
    iterations = int(re.findall(r"(?m)^PASTIX_REFINEMENT_ITERATIONS,(\d+)$", text)[-1])
    assert residual == float(row["residual"])
    assert iterations == float(row["refine_iterations"])
    if (row["performance_valid"] == "True" and 0 <= residual <= meta["residual_limit"]
            and iterations == 0 and row["backward_check"] == "SUCCESS"):
        accepted.append(row)

summary = []
for matrix in meta["matrices"]:
    group = [r for r in accepted if r["matrix"] == matrix]
    times = [float(r["factorize_s"]) for r in group]
    summary.append(dict(matrix=matrix, accepted=len(group), planned=meta["repeats"],
                        median_factorize_s=statistics.median(times) if times else "",
                        min_factorize_s=min(times) if times else "", max_factorize_s=max(times) if times else "",
                        median_gflops=statistics.median(float(r["gflops"]) for r in group) if group else "",
                        max_residual=max(float(r["residual"]) for r in group) if group else "",
                        refinement_iterations=0 if group else "",
                        forward_check=",".join(sorted({r["forward_check"] for r in group}))))
with (out / "accepted_statistics.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, list(summary[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(summary)
qualification = dict(rule="residual <= 1e-12, backward check SUCCESS, refinement iterations == 0; forward-check failures accepted",
                     completed=len(rows), accepted=len(accepted), all_requested_trials_accepted=len(accepted) == len(rows),
                     source="samples.csv and independently checked raw logs", summary="accepted_statistics.csv")
(out / "acceptance.json").write_text(json.dumps(qualification, indent=2) + "\n")
print(json.dumps(qualification, indent=2))
for row in summary:
    print(row)
