#!/usr/bin/env python3
"""Measure CDLS + 1D update1 on the requested MI300A matrix suite."""
import argparse
import csv
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import statistics
import subprocess
import time

MATRICES = ["CoupCons3D", "dielFilterV3real", "inline_1", "audikw_1", "nd6k",
            "xenon2", "radiation", "dgreen", "pwtk", "cage12"]
BACKEND = "hip-hipmm-cdls-getrf"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


def number(text, label):
    hits = re.findall(re.escape(label) + r"\s*(\S+)", text)
    try:
        value = float(hits[-1])
        return value if math.isfinite(value) else None
    except (IndexError, ValueError):
        return None


def parse(text, rc, reason):
    measured = re.findall(r"(?m)^\s*Time to factorize\s+(\S+)\s+s\s+\(\s*(\S+)\s+([KMGT]?)Flop/s\s*\)", text)
    rate = float(measured[-1][1]) * {"": 1e-9, "K": 1e-6, "M": 1e-3, "G": 1, "T": 1e3}[measured[-1][2]] if measured else None
    factor = number(text, "PASTIX_TIMING,pastix_numfact,")
    residual = number(text, "max(|| b_i - A x_i ||_2 / || b_i ||_2)")
    backward = re.findall(r"max\(\|\| b_i - A x_i \|\|_1 /.*?\)\)\s+(\S+) \((SUCCESS|FAILED)\)", text)
    forward = re.findall(r"max\(\|\| x0_i - x_i \|\|_oo /.*?\)\s+(\S+) \((SUCCESS|FAILED)\)", text)
    iterations = number(text, "PASTIX_REFINEMENT_ITERATIONS,")
    backend_ok = bool(re.search(r"(?m)^PaStiX FP64 Numfact device backend: " + re.escape(BACKEND) + r"$", text))
    layout_ok = "PaStiX FP64 device layout: 1D" in text
    solve = number(text, "PASTIX_TIMING,pastix_solve,")
    residual_ok = residual is not None and 0 <= residual <= 1e-12
    valid = (reason is None and rc in (0, 255) and backend_ok and layout_ok
             and factor is not None and factor > 0 and rate is not None and math.isfinite(rate)
             and solve is not None and solve > 0 and residual_ok
             and bool(backward) and backward[-1][1] == "SUCCESS"
             and "PaStiX numerical factorization failed:" not in text
             and "PaStiX solve/refine failed:" not in text)
    status = reason or "failed"
    if valid:
        status = "full_check_pass" if rc == 0 and "(FAILED)" not in text else "forward_error_only"
        if status == "forward_error_only" and not (forward and forward[-1][1] == "FAILED"):
            valid, status = False, "other_check_failure"
    return dict(status=status, performance_valid=bool(valid), factorize_s=factor,
                gflops=rate, residual=residual, backward_check=backward[-1][1] if backward else "",
                forward_check=forward[-1][1] if forward else "", refine_iterations=iterations,
                no_refine_pass=bool(valid and iterations == 0), backend_ok=backend_ok,
                solution_error=number(text, "max(|| x0_i - x_i ||_oo)"),
                solve_s=solve, refine_s=number(text, "PASTIX_TIMING,pastix_refine,"))


def report(out, meta, records):
    rows = []
    for name in MATRICES:
        group = [r for r in records if r["matrix"] == name]
        valid = [r for r in group if str(r["performance_valid"]) == "True"]
        times = [float(r["factorize_s"]) for r in valid]
        rows.append(dict(matrix=name, completed=len(group), valid=len(valid), planned=meta["repeats"],
                         median_factorize_s=statistics.median(times) if times else "",
                         min_factorize_s=min(times) if times else "", max_factorize_s=max(times) if times else "",
                         median_gflops=statistics.median(float(r["gflops"]) for r in valid) if valid else "",
                         max_residual=max(float(r["residual"]) for r in valid) if valid else "",
                         no_refine_pass=sum(str(r["no_refine_pass"]) == "True" for r in valid),
                         full_check_pass=sum(r["status"] == "full_check_pass" for r in valid),
                         statuses=",".join(sorted({r["status"] for r in group}))))
    for filename, data in [("samples.csv", records), ("statistics.csv", rows)]:
        if not data:
            continue
        temp = out / (filename + ".tmp")
        with temp.open("w", newline="") as f:
            w = csv.DictWriter(f, list(data[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(data)
        temp.replace(out / filename)
    meta.update(completed=len(records), valid=sum(str(r["performance_valid"]) == "True" for r in records), updated_utc=utc())
    save_json(out / "manifest.json", meta)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--timeout", type=int, default=1200)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    root, out = Path(__file__).resolve().parents[1], a.output.resolve()
    if a.repeats < 1 or a.timeout < 1 or not Path("/dev/kfd").exists():
        p.error("positive repeats/timeout and an AMD GPU node are required")
    out.mkdir(parents=True, exist_ok=a.resume)
    lock = (out / ".run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not a.resume:
        (out / "binaries").mkdir()
        (out / "logs").mkdir()
        sources = {"solver": root / "build-mi300a-core/example/simple_solve_and_refine",
                   "cdls.so": root / "build-mi300a-roc/libpastix_device_hip.so",
                   "libcdls_rocm.so": root / "build-mi300a-roc/cdls/src/rocm/libcdls_rocm.so"}
        for name, src in sources.items():
            shutil.copy2(src, out / "binaries" / name)
        print("Freezing binaries and hashing matrix inputs", flush=True)
        matrices = {}
        for name in MATRICES:
            source_dir = "matrix-repaired" if name == "audikw_1" else "matrix"
            path = root.parent / source_dir / (name + ".mtx")
            matrices[name] = dict(path=str(path), sha256=sha(path), size=path.stat().st_size,
                                  repaired_input=name == "audikw_1")
        meta = dict(start_utc=utc(), status="RUNNING", hostname=os.uname().nodename,
                    slurm_job_id=os.getenv("SLURM_JOB_ID"), backend=BACKEND, layout="1d", update=1,
                    threads=2, cpu_affinity=sorted(os.sched_getaffinity(0))[:2], pool_bytes=32 << 30,
                    repeats=a.repeats, timeout=a.timeout, residual_limit=1e-12, matrices=matrices,
                    binaries={name: dict(source=str(src), sha256=sha(out / "binaries" / name)) for name, src in sources.items()},
                    cdls_commit=subprocess.check_output(["git", "-C", str(root / "third_party/cdls"), "rev-parse", "HEAD"], text=True).strip(),
                    pastix_commit=subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
                    runner_sha256=sha(Path(__file__)))
        shutil.copy2(__file__, out / "runner.py")
        (out / "pastix-working-tree.patch").write_bytes(subprocess.check_output(["git", "-C", str(root), "diff", "HEAD"]))
        records = []
    else:
        meta = json.loads((out / "manifest.json").read_text())
        if meta["runner_sha256"] != sha(Path(__file__)) or meta["repeats"] != a.repeats or meta["timeout"] != a.timeout:
            p.error("resume requires the original runner and settings")
        for name, data in meta["binaries"].items():
            if sha(out / "binaries" / name) != data["sha256"]:
                p.error("frozen binary changed")
        for data in meta["matrices"].values():
            if sha(Path(data["path"])) != data["sha256"]:
                p.error("matrix input changed")
        records = list(csv.DictReader((out / "samples.csv").open())) if (out / "samples.csv").exists() else []
    env = dict(os.environ, PASTIX_DEVICE_BACKEND=str(out / "binaries/cdls.so"),
               PASTIX_DEVICE_POOL_BYTES=str(meta["pool_bytes"]), PASTIX_DEVICE_LAYOUT="1d",
               PASTIX_GPU_UPDATE_STRATEGY="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    env["LD_LIBRARY_PATH"] = ":".join([str(out / "binaries"),
        "/nfsapps/ubuntu-24.04/opt/rocmplus-7.2.4/openblas-v0.3.33/lib",
        "/nfsapps/ubuntu-24.04/opt/rocm-7.2.4/lib", env.get("LD_LIBRARY_PATH", "")])
    meta["environment"] = {k: env[k] for k in ("LD_LIBRARY_PATH", "PASTIX_DEVICE_BACKEND",
        "PASTIX_DEVICE_POOL_BYTES", "PASTIX_DEVICE_LAYOUT", "PASTIX_GPU_UPDATE_STRATEGY", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")}
    (out / "libraries.txt").write_text(subprocess.check_output(["ldd", str(out / "binaries/cdls.so")], env=env, text=True))
    (out / "hardware.txt").write_text(subprocess.check_output(["/nfsapps/ubuntu-24.04/opt/rocm-7.2.4/bin/rocm-smi", "--showproductname", "--showmeminfo", "vram"], text=True))
    meta["status"] = "RUNNING"
    report(out, meta, records)
    done = {(r["matrix"], int(r["trial"])) for r in records}
    def interrupt(_sig, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    try:
        for trial in range(1, a.repeats + 1):
            for name in MATRICES:
                if (name, trial) in done:
                    continue
                log = out / "logs" / f"{name}-trial{trial}.log"
                cmd = ["taskset", "-c", ",".join(map(str, meta["cpu_affinity"])), "stdbuf", "-oL", "-eL",
                       str(out / "binaries/solver"), "--mm", meta["matrices"][name]["path"],
                       "-g", "1", "-t", "2", "-s", "1", "-c", "2", "-i", "iparm_tasks2d_level", "0",
                       "-i", "iparm_factorization", "2", "-i", "iparm_gpu_update_strategy", "1"]
                job = dict(start_utc=utc(), command=cmd, environment=meta["environment"], cpu_affinity=meta["cpu_affinity"])
                save_json(log.with_suffix(".json"), job)
                print(f"RUN {len(records)+1}/{len(MATRICES)*a.repeats}: {name} trial {trial}", flush=True)
                start, reason, minimum = time.monotonic(), None, None
                with log.open("w") as f:
                    proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        while proc.poll() is None:
                            mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
                            available = int(mem["MemAvailable"].split()[0]) << 10
                            minimum = available if minimum is None else min(minimum, available)
                            if time.monotonic() - start > a.timeout:
                                reason = "timeout"
                            elif available < 3 << 30:
                                reason = "memory_pressure"
                            elif log.stat().st_size > 64 << 20:
                                reason = "log_limit"
                            if reason:
                                os.killpg(proc.pid, signal.SIGKILL)
                                break
                            time.sleep(.5)
                        rc = proc.wait()
                    except BaseException:
                        if proc.poll() is None:
                            os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                        raise
                job.update(end_utc=utc(), exit_code=rc, wall_s=time.monotonic()-start, stop_reason=reason,
                           minimum_available_host_bytes=minimum)
                save_json(log.with_suffix(".json"), job)
                row = dict(matrix=name, trial=trial, **parse(log.read_text(errors="replace"), rc, reason),
                           exit_code=rc, wall_s=job["wall_s"], log=str(log.relative_to(out)), log_sha256=sha(log))
                records.append(row)
                report(out, meta, records)
                print(f"RESULT {name}: {row['status']}; factorize={row['factorize_s']}s; residual={row['residual']}; refine={row['refine_iterations']}", flush=True)
        meta.update(status="COMPLETE", end_utc=utc())
    except KeyboardInterrupt:
        meta["status"] = "INTERRUPTED"
        report(out, meta, records)
        return 130
    report(out, meta, records)
    print(f"Statistics: {out / 'statistics.csv'}", flush=True)
    return 0 if meta["valid"] == len(MATRICES)*a.repeats else 1


if __name__ == "__main__":
    raise SystemExit(main())
