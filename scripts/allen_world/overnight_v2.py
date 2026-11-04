"""Bounded, resumable anatomical JEPA experiment queue; Linux A100 launcher.

All real-data runs are development comparisons, NOT a locked confirmatory study.
Every planned variant/result is retained, including failed and null results.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
VARIANTS = ("flat", "allocation", "proximal", "distal")
SCENARIOS = ("null", "motion", "temperature", "joint", "equal", "reverse")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def code_hashes():
    paths = (list((ROOT / "src/evowm").rglob("*.py")) + list(HERE.glob("*.py")) +
             list(HERE.glob("*.sh")) + [ROOT / "scripts/train_hjepa.sh", ROOT / "environment.yml"] +
             list((ROOT / "configs/hjepa").rglob("*.yaml")) + list((ROOT / "hjepa/h_jepa").rglob("*.py")))
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths)}


def make_jobs(variants, scenarios, seeds, folds, stages):
    jobs = []
    for stage in stages:
        for fold in folds:
            for seed in seeds:
                for scenario in (scenarios if stage == "simulation" else ["observed"]):
                    for variant in variants:
                        name = f"{stage}_{scenario}_{variant}_s{seed}_f{fold}"
                        jobs.append({"name": name, "stage": stage, "scenario": scenario,
                                     "variant": variant, "seed": seed, "fold": fold,
                                     "config": "allen_v2_" + variant})
    return jobs


def train_command(job, args, run):
    steps = args.sim_steps if job["stage"] == "simulation" else args.real_steps
    return ["bash", "scripts/train_hjepa.sh", job["config"],
            f"data.dataset.species_table={job['table']}", f"data.dataset.tree_file={args.tree}",
            f"data.dataset.holdout.fold={job['fold']}", f"seed={job['seed']}",
            f"num_workers={args.loader_workers}", "trainer.max_epochs=1", "save_every_n_epochs=1",
            f"data.dataset.samples_per_epoch={steps * 128}", "trainer.limit_val_batches=5",
            f"subdir={job['name']}"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpus", nargs="+", required=True)
    p.add_argument("--hours", type=float, default=8.0, help="Stop launching jobs at deadline; finish active jobs")
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    p.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    p.add_argument("--seeds", nargs="+", type=int, default=[13, 14])
    p.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    p.add_argument("--stages", nargs="+", choices=["simulation", "real_development"], default=["simulation", "real_development"])
    p.add_argument("--sim-steps", type=int, default=2000)
    p.add_argument("--real-steps", type=int, default=4000)
    p.add_argument("--eval-workers", type=int, default=32)
    p.add_argument("--loader-workers", type=int, default=8)
    p.add_argument("--threads-per-job", type=int, default=4)
    p.add_argument("--job-timeout-minutes", type=float, default=90, help="Timeout per training or evaluation subprocess")
    p.add_argument("--table", default="outputs/allen_world_v2/input/species_traits.csv")
    p.add_argument("--tree", default="data/processed/trees/hackett_t021.nwk")
    p.add_argument("--resume", type=Path, help="Resume exact saved plan; only GPUs/hours/resources may change")
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if not 1 <= len(a.gpus) <= 4 or len(a.gpus) != len(set(a.gpus)):
        p.error("Use one to four distinct explicitly selected idle GPU indices")
    if not 0 < a.hours < 168 or not 0 < a.job_timeout_minutes < 1440 or min(a.sim_steps, a.real_steps, a.eval_workers, a.loader_workers, a.threads_per_job) < 1:
        p.error("hours must be in (0,168); step/thread settings must be positive")
    if any(f not in range(5) for f in a.folds):
        p.error("folds must be 0..4")
    for values in (a.variants, a.scenarios, a.seeds, a.folds, a.stages):
        if len(values) != len(set(values)):
            p.error("No duplicate matrix entries")
    if a.resume:
        run = a.resume.resolve()
        if not run.is_relative_to((ROOT / "outputs").resolve()):
            p.error("resume directory must be inside this repository's outputs")
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        if manifest["code_sha256"] != code_hashes():
            raise SystemExit("Code/config changed since this run. Start a new run; do not mix implementations.")
        for rel, sha in manifest["input_sha256"].items():
            if digest(ROOT / rel) != sha:
                raise SystemExit(f"Input changed: {rel}. Start a new run.")
        for key in ("table", "tree", "sim_steps", "real_steps"):
            setattr(a, key, manifest["settings"][key])
        jobs = manifest["jobs"]
    else:
        for rel in (a.table, a.tree):
            if not (ROOT / rel).is_file():
                raise SystemExit(f"Missing {rel}; use run_overnight_v2.sh to prepare inputs first")
        run = ROOT / "outputs" / ("allen_overnight_v2_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        run.mkdir(parents=True, exist_ok=False)
        (run / "logs").mkdir()
        from simulate_v2 import generate
        jobs = make_jobs(a.variants, a.scenarios, a.seeds, a.folds, a.stages)
        sim_tables = {}
        if "simulation" in a.stages:
            for scenario in a.scenarios:
                for seed in a.seeds:
                    sim_tables[(scenario, seed)] = generate(ROOT / a.table, ROOT / a.tree,
                        run / "inputs" / f"{scenario}_s{seed}", scenario, seed).relative_to(ROOT).as_posix()
        for job in jobs:
            job["table"] = sim_tables[(job["scenario"], job["seed"])] if job["stage"] == "simulation" else a.table
        inputs = {a.table, a.tree, str(Path(a.table).with_suffix(".columns.json")).replace("\\", "/")}
        for rel in sim_tables.values():
            inputs.update([rel, Path(rel).with_suffix(".columns.json").as_posix(),
                           (Path(rel).parent / "truth.json").as_posix(), (Path(rel).parent / "true_slopes.csv").as_posix()])
        manifest = {"kind": "allen_anatomical_development_v2", "created": datetime.now().isoformat(),
                    "settings": {k: v for k, v in vars(a).items() if k != "resume"},
                    "jobs": jobs, "input_sha256": {r: digest(ROOT / r) for r in sorted(inputs)},
                    "code_sha256": code_hashes(),
                    "interpretation": "Full matrix retained. Real runs are development only; no confirmatory claims or gradient-based winner selection."}
        save_json(run / "manifest.json", manifest)
        for command, name in [(["rev-parse", "HEAD"], "git_commit.txt"), (["status", "--short"], "git_status.txt"),
                              (["diff"], "tracked_changes.patch")]:
            result = subprocess.run(["git", "-c", "safe.directory=" + ROOT.as_posix()] + command, cwd=ROOT, capture_output=True)
            (run / name).write_bytes(result.stdout + result.stderr)
        # Snapshot both tracked and new implementation files for exact auditability.
        for rel in manifest["code_sha256"]:
            dest = run / "source" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((ROOT / rel).read_bytes())
    print(f"RUN_DIRECTORY={run}\nPLANNED_JOBS={len(jobs)} GPU_SLOTS={len(a.gpus)} LAUNCH_WINDOW_HOURS={a.hours}", flush=True)
    if a.dry_run:
        print("Dry-run only: no training. Use --resume with this directory to execute the exact saved plan.")
        return
    if os.name == "nt":
        raise SystemExit("Training launcher requires the Linux server; local inspection supports --dry-run")
    # A second tmux --resume must never write the same checkpoints/status files.
    import fcntl
    run_lock = (run / ".run.lock").open("a+")
    try:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        run_lock.close()
        raise SystemExit(f"Another launcher is active in {run}; do not resume a live run")
    run_lock.seek(0)
    run_lock.truncate()
    run_lock.write(str(os.getpid()) + "\n")
    run_lock.flush()  # Descriptor remains open until main exits; stale files are harmless.
    gpu = subprocess.run(["nvidia-smi"], capture_output=True, check=True)
    (run / ("gpu_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt")).write_bytes(gpu.stdout)
    env_report = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True)
    (run / "environment.txt").write_bytes(env_report.stdout + env_report.stderr)
    state_file = run / "status.json"
    status = json.loads(state_file.read_text(encoding="utf-8")) if state_file.exists() else {}
    lock = threading.Lock()
    deadline = time.monotonic() + a.hours * 3600
    eval_futures = []

    def update(name, **values):
        with lock:
            status.setdefault(name, {}).update(values, updated=datetime.now().isoformat())
            save_json(state_file, status)

    def evaluate(job, checkpoint):
        out = run / "evaluation" / job["name"]
        update(job["name"], evaluation="running")
        env = os.environ.copy()
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[key] = "1"
        cmd = [sys.executable, "scripts/allen_world/evaluate.py", "--checkpoint", str(checkpoint),
               "--out-dir", str(out), "--threads", "1", "--all-temperature-axes"]
        with (run / "logs" / (job["name"] + "_eval.log")).open("ab") as log:
            try:
                result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=a.job_timeout_minutes * 60)
                returncode = result.returncode
            except subprocess.TimeoutExpired:
                log.write(b"\nTIMEOUT: evaluation process exceeded per-job limit\n")
                returncode = 124
        ok = returncode == 0 and (out / "pilot_results.json").is_file()
        update(job["name"], evaluation="done" if ok else "failed", evaluation_returncode=returncode)
        return ok

    queue = []
    for job in jobs:
        previous = status.get(job["name"], {})
        if previous.get("evaluation") == "done" and (run / "evaluation" / job["name"] / "pilot_results.json").is_file():
            continue
        if (previous.get("training") == "failed" or previous.get("evaluation") == "failed") and not a.retry_failed:
            continue
        queue.append(job)
    next_index = 0

    def worker(slot, eval_pool):
        nonlocal next_index
        while True:
            with lock:
                if next_index >= len(queue) or time.monotonic() >= deadline:
                    return
                job = queue[next_index]
                next_index += 1
            checkpoint = run / "training/ckpts" / job["name"] / "allen_v2_object.ckpt"
            # A checkpoint alone is not proof of successful trainer termination.
            trained = status.get(job["name"], {}).get("training") == "done" and checkpoint.is_file()
            if not trained:
                start = time.monotonic()
                update(job["name"], training="running", gpu=a.gpus[slot])
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=a.gpus[slot], HJEPA_HOME=str(run / "training"))
                for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                    env[key] = str(a.threads_per_job)
                cmd = train_command(job, a, run)
                print(f"START {job['name']} gpu={a.gpus[slot]}", flush=True)
                with (run / "logs" / (job["name"] + ".log")).open("ab") as log:
                    try:
                        result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                                timeout=a.job_timeout_minutes * 60)
                        returncode = result.returncode
                    except subprocess.TimeoutExpired:
                        log.write(b"\nTIMEOUT: training process exceeded per-job limit\n")
                        returncode = 124
                trained = returncode == 0 and checkpoint.is_file()
                elapsed = time.monotonic() - start
                update(job["name"], training="done" if trained else "failed", training_returncode=returncode,
                       training_seconds=elapsed)
                print(f"{'DONE' if trained else 'FAILED'} {job['name']} seconds={elapsed:.1f}", flush=True)
                if not trained:
                    # Fail this slot, retain other active work; do not produce a storm of identical errors.
                    return
            with lock:
                eval_futures.append(eval_pool.submit(evaluate, job, checkpoint))

    with ThreadPoolExecutor(max_workers=a.eval_workers) as eval_pool:
        with ThreadPoolExecutor(max_workers=len(a.gpus)) as gpu_pool:
            for future in as_completed([gpu_pool.submit(worker, i, eval_pool) for i in range(len(a.gpus))]):
                future.result()
        for future in as_completed(eval_futures):
            future.result()
    complete = sum(status.get(j["name"], {}).get("evaluation") == "done" for j in jobs)
    failed = [j["name"] for j in jobs if any(status.get(j["name"], {}).get(k) == "failed" for k in ("training", "evaluation"))]
    pending = [j["name"] for j in jobs if status.get(j["name"], {}).get("evaluation") != "done"]
    save_json(run / "completion.json", {"completed": complete, "planned": len(jobs), "failed": failed,
              "pending": pending, "all_done": complete == len(jobs), "launch_window_hours": a.hours,
              "note": "Deadline stops new launches; active training/evaluation may finish later."})
    summarize = subprocess.run([sys.executable, "scripts/allen_world/summarize_v2.py", str(run)], cwd=ROOT)
    print(f"FINISHED completed={complete}/{len(jobs)} failed={len(failed)} pending={len(pending)}\nDOWNLOAD={run}", flush=True)
    if failed or summarize.returncode:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
