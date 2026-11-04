"""Parallel simulation-only training on explicitly selected GPUs (maximum four).

Uses the unchanged upstream training entry through scripts/train_hjepa.sh.
No real-data training or automatic scientific approval is performed here.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from simulate import generate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpus", nargs="+", required=True, help="Explicit idle GPU indices, at most four")
    p.add_argument("--levels", nargs="+", choices=["l1", "l2"], default=["l1", "l2"])
    p.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    p.add_argument("--seeds", nargs="+", type=int, default=[13, 14])
    p.add_argument("--scenarios", nargs="+", choices=["null", "temperature", "motion", "joint"],
                   default=["null", "temperature", "motion", "joint"])
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--loader-workers", type=int, default=8)
    p.add_argument("--threads-per-job", type=int, default=4)
    p.add_argument("--eval-workers", type=int, default=32)
    p.add_argument("--tree", default="data/processed/trees/hackett_t021.nwk")
    p.add_argument("--table", default="outputs/allen_world_v1/input/species_traits.csv")
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if len(a.gpus) > 4 or len(set(a.gpus)) != len(a.gpus):
        p.error("Use one to four distinct GPU indices")
    if any(f not in range(5) for f in a.folds) or not a.folds or a.steps < 1:
        p.error("folds must be in 0..4 and steps positive")
    if min(a.loader_workers, a.threads_per_job, a.eval_workers) < 1:
        p.error("worker/thread settings must be positive")
    if any(len(v) != len(set(v)) for v in (a.folds, a.levels, a.seeds, a.scenarios)):
        p.error("Duplicate jobs requested")
    run = (a.run_dir or ROOT / "outputs" / ("allen_world_sim_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))).resolve()
    if not run.is_relative_to((ROOT / "outputs").resolve()):
        p.error("run-dir must be inside this repository's outputs directory")
    run.mkdir(parents=True, exist_ok=False)
    (run / "logs").mkdir()
    jobs = []
    for scenario in a.scenarios:
        for seed in a.seeds:
            table = generate(ROOT / a.table, ROOT / a.tree, run / "inputs" / f"{scenario}_s{seed}", scenario, seed)
            for level in a.levels:
                for fold in a.folds:
                    name = f"{scenario}_{level}_s{seed}_f{fold}"
                    jobs.append({"name": name, "config": "allen_world_v1_" + level,
                                 "table": table.relative_to(ROOT).as_posix(), "fold": fold, "seed": seed})
    manifest = {"kind": "simulation_only", "run_dir": str(run), "args": vars(a) | {"run_dir": str(run)},
                "input_sha256": hashlib.sha256((ROOT / a.table).read_bytes()).hexdigest(), "jobs": jobs}
    # Capture the complete code diff as well as tracked commit; no repository mutation.
    git_prefix = ["git", "-c", "safe.directory=" + ROOT.as_posix()]
    for tail, dest in [(["rev-parse", "HEAD"], "git_commit.txt"), (["status", "--short"], "git_status.txt"),
                       (["diff"], "tracked_changes.patch")]:
        # Preserve Git's exact bytes: Windows' locale may be GBK while diffs
        # contain UTF-8 Chinese documentation. Locale decoding corrupts/raises.
        result = subprocess.run(git_prefix + tail, cwd=ROOT, capture_output=True)
        (run / dest).write_bytes(result.stdout + result.stderr)
    code_paths = list((ROOT / "src/evowm").rglob("*.py")) + list((ROOT / "scripts/allen_world").glob("*.py")) + list((ROOT / "configs/hjepa").rglob("*.yaml"))
    manifest["code_sha256"] = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths}
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Run: {run}\nJobs: {len(jobs)}; GPU slots: {len(a.gpus)}; steps/model: {a.steps}", flush=True)
    if a.dry_run:
        print("Dry run: simulated tables and manifest only; no GPU processes launched.")
        return
    if os.name == "nt":
        raise SystemExit("Training launcher is for the Linux A100 server; use --dry-run locally")
    gpu_info = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
    (run / "gpu_before.txt").write_text(gpu_info.stdout, encoding="utf-8")

    def worker(slot):
        results = []
        for job in jobs[slot::len(a.gpus)]:
            overrides = [f"data.dataset.species_table={job['table']}", f"data.dataset.tree_file={a.tree}",
                         f"data.dataset.holdout.fold={job['fold']}", f"seed={job['seed']}",
                         f"num_workers={a.loader_workers}", "trainer.max_epochs=1", "save_every_n_epochs=1",
                         f"data.dataset.samples_per_epoch={a.steps * 128}", "trainer.limit_val_batches=5",
                         f"subdir={job['name']}"]
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=a.gpus[slot], HJEPA_HOME=str(run / "training"))
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                env[key] = str(a.threads_per_job)
            cmd = ["bash", "scripts/train_hjepa.sh", job["config"]] + overrides
            print(f"START {job['name']} GPU {a.gpus[slot]}", flush=True)
            with (run / "logs" / (job["name"] + ".log")).open("w", encoding="utf-8") as log:
                result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            checkpoint = run / "training/ckpts" / job["name"] / "allen_world_v1_object.ckpt"
            ok = result.returncode == 0 and checkpoint.is_file()
            results.append(job | {"training_ok": ok, "returncode": result.returncode, "checkpoint": str(checkpoint)})
            print(f"{'DONE' if ok else 'FAIL'} {job['name']}", flush=True)
            if not ok:
                break  # Stop this slot, leaving other running jobs intact and collecting failures.
        return results

    with ThreadPoolExecutor(max_workers=len(a.gpus)) as pool:
        trained = [r for batch in pool.map(worker, range(len(a.gpus))) for r in batch]
    (run / "training_status.json").write_text(json.dumps(trained, indent=2), encoding="utf-8")
    if len(trained) != len(jobs) or not all(r["training_ok"] for r in trained):
        raise SystemExit(f"Training failed; inspect {run / 'logs'}. Evaluation not started.")

    def evaluate(job):
        cfg = run / "training/ckpts" / job["name"] / "config.yaml"
        env = os.environ.copy()
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[key] = "1"
        cmd = [sys.executable, "scripts/allen_world/evaluate.py", "--checkpoint", job["checkpoint"],
               "--config", str(cfg), "--out-dir", str(run / "evaluation" / job["name"]), "--threads", "1"]
        with (run / "logs" / (job["name"] + "_eval.log")).open("w", encoding="utf-8") as log:
            result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        return {"name": job["name"], "evaluation_ok": result.returncode == 0}

    with ThreadPoolExecutor(max_workers=min(a.eval_workers, len(trained))) as pool:
        evaluated = list(pool.map(evaluate, trained))
    (run / "evaluation_status.json").write_text(json.dumps(evaluated, indent=2), encoding="utf-8")
    if not all(x["evaluation_ok"] for x in evaluated):
        raise SystemExit(f"Evaluation failed; inspect {run / 'logs'}")
    print(f"FINISHED: {run}\nDownload this complete directory. Scientific recovery is awaiting review, not automatically approved.")


if __name__ == "__main__":
    main()
