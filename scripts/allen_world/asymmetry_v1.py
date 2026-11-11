"""Paired Allen experiments with scarce and taxonomically uneven bone records.

No model selection and no changes to the existing JEPA or classical models.
Linux GPU launcher; Windows supports input preparation with --dry-run.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(ROOT / "src")]
from overnight_v3 import code_hashes, digest, inside, run_lock, save_json
from optimize_v4 import invoke, pack
from asymmetry_v1_plan import make_plan, prepare_inputs

DESIGN = "docs/design/Allen不对称数据实验_v1_2026-11-12.md"
THREAD_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def source_hashes():
    return code_hashes(ROOT) | {DESIGN: digest(ROOT / DESIGN)}


def prepare(args):
    from phylo_baseline_data import load_source_run
    source = load_source_run(ROOT, args.source_run)
    plan = make_plan(args.mode)
    run = ROOT / "outputs" / ("allen_asymmetry_v1_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    run.mkdir()
    for name in ("source", "logs", "evaluation", "attempts"):
        (run / name).mkdir()
    inputs = prepare_inputs(source, run / "inputs", ROOT, args.mode)
    hashes = source_hashes()
    for rel in hashes:
        dest = run / "source" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, dest)
    manifest = dict(kind="allen_asymmetric_data_v1", mode=args.mode,
        created=datetime.now().astimezone().isoformat(),
        source_run=source.path.relative_to(ROOT).as_posix(),
        jobs=plan["jobs"], counts=plan["counts"], inputs=inputs,
        settings=dict(loader_workers=args.loader_workers, threads_per_job=args.threads_per_job),
        source_sha256=hashes,
        input_sha256={p.relative_to(ROOT).as_posix(): digest(p)
                      for p in (run / "inputs").rglob("*") if p.is_file()},
        protocol=dict(reference_cases=["observed", "hide_half_bones", "hide_all_bones", "hide_distal"],
            primary_metric="allocation_logratio_joint",
            primary_contexts=["uniform25", "family25"],
            primary_comparators=["OU", "single", "supervised"],
            inference="Descriptive family-paired intervals; development on previously inspected outer folds.",
            recipe="Fixed original v3 recipe for all architectures and label budgets; no search or winner selection.",
            hidden_labels="Whole training bone triplets removed from the physical input table, including teacher/context/normalization.",
            references="Same three held-out reference species; their visible morphology is allowed, never target morphology.",
            eco_policy="Ecological inputs unchanged. This experiment tests missing bone measurements, not ecological imputation.",
            replication="Two paired mask/training-seed replicates; these do not separately estimate mask and training variance.",
            classic="Multivariate fixed-root BM/OU, shared-alpha covariance and the existing interaction mean design; retained-bone support selects identifiable columns."))
    save_json(run / "manifest.json", manifest)
    for argv, name in ((["rev-parse", "HEAD"], "git_commit.txt"),
                       (["status", "--short"], "git_status.txt"),
                       (["diff", "--", "src/evowm", "scripts/allen_world", "configs/hjepa"], "tracked_changes.patch")):
        result = subprocess.run(["git", "-c", "safe.directory=" + ROOT.as_posix(), *argv], cwd=ROOT, capture_output=True)
        (run / name).write_bytes(result.stdout + result.stderr)
    print("RUN_DIRECTORY=" + run.relative_to(ROOT).as_posix(), flush=True)
    return run, manifest


def verify(run, manifest):
    if manifest.get("kind") != "allen_asymmetric_data_v1":
        raise ValueError("Not an Allen asymmetry v1 run")
    if source_hashes() != manifest["source_sha256"]:
        raise ValueError("Source/config/design changed; start a new run")
    for rel, expected in manifest["source_sha256"].items():
        if digest(inside(run / "source" / rel, run)) != expected:
            raise ValueError("Archived source changed: " + rel)
    for rel, expected in manifest["input_sha256"].items():
        if digest(inside(ROOT / rel, run / "inputs")) != expected:
            raise ValueError("Frozen input changed: " + rel)
    plan = make_plan(manifest["mode"])
    if manifest["jobs"] != plan["jobs"] or manifest["counts"] != plan["counts"]:
        raise ValueError("Job matrix differs from the fixed protocol")
    if len({j["name"] for j in manifest["jobs"]}) != len(manifest["jobs"]):
        raise ValueError("Duplicate job names")
    for job in manifest["jobs"]:
        entry = manifest["inputs"]["tables"][job["input_key"]]
        for key in ("condition", "fold", "replicate", "mask_seed", "seed"):
            if entry[key] != job[key]:
                raise ValueError("Job/input identity mismatch: " + key)


def train_command(job, manifest):
    entry = manifest["inputs"]["tables"][job["input_key"]]
    settings = manifest["settings"]
    command = ["bash", "scripts/train_hjepa.sh", job["config"],
        f'data.dataset.species_table={entry["table"]}', f'data.dataset.tree_file={entry["tree"]}',
        f'data.dataset.holdout.fold={job["fold"]}', f'seed={job["seed"]}',
        f'num_workers={settings["loader_workers"]}', "loader.batch_size=128",
        "output_model_name=allen_asymmetry_v1", "trainer.max_epochs=1", "save_every_n_epochs=1",
        f'data.dataset.samples_per_epoch={job["steps"] * 128}',
        "trainer.limit_val_batches=0", "trainer.num_sanity_val_steps=0", f'subdir={job["name"]}',
        *job.get("overrides", [])]
    if settings["loader_workers"] == 0:
        command += ["loader.persistent_workers=false", "loader.prefetch_factor=null"]
    return command


def evaluation_command(run, job, checkpoint=None, threads=1):
    command = [sys.executable, "scripts/allen_world/asymmetry_v1_evaluate.py", job["kind"],
               "--run", str(run), "--job", job["name"], "--threads", str(threads)]
    if job["kind"] == "neural":
        if checkpoint is None:
            raise ValueError("Neural evaluation requires a checkpoint")
        command += ["--checkpoint", str(checkpoint)]
    return command


def completed(run, job, state):
    if state.get("evaluation") != "done":
        return False
    if state.get("job") != job:
        raise ValueError("Completed job declaration changed")
    hashes = state.get("output_sha256", {})
    required = {f'evaluation/{job["name"]}/{f}' for f in ("predictions.csv.gz", "result.json")}
    if not required <= set(hashes):
        raise ValueError("Completed job lacks required evidence hashes")
    for rel, expected in hashes.items():
        if digest(inside(run / rel, run / "evaluation")) != expected:
            raise ValueError("Completed output changed: " + rel)
    if json.loads((run / "evaluation" / job["name"] / "result.json").read_text(encoding="utf-8")).get("status") != "complete":
        raise ValueError("Evaluator did not declare completion")
    return True


def archive_attempt(run, job, include_training):
    candidates = [run / "evaluation" / job["name"]]
    if include_training:
        candidates.append(run / "training" / "ckpts" / job["name"])
    existing = [inside(p, run) for p in candidates if p.exists()]
    if not existing:
        return []
    folder = inside(run / "attempts" / (job["name"] + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")), run)
    folder.mkdir(parents=True)
    result = []
    for old in existing:
        dest = inside(folder / ("training" if old.parent.name == "ckpts" else "evaluation"), run)
        old.rename(dest)
        result.append(dest.relative_to(run).as_posix())
    return result


def execute(run, manifest, args):
    verify(run, manifest)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    save_json(run / ("launch_" + stamp + ".json"),
              {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    for command, name in ((["nvidia-smi"], "gpu"), ([sys.executable, "-m", "pip", "freeze"], "environment")):
        result = subprocess.run(command, capture_output=True, check=True)
        (run / (name + "_" + stamp + ".txt")).write_bytes(result.stdout + result.stderr)
    path = run / "status.json"
    status = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    pending = [j for j in manifest["jobs"] if not completed(run, j, status.get(j["name"], {}))]
    if not args.retry_failed and any(status.get(j["name"], {}).get(k) == "failed" for j in pending for k in ("training", "evaluation")):
        raise ValueError("A previous job failed; inspect logs, then resume with --retry-failed")
    lock, stop = threading.Lock(), threading.Event()
    deadline = time.monotonic() + args.hours * 3600
    neural = [j for j in pending if j["kind"] == "neural"]
    classic = [j for j in pending if j["kind"] == "classic"]
    cursor = {"neural": 0, "classic": 0}
    evaluation_futures = []

    def update(job, **fields):
        with lock:
            state = status.setdefault(job["name"], {"job": job})
            if state["job"] != job:
                raise ValueError("Job configuration changed on resume")
            state.update(fields, updated=datetime.now().astimezone().isoformat())
            save_json(path, status)

    def take(kind, jobs):
        with lock:
            if stop.is_set() or time.monotonic() >= deadline or cursor[kind] >= len(jobs):
                return None
            job = jobs[cursor[kind]]
            cursor[kind] += 1
            return job

    def evaluate_job(job, checkpoint=None):
        try:
            threads = args.classic_threads if job["kind"] == "classic" else 1
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES="", **{key: str(threads) for key in THREAD_KEYS})
            update(job, evaluation="running")
            command = evaluation_command(run, job, checkpoint, threads)
            save_json(run / "logs" / (job["name"] + "_evaluation_command.json"), command)
            before = time.monotonic()
            code = invoke(command, run / "logs" / (job["name"] + "_evaluation.log"), env, args.job_timeout_minutes)
            folder = run / "evaluation" / job["name"]
            okay = code == 0 and all((folder / f).is_file() for f in ("predictions.csv.gz", "result.json"))
            if okay:
                okay = json.loads((folder / "result.json").read_text(encoding="utf-8")).get("status") == "complete"
            hashes = {p.relative_to(run).as_posix(): digest(p) for p in folder.rglob("*") if p.is_file()} if okay else {}
            update(job, evaluation="done" if okay else "failed", evaluation_returncode=code,
                   evaluation_seconds=time.monotonic() - before, output_sha256=hashes)
            if not okay:
                stop.set()
            print(("EVALUATED " if okay else "FAILED_EVAL ") + job["name"], flush=True)
        except Exception as exc:
            stop.set()
            update(job, evaluation="failed", error=str(exc))
            raise

    def gpu_worker(slot, pool):
        while (job := take("neural", neural)) is not None:
            try:
                checkpoint = run / "training" / "ckpts" / job["name"] / "allen_asymmetry_v1_object.ckpt"
                state = status.get(job["name"], {})
                trained = state.get("training") == "done" and checkpoint.is_file()
                if trained and digest(checkpoint) != state.get("checkpoint_sha256"):
                    raise ValueError("Saved trained checkpoint changed")
                archived = archive_attempt(run, job, include_training=not trained)
                if not trained:
                    env = os.environ.copy()
                    env.update(CUDA_VISIBLE_DEVICES=args.gpus[slot], HJEPA_HOME=str(run / "training"), EVOWM_ROOT=str(ROOT))
                    env.update({k: str(manifest["settings"]["threads_per_job"]) for k in THREAD_KEYS})
                    command = train_command(job, manifest)
                    save_json(run / "logs" / (job["name"] + "_training_command.json"), command)
                    update(job, training="running", evaluation="pending", gpu=args.gpus[slot],
                           archived_attempt=archived, checkpoint_sha256=None)
                    print(f'START {job["name"]} GPU={args.gpus[slot]} steps={job["steps"]}', flush=True)
                    before = time.monotonic()
                    code = invoke(command, run / "logs" / (job["name"] + "_training.log"), env, args.job_timeout_minutes)
                    trained = code == 0 and checkpoint.is_file()
                    update(job, training="done" if trained else "failed", training_returncode=code,
                           training_seconds=time.monotonic() - before,
                           checkpoint_sha256=digest(checkpoint) if trained else None)
                    if not trained:
                        stop.set()
                        print("FAILED_TRAIN " + job["name"], flush=True)
                        return
                with lock:
                    evaluation_futures.append(pool.submit(evaluate_job, job, checkpoint))
            except Exception as exc:
                stop.set()
                update(job, training="failed", error=str(exc))
                raise

    def classic_worker():
        while (job := take("classic", classic)) is not None:
            archived = archive_attempt(run, job, include_training=False)
            update(job, archived_attempt=archived)
            print("START_CLASSIC " + job["name"], flush=True)
            evaluate_job(job)

    errors = []
    with ThreadPoolExecutor(max_workers=args.eval_workers) as eval_pool:
        with ThreadPoolExecutor(max_workers=len(args.gpus) + args.classic_workers) as workers:
            futures = [workers.submit(gpu_worker, slot, eval_pool) for slot in range(len(args.gpus))]
            futures += [workers.submit(classic_worker) for _ in range(args.classic_workers)]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    errors.append(str(exc))
                    stop.set()
        for future in as_completed(evaluation_futures):
            try:
                future.result()
            except Exception as exc:
                errors.append(str(exc))
                stop.set()
    verify(run, manifest)
    finished = [j["name"] for j in manifest["jobs"] if completed(run, j, status.get(j["name"], {}))]
    failed = [j["name"] for j in manifest["jobs"] if any(status.get(j["name"], {}).get(k) == "failed" for k in ("training", "evaluation"))]
    evaluations_done = len(finished) == len(manifest["jobs"])
    summary_done = False
    if evaluations_done:
        try:
            from summarize_asymmetry_v1 import summarize
            summarize(run, bootstrap=1000 if manifest["mode"] == "smoke" else 10000)
            summary_done = True
        except Exception as exc:
            errors.append("Summary failed: " + str(exc))
            print(errors[-1], flush=True)
    all_done = evaluations_done and summary_done
    save_json(run / "completion.json", dict(all_done=all_done, mode=manifest["mode"],
        planned=len(manifest["jobs"]), completed=len(finished), failed=failed, errors=errors,
        evaluations_done=evaluations_done, summary_done=summary_done,
        stage="complete" if all_done else "failed" if failed or errors else "time_budget_reached",
        finished=datetime.now().astimezone().isoformat()))
    pack(run)
    return 0 if all_done else 1 if failed or errors else 2


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--smoke", dest="mode", action="store_const", const="smoke")
    modes.add_argument("--full", dest="mode", action="store_const", const="full")
    p.add_argument("--source-run", type=Path)
    p.add_argument("--resume", type=Path)
    p.add_argument("--gpus", nargs="+", default=[])
    p.add_argument("--hours", type=float, default=12.)
    p.add_argument("--eval-workers", "--workers", type=int, default=32)
    p.add_argument("--classic-workers", type=int, default=4)
    p.add_argument("--classic-threads", type=int, default=2)
    p.add_argument("--loader-workers", type=int, default=4)
    p.add_argument("--threads-per-job", type=int, default=4)
    p.add_argument("--job-timeout-minutes", type=float, default=120.)
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def validate_args(args):
    if (not math.isfinite(args.hours) or not 0 < args.hours <= 168
            or not 1 <= args.eval_workers <= 64 or not 1 <= args.classic_workers <= 16
            or not 1 <= args.classic_threads <= 16 or not 0 <= args.loader_workers <= 32
            or not 1 <= args.threads_per_job <= 16 or not math.isfinite(args.job_timeout_minutes)
            or not 0 < args.job_timeout_minutes <= 720):
        raise ValueError("Invalid resource limits")
    if not args.dry_run:
        if os.name != "posix":
            raise ValueError("Training launcher requires Linux; Windows supports --dry-run")
        if not 1 <= len(args.gpus) <= 4 or any(not value.isdecimal() for value in args.gpus):
            raise ValueError("Select one to four available GPU indices")
        args.gpus = [str(int(value)) for value in args.gpus]
        if len(set(args.gpus)) != len(args.gpus):
            raise ValueError("Duplicate GPU indices")


def main(argv=None):
    args = parser().parse_args(argv)
    validate_args(args)
    if args.resume:
        run = inside(ROOT / args.resume, ROOT / "outputs")
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        if args.mode and args.mode != manifest["mode"]:
            raise ValueError("Resume cannot change smoke/full mode")
        if args.source_run and (ROOT / args.source_run).resolve() != (ROOT / manifest["source_run"]).resolve():
            raise ValueError("Resume cannot change source run")
        # Training resource settings that change RNG streams are frozen in the manifest.
        args.loader_workers = manifest["settings"]["loader_workers"]
        args.threads_per_job = manifest["settings"]["threads_per_job"]
        print("RESUME_DIRECTORY=" + run.relative_to(ROOT).as_posix(), flush=True)
    else:
        if not args.source_run:
            raise ValueError("--source-run is required for a new experiment")
        args.mode = args.mode or "smoke"
        run, manifest = prepare(args)
    verify(run, manifest)
    save_json(run / "commands.json", {j["name"]: train_command(j, manifest)
              if j["kind"] == "neural" else evaluation_command(run, j, threads=args.classic_threads)
              for j in manifest["jobs"]})
    if args.dry_run:
        save_json(run / "preflight.json", dict(validated=True, training_launched=False,
            mode=manifest["mode"], counts=manifest["counts"]))
        print("DRY_RUN: frozen inputs validated; no new model performance result", flush=True)
        return 0
    with run_lock(run):
        return execute(run, manifest, args)


if __name__ == "__main__":
    raise SystemExit(main())
