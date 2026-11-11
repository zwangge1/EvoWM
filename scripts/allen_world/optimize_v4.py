"""Nested family validation of a fixed, small Allen JEPA training recipe grid.

Preserves all v3 model code. Linux GPU training runs only after an input/code
snapshot; an inner-only selector precedes every outer-fold prediction.
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
sys.path[:0] = [str(HERE), str(ROOT/'src')]
from overnight_v3 import code_hashes, digest, inside, run_lock, save_json
from optimize_v4_plan import CANDIDATES, make_plan, make_outer_jobs, prepare_inputs
from optimize_v4_selection import select_candidates, summarize_outer

THREAD_KEYS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
DESIGN = 'docs/design/Allen训练优化_v4_2026-11-11.md'


def source_hashes():
    return code_hashes(ROOT) | {DESIGN: digest(ROOT/DESIGN)}


def bind_jobs(jobs, inputs):
    result = []
    for job in jobs:
        item = dict(job)
        entry = inputs['tables'][job['input_key']]
        item.update(table=entry['table'], tree=entry['tree'])
        result.append(item)
    return result


def prepare(args):
    from phylo_baseline_data import load_source_run
    source = load_source_run(ROOT, args.source_run)
    plan = make_plan(args.mode)
    run = ROOT/'outputs'/('allen_optimize_v4_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    run.mkdir()
    for folder in ('source', 'logs', 'evaluation', 'attempts'):
        (run/folder).mkdir()
    table = ROOT/source.jobs[0]['table']
    tree = ROOT/source.jobs[0]['tree']
    inputs = prepare_inputs(table, table.with_suffix('.columns.json'), tree, run/'inputs', ROOT,
                            outer_folds=list(range(5)) if args.mode == 'full' else [0])
    hashes = source_hashes()
    for rel in hashes:
        dest = run/'source'/rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/rel, dest)
    manifest = dict(kind='allen_training_optimization_v4', created=datetime.now().astimezone().isoformat(),
        mode=args.mode, source_run=source.path.relative_to(ROOT).as_posix(),
        baseline_run=args.baseline_run.as_posix() if args.baseline_run else None,
        candidates=list(CANDIDATES), reference_candidate='reference',
        tune_jobs=bind_jobs(plan['inner_jobs'], inputs), outer_jobs=[], inputs=inputs,
        settings=dict(loader_workers=args.loader_workers, threads_per_job=args.threads_per_job),
        source_sha256=hashes,
        input_sha256={p.relative_to(ROOT).as_posix(): digest(p) for p in (run/'inputs').rglob('*') if p.is_file()},
        original_evidence_sha256={p.relative_to(ROOT).as_posix(): digest(p) for p in
            [source.path/'manifest.json', source.path/'completion.json']+
            [source.path/'evaluation'/j['name']/f for j in source.jobs for f in ('pilot_results.json','prediction_windows.csv.gz')]},
        scope='Development on previously inspected outer folds; inner validation prevents direct per-fold tuning, not a new independent confirmation.')
    save_json(run/'manifest.json', manifest)
    for argv, name in ((['rev-parse', 'HEAD'], 'git_commit.txt'), (['status', '--short'], 'git_status.txt')):
        result = subprocess.run(['git', '-c', 'safe.directory='+ROOT.as_posix(), *argv], cwd=ROOT, capture_output=True)
        (run/name).write_bytes(result.stdout+result.stderr)
    print('RUN_DIRECTORY='+run.relative_to(ROOT).as_posix(), flush=True)
    return run, manifest


def verify(run, manifest):
    if manifest.get('kind') != 'allen_training_optimization_v4':
        raise ValueError('Not an Allen v4 optimization run')
    if source_hashes() != manifest['source_sha256']:
        raise ValueError('Code/config/design changed; start a new run')
    for rel, value in manifest['source_sha256'].items():
        if digest(inside(run/'source'/rel, run)) != value:
            raise ValueError('Archived source changed: '+rel)
    for rel, value in manifest['input_sha256'].items():
        if digest(inside(ROOT/rel, run)) != value:
            raise ValueError('Frozen input changed: '+rel)
    if manifest['candidates'] != list(CANDIDATES):
        raise ValueError('Candidate grid changed')
    expected = bind_jobs(make_plan(manifest['mode'])['inner_jobs'], manifest['inputs'])
    if expected != manifest['tune_jobs']:
        raise ValueError('Tuning jobs differ from the frozen experiment')


def train_command(job, manifest):
    settings = manifest['settings']
    command = ['bash', 'scripts/train_hjepa.sh', job['config'],
        f'data.dataset.species_table={job["table"]}', f'data.dataset.tree_file={job["tree"]}',
        f'data.dataset.holdout.fold={job["fold"]}', f'seed={job["seed"]}',
        f'num_workers={settings["loader_workers"]}', 'loader.batch_size=128', 'output_model_name=allen_v4',
        'trainer.max_epochs=1', 'save_every_n_epochs=1',
        f'data.dataset.samples_per_epoch={job["steps"]*128}',
        'trainer.limit_val_batches=0', 'trainer.num_sanity_val_steps=0', f'subdir={job["name"]}',
        *job['overrides']]
    if settings['loader_workers'] == 0:
        command += ['loader.persistent_workers=false', 'loader.prefetch_factor=null']
    return command


def valid_completed(run, job, state):
    if state.get('evaluation') != 'done':
        return False
    if state.get('job') != job:
        raise ValueError('Completed job metadata changed: '+job['name'])
    for rel, value in state.get('evaluation_sha256', {}).items():
        if digest(inside(run/rel, run)) != value:
            raise ValueError('Completed evaluation changed: '+rel)
    required = {f'evaluation/{job["name"]}/{name}' for name in ('pilot_results.json', 'prediction_windows.csv.gz')}
    if not required <= set(state.get('evaluation_sha256', {})):
        raise ValueError('Completed job has incomplete evidence hashes')
    return True


def invoke(command, log, env, timeout_minutes):
    import signal
    with Path(log).open('ab') as stream:
        try:
            with subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                  start_new_session=True) as process:
                try:
                    return process.wait(timeout=timeout_minutes*60)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    stream.write(b'\nTIMEOUT: process group stopped\n')
                    return 124
        except OSError as exc:
            stream.write(('\nLAUNCH_ERROR: '+str(exc)+'\n').encode())
            return 127


def archive_incomplete_training(run, job):
    """Preserve interrupted artifacts, then restart the declared fixed budget."""
    sources = [run/'training'/'ckpts'/job['name'], run/'evaluation'/job['name']]
    existing = [inside(p, run) for p in sources if p.exists()]
    if not existing:
        return []
    archive = inside(run/'attempts'/(job['name']+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')), run)
    archive.mkdir(parents=True, exist_ok=False)
    moved = []
    for old in existing:
        label = 'training' if old.parent.name == 'ckpts' else 'evaluation'
        destination = inside(archive/label, run)
        old.rename(destination)
        moved.append(destination.relative_to(run).as_posix())
    return moved


def execute_stage(run, manifest, jobs, args, deadline):
    path = run/'status.json'
    status = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    lock, stop = threading.Lock(), threading.Event()
    pending = [j for j in jobs if not valid_completed(run, j, status.get(j['name'], {}))]
    cursor, futures = 0, []
    def update(job, **fields):
        with lock:
            state = status.setdefault(job['name'], {'job': job})
            if state['job'] != job:
                raise ValueError('Cannot resume a job with different settings')
            state.update(fields, updated=datetime.now().astimezone().isoformat())
            save_json(path, status)
    def evaluate(job, checkpoint):
        update(job, evaluation='running')
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES='', **{key:'1' for key in THREAD_KEYS})
        command = [sys.executable, 'scripts/allen_world/evaluate.py', '--checkpoint', str(checkpoint),
                   '--out-dir', str(run/'evaluation'/job['name']), '--threads', '1']
        command += ['--temperature-step', '0'] if job['stage'] in ('inner', 'tune') else ['--all-temperature-axes']
        code = invoke(command, run/'logs'/(job['name']+'_eval.log'), env, args.job_timeout_minutes)
        files = [run/'evaluation'/job['name']/f for f in ('pilot_results.json', 'prediction_windows.csv.gz')]
        okay = code == 0 and all(f.is_file() for f in files)
        update(job, evaluation='done' if okay else 'failed', evaluation_returncode=code,
               evaluation_sha256={f.relative_to(run).as_posix():digest(f) for f in files} if okay else {})
        if not okay:
            stop.set()
        print(('EVALUATED ' if okay else 'FAILED_EVAL ')+job['name'], flush=True)
        return okay
    def worker(slot, pool):
        nonlocal cursor
        while True:
            with lock:
                if cursor == len(pending) or stop.is_set() or time.monotonic() >= deadline:
                    return
                job = pending[cursor]; cursor += 1
            state = status.get(job['name'], {})
            if any(state.get(k) == 'failed' for k in ('training', 'evaluation')) and not args.retry_failed:
                stop.set(); return
            checkpoint = run/'training'/'ckpts'/job['name']/'allen_v4_object.ckpt'
            trained = state.get('training') == 'done' and checkpoint.is_file()
            if trained and digest(checkpoint) != state.get('checkpoint_sha256'):
                raise ValueError('Saved trained checkpoint changed')
            if not trained:
                archived = archive_incomplete_training(run, job)
                update(job, training='running', evaluation='pending', gpu=args.gpus[slot],
                       archived_incomplete_attempt=archived, checkpoint_sha256=None)
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=args.gpus[slot], HJEPA_HOME=str(run/'training'), EVOWM_ROOT=str(ROOT))
                env.update({k:str(manifest['settings']['threads_per_job']) for k in THREAD_KEYS})
                command = train_command(job, manifest)
                save_json(run/'logs'/(job['name']+'_command.json'), command)
                before = time.monotonic()
                print(f'START {job["name"]} GPU={args.gpus[slot]} steps={job["steps"]}', flush=True)
                code = invoke(command, run/'logs'/(job['name']+'.log'), env, args.job_timeout_minutes)
                trained = code == 0 and checkpoint.is_file()
                update(job, training='done' if trained else 'failed', training_returncode=code,
                       training_seconds=time.monotonic()-before,
                       checkpoint_sha256=digest(checkpoint) if trained else None)
                if not trained:
                    stop.set(); return
            with lock:
                futures.append(pool.submit(evaluate, job, checkpoint))
    with ThreadPoolExecutor(max_workers=args.eval_workers) as pool:
        with ThreadPoolExecutor(max_workers=len(args.gpus)) as gpu_pool:
            for future in as_completed([gpu_pool.submit(worker, slot, pool) for slot in range(len(args.gpus))]):
                future.result()
        for future in as_completed(futures):
            future.result()
    return all(valid_completed(run, j, status.get(j['name'], {})) for j in jobs)


def ensure_baseline(run, manifest, args):
    if not manifest.get('baseline_run'):
        command = [sys.executable, 'scripts/allen_world/phylo_baselines_v1.py', '--source-run', manifest['source_run'],
                   '--full', '--workers', '5', '--threads', '2']
        env = os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES='', **{k:'1' for k in THREAD_KEYS})
        path = run/'logs'/'classic_baselines.log'
        code = invoke(command, path, env, args.job_timeout_minutes)
        for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
            if line.startswith('RUN_DIRECTORY='):
                manifest['baseline_run'] = line.split('=', 1)[1]
        save_json(run/'manifest.json', manifest)
        if code or not manifest.get('baseline_run'):
            raise RuntimeError('Classic baseline run failed; inspect logs/classic_baselines.log')
    from phylo_baselines_v1 import verify_snapshot, completed_result
    baseline = inside(ROOT/manifest['baseline_run'], ROOT/'outputs')
    declaration = json.loads((baseline/'manifest.json').read_text(encoding='utf-8'))
    verify_snapshot(baseline, declaration)
    if declaration['source_run'] != manifest['source_run'] or declaration['folds'] != list(range(5)):
        raise ValueError('Classic baseline is not the matching complete source experiment')
    completion = baseline/'completion.json'
    complete = completion.is_file() and json.loads(completion.read_text(encoding='utf-8')).get('all_done')
    if not complete or not all(completed_result(baseline, f) for f in range(5)):
        command = [sys.executable, 'scripts/allen_world/phylo_baselines_v1.py', '--resume',
                   manifest['baseline_run'], '--workers', '5', '--threads', '2']
        env = os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES='', **{k:'1' for k in THREAD_KEYS})
        code = invoke(command, run/'logs'/'classic_baselines_resume.log', env, args.job_timeout_minutes)
        if code or not completion.is_file() or not json.loads(completion.read_text(encoding='utf-8')).get('all_done'):
            raise ValueError('Classic baseline resume failed; inspect logs/classic_baselines_resume.log')
        if not all(completed_result(baseline, f) for f in range(5)):
            raise ValueError('Classic baseline still has incomplete folds')


def pack(run):
    import tarfile
    path = run.with_name(run.name+'_review.tar.gz')
    tmp = path.with_suffix('.gz.tmp')
    with tarfile.open(tmp, 'w:gz') as archive:
        for item in sorted(run.rglob('*')):
            rel = item.relative_to(run)
            if item.is_symlink():
                raise ValueError('Unexpected symlink in run')
            if item.is_file() and item.suffix != '.ckpt' and 'attempts' not in rel.parts and item.name != '.run.lock':
                archive.add(item, arcname=(Path(run.name)/rel).as_posix(), recursive=False)
    os.replace(tmp, path)
    print('DOWNLOAD='+path.relative_to(ROOT).as_posix(), flush=True)


def run_experiment(args, run, manifest):
    verify(run, manifest)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_json(run/('launch_'+stamp+'.json'), {k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()})
    gpu = subprocess.run(['nvidia-smi'], capture_output=True, check=True)
    (run/('gpu_'+stamp+'.txt')).write_bytes(gpu.stdout)
    environment = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True)
    (run/('environment_'+stamp+'.txt')).write_bytes(environment.stdout+environment.stderr)
    deadline = time.monotonic()+args.hours*3600
    if not execute_stage(run, manifest, manifest['tune_jobs'], args, deadline):
        save_json(run/'completion.json', dict(all_done=False, stage='inner', mode=manifest['mode']))
        pack(run); return 2
    selection = select_candidates(run, manifest, ROOT)
    jobs = make_outer_jobs(manifest['mode'], {int(k):v for k,v in selection['selected_by_outer'].items()})
    outer = bind_jobs(jobs, manifest['inputs'])
    if manifest['outer_jobs'] and manifest['outer_jobs'] != outer:
        raise ValueError('Inner selection changed after outer jobs were declared')
    manifest['outer_jobs'] = outer
    save_json(run/'manifest.json', manifest)
    if not execute_stage(run, manifest, outer, args, deadline):
        save_json(run/'completion.json', dict(all_done=False, stage='outer', mode=manifest['mode']))
        pack(run); return 2
    ensure_baseline(run, manifest, args)
    for rel, value in manifest['original_evidence_sha256'].items():
        if digest(ROOT/rel) != value:
            raise ValueError('Original comparison evidence changed: '+rel)
    summary = summarize_outer(run, manifest, selection, ROOT)
    save_json(run/'completion.json', dict(all_done=True, stage='complete', mode=manifest['mode'],
        inner_jobs=len(manifest['tune_jobs']), outer_jobs=len(outer), summary=summary,
        completed=datetime.now().astimezone().isoformat()))
    pack(run)
    return 0


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--smoke', dest='mode', action='store_const', const='smoke')
    mode.add_argument('--full', dest='mode', action='store_const', const='full')
    p.add_argument('--source-run', type=Path)
    p.add_argument('--baseline-run', type=Path)
    p.add_argument('--gpus', nargs='+', default=[])
    p.add_argument('--hours', type=float, default=8.)
    p.add_argument('--eval-workers', type=int, default=32)
    p.add_argument('--loader-workers', type=int, default=4)
    p.add_argument('--threads-per-job', type=int, default=4)
    p.add_argument('--job-timeout-minutes', type=float, default=120.)
    p.add_argument('--resume', type=Path)
    p.add_argument('--retry-failed', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    return p


def main():
    args = parser().parse_args()
    if not args.dry_run:
        if os.name != 'posix':
            raise ValueError('GPU launcher is for Linux; local Windows supports --dry-run')
        if not 1 <= len(args.gpus) <= 4 or any(not x.isdecimal() for x in args.gpus):
            raise ValueError('Select one to four GPU indices')
        args.gpus = [str(int(x)) for x in args.gpus]
        if len(set(args.gpus)) != len(args.gpus):
            raise ValueError('Duplicate GPU indices')
    if (not math.isfinite(args.hours) or not 0 < args.hours <= 168 or not 1 <= args.eval_workers <= 64
        or not 0 <= args.loader_workers <= 32 or not 1 <= args.threads_per_job <= 16
        or not math.isfinite(args.job_timeout_minutes) or not 0 < args.job_timeout_minutes <= 720):
        raise ValueError('Invalid resource limits')
    if args.resume:
        run = inside(ROOT/args.resume, ROOT/'outputs')
        manifest = json.loads((run/'manifest.json').read_text(encoding='utf-8'))
        if args.mode and args.mode != manifest['mode']:
            raise ValueError('Resume cannot change smoke/full mode')
        if args.source_run and (ROOT/args.source_run).resolve() != (ROOT/manifest['source_run']).resolve():
            raise ValueError('Resume cannot change source run')
        if args.baseline_run and (ROOT/args.baseline_run).resolve() != (ROOT/(manifest.get('baseline_run') or '')).resolve():
            raise ValueError('Resume cannot change baseline run')
        print('RESUME_DIRECTORY='+run.relative_to(ROOT).as_posix(), flush=True)
    else:
        if args.source_run is None:
            raise ValueError('--source-run is required for a new run')
        args.mode = args.mode or 'smoke'
        if args.baseline_run:
            args.baseline_run = inside(ROOT/args.baseline_run, ROOT/'outputs').relative_to(ROOT)
        run, manifest = prepare(args)
    verify(run, manifest)
    if args.dry_run:
        save_json(run/'preflight.json', dict(validated=True, training_launched=False,
            mode=manifest['mode'], tune_jobs=len(manifest['tune_jobs']),
            outer_jobs_after_selection=30 if manifest['mode']=='full' else 3))
        print('DRY_RUN: no model training or new performance result', flush=True)
        return 0
    with run_lock(run):
        return run_experiment(args, run, manifest)


if __name__ == '__main__':
    raise SystemExit(main())
