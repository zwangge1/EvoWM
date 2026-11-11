"""CPU-only, read-only diagnostics of a completed v3 checkpoint matrix.

Creates a separate result directory; never trains or changes the source run.
Default smoke checks two saved real-data models before the full 50-model audit.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ('full', 'no_link', 'scalar', 'supervised', 'single')
THREAD_KEYS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
DIAGNOSTIC_FILES = (
    'scripts/allen_world/diagnose_v3.py', 'scripts/allen_world/diagnose_v3_checkpoint.py',
    'scripts/allen_world/summarize_diagnose_v3.py', 'scripts/allen_world/run_diagnose_v3.sh',
    'tests/test_allen_diagnose_v3_runner.py', 'tests/test_allen_diagnose_v3_checkpoint.py',
    'tests/test_allen_diagnose_v3_summary.py',
    'docs/design/Allen_v3_checkpoint_diagnostics_2026-11-11.md',
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def contained(path, parent):
    path, parent = Path(path).resolve(), Path(parent).resolve()
    if path == parent or not path.is_relative_to(parent):
        raise ValueError(f'Expected a path strictly within {parent}: {path}')
    return path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', type=Path, required=True)
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--smoke', dest='mode', action='store_const', const='smoke')
    modes.add_argument('--full', dest='mode', action='store_const', const='full')
    p.set_defaults(mode='smoke')
    p.add_argument('--workers', type=int, default=32)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--batch', type=int, default=512)
    p.add_argument('--permutations', type=int, default=5)
    p.add_argument('--bootstrap', type=int, default=10000)
    p.add_argument('--job-timeout-minutes', type=int, default=60)
    p.add_argument('--plan-only', action='store_true', help='Validate saved evidence and list jobs; do not require weights or run inference')
    return p


def validate_args(a):
    if not 1 <= a.workers <= 64 or not 1 <= a.threads <= 8 or a.workers * a.threads > 128:
        raise ValueError('Use 1..64 workers, 1..8 threads, and at most 128 combined CPU threads')
    if not 1 <= a.batch <= 8192 or not 1 <= a.permutations <= 20:
        raise ValueError('Batch must be 1..8192 and permutation repeats 1..20')
    if not 100 <= a.bootstrap <= 100000 or not 1 <= a.job_timeout_minutes <= 180:
        raise ValueError('Bootstrap must be 100..100000; timeout 1..180 minutes')
    return a


def select_jobs(manifest, mode):
    jobs = [dict(j) for j in manifest['jobs'] if j['stage'] == 'real_development']
    expected = {(v, s, f) for v in VARIANTS for s in (13, 14) for f in range(5)}
    identities = [(j['variant'], j['seed'], j['fold']) for j in jobs]
    if len(jobs) != 50 or set(identities) != expected or len({j['name'] for j in jobs}) != 50:
        raise ValueError('Source must contain the complete 50-job real-data v3 matrix')
    if any(not re.fullmatch(r'[A-Za-z0-9_]+', j['name']) for j in jobs):
        raise ValueError('Unsafe job name')
    if mode == 'smoke':
        jobs = [j for j in jobs if j['variant'] in ('full', 'single') and j['seed'] == 13 and j['fold'] == 0]
    return jobs


def inspect_source(source_run, mode, *, require_checkpoints=True, root=ROOT):
    root = Path(root).resolve()
    source_run = contained(root / source_run, root / 'outputs')
    manifest_path = source_run / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('kind') != 'allen_anatomical_development_v3' or manifest['settings'].get('mode') != 'full':
        raise ValueError('Use the completed full v3 run, not the 64-step smoke run')
    completion = json.loads((source_run / 'completion.json').read_text(encoding='utf-8'))
    if not completion.get('all_done') or completion.get('failed') or completion.get('pending'):
        raise ValueError('Source experiment is incomplete')
    jobs = select_jobs(manifest, mode)
    for rel, digest in manifest['input_sha256'].items():
        path = contained(root / rel, source_run)
        if not path.is_file() or sha(path) != digest:
            raise ValueError(f'Source input missing or changed: {rel}')
    runtime_checked = 0
    for rel, digest in manifest['code_sha256'].items():
        archived = contained(source_run / 'source' / rel, source_run / 'source')
        if not archived.is_file() or sha(archived) != digest:
            raise ValueError(f'Source code snapshot missing or changed: {rel}')
        runtime = (rel.startswith('src/evowm/') or rel.startswith('hjepa/h_jepa/')
                   or rel == 'scripts/allen_world/evaluate.py')
        if runtime:
            current = contained(root / rel, root)
            if not current.is_file() or current.read_bytes().replace(b'\r\n', b'\n') != archived.read_bytes().replace(b'\r\n', b'\n'):
                raise ValueError(f'Inference source differs from original run: {rel}; restore matching implementation before diagnosing')
            runtime_checked += 1
    missing = []
    for job in jobs:
        name = job['name']
        for filename in ('pilot_results.json', 'prediction_windows.csv.gz'):
            path = source_run / 'evaluation' / name / filename
            if not path.is_file() or not path.stat().st_size:
                raise ValueError(f'Missing original evaluation: {path}')
        checkpoint = source_run / 'training' / 'ckpts' / name / 'allen_v3_object.ckpt'
        if not checkpoint.is_file():
            missing.append(str(checkpoint))
    if require_checkpoints and missing:
        raise ValueError(f'{len(missing)} checkpoint(s) missing. Run on the server where training weights were retained. First: {missing[0]}')
    return source_run, manifest, jobs, {'source_manifest_sha256': sha(manifest_path),
        'inputs_verified': len(manifest['input_sha256']), 'source_snapshots_verified': len(manifest['code_sha256']),
        'runtime_sources_verified': runtime_checked, 'missing_checkpoints': missing}


def command(job, source_run, output_run, a, root=ROOT):
    return [sys.executable, str(Path(root) / 'scripts/allen_world/diagnose_v3_checkpoint.py'),
        '--run', str(source_run), '--job', job['name'], '--out-dir', str(output_run / 'jobs' / job['name']),
        '--root', str(root), '--threads', str(a.threads), '--batch', str(a.batch),
        '--permutations', str(a.permutations)]


def package_run(run):
    """Package only this new audit; no model weights, original data, or symlinks."""
    archive = run.with_name(run.name + '_review.tar.gz')
    files = sorted(run.rglob('*'))
    if any(p.is_symlink() or p.suffix == '.ckpt' for p in files):
        raise ValueError('Unexpected link/checkpoint in diagnostic output')
    with tarfile.open(archive, 'x:gz') as tar:
        for path in files:
            tar.add(path, arcname=(Path(run.name) / path.relative_to(run)).as_posix(), recursive=False)
    return archive


def execute(a, source_run, original, jobs, audit, root=ROOT):
    root = Path(root).resolve()
    run = root / 'outputs' / ('allen_diagnose_v3_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    run.mkdir(parents=True, exist_ok=False)
    (run / 'jobs').mkdir()
    (run / 'logs').mkdir()
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    manifest = {'kind': 'allen_checkpoint_diagnosis_v3', 'created': datetime.now().isoformat(),
        'mode': a.mode, 'source_run': str(source_run), 'settings': settings, 'jobs': jobs, 'source_audit': audit,
        'interpretation': 'Frozen-weight post hoc diagnostics, not retraining, evolutionary causation or a non-inferiority test. Target-truth replacements are labelled oracle diagnostics only.'}
    hashes = {}
    for rel in DIAGNOSTIC_FILES:
        path = root / rel
        if not path.is_file():
            raise FileNotFoundError(f'Pull the complete diagnostic implementation: {rel}')
        dest = run / 'source' / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        hashes[rel] = sha(path)
    manifest['diagnostic_sha256'] = hashes
    save_json(run / 'manifest.json', manifest)
    shutil.copyfile(source_run / 'manifest.json', run / 'source_manifest.json')
    save_json(run / 'commands.json', {j['name']: command(j, source_run, run, a, root) for j in jobs})
    for args, filename in ((['rev-parse', 'HEAD'], 'git_commit.txt'), (['status', '--short'], 'git_status.txt')):
        result = subprocess.run(['git', '-c', 'safe.directory=' + root.as_posix(), *args], cwd=root, capture_output=True)
        (run / filename).write_bytes(result.stdout + result.stderr)
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True)
    (run / 'environment.txt').write_bytes(freeze.stdout + freeze.stderr)
    print('DIAG_DIRECTORY=' + run.relative_to(root).as_posix(), flush=True)
    env = os.environ.copy()
    env.update({k: str(a.threads) for k in THREAD_KEYS})
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONUNBUFFERED='1', EVOWM_ROOT=str(root))
    env['PYTHONPATH'] = os.pathsep.join([str(root / 'src'), str(root / 'hjepa/h_jepa'), str(root / 'hjepa'), env.get('PYTHONPATH', '')])
    status = {}

    def worker(job):
        name, start = job['name'], time.monotonic()
        log_path = run / 'logs' / (name + '.log')
        with log_path.open('wb') as log:
            try:
                result = subprocess.run(command(job, source_run, run, a, root), cwd=root, env=env,
                    stdout=log, stderr=subprocess.STDOUT, timeout=a.job_timeout_minutes * 60)
                rc = result.returncode
            except subprocess.TimeoutExpired:
                log.write(b'\nDIAGNOSTIC TIMEOUT\n')
                rc = 124
            except OSError as exc:
                log.write(str(exc).encode('utf-8'))
                rc = 127
        complete = rc == 0 and all((run / 'jobs' / name / f).is_file() for f in ('diagnosis.json', 'diagnostic_windows.csv.gz'))
        return name, {'complete': complete, 'returncode': rc, 'seconds': time.monotonic() - start}

    with ThreadPoolExecutor(max_workers=min(a.workers, len(jobs))) as pool:
        for future in as_completed([pool.submit(worker, j) for j in jobs]):
            name, result = future.result()
            status[name] = result
            save_json(run / 'status.json', status)
            print(('DONE ' if result['complete'] else 'FAILED ') + name + f' ({len(status)}/{len(jobs)})', flush=True)
    failed = [j['name'] for j in jobs if not status[j['name']]['complete']]
    summary_rc = None
    if not failed:
        with (run / 'logs' / 'summary.log').open('wb') as log:
            summary_rc = subprocess.run([sys.executable, str(root / 'scripts/allen_world/summarize_diagnose_v3.py'),
                '--run', str(run), '--root', str(root), '--bootstrap', str(a.bootstrap)],
                cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    completion = {'planned': len(jobs), 'completed': len(jobs) - len(failed), 'failed': failed,
        'summary_returncode': summary_rc, 'all_done': not failed and summary_rc == 0,
        'mode': a.mode, 'completed_at': datetime.now().isoformat(),
        'interpretation': 'Smoke verifies execution only. Full output diagnoses fixed fitted models, with no training or original-run edits.'}
    save_json(run / 'completion.json', completion)
    archive = package_run(run)
    print('DOWNLOAD=' + archive.relative_to(root).as_posix(), flush=True)
    print(json.dumps(completion, ensure_ascii=False), flush=True)
    return 0 if completion['all_done'] else 1


def main():
    p = parser()
    a = validate_args(p.parse_args())
    source_run, original, jobs, audit = inspect_source(a.source_run, a.mode, require_checkpoints=not a.plan_only)
    if a.plan_only:
        print(json.dumps({'mode': a.mode, 'jobs': len(jobs), 'selected': [j['name'] for j in jobs],
                          'source_audit': audit, 'note': 'Plan only; no weights loaded or outputs changed.'}, indent=2))
        return 0
    return execute(a, source_run, original, jobs, audit)


if __name__ == '__main__':
    raise SystemExit(main())
