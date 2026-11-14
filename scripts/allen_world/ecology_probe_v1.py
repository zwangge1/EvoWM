"""Run matched frozen-model ecological reallocation probes, never training.

Target-only conditions; all 42 directed pairs; all six corners must satisfy
the same training-only support screen. Original runs are read-only. Missing
checkpoints are reported during preflight, not downloaded or reconstructed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'src'), str(HERE)]
DEFAULT_SOURCE = 'outputs/allen_asymmetry_v1_20261112_053704_445314'
VARIANTS = ('full', 'single', 'ou')
NEW_FILES = (
    'scripts/allen_world/ecology_probe_v1.py',
    'scripts/allen_world/ecology_probe_core.py',
    'scripts/allen_world/ecology_probe_backends.py',
    'scripts/allen_world/ecology_probe_support.py',
    'scripts/allen_world/summarize_ecology_probe_v1.py',
    'scripts/allen_world/run_ecology_probe_v1.sh',
    'docs/design/Allen觅食条件探针_v1_2026-11-14.md',
    'tests/test_allen_ecology_probe_core.py',
    'tests/test_allen_ecology_probe_backends.py',
    'tests/test_allen_ecology_probe_support.py',
    'tests/test_allen_ecology_probe_summary.py',
    'tests/test_allen_ecology_probe_runner.py',
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def select_jobs(mode, models):
    if mode not in ('smoke', 'full') or not models or not set(models) <= set(VARIANTS):
        raise ValueError('Unknown mode/model selection')
    if len(models) != len(set(models)):
        raise ValueError('Duplicate model selection')
    folds, seeds = ((0,), (13,)) if mode == 'smoke' else (range(5), (13, 14))
    return [{'name': f'{variant}_f{fold}_s{seed}', 'fold': fold, 'seed': seed, 'variant': variant}
            for fold, seed, variant in itertools.product(folds, seeds, models)]


def select_windows(cell, *, max_species=None, max_windows_per_species=2):
    """Input-only deterministic selection; never use morphology or error."""
    import numpy as np
    from ecology_probe_core import CONDITION_COLUMNS
    if max_windows_per_species < 1 or (max_species is not None and max_species < 1):
        raise ValueError('Invalid window/species limits')
    raw = cell.table[list(CONDITION_COLUMNS)].to_numpy(float)[cell.target_rows]
    eligible = np.isfinite(raw).all(axis=1)
    tips = cell.table.tip.to_numpy()[cell.target_rows]
    names = sorted(set(tips[eligible]), key=lambda s: hashlib.sha256(('ecology_probe_v1|' + s).encode()).hexdigest())
    if max_species is not None:
        names = names[:max_species]
    allowed, count, indices = set(names), {}, []
    for i, tip in enumerate(tips):
        if eligible[i] and tip in allowed and count.get(tip, 0) < max_windows_per_species:
            indices.append(i)
            count[tip] = count.get(tip, 0) + 1
    if not indices:
        raise ValueError('No complete-condition target windows')
    return np.asarray(indices, dtype=int)


def _worker(run, job_name, threads, batch):
    # Fresh subprocess: set BLAS limits before importing numerical libraries.
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = str(threads)
    import numpy as np
    import pandas as pd
    from asymmetry_v1_evaluate import load_cell
    from ecology_probe_backends import find_job, load_backend
    from ecology_probe_core import (CONDITION_COLUMNS, FORAGING_COLUMNS, SCENARIO_NAMES,
                                     target_condition_scenarios)
    from ecology_probe_support import JointSupport

    started = time.monotonic()
    run = Path(run).resolve()
    manifest = read_json(run / 'manifest.json')
    found = [j for j in manifest['jobs'] if j['name'] == job_name]
    if len(found) != 1:
        raise ValueError('Unknown frozen probe job')
    job, settings = found[0], manifest['settings']
    for rel, digest in manifest['code_sha256'].items():
        if sha(ROOT / rel) != digest or sha(run / 'source' / rel) != digest:
            raise ValueError('Probe code changed after snapshot: ' + rel)
    source = ROOT / manifest['source_run']
    if sha(source / 'manifest.json') != manifest['source_manifest_sha256']:
        raise ValueError('Source manifest changed after probe preparation')
    original_checks = [item for item in manifest['preflight']
                       if item['variant'] == job['variant']
                       and item['job']['fold'] == job['fold'] and item['job']['seed'] == job['seed']]
    if len(original_checks) != 1:
        raise ValueError('Missing unique frozen preflight record')
    original_check = original_checks[0]
    if sha(original_check['paths']['result']) != original_check['actual_sha256']['result']:
        raise ValueError('Source result metadata changed after probe preparation')
    _, original_job, _ = find_job(source, job['fold'], job['seed'], job['variant'])
    cell = load_cell(source, original_job['name'])
    indices = select_windows(cell, max_species=settings['max_species'],
                              max_windows_per_species=settings['max_windows_per_species'])
    out = run / 'jobs' / job_name
    out.mkdir(parents=True, exist_ok=False)
    (out / 'pairs').mkdir()
    (out / 'eligibility').mkdir()
    target_rows = cell.target_rows[indices]
    targets = cell.table.iloc[target_rows]
    windows = pd.DataFrame({'selected_index': np.arange(len(indices)), 'window': indices,
                            'target_row': target_rows, 'target_tip': targets.tip.to_numpy(),
                            'family': targets.family.to_numpy(), 'order': targets.order.to_numpy(),
                            'mode': targets['mode'].to_numpy(),
                            'complete_three_bones': np.isfinite(cell.states[target_rows, 1:]).all(axis=1)})
    for i in range(3):
        windows[f'context_{i}_tip'] = cell.table.tip.to_numpy()[cell.reference_rows[indices, i]]
    for column in CONDITION_COLUMNS:
        windows[column] = targets[column].to_numpy()
    windows.to_csv(out / 'windows.csv', index=False)
    raw = cell.table[list(CONDITION_COLUMNS)].to_numpy(float)[target_rows]
    groups = targets['mode'].to_numpy()
    if job['variant'] in ('full', 'single'):
        import torch
        torch.set_num_threads(threads)
    support = JointSupport.fit(cell.table, cell.train_rows, condition_columns=CONDITION_COLUMNS)
    save_json(out / 'support_model.json', support.metadata())
    support.training_records.to_csv(out / 'support_training.csv', index=False)
    backend = load_backend(source, job['fold'], job['seed'], job['variant'],
                           window_indices=indices, batch=batch, device='cpu')
    if tuple(backend.condition_columns) != tuple(CONDITION_COLUMNS):
        raise ValueError('Backend condition order differs from probe')
    if not np.array_equal(backend.target_conditions, raw, equal_nan=True):
        raise ValueError('Backend target conditions differ from frozen windows')
    save_json(out / 'backend_audit.json', backend.audit)
    cache, counts = {}, []
    for recipient, donor in itertools.permutations(FORAGING_COLUMNS, 2):
        pair = recipient.removeprefix('e_fs_') + '_from_' + donor.removeprefix('e_fs_')
        scenarios = target_condition_scenarios(raw, recipient=recipient, donor=donor,
                     fraction_shift=settings['fraction_shift'], temperature_step=settings['temperature_step'])
        diagnostics = {name: support.evaluate(scenarios.values[name], groups) for name in SCENARIO_NAMES}
        corner_support = np.column_stack([diagnostics[n].supported.to_numpy(bool) for n in SCENARIO_NAMES])
        passed = scenarios.joint_feasible & corner_support.all(axis=1)
        selected = np.flatnonzero(passed)
        # Unicode arrays only: review NPZ loads safely with allow_pickle=False.
        np.savez_compressed(out / 'eligibility' / (pair + '.npz'),
            ecology_feasible=scenarios.ecology_feasible, temperature_feasible=scenarios.temperature_feasible,
            passed=passed, ecology_reason=scenarios.ecology_reason.astype(str),
            temperature_reason=scenarios.temperature_reason.astype(str), corner_support=corner_support,
            distance=np.column_stack([diagnostics[n].standardized_neighbor_distance for n in SCENARIO_NAMES]),
            threshold=np.column_stack([diagnostics[n].threshold for n in SCENARIO_NAMES]),
            neighbour_families=np.column_stack([diagnostics[n].neighbor_family_count for n in SCENARIO_NAMES]),
            corner_reason=np.column_stack([diagnostics[n].reason.to_numpy(str) for n in SCENARIO_NAMES]))
        kept = windows.iloc[selected]
        count = {'pair': pair, 'recipient': recipient, 'donor': donor,
                 'n_candidate_windows': len(indices), 'n_numeric_feasible': int(scenarios.joint_feasible.sum()),
                 'n_supported_windows': len(selected), 'n_supported_species': int(kept.target_tip.nunique()),
                 'n_supported_families': int(kept.family.nunique()),
                 'minimum_neighbor_families': int(min(diagnostics[n].iloc[selected].neighbor_family_count.min()
                                                      for n in SCENARIO_NAMES)) if len(selected) else None,
                 'status': 'supported' if len(selected) else 'unsupported',
                 'ecology_exclusion_counts': {str(k): int(v) for k, v in pd.Series(scenarios.ecology_reason).value_counts().items()}}
        counts.append(count)
        if len(selected):
            predictions = {'observed': backend.baseline_predictions[selected]}
            for name in ('cold', 'warm'):
                if name not in cache:
                    cache[name] = np.full((len(indices), 4), np.nan)
                needed = selected[~np.isfinite(cache[name][selected]).all(axis=1)]
                if len(needed):
                    cache[name][needed] = backend.predict(scenarios.values[name][needed], window_indices=indices[needed])
                predictions[name] = cache[name][selected]
            for name in ('shifted', 'cold_shifted', 'warm_shifted'):
                predictions[name] = backend.predict(scenarios.values[name][selected], window_indices=indices[selected])
            if set(predictions) != set(SCENARIO_NAMES) or any(v.shape != (len(selected), 4) or not np.isfinite(v).all() for v in predictions.values()):
                raise ValueError('Invalid six-corner paired predictions')
            np.savez_compressed(out / 'pairs' / (pair + '.npz'), selected_indices=selected,
                                **{name: predictions[name].astype(np.float64) for name in SCENARIO_NAMES})
        print(f"PAIR {job_name} {pair} windows={len(selected)} species={count['n_supported_species']}", flush=True)
    save_json(out / 'pair_support.json', counts)
    duration = time.monotonic() - started
    save_json(out / 'result.json', {'status': 'complete', 'job': job, 'seconds': duration,
               'n_selected_windows': len(indices), 'n_selected_species': int(windows.target_tip.nunique()),
               'n_pairs': len(counts), 'n_supported_pairs': sum(x['status'] == 'supported' for x in counts),
               'selection_sha256': sha(out / 'windows.csv'), 'model_retrained': False,
               'baseline_gate': backend.audit, 'source_input_sha256': cell.hashes})
    print(f'COMPLETED {job_name} seconds={duration:.2f}', flush=True)


def prepare(args):
    from ecology_probe_backends import inspect_checkpoint_paths
    from ecology_probe_core import METRIC_DESCRIPTIONS
    source = (ROOT / args.source_run).resolve()
    if not source.is_relative_to((ROOT / 'outputs').resolve()):
        raise ValueError('Source run must be inside this repository outputs directory')
    completion = read_json(source / 'completion.json')
    if not completion.get('all_done') or completion.get('failed'):
        raise ValueError('Use a successfully completed asymmetry source run')
    jobs = select_jobs(args.mode, args.models)
    preflight = [inspect_checkpoint_paths(source, j['fold'], j['seed'], j['variant']) for j in jobs]
    missing = [p for info in preflight for p in info['missing']]
    if missing and not args.plan_only:
        raise FileNotFoundError('Frozen model artifacts are missing. Run on the original server; no training is required. First: ' + missing[0])
    run = ROOT / 'outputs' / ('allen_ecology_probe_v1_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    run.mkdir(parents=True, exist_ok=False)
    for folder in ('source', 'jobs', 'logs'):
        (run / folder).mkdir()
    hashes = {}
    for rel in NEW_FILES:
        original = ROOT / rel
        if not original.is_file():
            raise FileNotFoundError('Pull the complete probe implementation: ' + rel)
        destination = run / 'source' / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, destination)
        hashes[rel] = sha(original)
    max_species = args.max_species if args.max_species else (60 if args.mode == 'smoke' else None)
    manifest = {'kind': 'allen_ecology_probe_v1', 'created': datetime.now().astimezone().isoformat(),
        'source_run': source.relative_to(ROOT).as_posix(), 'jobs': jobs, 'code_sha256': hashes,
        'source_manifest_sha256': sha(source / 'manifest.json'), 'preflight': preflight,
        'settings': {'mode': args.mode, 'models': args.models, 'model_subset': set(args.models) != set(VARIANTS),
            'fraction_shift': args.fraction_shift, 'temperature_step': args.temperature_step,
            'temperature_axis': 'joint_bio1_bio5_bio6', 'max_species': max_species,
            'max_windows_per_species': args.max_windows_per_species, 'workers': args.workers,
            'threads': args.threads, 'batch': args.batch, 'bootstrap': args.bootstrap,
            'plan_only': args.plan_only, 'missing_checkpoint_count': len(missing)},
        'metrics': METRIC_DESCRIPTIONS,
        'protocol': f'Frozen complete100/observed predictors; directed {100 * args.fraction_shift:g}pp replacement; all six corners pass training-bone development-stratified support; exploratory association, not causal/evolutionary contribution; never score shifted prediction against original truth.'}
    save_json(run / 'manifest.json', manifest)
    for argv, filename in ((['rev-parse', 'HEAD'], 'git_commit.txt'), (['status', '--short'], 'git_status.txt'),
                           (['diff', '--', 'scripts/allen_world', 'src/evowm'], 'tracked_changes.patch')):
        result = subprocess.run(['git', *argv], cwd=ROOT, capture_output=True)
        (run / filename).write_bytes(result.stdout + result.stderr)
    print('RUN=' + run.relative_to(ROOT).as_posix(), flush=True)
    return run, manifest


def run_all(run, manifest, args):
    jobs = manifest['jobs']
    status = {j['name']: {'status': 'pending'} for j in jobs}
    save_json(run / 'status.json', status)
    def launch(job):
        command = [sys.executable, str(Path(__file__).resolve()), '--worker-run', str(run),
                   '--job', job['name'], '--workers', '1', '--threads', str(args.threads), '--batch', str(args.batch)]
        start = time.monotonic()
        with (run / 'logs' / (job['name'] + '.log')).open('wb') as log:
            try:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=args.job_timeout_minutes * 60)
                state = {'status': 'complete' if result.returncode == 0 else 'failed',
                         'returncode': result.returncode, 'seconds': time.monotonic() - start}
            except subprocess.TimeoutExpired:
                state = {'status': 'failed', 'reason': 'timeout', 'seconds': time.monotonic() - start}
        return job['name'], state
    with ThreadPoolExecutor(max_workers=min(args.workers, len(jobs))) as executor:
        futures = [executor.submit(launch, job) for job in jobs]
        for future in as_completed(futures):
            name, state = future.result()
            status[name] = state
            save_json(run / 'status.json', status)
            print(state['status'].upper() + ' ' + name + f" seconds={state['seconds']:.1f}", flush=True)
    failed = [name for name, state in status.items() if state['status'] != 'complete']
    save_json(run / 'completion.json', {'all_done': not failed, 'failed': failed,
              'planned': len(jobs), 'completed': len(jobs) - len(failed),
              'mode': args.mode, 'summary_done': False})
    if failed:
        raise RuntimeError('Probe job(s) failed; inspect ' + str(run / 'logs') + ': ' + ', '.join(failed))
    from summarize_ecology_probe_v1 import summarize
    summary = summarize(run, bootstrap=args.bootstrap)
    save_json(run / 'completion.json', {'all_done': True, 'failed': [], 'planned': len(jobs),
               'completed': len(jobs), 'mode': args.mode, 'summary_done': True,
               'summary': summary, 'model_retrained': False})
    archive = Path(str(run) + '_review.tar.gz')
    with tarfile.open(archive, 'w:gz') as tf:
        tf.add(run, arcname=run.name)
    print('DOWNLOAD=' + archive.relative_to(ROOT).as_posix(), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', type=Path, default=Path(DEFAULT_SOURCE))
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--smoke', dest='mode', action='store_const', const='smoke')
    modes.add_argument('--full', dest='mode', action='store_const', const='full')
    p.set_defaults(mode='smoke')
    p.add_argument('--models', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    p.add_argument('--workers', type=int, default=32)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--batch', type=int, default=512)
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--fraction-shift', type=float, default=.1)
    p.add_argument('--temperature-step', type=float, default=1.)
    p.add_argument('--max-species', type=int, default=0)
    p.add_argument('--max-windows-per-species', type=int, default=2)
    p.add_argument('--job-timeout-minutes', type=int, default=180)
    p.add_argument('--plan-only', action='store_true')
    p.add_argument('--worker-run', type=Path, help=argparse.SUPPRESS)
    p.add_argument('--job', help=argparse.SUPPRESS)
    return p


def main():
    args = parser().parse_args()
    if (not 1 <= args.workers <= 64 or not 1 <= args.threads <= 8 or args.workers * args.threads > 128
            or not 1 <= args.batch <= 8192 or not 100 <= args.bootstrap <= 100000
            or not 0 < args.fraction_shift <= .25 or not 0 < args.temperature_step <= 5
            or args.max_species < 0 or not 1 <= args.max_windows_per_species <= 10
            or not 1 <= args.job_timeout_minutes <= 720):
        raise ValueError('Invalid resource/probe settings; max 128 CPU threads, transfer<=25pp, local climate step<=5C')
    if args.worker_run:
        _worker(args.worker_run, args.job, args.threads, args.batch)
        return
    run, manifest = prepare(args)
    if args.plan_only:
        save_json(run / 'completion.json', {'all_done': False, 'stage': 'plan_only',
                   'planned': len(manifest['jobs']), 'executed': 0,
                   'missing_artifacts': [p for i in manifest['preflight'] for p in i['missing']]})
        print('PLAN_ONLY jobs=' + str(len(manifest['jobs'])) + ' missing=' + str(manifest['settings']['missing_checkpoint_count']), flush=True)
        return
    try:
        run_all(run, manifest, args)
    except Exception:
        (run / 'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
