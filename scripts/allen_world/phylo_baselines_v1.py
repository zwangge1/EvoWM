"""Add exact multivariate BM/OU predictors to a frozen Allen v3 evaluation.

CPU only, no checkpoints, no neural retraining. New outputs preserve the saved
target/reference windows and fit classical parameters only to training species.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT/'src'))
SOURCE_FILES = (
    'scripts/allen_world/phylo_baselines_v1.py',
    'scripts/allen_world/phylo_baseline_core.py',
    'scripts/allen_world/phylo_baseline_data.py',
    'scripts/allen_world/summarize_phylo_baselines.py',
    'scripts/allen_world/run_phylo_baselines_v1.sh',
    'scripts/allen_world/diagnose_v3.py', 'src/evowm/data/tree.py',
    'tests/test_allen_phylo_core.py', 'tests/test_allen_phylo_data.py',
    'tests/test_allen_phylo_runner.py', 'tests/test_allen_phylo_summary.py',
    'docs/design/Allen_BM_OU预测基线_v1_2026-11-11.md',
)
THREAD_KEYS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def serializable(value):
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(serializable(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def inside(path, parent):
    path, parent = Path(path).resolve(), Path(parent).resolve()
    if not path.is_relative_to(parent) or path == parent:
        raise ValueError(f'Expected a path within {parent}: {path}')
    return path


def verify_snapshot(run, manifest):
    for rel, digest in manifest['source_sha256'].items():
        if sha(ROOT/rel) != digest or sha(run/'source'/rel) != digest:
            raise ValueError('Code/design changed after snapshot: '+rel)
    for rel, digest in manifest['frozen_sha256'].items():
        if sha(inside(run/rel, run)) != digest:
            raise ValueError('Frozen input changed: '+rel)


def prepare(args):
    import numpy as np
    from phylo_baseline_data import load_source_run, load_fold, STATE_COLUMNS
    source = load_source_run(ROOT, args.source_run)
    folds = list(range(5)) if args.mode == 'full' else [0]
    run = ROOT/'outputs'/('allen_phylo_baselines_v1_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    run.mkdir()
    for folder in ('inputs', 'source', 'logs', 'folds', 'attempts'):
        (run/folder).mkdir()
    table = ROOT/source.jobs[0]['table']
    tree = ROOT/source.jobs[0]['tree']
    for original, name in ((table, 'species_traits.csv'), (tree, 'tree.nwk'),
                           (table.with_suffix('.columns.json'), 'species_traits.columns.json')):
        shutil.copyfile(original, run/'inputs'/name)
    for fold in folds:
        data = load_fold(source, fold)
        if not np.isfinite(data.states[data.train_rows, 0]).all():
            raise ValueError('Training eligibility requires external tarsus observations')
        np.savez_compressed(run/'inputs'/f'fold_{fold}.npz', X=data.X, states=data.states,
                            train_rows=data.train_rows, reference_rows=data.reference_rows,
                            target_rows=data.target_rows, tip_ids_of_rows=data.tip_ids_of_rows)
        identity = ['window', 'target_row', 'source_row', 'target_tip', 'source_tip', 'family',
                    'context_0_tip', 'context_1_tip', 'context_2_tip', 'complete_three_bones']
        identity += ['observed_'+c for c in STATE_COLUMNS]
        data.saved_windows[identity].to_csv(run/'inputs'/f'fold_{fold}_windows.csv.gz', index=False, compression='gzip')
        save_json(run/'inputs'/f'fold_{fold}_design.json', data.design_info)
        save_json(run/'inputs'/f'fold_{fold}_source.json', data.source_refs)
    hashes = {}
    for rel in SOURCE_FILES:
        original = ROOT/rel
        if not original.is_file():
            raise FileNotFoundError('Missing complete implementation: '+rel)
        dest = run/'source'/rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, dest)
        hashes[rel] = sha(original)
    evidence = [source.path/'manifest.json', source.path/'completion.json']
    evidence += [source.path/'evaluation'/j['name']/file for j in source.jobs
                 for file in ('pilot_results.json', 'prediction_windows.csv.gz')]
    manifest = {
        'kind': 'allen_phylogenetic_baselines_v1', 'created': datetime.now().astimezone().isoformat(),
        'mode': args.mode, 'folds': folds, 'source_run': source.path.relative_to(ROOT).as_posix(),
        'arguments': vars(args), 'source_sha256': hashes,
        'frozen_sha256': {p.relative_to(run).as_posix(): sha(p) for p in (run/'inputs').iterdir()},
        'source_evidence_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in evidence},
        'source_audit': source.audit,
        'protocol': 'Training excludes held-out families. At prediction, the same three saved reference birds may belong to held-out families; target morphology is hidden. Multivariate fixed-root BM/OU regression, shared process covariance, proportional tip error estimated by train-only ML.',
        'mean_design': 'Shared train/bone-support identifiable main effects plus prespecified temperature interactions.',
        'primary_metric': 'allocation_logratio_joint',
    }
    save_json(run/'manifest.json', manifest)
    for argv, filename in ((['rev-parse', 'HEAD'], 'git_commit.txt'), (['status', '--short'], 'git_status.txt')):
        result = subprocess.run(['git', '-c', 'safe.directory='+ROOT.as_posix(), *argv], cwd=ROOT, capture_output=True)
        (run/filename).write_bytes(result.stdout+result.stderr)
    result = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True)
    (run/'environment.txt').write_bytes(result.stdout+result.stderr)
    print('RUN_DIRECTORY='+run.relative_to(ROOT).as_posix(), flush=True)
    return run, manifest


def run_fold(run, fold):
    import numpy as np
    import pandas as pd
    from evowm.data.tree import read_trees
    from phylo_baseline_core import PhyloKernel, PhyloOptimizationError, fit_nested, predict_joint
    run = Path(run).resolve()
    manifest = json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    if fold not in manifest['folds']:
        raise ValueError('Fold not configured in this run')
    verify_snapshot(run, manifest)
    out = run/'folds'/f'fold_{fold}'
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    with np.load(run/'inputs'/f'fold_{fold}.npz', allow_pickle=False) as z:
        X, states, train, refs, target, tips = (z[k] for k in
            ('X', 'states', 'train_rows', 'reference_rows', 'target_rows', 'tip_ids_of_rows'))
    if np.intersect1d(train, np.r_[target, refs.ravel()]).size or np.any(refs == target[:, None]):
        raise ValueError('Train/test or target/reference leakage in frozen inputs')
    if (tips[np.r_[train, target, refs.ravel()]] < 0).any():
        raise ValueError('Required species missing from tree')
    kernel = PhyloKernel(read_trees(run/'inputs'/'tree.nwk', max_trees=1)[0])
    frame = pd.read_csv(run/'inputs'/f'fold_{fold}_windows.csv.gz')
    fits = {}
    lengths = ('m_tarsus', 'b_fem', 'b_tbt', 'b_tmt')
    for model in ('BM', 'OU'):
        before = time.monotonic()
        print(f'FIT_BEGIN fold={fold} {model} training_species={len(train)} predictors={X.shape[1]}', flush=True)
        try:
            fit = fit_nested(kernel, tips[train], X[train], states[train], model=model, estimate_eta=True)
        except PhyloOptimizationError as exc:
            save_json(out/(model.lower()+'_optimization_failure.json'), exc.diagnostics)
            raise
        if model == 'OU' and fit['loglik'] < fits['BM']['loglik'] - 1e-6:
            raise ValueError('OU optimization is worse than its included BM boundary')
        save_json(out/(model.lower()+'_fit.json'), fit)
        mean = X[target] @ fit['beta']
        predicted = np.empty((len(target), 4), dtype=float)
        covariance = np.empty((len(target), 4, 4), dtype=float)
        for k, (reference, query) in enumerate(zip(refs, target)):
            value, uncertainty = predict_joint(kernel, fit, tips[reference], X[reference], states[reference],
                tips[query:query+1], X[query:query+1], return_cov=True)
            predicted[k], covariance[k] = value[0], uncertainty
        if not np.isfinite(predicted).all() or not np.isfinite(covariance).all():
            raise ValueError('Nonfinite conditional predictions')
        if np.linalg.eigvalsh(covariance).min() < -1e-8:
            raise ValueError('Conditional covariance is not positive semidefinite')
        for j, col in enumerate(lengths):
            frame[f'{model.lower()}_{col}'] = predicted[:, j]
            frame[f'{model.lower()}_mean_{col}'] = mean[:, j]
            frame[f'{model.lower()}_conditional_variance_{col}'] = covariance[:, j, j]
        np.savez_compressed(out/(model.lower()+'_conditional_covariance.npz'), covariance=covariance)
        fits[model] = {k: fit[k] for k in ('alpha', 'eta', 'loglik')}
        fits[model]['seconds'] = time.monotonic()-before
        print(f'FIT fold={fold} {model} seconds={fits[model]["seconds"]:.1f}', flush=True)
    frame.to_csv(out/'predictions.csv.gz', index=False, compression='gzip')
    files = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
    save_json(out/'result.json', {'fold': fold, 'status': 'complete', 'n_training': len(train),
        'n_training_bones': int(np.isfinite(states[train, 1:]).all(axis=1).sum()),
        'n_windows': len(target), 'n_predictors': X.shape[1], 'fits': fits,
        'output_sha256': files, 'seconds': time.monotonic()-started,
        'conditional_variance': 'Plug-in residual predictive variance; does not include parameter/phylogenetic uncertainty.'})


def completed_result(run, fold):
    folder = run/'folds'/f'fold_{fold}'
    path = folder/'result.json'
    if not path.is_file():
        return None
    result = json.loads(path.read_text(encoding='utf-8'))
    if result.get('status') != 'complete':
        return None
    for file, digest in result['output_sha256'].items():
        if sha(inside(folder/file, folder)) != digest:
            raise ValueError('Completed fold output changed: '+str(folder/file))
    return result


def package(run):
    path = run.with_name(run.name+'_review.tar.gz')
    temporary = path.with_suffix(path.suffix+'.tmp')
    with tarfile.open(temporary, 'w:gz') as tar:
        for item in sorted(run.rglob('*')):
            if item.is_symlink():
                raise ValueError('Unexpected symlink in results')
            if item.is_file() and 'attempts' not in item.relative_to(run).parts:
                tar.add(item, arcname=(Path(run.name)/item.relative_to(run)).as_posix(), recursive=False)
    os.replace(temporary, path)
    print('DOWNLOAD='+path.relative_to(ROOT).as_posix(), flush=True)


def execute(run, manifest, args):
    from summarize_phylo_baselines import summarize
    verify_snapshot(run, manifest)
    status = {}
    pending = []
    for fold in manifest['folds']:
        result = completed_result(run, fold)
        if result is not None:
            status[str(fold)] = result
        else:
            folder = run/'folds'/f'fold_{fold}'
            if folder.exists():
                old = inside(folder, run)
                new = inside(run/'attempts'/(folder.name+'_'+datetime.now().strftime('%H%M%S_%f')), run)
                old.rename(new)
            pending.append(fold)
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'], env['CUDA_VISIBLE_DEVICES'] = '1', ''
    for name in THREAD_KEYS:
        env[name] = str(args.threads)
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT/'src'), str(HERE), env.get('PYTHONPATH', '')])
    def worker(fold):
        command = [sys.executable, str(Path(__file__).resolve()), '--worker-run', str(run), '--fold', str(fold)]
        with (run/'logs'/f'fold_{fold}.log').open('ab') as log:
            try:
                code = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                      timeout=args.timeout_minutes*60).returncode
            except subprocess.TimeoutExpired:
                log.write(b'\nTIMEOUT\n'); code = 124
        result = completed_result(run, fold) if code == 0 else None
        return fold, result or {'fold': fold, 'status': 'failed', 'returncode': code}
    with ThreadPoolExecutor(max_workers=min(args.workers, max(1, len(pending)))) as pool:
        for future in as_completed([pool.submit(worker, fold) for fold in pending]):
            fold, result = future.result()
            status[str(fold)] = result
            save_json(run/'status.json', status)
            print(f'{result["status"].upper()} fold={fold} ({len(status)}/{len(manifest["folds"])})', flush=True)
    save_json(run/'status.json', status)
    failed = sum(v['status'] != 'complete' for v in status.values())
    details = {}
    if not failed:
        for rel, digest in manifest['source_evidence_sha256'].items():
            if sha(ROOT/rel) != digest:
                raise ValueError('Original comparison evidence changed: '+rel)
        details = summarize(run, manifest, ROOT, args.bootstrap)
    save_json(run/'completion.json', {'planned': len(manifest['folds']), 'complete': len(status)-failed,
        'failed': failed, 'all_done': failed == 0, 'mode': manifest['mode'],
        'full_five_folds': manifest['folds'] == list(range(5)),
        'completed_at': datetime.now().astimezone().isoformat(), **details})
    package(run)
    return 0 if not failed else 1


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', type=Path)
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--smoke', action='store_const', dest='mode', const='smoke')
    modes.add_argument('--full', action='store_const', dest='mode', const='full')
    p.set_defaults(mode='smoke')
    p.add_argument('--workers', type=int, default=5)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--bootstrap', type=int, default=10000)
    p.add_argument('--timeout-minutes', type=int, default=180)
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--resume', type=Path)
    p.add_argument('--worker-run', type=Path, help=argparse.SUPPRESS)
    p.add_argument('--fold', type=int, help=argparse.SUPPRESS)
    return p


def main():
    args = parser().parse_args()
    if args.worker_run:
        run_fold(inside(ROOT/args.worker_run, ROOT/'outputs'), args.fold)
        return 0
    if not 1 <= args.workers <= 5 or not 1 <= args.threads <= 16 or args.workers*args.threads > 64:
        raise ValueError('Use 1..5 fold workers, 1..16 BLAS threads, combined at most64')
    if not 100 <= args.bootstrap <= 100000 or not 1 <= args.timeout_minutes <= 720:
        raise ValueError('Bootstrap/timeout outside supported bounds')
    if args.resume:
        run = inside(ROOT/args.resume, ROOT/'outputs')
        manifest = json.loads((run/'manifest.json').read_text(encoding='utf-8'))
        if manifest.get('kind') != 'allen_phylogenetic_baselines_v1':
            raise ValueError('Resume only this baseline experiment')
        if args.source_run and (ROOT/args.source_run).resolve() != (ROOT/manifest['source_run']).resolve():
            raise ValueError('Resume source run mismatch')
        print('RESUME_DIRECTORY='+run.relative_to(ROOT).as_posix(), flush=True)
    else:
        if args.source_run is None:
            raise ValueError('--source-run is required for a new experiment')
        run, manifest = prepare(args)
    if args.check_only:
        verify_snapshot(run, manifest)
        save_json(run/'preflight.json', {'validated': True, 'folds': manifest['folds'], 'models_fitted': 0})
        package(run)
        return 0
    return execute(run, manifest, args)


if __name__ == '__main__':
    raise SystemExit(main())
