"""Development-only, paired family summaries of asymmetric bone availability.

Errors, not predictions, are averaged over windows, species, and repetitions.
Family bootstrap intervals condition on the frozen folds, fitted models and
realized masks. They are descriptive, not confirmatory noninferiority tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from summarize_phylo_baselines import LENGTHS, TARGETS, coordinates

CELLS = ('complete100', 'uniform50', 'uniform25', 'family50', 'family25')
REFCASES = ('observed', 'hide_half_bones', 'hide_all_bones', 'hide_distal')
METHODS = ('full', 'single', 'supervised', 'bm', 'ou')
PRIMARY_COMPARATORS = ('ou', 'single', 'supervised')
PRIMARY_CELLS = ('uniform25', 'family25')
JOINT = 'allocation_logratio_joint'
GROUP = ['cell', 'refcase', 'method', 'target', 'support', 'reference_support']


def _reference_flag(values):
    """Accept genuine Boolean/0/1 flags; never bool('False')."""
    values = pd.Series(values)
    if values.isna().any():
        raise ValueError('Missing has_reference_bones flag')
    mapping = {True: True, False: False, 1: True, 0: False,
               'True': True, 'False': False, 'true': True, 'false': False,
               '1': True, '0': False}
    if not values.isin(list(mapping)).all():
        raise ValueError('Invalid has_reference_bones flag')
    return values.map(mapping).to_numpy(bool)


def species_errors(identity, observed, predicted, method, fold, cell, refcase,
                   replicate, mask_seed, training_seed=-1):
    """Score fixed observed targets, with optional ORIGINAL-reference support.

    has_reference_bones describes the original observed reference measurements,
    before either experimental mask. Thus hide_all_bones remains scoreable on
    the same reference-supported target windows as the other reference cases.
    """
    observed, predicted = np.asarray(observed, float), np.asarray(predicted, float)
    if observed.shape != predicted.shape or observed.shape != (len(identity), 4):
        raise ValueError('Observed/predicted arrays must be matched n-by-four lengths')
    if not np.isfinite(predicted).all():
        raise ValueError('Nonfinite prediction: ' + str(method))
    if method not in METHODS or cell not in CELLS or refcase not in REFCASES:
        raise ValueError('Unknown method/cell/refcase')
    if identity[['target_tip', 'family']].isna().any().any():
        raise ValueError('Missing target or family identity')
    flags = _reference_flag(identity['has_reference_bones'])
    obs, pred = coordinates(observed), coordinates(predicted)
    complete = np.isfinite(observed[:, 1:]).all(axis=1)
    records = []
    for target in TARGETS:
        error = pred[target] - obs[target]
        squared = np.mean(error**2, axis=1) if error.ndim == 2 else error**2
        measured = np.isfinite(squared) & (True if target == 'm_tarsus' else complete)
        for label, select in [('all', np.ones(len(identity), bool)),
                              ('has_reference_bones', flags)]:
            keep = measured & select
            frame = identity.loc[keep, ['target_tip', 'family']].copy()
            frame['mse'] = squared[keep]
            frame = frame.groupby(['target_tip', 'family'], as_index=False).agg(
                mse=('mse', 'mean'), n_windows=('mse', 'size'))
            frame['method'], frame['target'], frame['fold'] = method, target, fold
            frame['cell'], frame['refcase'], frame['replicate'] = cell, refcase, replicate
            frame['mask_seed'], frame['training_seed'] = mask_seed, training_seed
            frame['support'] = 'available' if target == 'm_tarsus' else 'complete_three_bones'
            frame['reference_support'] = label
            records.append(frame)
    return pd.concat(records, ignore_index=True)


def aggregate(species):
    """Window MSE -> species MSE -> family MSE -> mean rep MSE -> RMSE."""
    if species.empty:
        raise ValueError('No measured species errors')
    required = GROUP + ['replicate', 'mask_seed', 'training_seed', 'target_tip',
                        'family', 'fold', 'mse', 'n_windows']
    if species[required].isna().any().any():
        raise ValueError('Missing species error metadata')
    if (not np.isfinite(species.mse.to_numpy(float)).all() or species.mse.lt(0).any()
            or species.n_windows.le(0).any()):
        raise ValueError('Invalid species error or window count')
    if species.groupby('target_tip').family.nunique().gt(1).any():
        raise ValueError('Species has inconsistent family identity')
    if species.groupby('target_tip').fold.nunique().gt(1).any():
        raise ValueError('Species occurs in multiple held-out folds')
    if species.duplicated(GROUP + ['replicate', 'target_tip']).any():
        raise ValueError('Duplicate species error realization')
    signatures, rep_sets = {}, {}
    for key, part in species.groupby(GROUP + ['replicate'], sort=False):
        cell, refcase, method, target, support, reference_support, rep = key
        sig = set(map(tuple, part[['target_tip', 'family', 'fold', 'n_windows']].to_numpy()))
        pair = (target, support, reference_support)
        if pair in signatures and signatures[pair] != sig:
            raise ValueError('Unmatched fixed species/window support across methods, cells, cases or reps')
        signatures[pair] = sig
        if part.mask_seed.nunique() != 1 or part.training_seed.nunique() != 1:
            raise ValueError('A repetition must have one mask seed and one training seed')
        rep_sets.setdefault((cell, refcase, method), set()).add(rep)
    if len({tuple(sorted(v)) for v in rep_sets.values()}) != 1:
        raise ValueError('Unmatched repetitions across methods/cells/reference cases')
    # A mask seed denotes the same missingness realization for every method.
    if species.groupby(['cell', 'replicate']).mask_seed.nunique().gt(1).any():
        raise ValueError('Methods have unmatched mask seeds')
    family_rep = species.groupby(GROUP + ['replicate', 'family'], as_index=False).mse.mean()
    family = family_rep.groupby(GROUP + ['family'], as_index=False).mse.mean()
    metrics = family.groupby(GROUP, as_index=False).agg(mse=('mse', 'mean'),
                                                       n_families=('family', 'nunique'))
    metrics['rmse'] = np.sqrt(metrics.mse)
    counts = species.groupby(GROUP, as_index=False).agg(
        n_species=('target_tip', 'nunique'), n_folds=('fold', 'nunique'),
        n_repetitions=('replicate', 'nunique'), n_mask_seeds=('mask_seed', 'nunique'),
        n_training_seeds=('training_seed', lambda v: v[v >= 0].nunique()))
    metrics = metrics.merge(counts, on=GROUP, validate='one_to_one')
    return family_rep, family, metrics


def _paired_values(series):
    ordered = [s.sort_index() for s in series]
    first = ordered[0]
    if not len(first) or not first.index.is_unique:
        raise ValueError('Paired bootstrap requires nonempty unique family support')
    for item in ordered:
        if not first.index.equals(item.index):
            raise ValueError('Paired bootstrap requires identical family support')
    values = np.stack([s.to_numpy(float) for s in ordered])
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Invalid family MSE')
    return values


def _bootstrap(values, statistic, bootstrap, seed):
    if not isinstance(bootstrap, (int, np.integer)) or bootstrap < 1:
        raise ValueError('Bootstrap requires a positive integer draw count')
    rng, samples = np.random.default_rng(seed), []
    for offset in range(0, bootstrap, 256):
        index = rng.integers(values.shape[1], size=(min(256, bootstrap-offset), values.shape[1]))
        rmse = np.sqrt(values[:, index].mean(axis=2))
        samples.append(statistic(rmse))
    low, high = np.quantile(np.concatenate(samples), [.025, .975])
    return float(low), float(high)


def paired_difference(full, comparator, bootstrap=10000, seed=20261112):
    """Comparator minus full RMSE; positive means full is more accurate."""
    values = _paired_values([full, comparator])
    rmse = np.sqrt(values.mean(axis=1))
    low, high = _bootstrap(values, lambda v: v[1] - v[0], bootstrap, seed)
    return {'full_rmse': float(rmse[0]), 'comparator_rmse': float(rmse[1]),
            'comparator_minus_full_rmse': float(rmse[1] - rmse[0]),
            'ci95_low': low, 'ci95_high': high, 'n_families': values.shape[1]}


def difference_in_differences(full_complete, full_cell, comparator_complete,
                              comparator_cell, bootstrap=10000, seed=20261112):
    """Comparator degradation minus full degradation relative to complete100.

    Positive means full loses less accuracy under training-bone deletion; it
    does not assert that full has better absolute accuracy in that cell.
    """
    values = _paired_values([full_complete, full_cell, comparator_complete, comparator_cell])
    rmse = np.sqrt(values.mean(axis=1))
    statistic = lambda v: (v[3] - v[2]) - (v[1] - v[0])
    low, high = _bootstrap(values, statistic, bootstrap, seed)
    return {'full_complete_rmse': float(rmse[0]), 'full_cell_rmse': float(rmse[1]),
            'comparator_complete_rmse': float(rmse[2]), 'comparator_cell_rmse': float(rmse[3]),
            'full_rmse_increase': float(rmse[1] - rmse[0]),
            'comparator_rmse_increase': float(rmse[3] - rmse[2]),
            'comparator_minus_full_rmse_increase': float(statistic(rmse)),
            'ci95_low': low, 'ci95_high': high, 'n_families': values.shape[1]}


def comparisons(family, bootstrap=10000, seed=20261112):
    """Keep raw performance and robustness-to-deletion results separate."""
    raw, did = [], []
    for (refcase, target, support, reference_support), frame in family.groupby(
            ['refcase', 'target', 'support', 'reference_support']):
        lookup = {(cell, method): rows.set_index('family').mse
                  for (cell, method), rows in frame.groupby(['cell', 'method'])}
        for cell in CELLS:
            if (cell, 'full') not in lookup:
                continue
            for comparator in METHODS[1:]:
                if (cell, comparator) not in lookup:
                    raise ValueError('Missing comparator: ' + str((cell, comparator)))
                primary = (cell in PRIMARY_CELLS and refcase == 'observed'
                           and target == JOINT and reference_support == 'all'
                           and comparator in PRIMARY_COMPARATORS)
                meta = dict(cell=cell, refcase=refcase, target=target, support=support,
                            reference_support=reference_support, full='full', comparator=comparator,
                            primary_comparison=primary, interval_role='descriptive development 95% family interval')
                raw.append(dict(**meta, **paired_difference(lookup[cell, 'full'],
                                lookup[cell, comparator], bootstrap, seed)))
                if cell == 'complete100':
                    continue
                if ('complete100', 'full') not in lookup or ('complete100', comparator) not in lookup:
                    raise ValueError('Missing complete100 baseline for degradation comparison')
                did.append(dict(**meta, baseline_cell='complete100',
                    **difference_in_differences(lookup['complete100', 'full'], lookup[cell, 'full'],
                        lookup['complete100', comparator], lookup[cell, comparator], bootstrap, seed)))
    return pd.DataFrame(raw), pd.DataFrame(did)


def reference_comparisons(family, bootstrap=10000, seed=20261112):
    """Reference-case degradation relative to observed within the SAME cell."""
    rows = []
    for (cell, target, support, reference_support), frame in family.groupby(
            ['cell', 'target', 'support', 'reference_support']):
        lookup = {(case, method): part.set_index('family').mse
                  for (case, method), part in frame.groupby(['refcase', 'method'])}
        for refcase in REFCASES[1:]:
            if (refcase, 'full') not in lookup:
                continue
            for comparator in METHODS[1:]:
                needed = [('observed', 'full'), (refcase, 'full'),
                          ('observed', comparator), (refcase, comparator)]
                if any(key not in lookup for key in needed):
                    raise ValueError('Missing same-cell observed baseline or reference comparator')
                result = difference_in_differences(*(lookup[key] for key in needed),
                                                   bootstrap=bootstrap, seed=seed)
                # The numerical contrast is shared, but its baseline is reference
                # availability, not the complete100 training-label condition.
                renamed = {'full_complete_rmse': 'full_observed_rmse',
                           'full_cell_rmse': 'full_reference_case_rmse',
                           'comparator_complete_rmse': 'comparator_observed_rmse',
                           'comparator_cell_rmse': 'comparator_reference_case_rmse'}
                result = {renamed.get(key, key): value for key, value in result.items()}
                rows.append(dict(cell=cell, refcase=refcase, target=target, support=support,
                    reference_support=reference_support, full='full', comparator=comparator,
                    baseline_reference_case='observed', primary_comparison=False,
                    interval_role='descriptive development 95% family interval', **result))
    return pd.DataFrame(rows)


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _read_species(run, manifest):
    """Fail closed on unfinished jobs or unmatched frozen evaluation windows."""
    rows, canonical, job_keys = [], {}, set()
    jobs = manifest['jobs']
    if not jobs:
        raise ValueError('Manifest has no jobs')
    mode = manifest.get('mode')
    if mode not in ('full', 'smoke'):
        raise ValueError('Expected frozen full or smoke mode')
    expected_cells = CELLS if mode == 'full' else ('complete100', 'uniform25', 'family25')
    expected_folds = range(5) if mode == 'full' else (0,)
    expected_reps = (0, 1) if mode == 'full' else (0,)
    declared = []
    for job in jobs:
        if job['replicate'] not in expected_reps:
            raise ValueError('Unexpected repetition for run mode')
        if job['mask_seed'] != (101, 202)[job['replicate']] or job['seed'] != (13, 14)[job['replicate']]:
            raise ValueError('Mask/training seeds differ from frozen paired repetitions')
        methods = [job['variant']] if job['kind'] == 'neural' else ['bm', 'ou']
        declared.extend((method, job['condition'], job['fold'], job['replicate']) for method in methods)
    expected = {(m, c, f, r) for m in METHODS for c in expected_cells
                for f in expected_folds for r in expected_reps}
    if len(set(declared)) != len(declared) or set(declared) != expected:
        raise ValueError('Manifest does not contain the complete frozen method/cell/fold/repetition matrix')
    for job in jobs:
        if job['kind'] not in ('neural', 'classic'):
            raise ValueError('Unknown job kind')
        methods = [job['variant']] if job['kind'] == 'neural' else ['bm', 'ou']
        if any(m not in METHODS for m in methods):
            raise ValueError('Unknown neural variant')
        folder = run / 'evaluation' / job['name']
        result = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
        if result.get('status') != 'complete':
            raise ValueError('Incomplete evaluation job: ' + job['name'])
        prediction_path = folder / 'predictions.csv.gz'
        expected_hash = result.get('output_sha256', {}).get('predictions.csv.gz')
        if not expected_hash or expected_hash != _file_hash(prediction_path):
            raise ValueError('Prediction output SHA256 mismatch: ' + job['name'])
        frame = pd.read_csv(prediction_path)
        frame['model'] = frame.model.str.lower()
        expected = {(m, case) for m in methods for case in REFCASES}
        actual = set(map(tuple, frame[['model', 'reference_case']].drop_duplicates().to_numpy()))
        if expected != actual:
            raise ValueError('Missing or extra model/reference case: ' + job['name'])
        for field in ('condition', 'fold', 'replicate', 'mask_seed'):
            if not frame[field].eq(job[field]).all():
                raise ValueError('Prediction/manifest mismatch: ' + field)
        if job['kind'] == 'neural' and not frame.seed.eq(job['seed']).all():
            raise ValueError('Prediction/manifest mismatch: training seed')
        for (method, refcase), part in frame.groupby(['model', 'reference_case']):
            part = part.sort_values('window').reset_index(drop=True)
            if part.window.isna().any() or part.window.duplicated().any():
                raise ValueError('Missing or duplicate prediction window')
            part['has_reference_bones'] = _reference_flag(part.has_reference_bones)
            part['complete_three_bones'] = _reference_flag(part.complete_three_bones)
            observed = part[['observed_' + c for c in LENGTHS]].to_numpy(float)
            if np.isinf(observed).any():
                raise ValueError('Infinite observed length')
            if not np.array_equal(part.complete_three_bones.to_numpy(), np.isfinite(observed[:, 1:]).all(axis=1)):
                raise ValueError('complete_three_bones disagrees with fixed observed target')
            identity = ['window', 'target_row', 'target_tip', 'family',
                        'has_reference_bones', 'complete_three_bones',
                        'context_0_tip', 'context_1_tip', 'context_2_tip']
            if part[identity].isna().any().any():
                raise ValueError('Missing prediction-window identity')
            if job['fold'] not in canonical:
                canonical[job['fold']] = (part[identity].copy(), observed.copy())
            else:
                saved_identity, saved_observed = canonical[job['fold']]
                try:
                    pd.testing.assert_frame_equal(part[identity], saved_identity, check_dtype=False)
                    np.testing.assert_allclose(observed, saved_observed, rtol=0, atol=1e-12, equal_nan=True)
                except AssertionError as exc:
                    raise ValueError('Frozen target/reference support changed across methods, cells, cases or reps') from exc
            if refcase == 'observed':
                key = (method, job['condition'], job['fold'], job['replicate'])
                if key in job_keys:
                    raise ValueError('Duplicate model/fold/repetition/cell job')
                job_keys.add(key)
            predicted = part[['predicted_' + c for c in LENGTHS]].to_numpy(float)
            rows.append(species_errors(part, observed, predicted, method, job['fold'],
                        job['condition'], refcase, job['replicate'], job['mask_seed'],
                        job['seed'] if job['kind'] == 'neural' else -1))
    # Every declared cell/fold/repetition must include all five methods.
    for cell, fold, rep in {(c, f, r) for _, c, f, r in job_keys}:
        if {m for m, c, f, r in job_keys if (c, f, r) == (cell, fold, rep)} != set(METHODS):
            raise ValueError('Incomplete methods for a declared cell/fold/repetition')
    return pd.concat(rows, ignore_index=True)


def _output_names(frame):
    return frame.rename(columns={'cell': 'condition', 'refcase': 'reference_case'})


def summarize(run, bootstrap=10000):
    """Read frozen job predictions and persist every planned comparison."""
    run = Path(run)
    manifest = json.loads((run / 'manifest.json').read_text(encoding='utf-8'))
    species = _read_species(run, manifest)
    family_rep, family, metrics = aggregate(species)
    raw, did = comparisons(family, bootstrap=bootstrap)
    reference_did = reference_comparisons(family, bootstrap=bootstrap)
    raw['comparison_type'] = 'absolute_performance'
    raw['effect'] = raw.comparator_minus_full_rmse
    did['comparison_type'] = 'difference_in_differences'
    did['effect'] = did.comparator_minus_full_rmse_increase
    reference_did['comparison_type'] = 'reference_difference_in_differences'
    reference_did['effect'] = reference_did.comparator_minus_full_rmse_increase
    joined = pd.concat([raw, did, reference_did], ignore_index=True)
    scope = manifest['mode']
    pipeline_only = scope != 'full' or bool(manifest.get('local_pipeline_only', False))
    joined['pipeline_only'] = pipeline_only
    metrics['pipeline_only'] = pipeline_only
    _output_names(species).to_csv(run / 'species_errors.csv.gz', index=False, compression='gzip')
    _output_names(family_rep).to_csv(run / 'per_repetition_family_errors.csv', index=False)
    _output_names(family).to_csv(run / 'per_family_errors.csv', index=False)
    _output_names(metrics).to_csv(run / 'summary.csv', index=False)
    _output_names(joined).to_csv(run / 'comparisons.csv', index=False)
    summary = {
        'status': 'complete', 'mode': scope, 'pipeline_only': pipeline_only,
        'interpretation': ('Pipeline verification only; no scientific conclusion.' if pipeline_only else
                           'Development analysis; descriptive intervals, not confirmatory or noninferiority evidence.'),
        'weighting': 'window MSE -> species MSE -> family MSE -> repetition mean MSE -> equal-family RMSE',
        'target_support': 'external: all measured; every bone, total and ratio: fixed complete three bones',
        'reference_support': 'has_reference_bones is fixed original-reference support before experimental masking',
        'intervals': '95% paired family bootstrap conditional on fitted models, realized masks and frozen folds/tree',
        'effect_directions': {
            'absolute_performance': 'comparator RMSE minus full RMSE; positive means full is more accurate',
            'difference_in_differences': '(comparator cell - comparator complete100) - (full cell - full complete100); '
                                         'positive means full has the smaller error increase',
            'reference_difference_in_differences': '(comparator reference case - comparator observed) - '
                '(full reference case - full observed), within the same training cell; '
                'positive means full has the smaller increase under reference-bone masking'},
        'primary_definition': 'observed reference case, all reference windows, joint allocation target; '
                              'uniform25/family25 each versus OU/single/supervised; raw and degradation contrasts',
        'bootstrap': int(bootstrap), 'bootstrap_seed': 20261112,
        'n_jobs': len(manifest['jobs']), 'n_species_error_rows': len(species),
        'n_metric_rows': len(metrics), 'n_comparisons': len(joined),
        'n_primary_comparisons': int(joined.primary_comparison.sum()),
        'primary_comparisons': json.loads(_output_names(joined[joined.primary_comparison]).to_json(orient='records')),
        'all_results_saved': True,
    }
    (run / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--bootstrap', type=int, default=10000)
    args = parser.parse_args()
    summary = summarize(args.run, bootstrap=args.bootstrap)
    print(json.dumps({k: summary[k] for k in ('status', 'mode', 'pipeline_only', 'n_metric_rows', 'n_comparisons')}))


if __name__ == '__main__':
    main()
