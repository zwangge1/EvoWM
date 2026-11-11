"""Matched family-weighted comparisons of classical phylogenetic predictors.

Neural seeds are averaged as errors, not prediction ensembles. BM/OU are fitted
once per fold and are never duplicated as independent training replicates.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

LENGTHS = ('m_tarsus', 'b_fem', 'b_tbt', 'b_tmt')
TARGETS = (*LENGTHS, 'log10_total_leg', 'log10_tmt_over_fem',
           'log10_tbt_over_tmt', 'allocation_logratio_joint')


def coordinates(values):
    values = np.asarray(values, float)
    if values.ndim != 2 or values.shape[1] != 4 or np.isinf(values).any():
        raise ValueError('Expected four finite-or-missing log lengths')
    complete = np.isfinite(values[:, 1:]).all(axis=1)
    safe = np.where(complete[:, None], values[:, 1:], 0)
    total = np.logaddexp.reduce(safe * np.log(10), axis=1) / np.log(10)
    return dict(zip(LENGTHS, values.T)) | {
        'log10_total_leg': np.where(complete, total, np.nan),
        'log10_tmt_over_fem': values[:, 3] - values[:, 1],
        'log10_tbt_over_tmt': values[:, 2] - values[:, 3],
        'allocation_logratio_joint': values[:, 1:3] - values[:, 3:4],
    }


def species_errors(identity, observed, predicted, method, fold, training_seed=-1):
    predicted = np.asarray(predicted, float)
    if not np.isfinite(predicted).all():
        raise ValueError('A prediction is nonfinite: ' + method)
    obs, pred = coordinates(observed), coordinates(predicted)
    complete = np.isfinite(observed[:, 1:]).all(axis=1)
    records = []
    for target in TARGETS:
        error = pred[target] - obs[target]
        sq = np.mean(error**2, axis=1) if error.ndim == 2 else error**2
        keep = np.isfinite(sq) & (True if target == 'm_tarsus' else complete)
        frame = identity.loc[keep, ['target_tip', 'family']].copy()
        frame['mse'] = sq[keep]
        frame = frame.groupby(['target_tip', 'family'], as_index=False).agg(
            mse=('mse', 'mean'), n_windows=('mse', 'size'))
        frame['method'], frame['target'], frame['fold'] = method, target, fold
        frame['training_seed'] = training_seed
        frame['support'] = 'available' if target == 'm_tarsus' else 'complete_three_bones'
        records.append(frame)
    return pd.concat(records, ignore_index=True)


def aggregate(species):
    keys = ['method', 'target', 'support']
    if species.groupby(keys + ['training_seed', 'target_tip']).fold.nunique().gt(1).any():
        raise ValueError('Species occurs in multiple held-out folds')
    signatures = {}
    for (method, target, support, seed), part in species.groupby(keys + ['training_seed']):
        sig = set(map(tuple, part[['target_tip', 'family', 'fold']].to_numpy()))
        pair = (target, support)
        if pair in signatures and signatures[pair] != sig:
            raise ValueError('Methods or seeds have unmatched measured species support')
        signatures[pair] = sig
    family_seed = species.groupby(keys + ['training_seed', 'family'], as_index=False).mse.mean()
    family = family_seed.groupby(keys + ['family'], as_index=False).mse.mean()
    summary = family.groupby(keys, as_index=False).agg(mse=('mse', 'mean'), n_families=('family', 'nunique'))
    summary['rmse'] = np.sqrt(summary.mse)
    counts = species.groupby(keys, as_index=False).agg(n_species=('target_tip', 'nunique'),
        n_folds=('fold', 'nunique'), n_error_realizations=('training_seed', 'nunique'),
        n_training_seeds=('training_seed', lambda values: values[values >= 0].nunique()))
    fits = species[keys+['fold', 'training_seed']].drop_duplicates().groupby(keys, as_index=False).size()
    counts = counts.merge(fits.rename(columns={'size': 'n_underlying_fits'}), on=keys, validate='one_to_one')
    return family, summary.merge(counts, on=keys, validate='one_to_one')


def paired_difference(first, second, bootstrap=10000, seed=20261111):
    """second minus first; positive means first is more accurate."""
    if bootstrap < 1:
        raise ValueError('Bootstrap requires at least one draw')
    first, second = first.sort_index(), second.sort_index()
    if not first.index.equals(second.index) or not first.index.is_unique or not len(first):
        raise ValueError('Paired bootstrap requires identical nonempty family support')
    a, b = first.to_numpy(float), second.to_numpy(float)
    if not np.isfinite(np.r_[a, b]).all() or np.any(np.r_[a, b] < 0):
        raise ValueError('Invalid family MSE')
    rng, draws = np.random.default_rng(seed), []
    for offset in range(0, bootstrap, 256):
        idx = rng.integers(len(a), size=(min(256, bootstrap-offset), len(a)))
        draws.append(np.sqrt(b[idx].mean(axis=1)) - np.sqrt(a[idx].mean(axis=1)))
    quantiles = np.quantile(np.concatenate(draws), [.0125, .025, .975, .9875])
    return {'first_rmse': float(np.sqrt(a.mean())), 'second_rmse': float(np.sqrt(b.mean())),
            'second_minus_first_rmse': float(np.sqrt(b.mean()) - np.sqrt(a.mean())),
            'ci95_low': float(quantiles[1]), 'ci95_high': float(quantiles[2]),
            'ci975_low': float(quantiles[0]), 'ci975_high': float(quantiles[3]),
            'n_families': len(a)}


def summarize(run, manifest, root, bootstrap=10000):
    run, root = Path(run), Path(root)
    source = root / manifest['source_run']
    source_manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    rows, diagnostics = [], []
    for fold in manifest['folds']:
        for model in ('bm', 'ou'):
            fit = json.loads((run/'folds'/f'fold_{fold}'/(model+'_fit.json')).read_text(encoding='utf-8'))
            diagnostics.append(dict(fold=fold, method=model, alpha=fit['alpha'], eta=fit['eta'],
                loglik=fit['loglik'], aic=fit['aic'], n_training=fit['n_morph'], n_training_bones=fit['n_bone'],
                n_predictors=fit['p'], convergence=fit['convergence']['kind'],
                failed_optimizer_count=fit['convergence']['failed_optimizer_count'],
                refinement_attempted=fit['convergence']['refinement_attempted'],
                alpha_location=fit['convergence']['parameter_boundaries'].get('alpha', 'fixed_zero'),
                eta_location=fit['convergence']['parameter_boundaries'].get('eta', 'fixed')))
        baseline = pd.read_csv(run / 'folds' / f'fold_{fold}' / 'predictions.csv.gz')
        observed = baseline[['observed_' + c for c in LENGTHS]].to_numpy(float)
        identities = ['window', 'target_tip', 'source_tip', 'family',
                      'context_0_tip', 'context_1_tip', 'context_2_tip']
        for method in ('bm', 'ou', 'bm_mean', 'ou_mean'):
            rows.append(species_errors(baseline, observed,
                baseline[[method + '_' + c for c in LENGTHS]], method, fold))
        ridge = None
        jobs = [j for j in source_manifest['jobs'] if j['stage'] == 'real_development' and j['fold'] == fold]
        for job in jobs:
            path = source / 'evaluation' / job['name'] / 'prediction_windows.csv.gz'
            saved = pd.read_csv(path, usecols=identities + ['observed_'+c for c in LENGTHS]
                                + [m+'_'+c for m in ('jepa_full', 'ridge_full') for c in LENGTHS])
            pd.testing.assert_frame_equal(saved[identities], baseline[identities], check_dtype=False)
            np.testing.assert_allclose(saved[['observed_'+c for c in LENGTHS]], observed,
                                       equal_nan=True, atol=1e-12, rtol=0)
            rows.append(species_errors(baseline, observed, saved[['jepa_full_'+c for c in LENGTHS]],
                                      'jepa_'+job['variant'], fold, job['seed']))
            value = saved[['ridge_full_'+c for c in LENGTHS]].to_numpy(float)
            if ridge is None:
                ridge = value
            elif not np.allclose(ridge, value, atol=1e-10, rtol=0):
                raise ValueError('Original deterministic Ridge predictions differ across source jobs')
        rows.append(species_errors(baseline, observed, ridge, 'ridge_existing', fold))
    species = pd.concat(rows, ignore_index=True)
    family, metrics = aggregate(species)
    comparisons = []
    for (target, support), data in family.groupby(['target', 'support']):
        lookup = {m: v.set_index('family').mse for m, v in data.groupby('method')}
        for method in lookup:
            if method == 'jepa_full':
                continue
            result = paired_difference(lookup['jepa_full'], lookup[method], bootstrap)
            primary = method in ('bm', 'ou') and target == 'allocation_logratio_joint'
            comparisons.append(dict(first='jepa_full', second=method, target=target, support=support,
                                    primary_comparison=primary,
                                    interval_role='97.5% interval for two primary comparisons' if primary else 'descriptive 95% interval',
                                    **result))
    species.to_csv(run/'species_errors.csv.gz', index=False, compression='gzip')
    family.to_csv(run/'family_errors.csv', index=False)
    metrics.to_csv(run/'metrics.csv', index=False)
    pd.DataFrame(comparisons).to_csv(run/'comparisons.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(run/'fit_diagnostics.csv', index=False)
    report = ['# BM OU conditional prediction comparison', '',
              'Positive comparator-minus-EvoWM RMSE means EvoWM is more accurate. '
              'BM/OU use the same three reference species, a common four-trait covariance and fixed-root evolutionary kernels. '
              'All covariance and mean parameters are trained outside held-out families.', '',
              'Neural seeds are averaged at family MSE level; deterministic phylogenetic fits are not duplicated. '
              'Bootstrap intervals condition on fitted models and the frozen tree.', '',
              f'Scope: {manifest["mode"]}; folds: {manifest["folds"]}.', '',
              '| Method | Joint bone ratios RMSE | Total leg RMSE | External tarsus RMSE |',
              '|---|---:|---:|---:|']
    for method, data in metrics.groupby('method'):
        scores = data.set_index('target').rmse
        report.append(f'| {method} | {scores["allocation_logratio_joint"]:.6f} | '
                      f'{scores["log10_total_leg"]:.6f} | {scores["m_tarsus"]:.6f} |')
    report += ['', '| Primary comparison | Comparator minus full RMSE | 97.5% family interval |',
               '|---|---:|---|']
    for row in comparisons:
        if row['primary_comparison']:
            report.append(f'| {row["second"]} | {row["second_minus_first_rmse"]:+.6f} | '
                          f'[{row["ci975_low"]:+.6f}, {row["ci975_high"]:+.6f}] |')
    report += ['', 'Optimization diagnostics (alpha is per normalized full-tree height):', '',
               '| Fold | Model | alpha | eta | alpha location | eta location | Failed starts | Refinement |',
               '|---|---|---:|---:|---|---|---:|---|']
    for row in diagnostics:
        report.append(f'| {row["fold"]} | {row["method"]} | {row["alpha"]:.6g} | {row["eta"]:.6g} | '
            f'{row["alpha_location"]} | {row["eta_location"]} | {row["failed_optimizer_count"]} | {row["refinement_attempted"]} |')
    if any('search_upper' in (row['alpha_location'], row['eta_location']) for row in diagnostics):
        report += ['', 'At least one fitted parameter reaches the declared search upper bound; '
                   'these predictions describe the bounded fit, not an established unconstrained optimum.']
    (run/'report.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
    return {'n_species_error_rows': len(species), 'n_metric_rows': len(metrics),
            'n_comparisons': len(comparisons), 'bootstrap': bootstrap}
