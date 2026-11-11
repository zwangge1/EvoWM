"""Paired summaries of frozen-checkpoint Allen anatomical diagnostics.

No fit, model selection, causal test, equivalence or noninferiority test is run.
Oracle substitutions use target truth for error decomposition only. A substitution
can worsen error by removing cancellation; these are not attainable error bounds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LENGTHS = ('m_tarsus', 'b_fem', 'b_tbt', 'b_tmt')
BONES = LENGTHS[1:]
TARGETS = (*LENGTHS, 'log10_total_leg', 'log10_tmt_over_fem',
           'log10_tbt_over_tmt', 'allocation_logratio_joint')
VARIANTS = {'full', 'no_link', 'scalar', 'supervised', 'single'}
META = ['stage', 'scenario', 'variant', 'seed', 'fold']
GROUP = ['stage', 'scenario', 'variant', 'method', 'target', 'support']
COMPONENTS = ['total_mse', 'proportion_mse', 'cross_term', 'bone_mse']


def coordinate_values(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 4 or np.isinf(values).any():
        raise ValueError('Expected N by four finite-or-missing log lengths')
    complete = np.isfinite(values[:, 1:]).all(axis=1)
    safe = np.where(complete[:, None], values[:, 1:], 0.)
    total = np.logaddexp.reduce(safe * np.log(10.), axis=1) / np.log(10.)
    total = np.where(complete, total, np.nan)
    result = dict(zip(LENGTHS, values.T))
    result.update(log10_total_leg=total, log10_tmt_over_fem=values[:, 3]-values[:, 1],
                  log10_tbt_over_tmt=values[:, 2]-values[:, 3],
                  allocation_logratio_joint=np.column_stack((values[:, 1]-values[:, 3],
                                                            values[:, 2]-values[:, 3])))
    return result


def boolean(values, label):
    out = values.astype(str).str.lower().map({'true': True, 'false': False, '1': True, '0': False})
    if out.isna().any():
        raise ValueError(f'Invalid boolean flag: {label}')
    return out.to_numpy(bool)


def check_windows(data, methods):
    identity = ['window', 'target_tip', 'source_tip', 'family']
    contexts = sorted(c for c in data if re.fullmatch(r'context_\d+_tip', c))
    if not contexts:
        raise ValueError('Missing reference context identity columns')
    identity += contexts
    required = identity + ['complete_three_bones'] + ['observed_' + c for c in LENGTHS]
    required += [m + '_' + c for m in methods for c in LENGTHS]
    missing = set(required) - set(data)
    if missing:
        raise ValueError(f'Missing diagnostic columns: {sorted(missing)}')
    if data.empty or data.window.duplicated().any():
        raise ValueError('Prediction windows must be nonempty and uniquely identified')
    for col in identity:
        if data[col].isna().any() or data[col].astype(str).eq('').any():
            raise ValueError(f'Missing window identity: {col}')
    observed = data[['observed_' + c for c in LENGTHS]].to_numpy(float)
    if np.isinf(observed).any():
        raise ValueError('Infinite observed length')
    complete = np.isfinite(observed[:, 1:]).all(axis=1)
    if not np.array_equal(complete, boolean(data.complete_three_bones, 'complete_three_bones')):
        raise ValueError('Complete-bone flag disagrees with observed triplets')
    for method in methods:
        if not np.isfinite(data[[method + '_' + c for c in LENGTHS]].to_numpy(float)).all():
            raise ValueError(f'Nonfinite prediction: {method}')
    return identity, observed, complete


def error_tables(data, methods, job):
    """Collapse repeated windows within species before any family weighting."""
    _, observed, complete = check_windows(data, methods)
    obs = coordinate_values(observed)
    predictions = {m: data[[m + '_' + c for c in LENGTHS]].to_numpy(float) for m in methods}
    baseline = predictions['baseline']
    pred_total = coordinate_values(baseline)['log10_total_leg']
    obs_total = obs['log10_total_leg']
    # Explicitly unavailable on incomplete triplets; never feed truth to a model.
    oracle_total, oracle_allocation = baseline.copy(), baseline.copy()
    oracle_total[:, 1:] = baseline[:, 1:] - pred_total[:, None] + obs_total[:, None]
    oracle_allocation[:, 1:] = observed[:, 1:] - obs_total[:, None] + pred_total[:, None]
    predictions.update(oracle_total=oracle_total, oracle_allocation=oracle_allocation)
    records = []
    for method, lengths in predictions.items():
        values = coordinate_values(lengths)
        for target in TARGETS:
            if method.startswith('oracle_') and target == 'm_tarsus':
                continue
            support = 'available' if target == 'm_tarsus' else 'complete_three_bones'
            err = values[target] - obs[target]
            squared = np.mean(err**2, axis=1) if err.ndim == 2 else err**2
            selected = np.isfinite(squared)
            if support == 'complete_three_bones':
                selected &= complete
            x = data.loc[selected, ['target_tip', 'family']].copy()
            x['mse'] = squared[selected]
            x = x.groupby(['target_tip', 'family'], sort=True).mse.agg(['mean', 'size']).reset_index()
            x = x.rename(columns={'mean': 'mse', 'size': 'n_windows'})
            for field in META:
                x[field] = job[field]
            x['method'], x['target'], x['support'] = method, target, support
            records.append(x)
    species = pd.concat(records, ignore_index=True)
    repeats = sorted(m for m in methods if re.fullmatch(r'shuffle_rich_\d+', m))
    if repeats:
        # Average intervention MSEs, not predictions; repeats are not new models.
        sh = species[species.method.isin(repeats)]
        key = META + ['target', 'support', 'target_tip', 'family']
        if not sh.groupby(key).method.nunique().eq(len(repeats)).all():
            raise ValueError('Unequal permutation support')
        sh = sh.groupby(key, as_index=False).agg(mse=('mse', 'mean'), n_windows=('n_windows', 'first'))
        sh['method'] = 'shuffle_rich_average'
        species = pd.concat([species, sh], ignore_index=True)
    terms = []
    total_error = pred_total - obs_total
    for i, bone in enumerate(BONES, 1):
        bone_error = baseline[:, i] - observed[:, i]
        proportion_error = bone_error - total_error
        vals = np.column_stack((total_error**2, proportion_error**2,
                                2*total_error*proportion_error, bone_error**2))
        if not np.allclose(vals[complete, :3].sum(axis=1), vals[complete, 3], atol=1e-12, rtol=1e-10):
            raise ValueError('Bone error decomposition identity failed')
        x = data.loc[complete, ['target_tip', 'family']].copy()
        x[COMPONENTS] = vals[complete]
        x = x.groupby(['target_tip', 'family'], as_index=False)[COMPONENTS].mean()
        for field in META:
            x[field] = job[field]
        x['bone'] = bone
        terms.append(x)
    return species, pd.concat(terms, ignore_index=True)


def validate_species_support(species, expected_seeds):
    grouping = ['stage', 'scenario', 'variant', 'seed', 'method', 'target', 'support']
    if species.groupby(grouping + ['target_tip']).fold.nunique().gt(1).any():
        raise ValueError('A target species occurs in multiple held-out folds')
    if species.groupby(['stage', 'scenario', 'target_tip']).family.nunique().gt(1).any():
        raise ValueError('Conflicting target family labels')
    if species.groupby(['stage', 'scenario', 'seed', 'family']).fold.nunique().gt(1).any():
        raise ValueError('A family occurs in multiple held-out folds')
    # Pair exact species/family/fold supports across all methods and all seeds.
    supports = {}
    for key, frame in species.groupby(grouping, sort=True):
        scope = (key[0], key[1], key[5], key[6])
        signature = set(map(tuple, frame[['target_tip', 'family', 'fold']].to_numpy()))
        if scope in supports and signature != supports[scope]:
            raise ValueError('Unmatched species support across variants/methods/seeds')
        supports[scope] = signature
    for key, frame in species.groupby(GROUP):
        desired = expected_seeds[(key[0], key[1], key[2])]
        if set(frame.seed) != desired:
            raise ValueError(f'Missing configured seed support: {key}')


def aggregate_species(species):
    family = species.groupby(GROUP + ['seed', 'family'], as_index=False).mse.mean()
    pooled = family.groupby(GROUP + ['family'], as_index=False).mse.mean()
    summary = pooled.groupby(GROUP, as_index=False).agg(mse=('mse', 'mean'), n_families=('family', 'nunique'))
    summary['rmse'] = np.sqrt(summary.mse)
    counts = species.groupby(GROUP, as_index=False).agg(n_species=('target_tip', 'nunique'), n_seeds=('seed', 'nunique'))
    summary = summary.merge(counts, on=GROUP, validate='one_to_one')
    summary['uses_target_truth'] = summary.method.str.startswith('oracle_')
    per_seed = family.groupby(GROUP + ['seed'], as_index=False).agg(mse=('mse', 'mean'), n_families=('family', 'nunique'))
    per_seed['rmse'] = np.sqrt(per_seed.mse)
    return family, pooled, summary, per_seed


def paired_delta(baseline, alternative, bootstrap=10000, seed=20261111):
    """Positive alternative-minus-baseline means the baseline has less error."""
    if bootstrap < 1:
        raise ValueError('bootstrap must be positive')
    baseline, alternative = baseline.sort_index(), alternative.sort_index()
    if not baseline.index.equals(alternative.index) or not baseline.index.is_unique or not len(baseline):
        raise ValueError('Paired comparison requires identical unique family support')
    a, b = baseline.to_numpy(float), alternative.to_numpy(float)
    if not np.isfinite(np.r_[a, b]).all() or np.any(np.r_[a, b] < 0):
        raise ValueError('Family MSE must be finite and nonnegative')
    rng = np.random.default_rng(seed)
    # Bound memory for larger taxon sets while maintaining paired resampling.
    sampled = []
    for offset in range(0, bootstrap, 512):
        indices = rng.integers(0, len(a), size=(min(512, bootstrap-offset), len(a)))
        sampled.append(np.sqrt(b[indices].mean(axis=1)) - np.sqrt(a[indices].mean(axis=1)))
    low, high = np.quantile(np.concatenate(sampled), [.025, .975])
    base_rmse, alt_rmse = float(np.sqrt(a.mean())), float(np.sqrt(b.mean()))
    return {'baseline_rmse': base_rmse, 'alternative_rmse': alt_rmse,
            'alternative_minus_baseline_rmse': alt_rmse-base_rmse,
            'ci_low': float(low), 'ci_high': float(high), 'n_families': len(a)}


def comparison_table(pooled, bootstrap):
    rows = []
    for (stage, scenario, target, support), group in pooled.groupby(['stage', 'scenario', 'target', 'support']):
        lookup = {(v, m): x.set_index('family').mse for (v, m), x in group.groupby(['variant', 'method'])}
        comparisons = []
        for (variant, method) in lookup:
            if method != 'baseline':
                kind = 'truth_substitution' if method.startswith('oracle_') else 'frozen_checkpoint_intervention'
                comparisons.append((variant, 'baseline', variant, method, kind))
            elif variant != 'full' and ('full', 'baseline') in lookup:
                comparisons.append(('full', 'baseline', variant, method, 'separately_trained_architecture'))
        for bv, bm, av, am, kind in comparisons:
            row = dict(stage=stage, scenario=scenario, target=target, support=support,
                       baseline_variant=bv, baseline_method=bm, alternative_variant=av,
                       alternative_method=am, comparison_type=kind)
            row.update(paired_delta(lookup[bv, bm], lookup[av, am], bootstrap))
            rows.append(row)
    return pd.DataFrame(rows)


def checkpoint_tables(reports):
    """Flatten local diagnostics without treating their rows as replicates."""
    gradients, internal = [], []
    for report in reports:
        job = report['job']
        meta = {k: job[k] for k in ['name'] + META}
        grad = report.get('gradient_diagnostics', {})
        status = grad.get('status', 'not_recorded')
        scope = grad.get('scope', '')
        groups = grad.get('parameter_groups', {})
        if not groups:
            gradients.append(meta | {'status': status, 'paramgroup': None, 'loss': None,
                                      'value': None, 'norm': None, 'cosine': None,
                                      'n_complete': grad.get('n_complete'),
                                      'n_windows': grad.get('n_windows'), 'scope': scope})
        for group, values in groups.items():
            for loss, norm in values.get('gradient_norms', {}).items():
                gradients.append(meta | {'status': status, 'paramgroup': group, 'loss': loss,
                    'value': grad['losses'].get(loss), 'norm': norm,
                    'cosine': values.get('cosine_with_supervised_sum', {}).get(loss),
                    'n_complete': grad.get('n_complete'), 'n_windows': grad.get('n_windows'), 'scope': scope})
        for split, values in report.get('upper_latent_statistics', {}).items():
            for metric, value in values.items():
                if metric == 'note':
                    continue
                if isinstance(value, list):
                    for dimension, element in enumerate(value):
                        internal.append(meta | {'category': 'upper_latent', 'split': split,
                            'method': None, 'metric': metric, 'dimension': dimension, 'value': element})
                elif isinstance(value, (int, float)):
                    internal.append(meta | {'category': 'upper_latent', 'split': split,
                        'method': None, 'metric': metric, 'dimension': None, 'value': value})
        for method, value in report.get('maximum_absolute_prediction_change', {}).items():
            internal.append(meta | {'category': 'prediction_change', 'split': 'held_out',
                'method': method, 'metric': 'maximum_absolute_change_log10', 'dimension': None, 'value': value})
        for metric, value in report.get('adapter_first_layer_weight_norms', {}).items():
            if isinstance(value, (int, float)):
                internal.append(meta | {'category': 'adapter_weight_norm', 'split': 'checkpoint',
                    'method': None, 'metric': metric, 'dimension': None, 'value': value})
        internal.append(meta | {'category': 'verification', 'split': 'held_out', 'method': 'baseline',
            'metric': 'maximum_absolute_baseline_reproduction_difference', 'dimension': None,
            'value': report['baseline_max_absolute_difference']})
    return pd.DataFrame(gradients), pd.DataFrame(internal)


def summarize(run, *, bootstrap=10000, root=ROOT):
    run = Path(run)
    if not run.is_absolute():
        run = Path(root) / run
    run = run.resolve()
    manifest = json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    jobs = manifest['jobs']
    if not jobs or len({j['name'] for j in jobs}) != len(jobs):
        raise ValueError('Manifest must contain uniquely named configured jobs')
    species_frames, terms_frames, identities, expected_seeds, method_sets = [], [], {}, {}, {}
    checkpoint_reports = []
    for job in jobs:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', job['name']):
            raise ValueError('Invalid diagnostic job name')
        folder = run / 'jobs' / job['name']
        missing = [f for f in ('diagnosis.json', 'diagnostic_windows.csv.gz') if not (folder/f).is_file()]
        if missing:
            raise ValueError(f'Missing configured diagnostic output: {job["name"]}: {missing}')
        report = json.loads((folder/'diagnosis.json').read_text(encoding='utf-8'))
        if any(report['job'].get(k) != job[k] for k in ['name'] + META):
            raise ValueError('Diagnostic identity disagrees with manifest: ' + job['name'])
        reproduced_error = report.get('baseline_max_absolute_difference')
        if not isinstance(reproduced_error, (int, float)) or not np.isfinite(reproduced_error) or not 0 <= reproduced_error <= 1e-5:
            raise ValueError('Missing or failed original-baseline reproduction: ' + job['name'])
        checkpoint_reports.append(report)
        methods = report['prediction_methods']
        if ('baseline' not in methods or len(set(methods)) != len(methods)
                or any(not re.fullmatch(r'[a-z][a-z0-9_]*', m) or m.startswith('oracle_') for m in methods)):
            raise ValueError('Invalid/duplicate prediction methods')
        mode_key = (job['stage'], job['scenario'], job['variant'])
        if mode_key in method_sets and set(methods) != method_sets[mode_key]:
            raise ValueError('Inconsistent intervention methods across configured jobs')
        method_sets[mode_key] = set(methods)
        expected_seeds.setdefault(mode_key, set()).add(job['seed'])
        data = pd.read_csv(folder/'diagnostic_windows.csv.gz', keep_default_na=False, na_values=[''])
        identity, _, _ = check_windows(data, methods)
        signature = data[identity + ['complete_three_bones'] + ['observed_' + c for c in LENGTHS]]
        # Reference contexts are identical across architectures and training seeds.
        key = (job['stage'], job['scenario'], job['fold'])
        if key in identities:
            try:
                pd.testing.assert_frame_equal(identities[key], signature, check_dtype=False, check_exact=True)
            except AssertionError as exc:
                raise ValueError('Unmatched window identity or observed support: ' + job['name']) from exc
        else:
            identities[key] = signature
        sp, terms = error_tables(data, methods, job)
        species_frames.append(sp)
        terms_frames.append(terms)
    species = pd.concat(species_frames, ignore_index=True)
    validate_species_support(species, expected_seeds)
    family, pooled, summary, per_seed = aggregate_species(species)
    paired = comparison_table(pooled, bootstrap)
    gradient_summary, internal_summary = checkpoint_tables(checkpoint_reports)
    terms = pd.concat(terms_frames, ignore_index=True)
    dkeys = ['stage', 'scenario', 'variant', 'bone']
    terms_family = terms.groupby(dkeys + ['seed', 'family'], as_index=False)[COMPONENTS].mean()
    terms_pooled = terms_family.groupby(dkeys + ['family'], as_index=False)[COMPONENTS].mean()
    decomp = terms_pooled.groupby(dkeys, as_index=False)[COMPONENTS].mean()
    decomp['identity_residual'] = decomp.total_mse + decomp.proportion_mse + decomp.cross_term - decomp.bone_mse
    if not np.allclose(decomp.identity_residual, 0., atol=1e-12):
        raise ValueError('Aggregated decomposition identity failed')
    planned = {(j['stage'], j['scenario'], j['variant'], int(j['seed']), int(j['fold'])) for j in jobs}
    if len(planned) != len(jobs):
        raise ValueError('Duplicate scientific job configuration')
    complete_matrix = True
    for stage, scenario in {(j['stage'], j['scenario']) for j in jobs}:
        group = [j for j in jobs if (j['stage'], j['scenario']) == (stage, scenario)]
        seeds = {j['seed'] for j in group}
        complete_matrix &= len(seeds) >= 2 and all((stage, scenario, v, s, f) in planned
                                                 for v in VARIANTS for s in seeds for f in range(5))
    result = {
        'all_configured_jobs_complete': True, 'completed_jobs': len(jobs),
        'complete_five_variant_two_seed_five_fold_matrix': bool(complete_matrix),
        'scope': 'Fixed-checkpoint descriptive diagnosis. Smoke/subsets are workflow checks, not evidence of architecture superiority.',
        'aggregation': 'Window squared errors averaged within species, species within family; pool held-out folds, average family MSE across trained seeds, sqrt only last. Not an ensemble of averaged predictions.',
        'shuffle_aggregation': 'shuffle_rich_average averages repeat MSEs within species before family and seed averaging. Permutation repeats are not independent models.',
        'oracle_interpretation': 'oracle_total substitutes observed total with baseline proportions; oracle_allocation substitutes observed proportions with baseline total. Target-truth diagnostics only, not deployable predictions or attainable bounds; removing error cancellation can worsen error.',
        'decomposition': 'bone_MSE = total_MSE + logproportion_MSE + signed_cross_term, with identical species/family/seed weights.',
        'uncertainty': 'Paired family percentile bootstrap conditional on fitted models, not retraining uncertainty, not phylogenetic-tree uncertainty, not multiplicity adjusted, not an equivalence or noninferiority test.',
        'difference_sign': 'Positive alternative_minus_baseline_rmse means baseline has lower error. A frozen intervention is distinct from separately retraining an ablated model.',
        'internal_diagnostics': 'Gradient rows are local training-window diagnostics and omit SIGReg/other mask modes/earlier positions; do not infer complete-objective conflict or causality. Raw weight norms and latent variance are not feature importance. See the preserved per-job JSON scope.',
        'bootstrap_replicates': int(bootstrap), 'bootstrap_seed': 20261111,
        'identity_max_absolute_residual': float(np.abs(decomp.identity_residual).max()) if len(decomp) else None,
        'files': ['summary.csv', 'seed_summary.csv', 'paired_effects.csv', 'family_mse.csv.gz',
                  'decomposition.csv', 'decomposition_family.csv.gz', 'gradient_summary.csv', 'internal_summary.csv'],
    }
    summary.to_csv(run/'summary.csv', index=False)
    per_seed.to_csv(run/'seed_summary.csv', index=False)
    paired.to_csv(run/'paired_effects.csv', index=False)
    family.to_csv(run/'family_mse.csv.gz', index=False, compression='gzip')
    decomp.to_csv(run/'decomposition.csv', index=False)
    terms_family.to_csv(run/'decomposition_family.csv.gz', index=False, compression='gzip')
    gradient_summary.to_csv(run/'gradient_summary.csv', index=False)
    internal_summary.to_csv(run/'internal_summary.csv', index=False)
    (run/'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--bootstrap', type=int, default=10000)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error('--bootstrap must be positive')
    print(json.dumps(summarize(args.run, bootstrap=args.bootstrap, root=args.root), ensure_ascii=False))


if __name__ == '__main__':
    main()
