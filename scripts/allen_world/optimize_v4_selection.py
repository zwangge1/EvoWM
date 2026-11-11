"""Inner-only configuration selection and matched outer evaluation for Allen v4.

The selector reads only explicitly declared tune jobs. Outer observations,
previous neural performance, and BM/OU performance never enter selection.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from summarize_phylo_baselines import LENGTHS, TARGETS, aggregate, paired_difference, species_errors

SELECTION_TARGETS = (*LENGTHS, 'log10_total_leg', 'allocation_logratio_joint')
OUTER_VARIANTS = ('full', 'single', 'supervised')
RATIO_LIMIT = 1.03


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def _child(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError('Expected evidence strictly within its declared run: ' + str(path))
    return path


def _scope(manifest):
    mode = manifest.get('mode', 'full')
    if mode == 'full':
        return list(range(5)), [0, 1], [13, 14]
    if mode == 'smoke':
        return [0], [0], [13]
    raise ValueError('Mode must be full or smoke')


def _candidate_spec(manifest):
    candidates = manifest['candidates']
    if not isinstance(candidates, list) or not candidates:
        raise ValueError('A fixed nonempty candidate list is required')
    names = [c['name'] for c in candidates]
    if len(set(names)) != len(names) or any(not re.fullmatch(r'[A-Za-z0-9_]+', n) for n in names):
        raise ValueError('Candidate names must be unique safe identifiers')
    reference = manifest.get('reference_candidate', 'reference')
    if reference not in names:
        raise ValueError('Reference candidate is absent from the fixed candidate list')
    return candidates, names, reference


def _plan_digest(manifest):
    allowed = {key: manifest.get(key) for key in ('mode', 'candidates', 'reference_candidate', 'tune_jobs')}
    return hashlib.sha256(json.dumps(allowed, sort_keys=True, separators=(',', ':'), default=str).encode('utf-8')).hexdigest()


def _evaluation(run, name, evidence, *, prefix='jepa_full'):
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_]+', name):
        raise ValueError('Unsafe evaluation job name')
    folder = _child(Path(run) / 'evaluation' / name, run)
    result_path, raw_path = folder / 'pilot_results.json', folder / 'prediction_windows.csv.gz'
    for path in (result_path, raw_path):
        if not path.is_file() or not path.stat().st_size:
            raise ValueError('Missing complete evaluation evidence: ' + str(path))
        evidence[path.relative_to(run).as_posix()] = _sha(path)
    result, frame = _json(result_path), pd.read_csv(raw_path)
    if int(result.get('history', -1)) != 3:
        raise ValueError('Selection requires the original three-reference prediction task')
    if result.get('n_windows', len(frame)) != len(frame):
        raise ValueError('Saved evaluation window count disagrees with its CSV')
    required = {'window', 'target_row', 'source_row', 'target_tip', 'source_tip', 'family',
                'context_0_tip', 'context_1_tip', 'context_2_tip', 'complete_three_bones',
                *['observed_' + c for c in LENGTHS], *[prefix + '_' + c for c in LENGTHS]}
    if not required <= set(frame):
        raise ValueError('Evaluation lacks required identities, observations, or predictions')
    if not np.array_equal(frame.window.to_numpy(), np.arange(len(frame))):
        raise ValueError('Evaluation window identifiers must retain their original order')
    contexts = frame[['context_0_tip', 'context_1_tip', 'context_2_tip']].to_numpy()
    if (contexts == frame.target_tip.to_numpy()[:, None]).any():
        raise ValueError('Target appears among reference species')
    if not np.array_equal(frame.source_tip.to_numpy(), contexts[:, -1]):
        raise ValueError('Final reference and source tip disagree')
    observed = frame[['observed_' + c for c in LENGTHS]].to_numpy(float)
    if not np.array_equal(frame.complete_three_bones.to_numpy(bool), np.isfinite(observed[:, 1:]).all(axis=1)):
        raise ValueError('Complete-bone observation mask differs from observed values')
    for c in LENGTHS:
        mask_column = 'observed_mask_' + c
        if mask_column in frame and not np.array_equal(frame[mask_column].to_numpy(bool), np.isfinite(frame['observed_' + c])):
            raise ValueError('Saved observation mask disagrees with its values')
    return result, frame


def _pair(first, second, *, exact_observations=True):
    identities = ['window', 'target_row', 'source_row', 'target_tip', 'source_tip', 'family',
                  'context_0_tip', 'context_1_tip', 'context_2_tip', 'complete_three_bones']
    observations = sorted(c for c in first if c.startswith('observed_'))
    if observations != sorted(c for c in second if c.startswith('observed_')):
        raise ValueError('Compared evaluations have different saved observation schemas')
    pd.testing.assert_frame_equal(first[identities], second[identities], check_dtype=False, check_exact=True)
    if exact_observations:
        pd.testing.assert_frame_equal(first[observations], second[observations], check_dtype=False, check_exact=True)
    else:
        masks = [c for c in observations if c.startswith('observed_mask_')]
        values = [c for c in observations if c not in masks]
        pd.testing.assert_frame_equal(first[masks], second[masks], check_dtype=False, check_exact=True)
        np.testing.assert_allclose(first[values], second[values], atol=1e-12, rtol=0, equal_nan=True)


def _sampling_signature(result):
    provenance = result.get('provenance', {})
    return {'history': result['history'], 'n_orders': result.get('n_orders'), 'evaluation_seed': result.get('seed'),
            **{key: provenance.get(key) for key in ('table_sha256', 'table_metadata_sha256', 'tree_sha256')}}


def _errors(frame, method, fold, seed, prefix='jepa_full'):
    observed = frame[['observed_' + c for c in LENGTHS]].to_numpy(float)
    predicted = frame[[prefix + '_' + c for c in LENGTHS]].to_numpy(float)
    return species_errors(frame, observed, predicted, method, int(fold), int(seed))


def select_candidates(run, manifest, root):
    """Select one configuration per outer fold, exclusively from inner evidence.

    ``selected_by_outer`` has string keys so in-memory and JSON APIs agree.
    A reference-relative cap applies to all six metrics; the score is their
    unweighted mean relative MSE. Ties use actual training steps then fixed order.
    """
    del root  # Intentionally unavailable to evidence reads in this function.
    run = Path(run).resolve()
    candidates, names, reference = _candidate_spec(manifest)
    outer_folds, inner_folds, seeds = _scope(manifest)
    jobs = manifest['tune_jobs']
    expected = {(outer, candidate, inner, seed) for outer in outer_folds
                for candidate in names for inner in inner_folds for seed in seeds}
    actual, by_key, job_names = [], {}, []
    for job in jobs:
        if job.get('stage') not in ('inner', 'tune') or job.get('variant') != 'full':
            raise ValueError('Selection may only read full-architecture inner/tune jobs')
        inner = int(job.get('inner_fold', job['fold']))
        if inner != int(job['fold']):
            raise ValueError('fold and inner_fold disagree')
        key = (int(job['outer_fold']), job['candidate'], inner, int(job['seed']))
        actual.append(key)
        by_key[key] = job
        job_names.append(job['name'])
    if len(actual) != len(expected) or set(actual) != expected or len(set(job_names)) != len(job_names):
        raise ValueError('Incomplete or duplicated inner candidate/fold/seed matrix; selection refused')
    evidence, identity_audit, scores, selected = {}, [], [], {}
    for outer in outer_folds:
        error_frames, canonicals, signatures = [], {}, {}
        for candidate in names:
            for inner in inner_folds:
                for seed in seeds:
                    job = by_key[outer, candidate, inner, seed]
                    result, frame = _evaluation(run, job['name'], evidence)
                    signature = _sampling_signature(result)
                    if inner not in canonicals:
                        canonicals[inner], signatures[inner] = frame, signature
                    else:
                        _pair(canonicals[inner], frame)
                        if signature != signatures[inner]:
                            raise ValueError('Inner sampling/input provenance differs across candidates or seeds')
                    error_frames.append(_errors(frame, candidate, inner, seed))
            job_steps = {int(by_key[outer, candidate, inner, seed].get('steps', candidates[names.index(candidate)].get('steps', 0)))
                         for inner in inner_folds for seed in seeds}
            if len(job_steps) != 1 or min(job_steps) <= 0:
                raise ValueError('A candidate must have one positive step budget across its inner jobs')
        _, metrics = aggregate(pd.concat(error_frames, ignore_index=True))
        ref = metrics[metrics.method.eq(reference)].set_index('target').rmse
        if not set(SELECTION_TARGETS) <= set(ref.index) or not np.isfinite(ref.loc[list(SELECTION_TARGETS)]).all() or (ref.loc[list(SELECTION_TARGETS)] <= 0).any():
            raise ValueError('Reference must have positive finite RMSE for all six selection metrics')
        outer_scores = []
        for order, candidate in enumerate(names):
            table = metrics[metrics.method.eq(candidate)].set_index('target')
            rmse = {target: float(table.loc[target, 'rmse']) for target in SELECTION_TARGETS}
            ratios = {target: rmse[target] / float(ref[target]) for target in SELECTION_TARGETS}
            if not np.isfinite(list(ratios.values())).all():
                raise ValueError('Nonfinite candidate ratio')
            row = {'outer_fold': outer, 'candidate': candidate, 'candidate_order': order,
                   'steps': int(by_key[outer, candidate, inner_folds[0], seeds[0]].get('steps', candidates[order].get('steps', 0))),
                   'eligible': bool(all(value <= RATIO_LIMIT for value in ratios.values())),
                   'score': float(np.mean(np.square(list(ratios.values())))), 'rmse': rmse, 'ratios': ratios}
            outer_scores.append(row)
        if not next(row for row in outer_scores if row['candidate'] == reference)['eligible']:
            raise AssertionError('Reference must always be eligible')
        winner = min((row for row in outer_scores if row['eligible']), key=lambda row: (row['score'], row['steps'], row['candidate_order']))
        selected[str(outer)] = winner['candidate']
        for row in outer_scores:
            row['selected'] = row['candidate'] == winner['candidate']
        scores.extend(outer_scores)
        for inner, frame in canonicals.items():
            identity_audit.append({'outer_fold': outer, 'inner_fold': inner, 'n_windows': len(frame),
                'n_jobs_matched': len(names) * len(seeds), 'all_context_target_observed_values_masks_exact': True,
                'sampling': signatures[inner]})
    protocol = {'targets': list(SELECTION_TARGETS), 'max_each_rmse_ratio': RATIO_LIMIT,
                'score': 'mean of six squared candidate/reference RMSE ratios',
                'ties': 'lower actual steps, then fixed candidate order', 'candidate_order': names,
                'reference_candidate': reference, 'configuration_plan_sha256': _plan_digest(manifest),
                'outer_folds': outer_folds, 'inner_folds': inner_folds, 'seeds': seeds,
                'selection_reads': 'Only manifest.tune_jobs evaluation files; no outer, v3, or BM/OU performance.'}
    selection = {'kind': 'allen_v4_inner_selection', 'protocol': protocol, 'selected_by_outer': selected,
                 'candidate_scores': scores, 'tune_evidence_sha256': dict(sorted(evidence.items())),
                 'identity_audit': identity_audit}
    path = run / 'selection.json'
    if path.exists() and _json(path) != selection:
        raise ValueError('Existing locked selection differs; refusing reselection or evidence changes')
    flat = []
    for row in scores:
        flat.append({k: v for k, v in row.items() if k not in ('rmse', 'ratios')} |
                    {'rmse_' + k: v for k, v in row['rmse'].items()} |
                    {'ratio_' + k: v for k, v in row['ratios'].items()})
    pd.DataFrame(flat).to_csv(run / 'scores.csv', index=False)
    _save(path, selection)
    return selection


def _verify_selection(run, manifest, selection):
    if selection.get('kind') != 'allen_v4_inner_selection':
        raise ValueError('Outer summary requires a locked inner selection')
    if _json(Path(run) / 'selection.json') != selection:
        raise ValueError('Passed selection differs from the locked selection file')
    candidates, names, reference = _candidate_spec(manifest)
    if selection['protocol']['candidate_order'] != names or selection['protocol']['reference_candidate'] != reference:
        raise ValueError('Candidate protocol changed after inner selection')
    if selection['protocol']['configuration_plan_sha256'] != _plan_digest(manifest):
        raise ValueError('Inner configuration plan changed after selection')
    for rel, digest in selection['tune_evidence_sha256'].items():
        if _sha(_child(Path(run) / rel, run)) != digest:
            raise ValueError('Inner evidence changed after selection: ' + rel)


def summarize_outer(run, manifest, selection, root):
    """Report all three v4 architectures against matched v3 and BM/OU evidence."""
    run, root = Path(run).resolve(), Path(root).resolve()
    _verify_selection(run, manifest, selection)
    outer_folds, _, seeds = _scope(manifest)
    expected = {(fold, variant, seed) for fold in outer_folds for variant in OUTER_VARIANTS for seed in seeds}
    jobs, keyed, identities = manifest['outer_jobs'], {}, []
    for job in jobs:
        key = (int(job['outer_fold']), job['variant'], int(job['seed']))
        if job.get('stage') != 'outer' or int(job['fold']) != key[0]:
            raise ValueError('Outer job stage/fold mismatch')
        if job['candidate'] != selection['selected_by_outer'].get(str(key[0])):
            raise ValueError('Outer job does not use the inner-selected configuration')
        identities.append(key)
        keyed[key] = job
    if len(identities) != len(expected) or set(identities) != expected or len({j['name'] for j in jobs}) != len(jobs):
        raise ValueError('Incomplete or duplicated outer architecture/fold/seed matrix')
    source, baseline = _child(root / manifest['source_run'], root), _child(root / manifest['baseline_run'], root)
    old_manifest = _json(source / 'manifest.json')
    if not _json(source / 'completion.json').get('all_done') or not _json(baseline / 'completion.json').get('all_done'):
        raise ValueError('Matched old neural or classical source run is incomplete')
    old_jobs = {(int(j['fold']), j['variant'], int(j['seed'])): j for j in old_manifest['jobs']
                if j['stage'] == 'real_development' and j['variant'] in OUTER_VARIANTS}
    errors, evidence, audit = [], {}, []
    for fold in outer_folds:
        canonical, ridge = None, None
        for variant in OUTER_VARIANTS:
            for seed in seeds:
                job = keyed[fold, variant, seed]
                current_evidence = {}
                _, frame = _evaluation(run, job['name'], current_evidence)
                evidence.update({run.relative_to(root).as_posix() + '/' + k: v for k, v in current_evidence.items()})
                if canonical is None:
                    canonical = frame
                else:
                    _pair(canonical, frame)
                errors.append(_errors(frame, 'v4_' + variant, fold, seed))
                if (fold, variant, seed) not in old_jobs:
                    raise ValueError('Missing matched original v3 job')
                old_evidence = {}
                _, saved = _evaluation(source, old_jobs[fold, variant, seed]['name'], old_evidence)
                _pair(canonical, saved, exact_observations=False)
                evidence.update({source.relative_to(root).as_posix() + '/' + k: v for k, v in old_evidence.items()})
                errors.append(_errors(saved, 'v3_' + variant, fold, seed))
                value = saved[['ridge_full_' + c for c in LENGTHS]].to_numpy(float)
                if ridge is None:
                    ridge = value
                elif not np.allclose(ridge, value, atol=1e-10, rtol=0):
                    raise ValueError('Original deterministic Ridge differs across matched jobs')
        errors.append(species_errors(canonical, canonical[['observed_' + c for c in LENGTHS]].to_numpy(float),
                                     ridge, 'ridge', fold, -1))
        path = baseline / 'folds' / f'fold_{fold}' / 'predictions.csv.gz'
        if not path.is_file():
            raise ValueError('Missing matched classical fold prediction')
        classical = pd.read_csv(path)
        # The classical artifact records only the four observed lengths. Check
        # that common schema exactly, allowing CSV round-trip floating error.
        common = ['window', 'target_row', 'source_row', 'target_tip', 'source_tip', 'family',
                  'context_0_tip', 'context_1_tip', 'context_2_tip', 'complete_three_bones']
        pd.testing.assert_frame_equal(canonical[common], classical[common], check_dtype=False, check_exact=True)
        np.testing.assert_allclose(canonical[['observed_' + c for c in LENGTHS]],
                                   classical[['observed_' + c for c in LENGTHS]], atol=1e-12, rtol=0, equal_nan=True)
        for method in ('bm', 'ou'):
            errors.append(_errors(classical, method, fold, -1, prefix=method))
        evidence[path.relative_to(root).as_posix()] = _sha(path)
        audit.append({'outer_fold': fold, 'matched_v4_jobs': len(OUTER_VARIANTS) * len(seeds),
                      'matched_v3_jobs': len(OUTER_VARIANTS) * len(seeds), 'n_windows': len(canonical),
                      'context_target_observation_support_matched': True})
    species = pd.concat(errors, ignore_index=True)
    family, metrics = aggregate(species)
    comparisons = []
    for (target, support), part in family.groupby(['target', 'support']):
        lookup = {method: frame.set_index('family').mse for method, frame in part.groupby('method')}
        for method in sorted(lookup):
            if method == 'v4_full':
                continue
            primary = target == 'allocation_logratio_joint' and method in ('bm', 'ou')
            comparisons.append({'first': 'v4_full', 'second': method, 'target': target, 'support': support,
                'primary_comparison': primary,
                'interval_role': '97.5% interval for two primary comparisons' if primary else 'descriptive 95% interval',
                **paired_difference(lookup['v4_full'], lookup[method], bootstrap=10000, seed=20261111)})
    species.to_csv(run / 'species_errors_v4.csv.gz', index=False, compression='gzip')
    family.to_csv(run / 'family_errors_v4.csv', index=False)
    metrics.to_csv(run / 'metrics_v4.csv', index=False)
    pd.DataFrame(comparisons).to_csv(run / 'comparisons_v4.csv', index=False)
    result = {'kind': 'allen_v4_outer_summary', 'n_metric_rows': len(metrics), 'n_comparisons': len(comparisons),
              'n_species_error_rows': len(species), 'folds': outer_folds, 'bootstrap': 10000,
              'methods': sorted(metrics.method.unique().tolist()), 'selection_architecture': 'full only on inner folds',
              'architecture_reporting': 'All full/single/supervised outer results retained; no outer architecture winner selected.',
              'difference_sign': 'second minus v4_full RMSE; positive favors v4_full',
              'interval_scope': 'Fixed fitted models/tree; outer-family resampling does not include retraining uncertainty.',
              'outer_identity_audit': audit, 'evidence_sha256': dict(sorted(evidence.items()))}
    _save(run / 'outer_summary_v4.json', result)
    return result
