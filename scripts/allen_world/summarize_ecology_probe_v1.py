"""Summarize all frozen ecological probes with paired support and family weights.

No clustering, new-X selection, model fitting or causal attribution is performed.
Species first average their windows, families then average species, repeats then
average within family; the final mean and bootstrap weight families equally.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from ecology_probe_core import (FORAGING_COLUMNS, OUTCOME_NAMES, SCENARIO_NAMES,
                                SHARE_INDICES, percent_change_from_log,
                                summarize_corner_effects)


COMPARISONS = ('observed_ecology', 'cold_ecology', 'warm_ecology', 'temperature_by_ecology')
IDENTITY = ('selected_index', 'window', 'target_row', 'target_tip', 'family', 'order', 'mode', 'complete_three_bones')
SCOPES = (('all', 'all'), ('all', 'eliason_complete'), ('all', 'eliason_missing'),
          ('passerines', 'all'), ('passerines', 'eliason_complete'), ('passerines', 'eliason_missing'),
          ('nonpasserines', 'all'), ('Anatidae', 'all'))
PAIR_NAMES = {a.removeprefix('e_fs_') + '_from_' + b.removeprefix('e_fs_'): (a, b)
              for a, b in itertools.permutations(FORAGING_COLUMNS, 2)}
LOG_FEATURES = tuple('log__' + comparison + '__' + outcome
                     for comparison in COMPARISONS for outcome in OUTCOME_NAMES)
PP_FEATURES = tuple('pp__' + comparison + '__' + OUTCOME_NAMES[i]
                    for comparison in COMPARISONS[:-1] for i in SHARE_INDICES)
FEATURES = LOG_FEATURES + PP_FEATURES
COVERAGE_FIELDS = ('n_supported_families_before_seed_filter', 'n_supported_species_before_seed_filter',
                   'n_supported_windows_before_seed_filter', 'n_excluded_families_incomplete_seeds',
                   'n_excluded_species_incomplete_seeds', 'n_excluded_windows_incomplete_seeds', 'required_seeds')
BOOTSTRAP_SEED = 20261114


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _bool_column(values):
    mapping = {True: True, False: False, 'True': True, 'False': False,
               'true': True, 'false': False, 1: True, 0: False}
    if values.isna().any() or not values.isin(mapping).all():
        raise ValueError('complete_three_bones must be an observed boolean flag')
    return values.map(mapping).astype(bool)


def _load_and_validate(run, manifest):
    jobs = manifest.get('jobs', [])
    if not jobs or len({j['name'] for j in jobs}) != len(jobs):
        raise ValueError('Manifest needs distinct nonempty jobs')
    identities, loaded, grids = {}, [], {}
    for job in jobs:
        name = job['name']
        if not isinstance(name, str) or re.fullmatch(r'[A-Za-z0-9_-]+', name) is None:
            raise ValueError('Unsafe job name')
        key = (int(job['fold']), int(job['seed']))
        variant = str(job['variant'])
        if key in grids.setdefault(variant, set()):
            raise ValueError('Duplicate variant/fold/seed')
        grids[variant].add(key)
        directory = run / 'jobs' / name
        result = _json(directory / 'result.json')
        if result.get('status') != 'complete':
            raise ValueError('Cannot summarize unfinished job: ' + name)
        windows = pd.read_csv(directory / 'windows.csv')
        if not set(IDENTITY) <= set(windows):
            raise ValueError('Missing window identity fields: ' + name)
        windows = windows[list(IDENTITY)].copy()
        windows['complete_three_bones'] = _bool_column(windows.complete_three_bones)
        if (not np.array_equal(windows.selected_index.to_numpy(), np.arange(len(windows)))
                or windows.window.duplicated().any()
                or windows[['target_row', 'target_tip', 'family', 'order']].isna().any().any()):
            raise ValueError('Invalid selected window identities: ' + name)
        if key in identities and not windows.equals(identities[key]):
            raise ValueError('Unpaired model window identities at ' + str(key))
        identities.setdefault(key, windows)
        records = _json(directory / 'pair_support.json')
        if not isinstance(records, list):
            raise ValueError('pair_support.json must be a list')
        support = {r['pair']: r for r in records}
        if len(support) != len(records) or set(support) != set(PAIR_NAMES):
            raise ValueError('Expected all 42 directed pairs, including unsupported ones')
        indices = {}
        expected_files = set()
        for pair, record in support.items():
            if (record['recipient'], record['donor']) != PAIR_NAMES[pair]:
                raise ValueError('Pair direction disagrees with component labels')
            n = int(record['n_supported_windows'])
            if n < 0:
                raise ValueError('Negative support count')
            path = directory / 'pairs' / (pair + '.npz')
            if n == 0:
                if path.exists():
                    raise ValueError('Unsupported pair unexpectedly has predictions')
                selected = np.array([], dtype=int)
            else:
                expected_files.add(path.name)
                with np.load(path, allow_pickle=False) as values:
                    if not {'selected_indices', *SCENARIO_NAMES} <= set(values.files):
                        raise ValueError('Missing paired scenario arrays: ' + str(path))
                    selected = values['selected_indices'].copy()
                    if (selected.ndim != 1 or selected.dtype.kind not in 'iu' or len(selected) != n
                            or np.any(selected < 0) or np.any(selected >= len(windows))
                            or np.any(np.diff(selected) <= 0)):
                        raise ValueError('Invalid selected pair indices: ' + str(path))
                    for scenario in SCENARIO_NAMES:
                        arr = values[scenario]
                        if arr.shape != (n, 4) or not np.isfinite(arr).all():
                            raise ValueError('Nonfinite or unpaired corner prediction: ' + str(path))
                selected_windows = windows.iloc[selected]
                if (selected_windows.target_tip.nunique() != int(record['n_supported_species'])
                        or selected_windows.family.nunique() != int(record['n_supported_families'])):
                    raise ValueError('Saved support counts disagree with identities')
            indices[pair] = selected
        actual_files = {p.name for p in (directory / 'pairs').glob('*.npz')}
        if actual_files != expected_files:
            raise ValueError('Unlisted pair prediction files')
        loaded.append({'job': job, 'path': directory, 'windows': windows,
                       'support': support, 'indices': indices})
    if any(grid != next(iter(grids.values())) for grid in grids.values()):
        raise ValueError('Models do not have identical fold/seed grids')
    for key in next(iter(grids.values())):
        matched = [j for j in loaded if (j['job']['fold'], j['job']['seed']) == key]
        for pair in PAIR_NAMES:
            if any(not np.array_equal(j['indices'][pair], matched[0]['indices'][pair]) for j in matched[1:]):
                raise ValueError('Models do not share pair support: ' + str(key) + ' ' + pair)
    # Target identities and fold assignment must not change between repeats.
    targets = pd.concat([j['windows'].assign(fold=int(j['job']['fold'])) for j in loaded], ignore_index=True)
    for field in ('target_row', 'family', 'order', 'complete_three_bones', 'fold'):
        if (targets.groupby('target_tip')[field].nunique() > 1).any():
            raise ValueError('Target identity or fold changes between jobs: ' + field)
    if (targets.groupby('family').fold.nunique() > 1).any():
        raise ValueError('A family appears in multiple outer folds')
    return loaded


def aggregate_family_seed_profiles(frame, feature_columns=FEATURES):
    """Retain separately inspectable species-equal family means for every seed."""
    features = list(feature_columns)
    if frame.empty:
        return pd.DataFrame(columns=['fold', 'seed', 'family', 'n_species', 'n_windows', *features])
    if not np.isfinite(frame[features].to_numpy(float)).all():
        raise ValueError('Nonfinite effects cannot silently receive different aggregation support')
    species = frame.groupby(['fold', 'seed', 'target_tip', 'family'], sort=True)[features].mean()
    family_seed = species.groupby(level=['fold', 'seed', 'family'])[features].mean()
    counts = frame.groupby(['fold', 'seed', 'family']).agg(n_species=('target_tip', 'nunique'),
                                                         n_windows=('window', 'size'))
    return counts.join(family_seed).reset_index().sort_values(['family', 'seed']).reset_index(drop=True)


def complete_seed_support(frame, required_seeds=None):
    """Filter whole families to shared repeat support; never fill absent seeds."""
    if frame.empty or required_seeds is None:
        return frame.copy()
    required_seeds = tuple(required_seeds)
    if not required_seeds or len(set(required_seeds)) != len(required_seeds):
        raise ValueError('Required seeds must be nonempty and distinct')
    required = set(required_seeds)
    if set(frame.seed.unique()) - required:
        raise ValueError('Unexpected seed outside the required repeat set')
    available = frame.groupby('family').seed.apply(lambda x: set(x.unique()))
    keep = set(available.index[available.map(lambda seeds: seeds == required)])
    return frame.loc[frame.family.isin(keep)].copy()


def aggregate_family_profiles(frame, feature_columns=FEATURES, required_seeds=None):
    """Window -> species -> family -> repeat means on complete repeat support."""
    features = list(feature_columns)
    frame = complete_seed_support(frame, required_seeds)
    if frame.empty:
        return pd.DataFrame(columns=['family', 'n_species', 'n_windows', 'n_seeds', 'seeds', *features])
    family_seed = aggregate_family_seed_profiles(frame, features)
    values = family_seed.groupby('family')[features].mean()
    counts = frame.groupby('family').agg(n_species=('target_tip', 'nunique'),
                                        n_windows=('window', 'size'), n_seeds=('seed', 'nunique'))
    counts['seeds'] = frame.groupby('family').seed.apply(lambda s: '|'.join(map(str, sorted(s.unique()))))
    return counts.join(values).reset_index().sort_values('family').reset_index(drop=True)


@lru_cache(maxsize=16)
def _bootstrap_weights(families, repeats, seed):
    digest = hashlib.sha256(('\0'.join(families) + '|' + str(seed)).encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], 'little'))
    n = len(families)
    return rng.multinomial(n, np.full(n, 1 / n), size=repeats).astype(float) / n


def family_mean_interval(profiles, columns, bootstrap=2000):
    """Shared family resampling across all columns and model-matched supports."""
    profiles = profiles.sort_values('family')
    values = profiles[list(columns)].to_numpy(float)
    if len(values) == 0:
        blank = np.full(len(columns), np.nan)
        return blank, blank.copy(), blank.copy()
    mean = values.mean(axis=0)
    if len(values) == 1:
        return mean, np.full_like(mean, np.nan), np.full_like(mean, np.nan)
    weights = _bootstrap_weights(tuple(profiles.family.astype(str)), int(bootstrap), BOOTSTRAP_SEED)
    means = weights @ values
    lo, hi = np.quantile(means, [.025, .975], axis=0)
    return mean, lo, hi


def _scope(frame, scope, subset):
    keep = np.ones(len(frame), dtype=bool)
    if scope == 'passerines':
        keep &= frame.order.eq('Passeriformes').to_numpy()
    elif scope == 'nonpasserines':
        keep &= frame.order.notna().to_numpy() & ~frame.order.eq('Passeriformes').to_numpy()
    elif scope == 'Anatidae':
        keep &= frame.family.eq('Anatidae').to_numpy()
    if subset == 'eliason_complete':
        keep &= frame.complete_three_bones.to_numpy(bool)
    elif subset == 'eliason_missing':
        keep &= ~frame.complete_three_bones.to_numpy(bool)
    return frame.loc[keep]


def _effects_frame(item, pair, step):
    selected = item['indices'][pair]
    if not len(selected):
        return None
    with np.load(item['path'] / 'pairs' / (pair + '.npz'), allow_pickle=False) as archive:
        predictions = {scenario: archive[scenario] for scenario in SCENARIO_NAMES}
    effects = summarize_corner_effects(predictions, temperature_step=step)
    frame = item['windows'].iloc[selected].copy().reset_index(drop=True)
    frame['fold'], frame['seed'] = int(item['job']['fold']), int(item['job']['seed'])
    arrays = {}
    for comparison in COMPARISONS:
        for i, outcome in enumerate(OUTCOME_NAMES):
            arrays['log__' + comparison + '__' + outcome] = effects[comparison]['delta_log10'][:, i]
        if comparison != 'temperature_by_ecology':
            for i in SHARE_INDICES:
                arrays['pp__' + comparison + '__' + OUTCOME_NAMES[i]] = effects[comparison]['share_pp'][:, i]
    return pd.concat([frame, pd.DataFrame(arrays)], axis=1)


def _summary_rows(profiles, frame, base, bootstrap, step, smoke):
    n_species = int(frame.target_tip.nunique()) if len(frame) else 0
    n_families = len(profiles)
    support = 'unsupported' if not n_families else ('limited' if n_species < 20 or n_families < 5 else 'supported')
    meta = dict(base, n_species=n_species, n_families=n_families, n_windows=len(frame),
                n_seeds=int(frame.seed.nunique()) if len(frame) else 0,
                min_seeds_per_family=int(profiles.n_seeds.min()) if len(profiles) else 0,
                max_seeds_per_family=int(profiles.n_seeds.max()) if len(profiles) else 0,
                support_status=support,
                interpretation='implementation_only' if smoke else 'conditional_model_sensitivity_descriptive',
                ci_method='family_bootstrap_conditional_on_fixed_fits' if n_families > 1 else 'not_available',
                bootstrap_repeats=bootstrap)
    means, lo, hi = family_mean_interval(profiles, FEATURES, bootstrap)
    lookup = {name: i for i, name in enumerate(FEATURES)}
    rows = []
    for comparison in COMPARISONS:
        for outcome in OUTCOME_NAMES:
            j = lookup['log__' + comparison + '__' + outcome]
            value = [float(v[j]) for v in (means, lo, hi)]
            row = dict(meta, comparison=comparison, outcome=outcome,
                       delta_log10=value[0], delta_log10_lo95=value[1], delta_log10_hi95=value[2])
            for field in ('percent_change', 'share_pp', 'response_factor_change_percent', 'interaction_per10C_log10'):
                for suffix in ('', '_lo95', '_hi95'):
                    row[field + suffix] = np.nan
            transformed = percent_change_from_log(value)
            field = 'response_factor_change_percent' if comparison == 'temperature_by_ecology' else 'percent_change'
            for suffix, number in zip(('', '_lo95', '_hi95'), transformed):
                row[field + suffix] = float(number)
            if comparison == 'temperature_by_ecology':
                for suffix, number in zip(('', '_lo95', '_hi95'), np.asarray(value) / (2 * step) * 10):
                    row['interaction_per10C_log10' + suffix] = float(number)
            else:
                name = 'pp__' + comparison + '__' + outcome
                if name in lookup:
                    index = lookup[name]
                    for suffix, values in zip(('', '_lo95', '_hi95'), (means, lo, hi)):
                        row['share_pp' + suffix] = float(values[index])
            rows.append(row)
    return rows


def summarize(run, bootstrap=None):
    """Validate all completed jobs, write all summaries, return completion data."""
    run = Path(run).resolve()
    manifest = _json(run / 'manifest.json')
    settings = manifest['settings']
    bootstrap = settings.get('bootstrap', 2000) if bootstrap is None else bootstrap
    if not isinstance(bootstrap, (int, np.integer)) or isinstance(bootstrap, bool) or bootstrap < 1:
        raise ValueError('Bootstrap count must be a positive integer')
    step = float(settings['temperature_step'])
    if not np.isfinite(step) or step <= 0:
        raise ValueError('Temperature step must be finite and positive')
    loaded = _load_and_validate(run, manifest)
    variants = sorted({item['job']['variant'] for item in loaded})
    smoke = settings['mode'] == 'smoke'
    rows, support_rows = [], []
    family_count, seed_family_count, wrote_header, wrote_seed_header = 0, 0, False, False
    family_path = run / 'family_profiles.csv.gz'
    seed_family_path = run / 'family_seed_profiles.csv.gz'
    required_seeds = {variant: tuple(sorted({int(item['job']['seed']) for item in loaded
                                             if item['job']['variant'] == variant})) for variant in variants}
    with gzip.open(family_path, 'wt', encoding='utf-8', newline='') as stream, \
            gzip.open(seed_family_path, 'wt', encoding='utf-8', newline='') as seed_stream:
        for pair in sorted(PAIR_NAMES):
            for variant in variants:
                parts = [_effects_frame(item, pair, step) for item in loaded if item['job']['variant'] == variant]
                parts = [p for p in parts if p is not None]
                frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=[*IDENTITY, 'fold', 'seed', *FEATURES])
                recipient, donor = PAIR_NAMES[pair]
                for scope, subset in SCOPES:
                    before = _scope(frame, scope, subset)
                    selected = complete_seed_support(before, required_seeds[variant])
                    profiles = aggregate_family_profiles(before, required_seeds=required_seeds[variant])
                    seed_profiles = aggregate_family_seed_profiles(before)
                    coverage_counts = dict(
                        n_supported_families_before_seed_filter=int(before.family.nunique()),
                        n_supported_species_before_seed_filter=int(before.target_tip.nunique()),
                        n_supported_windows_before_seed_filter=len(before),
                        n_excluded_families_incomplete_seeds=int(before.family.nunique() - selected.family.nunique()),
                        n_excluded_species_incomplete_seeds=int(before.target_tip.nunique() - selected.target_tip.nunique()),
                        n_excluded_windows_incomplete_seeds=len(before) - len(selected),
                        required_seeds='|'.join(map(str, required_seeds[variant])))
                    base = dict(model=variant, pair=pair, recipient=recipient, donor=donor,
                                scope=scope, label_subset=subset,
                                fraction_shift=float(settings['fraction_shift']), temperature_step_C=step,
                                **coverage_counts)
                    rows.extend(_summary_rows(profiles, selected, base, int(bootstrap), step, smoke))
                    if len(profiles):
                        output = profiles.assign(**base)
                        output.to_csv(stream, index=False, header=not wrote_header)
                        wrote_header = True
                        family_count += len(output)
                    if len(seed_profiles):
                        seed_output = seed_profiles.assign(included_in_combined=seed_profiles.family.isin(profiles.family), **base)
                        seed_output.to_csv(seed_stream, index=False, header=not wrote_seed_header)
                        wrote_seed_header = True
                        seed_family_count += len(seed_output)
                    if scope == 'all' and subset == 'all':
                        support_rows.append(dict(base, n_species=int(selected.target_tip.nunique()),
                                                 n_families=len(profiles), n_windows=len(selected)))
        if not wrote_header:
            pd.DataFrame(columns=['family', 'n_species', 'n_windows', 'n_seeds', 'seeds', *FEATURES,
                                  'model', 'pair', 'recipient', 'donor', 'scope', 'label_subset',
                                  'fraction_shift', 'temperature_step_C', *COVERAGE_FIELDS]).to_csv(stream, index=False)
        if not wrote_seed_header:
            pd.DataFrame(columns=['fold', 'seed', 'family', 'n_species', 'n_windows', *FEATURES,
                                  'included_in_combined', 'model', 'pair', 'recipient', 'donor',
                                  'scope', 'label_subset', 'fraction_shift', 'temperature_step_C',
                                  *COVERAGE_FIELDS]).to_csv(seed_stream, index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(run / 'summary.csv', index=False, encoding='utf-8-sig')
    support = pd.DataFrame(support_rows)
    support.to_csv(run / 'summary_support.csv', index=False, encoding='utf-8-sig')
    # Every variant has the same selected species/windows; display coverage once.
    coverage = support.loc[support.model.eq(variants[0])]
    lines = ['# Frozen ecological probe summary', '',
             '**IMPLEMENTATION CHECK ONLY — no biological conclusions.**' if smoke else
             'Descriptive conditional model responses, not causal effects or shares of evolution.', '',
             f'Models: {", ".join(variants)}. All {len(PAIR_NAMES)} directed reallocation pairs retained, including empty support.',
             f'Transfer: {100 * float(settings["fraction_shift"]):g} percentage points. Cold/warm shifts: ±{step:g}°C; span {2*step:g}°C.', '',
             'Window log differences are averaged within species, species within family/repeat, repeats within family, then families equally. Combined results include only families present in every expected seed from the manifest; exclusions and pre-filter counts are retained. Percent changes are transformed from that aggregate log mean (geometric response factors), not means of window percentages.', '',
             'Family bootstrap intervals condition on the fitted models and selected support. They omit retraining, tree selection and measurement uncertainty; seed variation is averaged, not fully propagated. One-family scopes have no interval. Fewer than 20 species or 5 families is flagged limited; smoke remains implementation_only regardless of counts.', '',
             'The interaction is warm ecological log response minus cold ecological log response. Its percentage is a ratio of response factors, not a subtraction of percentage changes or an evolutionary contribution. Share percentage points are separately averaged and bootstrapped.', '',
             'All model variants share exactly the same window identities and supported pair indices within fold/seed. The same family bootstrap draws are reused wherever family sets agree. No clustering, significance ranking or new X was selected.', '',
             'Different directed pairs can have different supported species and families. Their aggregate percentages are not directly comparable as a seven-category importance ranking or a basis for pooling. A later grouping analysis must compare multi-bone response profiles on explicitly shared support, with uncertainty and temperature dependence retained.', '',
             'Scopes: all, passerines, nonpasserines and Anatidae. Complete/missing Eliason labels additionally subdivide all and passerines; labels are not used as prediction truth here.', '',
             'summary.csv contains every comparison/outcome/scope, including unsupported rows. family_profiles.csv.gz is wide: log__comparison__outcome and pp__comparison__share columns retain combined family profiles. family_seed_profiles.csv.gz separately retains every seed/family profile, including excluded families marked included_in_combined=False. summary_support.csv retains coverage and incomplete-seed exclusions for every model.', '',
             '| Directed pair | Retained species | Retained families | Retained windows (including repeats) | Families excluded for incomplete seeds |', '|---|---:|---:|---:|---:|']
    lines += [f'| {r.pair} | {r.n_species} | {r.n_families} | {r.n_windows} | {r.n_excluded_families_incomplete_seeds} |' for r in coverage.itertuples()]
    (run / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    info = {'status': 'complete', 'summary_rows': len(summary), 'family_profile_rows': family_count,
            'family_seed_profile_rows': seed_family_count,
            'models': variants, 'mode': settings['mode'], 'bootstrap': int(bootstrap),
            'bootstrap_seed': BOOTSTRAP_SEED, 'directed_pairs': len(PAIR_NAMES),
            'manifest_sha256': _sha(run / 'manifest.json'), 'summarizer_sha256': _sha(__file__),
            'all_models_paired_support': True, 'model_retrained': False,
            'inference': 'implementation_only' if smoke else 'descriptive_fixed_model_conditional_sensitivity'}
    (run / 'summary_completion.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    return info


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--bootstrap', type=int)
    args = p.parse_args()
    print(json.dumps(summarize(args.run, args.bootstrap), ensure_ascii=False))


if __name__ == '__main__':
    main()
