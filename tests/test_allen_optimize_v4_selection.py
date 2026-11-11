"""GPU-free selection isolation, completeness, constraints, and outer matching."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/allen_world'))
from optimize_v4_selection import SELECTION_TARGETS, select_candidates, summarize_outer

LENGTHS = ('m_tarsus', 'b_fem', 'b_tbt', 'b_tmt')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def frame(label, factor=1., n=6):
    observed = np.column_stack([1 + .03 * np.arange(n), 1.1 + .05 * np.arange(n),
                                1.4 + .04 * np.arange(n), 1.25 + .06 * np.arange(n)])
    offsets = (1 + .03 * np.arange(n))[:, None] * np.array([.11, .08, -.06, .04])
    predicted = observed + offsets * np.asarray(factor)
    ridge = observed + offsets * .95
    observed[0, 1:] = np.nan
    data = pd.DataFrame({'window': np.arange(n), 'target_row': np.arange(n) + 3,
        'source_row': np.full(n, 2), 'target_tip': [f'{label}_bird{i}' for i in range(n)],
        'source_tip': [f'{label}_ref2'] * n, 'family': [f'{label}_family{i//2}' for i in range(n)],
        'context_0_tip': [f'{label}_ref0'] * n, 'context_1_tip': [f'{label}_ref1'] * n,
        'context_2_tip': [f'{label}_ref2'] * n, 'complete_three_bones': np.isfinite(observed[:, 1:]).all(axis=1)})
    for j, col in enumerate(LENGTHS):
        data['observed_' + col] = observed[:, j]
        data['observed_mask_' + col] = np.isfinite(observed[:, j])
        data['jepa_full_' + col] = predicted[:, j]
        data['ridge_full_' + col] = ridge[:, j]
    return data


def write_evaluation(run, name, label, factor=1.):
    folder = run / 'evaluation' / name
    folder.mkdir(parents=True)
    raw = frame(label, factor)
    raw.to_csv(folder / 'prediction_windows.csv.gz', index=False, compression='gzip')
    write_json(folder / 'pilot_results.json', {'history': 3, 'n_orders': 2, 'seed': 0, 'n_windows': len(raw),
        'provenance': {'table_sha256': label, 'table_metadata_sha256': label, 'tree_sha256': 'same-tree'}})


def make_inner(run, factors, steps=None, mode='smoke'):
    steps = steps or {}
    candidates = [{'name': name, 'steps': steps.get(name, 100)} for name in factors]
    manifest = {'mode': mode, 'candidates': candidates, 'reference_candidate': 'reference',
                'tune_jobs': [], 'source_run': 'DO_NOT_READ_SOURCE', 'baseline_run': 'DO_NOT_READ_BASELINE'}
    outers, inners, seeds = (range(5), (0, 1), (13, 14)) if mode == 'full' else ((0,), (0,), (13,))
    for outer in outers:
        for inner in inners:
            for seed in seeds:
                for candidate, factor in factors.items():
                    name = f'inner_o{outer}_f{inner}_s{seed}_{candidate}'
                    manifest['tune_jobs'].append({'name': name, 'stage': 'inner', 'outer_fold': outer,
                        'inner_fold': inner, 'fold': inner, 'seed': seed, 'candidate': candidate,
                        'variant': 'full', 'steps': steps.get(candidate, 100)})
                    write_evaluation(run, name, f'inner_o{outer}_f{inner}', factor)
    return manifest


class SelectionTests(unittest.TestCase):
    def test_six_metric_cap_beats_better_average_and_ignores_outer_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            manifest = make_inner(run, {'reference': 1., 'balanced': .9, 'bad_tarsus': [1.04, .5, .5, .5]})
            # This job is intentionally nonexistent: selection must not inspect it.
            manifest['outer_jobs'] = [{'name': 'outer_answers_must_not_be_read'}]
            selected = select_candidates(run, manifest, Path('/inaccessible-root'))
            self.assertEqual(selected['selected_by_outer'], {'0': 'balanced'})
            scores = {row['candidate']: row for row in selected['candidate_scores']}
            self.assertLess(scores['bad_tarsus']['score'], scores['balanced']['score'])
            self.assertFalse(scores['bad_tarsus']['eligible'])
            self.assertTrue(scores['reference']['eligible'])
            self.assertEqual(set(scores['reference']['ratios'].values()), {1.})
            self.assertEqual(len(selected['protocol']['targets']), 6)
            self.assertEqual(len(selected['tune_evidence_sha256']), 6)
            self.assertTrue((run / 'scores.csv').is_file())
            # Outer outcomes and paths can change without affecting inner selection.
            manifest['outer_jobs'] = [{'name': 'another_unread_outer_job'}]
            manifest['source_run'] = 'another_unread_old_run'
            self.assertEqual(select_candidates(run, manifest, Path('/elsewhere')), selected)

    def test_identical_scores_tie_by_actual_steps_then_fixed_order(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            manifest = make_inner(run, {'reference': 1., 'long': 1., 'short_first': 1., 'short_second': 1.},
                                  {'reference': 100, 'long': 200, 'short_first': 50, 'short_second': 50})
            selected = select_candidates(run, manifest, run)
            self.assertEqual(selected['selected_by_outer'], {'0': 'short_first'})

    def test_full_matrix_and_paired_support_are_required(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            manifest = make_inner(run, {'reference': 1., 'balanced': .9}, mode='full')
            incomplete = copy.deepcopy(manifest)
            incomplete['tune_jobs'].pop()
            with self.assertRaisesRegex(ValueError, 'matrix'):
                select_candidates(run, incomplete, run)
            self.assertFalse((run / 'selection.json').exists())
            selected = select_candidates(run, manifest, run)
            self.assertEqual(selected['selected_by_outer'], {str(i): 'balanced' for i in range(5)})
            self.assertEqual(len(selected['identity_audit']), 10)

    def test_context_and_observation_mask_changes_fail_before_selection(self):
        for changed_column, changed_value in [('context_0_tip', 'unknown_context'), ('observed_mask_b_fem', True)]:
            with self.subTest(column=changed_column), tempfile.TemporaryDirectory() as temp:
                run = Path(temp)
                manifest = make_inner(run, {'reference': 1., 'balanced': .9})
                path = run / 'evaluation' / manifest['tune_jobs'][-1]['name'] / 'prediction_windows.csv.gz'
                raw = pd.read_csv(path)
                raw.loc[0, changed_column] = changed_value
                raw.to_csv(path, index=False, compression='gzip')
                with self.assertRaises((AssertionError, ValueError)):
                    select_candidates(run, manifest, run)
                self.assertFalse((run / 'selection.json').exists())

    def test_changed_inner_evidence_cannot_reselect_after_lock(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            manifest = make_inner(run, {'reference': 1.})
            select_candidates(run, manifest, run)
            path = run / 'evaluation' / manifest['tune_jobs'][0]['name'] / 'prediction_windows.csv.gz'
            raw = pd.read_csv(path)
            raw['jepa_full_m_tarsus'] += .01
            raw.to_csv(path, index=False, compression='gzip')
            with self.assertRaisesRegex(ValueError, 'locked selection'):
                select_candidates(run, manifest, run)


class OuterSummaryTests(unittest.TestCase):
    def fixture(self, root):
        run, old, baseline = [root / 'outputs' / name for name in ('new', 'v3', 'bmou')]
        run.mkdir(parents=True)
        manifest = make_inner(run, {'reference': 1.})
        selected = select_candidates(run, manifest, root)
        manifest.update(source_run=old.relative_to(root).as_posix(), baseline_run=baseline.relative_to(root).as_posix())
        manifest['outer_jobs'] = []
        old_jobs = []
        for i, variant in enumerate(('full', 'single', 'supervised')):
            name = 'outer_' + variant
            manifest['outer_jobs'].append({'name': name, 'stage': 'outer', 'outer_fold': 0, 'fold': 0,
                'seed': 13, 'candidate': 'reference', 'variant': variant})
            write_evaluation(run, name, 'outer0', .8 + .1 * i)
            old_name = 'old_' + variant
            old_jobs.append({'name': old_name, 'stage': 'real_development', 'fold': 0, 'seed': 13, 'variant': variant})
            write_evaluation(old, old_name, 'outer0', 1. + .1 * i)
        write_json(old / 'manifest.json', {'jobs': old_jobs})
        write_json(old / 'completion.json', {'all_done': True})
        write_json(baseline / 'completion.json', {'all_done': True})
        classical = frame('outer0', .9)
        for col in LENGTHS:
            classical['bm_' + col] = classical['jepa_full_' + col]
            classical['ou_' + col] = classical['jepa_full_' + col]
        classical = classical.drop(columns=[c for c in classical if c.startswith(('observed_mask_', 'jepa_', 'ridge_'))])
        folder = baseline / 'folds/fold_0'
        folder.mkdir(parents=True)
        classical.to_csv(folder / 'predictions.csv.gz', index=False, compression='gzip')
        return run, manifest, selected

    def test_outer_reports_all_architectures_without_reselecting(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run, manifest, selected = self.fixture(root)
            before = (run / 'selection.json').read_bytes()
            result = summarize_outer(run, manifest, selected, root)
            self.assertEqual((run / 'selection.json').read_bytes(), before)
            self.assertEqual(result['n_metric_rows'], 72)
            self.assertEqual(result['n_comparisons'], 64)
            metrics = pd.read_csv(run / 'metrics_v4.csv')
            self.assertEqual(set(metrics.method), {'v4_full', 'v4_single', 'v4_supervised',
                'v3_full', 'v3_single', 'v3_supervised', 'bm', 'ou', 'ridge'})
            self.assertTrue(metrics[metrics.target.eq('m_tarsus')].n_species.eq(6).all())
            self.assertTrue(metrics[~metrics.target.eq('m_tarsus')].n_species.eq(5).all())
            comparisons = pd.read_csv(run / 'comparisons_v4.csv')
            primary = comparisons[comparisons.primary_comparison]
            self.assertEqual(set(primary.second), {'bm', 'ou'})
            self.assertTrue(primary.target.eq('allocation_logratio_joint').all())
            self.assertTrue(primary.interval_role.str.contains('97.5%').all())
            self.assertTrue((comparisons.ci975_low <= comparisons.ci95_low + 1e-14).all())
            self.assertTrue((comparisons.ci975_high >= comparisons.ci95_high - 1e-14).all())

    def test_outer_cannot_substitute_a_new_candidate_or_mismatched_species(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run, manifest, selected = self.fixture(root)
            wrong = copy.deepcopy(manifest)
            wrong['outer_jobs'][0]['candidate'] = 'posthoc_outer_winner'
            with self.assertRaisesRegex(ValueError, 'inner-selected'):
                summarize_outer(run, wrong, selected, root)
            path = root / manifest['baseline_run'] / 'folds/fold_0/predictions.csv.gz'
            raw = pd.read_csv(path)
            raw.loc[0, 'family'] = 'changed_family'
            raw.to_csv(path, index=False, compression='gzip')
            with self.assertRaises(AssertionError):
                summarize_outer(run, manifest, selected, root)
            self.assertFalse((run / 'metrics_v4.csv').exists())


if __name__ == '__main__':
    unittest.main()
