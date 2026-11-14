"""Tests of pairing, hierarchical weights and descriptive probe summaries."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'allen_world'))
from ecology_probe_core import SCENARIO_NAMES
from summarize_ecology_probe_v1 import (
    PAIR_NAMES, _load_and_validate, aggregate_family_profiles, aggregate_family_seed_profiles,
    family_mean_interval, summarize,
)


PAIR = 'ground_from_canopy'


def make_run(path, variants=('ou',), seeds=(13, 14), mode='smoke', empty=False):
    jobs = [dict(name=f'{variant}_f0_s{seed}', variant=variant, fold=0, seed=seed)
            for variant in variants for seed in seeds]
    manifest = dict(jobs=jobs, settings=dict(mode=mode, fraction_shift=.1,
                    temperature_step=1., bootstrap=100, model_subset=len(variants) < 3))
    (path / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    windows = pd.DataFrame(dict(selected_index=range(4), window=[7, 9, 10, 20],
        target_row=[0, 0, 1, 2], target_tip=['A', 'A', 'B', 'C'],
        family=['FamilyA', 'FamilyA', 'FamilyB', 'Anatidae'],
        order=['Passeriformes', 'Passeriformes', 'Passeriformes', 'Anseriformes'],
        mode=['A', 'A', 'A', 'P'], complete_three_bones=[False, False, True, True]))
    for job in jobs:
        directory = path / 'jobs' / job['name']
        (directory / 'pairs').mkdir(parents=True)
        windows.to_csv(directory / 'windows.csv', index=False)
        (directory / 'result.json').write_text(json.dumps(dict(status='complete')), encoding='utf-8')
        records = []
        for pair, (recipient, donor) in PAIR_NAMES.items():
            selected = pair == PAIR and not empty
            records.append(dict(pair=pair, recipient=recipient, donor=donor,
                n_candidate_windows=4, n_numeric_feasible=4 if selected else 0,
                n_supported_windows=4 if selected else 0,
                n_supported_species=3 if selected else 0,
                n_supported_families=3 if selected else 0,
                status='supported' if selected else 'unsupported'))
            if selected:
                baseline = np.log10(np.tile([40., 20., 30., 50.], (4, 1)))
                values = {name: baseline.copy() for name in SCENARIO_NAMES}
                # Differing individual effects verify transform-after-log-mean.
                changes = np.array([0., 0., np.log10(1.21), np.log10(1.44)])
                values['shifted'] += changes[:, None]
                values['cold_shifted'] += np.log10(1.01)
                values['warm_shifted'] += np.log10(1.06)
                np.savez_compressed(directory / 'pairs' / (pair + '.npz'),
                                    selected_indices=np.arange(4), **values)
        (directory / 'pair_support.json').write_text(json.dumps(records), encoding='utf-8')
    return manifest


class WeightingTest(unittest.TestCase):
    def test_species_family_repeat_weights_are_not_window_weights(self):
        frame = pd.DataFrame([
            [0, 13, 'a', 'F1', 0, 0.], [0, 13, 'a', 'F1', 1, 0.],
            [0, 13, 'a', 'F1', 2, 0.], [0, 13, 'b', 'F1', 3, 4.],
            [0, 14, 'a', 'F1', 0, 2.], [0, 14, 'b', 'F1', 1, 6.],
            [0, 13, 'c', 'F2', 4, 8.], [0, 14, 'c', 'F2', 4, 12.],
        ], columns=['fold', 'seed', 'target_tip', 'family', 'window', 'effect'])
        profiles = aggregate_family_profiles(frame, ['effect'])
        np.testing.assert_allclose(profiles.effect, [3., 10.])
        self.assertEqual(profiles.n_species.tolist(), [2, 1])
        self.assertEqual(profiles.n_windows.tolist(), [6, 2])
        self.assertEqual(profiles.n_seeds.tolist(), [2, 2])
        mean, lo, hi = family_mean_interval(profiles, ['effect'], 500)
        self.assertAlmostEqual(mean[0], 6.5)
        self.assertAlmostEqual(lo[0], 3.)
        self.assertAlmostEqual(hi[0], 10.)
        other = profiles.copy()
        other.effect += 5
        m2, l2, h2 = family_mean_interval(other, ['effect'], 500)
        np.testing.assert_allclose([m2[0] - mean[0], l2[0] - lo[0], h2[0] - hi[0]], 5)

    def test_single_family_has_no_degenerate_interval(self):
        profiles = pd.DataFrame(dict(family=['Anatidae'], effect=[.2]))
        mean, lo, hi = family_mean_interval(profiles, ['effect'], 100)
        self.assertEqual(mean[0], .2)
        self.assertTrue(np.isnan(lo[0]) and np.isnan(hi[0]))

    def test_combining_requires_every_expected_seed(self):
        frame = pd.DataFrame([
            [0, 13, 'a', 'F1', 0, 2.], [0, 14, 'a', 'F1', 1, 4.],
            [0, 13, 'b', 'F2', 2, 100.],
        ], columns=['fold', 'seed', 'target_tip', 'family', 'window', 'effect'])
        combined = aggregate_family_profiles(frame, ['effect'], required_seeds=(13, 14))
        self.assertEqual(combined.family.tolist(), ['F1'])
        self.assertEqual(combined.effect.tolist(), [3.])
        self.assertEqual(combined.n_windows.tolist(), [2])
        self.assertEqual(len(aggregate_family_profiles(frame, ['effect'])), 2)
        seed_rows = aggregate_family_seed_profiles(frame, ['effect'])
        self.assertEqual(len(seed_rows), 3)
        self.assertEqual(seed_rows.loc[seed_rows.family.eq('F2'), 'effect'].tolist(), [100.])


class SummaryTest(unittest.TestCase):
    def test_complete_summary_percent_transformation_and_unsupported_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            make_run(path)
            result = summarize(path, bootstrap=100)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['directed_pairs'], 42)
            self.assertEqual(result['inference'], 'implementation_only')
            table = pd.read_csv(path / 'summary.csv')
            self.assertTrue(table.interpretation.eq('implementation_only').all())
            row = table.loc[table.pair.eq(PAIR) & table.scope.eq('all')
                & table.label_subset.eq('all') & table.comparison.eq('observed_ecology')
                & table.outcome.eq('b_fem')].iloc[0]
            expected_log = (0 + np.log10(1.21) + np.log10(1.44)) / 3
            self.assertAlmostEqual(row.delta_log10, expected_log)
            self.assertAlmostEqual(row.percent_change, 100 * (10**expected_log - 1))
            self.assertEqual(row.n_species, 3)
            self.assertEqual(row.n_windows, 8)
            self.assertEqual(row.min_seeds_per_family, 2)
            self.assertEqual(row.support_status, 'limited')
            duck = table.loc[table.pair.eq(PAIR) & table.scope.eq('Anatidae')]
            self.assertTrue(duck.delta_log10_lo95.isna().all())
            self.assertTrue(duck.ci_method.eq('not_available').all())
            interaction = table.loc[table.pair.eq(PAIR) & table.scope.eq('all')
                & table.label_subset.eq('all') & table.comparison.eq('temperature_by_ecology')
                & table.outcome.eq('b_fem')].iloc[0]
            self.assertTrue(np.isnan(interaction.percent_change))
            self.assertAlmostEqual(interaction.response_factor_change_percent, 100 * (1.06 / 1.01 - 1))
            self.assertAlmostEqual(interaction.interaction_per10C_log10, np.log10(1.06 / 1.01) * 5)
            unsupported = table.loc[table.pair.ne(PAIR)]
            self.assertTrue(unsupported.support_status.eq('unsupported').all())
            self.assertTrue(unsupported.delta_log10.isna().all())
            profiles = pd.read_csv(path / 'family_profiles.csv.gz')
            self.assertIn('log__cold_ecology__b_fem', profiles)
            self.assertTrue((profiles.n_seeds == 2).all())

    def test_all_empty_support_still_writes_complete_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            make_run(path, seeds=(13,), empty=True)
            result = summarize(path, bootstrap=20)
            self.assertEqual(result['family_profile_rows'], 0)
            table = pd.read_csv(path / 'summary.csv')
            self.assertTrue(table.support_status.eq('unsupported').all())
            self.assertEqual(len(table), 42 * 8 * 4 * 10)
            self.assertEqual(len(pd.read_csv(path / 'family_profiles.csv.gz')), 0)
            self.assertEqual(len(pd.read_csv(path / 'family_seed_profiles.csv.gz')), 0)

    def test_incomplete_seed_families_are_excluded_but_seed_profiles_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            make_run(path)
            directory = path / 'jobs' / 'ou_f0_s14'
            archive = directory / 'pairs' / (PAIR + '.npz')
            with np.load(archive) as saved:
                values = {name: saved[name][:3].copy() for name in SCENARIO_NAMES}
            np.savez_compressed(archive, selected_indices=np.arange(3), **values)
            support = json.loads((directory / 'pair_support.json').read_text())
            next(r for r in support if r['pair'] == PAIR).update(
                n_supported_windows=3, n_supported_species=2, n_supported_families=2)
            (directory / 'pair_support.json').write_text(json.dumps(support))
            summarize(path, bootstrap=20)
            table = pd.read_csv(path / 'summary.csv')
            row = table.loc[table.pair.eq(PAIR) & table.scope.eq('all') & table.label_subset.eq('all')
                            & table.comparison.eq('observed_ecology') & table.outcome.eq('b_fem')].iloc[0]
            self.assertEqual(row.n_excluded_families_incomplete_seeds, 1)
            self.assertEqual(row.n_excluded_species_incomplete_seeds, 1)
            self.assertEqual(row.n_supported_windows_before_seed_filter, 7)
            self.assertEqual(row.n_windows, 6)
            self.assertEqual(row.n_species, 2)
            self.assertEqual(row.n_families, 2)
            self.assertAlmostEqual(row.delta_log10, np.log10(1.21) / 2)
            seed_profiles = pd.read_csv(path / 'family_seed_profiles.csv.gz')
            excluded = seed_profiles.loc[seed_profiles.family.eq('Anatidae')]
            self.assertTrue(len(excluded) > 0)
            self.assertTrue((excluded.seed == 13).all())
            self.assertTrue((~excluded.included_in_combined).all())
            profiles = pd.read_csv(path / 'family_profiles.csv.gz')
            self.assertFalse(profiles.family.eq('Anatidae').any())

    def test_identity_mismatch_is_rejected_before_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            manifest = make_run(path, variants=('full', 'ou'), seeds=(13,))
            window_file = path / 'jobs' / 'ou_f0_s13' / 'windows.csv'
            windows = pd.read_csv(window_file)
            windows.loc[3, 'target_tip'] = 'D'
            windows.to_csv(window_file, index=False)
            with self.assertRaisesRegex(ValueError, 'Unpaired model window'):
                _load_and_validate(path, manifest)

    def test_pair_support_mismatch_is_not_silently_intersected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            manifest = make_run(path, variants=('full', 'ou'), seeds=(13,))
            directory = path / 'jobs' / 'ou_f0_s13'
            archive_path = directory / 'pairs' / (PAIR + '.npz')
            with np.load(archive_path) as saved:
                values = {name: saved[name][:3].copy() for name in SCENARIO_NAMES}
            np.savez_compressed(archive_path, selected_indices=np.arange(3), **values)
            records = json.loads((directory / 'pair_support.json').read_text())
            record = next(r for r in records if r['pair'] == PAIR)
            record.update(n_supported_windows=3, n_supported_species=2, n_supported_families=2)
            (directory / 'pair_support.json').write_text(json.dumps(records))
            with self.assertRaisesRegex(ValueError, 'do not share pair support'):
                _load_and_validate(path, manifest)

    def test_missing_pair_and_nonfinite_predictions_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            manifest = make_run(path, seeds=(13,))
            directory = path / 'jobs' / 'ou_f0_s13'
            filename = directory / 'pair_support.json'
            records = json.loads(filename.read_text())
            filename.write_text(json.dumps(records[:-1]))
            with self.assertRaisesRegex(ValueError, 'all 42'):
                _load_and_validate(path, manifest)
            filename.write_text(json.dumps(records))
            archive = directory / 'pairs' / (PAIR + '.npz')
            with np.load(archive) as saved:
                values = {name: saved[name].copy() for name in SCENARIO_NAMES}
            values['warm'][0, 0] = np.nan
            np.savez_compressed(archive, selected_indices=np.arange(4), **values)
            with self.assertRaisesRegex(ValueError, 'Nonfinite or unpaired'):
                _load_and_validate(path, manifest)


if __name__ == '__main__':
    unittest.main()
