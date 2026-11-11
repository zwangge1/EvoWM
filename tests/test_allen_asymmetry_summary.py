"""Scientific weighting, fixed support and degradation contrasts for asymmetry."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'allen_world'))
from summarize_asymmetry_v1 import (species_errors, aggregate, paired_difference,
                                    difference_in_differences, comparisons, summarize,
                                    reference_comparisons, METHODS, LENGTHS, REFCASES)


class AsymmetrySummaryTests(unittest.TestCase):
    def fixture(self, method='full', cell='complete100', refcase='observed', rep=0, scale=1.):
        identity = pd.DataFrame({'target_tip': ['a', 'a', 'b', 'c'], 'family': ['F', 'F', 'F', 'G'],
                                 'has_reference_bones': [True, False, True, True]})
        observed, predicted = np.zeros((4, 4)), np.zeros((4, 4))
        predicted[:, 0] = np.array([0., 2., 4., 10.]) * scale
        return species_errors(identity, observed, predicted, method, 0, cell, refcase, rep,
                              [101, 202][rep], [13, 14][rep] if method in METHODS[:3] else -1)

    def test_errors_then_species_family_repetition_weights(self):
        one, two = self.fixture(), self.fixture(rep=1, scale=2.)
        family_rep, family, metrics = aggregate(pd.concat([one, two], ignore_index=True))
        row = metrics.query("target == 'm_tarsus' and reference_support == 'all'").iloc[0]
        # F averages species a=(0+4)/2 and b=16 -> 9; G=100.
        self.assertAlmostEqual(row.mse, 54.5 * 2.5)
        self.assertAlmostEqual(row.rmse, np.sqrt(54.5 * 2.5))
        self.assertEqual(row.n_repetitions, 2)
        # Averaging predictions before squaring would give 54.5 * 2.25: prohibited.
        self.assertNotAlmostEqual(row.mse, 54.5 * 2.25)
        self.assertEqual(len(family_rep), 2 * len(family))

    def test_all_bone_metrics_use_complete_three_bones_and_original_reference_support(self):
        identity = pd.DataFrame({'target_tip': ['a', 'b'], 'family': ['F', 'G'],
                                 'has_reference_bones': ['True', 'False']})
        obs = np.array([[1., 1., 1., 1.], [1., 1., np.nan, 1.]])
        rows = species_errors(identity, obs, np.ones((2, 4)), 'ou', 0, 'uniform25',
                              'hide_all_bones', 0, 101)
        self.assertEqual(len(rows.query("target == 'm_tarsus' and reference_support == 'all'")), 2)
        for target in ('b_fem', 'b_tmt', 'log10_tmt_over_fem', 'allocation_logratio_joint'):
            self.assertEqual(len(rows.query('target == @target and reference_support == "all"')), 1)
        self.assertEqual(len(rows.query("target == 'm_tarsus' and reference_support == 'has_reference_bones'")), 1)

    def test_invalid_support_replicates_and_duplicate_rows_rejected(self):
        a, b = self.fixture(), self.fixture(method='ou')
        with self.assertRaisesRegex(ValueError, 'Unmatched fixed'):
            aggregate(pd.concat([a, b.query("target_tip != 'b'")]))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            aggregate(pd.concat([a, a]))
        with self.assertRaisesRegex(ValueError, 'multiple held-out folds'):
            aggregate(pd.concat([a, b.assign(fold=1)]))
        with self.assertRaisesRegex(ValueError, 'Unmatched repetitions'):
            aggregate(pd.concat([a, b, self.fixture(rep=1)]))
        with self.assertRaisesRegex(ValueError, 'unmatched mask seeds'):
            aggregate(pd.concat([a, b.assign(mask_seed=999)]))

    def test_robustness_and_absolute_accuracy_can_disagree(self):
        idx = ['F', 'G', 'H']
        series = lambda x: pd.Series(np.full(3, x*x), index=idx)
        # Full starts worse and remains worse, but its error increase is smaller.
        raw = paired_difference(series(3.5), series(2.), bootstrap=99)
        did = difference_in_differences(series(3.), series(3.5), series(1.), series(2.), bootstrap=99)
        self.assertAlmostEqual(raw['comparator_minus_full_rmse'], -1.5)
        self.assertAlmostEqual(did['comparator_minus_full_rmse_increase'], .5)
        self.assertAlmostEqual(did['ci95_low'], .5)
        self.assertAlmostEqual(did['ci95_high'], .5)

    def test_paired_bootstrap_reproducibility_and_same_family_requirement(self):
        a = pd.Series([1., 4., 9.], index=['F', 'G', 'H'])
        result = paired_difference(a, a * 4, 2000)
        self.assertEqual(result, paired_difference(a, (a * 4).iloc[::-1], 2000))
        self.assertGreater(result['ci95_low'], 0)
        with self.assertRaises(ValueError): paired_difference(a, a.iloc[:2])
        with self.assertRaises(ValueError): difference_in_differences(a, a, a, a.iloc[:2])
        with self.assertRaises(ValueError): paired_difference(a, a, 0)

    def test_primary_comparisons_and_development_interval_labels(self):
        rows = [self.fixture(method=method, cell=cell, rep=rep, scale=1.+rep)
                for method in METHODS for cell in ('complete100', 'uniform25', 'family25')
                for rep in (0, 1)]
        _, family, _ = aggregate(pd.concat(rows, ignore_index=True))
        raw, did = comparisons(family, bootstrap=9)
        self.assertEqual(int(raw.primary_comparison.sum()), 6)
        self.assertEqual(int(did.primary_comparison.sum()), 6)
        self.assertTrue(raw.interval_role.str.contains('descriptive development').all())
        self.assertTrue(did.baseline_cell.eq('complete100').all())
        self.assertTrue(did.comparator_minus_full_rmse_increase.eq(0).all())

    def test_training_and_reference_degradation_use_distinct_baselines(self):
        rows = []
        rmse = {('complete100', 'observed'): (3., 1.),
                ('uniform25', 'observed'): (3.5, 2.),
                ('complete100', 'hide_all_bones'): (3.2, 1.1),
                ('uniform25', 'hide_all_bones'): (5., 2.2)}
        for (cell, refcase), (full, other) in rmse.items():
            for method in METHODS:
                for family in ('F', 'G'):
                    rows.append(dict(cell=cell, refcase=refcase, method=method,
                        target='allocation_logratio_joint', support='complete_three_bones',
                        reference_support='all', family=family, mse=(full if method == 'full' else other)**2))
        frame = pd.DataFrame(rows)
        _, train = comparisons(frame, bootstrap=9)
        reference = reference_comparisons(frame, bootstrap=9)
        train_row = train.query("cell == 'uniform25' and refcase == 'observed' and comparator == 'ou'").iloc[0]
        ref_row = reference.query("cell == 'uniform25' and comparator == 'ou'").iloc[0]
        self.assertAlmostEqual(train_row.comparator_minus_full_rmse_increase, .5)
        self.assertAlmostEqual(ref_row.comparator_minus_full_rmse_increase, -1.3)
        self.assertEqual(train_row.baseline_cell, 'complete100')
        self.assertEqual(ref_row.baseline_reference_case, 'observed')
        self.assertAlmostEqual(ref_row.full_observed_rmse, 3.5)
        self.assertAlmostEqual(ref_row.full_reference_case_rmse, 5.)
        self.assertFalse(reference.primary_comparison.any())

    def test_manifest_pipeline_outputs_and_frozen_support_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            run, jobs = Path(temp), []
            for cell in ('complete100', 'uniform25', 'family25'):
                for kind, variant in [('neural', 'full'), ('neural', 'single'),
                                      ('neural', 'supervised'), ('classic', 'classic')]:
                    name = f'{variant}_f0_r0_{cell}'
                    job = dict(name=name, kind=kind, variant=variant, condition=cell,
                               fold=0, replicate=0, mask_seed=101, seed=13)
                    jobs.append(job)
                    folder = run / 'evaluation' / name
                    folder.mkdir(parents=True)
                    frames = []
                    for method in ([variant] if kind == 'neural' else ['BM', 'OU']):
                        for refcase in REFCASES:
                            frame = pd.DataFrame(dict(model=method, condition=cell, reference_case=refcase,
                                fold=0, replicate=0, seed=13, mask_seed=101, window=[0, 1], target_row=[9, 10],
                                target_tip=['a', 'b'], family=['F', 'G'], has_reference_bones=[True, False],
                                complete_three_bones=[True, True], context_0_tip=['r0', 'r3'],
                                context_1_tip=['r1', 'r4'], context_2_tip=['r2', 'r5']))
                            for column in LENGTHS:
                                frame['observed_' + column] = 1.
                                frame['predicted_' + column] = [1.1, 1.2]
                            frames.append(frame)
                    pd.concat(frames).to_csv(folder / 'predictions.csv.gz', index=False, compression='gzip')
                    digest = hashlib.sha256((folder / 'predictions.csv.gz').read_bytes()).hexdigest()
                    (folder / 'result.json').write_text(json.dumps(dict(status='complete',
                        output_sha256={'predictions.csv.gz': digest})), encoding='utf-8')
            (run / 'manifest.json').write_text(json.dumps(dict(mode='smoke', jobs=jobs)), encoding='utf-8')
            result = summarize(run, bootstrap=9)
            self.assertTrue(result['pipeline_only'])
            self.assertEqual(result['n_jobs'], 12)
            self.assertEqual(result['n_primary_comparisons'], 12)
            metrics = pd.read_csv(run / 'summary.csv')
            self.assertTrue(metrics.pipeline_only.all())
            compare = pd.read_csv(run / 'comparisons.csv')
            self.assertEqual(set(compare.comparison_type), {'absolute_performance', 'difference_in_differences',
                                                            'reference_difference_in_differences'})
            self.assertFalse(compare.query("comparison_type == 'reference_difference_in_differences'").primary_comparison.any())
            self.assertTrue((run / 'per_family_errors.csv').exists())
            # A full label cannot turn a partial matrix into a full experiment.
            (run / 'manifest.json').write_text(json.dumps(dict(mode='full', jobs=jobs)), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'complete frozen'):
                summarize(run, bootstrap=9)
            (run / 'manifest.json').write_text(json.dumps(dict(mode='smoke', jobs=jobs)), encoding='utf-8')
            # Experimental deletions may change inputs, never scoring truths or
            # the original-reference support defining the paired population.
            path = run / 'evaluation' / jobs[-1]['name'] / 'predictions.csv.gz'
            wrong = pd.read_csv(path)
            wrong.loc[0, 'observed_b_fem'] += .01
            wrong.to_csv(path, index=False, compression='gzip')
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                summarize(run, bootstrap=9)
            def update_hash():
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                (path.parent / 'result.json').write_text(json.dumps(dict(status='complete',
                    output_sha256={'predictions.csv.gz': digest})), encoding='utf-8')
            update_hash()
            with self.assertRaisesRegex(ValueError, 'Frozen target/reference support'):
                summarize(run, bootstrap=9)
            wrong.loc[0, 'observed_b_fem'] = 1.
            wrong.loc[0, 'context_1_tip'] = 'wrong_reference'
            wrong.to_csv(path, index=False, compression='gzip')
            update_hash()
            with self.assertRaisesRegex(ValueError, 'Frozen target/reference support'):
                summarize(run, bootstrap=9)
            wrong.loc[0, 'context_1_tip'] = 'r1'
            wrong.loc[0, 'mask_seed'] = 202
            wrong.to_csv(path, index=False, compression='gzip')
            update_hash()
            with self.assertRaisesRegex(ValueError, 'mismatch: mask_seed'):
                summarize(run, bootstrap=9)


if __name__ == '__main__':
    unittest.main()
