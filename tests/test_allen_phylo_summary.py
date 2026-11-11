"""Tests for matched support and the scientific weighting of predictive error."""
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'/'allen_world'))
from summarize_phylo_baselines import coordinates, species_errors, aggregate, paired_difference


class PhyloSummaryTests(unittest.TestCase):
    def test_derived_coordinates_and_missing_bones(self):
        values = np.log10([[9., 2., 3., 5.], [4., 1., 2., 3.]])
        values[1, 2] = np.nan
        result = coordinates(values)
        self.assertAlmostEqual(result['log10_total_leg'][0], 1.)
        self.assertAlmostEqual(result['log10_tmt_over_fem'][0], np.log10(2.5))
        np.testing.assert_allclose(result['allocation_logratio_joint'][0], np.log10([.4, .6]))
        self.assertTrue(np.isnan(result['log10_total_leg'][1]))

    def test_window_species_family_and_seed_weights(self):
        identity = pd.DataFrame({'target_tip': ['a', 'a', 'b', 'c'], 'family': ['F', 'F', 'F', 'G']})
        observed = np.zeros((4, 4))
        first = np.zeros((4, 4)); first[:, 0] = [0, 2, 4, 10]
        second = first * 2
        one = species_errors(identity, observed, first, 'neural', 0, 13)
        two = species_errors(identity, observed, second, 'neural', 0, 14)
        fixed = species_errors(identity, observed, first, 'bm', 0)
        family, metrics = aggregate(pd.concat([one, two, fixed], ignore_index=True))
        score = metrics.query("target == 'm_tarsus'").set_index('method')
        # Family F has species MSE 2 and 16, G has 100. Equal family weights.
        self.assertAlmostEqual(score.loc['bm', 'mse'], 54.5)
        self.assertAlmostEqual(score.loc['neural', 'mse'], 54.5 * 2.5)
        self.assertEqual(score.loc['bm', 'n_error_realizations'], 1)
        self.assertEqual(score.loc['neural', 'n_error_realizations'], 2)
        self.assertEqual(score.loc['neural', 'n_species'], 3)

    def test_missing_bone_support_not_replaced_by_extra_external_rows(self):
        identity = pd.DataFrame({'target_tip': ['a', 'b'], 'family': ['F', 'G']})
        observed = np.array([[1., 1., 1., 1.], [1., np.nan, np.nan, np.nan]])
        errors = species_errors(identity, observed, np.ones((2, 4)), 'bm', 0)
        self.assertEqual(len(errors.query("target == 'm_tarsus'")), 2)
        self.assertEqual(len(errors.query("target == 'allocation_logratio_joint'")), 1)

    def test_support_and_fold_mismatches_fail(self):
        identity = pd.DataFrame({'target_tip': ['a', 'b'], 'family': ['F', 'G']})
        a = species_errors(identity, np.zeros((2, 4)), np.ones((2, 4)), 'a', 0)
        b = a.assign(method='b').query("target_tip == 'a'")
        with self.assertRaisesRegex(ValueError, 'unmatched'):
            aggregate(pd.concat([a, b]))
        with self.assertRaisesRegex(ValueError, 'multiple held-out folds'):
            aggregate(pd.concat([a, a.assign(fold=1)]))

    def test_bootstrap_is_paired_reproducible_and_sign_defined(self):
        first = pd.Series([1., 4., 9.], index=['F', 'G', 'H'])
        second = first * 4
        a = paired_difference(first, second, 2000)
        self.assertEqual(a, paired_difference(first, second.iloc[::-1], 2000))
        self.assertAlmostEqual(a['second_minus_first_rmse'], np.sqrt(14/3))
        self.assertGreater(a['ci95_low'], 0)
        self.assertLessEqual(a['ci975_low'], a['ci95_low'])
        self.assertGreaterEqual(a['ci975_high'], a['ci95_high'])
        with self.assertRaises(ValueError):
            paired_difference(first, second.iloc[:2])


if __name__ == '__main__':
    unittest.main()
