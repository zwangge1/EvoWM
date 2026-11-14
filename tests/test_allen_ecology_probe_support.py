"""Tests for training-only support and rejection of implausible joint inputs."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('ecology_probe_support', ROOT/'scripts/allen_world/ecology_probe_support.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
COLS = ('x', 'y', 'constant', 'e_dev_P', 'e_dev_SP', 'e_dev_SA', 'e_dev_A')


def fixture(n=30):
    x = np.linspace(-2., 2., n)
    return pd.DataFrame({'tip': ['Bird_'+str(i) for i in range(n)],
                         'family': ['Fam_'+str(i % 5) for i in range(n)],
                         'mode': ['A'] * n, 'x': x, 'y': x, 'constant': np.ones(n),
                         'e_dev_P': np.zeros(n), 'e_dev_SP': np.zeros(n),
                         'e_dev_SA': np.zeros(n), 'e_dev_A': np.ones(n),
                         'b_fem': np.ones(n), 'b_tbt': np.ones(n), 'b_tmt': np.ones(n)})


class JointSupportTests(unittest.TestCase):
    def fit(self, d, rows=None):
        return module.JointSupport.fit(d, np.arange(len(d)) if rows is None else rows,
                                       condition_columns=COLS, k=5)

    def test_joint_screen_rejects_marginally_valid_but_inconsistent_pair(self):
        d = fixture()
        s = self.fit(d)
        raw = d.loc[[15, 15], COLS].to_numpy().copy()
        raw[1, :2] = [-1.8, 1.8]
        result = s.evaluate(raw, ['A', 'A'])
        self.assertTrue(result.loc[0, 'supported'])
        self.assertTrue(result.loc[1, 'within_marginal_range'])
        self.assertFalse(result.loc[1, 'supported'])
        self.assertEqual(result.loc[1, 'reason'], 'outside_training_joint_neighbourhood')

    def test_unseen_condition_cannot_disappear_in_constant_column(self):
        d = fixture(); s = self.fit(d)
        raw = d.loc[[15], COLS].to_numpy().copy()
        raw[0, 2] = 2.
        result = s.evaluate(raw, ['A'])
        self.assertFalse(result.loc[0, 'supported'])
        self.assertEqual(result.loc[0, 'reason'], 'outside_training_marginal_range')

    def test_holdout_values_and_bone_magnitudes_do_not_calibrate_support(self):
        d = fixture(35); rows = np.arange(30)
        a = self.fit(d, rows)
        d.loc[30:, ['x', 'y']] = 1e8
        d.loc[:, ['b_fem', 'b_tbt', 'b_tmt']] = 1e6
        b = self.fit(d, rows)
        self.assertEqual(a.metadata(), b.metadata())

    def test_missing_training_bones_excluded_and_insufficient_groups_recorded(self):
        d = fixture(15)
        d.loc[:5, 'b_fem'] = np.nan
        s = self.fit(d)
        self.assertEqual(s.metadata()['n_training_support_rows'], 9)
        self.assertFalse(s.metadata()['groups']['A']['supported'])
        self.assertEqual(s.evaluate(d.loc[[10], COLS], ['A']).loc[0, 'reason'], 'too_few_training_species')

    def test_development_groups_and_malformed_indicators(self):
        d = fixture(); s = self.fit(d)
        raw = d.loc[[15, 15], COLS].to_numpy().copy()
        raw[1, -1] = np.nan
        out = s.evaluate(raw, ['P', 'A'])
        self.assertEqual(out.loc[0, 'reason'], 'development_label_mismatch')
        self.assertEqual(out.loc[1, 'reason'], 'nonfinite_conditions')
        d.loc[0, 'mode'] = 'P'
        with self.assertRaisesRegex(ValueError, 'one-hot mismatch'):
            self.fit(d)

    def test_threshold_matches_explicit_leave_one_out_distances(self):
        d = fixture(); s = self.fit(d)
        info = s.metadata()['groups']['A']
        raw = d[list(COLS)].to_numpy(float)
        mean, sd = np.asarray(info['mean']), np.asarray(info['sd'])
        active = sd > 1e-12
        z = (raw[:, active]-mean[active])/sd[active]/np.sqrt(active.sum())
        distances = np.linalg.norm(z[:, None]-z[None, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        kth = np.sort(distances, axis=1)[:, 4]
        np.testing.assert_allclose(info['calibration_distances'], kth, atol=1e-12)
        self.assertAlmostEqual(info['threshold'], np.quantile(kth, .95))

    def test_coincident_training_species_and_metadata_copy(self):
        d = fixture(); d[['x', 'y']] = 0.
        s = self.fit(d)
        self.assertEqual(s.metadata()['groups']['A']['threshold'], 0.)
        self.assertTrue(s.evaluate(d.loc[[0], COLS], ['A']).loc[0, 'supported'])
        changed = s.metadata(); changed['groups']['A']['threshold'] = 100.
        self.assertEqual(s.metadata()['groups']['A']['threshold'], 0.)


if __name__ == '__main__':
    unittest.main()
