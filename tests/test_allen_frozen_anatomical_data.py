"""Frozen split portability and measured-only anatomical training labels."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from evowm.data.allen import AllenDataset


class FrozenAnatomicalDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        n = 30
        self.cols = ['m_tarsus', 'b_fem', 'b_tbt', 'b_tmt']
        self.table = pd.DataFrame({'tip': [f'Bird_{i}' for i in range(n)],
                                   'family': [f'F{i // 3}' for i in range(n)]})
        groups = {'temperature': ['bio1_c', 'bio5_c', 'bio6_c'],
                  'exposure': ['ground'], 'motion': ['walk'],
                  'development': ['early'], 'background': ['m_mass']}
        for j, col in enumerate(self.cols + sum(groups.values(), [])):
            self.table[col] = np.log10(np.arange(n) + j + 2.)
        self.table.loc[2, 'b_tbt'] = np.nan
        mapping = {f'F{i}': (i + 1) % 5 for i in range(10)}
        self.table['allen_frozen_family_fold'] = self.table.family.map(mapping)
        self.meta = {'morph': self.cols, 'skeleton': [], 'state_default': self.cols,
                     'eco_groups': groups, 'frozen_split': {
                         'level': 'family', 'folds': 5, 'seed': 0,
                         'require_cols': ['m_tarsus', 'm_mass'],
                         'family_to_fold': mapping, 'audit_column': 'allen_frozen_family_fold'}}
        self.write()
        (self.root / 'tree.nwk').write_text('(' + ','.join(t + ':10' for t in self.table.tip) + ');')

    def tearDown(self):
        self.temp.cleanup()

    def write(self):
        self.table.to_csv(self.root / 'traits.csv', index=False)
        (self.root / 'traits.columns.json').write_text(json.dumps(self.meta), encoding='utf8')

    def ds(self, val=False, **updates):
        args = dict(name='allen_val' if val else 'allen_train', evowm_root=str(self.root),
                    species_table='traits.csv', tree_file='tree.nwk', n_trees=1,
                    level1={'num_steps': 4, 'frameskip': 1},
                    holdout={'level': 'family', 'folds': 5, 'fold': 0, 'seed': 0},
                    n_orders=2, refresh_every=0, anatomical_targets=True,
                    frozen_fold_column='allen_frozen_family_fold')
        args.update(updates)
        return AllenDataset(**args)

    def test_frozen_mapping_is_used_without_recomputing_sort_order(self):
        with patch('evowm.data.transect.family_folds', side_effect=AssertionError('must not repartition')):
            tr, va = self.ds(), self.ds(val=True)
        expected = self.table.allen_frozen_family_fold.eq(0).to_numpy()
        np.testing.assert_array_equal(va.rows_mask, expected)
        np.testing.assert_array_equal(tr.rows_mask, ~expected)
        self.assertFalse(np.any(tr.rows_mask & va.rows_mask))
        np.testing.assert_array_equal(tr.morph_mu, va.morph_mu)

    def test_raw_anatomical_labels_obey_identity_and_missing_mask(self):
        ds = self.ds()
        complete = ds.complete_three_bones
        np.testing.assert_allclose(10 ** ds.anatomical_log_total[complete],
                                   (10 ** ds.S[complete, 1:]).sum(axis=1), rtol=1e-6)
        np.testing.assert_allclose(ds.anatomical_log_ratios[complete],
                                   ds.S[complete, 1:3] - ds.S[complete, 3:], atol=1e-6)
        self.assertFalse(complete[2])
        self.assertTrue(np.isnan(ds.anatomical_log_total[2]))
        self.assertTrue(np.isnan(ds.anatomical_log_ratios[2]).all())
        sample = ds.sample(np.random.default_rng(3))
        rows = sample['rows_level1'].numpy()
        np.testing.assert_array_equal(sample['complete_three_bones'], complete[rows])
        np.testing.assert_allclose(sample['anatomical_log_total'], ds.anatomical_log_total[rows], equal_nan=True)

    def test_fold_metadata_and_column_disagreement_is_rejected(self):
        self.table.loc[0, 'allen_frozen_family_fold'] = 4
        self.write()
        with self.assertRaisesRegex(ValueError, 'Frozen fold column'):
            self.ds()

    def test_metadata_seed_and_eligibility_cannot_silently_change(self):
        with self.assertRaisesRegex(ValueError, 'metadata'):
            self.ds(holdout={'level': 'family', 'folds': 5, 'fold': 0, 'seed': 123})
        with self.assertRaisesRegex(ValueError, 'Eligibility'):
            self.ds(require_cols=['m_tarsus', 'm_mass', 'b_tbt'])

    def test_heldout_values_do_not_change_training_scales_or_samples(self):
        before = self.ds()
        held = ~before.rows_mask
        self.table.loc[held, self.cols] += 100
        self.write()
        after = self.ds()
        np.testing.assert_array_equal(before.morph_mu, after.morph_mu)
        a, b = before.sample(np.random.default_rng(4)), after.sample(np.random.default_rng(4))
        for k in a:
            np.testing.assert_allclose(a[k].numpy(), b[k].numpy(), equal_nan=True)


if __name__ == '__main__':
    unittest.main()
