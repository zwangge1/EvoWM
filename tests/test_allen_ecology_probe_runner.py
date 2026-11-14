import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('ecology_probe_runner', ROOT/'scripts/allen_world/ecology_probe_v1.py')
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)
from ecology_probe_core import CONDITION_COLUMNS


class RunnerTests(unittest.TestCase):
    def test_matched_job_matrix(self):
        jobs = MOD.select_jobs('full', ['full', 'single', 'ou'])
        self.assertEqual(len(jobs), 30)
        self.assertEqual(len({j['name'] for j in jobs}), 30)
        self.assertEqual(len(MOD.select_jobs('smoke', ['full', 'single', 'ou'])), 3)
        with self.assertRaises(ValueError):
            MOD.select_jobs('full', ['ou', 'ou'])

    def test_selection_uses_conditions_not_morphology_and_keeps_pairs(self):
        table = pd.DataFrame(np.ones((5, len(CONDITION_COLUMNS))), columns=CONDITION_COLUMNS)
        table['tip'] = ['a', 'b', 'c', 'd', 'e']
        table.loc[3, 'bio1_c'] = np.nan
        cell = SimpleNamespace(table=table, target_rows=np.array([0, 1, 0, 2, 3, 4, 1, 0]))
        original = MOD.select_windows(cell, max_windows_per_species=2)
        np.testing.assert_array_equal(original, [0, 1, 2, 3, 5, 6])
        for c in ('b_fem', 'b_tbt', 'b_tmt'):
            table[c] = [np.nan, 999, -999, 0, 1]
        np.testing.assert_array_equal(MOD.select_windows(cell), original)
        chosen = MOD.select_windows(cell, max_species=2)
        self.assertEqual(len(set(table.tip.iloc[cell.target_rows[chosen]])), 2)
        np.testing.assert_array_equal(MOD.select_windows(cell, max_species=2), chosen)


if __name__ == '__main__':
    unittest.main()
