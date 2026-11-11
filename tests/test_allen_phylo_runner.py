"""End-to-end frozen-fold and evidence integrity checks on a small tree."""
from pathlib import Path
import sys
import json
import tempfile
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'/'allen_world'))
sys.path.insert(0, str(ROOT/'src'))
from phylo_baselines_v1 import save_json, sha, run_fold, completed_result, verify_snapshot, inside


def setup_run(root, target_shift=0., leak=False):
    for name in ('inputs', 'folds', 'source'):
        (root/name).mkdir()
    rng = np.random.default_rng(2026)
    def branch(ids):
        return (f's{ids[0]}' if len(ids) == 1 else '('+branch(ids[:len(ids)//2])+','+branch(ids[len(ids)//2:])+')')+':1'
    (root/'inputs'/'tree.nwk').write_text(branch(list(range(36)))+';', encoding='utf-8')
    X = np.column_stack([np.ones(36), rng.normal(size=36)])
    states = X @ rng.normal(size=(2, 4)) + rng.normal(size=(36, 4)) * .3
    states[20:32, 1:] = np.nan
    states[35] += target_shift  # Neither training nor reference: must never affect prediction.
    refs = np.array([[32, 33, 35 if leak else 34]])
    np.savez_compressed(root/'inputs'/'fold_0.npz', X=X, states=states, train_rows=np.arange(32),
                        reference_rows=refs, target_rows=np.array([35]), tip_ids_of_rows=np.arange(36))
    frame = pd.DataFrame({'window': [0], 'target_tip': ['s35'], 'source_tip': ['s34'], 'family': ['test'],
                          'context_0_tip': ['s32'], 'context_1_tip': ['s33'], 'context_2_tip': ['s34']})
    for j, col in enumerate(('m_tarsus', 'b_fem', 'b_tbt', 'b_tmt')):
        frame['observed_'+col] = states[35, j]
    frame.to_csv(root/'inputs'/'fold_0_windows.csv.gz', index=False)
    manifest = {'folds': [0], 'source_sha256': {}, 'frozen_sha256':
                {p.relative_to(root).as_posix(): sha(p) for p in (root/'inputs').iterdir()}}
    save_json(root/'manifest.json', manifest)
    return manifest


class PhyloRunnerTests(unittest.TestCase):
    def test_real_fold_pipeline_target_invariance_and_completed_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            a, b = base/'a', base/'b'; a.mkdir(); b.mkdir()
            setup_run(a); setup_run(b, target_shift=100.)
            run_fold(a, 0); run_fold(b, 0)
            first = pd.read_csv(a/'folds'/'fold_0'/'predictions.csv.gz')
            second = pd.read_csv(b/'folds'/'fold_0'/'predictions.csv.gz')
            cols = [c for c in first if c.startswith(('bm_', 'ou_'))]
            np.testing.assert_array_equal(first[cols], second[cols])
            result = completed_result(a, 0)
            self.assertEqual(result['n_training'], 32)
            self.assertEqual(result['n_training_bones'], 20)
            self.assertGreaterEqual(result['fits']['OU']['loglik']+1e-8, result['fits']['BM']['loglik'])
            with (a/'folds'/'fold_0'/'predictions.csv.gz').open('ab') as f:
                f.write(b'corrupted')
            with self.assertRaisesRegex(ValueError, 'output changed'):
                completed_result(a, 0)

    def test_target_reference_leakage_is_blocked(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp); setup_run(run, leak=True)
            with self.assertRaisesRegex(ValueError, 'leakage'):
                run_fold(run, 0)

    def test_frozen_input_and_path_boundaries(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp); manifest = setup_run(run)
            verify_snapshot(run, manifest)
            (run/'inputs'/'tree.nwk').write_text('(altered:1,tree:1);')
            with self.assertRaisesRegex(ValueError, 'Frozen input changed'):
                verify_snapshot(run, manifest)
            with self.assertRaises(ValueError):
                inside(run/'..'/'outside', run)
            with self.assertRaises(ValueError):
                inside(run, run)


if __name__ == '__main__':
    unittest.main()
