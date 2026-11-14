"""Numerical and information-boundary checks for frozen ecology backends."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/allen_world'))
sys.path.insert(0, str(ROOT / 'src'))
from ecology_probe_backends import (ClassicBackend, FrozenConditionDesign,
                                    inspect_checkpoint_paths, validate_target_changes)
from phylo_baseline_data import (CONDITION_COLUMNS, DEVELOPMENT_COLUMNS,
                                FORAGING_COLUMNS, MOTION_COLUMNS, build_condition_design)


def conditions(n=240, seed=91):
    rng = np.random.default_rng(seed)
    data = pd.DataFrame(rng.normal(size=(n, len(CONDITION_COLUMNS))), columns=CONDITION_COLUMNS)
    data[list(FORAGING_COLUMNS)] = rng.dirichlet(np.ones(7), size=n)
    data.loc[::9, list(FORAGING_COLUMNS)] *= .99
    for cols in (DEVELOPMENT_COLUMNS, MOTION_COLUMNS):
        data[list(cols)] = np.eye(len(cols))[rng.integers(0, len(cols), size=n)]
    data['tip'] = [f's{i}' for i in range(n)]
    data['family'] = ['train' if i < n - 20 else 'heldout' for i in range(n)]
    return data


class EcologyProbeBackendTests(unittest.TestCase):
    def test_saved_design_reproduces_original_training_basis_and_missingness(self):
        data = conditions()
        data.loc[[3, 20, 90], 'e_habdens'] = np.nan
        data.loc[[4, 21, 91], list(FORAGING_COLUMNS)] = np.nan
        x, description = build_condition_design(data, np.arange(220), rank_support_rows=np.arange(200))
        rebuilt = FrozenConditionDesign(json.loads(json.dumps(description)))
        got = rebuilt.transform(data[list(CONDITION_COLUMNS)].to_numpy())
        np.testing.assert_allclose(got, x, rtol=0, atol=1e-14)
        changed = data[list(CONDITION_COLUMNS)].to_numpy()
        changed[230, CONDITION_COLUMNS.index('bio1_c')] += 1
        expected = changed[230, CONDITION_COLUMNS.index('bio1_c')]
        self.assertTrue(np.isfinite(rebuilt.transform(changed)).all())
        self.assertEqual(expected, changed[230, CONDITION_COLUMNS.index('bio1_c')])

    def test_probe_rejects_mass_mask_and_composition_changes(self):
        raw = conditions(12)[list(CONDITION_COLUMNS)].to_numpy()
        ground = CONDITION_COLUMNS.index('e_fs_ground')
        canopy = CONDITION_COLUMNS.index('e_fs_canopy')
        change = raw.copy()
        amount = np.minimum(.01, change[:, canopy] / 2)
        change[:, ground] += amount
        change[:, canopy] -= amount
        change[:, CONDITION_COLUMNS.index('bio1_c')] += 1
        np.testing.assert_array_equal(validate_target_changes(raw, change), change)
        invalid = change.copy()
        invalid[0, CONDITION_COLUMNS.index('m_mass')] += .1
        with self.assertRaisesRegex(ValueError, 'non-ecology'):
            validate_target_changes(raw, invalid)
        invalid = change.copy()
        invalid[0, ground] += .1
        with self.assertRaisesRegex(ValueError, 'composition sum'):
            validate_target_changes(raw, invalid)
        invalid = change.copy()
        invalid[0, ground] = np.nan
        with self.assertRaisesRegex(ValueError, 'observation mask'):
            validate_target_changes(raw, invalid)
        np.testing.assert_array_equal(raw, conditions(12)[list(CONDITION_COLUMNS)].to_numpy())

    def test_classic_fast_update_matches_full_nonzero_ou_posterior(self):
        from evowm.data.tree import read_trees
        from phylo_baseline_core import PhyloKernel, fit_nested_fixed, predict_joint
        from asymmetry_v1_evaluate import visible_references
        data = conditions()
        train = np.arange(220)
        x, design = build_condition_design(data, train, rank_support_rows=np.arange(205))
        rng = np.random.default_rng(812)
        beta = rng.normal(scale=.03, size=(x.shape[1], 4))
        states = x @ beta + rng.normal(scale=.07, size=(len(data), 4))
        states[205:220, 1:] = np.nan
        states[222, 2:] = np.nan
        refs = np.array([[220, 221, 222], [221, 222, 223], [224, 225, 226]])
        target = np.array([230, 231, 232])
        def branch(ids):
            if len(ids) == 1:
                return f's{ids[0]}:1'
            mid = len(ids) // 2
            return '(' + branch(ids[:mid]) + ',' + branch(ids[mid:]) + '):1'
        with tempfile.TemporaryDirectory() as temp:
            tree_path = Path(temp) / 'tree.nwk'
            tree_path.write_text(branch(list(range(len(data)))) + ';', encoding='utf-8')
            tree = read_trees(tree_path)[0]
            tip_lookup = {name: i for i, name in enumerate(tree.tip_names)}
            ids = np.array([tip_lookup[name] for name in data.tip])
            kernel = PhyloKernel(tree)
            fit = fit_nested_fixed(kernel, ids[train], x[train], states[train], alpha=.7, eta=.08)
            fit['model'] = 'OU'
            cell = SimpleNamespace(table=data, states=states, train_rows=train,
                reference_rows=refs, target_rows=target, tip_ids_of_rows=ids,
                tree_path=tree_path, job={'mask_seed': 101, 'fold': 0})
            visible = visible_references(cell, 'observed')
            original = np.array([predict_joint(kernel, fit, ids[ref], x[ref], visible[i],
                                ids[t:t+1], x[t:t+1], ref_mask=np.isfinite(visible[i]))[0]
                                 for i, (ref, t) in enumerate(zip(refs, target))])
            backend = ClassicBackend(cell, 'ou', [2, 0], {}, fit, design, original)
            np.testing.assert_allclose(backend.baseline_predictions, original[[2, 0]], rtol=0, atol=1e-12)
            changed = backend.target_conditions.copy()
            j = CONDITION_COLUMNS.index('e_fs_ground')
            k = CONDITION_COLUMNS.index('e_fs_canopy')
            shift = np.minimum(.02, changed[:, k] / 2)
            changed[:, j] += shift
            changed[:, k] -= shift
            changed[:, CONDITION_COLUMNS.index('bio1_c')] += .5
            new_x = backend.design.transform(changed)
            expected = np.array([predict_joint(kernel, fit, ids[refs[w]], x[refs[w]], visible[w],
                ids[target[w]:target[w]+1], new_x[i:i+1], ref_mask=np.isfinite(visible[w]))[0]
                for i, w in enumerate([2, 0])])
            np.testing.assert_allclose(backend.predict(changed), expected, rtol=0, atol=1e-12)
            # Target measurements do not participate in the probe prediction.
            cell.states[target] = 1e30
            np.testing.assert_allclose(backend.predict(changed), expected, rtol=0, atol=1e-12)
            self.assertGreater(backend.audit['baseline_gate_windows'], backend.audit['selected_windows'])

    def test_preflight_reports_missing_original_checkpoint_and_rejects_changed_config(self):
        from asymmetry_v1_evaluate import sha
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / 'outputs/run'
            job = dict(name='full_f0_r0_complete100', condition='complete100', fold=0,
                       seed=13, replicate=0, variant='full', kind='neural')
            ev = run / 'evaluation' / job['name']
            ck = run / 'training/ckpts' / job['name']
            ev.mkdir(parents=True); ck.mkdir(parents=True)
            (run / 'manifest.json').write_text(json.dumps({'kind': 'allen_asymmetric_data_v1', 'jobs': [job]}))
            (ev / 'predictions.csv.gz').write_bytes(b'predictions')
            (ck / 'config.yaml').write_bytes(b'config')
            result = {'status': 'complete', 'job': job,
                'checkpoint': '/mnt/server/run/training/ckpts/' + job['name'] + '/allen_asymmetry_v1_object.ckpt',
                'config': '/mnt/server/run/training/ckpts/' + job['name'] + '/config.yaml',
                'checkpoint_sha256': '0' * 64, 'config_sha256': sha(ck / 'config.yaml'),
                'output_sha256': {'predictions.csv.gz': sha(ev / 'predictions.csv.gz')}}
            (ev / 'result.json').write_text(json.dumps(result))
            got = inspect_checkpoint_paths(run, 0, 13, 'full', root)
            self.assertFalse(got['ready'])
            self.assertEqual([Path(p) for p in got['missing']],
                             [(ck / 'allen_asymmetry_v1_object.ckpt').resolve()])
            (ck / 'config.yaml').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                inspect_checkpoint_paths(run, 0, 13, 'full', root)


if __name__ == '__main__':
    unittest.main()
