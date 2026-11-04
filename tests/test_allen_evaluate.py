"""Guard against repeated-window weighting and accidental counterfactual edits."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("allen_evaluate", ROOT / "scripts/allen_world/evaluate.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


class EvaluationTests(unittest.TestCase):
    def test_window_count_does_not_reweight_species_or_family(self):
        # Family A: one species with RMSE 1, one with RMSE 3 => MSE 5.
        # Family B: one species with RMSE 5 => MSE 25. Equal family MSE = 15.
        rows = np.array([0] * 100 + [1, 2])
        errors = np.array([1.] * 100 + [3., 5.])
        metric = ev.family_metric(errors, rows, pd.DataFrame({"family": ["A", "A", "B"]}))
        self.assertAlmostEqual(metric["family_weighted_rmse"], np.sqrt(15.))
        self.assertEqual(metric["n_species"], 3)
        self.assertEqual(metric["n_families"], 2)

    def test_temperature_only_final_target_condition_changes(self):
        class FakeDataset:
            eco_group_names = ["temperature", "motion", "background"]
            col_group = np.array([0, 0, 0, 1, 2])
            condition_raw = np.arange(20, dtype=float).reshape(4, 5)
            seen = []
            def build_action(self, rows, dist, mode, raw_conditions):
                self.seen.append(raw_conditions.copy())
                return np.zeros((4, 4))
        ds = FakeDataset()
        w = SimpleNamespace(rows=np.array([[0, 1, 2, 3]]), dist=np.ones((1, 3)))
        act = ev.action_windows(ds, w, 3, temperature_delta=1.)
        expected = ds.condition_raw.copy()
        expected[-1, :3] += 1
        np.testing.assert_array_equal(ds.seen[0], expected)
        self.assertEqual(act.shape, (1, 3, 4))
        np.testing.assert_array_equal(ds.condition_raw, np.arange(20).reshape(4, 5))

    def test_ratio_uses_paired_predictions_not_independent_ratio_fit(self):
        cols = ["m_tarsus", "b_fem", "b_tbt", "b_tmt"]
        result = ev.with_ratio(np.array([[1., 2., 3., 4.], [1., np.nan, 3., 4.]]), cols)
        self.assertEqual(result[0, -1], 2.)
        self.assertTrue(np.isnan(result[1, -1]))

    def test_config_resolver_does_not_execute_code(self):
        self.assertEqual(ev.arithmetic("3 + 1"), 4)
        with self.assertRaises(ValueError):
            ev.arithmetic("__import__('os').getcwd()")

    def test_single_temperature_axis_leaves_other_temperatures_unchanged(self):
        class FakeDataset:
            eco_group_names = ["temperature", "motion", "background"]
            eco_cols = ["bio1_c", "bio5_c", "bio6_c", "motion", "mass"]
            col_group = np.array([0, 0, 0, 1, 2])
            condition_raw = np.arange(20, dtype=float).reshape(4, 5)
            def build_action(self, rows, dist, mode, raw_conditions):
                self.seen = raw_conditions.copy()
                return np.zeros((4, 4))
        ds = FakeDataset()
        w = SimpleNamespace(rows=np.array([[0, 1, 2, 3]]), dist=np.ones((1, 3)))
        ev.action_windows(ds, w, 3, temperature_delta=1., temperature_axis="bio5_c")
        expected = ds.condition_raw.copy()
        expected[-1, 1] += 1
        np.testing.assert_array_equal(ds.seen, expected)
        with self.assertRaises(ValueError):
            ev.temperature_indices(ds, "unknown")

    def test_trained_head_is_used_without_fitting_a_probe(self):
        import torch
        class Head(torch.nn.Module):
            variant = "test"
            def forward(self, z):
                return z[..., :1].expand(*z.shape[:-1], 4) + 10
        model = SimpleNamespace(get_level=lambda level: SimpleNamespace(allen_readout=Head()))
        # Empty state/train rows would make legacy probe fitting fail.
        decode, info = ev.prediction_decoder(model, None, np.empty((0, 4)), np.array([], dtype=int),
                                             ["m_tarsus", "b_fem", "b_tbt", "b_tmt"], batch=2)
        z = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
        result = decode(z)
        np.testing.assert_array_equal(result, np.repeat(z[..., :1] + 10, 4, axis=-1))
        self.assertEqual(info["kind"], "trained_anatomical_readout")

    def test_primary_p_group_is_not_replaced_by_combined_early_group(self):
        table = pd.DataFrame({"mode": ["P", "SP", "A", "SA"],
                              "order": ["Other", "Other", "Passeriformes", "Other"]})
        groups = ev.biological_groups(table, np.arange(4))
        np.testing.assert_array_equal(groups["P_nonpasserine"], [True, False, False, False])
        np.testing.assert_array_equal(groups["SP_nonpasserine"], [False, True, False, False])
        self.assertEqual(groups["early_P_SP"].sum(), 2)

    def test_thermal_bone_contrasts_share_complete_three_bone_support(self):
        class FakeDataset:
            state_cols = ["m_tarsus", "b_fem", "b_tbt", "b_tmt"]
            eco_group_names = ["temperature"]
            eco_cols = ["bio1_c", "bio5_c", "bio6_c"]
            col_group = np.array([0, 0, 0])
            condition_raw = np.array([[0., 0., 0.], [30., 30., 30.], [15., 15., 15.], [20., 20., 20.]])
            rows_mask = np.array([True, True, False, False])
            table = pd.DataFrame({"mode": ["A", "A", "P", "SP"], "order": ["Other"] * 4,
                                  "family": ["F0", "F1", "F2", "F3"]})
            def build_action(self, rows, dist, mode, raw_conditions):
                return np.zeros((len(rows), 2))
        class Predictor:
            def predict(self, embeddings, actions, batch):
                return np.ones((len(actions), 1, 4))
        ds = FakeDataset()
        windows = SimpleNamespace(rows=np.array([[0, 2], [1, 3]]), dist=np.ones((2, 1)))
        raw = pd.DataFrame({"window": [0, 1], "target_row": [2, 3], "target_tip": ["a", "b"], "family": ["F2", "F3"]})
        observed = np.array([[1., 1., 1., 1., 0.], [1., np.nan, 1., 1., np.nan]])
        with tempfile.TemporaryDirectory() as tmp:
            report = ev.temperature_diagnostic(ds, ds, Predictor(), lambda z: z, np.ones((2, 1, 4)), windows,
                                               observed, ds.state_cols + ["log10_tmt_over_fem"], raw,
                                               Path(tmp), "joint", 1., 8, 1, primary=True)
        summary = report["summary_measured_outcomes_only"]
        for key in ("b_fem", "b_tbt", "b_tmt", "b_tmt_minus_b_tbt", "b_tbt_minus_b_fem", "log10_total_leg"):
            self.assertEqual(summary[key]["n_species"], 1)
        self.assertEqual(summary["m_tarsus"]["n_species"], 2)

    def test_observational_ols_uses_species_means_and_common_bone_sample(self):
        rng = np.random.default_rng(7)
        n = 40
        table = pd.DataFrame({"mode": ["P"] * n, "order": ["Other"] * n,
                              "family": [f"F{i // 2}" for i in range(n)], "m_mass": rng.normal(size=n),
                              "bio1_c": rng.normal(15, 5, n), "e_habdens": rng.normal(size=n), "e_migr": rng.normal(size=n)})
        for name in ("aerial", "aquatic", "generalist", "terrestrial"):
            table["e_life_" + name] = 0.
        slopes = np.array([0., 0.01, 0.03])
        y = table.m_mass.to_numpy()[:, None] * .3 + table.bio1_c.to_numpy()[:, None] / 10 * slopes
        values = np.column_stack([np.ones(n), y])
        rows = np.r_[np.arange(n), np.zeros(15, dtype=int)]
        report = ev.observational_gradient(values[rows], values[rows], rows, table,
                                          ["m_tarsus", "b_fem", "b_tbt", "b_tmt"])
        self.assertEqual(report["n_species"], n)
        self.assertAlmostEqual(report["predicted"]["b_tmt_minus_b_tbt"], .02)
        self.assertEqual(report["observed"], report["predicted"])
        table["m_mass"] = table["bio1_c"]
        report = ev.observational_gradient(values[rows], values[rows], rows, table,
                                          ["m_tarsus", "b_fem", "b_tbt", "b_tmt"])
        self.assertEqual(report["status"], "unidentifiable_or_insufficient")


if __name__ == "__main__":
    unittest.main()
