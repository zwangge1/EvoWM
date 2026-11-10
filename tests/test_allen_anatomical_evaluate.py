"""Whole-path inference, target isolation and paired anatomical coordinates."""
import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import copy
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("anatomical_evaluate", ROOT / "scripts/allen_world/evaluate.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

COLS = ["m_tarsus", "b_fem", "b_tbt", "b_tmt"]


class FakeModel(torch.nn.Module):
    num_levels = 2

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.allen_anatomical_settings = {"mode": "hierarchical", "connection": "full", "objective": "jepa"}

    def get_level(self, level):
        raise AssertionError("New inference must not enter legacy Level1/probe path")


class FakeDataset:
    state_cols = COLS
    eco_group_names = ["temperature"]
    eco_cols = ["bio1_c", "bio5_c", "bio6_c"]
    col_group = np.array([0, 0, 0])
    mask_modes = ("full", "no_temperature")

    def __init__(self, train):
        self.S = 1. + np.arange(32).reshape(8, 4) / 100.
        self.S[7, 1] = np.nan
        self.pix = np.concatenate([np.nan_to_num(self.S), np.isfinite(self.S)], axis=1).astype(np.float32)
        self.rows_mask = np.array([True] * 4 + [False] * 4) if train else np.array([False] * 4 + [True] * 4)
        self.condition_raw = np.repeat(np.array([-20., 50., 0., 30., 10., 15., 20., 25.])[:, None], 3, axis=1)
        self.table = pd.DataFrame({"tip": [f"sp{i}" for i in range(8)], "family": [f"F{i}" for i in range(8)],
                                   "mode": ["P"] * 8, "order": ["Other"] * 8})

    def build_action(self, rows, dist, mode="full", raw_conditions=None):
        c = self.condition_raw[rows] if raw_conditions is None else raw_conditions
        action = np.concatenate([c[1:], np.zeros((1, 3))])
        return np.zeros_like(action) if mode == "no_temperature" else action

    def audit(self):
        return {"n_species": int(self.rows_mask.sum())}


class AnatomicalEvaluationTests(unittest.TestCase):
    def inference_stub(self, seen):
        def predict_from_context(model, pixels, actions):
            seen.append((pixels.detach().cpu().numpy().copy(), actions.detach().cpu().numpy().copy()))
            cumulative = pixels[..., :4].cumsum(dim=1) / torch.arange(1, pixels.shape[1] + 1)[None, :, None]
            lengths = cumulative + actions[..., :1] * pixels.new_tensor([.001, 0., .002, .004])
            return {"predicted_log_lengths": lengths}

        return predict_from_context

    def test_coordinates_use_paired_bones_and_complete_observations(self):
        x = np.log10([[9., 2., 3., 5.], [9., 2., 3., 5.]])
        x[1, 2] = np.nan
        result = ev.anatomical_coordinates(x, COLS)
        self.assertAlmostEqual(result["log10_total_leg"][0], 1.)
        self.assertAlmostEqual(result["log10_fem_over_tmt"][0], np.log10(2 / 5))
        self.assertAlmostEqual(result["log10_tbt_over_tmt"][0], np.log10(3 / 5))
        np.testing.assert_allclose(sum(10 ** result[f"log10_proportion_{b}"][0] for b in ev.BONES), 1.)
        self.assertTrue(all(np.isnan(values[1]) for values in result.values()))
        # Three log lengths of 300 are finite: never exponentiate raw mm first.
        self.assertAlmostEqual(ev.anatomical_coordinates(np.array([[1., 300., 300., 300.]]), COLS)["log10_total_leg"][0],
                               300. + np.log10(3))

    def test_backend_never_encodes_targets_or_fits_probe(self):
        model = FakeModel()
        pixels = np.arange(48, dtype=np.float32).reshape(6, 8)
        seen = []
        anatomical = importlib.import_module("evowm.models.allen_anatomical")
        with patch.object(anatomical, "predict_from_context", self.inference_stub(seen)):
            predictor, refs, decode, info = ev.prediction_backend(model, pixels, None, [], COLS)
            out = decode(predictor.predict(refs[[[0, 1, 2], [3, 4, 5]]], np.zeros((2, 3, 3)), batch=1))
        self.assertEqual(out.shape, (2, 3, 4))
        self.assertEqual(len(seen), 2)
        self.assertEqual(info["levels"], 2)
        self.assertEqual(info["connection"], "full")
        self.assertEqual(info["objective"], "jepa")
        self.assertEqual(info["kind"], "trained_anatomical_hierarchy")

    def test_joint_allocation_metric_averages_coordinate_mse_before_families(self):
        rows = np.array([0] * 20 + [1, 2])
        errors = np.array([[1., 3.]] * 20 + [[3., 1.], [5., 5.]])
        metric = ev.allocation_metric(errors, rows, pd.DataFrame({"family": ["A", "A", "B"]}))
        self.assertAlmostEqual(metric["family_weighted_rmse"], np.sqrt(15.))
        self.assertIsNone(metric["family_weighted_mean_error"])
        self.assertEqual(metric["n_species"], 3)

    def evaluate_fake(self, path, mutate_target=False, temperature=True):
        train, val = FakeDataset(True), FakeDataset(False)
        if mutate_target:
            train.pix[7, :4] += 10000.
            val.pix[7, :4] += 10000.
        wt = SimpleNamespace(rows=np.array([[0, 1, 2, 3], [3, 2, 1, 0]]), dist=np.ones((2, 3)))
        wv = SimpleNamespace(rows=np.array([[4, 5, 6, 7], [5, 4, 7, 6]]), dist=np.ones((2, 3)))
        seen = []
        # Patch the actual imported objects. Replacing sys.modules can remove
        # lazy imports on exit while leaving stale attributes on parent packages.
        anatomical = importlib.import_module("evowm.models.allen_anatomical")
        recovery = importlib.import_module("evowm.eval.recovery")
        with patch.object(anatomical, "predict_from_context", self.inference_stub(seen)), \
             patch.object(recovery, "sample_windows", side_effect=[wt, wv]):
            result = ev.evaluate(train, val, FakeModel(), {"level1": {"wm": {"history_size": 3, "rollout_n": 1}}},
                                 path, batch=16, temperature_step=1. if temperature else 0., all_temperature_axes=temperature)
        return result, seen, pd.read_csv(path / "prediction_windows.csv.gz")

    def test_full_evaluation_uses_same_inference_for_temperature_and_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result, seen, raw = self.evaluate_fake(root / "normal")
            _, _, changed = self.evaluate_fake(root / "changed", mutate_target=True, temperature=False)
            # Target 7 is never included in its own first-window context. It is
            # a reference for the second window, where changes are allowed.
            for name in [f"jepa_full_{col}" for col in COLS]:
                self.assertAlmostEqual(raw.loc[0, name], changed.loc[0, name])
            self.assertEqual(len(seen), 2 + 2 * 4)  # masks, then +/- for each temperature axis
            np.testing.assert_array_equal(seen[0][0], FakeDataset(True).pix[np.array([[4, 5, 6], [5, 4, 7]])])
            for context, _ in seen:
                np.testing.assert_array_equal(context, seen[0][0])
            self.assertFalse(np.array_equal(seen[0][1], seen[1][1]))
            self.assertIn("Full anatomical inference path", result["scope"])
            self.assertEqual(result["metrics"]["available"]["m_tarsus"]["jepa_full"]["n_species"], 2)
            for name in ev.ANATOMICAL_OUTCOMES:
                self.assertEqual(result["metrics"]["available"][name]["jepa_full"]["n_species"], 1)
            for name in ev.BONES:
                self.assertEqual(result["metrics"]["complete_three_bones"][name]["jepa_full"]["n_species"], 1)
            self.assertEqual(result["metrics"]["complete_three_bones"]["allocation_logratio_joint"]["jepa_full"]["n_species"], 1)
            thermal = pd.read_csv(root / "normal/temperature_windows.csv.gz")
            # Anatomy uses the same +/- inference and a common sample. Total
            # plus allocation response must reconstruct each bone response.
            for name in ev.BONES:
                np.testing.assert_allclose(thermal[f"response_per_10C_{name}"],
                                           thermal["response_per_10C_log10_total_leg"]
                                           + thermal[f"response_per_10C_log10_proportion_{name}"], atol=1e-12)
            self.assertEqual(result["temperature_diagnostic"]["summary_measured_outcomes_only"]["log10_fem_over_tmt"]["n_species"], 1)

    def test_predictor_rejects_bad_shape_and_nonfinite_inputs(self):
        anatomical = importlib.import_module("evowm.models.allen_anatomical")
        with patch.object(anatomical, "predict_from_context", self.inference_stub([])):
            predictor = ev.AnatomicalPredictor(FakeModel())
            with self.assertRaises(ValueError):
                predictor.predict(np.zeros((1, 3, 8)), np.zeros((1, 4, 3)))
            with self.assertRaises(ValueError):
                predictor.predict(np.full((1, 3, 8), np.nan), np.zeros((1, 3, 3)))

    def test_real_model_adapter_matches_training_inference_across_batches(self):
        from omegaconf import OmegaConf
        sys.path.insert(0, str(ROOT / "hjepa/h_jepa"))
        from evowm.models.allen_anatomical import create_world_model, predict_from_context
        torch.set_num_threads(2)
        cfg = OmegaConf.load(ROOT / "configs/hjepa/base/evowm.yaml")
        cfg.num_levels = 2
        cfg.level1.wm.embed_dim = 16
        cfg.level1.wm.action_dim = 11
        cfg.level1.encoder.input_dim = 8
        cfg.level1.encoder.hidden_dim = 16
        cfg.level1.encoder.output_dim = 16
        cfg.level1.predictor.update({"depth": 1, "heads": 2, "dim_head": 8,
                                    "hidden_dim": 16, "mlp_dim": 32, "dropout": 0.})
        cfg.level2 = copy.deepcopy(cfg.level1)
        cfg.level2.encoder.input_dim = 16
        cfg.level2.encoder.hidden_dim = 8
        cfg.level2.encoder.output_dim = 8
        cfg.level2.wm.embed_dim = 8
        cfg.level2.predictor.hidden_dim = 8
        cfg.level2.predictor.dim_head = 4
        cfg.level2.action_embed.input_dim = 11
        cfg.evowm_allen_anatomical = {"mode": "hierarchical", "connection": "full", "objective": "jepa", "hidden_dim": 8}
        model, _ = create_world_model(cfg)
        model.get_level(1).anatomical_readout.configure(torch.tensor([[1., 1.1, 1.2, 1.3],
                                                                    [1.2, 1.3, 1.5, 1.8], [1.5, 1.7, 1.8, 2.]]))
        rng = np.random.default_rng(0)
        context = rng.normal(size=(5, 3, 8)).astype(np.float32)
        actions = rng.normal(size=(5, 3, 11)).astype(np.float32)
        predictor = ev.AnatomicalPredictor(model)
        actual = predictor.predict(context, actions, batch=2)
        with torch.no_grad():
            expected = predict_from_context(model, torch.from_numpy(context), torch.from_numpy(actions))["predicted_log_lengths"].numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
