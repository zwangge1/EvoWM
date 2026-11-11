"""Leakage, fair-reference visibility and actual classic-fit regression tests."""
from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "allen_world"))
sys.path.insert(0, str(ROOT / "src"))
import asymmetry_v1_evaluate as evaluation
from phylo_baseline_data import CONDITION_COLUMNS


def setup_run(root, target_shift=0., leak=False):
    run = root / "run"
    (run / "inputs").mkdir(parents=True)
    rng = np.random.default_rng(812)
    n = 64
    def branch(ids):
        return (f"s{ids[0]}" if len(ids) == 1 else
                "(" + branch(ids[:len(ids)//2]) + "," + branch(ids[len(ids)//2:]) + ")") + ":1"
    (run / "inputs" / "tree.nwk").write_text(branch(list(range(n))) + ";", encoding="utf-8")
    table = pd.DataFrame({"tip": [f"s{i}" for i in range(n)],
                          "family": [f"train{i//3}" if i < 54 else "heldout" for i in range(n)]})
    for col in CONDITION_COLUMNS:
        table[col] = 0.
    table["m_mass"] = rng.normal(size=n)
    table["bio1_c"] = rng.normal(size=n)
    X = np.column_stack([np.ones(n), table.m_mass, table.bio1_c,
                         table.m_mass * table.bio1_c])
    states = X @ rng.normal(size=(4, 4)) * .15 + rng.normal(size=(n, 4)) * .12
    states[40:54, 1:] = np.nan
    states[60:62] += target_shift
    for j, col in enumerate(evaluation.STATE_COLUMNS):
        table[col] = states[:, j]
    table.to_csv(run / "inputs" / "table.csv", index=False)
    refs = np.array([[54, 55, 60 if leak else 56], [55, 56, 57]])
    np.savez_compressed(run / "inputs" / "fold.npz", train_rows=np.arange(54),
        validation_rows=np.arange(54, n), reference_rows=refs, target_rows=np.array([60, 61]),
        distances=np.ones((2, 3)), tip_ids_of_rows=np.arange(n), order_ids=np.zeros(2, dtype=int),
        kept_train_bone_rows=np.arange(40), hidden_train_bone_rows=np.arange(40, 54))
    identity = {"condition": "bone_half", "fold": 0, "replicate": 0, "mask_seed": 917, "seed": 13}
    entry = {**identity, "table": "run/inputs/table.csv", "tree": "run/inputs/tree.nwk",
             "npz": "run/inputs/fold.npz", "mask_table": "run/inputs/masks.csv"}
    manifest = {"inputs": {"tables": {"cell": entry}},
                "jobs": [{**identity, "name": "classic", "kind": "classic", "variant": "classic", "input_key": "cell"}]}
    evaluation.save_json(run / "manifest.json", manifest)
    return run


class AsymmetryEvaluateTests(unittest.TestCase):
    def test_disconnected_checkpoint_cannot_be_scored_as_full_hierarchy(self):
        full = dict(mode="hierarchical", connection="full", objective="jepa")
        evaluation.validate_architecture(full, "full", 2)
        with self.assertRaisesRegex(ValueError, "architecture"):
            evaluation.validate_architecture(dict(full, connection="none"), "full", 2)
        with self.assertRaisesRegex(ValueError, "architecture"):
            evaluation.validate_architecture(full, "single", 2)
        with self.assertRaisesRegex(ValueError, "architecture"):
            evaluation.validate_architecture(dict(full, objective="supervised"), "full", 2)

    def test_hidden_reference_erases_value_and_mask_without_mutation(self):
        pixels = np.array([[[9., 8., 7., 0., 1., 1., 1., 0.],
                            [5., 4., 3., 2., 1., 1., 1., 1.]]])
        before = pixels.copy()
        available = np.array([[[True, False, False, False], [True, True, True, False]]])
        masked = evaluation.mask_reference_pixels(pixels, available)
        np.testing.assert_array_equal(pixels, before)
        np.testing.assert_array_equal(masked[..., 4:], available)
        self.assertTrue((masked[..., :4][~available] == 0).all())
        self.assertTrue((masked[..., :4][available] == pixels[..., :4][available]).all())
        poisoned = pixels.copy()
        poisoned[..., :4][~available] = 1e20
        np.testing.assert_array_equal(evaluation.mask_reference_pixels(poisoned, available), masked)
        available[0, 0, 3] = True
        with self.assertRaisesRegex(ValueError, "naturally missing"):
            evaluation.mask_reference_pixels(pixels, available)

    def test_frozen_split_and_target_reference_leakage_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = setup_run(root, leak=True)
            with self.assertRaisesRegex(ValueError, "leakage"):
                evaluation.load_cell(run, "classic", root)

    def test_training_family_overlap_and_hidden_labels_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = setup_run(root)
            table = pd.read_csv(run / "inputs" / "table.csv")
            table.loc[54, "family"] = table.loc[0, "family"]
            table.to_csv(run / "inputs" / "table.csv", index=False)
            with self.assertRaisesRegex(ValueError, "families overlap"):
                evaluation.load_cell(run, "classic", root)
            table.loc[54, "family"] = "heldout"
            table.loc[40, "b_fem"] = 1.
            table.to_csv(run / "inputs" / "table.csv", index=False)
            with self.assertRaisesRegex(ValueError, "triplet observation"):
                evaluation.load_cell(run, "classic", root)

    def test_predictions_keep_original_support_and_score_truth_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cell = evaluation.load_cell(setup_run(root), "classic", root)
            prediction = np.ones((2, 4))
            available = np.isfinite(cell.states[cell.reference_rows])
            available[..., 1:] = False
            frame = evaluation.prediction_frame(cell, prediction, "BM", "hide_all_bones", available)
            self.assertTrue(frame.has_reference_bones.all())
            self.assertTrue((frame.visible_reference_bone_values == 0).all())
            self.assertTrue(frame.complete_three_bones.all())
            np.testing.assert_array_equal(frame[["predicted_" + c for c in evaluation.STATE_COLUMNS]], prediction)
            np.testing.assert_array_equal(frame[["observed_" + c for c in evaluation.STATE_COLUMNS]],
                                          cell.states[cell.target_rows])

    def test_real_classical_fit_never_conditions_on_target_truth(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            a, b = base / "a", base / "b"
            a.mkdir(); b.mkdir()
            run_a, run_b = setup_run(a), setup_run(b, target_shift=100.)
            first = evaluation.evaluate_classic(run_a, "classic", a)
            second = evaluation.evaluate_classic(run_b, "classic", b)
            pred_a = pd.read_csv(run_a / "evaluation" / "classic" / "predictions.csv.gz")
            pred_b = pd.read_csv(run_b / "evaluation" / "classic" / "predictions.csv.gz")
            columns = ["predicted_" + c for c in evaluation.STATE_COLUMNS]
            np.testing.assert_array_equal(pred_a[columns], pred_b[columns])
            self.assertEqual(first["n_prediction_rows"], 16)
            self.assertEqual(first["n_training_bone_species"], 40)
            self.assertEqual(first["fits"], second["fits"])
            self.assertGreaterEqual(first["fits"]["OU"]["loglik"] + 1e-6,
                                    first["fits"]["BM"]["loglik"])
            self.assertEqual(set(pred_a.reference_case), set(evaluation.REFERENCE_CASES))
            self.assertEqual(set(pred_a.model), {"BM", "OU"})
            for name, digest in first["output_sha256"].items():
                self.assertEqual(evaluation.sha(run_a / "evaluation" / "classic" / name), digest)
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                evaluation.evaluate_classic(run_a, "classic", a)


if __name__ == "__main__":
    unittest.main()
