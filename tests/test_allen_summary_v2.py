"""Calibration summaries must use matched species/family weights and fail closed."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/allen_world"))
import summarize_v2 as summary


def inputs(n=50):
    truth = pd.DataFrame({"tip": [f"bird{i}" for i in range(n)], "family": [f"F{i % 5}" for i in range(n)],
                          "order": ["O"] * n, "mode": ["P"] * n})
    for j, bone in enumerate(summary.BONES):
        truth[f"true_slope_{bone}"] = j * 0.006 + np.arange(n) / 10000
        truth[f"observed_mask_{bone}"] = True
    win = pd.DataFrame({"target_tip": truth.tip, "family": truth.family, "development_mode": truth["mode"],
                        "order": truth.order, "temperature_axis": "joint", "within_training_temperature_marginals": True,
                        "complete_three_bones": True})
    for bone in summary.BONES:
        win[f"response_per_10C_{bone}"] = truth[f"true_slope_{bone}"]
        win[f"measured_{bone}"] = True
    return truth, win


class SummaryV2Tests(unittest.TestCase):
    def test_common_support_and_repeated_windows_do_not_reweight_truth(self):
        truth, win = inputs(10)
        # Exclude one high-truth species from both sides, then replicate another window.
        win.loc[0, "within_training_temperature_marginals"] = False
        win = pd.concat([win, pd.concat([win.iloc[[1]]] * 50)], ignore_index=True)
        paired = summary.paired_rows(win, truth, 0)
        report = summary.aggregate_recovery([paired], min_species=1, min_families=1)
        self.assertTrue(report["pass"])
        self.assertEqual(report["n_species"], 9)
        wanted = truth.iloc[1:].groupby("family").true_slope_b_tmt.mean().mean()
        self.assertAlmostEqual(report["metrics"]["b_tmt"]["truth"], wanted)
        self.assertAlmostEqual(report["metrics"]["b_tmt"]["prediction"], wanted)

    def test_empty_small_missing_flags_nonfinite_and_crossfold_duplicates_fail(self):
        self.assertFalse(summary.aggregate_recovery([])["pass"])
        truth, win = inputs(10)
        paired = summary.paired_rows(win, truth, 0)
        self.assertFalse(summary.aggregate_recovery([paired])["pass"])
        with self.assertRaisesRegex(ValueError, "Missing required"):
            summary.paired_rows(win.drop(columns="complete_three_bones"), truth, 0)
        bad = win.copy()
        bad.loc[0, "response_per_10C_b_fem"] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            summary.paired_rows(bad, truth, 0)
        with self.assertRaisesRegex(ValueError, "multiple held-out"):
            summary.aggregate_recovery([paired, summary.paired_rows(win, truth, 1)])

    def test_common_offset_cannot_pass_just_because_contrasts_match(self):
        truth, win = inputs()
        for bone in summary.BONES:
            win[f"response_per_10C_{bone}"] += 0.02
        report = summary.aggregate_recovery([summary.paired_rows(win, truth, 0)])
        self.assertFalse(report["pass"])
        self.assertTrue(report["metrics"]["b_tmt_minus_b_tbt"]["pass"])
        self.assertFalse(report["metrics"]["b_fem"]["pass"])

    def test_complete_matrix_passes_but_partial_and_real_outputs_are_retained(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "run"
            run.mkdir()
            jobs, status = [], {}
            truth, win = inputs()
            for scenario in sorted(summary.SCENARIOS):
                for seed in (13, 14):
                    table = root / "inputs" / f"{scenario}{seed}" / "species_traits.csv"
                    table.parent.mkdir(parents=True)
                    truth.to_csv(table.parent / "true_slopes.csv", index=False)
                    for fold in range(5):
                        name = f"{scenario}_{seed}_{fold}"
                        job = dict(name=name, stage="simulation", scenario=scenario, variant="flat", seed=seed, fold=fold,
                                   table=table.relative_to(root).as_posix())
                        jobs.append(job)
                        out = run / "evaluation" / name
                        out.mkdir(parents=True)
                        result = {"metrics": {"available": {"b_tmt": {"jepa_full": {"n_windows": 10, "n_species": 10,
                            "n_families": 5, "family_weighted_rmse": 0.1, "family_weighted_mean_error": 0.01}}}}}
                        (out / "pilot_results.json").write_text(json.dumps(result), encoding="utf-8")
                        win.iloc[np.arange(len(win)) % 5 == fold].to_csv(out / "temperature_windows.csv.gz", index=False)
                        status[name] = {"training": "done", "evaluation": "done"}
            real = dict(name="real", stage="real_development", scenario="observed", variant="flat", seed=13, fold=0, table="real.csv")
            jobs.append(real)
            out = run / "evaluation/real"
            out.mkdir()
            (out / "pilot_results.json").write_text(json.dumps(result), encoding="utf-8")
            status["real"] = {"training": "done", "evaluation": "done"}
            (run / "manifest.json").write_text(json.dumps({"jobs": jobs}), encoding="utf-8")
            (run / "status.json").write_text(json.dumps(status), encoding="utf-8")
            complete = summary.summarize(run, root)
            self.assertTrue(complete["calibration"]["flat"])
            first = jobs[0]["name"]
            status[first]["evaluation"] = "failed"
            (run / "status.json").write_text(json.dumps(status), encoding="utf-8")
            partial = summary.summarize(run, root)
            self.assertFalse(partial["calibration"]["flat"])
            inventory = pd.read_csv(run / "jobs.csv")
            self.assertEqual(len(inventory), len(jobs))
            self.assertEqual(inventory.loc[inventory.name == first, "evaluation"].item(), "failed")
            metrics = pd.read_csv(run / "summary.csv")
            self.assertIn("real", set(metrics.name))
            self.assertFalse(metrics.development_calibration_pass.any())


if __name__ == "__main__":
    unittest.main()
