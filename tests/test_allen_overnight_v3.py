"""V3 queue smoke/full intent, input immutability, and safe resume contracts."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("overnight_v3", ROOT / "scripts/allen_world/overnight_v3.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class V3RunnerTests(unittest.TestCase):
    def args(self, *extra):
        return runner.resolve_settings(runner.parser().parse_args(["--gpus", "4", "5", *extra]))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.table = self.root / "observed/species_traits.csv"
        self.table.parent.mkdir()
        self.table.write_text("tip,family,allen_frozen_family_fold\nA,F0,0\nB,F1,1\n", encoding="utf-8")
        self.tree = self.root / "tree.nwk"
        self.tree.write_text("(A:1,B:1);\n", encoding="utf-8")
        self.meta = {"table_sha256": runner.digest(self.table),
                     "frozen_split": {"level": "family", "folds": 5, "seed": 0,
                                      "family_to_fold": {"F0": 0, "F1": 1}}}
        self.table.with_suffix(".columns.json").write_text(json.dumps(self.meta), encoding="utf-8")
        for variant in runner.CONFIGS:
            p = self.root / "configs/hjepa" / (runner.CONFIGS[variant] + ".yaml")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("output_model_name: allen_v3\n", encoding="utf-8")
        (self.root / "scripts").mkdir()
        (self.root / "scripts/train_hjepa.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def new_plan(self):
        args = self.args("--stages", "real_development", "--table", str(self.table), "--tree", str(self.tree))
        return args, *runner.create_plan(args, self.root)

    def test_default_smoke_and_explicit_full_matrix(self):
        smoke = self.args()
        self.assertEqual(smoke.mode, "smoke")
        self.assertEqual(smoke.sim_steps, 64)
        jobs = runner.make_jobs(smoke.variants, smoke.scenarios, smoke.seeds, smoke.folds, smoke.stages)
        self.assertEqual(len(jobs), 10)
        full = self.args("--full")
        jobs = runner.make_jobs(full.variants, full.scenarios, full.seeds, full.folds, full.stages)
        self.assertEqual(len(jobs), 350)
        self.assertEqual(len({j["name"] for j in jobs}), 350)
        self.assertEqual(sum(j["stage"] == "simulation" for j in jobs), 300)
        self.assertEqual([j["variant"] for j in jobs[:5]], list(runner.VARIANTS))
        self.assertTrue(all(j["config"] == "allen_v3_" + j["variant"] for j in jobs))
        self.assertEqual(set(runner.CONFIGS), set(runner.VARIANTS))
        self.assertEqual(set(runner.COMPARISONS), set(runner.VARIANTS) - {"full"})

    def test_resource_and_plan_limits(self):
        for argv in (["--gpus", "0", "1", "2", "3", "4"], ["--gpus", "1", "1"],
                     ["--gpus", "04", "4"],
                     ["--gpus", "-1"], ["--gpus", "0", "--hours", "nan"],
                     ["--gpus", "0", "--eval-workers", "65"],
                     ["--gpus", "0", "--folds", "5"],
                     ["--gpus", "0", "--folds", "0", "1"],
                     ["--gpus", "0", "--sim-steps", "2000"]):
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                runner.resolve_settings(runner.parser().parse_args(argv))
        with self.assertRaises(SystemExit):
            runner.parser().parse_args(["--gpus", "0", "--full", "--smoke"])
        self.assertEqual(self.args("--eval-workers", "64", "--loader-workers", "0").eval_workers, 64)

    def test_snapshot_bytes_and_frozen_mapping_are_preserved(self):
        snap = runner.snapshot_inputs(self.table, self.tree, self.root / "snapshot")
        self.assertEqual(snap["table"].read_bytes(), self.table.read_bytes())
        self.assertEqual(snap["metadata"].read_bytes(), self.table.with_suffix(".columns.json").read_bytes())
        self.assertEqual(json.loads(snap["folds"].read_text())["family_to_fold"], {"F0": 0, "F1": 1})
        with self.assertRaises(FileExistsError):
            runner.snapshot_inputs(self.table, self.tree, self.root / "snapshot")
        self.table.write_text(self.table.read_text().replace("B,F1,1", "B,F1,0"), encoding="utf-8")
        self.meta["table_sha256"] = runner.digest(self.table)
        self.table.with_suffix(".columns.json").write_text(json.dumps(self.meta), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "folds disagree"):
            runner.snapshot_inputs(self.table, self.tree, self.root / "bad")

    def test_separate_mapping_disagreement_is_not_silently_overwritten(self):
        (self.table.parent / "family_folds.json").write_text(json.dumps({"family_to_fold": {"F0": 1}}))
        with self.assertRaisesRegex(ValueError, "family_folds.json disagrees"):
            runner.snapshot_inputs(self.table, self.tree, self.root / "bad")

    def test_saved_plan_uses_only_snapshot_and_resume_detects_mutation(self):
        args, run, manifest = self.new_plan()
        for job in manifest["jobs"]:
            self.assertTrue((self.root / job["table"]).is_relative_to(run))
            self.assertTrue((self.root / job["tree"]).is_relative_to(run))
        self.table.write_text("external original changed after snapshot\n", encoding="utf-8")
        _, loaded = runner.validate_resume(run, self.root)
        self.assertEqual(loaded["jobs"], manifest["jobs"])
        source = self.root / manifest["jobs"][0]["table"]
        source.write_text(source.read_text() + "changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Snapshot input changed"):
            runner.validate_resume(run, self.root)

    def test_code_mutation_rejects_resume(self):
        _, run, manifest = self.new_plan()
        p = self.root / "configs/hjepa/allen_v3_full.yaml"
        p.write_text("changed: true\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Code/config changed"):
            runner.validate_resume(run, self.root)

    def test_manifest_job_cannot_escape_saved_matrix_or_hashed_inputs(self):
        _, run, manifest = self.new_plan()
        manifest["jobs"][0]["name"] = "../../outside"
        runner.save_json(run / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "Saved job/config differs"):
            runner.validate_resume(run, self.root)

    def test_simulation_is_generated_from_observed_snapshot(self):
        args = self.args("--stages", "simulation", "--table", str(self.table), "--tree", str(self.tree))
        calls = []
        def generate(table, tree, output, scenario, seed):
            calls.append((table, tree, output, scenario, seed))
            self.assertNotEqual(table, self.table)
            self.assertIn("inputs/observed", table.as_posix())
            self.assertIn("inputs/observed", tree.as_posix())
            output.mkdir()
            (output / "species_traits.csv").write_bytes(table.read_bytes())
            (output / "species_traits.columns.json").write_bytes(table.with_suffix(".columns.json").read_bytes())
            (output / "truth.json").write_text("{}")
            (output / "true_slopes.csv").write_text("tip,slope\nA,0\n")
            return output / "species_traits.csv"
        from types import SimpleNamespace
        with patch.dict(sys.modules, {"simulate_v2": SimpleNamespace(generate=generate)}):
            run, manifest = runner.create_plan(args, self.root)
        self.assertEqual(len(calls), 1)
        self.assertTrue(all("temperature_s13" in j["table"] for j in manifest["jobs"]))
        self.assertTrue(any(p.endswith("true_slopes.csv") for p in manifest["input_sha256"]))
        runner.validate_resume(run, self.root)

    def test_resume_preserves_plan_and_rejects_changed_matrix_or_inputs(self):
        original = self.args("--full")
        resume = runner.parser().parse_args(["--gpus", "7", "--hours", "4"])
        runner.resolve_settings(resume, vars(original))
        self.assertEqual(resume.mode, "full")
        self.assertEqual(resume.folds, list(range(5)))
        self.assertEqual(resume.gpus, ["7"])
        for argv in (["--smoke"], ["--real-steps", "33"], ["--table", "another.csv"]):
            supplied = runner.parser().parse_args(["--gpus", "7", *argv])
            with self.assertRaises(ValueError):
                runner.resolve_settings(supplied, vars(original))

    def test_command_uses_snapshot_paths_and_fixed_step_budget(self):
        args = self.args("--loader-workers", "0")
        job = {"config": "allen_v3_full", "stage": "simulation", "table": "outputs/a b/table.csv",
               "tree": "outputs/a b/tree.nwk", "fold": 0, "seed": 13, "name": "smoke"}
        command = runner.train_command(job, args, self.root)
        self.assertIn("data.dataset.samples_per_epoch=8192", command)
        self.assertIn("loader.batch_size=128", command)
        self.assertIn("loader.persistent_workers=false", command)
        self.assertIn("data.dataset.tree_file=outputs/a b/tree.nwk", command)
        self.assertFalse(any("gradient" in x or "monotonic" in x for x in command))

    def test_dry_run_does_not_launch_training(self):
        with patch.object(runner, "ROOT", self.root), patch.object(runner, "execute") as execute:
            result = runner.main(["--gpus", "0", "--dry-run", "--stages", "real_development",
                                  "--table", str(self.table), "--tree", str(self.tree)])
        self.assertEqual(result, 0)
        execute.assert_not_called()
        self.assertEqual(len(list((self.root / "outputs").glob("*/manifest.json"))), 1)

    def test_completion_requires_both_stages_and_result_file(self):
        jobs = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
        status = {"a": {"training": "done", "evaluation": "done"}, "b": {"training": "failed"}}
        report = runner.completion_status(jobs, status, self.root)
        self.assertEqual(report["completed"], 0)
        self.assertEqual(report["failed"], ["b"])
        self.assertEqual(report["pending"], ["a", "c"])
        output = self.root / "evaluation/a/pilot_results.json"
        output.parent.mkdir(parents=True)
        output.write_text("{}")
        self.assertEqual(runner.completion_status(jobs, status, self.root)["completed"], 1)

    def test_existing_summarizer_accepts_new_variant_labels_and_partial_plan(self):
        sys.path.insert(0, str(ROOT / "scripts/allen_world"))
        from summarize_v2 import summarize
        args, run, manifest = self.new_plan()
        result = summarize(run, root=self.root)
        self.assertEqual(result["planned"], 5)
        self.assertEqual(result["complete"], 0)
        self.assertEqual(set(result["calibration"]), set(runner.VARIANTS))
        self.assertFalse(any(result["calibration"].values()))

    def completed_results(self, run, manifest):
        import pandas as pd
        state = {}
        for index, job in enumerate(manifest["jobs"]):
            folder = run / "evaluation" / job["name"]
            folder.mkdir(parents=True)
            # A joint vector error deliberately has no signed mean-error field.
            metric = {"family_weighted_rmse": .1 + index / 100,
                      "n_windows": 2, "n_species": 2, "n_families": 2}
            runner.save_json(folder / "pilot_results.json", {
                "metrics": {"complete_three_bones": {"allocation_logratio_joint": {"jepa_full": metric}}}})
            pd.DataFrame({"window": [0, 1], "target_tip": ["A", "B"], "source_tip": ["B", "A"],
                          "family": ["F0", "F1"], "context_0_tip": ["B", "A"]}).to_csv(
                              folder / "prediction_windows.csv.gz", index=False, compression="gzip")
            state[job["name"]] = {"training": "done", "evaluation": "done"}
        runner.save_json(run / "status.json", state)

    def test_joint_allocation_without_signed_mean_is_complete_and_four_comparisons_are_retained(self):
        import pandas as pd
        sys.path.insert(0, str(ROOT / "scripts/allen_world"))
        from summarize_v2 import summarize
        _, run, manifest = self.new_plan()
        self.completed_results(run, manifest)
        report = summarize(run, root=self.root)
        self.assertEqual(report["complete"], 5)
        summary = pd.read_csv(run / "summary.csv")
        self.assertTrue(summary.family_weighted_mean_error.isna().all())
        report = runner.write_comparisons(run, manifest)
        self.assertEqual(report, {"planned_pairs": 4, "completed_pairs": 4, "verified_pairs": 4})
        comparisons = pd.read_csv(run / "comparisons_v3.csv")
        self.assertEqual(set(comparisons.comparison), {"full_vs_" + c for c in runner.COMPARISONS})
        self.assertTrue(comparisons.comparable.all())
        self.assertTrue((comparisons.control_minus_full_rmse > 0).all())

    def test_comparison_refuses_unmatched_windows_and_records_missing_pairs(self):
        import pandas as pd
        _, run, manifest = self.new_plan()
        report = runner.write_comparisons(run, manifest)
        self.assertEqual(report, {"planned_pairs": 4, "completed_pairs": 0, "verified_pairs": 0})
        self.completed_results(run, manifest)
        path = run / "evaluation" / manifest["jobs"][1]["name"] / "prediction_windows.csv.gz"
        frame = pd.read_csv(path)
        frame.loc[0, "context_0_tip"] = "DifferentContext"
        frame.to_csv(path, index=False, compression="gzip")
        runner.write_comparisons(run, manifest)
        comparisons = pd.read_csv(run / "comparisons_v3.csv")
        row = comparisons.loc[comparisons.comparison.eq("full_vs_no_link")].iloc[0]
        self.assertFalse(row.comparable)
        self.assertTrue(pd.isna(row.control_minus_full_rmse))


if __name__ == "__main__":
    unittest.main()
