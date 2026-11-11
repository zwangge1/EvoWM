"""Leakage, exact-window reconstruction, and immutable provenance checks."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/allen_world"))
import phylo_baseline_data as data
from evowm.data.tree import parse_newick


def condition_table(n=160):
    rng = np.random.default_rng(41)
    return pd.DataFrame(rng.normal(size=(n, len(data.CONDITION_COLUMNS))), columns=data.CONDITION_COLUMNS)


class ConditionDesignTests(unittest.TestCase):
    def test_validation_values_and_outcomes_cannot_change_scaling_or_basis(self):
        table = condition_table()
        table.loc[2, "e_habdens"] = np.nan
        for col in data.STATE_COLUMNS:
            table[col] = np.arange(len(table), dtype=float)
        before = table.copy(deep=True)
        train, bone_support = np.arange(120), np.arange(100)
        first, info = data.build_condition_design(table, train, rank_support_rows=bone_support)
        changed = table.copy()
        changed.loc[120:, list(data.CONDITION_COLUMNS)] = 1e6
        changed.loc[:, list(data.STATE_COLUMNS)] = -1e9
        second, again = data.build_condition_design(changed, train, rank_support_rows=bone_support)
        np.testing.assert_array_equal(first[train], second[train])
        self.assertEqual(info, again)
        pd.testing.assert_frame_equal(table, before)
        self.assertTrue(np.isfinite(first).all())
        self.assertEqual(len(info["interaction_pairs"]), 51)
        self.assertEqual(info["n_candidates"], 96)

    def test_morphology_cannot_be_an_input_condition(self):
        with self.assertRaisesRegex(ValueError, "morphology"):
            data.build_condition_design(condition_table(), np.arange(100), ["m_tarsus"])

    def test_bone_support_must_support_the_same_basis(self):
        table = condition_table(80)
        table["e_migr"] = 0.
        table.loc[40:59, "e_migr"] = 1.
        matrix, info = data.build_condition_design(table, np.arange(60), mode="main", rank_support_rows=np.arange(40))
        drop = {d["column"]: d["reason"] for d in info["dropped_columns"]}
        self.assertEqual(drop["e_migr"], "bone_support_linear_dependency")
        self.assertEqual(np.linalg.matrix_rank(matrix[:40]), matrix.shape[1])
        self.assertEqual(np.linalg.matrix_rank(matrix[:60]), matrix.shape[1])
        self.assertEqual(info["condition_support"]["e_migr"]["bone_max"], 0.)
        self.assertEqual(info["condition_support"]["e_migr"]["train_max"], 1.)
        with self.assertRaisesRegex(ValueError, "training rows"):
            data.build_condition_design(table, np.arange(60), rank_support_rows=np.arange(65))

    def test_foraging_rounding_and_separate_fractions_remain_unchanged(self):
        table = condition_table()
        table.loc[0, list(data.FORAGING_COLUMNS)] = [0., 0., .33, .33, .33, 0., 0.]
        before = table.loc[0, list(data.FORAGING_COLUMNS)].copy()
        _, info = data.build_condition_design(table, np.arange(120))
        pd.testing.assert_series_equal(table.loc[0, list(data.FORAGING_COLUMNS)], before)
        self.assertAlmostEqual(before.sum(), .99)
        self.assertNotIn("X", info["condition_columns"])
        self.assertEqual(len([p for p in info["interaction_pairs"] if p[1] in data.FORAGING_COLUMNS]), 21)


class WindowTests(unittest.TestCase):
    def fixture(self):
        table = pd.DataFrame({"tip": [f"t{i}" for i in range(16)], "family": [f"f{i//2}" for i in range(16)]})
        tree = parse_newick("(" + ",".join(f"t{i}:1" for i in range(16)) + ");")
        mask = np.arange(16) % 2 == 0
        rows, dist, orders = data.replay_windows(tree, np.arange(16), mask, 3, 2, 1)
        raw = pd.DataFrame({"window": np.arange(len(rows)), "target_row": rows[:, -1], "source_row": rows[:, -2],
            "target_tip": table.tip.to_numpy()[rows[:, -1]], "source_tip": table.tip.to_numpy()[rows[:, -2]],
            "family": table.family.to_numpy()[rows[:, -1]],
            **{f"context_{j}_tip": table.tip.to_numpy()[rows[:, j]] for j in range(3)}})
        return table, mask, rows, dist, orders, raw

    def test_all_reference_rows_and_tree_distances_reconstruct(self):
        table, mask, rows, dist, orders, raw = self.fixture()
        ref, target = data.reconstruct_saved_rows(table, raw, 3, mask)
        np.testing.assert_array_equal(np.column_stack([ref, target]), rows)
        np.testing.assert_array_equal(dist, np.full((10, 3), 2., np.float32))
        np.testing.assert_array_equal(orders, np.repeat([0, 1], 5))
        wrong = raw.copy()
        wrong.loc[0, "context_0_tip"] = raw.loc[0, "target_tip"]
        with self.assertRaisesRegex(ValueError, "leak"):
            data.reconstruct_saved_rows(table, wrong, 3, mask)
        wrong = raw.copy()
        wrong.loc[0, "context_0_tip"] = "t1"
        with self.assertRaisesRegex(ValueError, "held-out"):
            data.reconstruct_saved_rows(table, wrong, 3, mask)


class SourceRunTests(unittest.TestCase):
    def make_source(self, root):
        run = root / "outputs/source"
        input_dir = run / "inputs/observed"
        input_dir.mkdir(parents=True)
        table = condition_table(100)
        table["tip"] = [f"t{i}" for i in range(100)]
        table["family"] = [f"f{i//2}" for i in range(100)]
        table[data.FOLD_COLUMN] = (np.arange(100) // 2) % 5
        for j, col in enumerate(data.STATE_COLUMNS):
            table[col] = np.log10(np.arange(100) + j + 2.)
        table.loc[0, "b_fem"] = np.nan
        table_path = input_dir / "species_traits.csv"
        table.to_csv(table_path, index=False)
        table = pd.read_csv(table_path)
        meta = {"state_default": list(data.STATE_COLUMNS), "eco_groups": {
            "temperature": list(data.TEMPERATURE_COLUMNS), "exposure": list(data.FORAGING_COLUMNS),
            "motion": list(data.MOTION_COLUMNS), "development": list(data.DEVELOPMENT_COLUMNS),
            "background": ["m_mass", "e_habdens", "e_migr"]},
            "frozen_split": {"level": "family", "folds": 5, "seed": 0,
                "require_cols": list(data.REQUIRE_COLUMNS), "audit_column": data.FOLD_COLUMN,
                "family_to_fold": {f"f{i}": i % 5 for i in range(50)}}}
        meta_path = table_path.with_suffix(".columns.json")
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        tree_path = input_dir / "tree.nwk"
        tree_path.write_text("(" + ",".join(f"t{i}:1" for i in range(100)) + ");")
        tree = parse_newick(tree_path.read_text())
        state = table[list(data.STATE_COLUMNS)].to_numpy(np.float32).astype(float)
        jobs = []
        for fold in range(5):
            val = table[data.FOLD_COLUMN].eq(fold).to_numpy()
            rows, _, _ = data.replay_windows(tree, np.arange(100), val, 3, 2, 1)
            target = rows[:, -1]
            frame = pd.DataFrame({"window": np.arange(len(rows)), "target_row": target, "source_row": rows[:, -2],
                "target_tip": table.tip.to_numpy()[target], "source_tip": table.tip.to_numpy()[rows[:, -2]],
                "family": table.family.to_numpy()[target],
                **{f"context_{i}_tip": table.tip.to_numpy()[rows[:, i]] for i in range(3)},
                "complete_three_bones": np.isfinite(state[target, 1:]).all(axis=1),
                **{"observed_" + c: state[target, j] for j, c in enumerate(data.STATE_COLUMNS)}})
            raw_train = table.loc[~val, list(data.CONDITION_COLUMNS)].to_numpy(float)
            for variant in ("full", "no_link", "scalar", "supervised", "single"):
                for seed in (13, 14):
                    name = f"real_development_observed_{variant}_s{seed}_f{fold}"
                    jobs.append({"name": name, "stage": "real_development", "variant": variant, "seed": seed, "fold": fold})
                    cfg = {"seed": seed, "data": {"dataset": {"species_table": table_path.relative_to(root).as_posix(),
                        "tree_file": tree_path.relative_to(root).as_posix(), "n_trees": 1,
                        "holdout": {"level": "family", "folds": 5, "seed": 0, "frac": 0, "fold": fold},
                        "require_cols": list(data.REQUIRE_COLUMNS), "frozen_fold_column": data.FOLD_COLUMN}}}
                    config = run / "training/ckpts" / name / "config.yaml"
                    config.parent.mkdir(parents=True)
                    config.write_text(json.dumps(cfg))
                    old = {"history": 3, "n_orders": 2, "seed": 0,
                        "train_audit": {"condition_mu": np.nanmean(raw_train, axis=0).tolist(),
                                        "condition_sd": np.nanstd(raw_train, axis=0).tolist()},
                        "provenance": {"resolved_config": cfg, "config_sha256": data.sha(config),
                            "table_sha256": data.sha(table_path), "table_metadata_sha256": data.sha(meta_path),
                            "tree_sha256": data.sha(tree_path)}}
                    dest = run / "evaluation" / name
                    dest.mkdir(parents=True)
                    (dest / "pilot_results.json").write_text(json.dumps(old))
                    frame.to_csv(dest / "prediction_windows.csv.gz", index=False)
        snap = run / "source/scripts/fixture.py"
        snap.parent.mkdir(parents=True)
        snap.write_text("# frozen source\n")
        manifest = {"kind": "allen_anatomical_development_v3", "settings": {"mode": "full"}, "jobs": jobs,
            "input_sha256": {p.relative_to(root).as_posix(): data.sha(p) for p in (table_path, tree_path, meta_path)},
            "code_sha256": {"scripts/fixture.py": data.sha(snap)}}
        (run / "manifest.json").write_text(json.dumps(manifest))
        (run / "completion.json").write_text(json.dumps({"all_done": True, "planned": 50, "completed": 50}))
        return run

    def test_complete_pairing_works_without_checkpoints_and_rejects_changed_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self.make_source(root)
            source = data.load_source_run(root=root, run=run)
            self.assertEqual(source.audit["real_jobs_paired"], 50)
            self.assertEqual(len(source.audit["missing_checkpoints"]), 50)
            fold = data.load_fold(source, 0)
            self.assertEqual(len(fold.saved_results), 10)
            self.assertEqual(fold.source_refs["evaluation_seed"], 0)
            np.testing.assert_array_equal(fold.tip_ids_of_rows, np.arange(100))
            self.assertFalse(np.isin(fold.reference_rows, fold.train_rows).any())
            changed = run / "evaluation/real_development_observed_single_s14_f0/prediction_windows.csv.gz"
            frame = pd.read_csv(changed)
            frame.loc[0, "context_0_tip"] = frame.loc[1, "context_0_tip"]
            frame.to_csv(changed, index=False)
            with self.assertRaises(AssertionError):
                data.load_source_run(root=root, run=run)

    def test_changed_input_is_rejected_before_loading_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self.make_source(root)
            with (run / "inputs/observed/tree.nwk").open("a") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "Source input"):
                data.load_source_run(root=root, run=run)


if __name__ == "__main__":
    unittest.main()
