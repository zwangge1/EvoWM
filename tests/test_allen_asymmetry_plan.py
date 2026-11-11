"""Synthetic, CPU-only checks of the prespecified missing-label experiment."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/allen_world"))
import asymmetry_v1_plan as plan


def fixture_table(n=75):
    table = pd.DataFrame(dict(tip=[f"tip_{i:03}" for i in range(n)],
                              family=[f"family_{i // 5:02}" for i in range(n)],
                              allen_frozen_family_fold=(np.arange(n) // 5) % 5))
    for j, name in enumerate(plan.STATE_COLUMNS):
        table[name] = np.log10(np.arange(n) + j + 3.)
    for j, name in enumerate(plan.REQUIRE_COLUMNS[1:]):
        table[name] = np.arange(n, dtype=float) / 7 + j
    table["e_fs_ground"] = .33
    table["e_fs_understory"] = .33
    table["e_fs_canopy"] = .33
    table["k_femur"] = np.arange(n, dtype=float) + 10
    table["k_humerus"] = np.arange(n, dtype=float) + 11
    table["n_femur"] = 2
    table["n_eliason_fem"] = 3
    table["src_skeleton"] = "legacy"
    table["anatomical_log_total"] = np.arange(n, dtype=float) / 4
    table["allocation_logratio_joint"] = np.arange(n, dtype=float) / 9
    table.loc[np.arange(n) % 7 == 0, list(plan.BONE_COLUMNS)] = np.nan
    return table


def fixture_source(root):
    run = root / "outputs/source"
    directory = run / "inputs/observed"
    directory.mkdir(parents=True)
    table_path = directory / "species_traits.csv"
    fixture_table().to_csv(table_path, index=False)
    table = pd.read_csv(table_path)
    metadata = dict(state_default=list(plan.STATE_COLUMNS), morph=["m_tarsus"],
                    skeleton=list(plan.BONE_COLUMNS), eco=list(plan.REQUIRE_COLUMNS[1:]),
                    eco_groups={"temperature": ["bio1_c", "bio5_c", "bio6_c"],
                                "background": ["m_mass", "e_dev_A"]},
                    frozen_split=dict(level="family", folds=5, seed=0,
                                      family_to_fold={f"family_{i:02}": i % 5 for i in range(15)}))
    metadata_path = table_path.with_suffix(".columns.json")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    tree_path = directory / "tree.nwk"
    tree_path.write_text("(" + ",".join(f"{tip}:1" for tip in table.tip) + ");", encoding="utf-8")
    audit = dict(table_sha256=hashlib.sha256(table_path.read_bytes()).hexdigest(),
                 tree_sha256=hashlib.sha256(tree_path.read_bytes()).hexdigest(),
                 table_metadata_sha256=hashlib.sha256(metadata_path.read_bytes()).hexdigest())
    config = dict(data=dict(dataset=dict(species_table=table_path.relative_to(root).as_posix(),
                                        tree_file=tree_path.relative_to(root).as_posix())))
    results = {"real_development_observed_full_s13_f0": dict(provenance=dict(resolved_config=config))}
    canonical = {}
    for fold in range(5):
        val = table[plan.FOLD_COLUMN].eq(fold).to_numpy()
        rows = np.flatnonzero(val)
        references = np.array([rows[i:i + 3] for i in range(len(rows) - 3)])
        canonical[fold] = dict(train_mask=~val, val_mask=val, reference_rows=references,
                               target_rows=rows[3:], distances=np.full(references.shape, 2, np.float32),
                               order_ids=np.zeros(len(references), np.int64))
    return SimpleNamespace(path=run, table=table, metadata=metadata, audit=audit,
                           states=table[list(plan.STATE_COLUMNS)].to_numpy(np.float32).astype(float),
                           tree_row_indices=np.arange(len(table)), tree=SimpleNamespace(n_tips=len(table)),
                           results=results, canonical_windows=canonical)


class PlanTests(unittest.TestCase):
    def test_full_and_smoke_are_complete_prespecified_matrices(self):
        full = plan.make_plan("full")
        self.assertEqual(full["counts"], dict(inputs=50, neural=150, classic=50, total=200))
        self.assertEqual(len({job["name"] for job in full["jobs"]}), 200)
        for job in full["jobs"]:
            self.assertEqual(job["name"], f"{job['variant']}_f{job['fold']}_r{job['replicate']}_{job['condition']}")
            self.assertEqual(job["mask_seed"], (101, 202)[job["replicate"]])
            self.assertEqual(job["seed"], (13, 14)[job["replicate"]])
            self.assertEqual(job["steps"], 4000 if job["kind"] == "neural" else 0)
            self.assertEqual(job["overrides"], [])
        smoke = plan.make_plan("smoke")
        self.assertEqual(smoke["counts"], dict(inputs=3, neural=9, classic=3, total=12))
        self.assertEqual({job["condition"] for job in smoke["jobs"]}, {"complete100", "uniform25", "family25"})
        self.assertEqual({job["fold"] for job in smoke["jobs"]}, {0})
        self.assertEqual({job["steps"] for job in smoke["jobs"] if job["kind"] == "neural"}, {64})
        full["jobs"][0]["overrides"].append("changed")
        self.assertEqual(plan.make_plan()["jobs"][0]["overrides"], [])
        with self.assertRaises(ValueError):
            plan.make_plan("quick")

    def test_exact_budgets_nested_sets_and_single_family_boundary(self):
        table, train = fixture_table(), np.arange(5, 75)
        selected = {condition: plan.choose_train_bone_rows(table, train, condition, 101, 0)
                    for condition in plan.CONDITIONS}
        n = selected["complete100"]["audit"]["original_train_bone_species"]
        for mechanism in ("uniform", "family"):
            for fraction in (25, 50):
                result = selected[f"{mechanism}{fraction}"]
                self.assertEqual(len(result["kept"]), n * fraction // 100)
                self.assertFalse(set(result["kept"]) & set(result["hidden"]))
                self.assertEqual(len(result["kept"]) + len(result["hidden"]), n)
            self.assertTrue(set(selected[mechanism + "25"]["kept"]) <= set(selected[mechanism + "50"]["kept"]))
        self.assertLessEqual(len(selected["family50"]["audit"]["partial_families"]), 1)
        self.assertLessEqual(len(selected["family25"]["audit"]["partial_families"]), 1)
        self.assertEqual(len(selected["complete100"]["hidden"]), 0)

    def test_selection_ignores_observed_values_and_row_order(self):
        table, train = fixture_table(), np.arange(5, 75)
        before = table.copy(deep=True)
        first = plan.choose_train_bone_rows(table, train, "family25", 202, 2)
        changed = table.copy(deep=True)
        changed[list(plan.BONE_COLUMNS)] *= -999
        changed["bio1_c"] += 1000
        again = plan.choose_train_bone_rows(changed, train, "family25", 202, 2)
        np.testing.assert_array_equal(first["kept"], again["kept"])
        self.assertEqual(first["audit"], again["audit"])
        shuffled = table.sample(frac=1, random_state=11).reset_index(drop=True)
        shuffled_train = np.flatnonzero(shuffled.tip.isin(table.tip.iloc[train]).to_numpy())
        reordered = plan.choose_train_bone_rows(shuffled, shuffled_train, "family25", 202, 2)
        self.assertEqual(set(table.tip.iloc[first["kept"]]), set(shuffled.tip.iloc[reordered["kept"]]))
        pd.testing.assert_frame_equal(before, table)
        changed.loc[6, "b_fem"] = np.nan
        with self.assertRaisesRegex(ValueError, "whole-triplet"):
            plan.choose_train_bone_rows(changed, train, "uniform25", 202, 2)

    def test_bone_scan_includes_legacy_counts_and_derived_fields(self):
        table = fixture_table()
        actual = plan.bone_related_columns(table, {})
        self.assertEqual(set(actual), set(plan.BONE_COLUMNS) | {"k_femur", "k_humerus", "n_femur",
                         "n_eliason_fem", "src_skeleton", "anatomical_log_total", "allocation_logratio_joint"})
        self.assertNotIn("m_tarsus", actual)
        with self.assertRaisesRegex(ValueError, "protected"):
            plan.bone_related_columns(table, {"eco": ["k_femur"]})


class ReferenceTests(unittest.TestCase):
    def test_reference_masks_are_species_stable_and_do_not_mutate_truth(self):
        states = np.arange(40, dtype=float).reshape(10, 4) + 1
        states[1, 1:] = np.nan
        original = states.copy()
        rows = np.array([[0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 8], [8, 9, 0]])
        tips = np.array([f"t{i}" for i in range(10)])
        observed = plan.make_reference_states(states, rows, tips, "observed", 101, 0)
        np.testing.assert_array_equal(observed, states[rows])
        half = plan.make_reference_states(states, rows, tips, "hide_half_bones", 101, 0)
        hidden_species = set()
        for row in range(10):
            locations = rows == row
            if np.isfinite(states[row, 1:]).all() and np.isnan(half[:, :, 1:][locations]).all():
                hidden_species.add(row)
            self.assertEqual(len(np.unique(np.isfinite(half[:, :, 1:][locations]), axis=0)), 1)
        self.assertEqual(len(hidden_species), 4)  # floor(9 originally observed / 2)
        reversed_result = plan.make_reference_states(states, rows[::-1], tips, "hide_half_bones", 101, 0)
        np.testing.assert_array_equal(half, reversed_result[::-1])
        np.testing.assert_array_equal(half[:, :, 0], states[rows, 0])
        all_missing = plan.make_reference_states(states, rows, tips, "hide_all_bones", 101, 0)
        self.assertTrue(np.isnan(all_missing[:, :, 1:]).all())
        distal = plan.make_reference_states(states, rows, tips, "hide_distal", 101, 0)
        np.testing.assert_array_equal(distal[:, :, :3], states[rows, :3])
        self.assertTrue(np.isnan(distal[:, :, 3]).all())
        np.testing.assert_array_equal(states, original)
        with self.assertRaises(ValueError):
            plan.make_reference_states(states, rows, tips, "not_a_case", 101, 0)


class FrozenInputTests(unittest.TestCase):
    def test_input_tables_delete_every_training_role_and_preserve_test_ecology(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = fixture_source(root)
            original = source.table.copy(deep=True)
            result = plan.prepare_inputs(source, root / "outputs/new/inputs", root, "smoke")
            self.assertEqual(len(result["tables"]), 3)
            for key, entry in result["tables"].items():
                table = pd.read_csv(root / entry["table"])
                metadata = json.loads((root / entry["metadata"]).read_text(encoding="utf-8"))
                with np.load(root / entry["npz"], allow_pickle=False) as data:
                    self.assertEqual(set(data.files), {"train_rows", "validation_rows", "reference_rows", "target_rows",
                         "distances", "tip_ids_of_rows", "order_ids", "kept_train_bone_rows", "hidden_train_bone_rows"})
                    hidden, kept, val = data["hidden_train_bone_rows"], data["kept_train_bone_rows"], data["validation_rows"]
                    self.assertTrue(table.loc[hidden, result["masked_columns"]].isna().all().all())
                    self.assertTrue(table.loc[kept, list(plan.BONE_COLUMNS)].notna().all().all())
                    pd.testing.assert_frame_equal(table.loc[val], original.loc[val], check_dtype=False, check_exact=True)
                    protected = [name for name in original if name not in result["masked_columns"]]
                    pd.testing.assert_frame_equal(table[protected], original[protected], check_exact=True)
                    np.testing.assert_array_equal(data["reference_rows"], source.canonical_windows[0]["reference_rows"])
                    np.testing.assert_array_equal(data["target_rows"], source.canonical_windows[0]["target_rows"])
                    self.assertFalse(np.isin(data["reference_rows"], hidden).any())
                    self.assertEqual(len(kept), entry["n_kept_train_bones"])
                    self.assertEqual(len(hidden), entry["n_hidden_train_bones"])
                self.assertEqual(metadata["frozen_split"], source.metadata["frozen_split"])
                self.assertEqual(metadata["state_default"], source.metadata["state_default"])
                self.assertEqual(list(metadata["eco_groups"]), list(source.metadata["eco_groups"]))
                self.assertEqual([c for group in metadata["eco_groups"].values() for c in group],
                                 [c for group in source.metadata["eco_groups"].values() for c in group])
                for path_field, hash_field in (("table", "table_sha256"), ("metadata", "table_metadata_sha256"),
                        ("tree", "tree_sha256"), ("npz", "npz_sha256"), ("mask_table", "mask_table_sha256"), ("audit", "audit_sha256")):
                    self.assertFalse(Path(entry[path_field]).is_absolute())
                    self.assertNotIn("\\", entry[path_field])
                    self.assertEqual(hashlib.sha256((root / entry[path_field]).read_bytes()).hexdigest(), entry["hashes"][hash_field])
            self.assertEqual(result["tables"]["f0_r0_uniform25"]["n_kept_train_bones"],
                             result["tables"]["f0_r0_family25"]["n_kept_train_bones"])
            self.assertEqual(result["tables"]["f0_r0_complete100"]["hashes"]["table_sha256"], source.audit["table_sha256"])
            pd.testing.assert_frame_equal(source.table, original)

    def test_frozen_artifacts_are_repeatable_and_never_overwrite_different_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = fixture_source(root)
            destination = root / "outputs/new/inputs"
            first = plan.prepare_inputs(source, destination, root, "smoke")
            self.assertEqual(first, plan.prepare_inputs(source, destination, root, "smoke"))
            table_path = root / first["tables"]["f0_r0_uniform25"]["table"]
            table_path.write_bytes(b"altered")
            with self.assertRaisesRegex(ValueError, "Refusing to replace"):
                plan.prepare_inputs(source, destination, root, "smoke")
            self.assertEqual(table_path.read_bytes(), b"altered")

    def test_source_provenance_and_frozen_split_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = fixture_source(root)
            with self.assertRaisesRegex(ValueError, "original source"):
                plan.prepare_inputs(source, source.path / "new_inputs", root, "smoke")
            source.audit["tree_sha256"] = "wrong"
            with self.assertRaisesRegex(ValueError, "hash differs"):
                plan.prepare_inputs(source, root / "outputs/new/inputs", root, "smoke")
            source.audit["tree_sha256"] = hashlib.sha256((source.path / "inputs/observed/tree.nwk").read_bytes()).hexdigest()
            source.canonical_windows[0]["train_mask"][0] = True
            with self.assertRaisesRegex(ValueError, "frozen folds"):
                plan.prepare_inputs(source, root / "outputs/new/inputs", root, "smoke")


if __name__ == "__main__":
    unittest.main()
