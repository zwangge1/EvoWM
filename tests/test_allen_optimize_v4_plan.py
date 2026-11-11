"""Frozen recipe matrix, whole-family inner isolation and original data API."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "allen_world"))
sys.path.insert(0, str(ROOT / "src"))
import optimize_v4_plan as plan
from evowm.data.allen import AllenDataset
from evowm.data.transect import family_folds, validated_frozen_folds


class PlanTests(unittest.TestCase):
    def test_full_and_smoke_matrices_are_exact_and_unique(self):
        full, smoke = plan.make_plan("full"), plan.make_plan("smoke")
        self.assertEqual(full["counts"], {"inner": 140, "outer": 30, "total": 170})
        self.assertEqual(smoke["counts"], {"inner": 7, "outer": 3, "total": 10})
        expected = {(outer, inner, seed, c["name"]) for outer in range(5)
                    for inner in (0, 1) for seed in (13, 14) for c in plan.CANDIDATES}
        actual = {(j["outer_fold"], j["inner_fold"], j["seed"], j["candidate"]) for j in full["inner_jobs"]}
        self.assertEqual(actual, expected)
        self.assertEqual(len({j["name"] for j in full["inner_jobs"]}), 140)
        self.assertEqual(full["inner_jobs"], full["tune_jobs"])
        self.assertEqual(json.loads(json.dumps(plan.CANDIDATES)), plan.CANDIDATES)
        self.assertEqual(json.loads(json.dumps(full)), full)
        for job in smoke["inner_jobs"]:
            self.assertEqual((job["outer_fold"], job["fold"], job["seed"], job["steps"]), (0, 0, 13, 64))
            self.assertEqual(job["config"], "allen_v3_full")
            self.assertEqual(job["input_key"], "inner_0")
        full["candidates"][0]["mask_probs"][0] = -123.
        self.assertEqual(plan.CANDIDATES[0]["mask_probs"][0], .125)

    def test_recipes_preserve_jepa_and_sigreg_and_encode_exact_changes(self):
        names = [c["name"] for c in plan.CANDIDATES]
        self.assertEqual(names, ["reference", "gentle_mask", "light_latent", "size_balance", "bone_balance", "long", "size_long"])
        c = {item["name"]: item for item in plan.CANDIDATES}
        for recipe in c.values():
            self.assertAlmostEqual(sum(recipe["mask_probs"]), 1.)
            self.assertEqual(recipe["observation_weight"], 1.)
            self.assertFalse(any("sigreg" in v or "objective" in v or "num_levels" in v for v in recipe["overrides"]))
            self.assertIn("++level1.loss.embed.pred.enabled=true", recipe["overrides"])
            self.assertGreater(recipe["pred_weight_level1"], 0.)
        self.assertEqual(c["reference"]["mask_probs"], [.125] * 8)
        self.assertEqual(c["gentle_mask"]["mask_probs"], [.75] + [.25/7] * 7)
        self.assertEqual(c["gentle_mask"]["pred_weight_level1"], 1.)
        self.assertEqual(c["light_latent"]["pred_weight_level1"], .25)
        self.assertEqual(c["light_latent"]["pred_weight_level2"], .25)
        self.assertEqual(c["size_balance"]["total_weight"], 1.)
        self.assertEqual((c["bone_balance"]["total_weight"], c["bone_balance"]["ratio_weight"]), (0., 0.))
        self.assertEqual([c[k]["steps"] for k in names], [4000] * 5 + [8000] * 2)

    def test_outer_controls_reuse_each_selected_recipe_and_validate_selection(self):
        chosen = {fold: "size_long" if fold % 2 else "light_latent" for fold in range(5)}
        jobs = plan.make_outer_jobs("full", chosen)
        self.assertEqual(len(jobs), 30)
        for outer in range(5):
            group = [j for j in jobs if j["outer_fold"] == outer]
            self.assertEqual({(j["variant"], j["seed"]) for j in group},
                             {(v, s) for v in ("full", "single", "supervised") for s in (13, 14)})
            self.assertTrue(all(j["candidate"] == chosen[outer] and j["fold"] == outer for j in group))
            self.assertTrue(all(j["overrides"] == group[0]["overrides"] for j in group))
            self.assertTrue(all(j["input_key"] == "outer" for j in group))
        self.assertTrue(all(j["steps"] == 64 for j in plan.make_outer_jobs("smoke", {"0": "long"})))
        for selections in ({}, {0: "missing"}, {0: "reference", 1: "reference"}):
            with self.assertRaises(ValueError):
                plan.make_outer_jobs("smoke", selections)
        with self.assertRaises(ValueError):
            plan.make_plan("unregistered")


class PreparedInputsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.groups = {
            "temperature": ["bio1_c", "bio5_c", "bio6_c"],
            "exposure": ["e_fs_ground", "e_fs_aerial"],
            "motion": ["e_life_aerial", "e_life_terrestrial"],
            "development": ["e_dev_A", "e_dev_P"],
            "background": ["m_mass", "e_habdens", "e_migr"],
        }
        self.require = ["m_tarsus", "m_mass", "e_dev_A", "bio1_c", "bio5_c", "bio6_c"]
        n = 100
        rng = np.random.default_rng(83)
        self.table = pd.DataFrame({"tip": [f"Bird_{i}" for i in range(n)],
                                   "family": [f"F{i//5:02d}" for i in range(n)],
                                   "order": [f"Order_{i//25}" for i in range(n)]})
        columns = list(plan.STATE_COLUMNS) + sum(self.groups.values(), [])
        for j, col in enumerate(columns):
            self.table[col] = rng.normal(j + 1., .2, size=n)
        self.table["k_legacy_bone"] = rng.normal(size=n)
        self.table["X"] = rng.normal(size=n)
        self.table["mode"] = "A"
        # Keep an ineligible member in each family to test whole-family erasure.
        self.table.loc[np.arange(0, n, 5), "m_tarsus"] = np.nan
        mapping = {f"F{i:02d}": i % 5 for i in range(20)}
        self.table[plan.FOLD_COLUMN] = self.table.family.map(mapping)
        self.meta = dict(morph=["m_tarsus"], skeleton=list(plan.STATE_COLUMNS[1:]),
                         state_default=list(plan.STATE_COLUMNS), eco_groups=self.groups,
                         eco=sum(self.groups.values(), []), audit_only_columns=["X", plan.FOLD_COLUMN],
                         frozen_split=dict(level="family", folds=5, seed=0, require_cols=self.require,
                                           family_to_fold=mapping, audit_column=plan.FOLD_COLUMN))
        self.source = self.root / "source.csv"
        self.metadata = self.root / "source.columns.json"
        self.tree = self.root / "tree.nwk"
        self.tree.write_text("(" + ",".join(t + ":10" for t in self.table.tip) + ");", encoding="utf-8")
        self.write_source()

    def tearDown(self):
        self.temp.cleanup()

    def write_source(self):
        self.table.to_csv(self.source, index=False, float_format="%.17g")
        self.meta["table_sha256"] = plan.sha256(self.source)
        self.metadata.write_text(json.dumps(self.meta), encoding="utf-8")

    def prepare(self, name="prepared", folds=None):
        return plan.prepare_inputs(self.source, self.metadata, self.tree, self.root / name,
                                   self.root, outer_folds=folds)

    def dataset(self, spec, *, fold=0, val=False):
        return AllenDataset(name="allen_val" if val else "allen_train", evowm_root=str(self.root),
            species_table=spec["table"], tree_file=spec["tree"], n_trees=1,
            level1={"num_steps": 4, "frameskip": 1}, require_cols=self.require,
            holdout={"level": "family", "folds": 5, "fold": fold, "seed": 0, "frac": 0},
            n_orders=2, refresh_every=0, anatomical_targets=True,
            frozen_fold_column=plan.FOLD_COLUMN)

    def test_original_bytes_preserved_and_all_tree_rows_retained(self):
        original_table, original_meta, original_tree = self.source.read_bytes(), self.metadata.read_bytes(), self.tree.read_bytes()
        prepared = self.prepare()
        self.assertEqual(set(prepared["tables"]), {"outer", *[f"inner_{i}" for i in range(5)]})
        self.assertEqual((self.root / prepared["tables"]["outer"]["table"]).read_bytes(), original_table)
        self.assertEqual((self.root / prepared["tables"]["outer"]["metadata"]).read_bytes(), original_meta)
        self.assertEqual((self.root / prepared["tables"]["outer"]["tree"]).read_bytes(), original_tree)
        for outer in range(5):
            spec = prepared["tables"][f"inner_{outer}"]
            inner = pd.read_csv(self.root / spec["table"])
            pd.testing.assert_frame_equal(inner[["tip", "family", "order"]], self.table[["tip", "family", "order"]])
            np.testing.assert_array_equal(inner[plan.OUTER_FOLD_COLUMN], self.table[plan.FOLD_COLUMN])
            excluded = self.table[plan.FOLD_COLUMN].eq(outer)
            erased = prepared["audit"][str(outer)]["erased_columns"]
            self.assertTrue(inner.loc[excluded, erased].isna().all().all())
            self.assertTrue(inner.loc[excluded, plan.FOLD_COLUMN].eq(-1).all())
            self.assertEqual(len(inner), len(self.table))
            self.assertEqual(spec["table_sha256"], plan.sha256(self.root / spec["table"]))
        self.assertEqual(self.source.read_bytes(), original_table)

    def test_inner_family_mapping_is_deterministic_complete_and_dataset_compatible(self):
        prepared = self.prepare(folds=[0])
        spec = prepared["tables"]["inner_0"]
        inner = pd.read_csv(self.root / spec["table"])
        meta = json.loads((self.root / spec["metadata"]).read_text(encoding="utf-8"))
        eligible = inner[self.require].notna().all(axis=1).to_numpy()
        expected = family_folds(inner, eligible, 5, 0)
        self.assertEqual(meta["frozen_split"]["family_to_fold"], expected)
        self.assertTrue(inner.groupby("family")[plan.FOLD_COLUMN].nunique().eq(1).all())
        self.assertEqual(meta["table_sha256"], plan.sha256(self.root / spec["table"]))
        mapping = validated_frozen_folds(inner, eligible,
            {"level": "family", "folds": 5, "fold": 0, "seed": 0}, plan.FOLD_COLUMN, meta)
        self.assertEqual(mapping, expected)
        for fold in (0, 1):
            train, val = self.dataset(spec, fold=fold), self.dataset(spec, fold=fold, val=True)
            self.assertFalse(np.any(train.rows_mask & val.rows_mask))
            np.testing.assert_array_equal(train.rows_mask | val.rows_mask, eligible)
            np.testing.assert_allclose(train.morph_mu, inner.loc[train.rows_mask, list(plan.STATE_COLUMNS)].mean(), rtol=1e-12)
            np.testing.assert_array_equal(train.morph_mu, val.morph_mu)
            self.assertTrue(np.isfinite(train.condition_mu).all())
            excluded = self.table[plan.FOLD_COLUMN].eq(0).to_numpy()
            for rng_seed in range(12):
                for dataset in (train, val):
                    sample = dataset.sample(np.random.default_rng(rng_seed))
                    self.assertFalse(excluded[sample["rows_level1"].numpy()].any())
                    self.assertTrue(np.isfinite(sample["pixels_level1"].numpy()).all())
                    self.assertTrue(np.isfinite(sample["action_level1"].numpy()).all())

    def test_outer_values_cannot_change_inner_table_normalization_or_windows(self):
        first = self.prepare("first", folds=[0])
        before = self.dataset(first["tables"]["inner_0"])
        excluded = self.table[plan.FOLD_COLUMN].eq(0)
        numeric = [c for c in self.table.select_dtypes(include=np.number) if c != plan.FOLD_COLUMN]
        self.table.loc[excluded, numeric] = 1e9
        self.table.loc[excluded, "mode"] = "CHANGED_OUTER_LABEL"
        self.write_source()
        second = self.prepare("second", folds=[0])
        a = self.root / first["tables"]["inner_0"]["table"]
        b = self.root / second["tables"]["inner_0"]["table"]
        self.assertEqual(a.read_bytes(), b.read_bytes())
        after = self.dataset(second["tables"]["inner_0"])
        for name in ("morph_mu", "morph_sd", "condition_mu", "condition_sd", "rows_mask"):
            np.testing.assert_array_equal(getattr(before, name), getattr(after, name))
        sa, sb = before.sample(np.random.default_rng(42)), after.sample(np.random.default_rng(42))
        for name in sa:
            np.testing.assert_allclose(sa[name].numpy(), sb[name].numpy(), equal_nan=True)

    def test_invalid_frozen_inputs_and_overwrite_are_rejected(self):
        self.prepare(folds=[0])
        with self.assertRaises(FileExistsError):
            self.prepare(folds=[0])
        self.meta["table_sha256"] = "changed"
        self.metadata.write_text(json.dumps(self.meta), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.prepare("hash_failure", folds=[0])
        self.write_source()
        self.table.loc[1, plan.FOLD_COLUMN] = 4
        self.write_source()
        with self.assertRaisesRegex(ValueError, "fold column"):
            self.prepare("fold_failure", folds=[0])
        with self.assertRaises(ValueError):
            self.prepare("invalid_folds", folds=[0, 0])
        with self.assertRaisesRegex(ValueError, "strictly inside"):
            plan.prepare_inputs(self.source, self.metadata, self.tree, self.root, self.root)


if __name__ == "__main__":
    unittest.main()
