"""Synthetic split, masking, and hierarchy tests (no real biology fitted)."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from evowm.data.allen import AllenDataset, MASK_MODES


class AllenDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.morph = ["m_tarsus", "k_femur", "k_tibiotarsus", "k_tarsometatarsus"]
        self.groups = {
            "temperature": ["bio1", "bio5", "bio6"],
            "motion": [f"motion_{i}" for i in range(12)],
            "background": ["m_mass"] + [f"background_{i}" for i in range(6)],
        }
        n = 24
        self.table = pd.DataFrame({"tip": [f"bird{i}" for i in range(n)],
                                   "family": [f"family{i // 2}" for i in range(n)]})
        for j, col in enumerate(self.morph):
            self.table[col] = np.log10(3 + np.arange(n) * (j + 1))
        for j, col in enumerate(sum(self.groups.values(), [])):
            self.table[col] = ((np.arange(n) * (j + 2)) % 31) / 31 + j
        self.table.loc[[1, 5, 13], "bio6"] = np.nan
        self.table.loc[[7, 17], "motion_11"] = np.nan
        self.write_table()
        metadata = {"morph": self.morph, "skeleton": [], "state_default": self.morph,
                    "eco_groups": self.groups}
        (self.root / "traits.columns.json").write_text(json.dumps(metadata), encoding="utf-8")
        # Twelve independent 5-Myr crown clades at a 50-Myr root; each is a family.
        nwk = "(" + ",".join(f"(bird{i}:5,bird{i+1}:5):45" for i in range(0, n, 2)) + ");"
        (self.root / "tree.nwk").write_text(nwk, encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def write_table(self):
        self.table.to_csv(self.root / "traits.csv", index=False)

    def dataset(self, *, val=False, hierarchy=False, **overrides):
        kwargs = dict(name="allen_val" if val else "allen_train", species_table="traits.csv",
                      evowm_root=str(self.root), tree_file="tree.nwk", n_trees=1,
                      level1={"num_steps": 4, "frameskip": 1}, morph_cols=self.morph,
                      state_cols=self.morph, require_cols=("m_tarsus", "m_mass"),
                      holdout={"folds": 2, "fold": 0, "seed": 17}, n_orders=2,
                      refresh_every=0, action_mode="change", seed=12)
        if hierarchy:
            kwargs["units"] = {"levels": ["phylo"], "history": 3, "max_members": 3}
        kwargs.update(overrides)
        return AllenDataset(**kwargs)

    def test_explicit_masks_hide_values_differences_and_missingness(self):
        ds = self.dataset()
        rows, dist = ds.pool[0]
        rows, dist = rows[:4], dist[:3]
        full = ds.build_action(rows, dist, mode="full")
        self.assertEqual(full.shape, (4, 115))
        for mode in MASK_MODES:
            action = ds.build_action(rows, dist, mode=mode)
            np.testing.assert_array_equal(action[:, -2:], full[:, -2:])
            for group, (start, end) in ds.action_layout.items():
                hidden = (group == "temperature" and mode in ("no_temperature", "neither")) or (
                    group == "motion" and mode in ("no_motion", "neither"))
                if hidden:
                    np.testing.assert_array_equal(action[:, start:end], 0)
                else:
                    np.testing.assert_array_equal(action[:, start:end], full[:, start:end])
                    np.testing.assert_array_equal(action[:-1, end - 1], 1)
            np.testing.assert_array_equal(action[-1], 0)
        # Missing observations and intentionally hidden information are distinct.
        raw = ds.condition_raw[rows].copy()
        raw[:, :3] = np.nan
        missing = ds.build_action(rows, dist, raw_conditions=raw)
        t_start, t_end = ds.action_layout["temperature"]
        np.testing.assert_array_equal(missing[:-1, t_start:t_end - 1], 0)
        np.testing.assert_array_equal(missing[:-1, t_end - 1], 1)

    def test_v2_factors_have_separate_masks_and_frozen_folds_are_enforced(self):
        from evowm.data.allen import FACTOR_MASK_MODES, HIDDEN_GROUPS
        from evowm.data.transect import family_folds
        groups = {"temperature": self.groups["temperature"], "exposure": self.groups["motion"][:7],
                  "motion": self.groups["motion"][7:], "development": self.groups["background"][1:5],
                  "background": ["m_mass"] + self.groups["background"][5:]}
        meta_path = self.root / "traits.columns.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["eco_groups"] = groups
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        folds = family_folds(self.table, np.ones(len(self.table), bool), 5, 0)
        self.table["allen_frozen_family_fold"] = self.table.family.map(folds)
        self.write_table()
        ds = self.dataset(holdout={"folds": 5, "seed": 0, "fold": 0})
        self.assertEqual(ds.mask_modes, FACTOR_MASK_MODES)
        self.assertEqual(ds.action_dim, 117)
        rows, distances = ds.pool[0]
        full = ds.build_action(rows[:4], distances[:3])
        for mode in ds.mask_modes:
            masked = ds.build_action(rows[:4], distances[:3], mode=mode)
            for group, (a, b) in ds.action_layout.items():
                np.testing.assert_array_equal(masked[:, a:b], 0 if group in HIDDEN_GROUPS[mode] else full[:, a:b])
        with self.assertRaises(ValueError):
            self.dataset()  # Old two-fold specification cannot silently replace frozen folds.
        index = self.table.index[self.table.allen_frozen_family_fold.eq(0)][0]
        self.table.loc[index, "allen_frozen_family_fold"] = 1
        self.write_table()
        with self.assertRaises(ValueError):
            self.dataset(holdout={"folds": 5, "seed": 0, "fold": 0})

    def test_heldout_values_cannot_change_training_normalization_or_batches(self):
        before = self.dataset(hierarchy=True)
        val = self.dataset(val=True, hierarchy=True)
        self.assertFalse(np.any(before.rows_mask & val.rows_mask))
        self.assertTrue(np.all(before.rows_mask | val.rows_mask))
        for col in self.morph + sum(self.groups.values(), []):
            self.table.loc[val.rows_mask, col] += 1000
        self.write_table()
        after = self.dataset(hierarchy=True)
        for name in ("morph_mu", "morph_sd", "condition_mu", "condition_sd"):
            np.testing.assert_array_equal(getattr(before, name), getattr(after, name))
        for mode in MASK_MODES:
            a = before.sample(np.random.default_rng(8), mode=mode)
            b = after.sample(np.random.default_rng(8), mode=mode)
            for key in a:
                np.testing.assert_array_equal(a[key].numpy(), b[key].numpy())

    def test_every_context_and_target_member_is_in_current_split(self):
        for val in (False, True):
            ds = self.dataset(val=val, hierarchy=True)
            for pool in ds.pool:
                self.assertTrue(ds.rows_mask[pool[0]].all())
            for seed in range(5):
                batch = ds.sample(np.random.default_rng(seed), mode="full")
                self.assertEqual(tuple(batch["pixels_level1"].shape), (4, 8))
                self.assertEqual(tuple(batch["u2_pixels"].shape), (4, 3, 8))
                self.assertEqual(tuple(batch["u2_action"].shape), (4, 115))
                self.assertTrue(ds.rows_mask[batch["rows_level1"].numpy()].all())
                pixels, mask = batch["u2_pixels"].numpy(), batch["u2_mask"].numpy()
                np.testing.assert_array_equal(pixels[~mask], 0)
                for pixel in pixels[mask]:
                    matches = np.flatnonzero(np.isclose(ds.pix, pixel).all(axis=1))
                    self.assertEqual(len(matches), 1)
                    self.assertTrue(ds.rows_mask[matches[0]])

    def test_unit_condition_means_use_exact_sampled_members_and_available_values(self):
        ds = self.dataset(hierarchy=True)
        for mode in MASK_MODES:
            batch = ds.sample(np.random.default_rng(14), mode=mode)
            pixels, mask = batch["u2_pixels"].numpy(), batch["u2_mask"].numpy()
            action = batch["u2_action"].numpy()
            unit_rows = [[int(np.flatnonzero(np.isclose(ds.pix, pixel).all(axis=1))[0])
                          for pixel in group[group_mask]] for group, group_mask in zip(pixels, mask)]
            for gi, (group, (start, end)) in enumerate(ds.action_layout.items()):
                hidden = (group == "temperature" and mode in ("no_temperature", "neither")) or (
                    group == "motion" and mode in ("no_motion", "neither"))
                if hidden:
                    np.testing.assert_array_equal(action[:, start:end], 0)
                    continue
                columns = np.flatnonzero(ds.col_group == gi)
                n = len(columns)
                means = []
                for members in unit_rows:
                    means.append([np.mean(ds.condition_values[members, c][ds.condition_valid[members, c]])
                                  if ds.condition_valid[members, c].any() else 0 for c in columns])
                means = np.asarray(means)
                np.testing.assert_allclose(action[:-1, start:start+n], means[:-1], atol=1e-6)
                np.testing.assert_allclose(action[:-1, start+n:start+2*n], means[1:], atol=1e-6)

    def test_bad_modes_distances_mask_probabilities_and_ecological_units_fail(self):
        ds = self.dataset()
        values, valid = ds.condition_values[:4], ds.condition_valid[:4]
        for mode in ("unknown", ""):
            with self.assertRaises(ValueError):
                ds.action_from_conditions(values, valid, [1, 2, 3], mode)
        for distance in ([1, -1, 2], [1, np.inf, 2], [1, 2]):
            with self.assertRaises(ValueError):
                ds.action_from_conditions(values, valid, distance)
        for probabilities in ([1, 1, 1, 1], [-1, 1, 1, 0], [1, 0, 0], [np.nan, 0, 0, 1]):
            with self.assertRaises(ValueError):
                self.dataset(condition_mask_probs=probabilities)
        with self.assertRaises(ValueError):
            self.dataset(units={"levels": ["locomotor"]})
        with self.assertRaises(ValueError):
            self.dataset(eco_dropout=0.5)


if __name__ == "__main__":
    unittest.main()
