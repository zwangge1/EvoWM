"""v2 metadata/simulation contracts: folds, data provenance and unforced signs."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/allen_world")]
import prepare_v2
import simulate_v2
from evowm.data.transect import family_folds


class AllenV2InputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source.csv"
        n = 8
        modes = ["P", "SP", "SA", "A"] * 2
        d = pd.DataFrame({"tip": [f"Bird_{i}" for i in range(n)],
                          "family": [f"F{i // 2}" for i in range(n)], "order": ["O"] * n,
                          "mode": modes})
        for c in prepare_v2.TARGETS:
            d[c] = 1 + np.arange(n) / 20
        d.loc[3, "b_tbt"] = np.nan
        for c in prepare_v2.GROUPS["temperature"]:
            d[c] = 10 + np.arange(n)
        for c in prepare_v2.GROUPS["motion"]:
            d[c] = float(c == "e_life_terrestrial")
        for c in prepare_v2.GROUPS["exposure"]:
            d[c] = 0.1
        d["e_fs_ground"] = 0.4
        for c in prepare_v2.GROUPS["development"]:
            d[c] = [float(c == "e_dev_" + m) for m in modes]
        d["m_mass"] = 2 + np.arange(n) / 10
        d["e_habdens"], d["e_migr"] = 0.5, 0.
        self.table = d
        d.to_csv(self.source, index=False)
        meta = {"schema_version": "allen_world_v1", "morph": ["m_tarsus"],
                "skeleton": ["b_fem", "b_tbt", "b_tmt"], "state_default": prepare_v2.TARGETS,
                "table_sha256": prepare_v2.sha(self.source)}
        self.source.with_suffix(".columns.json").write_text(json.dumps(meta), encoding="utf-8")
        self.tree = self.root / "tree.nwk"
        self.tree.write_text("(" + ",".join(t + ":10" for t in d.tip) + ");", encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def prepared(self):
        return prepare_v2.prepare(self.source, self.root / "prepared")

    def test_prepare_preserves_observations_source_and_folds(self):
        original = self.source.read_bytes()
        dest = self.prepared()
        new = pd.read_csv(dest)
        pd.testing.assert_frame_equal(new[self.table.columns], self.table, check_dtype=False)
        self.assertEqual(original, self.source.read_bytes())
        meta = json.loads(dest.with_suffix(".columns.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["action_dim"], 117)
        self.assertEqual(list(meta["eco_groups"]), ["temperature", "exposure", "motion", "development", "background"])
        inputs = sum(meta["eco_groups"].values(), [])
        self.assertEqual(len(inputs), len(set(inputs)))
        eligible = self.table[prepare_v2.REQUIRE_COLS].notna().all(axis=1).to_numpy()
        expected = family_folds(self.table, eligible, 5, 0)
        self.assertEqual(meta["frozen_split"]["family_to_fold"], expected)
        self.assertEqual(new.allen_frozen_family_fold.tolist(), new.family.map(expected).tolist())
        self.assertEqual(meta["table_sha256"], prepare_v2.sha(dest))

    def test_source_hash_mismatch_fails(self):
        d = self.table.copy()
        d.loc[0, "b_fem"] += 1
        d.to_csv(self.source, index=False)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.prepared()

    def test_known_gradient_is_P_only_and_equal_reverse_are_not_forced_positive(self):
        for scenario, sign in (("temperature", 1), ("joint", 1), ("equal", 0), ("reverse", -1)):
            slope, _, p_exact, _ = simulate_v2.conditional_truth(self.table, scenario)
            self.assertTrue(np.all(np.sign(slope[p_exact, 3] - slope[p_exact, 2]) == sign))
            self.assertTrue(np.all(np.sign(slope[p_exact, 2] - slope[p_exact, 1]) == sign))
            np.testing.assert_allclose(slope[~p_exact, 3], slope[~p_exact, 2])
            np.testing.assert_allclose(slope[~p_exact, 2], slope[~p_exact, 1])
        for scenario in ("null", "motion"):
            slope, _, _, _ = simulate_v2.conditional_truth(self.table, scenario)
            self.assertFalse(slope.any())
        with self.assertRaises(ValueError):
            simulate_v2.conditional_truth(self.table, "typo")

    def test_joint_modulation_and_exclusive_motion_feature(self):
        slope, motion, p_exact, factor = simulate_v2.conditional_truth(self.table, "joint")
        np.testing.assert_allclose(factor, 1.1)
        np.testing.assert_allclose(slope[p_exact], np.tile(simulate_v2.POSITIVE * 1.1, (p_exact.sum(), 1)))
        self.assertTrue(motion.any())
        no_exposure = self.table.copy()
        no_exposure[prepare_v2.GROUPS["exposure"]] = 0
        slope2, motion2, _, _ = simulate_v2.conditional_truth(no_exposure, "joint")
        np.testing.assert_array_equal(motion, motion2)
        self.assertTrue(np.all(np.abs(slope2[:, 3]) <= np.abs(slope[:, 3])))

    def test_simulations_preserve_missingness_conditions_fold_and_refresh_hashes(self):
        source = self.prepared()
        real = pd.read_csv(source)
        one = simulate_v2.generate(source, self.tree, self.root / "sim_one", "reverse", 13)
        two = simulate_v2.generate(source, self.tree, self.root / "sim_two", "reverse", 13)
        a, b = pd.read_csv(one), pd.read_csv(two)
        pd.testing.assert_frame_equal(a, b)
        pd.testing.assert_frame_equal(a.drop(columns=prepare_v2.TARGETS), real.drop(columns=prepare_v2.TARGETS))
        np.testing.assert_array_equal(a[prepare_v2.TARGETS].isna(), real[prepare_v2.TARGETS].isna())
        meta = json.loads(one.with_suffix(".columns.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["table_sha256"], prepare_v2.sha(one))
        self.assertNotEqual(meta["table_sha256"], prepare_v2.sha(source))
        truth = pd.read_csv(one.parent / "true_slopes.csv")
        self.assertEqual(truth.tip.tolist(), real.tip.tolist())
        self.assertTrue(truth.filter(like="true_slope_").notna().all().all())
        self.assertEqual(int(truth.loc[3, "observed_mask_b_tbt"]), 0)
        self.assertEqual(meta["simulation"]["true_slopes_sha256"], prepare_v2.sha(one.parent / "true_slopes.csv"))
        with self.assertRaises(FileExistsError):
            simulate_v2.generate(source, self.tree, self.root / "sim_one", "reverse", 13)


if __name__ == "__main__":
    unittest.main()
