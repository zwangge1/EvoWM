"""Input integrity tests: no source pooling, no duplicated joins, missing means missing."""
import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("allen_build_table", ROOT / "scripts/allen_world/build_table.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class AllenTableTests(unittest.TestCase):
    def setUp(self):
        self.traits = pd.DataFrame({"tip": ["Bird_b", "Bird_a", "Bird_c"], "family": ["F2", "F1", "F3"],
                                   "order": ["O", "Passeriformes", "O"], "m_tarsus": [1., 2., np.nan],
                                   "m_mass": [2., 3., 1.], "e_habdens": [0., .5, 1.], "e_migr": [0., 1., .5],
                                   "k_femur": [99., 99., 99.], "src_skeleton": ["legacy"] * 3})
        for col in builder.MOTION:
            self.traits[col] = [0., 0., np.nan]
        self.climate = pd.DataFrame({"tip": ["Bird_a", "Bird_b"]})
        for col in builder.TEMPERATURE:
            self.climate[col] = [10., 20.]
        self.dev = pd.DataFrame({"family": ["F2"], "mode": ["P"]})
        self.bones = pd.DataFrame({"tip": ["Bird_a"], "b_fem": [1.], "b_tbt": [2.], "b_tmt": [3.]})

    def test_join_keeps_missing_rows_and_does_not_pool_sources(self):
        d = builder.assemble_table(self.traits, self.climate, self.dev, self.bones)
        self.assertEqual(d.tip.tolist(), self.traits.tip.tolist())
        self.assertEqual(d.loc[1, "mode"], "A")
        self.assertTrue(d.loc[2, [f"e_dev_{m}" for m in builder.MODES]].isna().all())
        self.assertTrue(pd.isna(d.loc[0, "b_fem"]))
        self.assertEqual(d.loc[0, "k_femur"], 99.)
        self.assertTrue(pd.isna(d.loc[2, "X"]))
        self.assertEqual(d.loc[0, "bio1_c"], 20.)
        meta = builder.table_metadata(d)
        inputs = sum(meta["eco_groups"].values(), [])
        self.assertEqual(len(inputs), len(set(inputs)))
        self.assertFalse(set(builder.TARGETS) & set(inputs))
        self.assertNotIn("X", inputs)
        self.assertNotIn("k_femur", meta["skeleton"])

    def test_duplicate_key_and_invalid_mode_fail(self):
        with self.assertRaises(ValueError):
            builder.assemble_table(self.traits, pd.concat([self.climate, self.climate.iloc[:1]]), self.dev, self.bones)
        dev = self.dev.copy()
        dev.loc[0, "mode"] = "unknown_typo"
        with self.assertRaisesRegex(ValueError, "Unknown developmental"):
            builder.assemble_table(self.traits, self.climate, dev, self.bones)

    def test_eliason_matching_and_observation_scale(self):
        raw = pd.DataFrame({"Taxon": ["Bird a", "Alias a", "Unknown", "Bird b"],
                            "fem_leng_mm": [10., 30., 1000., -1.],
                            "tbt_leng_mm": [20., 40., 1000., np.inf],
                            "tmt_leng_mm": [30., 50., 1000., 50.]})
        b = builder.eliason_means(raw, {"Bird_a", "Bird_b"}, {"Alias_a": "Bird_a"}).set_index("tip")
        self.assertAlmostEqual(b.loc["Bird_a", "b_fem"], np.log10(20.))
        self.assertEqual(b.loc["Bird_a", "n_eliason_fem"], 2)
        self.assertNotIn("Unknown", b.index)
        self.assertTrue(pd.isna(b.loc["Bird_b", "b_fem"]))
        self.assertTrue(pd.isna(b.loc["Bird_b", "b_tbt"]))
        self.assertAlmostEqual(b.loc["Bird_b", "b_tmt"], np.log10(50.))


if __name__ == "__main__":
    unittest.main()
