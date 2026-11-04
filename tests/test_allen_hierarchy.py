"""Run with PYTHONPATH=src python -m unittest discover -s tests -p test_allen_hierarchy.py."""
import copy
import unittest

import numpy as np

from evowm.data.allen_hierarchy import clade_members, phylogenetic_clade_codes
from evowm.data.tree import leaf_order, parse_newick


class AllenHierarchyTests(unittest.TestCase):
    def setUp(self):
        # Total height 30; AB crown 10, CD crown 5, EF crown 25.
        self.tree = parse_newick("(((A:10,B:10):10,(C:5,D:5):15):10,(E:25,F:25):5);")

    def test_maximal_monophyletic_cut_and_exact_boundary(self):
        tree = self.tree
        codes = phylogenetic_clade_codes(tree, 20)
        self.assertEqual(codes[0], codes[3])
        self.assertNotEqual(codes[4], codes[5])
        self.assertEqual(codes[4], tree.tip_nodes[4])
        self.assertEqual(codes[5], tree.tip_nodes[5])
        for code in np.unique(codes):
            descendants = []
            stack = [int(code)]
            while stack:
                node = stack.pop()
                ch = tree.children(node)
                if len(ch):
                    stack.extend(ch.tolist())
                else:
                    descendants.append(node)
            self.assertEqual(set(descendants), set(tree.tip_nodes[codes == code]))

    def test_root_and_singleton_cuts(self):
        np.testing.assert_array_equal(phylogenetic_clade_codes(self.tree, 30), np.zeros(6, dtype=int))
        np.testing.assert_array_equal(phylogenetic_clade_codes(self.tree, 0), self.tree.tip_nodes)

    def test_splits_share_codes_but_never_share_members(self):
        codes = phylogenetic_clade_codes(self.tree)
        train = np.array([True, False, True, False, True, False])
        training = clade_members(codes, train)
        validation = clade_members(codes, ~train)
        for members in training.values():
            self.assertTrue(train[members].all())
        for members in validation.values():
            self.assertTrue((~train)[members].all())
        self.assertFalse(set(np.concatenate(list(training.values()))) & set(np.concatenate(list(validation.values()))))
        self.assertEqual(clade_members(codes, np.zeros(6, dtype=bool)), {})

    def test_rotation_and_extra_trait_attributes_do_not_change_units(self):
        expected = phylogenetic_clade_codes(self.tree)
        self.tree.temperature = np.arange(6)
        self.tree.morphology = np.arange(18).reshape(6, 3)
        self.tree.locomotion = ["water"] * 6
        np.testing.assert_array_equal(phylogenetic_clade_codes(self.tree), expected)
        for seed in range(5):
            order, _ = leaf_order(self.tree, np.random.default_rng(seed))
            rotated = expected[order]
            for code in np.unique(expected):
                positions = np.flatnonzero(rotated == code)
                # A complete monophyletic unit is contiguous under every DFS rotation.
                self.assertEqual(positions[-1] - positions[0] + 1, len(positions))

    def test_invalid_or_missing_age_information_is_rejected(self):
        for newick in ("(A:1,B:2);", "(A,B);", "(A:1,B);", "(A:nan,B:1);", "(A:-1,B:-1);"):
            with self.subTest(newick=newick), self.assertRaises(ValueError):
                phylogenetic_clade_codes(parse_newick(newick))
        for age in (-1, np.nan, np.inf):
            with self.subTest(age=age), self.assertRaises(ValueError):
                phylogenetic_clade_codes(self.tree, age)

    def test_zero_internal_edges_and_rounding_are_allowed(self):
        tree = parse_newick("((A:10,B:10):0,C:10.000001);")
        np.testing.assert_array_equal(phylogenetic_clade_codes(tree, 20), [0, 0, 0])

    def test_corrupt_topology_and_bad_split_masks_fail_closed(self):
        tree = copy.deepcopy(self.tree)
        tree.child_idx[0] = tree.child_idx[1]
        with self.assertRaises(ValueError):
            phylogenetic_clade_codes(tree)
        tree = copy.deepcopy(self.tree)
        tree.tip_nodes[0] = tree.root
        with self.assertRaises(ValueError):
            phylogenetic_clade_codes(tree)
        codes = phylogenetic_clade_codes(self.tree)
        for keep in (np.ones(6, dtype=int), np.ones(5, dtype=bool)):
            with self.assertRaises(ValueError):
                clade_members(codes, keep)


if __name__ == "__main__":
    unittest.main()
