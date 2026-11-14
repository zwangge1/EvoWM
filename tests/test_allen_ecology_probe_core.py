"""Conditional probes: conservation, reference exclusion and physical readouts."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'allen_world'))
from ecology_probe_core import (
    CONDITION_COLUMNS, FORAGING_COLUMNS, OUTCOME_NAMES, STATE_COLUMNS,
    SCENARIO_NAMES, SHARE_INDICES, outcome_coordinates, percent_change_from_log,
    summarize_corner_effects, target_condition_scenarios, target_scenarios_for_pairs,
    window_condition_scenarios,
)


def conditions(n=1):
    x = np.zeros((n, len(CONDITION_COLUMNS)), dtype=float)
    x[:, :3] = [20., 35., 5.]
    for name, value in {'ground': .3, 'canopy': .25, 'midhigh': .2,
                        'understory': .1, 'aerial': .05, 'wataroundsurf': .1}.items():
        x[:, CONDITION_COLUMNS.index('e_fs_' + name)] = value
    x[:, CONDITION_COLUMNS.index('m_mass')] = 1.5
    return x


def idx(name):
    return CONDITION_COLUMNS.index(name)


class ConditionsTest(unittest.TestCase):
    def test_original_sums_and_source_are_preserved(self):
        raw = conditions(2)
        fi = [idx(c) for c in FORAGING_COLUMNS]
        raw[1, fi] *= .99
        snapshot = raw.copy()
        result = target_condition_scenarios(raw, recipient='ground', donor='canopy')
        np.testing.assert_array_equal(raw, snapshot)
        self.assertTrue(result.joint_feasible.all())
        for name, values in result.values.items():
            np.testing.assert_allclose(values[:, fi].sum(axis=1), [1., .99], atol=1e-15)
            self.assertFalse(values.flags.writeable)
            unaltered = [i for i in range(raw.shape[1]) if i not in [*range(3), idx('e_fs_ground'), idx('e_fs_canopy')]]
            np.testing.assert_array_equal(values[:, unaltered], raw[:, unaltered])
        np.testing.assert_allclose(result.values['shifted'][:, idx('e_fs_ground')], raw[:, idx('e_fs_ground')] + .1)
        np.testing.assert_allclose(result.values['shifted'][:, idx('e_fs_canopy')], raw[:, idx('e_fs_canopy')] - .1)
        np.testing.assert_allclose(result.values['cold'][:, :3], raw[:, :3] - 1)
        np.testing.assert_allclose(result.values['warm'][:, :3], raw[:, :3] + 1)
        with self.assertRaises(ValueError):
            result.values['observed'][0, 0] = 0

    def test_invalid_reallocation_is_marked_and_not_clipped(self):
        raw = conditions(5)
        raw[1, idx('e_fs_canopy')] = .05
        raw[1, idx('e_fs_ground')] = .5
        raw[2, idx('e_fs_aerial')] = np.nan
        raw[3, idx('e_fs_aerial')] = -.01
        raw[4, idx('e_fs_midhigh')] = 0
        result = target_condition_scenarios(raw, recipient='ground', donor='canopy')
        np.testing.assert_array_equal(result.ecology_feasible, [True, False, False, False, False])
        self.assertEqual(result.ecology_reason.tolist(), ['ok', 'infeasible_requested_transfer',
                         'missing_foraging_component', 'fraction_outside_unit_interval', 'unsupported_fraction_sum'])
        np.testing.assert_array_equal(result.values['shifted'][1:], raw[1:])
        np.testing.assert_array_equal(result.valid_for('shifted'), result.ecology_feasible)

    def test_temperature_missingness_and_order_flags(self):
        raw = conditions(3)
        raw[1, idx('bio6_c')] = np.nan
        raw[2, idx('bio5_c')] = 20.5
        result = target_condition_scenarios(raw, recipient='ground', donor='canopy',
                                            temperature_axis='bio5_c')
        np.testing.assert_array_equal(result.temperature_feasible, [True, False, False])
        self.assertTrue(np.isnan(result.values['cold'][1, idx('bio6_c')]))
        np.testing.assert_array_equal(result.values['warm'][:, idx('bio1_c')], raw[:, idx('bio1_c')])
        self.assertEqual(result.temperature_reason.tolist(), ['ok', 'missing_temperature_coordinate', 'invalid_temperature_order'])

    def test_only_final_target_in_every_window_changes(self):
        targets = conditions(3)
        raw = np.tile(targets[:, None], (1, 4, 1))
        raw[0, 0, idx('m_mass')] = np.nan
        raw[1, 1, idx('e_fs_ground')] = .8
        before = raw.copy()
        result = window_condition_scenarios(raw, recipient='ground', donor='canopy')
        for values in result.values.values():
            np.testing.assert_array_equal(values[:, :-1], raw[:, :-1])
        np.testing.assert_array_equal(raw, before)
        self.assertEqual(result.metadata['n_reference_rows'], 3)

    def test_multiple_pairs_and_column_permutation(self):
        raw = conditions(2)
        cases = target_scenarios_for_pairs(raw, [('ground', 'canopy'), ('canopy', 'ground')])
        self.assertEqual(len(cases), 2)
        order = np.random.default_rng(7).permutation(len(CONDITION_COLUMNS))
        permuted = target_condition_scenarios(raw[:, order], [CONDITION_COLUMNS[i] for i in order],
                                               recipient='ground', donor='canopy')
        original = cases[('e_fs_ground', 'e_fs_canopy')]
        for name in SCENARIO_NAMES:
            np.testing.assert_array_equal(permuted.values[name][:, np.argsort(order)], original.values[name])
        with self.assertRaises(ValueError):
            target_scenarios_for_pairs(raw, [('ground', 'canopy'), ('e_fs_ground', 'canopy')])

    def test_input_errors_and_zero_transfer(self):
        raw = conditions()
        zero = target_condition_scenarios(raw, recipient='ground', donor='canopy', fraction_shift=0)
        np.testing.assert_array_equal(zero.values['shifted'], raw)
        for kwargs in ({'fraction_shift': -.1}, {'temperature_step': 0},
                       {'temperature_axis': 'invalid'}, {'donor': 'ground'}):
            args = dict(recipient='ground', donor='canopy')
            args.update(kwargs)
            with self.assertRaises(ValueError):
                target_condition_scenarios(raw, **args)
        invalid = raw.copy()
        invalid[0, idx('m_mass')] = np.inf
        with self.assertRaises(ValueError):
            target_condition_scenarios(invalid, recipient='ground', donor='canopy')


class OutcomesTest(unittest.TestCase):
    def test_coordinate_values_units_and_partial_observations(self):
        lengths = np.array([[40., 20., 30., 50.], [40., 20., np.nan, 50.]])
        values = outcome_coordinates(np.log10(lengths))
        self.assertEqual(values.shape, (2, 10))
        expected = [np.log10(40), 2., np.log10(20), np.log10(30), np.log10(50),
                    np.log10(.2), np.log10(.3), np.log10(.5), np.log10(.4), np.log10(.6)]
        np.testing.assert_allclose(values[0], expected, atol=1e-14)
        self.assertTrue(np.isnan(values[1, 1]))
        self.assertTrue(np.isnan(values[1, list(SHARE_INDICES)]).all())
        self.assertTrue(np.isfinite(values[1, 2]))
        np.testing.assert_allclose(percent_change_from_log(np.log10(1.03)), 3., atol=1e-12)
        extreme = outcome_coordinates(np.array([[1000., 1000., 1000., 1000.]]))
        self.assertTrue(np.isfinite(extreme).all())
        self.assertAlmostEqual(extreme[0, 1], 1000 + np.log10(3))

    def test_uniform_scaling_changes_size_but_not_allocation(self):
        log = np.log10(np.array([[40., 20., 30., 50.]]))
        base = outcome_coordinates(log)
        scaled = outcome_coordinates(log + np.log10(1.2))
        delta = scaled - base
        np.testing.assert_allclose(percent_change_from_log(delta[:, :5]), 20., atol=1e-12)
        np.testing.assert_allclose(delta[:, 5:], 0, atol=1e-14)

    def test_share_percentage_points_are_not_relative_percent(self):
        a = np.log10(np.array([[40., 30., 30., 40.]]))
        b = np.log10(np.array([[40., 30.6, 30., 39.4]]))
        predictions = {k: a.copy() for k in SCENARIO_NAMES}
        predictions['shifted'] = b
        result = summarize_corner_effects(predictions)['observed_ecology']
        fem_share = OUTCOME_NAMES.index('log10_proportion_b_fem')
        self.assertAlmostEqual(result['percent_change'][0, fem_share], 2., places=11)
        self.assertAlmostEqual(result['share_pp'][0, fem_share], .6, places=11)
        self.assertTrue(np.isnan(result['share_pp'][0, :5]).all())

    def test_four_corner_interaction_is_ratio_of_response_factors(self):
        lengths = np.array([[40., 20., 30., 50.]])
        predictions = {k: np.log10(lengths.copy()) for k in SCENARIO_NAMES}
        predictions['cold_shifted'] += np.log10(1.04)
        predictions['warm_shifted'] += np.log10(1.01)
        result = summarize_corner_effects(predictions, temperature_step=2.)
        out = result['temperature_by_ecology']
        delta = np.log10(1.01 / 1.04)
        self.assertAlmostEqual(out['delta_log10'][0, 2], delta)
        self.assertAlmostEqual(out['response_factor_change_percent'][0, 2], 100 * (1.01 / 1.04 - 1))
        self.assertAlmostEqual(out['interaction_per10C_log10'][0, 2], delta / 4 * 10)
        self.assertTrue(np.isnan(out['percent_change']).all())
        self.assertTrue(np.isnan(out['share_pp']).all())
        self.assertAlmostEqual(result['cold_ecology']['percent_change'][0, 2], 4.)
        self.assertAlmostEqual(result['warm_ecology']['percent_change'][0, 2], 1.)

    def test_masks_do_not_allow_invalid_rows_to_become_zero_effect(self):
        values = np.log10(np.tile([[40., 20., 30., 50.]], (3, 1)))
        predictions = {k: values.copy() for k in SCENARIO_NAMES}
        result = summarize_corner_effects(predictions, ecology_feasible=np.array([True, False, True]),
                                          temperature_feasible=np.array([True, True, False]))
        self.assertTrue(np.isnan(result['observed_ecology']['delta_log10'][1]).all())
        self.assertTrue(np.isnan(result['temperature_by_ecology']['delta_log10'][1:]).all())
        np.testing.assert_allclose(result['temperature_original']['delta_log10'][:2], 0.)
        self.assertTrue(np.isnan(result['temperature_original']['delta_log10'][2]).all())

    def test_missing_predictions_and_extreme_percent_are_explicit(self):
        values = np.log10(np.array([[40., 20., 30., 50.]]))
        predictions = {k: values.copy() for k in SCENARIO_NAMES}
        predictions['shifted'][0, 2] = np.nan
        result = summarize_corner_effects(predictions)['observed_ecology']
        self.assertTrue(np.isnan(result['delta_log10'][0, 1]))
        self.assertTrue(np.isnan(result['delta_log10'][0, 3]))
        self.assertAlmostEqual(result['delta_log10'][0, 2], 0.)
        with self.assertRaises(ValueError):
            percent_change_from_log([1000.])
        with self.assertRaises(ValueError):
            outcome_coordinates([[0., 0., 0., np.inf]])
        self.assertTrue(np.isnan(percent_change_from_log([np.nan])[0]))


if __name__ == '__main__':
    unittest.main()
