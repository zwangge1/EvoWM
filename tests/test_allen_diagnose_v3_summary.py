"""Scientific weighting, matched support and truthful diagnostic decomposition."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/allen_world'))
import summarize_diagnose_v3 as summary


def job(variant='full', seed=13, fold=0):
    return dict(name=f'real_development_observed_{variant}_s{seed}_f{fold}',
                stage='real_development', scenario='observed', variant=variant,
                seed=seed, fold=fold)


def windows(scale=1.):
    # Many repeated windows for A1 must not outweigh A2 or family B.
    tips = ['A1'] * 20 + ['A2', 'B1']
    d = pd.DataFrame({'window': np.arange(len(tips)), 'target_tip': tips,
                      'source_tip': ['ref2']*len(tips), 'family': ['A']*21+['B'],
                      'context_0_tip': 'ref0', 'context_1_tip': 'ref1',
                      'context_2_tip': 'ref2', 'complete_three_bones': True})
    error = np.array([1.]*20 + [3.,5.]) * scale
    for c in summary.LENGTHS:
        d['observed_'+c] = 1.
        d['baseline_'+c] = 1. + error
        d['zero_rich_'+c] = 1. + error*2
        d['shuffle_rich_00_'+c] = 1. + error*2
        d['shuffle_rich_01_'+c] = 1. + error*4
    return d


def write_job(root, j, frame, methods=('baseline', 'zero_rich')):
    path = root/'jobs'/j['name']
    path.mkdir(parents=True)
    (path/'diagnosis.json').write_text(json.dumps({'job':j, 'prediction_methods':list(methods),
                                               'baseline_max_absolute_difference':0.}))
    frame.to_csv(path/'diagnostic_windows.csv.gz', index=False, compression='gzip')


class DiagnosticSummaryTests(unittest.TestCase):
    def test_repeated_windows_species_family_then_seed_mse(self):
        a, _ = summary.error_tables(windows(), ['baseline'], job())
        b, _ = summary.error_tables(windows(3.), ['baseline'], job(seed=14))
        species = pd.concat([a,b], ignore_index=True)
        summary.validate_species_support(species, {('real_development','observed','full'):{13,14}})
        family, pooled, table, seeds = summary.aggregate_species(species)
        row = table[(table.method=='baseline') & (table.target=='m_tarsus')].iloc[0]
        self.assertAlmostEqual(row.mse, 75.)  # (15 + 9*15)/2
        self.assertAlmostEqual(row.rmse, np.sqrt(75.))
        self.assertEqual(row.n_species, 3)
        self.assertEqual(row.n_families, 2)
        self.assertNotAlmostEqual(row.rmse, (np.sqrt(15)+np.sqrt(135))/2)

    def test_partial_triplets_never_supply_total_or_bone_comparison(self):
        d = windows()
        d.loc[d.target_tip=='A2', 'observed_b_tbt'] = np.nan
        d.loc[d.target_tip=='A2', 'complete_three_bones'] = False
        species, terms = summary.error_tables(d, ['baseline'], job())
        self.assertIn('A2', set(species[species.target=='m_tarsus'].target_tip))
        self.assertNotIn('A2', set(species[species.target!='m_tarsus'].target_tip))
        self.assertNotIn('A2', set(terms.target_tip))
        d.loc[d.target_tip=='A2', 'complete_three_bones'] = True
        with self.assertRaisesRegex(ValueError, 'flag disagrees'):
            summary.error_tables(d, ['baseline'], job())

    def test_signed_cross_term_and_oracle_may_worsen(self):
        d = windows().iloc[:1].copy()
        obs, pred = np.log10([9.,2.,3.,5.]), np.log10([9.,2.,3.,50.])
        for i,c in enumerate(summary.LENGTHS):
            d['observed_'+c] = obs[i]
            d['baseline_'+c] = pred[i]
        species, terms = summary.error_tables(d, ['baseline'], job())
        fem = terms[terms.bone=='b_fem'].iloc[0]
        self.assertLess(fem.cross_term, 0.)
        self.assertAlmostEqual(fem.bone_mse, 0.)
        self.assertAlmostEqual(fem.total_mse+fem.proportion_mse+fem.cross_term, fem.bone_mse)
        errors = species[species.target=='b_fem'].set_index('method').mse
        self.assertEqual(errors['baseline'], 0.)
        self.assertGreater(errors['oracle_total'], 0.)
        self.assertAlmostEqual(errors['oracle_total'], fem.proportion_mse)
        self.assertAlmostEqual(errors['oracle_allocation'], fem.total_mse)

    def test_permutation_repeats_average_errors_not_predictions(self):
        sp, _ = summary.error_tables(windows(), ['baseline','shuffle_rich_00','shuffle_rich_01'], job())
        values = sp[(sp.target=='m_tarsus')&(sp.target_tip=='A1')].set_index('method').mse
        self.assertAlmostEqual(values['shuffle_rich_average'], (4+16)/2)
        self.assertNotAlmostEqual(values['shuffle_rich_average'], ((2+4)/2)**2)

    def test_pairing_rejects_mismatched_families_and_seed_support(self):
        with self.assertRaisesRegex(ValueError, 'identical unique family'):
            summary.paired_delta(pd.Series([1.,2.], index=['A','B']), pd.Series([1.,2.], index=['A','C']), bootstrap=10)
        a, _ = summary.error_tables(windows(), ['baseline'], job())
        b, _ = summary.error_tables(windows(), ['baseline'], job(seed=14))
        b = b[b.target_tip!='B1']
        with self.assertRaisesRegex(ValueError, 'Unmatched species support'):
            summary.validate_species_support(pd.concat([a,b]), {('real_development','observed','full'):{13,14}})
        c = a.copy(); c['fold'] = 1
        with self.assertRaisesRegex(ValueError, 'multiple held-out folds'):
            summary.validate_species_support(pd.concat([a,c]), {('real_development','observed','full'):{13}})

    def test_paired_bootstrap_zero_difference_exact(self):
        a = pd.Series([1.,4.,9.], index=['A','B','C'])
        r = summary.paired_delta(a,a,bootstrap=51)
        self.assertEqual(r['alternative_minus_baseline_rmse'],0.)
        self.assertEqual(r['ci_low'],0.)
        self.assertEqual(r['ci_high'],0.)

    def test_smoke_summary_no_evidence_flag_and_missing_job_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp)
            jobs=[job(),job('single')]
            (run/'manifest.json').write_text(json.dumps({'jobs':jobs,'mode':'smoke'}))
            write_job(run,jobs[0],windows())
            with self.assertRaisesRegex(ValueError,'Missing configured diagnostic output'):
                summary.summarize(run,bootstrap=20)
            write_job(run,jobs[1],windows(),methods=('baseline',))
            result=summary.summarize(run,bootstrap=20)
            self.assertTrue(result['all_configured_jobs_complete'])
            self.assertFalse(result['complete_five_variant_two_seed_five_fold_matrix'])
            for f in result['files']:
                self.assertTrue((run/f).is_file())
            pairs=pd.read_csv(run/'paired_effects.csv')
            self.assertIn('separately_trained_architecture',set(pairs.comparison_type))
            d=pd.read_csv(run/'decomposition.csv')
            np.testing.assert_allclose(d.identity_residual,0.,atol=1e-12)

    def test_observed_or_context_mismatch_fails_even_if_counts_equal(self):
        for column,value in [('observed_b_fem',9.),('context_1_tip','wrong_ref')]:
            with self.subTest(column=column), tempfile.TemporaryDirectory() as temp:
                run=Path(temp)
                jobs=[job(),job('single')]
                (run/'manifest.json').write_text(json.dumps({'jobs':jobs}))
                write_job(run,jobs[0],windows())
                changed=windows(); changed.loc[0,column]=value
                write_job(run,jobs[1],changed,methods=('baseline',))
                with self.assertRaisesRegex(ValueError,'Unmatched window identity'):
                    summary.summarize(run,bootstrap=10)


if __name__=='__main__':
    unittest.main()
