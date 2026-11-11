"""Execution sequencing and evidence integrity without launching training."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'/'allen_world'), str(ROOT/'src')]
import optimize_v4 as runner
from optimize_v4_plan import make_plan, make_outer_jobs


class OptimizeRunnerTests(unittest.TestCase):
    def fixture(self, folder):
        manifest = dict(mode='smoke', settings=dict(loader_workers=0, threads_per_job=2),
            tune_jobs=make_plan('smoke')['inner_jobs'], outer_jobs=[],
            inputs={'tables': {'outer': {'table':'outer.csv', 'tree':'tree.nwk'}}},
            original_evidence_sha256={})
        args = SimpleNamespace(hours=1, gpus=['0'], eval_workers=2, job_timeout_minutes=2, retry_failed=False)
        return manifest, args

    def test_command_keeps_training_and_final_evaluation_separate(self):
        job = make_outer_jobs('full', {f:'long' for f in range(5)})[0]
        job.update(table='frozen/table.csv', tree='frozen/tree.nwk')
        command = runner.train_command(job, {'settings': {'loader_workers':0}})
        self.assertIn('data.dataset.samples_per_epoch=1024000', command)
        self.assertIn('trainer.limit_val_batches=0', command)
        self.assertIn('trainer.num_sanity_val_steps=0', command)
        self.assertIn('loader.persistent_workers=false', command)
        self.assertIn('output_model_name=allen_v4', command)
        self.assertIn('++level1.loss.embed.pred.weight=0.25', command)

    def test_incomplete_inner_never_selects_or_launches_outer(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp); manifest, args = self.fixture(run)
            with patch.object(runner, 'verify'), patch.object(runner.subprocess, 'run', return_value=SimpleNamespace(stdout=b'', stderr=b'')), \
                 patch.object(runner, 'execute_stage', return_value=False) as execute, \
                 patch.object(runner, 'select_candidates') as select, patch.object(runner, 'pack'):
                self.assertEqual(runner.run_experiment(args, run, manifest), 2)
                select.assert_not_called()
                self.assertEqual(execute.call_count, 1)
                self.assertEqual(manifest['outer_jobs'], [])
                self.assertFalse(json.loads((run/'completion.json').read_text())['all_done'])

    def test_selection_precedes_outer_and_no_comparison_on_partial_outer(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp); manifest, args = self.fixture(run)
            selection = {'selected_by_outer': {'0':'light_latent'}}
            events = []
            def execute(*arguments):
                jobs = arguments[2]; events.append(jobs[0]['stage'])
                return len(events) == 1
            def select(*arguments):
                events.append('selection'); return selection
            with patch.object(runner, 'verify'), patch.object(runner.subprocess, 'run', return_value=SimpleNamespace(stdout=b'', stderr=b'')), \
                 patch.object(runner, 'execute_stage', side_effect=execute), \
                 patch.object(runner, 'select_candidates', side_effect=select), \
                 patch.object(runner, 'ensure_baseline') as classic, patch.object(runner, 'summarize_outer') as summary, \
                 patch.object(runner, 'pack'):
                self.assertEqual(runner.run_experiment(args, run, manifest), 2)
                self.assertEqual(events, ['inner', 'selection', 'outer'])
                self.assertEqual(len(manifest['outer_jobs']), 3)
                self.assertTrue(all(j['candidate']=='light_latent' for j in manifest['outer_jobs']))
                classic.assert_not_called(); summary.assert_not_called()

    def test_completed_evidence_must_match_job_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp); job = {'name':'test', 'candidate':'reference'}
            folder = run/'evaluation'/'test'; folder.mkdir(parents=True)
            files = [folder/'pilot_results.json', folder/'prediction_windows.csv.gz']
            for f in files:
                f.write_bytes(b'frozen')
            state = dict(evaluation='done', job=dict(job), evaluation_sha256={
                p.relative_to(run).as_posix():runner.digest(p) for p in files})
            self.assertTrue(runner.valid_completed(run, job, state))
            with self.assertRaisesRegex(ValueError, 'metadata changed'):
                runner.valid_completed(run, dict(job, candidate='long'), state)
            files[0].write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'evaluation changed'):
                runner.valid_completed(run, job, state)

    def test_interrupted_training_cannot_silently_load_partial_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp); job = {'name':'interrupted'}
            training = run/'training'/'ckpts'/job['name']
            evaluation = run/'evaluation'/job['name']
            training.mkdir(parents=True); evaluation.mkdir(parents=True)
            (training/'allen_v4_weights.ckpt').write_bytes(b'partial weights')
            (training/'lightning_resume').mkdir()
            (training/'lightning_resume'/'last.ckpt').write_bytes(b'partial optimizer')
            (evaluation/'pilot_results.json').write_text('{}')
            moved = runner.archive_incomplete_training(run, job)
            self.assertEqual(len(moved), 2)
            self.assertFalse(training.exists()); self.assertFalse(evaluation.exists())
            preserved = [run/p for p in moved if p.endswith('/training')][0]
            self.assertEqual((preserved/'allen_v4_weights.ckpt').read_bytes(), b'partial weights')
            self.assertTrue((preserved/'lightning_resume'/'last.ckpt').exists())

    def test_incomplete_classic_baseline_is_resumed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); run = root/'outputs'/'optimization'; run.mkdir(parents=True)
            baseline = root/'outputs'/'baseline'; baseline.mkdir()
            (run/'logs').mkdir()
            (baseline/'manifest.json').write_text(json.dumps({'source_run':'outputs/source', 'folds':list(range(5))}))
            def invoke(command, *args):
                self.assertIn('--resume', command)
                self.assertIn('outputs/baseline', command)
                (baseline/'completion.json').write_text('{"all_done":true}')
                return 0
            with patch.object(runner, 'ROOT', root), patch.object(runner, 'invoke', side_effect=invoke) as called, \
                 patch('phylo_baselines_v1.verify_snapshot'), \
                 patch('phylo_baselines_v1.completed_result', return_value={'status':'complete'}):
                runner.ensure_baseline(run, {'source_run':'outputs/source', 'baseline_run':'outputs/baseline'},
                                       SimpleNamespace(job_timeout_minutes=1))
                called.assert_called_once()


if __name__ == '__main__':
    unittest.main()
