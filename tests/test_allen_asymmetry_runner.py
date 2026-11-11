"""Launcher budget, isolation, and artifact integrity contracts."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/allen_world"))
from asymmetry_v1 import (archive_attempt, completed, digest, evaluation_command,
                         parser, train_command, validate_args)
import asymmetry_v1 as runner


class RunnerTests(unittest.TestCase):
    def job(self):
        return dict(name="full_f0_r0_uniform25", kind="neural", input_key="cell",
                    config="allen_v3_full", fold=0, seed=13, steps=4000, overrides=[])

    def test_training_uses_masked_table_not_original_source(self):
        manifest = dict(inputs=dict(tables=dict(cell=dict(table="masked.csv", tree="frozen.nwk"))),
                        settings=dict(loader_workers=0, threads_per_job=4))
        cmd = train_command(self.job(), manifest)
        self.assertIn("data.dataset.species_table=masked.csv", cmd)
        self.assertIn("data.dataset.samples_per_epoch=512000", cmd)
        self.assertIn("trainer.num_sanity_val_steps=0", cmd)
        self.assertIn("loader.persistent_workers=false", cmd)
        self.assertNotIn("--source-run", cmd)

    def test_eval_has_no_target_truth_argument(self):
        with self.assertRaises(ValueError):
            evaluation_command(Path("run"), self.job())
        cmd = evaluation_command(Path("run"), self.job(), Path("weights.ckpt"))
        self.assertIn("neural", cmd)
        self.assertIn("weights.ckpt", cmd)
        self.assertNotIn("--target", cmd)

    def test_resource_bounds_and_worker_alias(self):
        args = parser().parse_args(["--dry-run", "--workers", "64"])
        validate_args(args)
        self.assertEqual(args.eval_workers, 64)
        args.hours = float("nan")
        with self.assertRaises(ValueError):
            validate_args(args)
        args.hours = 1
        args.classic_workers = 0
        with self.assertRaises(ValueError):
            validate_args(args)

    def test_completion_requires_all_evidence_and_unchanged_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            job = self.job()
            folder = run / "evaluation" / job["name"]
            folder.mkdir(parents=True)
            (folder / "predictions.csv.gz").write_bytes(b"predictions")
            (folder / "result.json").write_text(json.dumps(dict(status="complete")))
            hashes = {p.relative_to(run).as_posix(): digest(p) for p in folder.iterdir()}
            state = dict(evaluation="done", job=job, output_sha256=hashes)
            self.assertTrue(completed(run, job, state))
            broken = copy.deepcopy(state)
            broken["output_sha256"].pop(next(iter(hashes)))
            with self.assertRaises(ValueError):
                completed(run, job, broken)
            (folder / "predictions.csv.gz").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                completed(run, job, state)

    def test_retry_preserves_checkpoint_when_only_evaluation_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            job = self.job()
            out = run / "evaluation" / job["name"]
            out.mkdir(parents=True)
            (out / "partial.txt").write_text("partial")
            checkpoint = run / "training" / "ckpts" / job["name"] / "model.ckpt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"trained")
            archived = archive_attempt(run, job, include_training=False)
            self.assertEqual(len(archived), 1)
            self.assertTrue(checkpoint.exists())
            self.assertFalse(out.exists())
            self.assertTrue((run / archived[0] / "partial.txt").exists())

    def test_scheduler_finishes_neural_and_classic_before_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / 'logs').mkdir()
            neural = self.job()
            classic = dict(neural, name='classic', kind='classic', variant='classic')
            manifest = dict(mode='smoke', jobs=[neural, classic],
                            settings=dict(loader_workers=0, threads_per_job=1),
                            inputs=dict(tables=dict(cell=dict(table='masked.csv', tree='tree.nwk'))))
            args = SimpleNamespace(gpus=['0'], hours=1, eval_workers=2, classic_workers=1,
                                   classic_threads=1, job_timeout_minutes=1, retry_failed=False)
            def invoke(command, *unused):
                if command[0] == 'bash':
                    folder = run / 'training/ckpts' / neural['name']
                    folder.mkdir(parents=True)
                    (folder / 'allen_asymmetry_v1_object.ckpt').write_bytes(b'trained')
                else:
                    name = command[command.index('--job') + 1]
                    folder = run / 'evaluation' / name
                    folder.mkdir(parents=True)
                    (folder / 'predictions.csv.gz').write_bytes(b'prediction')
                    (folder / 'result.json').write_text('{"status":"complete"}')
                return 0
            def summarize(*unused, **kwargs):
                status = json.loads((run / 'status.json').read_text())
                self.assertTrue(all(status[j['name']]['evaluation'] == 'done' for j in manifest['jobs']))
            with patch.object(runner, 'verify'), patch.object(runner, 'invoke', side_effect=invoke), \
                 patch.object(runner.subprocess, 'run', return_value=SimpleNamespace(stdout=b'', stderr=b'')), \
                 patch.object(runner, 'pack') as pack, \
                 patch('summarize_asymmetry_v1.summarize', side_effect=summarize) as summary:
                self.assertEqual(runner.execute(run, manifest, args), 0)
                summary.assert_called_once()
                pack.assert_called_once()
            done = json.loads((run / 'completion.json').read_text())
            self.assertTrue(done['all_done'])
            self.assertEqual(done['completed'], 2)

    def test_expired_budget_packages_partial_run_without_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            manifest = dict(mode='smoke', jobs=[self.job()])
            args = SimpleNamespace(gpus=['0'], hours=-1, eval_workers=1, classic_workers=1,
                                   classic_threads=1, job_timeout_minutes=1, retry_failed=False)
            with patch.object(runner, 'verify'), patch.object(runner, 'invoke') as invoke, \
                 patch.object(runner.subprocess, 'run', return_value=SimpleNamespace(stdout=b'', stderr=b'')), \
                 patch.object(runner, 'pack') as pack, patch('summarize_asymmetry_v1.summarize') as summary:
                self.assertEqual(runner.execute(run, manifest, args), 2)
                invoke.assert_not_called()
                summary.assert_not_called()
                pack.assert_called_once()
            self.assertFalse(json.loads((run / 'completion.json').read_text())['all_done'])


if __name__ == "__main__":
    unittest.main()
