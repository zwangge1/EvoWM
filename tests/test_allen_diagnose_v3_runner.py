"""Read-only source validation, resource bounds and diagnostic packaging."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('diagnose_v3_runner', ROOT / 'scripts/allen_world/diagnose_v3.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class DiagnosticRunnerTests(unittest.TestCase):
    def jobs(self):
        return [{'name': f'real_{v}_{s}_{f}', 'stage':'real_development', 'variant':v,'seed':s,'fold':f}
                for v in runner.VARIANTS for s in (13,14) for f in range(5)]

    def test_full_and_smoke_retain_matched_explicit_jobs(self):
        manifest = {'jobs':self.jobs()}
        self.assertEqual(len(runner.select_jobs(manifest,'full')),50)
        small=runner.select_jobs(manifest,'smoke')
        self.assertEqual({j['variant'] for j in small},{'full','single'})
        self.assertTrue(all(j['seed']==13 and j['fold']==0 for j in small))
        manifest['jobs'][-1]=manifest['jobs'][0]
        with self.assertRaises(ValueError): runner.select_jobs(manifest,'full')

    def test_cpu_thread_bounds_and_no_gpu_requirement(self):
        args=runner.parser().parse_args(['--source-run','outputs/source'])
        self.assertEqual(runner.validate_args(args).workers,32)
        args.workers=64; args.threads=2
        runner.validate_args(args)
        args.threads=3
        with self.assertRaises(ValueError): runner.validate_args(args)

    def test_source_snapshot_and_runtime_mutations_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); run=root/'outputs/source'; run.mkdir(parents=True)
            code=root/'src/evowm/example.py'; code.parent.mkdir(parents=True); code.write_bytes(b'good\n')
            snap=run/'source/src/evowm/example.py'; snap.parent.mkdir(parents=True); snap.write_bytes(code.read_bytes())
            data=run/'inputs/observed.csv'; data.parent.mkdir(); data.write_text('original')
            jobs=self.jobs()
            manifest={'kind':'allen_anatomical_development_v3','settings':{'mode':'full'},'jobs':jobs,
                      'input_sha256':{'outputs/source/inputs/observed.csv':runner.sha(data)},
                      'code_sha256':{'src/evowm/example.py':runner.sha(snap)}}
            runner.save_json(run/'manifest.json',manifest); runner.save_json(run/'completion.json',{'all_done':True})
            for job in runner.select_jobs(manifest,'smoke'):
                target=run/'evaluation'/job['name']; target.mkdir(parents=True)
                (target/'pilot_results.json').write_text('{}')
                (target/'prediction_windows.csv.gz').write_bytes(b'fixture')
            audit=runner.inspect_source(run,'smoke',require_checkpoints=False,root=root)[3]
            self.assertEqual(len(audit['missing_checkpoints']),2)
            with self.assertRaisesRegex(ValueError,'checkpoint'): runner.inspect_source(run,'smoke',root=root)
            code.write_bytes(b'good\r\n')
            runner.inspect_source(run,'smoke',require_checkpoints=False,root=root)
            code.write_text('changed')
            with self.assertRaisesRegex(ValueError,'Inference source'): runner.inspect_source(run,'smoke',require_checkpoints=False,root=root)
            code.write_bytes(b'good\n'); data.write_text('changed')
            with self.assertRaisesRegex(ValueError,'Source input'): runner.inspect_source(run,'smoke',require_checkpoints=False,root=root)

    def test_package_rejects_weights_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)/'audit'; run.mkdir(); (run/'completion.json').write_text('{}')
            ckpt=run/'model.ckpt'; ckpt.write_bytes(b'weight')
            with self.assertRaises(ValueError): runner.package_run(run)
            ckpt.unlink()
            result=runner.package_run(run)
            self.assertTrue(result.is_file())
            with self.assertRaises(FileExistsError): runner.package_run(run)

    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaises(ValueError): runner.contained(root/'../escape',root)
            with self.assertRaises(ValueError): runner.contained(root,root)


if __name__=='__main__': unittest.main()
