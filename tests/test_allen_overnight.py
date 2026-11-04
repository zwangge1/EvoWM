"""Queue coverage and command safety for the long-running experiment launcher."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("overnight_v2", ROOT / "scripts/allen_world/overnight_v2.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class OvernightTests(unittest.TestCase):
    def test_default_matrix_complete_unique_and_balanced(self):
        jobs = runner.make_jobs(runner.VARIANTS, runner.SCENARIOS, [13, 14], list(range(5)),
                                ["simulation", "real_development"])
        self.assertEqual(len(jobs), 280)
        self.assertEqual(len({j["name"] for j in jobs}), 280)
        self.assertEqual(sum(j["stage"] == "simulation" for j in jobs), 240)
        # Interleave variants: an interrupted night must not train all one variant first.
        self.assertEqual([j["variant"] for j in jobs[:4]], list(runner.VARIANTS))
        self.assertEqual({j["scenario"] for j in jobs if j["stage"] == "real_development"}, {"observed"})

    def test_commands_share_budget_and_never_add_target_gradient_constraints(self):
        args = SimpleNamespace(sim_steps=2000, real_steps=4000, loader_workers=8, tree="tree with spaces.nwk")
        commands = []
        for variant in runner.VARIANTS:
            job = {"stage": "simulation", "table": "outputs/table with spaces.csv", "fold": 2,
                   "seed": 13, "name": variant, "config": "allen_v2_" + variant}
            commands.append(runner.train_command(job, args, ROOT / "outputs"))
        for command in commands:
            self.assertIn("data.dataset.samples_per_epoch=256000", command)
            self.assertIn("data.dataset.species_table=outputs/table with spaces.csv", command)
            self.assertFalse(any("gradient" in arg or "monotonic" in arg for arg in command))


if __name__ == "__main__":
    unittest.main()
