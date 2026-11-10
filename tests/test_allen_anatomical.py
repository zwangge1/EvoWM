"""Causal isolation, anatomical reconstruction and ablations of the v3 adapter."""
from pathlib import Path
from types import SimpleNamespace
import copy
import io
import os
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "hjepa/h_jepa"))
from evowm.models.allen_anatomical import (
    AnatomicalReadout, anatomical_coordinates, anatomical_forward,
    configure_readout, create_world_model, predict_from_context,
)


class Harness(nn.Module):
    def __init__(self, model, losses):
        super().__init__()
        self.model = model
        for name, loss in losses.items():
            setattr(self, name, loss)

    def log_dict(self, *args, **kwargs):
        pass


class AnatomicalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(4)
        self.values = torch.tensor([[1.1, 1.2, 1.4, 1.3], [1.3, 1.5, 1.6, 1.2],
                                    [1.5, 1.6, 1.8, 1.7], [1.7, float("nan"), 1.5, 1.8]])

    def config(self, connection="full", objective="jepa", single=False):
        cfg = OmegaConf.load(ROOT / "configs/hjepa/base/evowm.yaml")
        cfg.num_levels = 1 if single else 2
        cfg.level1.wm.embed_dim = 16
        cfg.level1.wm.action_dim = 11
        cfg.level1.encoder.input_dim = 8
        cfg.level1.encoder.hidden_dim = 16
        cfg.level1.encoder.output_dim = 16
        cfg.level1.predictor.update({"depth": 1, "heads": 2, "dim_head": 8,
                                    "hidden_dim": 16, "mlp_dim": 32, "dropout": 0.})
        cfg.level1.loss.embed.sigreg.num_proj = 8
        if not single:
            cfg.level2 = copy.deepcopy(cfg.level1)
            cfg.level2.encoder.input_dim = 16
            cfg.level2.encoder.hidden_dim = 8
            cfg.level2.encoder.output_dim = 8
            cfg.level2.wm.embed_dim = 8
            cfg.level2.predictor.hidden_dim = 8
            cfg.level2.predictor.dim_head = 4
            cfg.level2.action_embed.input_dim = 11
        cfg.evowm_allen_anatomical = {
            "mode": "single" if single else "hierarchical", "connection": connection,
            "objective": objective, "hidden_dim": 8,
        }
        return cfg

    def build(self, **kwargs):
        cfg = self.config(**kwargs)
        model, losses = create_world_model(cfg)
        model.get_level(1).anatomical_readout.configure(self.values)
        return cfg, Harness(model, losses)

    def batch(self):
        return {"pixels_level1": torch.randn(3, 4, 8),
                "action_level1": torch.randn(3, 4, 11),
                "state_level1": self.values.repeat(3, 1, 1)}

    def run_forward(self, harness, batch, cfg):
        return anatomical_forward(harness, batch, "train", cfg, normalize_batch=lambda b: None)

    def test_real_upstream_forward_backward_all_variants(self):
        for kwargs in ({}, {"connection": "none"}, {"connection": "scalar"},
                       {"objective": "supervised"}, {"single": True}):
            with self.subTest(**kwargs):
                cfg, harness = self.build(**kwargs)
                output = self.run_forward(harness, self.batch(), cfg)
                self.assertEqual(output["predicted_log_lengths"].shape, (3, 3, 4))
                self.assertTrue(torch.isfinite(output["loss"]))
                output["loss"].backward()
                grads = [p.grad for p in harness.model.parameters() if p.grad is not None]
                self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads))
                for level in range(1, cfg.num_levels + 1):
                    self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                        for p in harness.model.get_level(level).parameters()))

    def test_last_target_never_enters_prediction_in_train_mode(self):
        cfg, harness = self.build()
        harness.train()
        batch = self.batch()
        first = self.run_forward(harness, batch, cfg)
        changed = {k: v.clone() for k, v in batch.items()}
        changed["pixels_level1"][:, -1] += 100.
        changed["state_level1"][:, -1] = 9.
        changed["action_level1"][:, -1] -= 100.
        second = self.run_forward(harness, changed, cfg)
        torch.testing.assert_close(first["predicted_log_lengths"], second["predicted_log_lengths"])
        torch.testing.assert_close(first["predicted_embedding_2"], second["predicted_embedding_2"])
        self.assertNotEqual(float(first["readout_loss"].detach()), float(second["readout_loss"].detach()))

    def test_earlier_predictions_do_not_see_future_reference_in_train_mode(self):
        _, harness = self.build()
        harness.train()
        context, actions = torch.randn(3, 3, 8), torch.randn(3, 3, 11)
        before = predict_from_context(harness.model, context, actions)
        later = context.clone(); later[:, -1] += 100.
        later_actions = actions.clone(); later_actions[:, -1] -= 100.
        after = predict_from_context(harness.model, later, later_actions)
        torch.testing.assert_close(before["predicted_log_lengths"][:, :2], after["predicted_log_lengths"][:, :2])
        self.assertFalse(any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in harness.model.modules()))
        self.assertGreater(harness.model.allen_anatomical_settings["batchnorm_replacements"], 0)

    def test_no_complete_targets_zeroes_global_coordinate_and_latent_losses(self):
        cfg, harness = self.build()
        batch = self.batch()
        batch["state_level1"][:, :, 1:] = float("nan")
        output = self.run_forward(harness, batch, cfg)
        for key in ("latent_loss_level2", "total_coordinate_loss", "ratio_coordinate_loss"):
            self.assertEqual(float(output[key].detach()), 0., key)
        output["loss"].backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in harness.parameters() if p.grad is not None))
        for module in (harness.model.get_level(2).encoder, harness.model.get_level(2).predictor):
            self.assertTrue(any(p.grad is not None for p in module.parameters()))

    def test_all_missing_targets_give_differentiable_zero(self):
        cfg, harness = self.build()
        batch = self.batch(); batch["state_level1"][:] = float("nan")
        output = self.run_forward(harness, batch, cfg)
        self.assertEqual(float(output["loss"].detach()), 0.)
        output["loss"].backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in harness.parameters() if p.grad is not None))

    def test_one_complete_target_skips_only_global_sigreg(self):
        cfg, harness = self.build()
        batch = self.batch()
        batch["state_level1"][:, :, 1:] = float("nan")
        batch["state_level1"][0, 1, 1:] = self.values[0, 1:]
        # SIGReg needs a population: absence of this module catches accidental
        # regularization of a singleton, while the upper prediction loss stays.
        del harness.sigreg_level2
        output = self.run_forward(harness, batch, cfg)
        self.assertGreater(float(output["latent_loss_level2"].detach()), 0.)
        output["loss"].backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in harness.parameters() if p.grad is not None))

    def test_none_and_scalar_channels_cannot_receive_hidden_global_information(self):
        head = AnatomicalReadout(16, 8, hidden_dim=8)
        head.configure(self.values)
        fine, global_state, total_z = torch.randn(3, 16), torch.randn(3, 8), torch.randn(3)
        for mode in ("none", "scalar"):
            first = head.condition(fine, global_state, total_z, mode)
            second = head.condition(fine, global_state + 99, total_z, mode)
            torch.testing.assert_close(first, second)
        first = head.condition(fine, global_state, total_z, "none")
        second = head.condition(fine, global_state, total_z + 99, "none")
        torch.testing.assert_close(first, second)
        a = head.condition(fine, global_state, total_z, "scalar")
        b = head.condition(fine, global_state, total_z + 99, "scalar")
        self.assertFalse(torch.allclose(a, b))
        a = head.condition(fine, global_state, total_z, "full")
        b = head.condition(fine, global_state + 99, total_z, "full")
        self.assertFalse(torch.allclose(a, b))

    def test_bone_sum_and_log_ratio_reconstruction(self):
        _, harness = self.build()
        output = predict_from_context(harness.model, torch.randn(3, 3, 8), torch.randn(3, 3, 11))
        bones = output["predicted_log_lengths"][..., 1:]
        torch.testing.assert_close(torch.pow(10., bones).sum(dim=-1), torch.pow(10., output["predicted_log_total"]))
        torch.testing.assert_close(bones[..., :2] - bones[..., 2:], output["predicted_log_ratios"])
        torch.testing.assert_close(torch.pow(10., output["predicted_log_proportions"]).sum(dim=-1), torch.ones(3, 3))

    def test_disconnected_total_still_changes_bones_but_not_proportions(self):
        _, harness = self.build(connection="none")
        model = harness.model.eval()
        context, actions = torch.randn(3, 3, 8), torch.randn(3, 3, 11)
        before = predict_from_context(model, context, actions)
        with torch.no_grad():
            model.get_level(2).anatomical_total[-1].bias.add_(1.)
        after = predict_from_context(model, context, actions)
        torch.testing.assert_close(before["predicted_log_proportions"], after["predicted_log_proportions"])
        change = after["predicted_log_total"] - before["predicted_log_total"]
        self.assertTrue((change > 0).all())
        torch.testing.assert_close(after["predicted_log_lengths"][..., 1:] - before["predicted_log_lengths"][..., 1:],
                                   change.unsqueeze(-1).expand(-1, -1, 3))

    def test_complete_mask_never_uses_partial_sum(self):
        total, ratios, mask = anatomical_coordinates(self.values)
        self.assertEqual(mask.tolist(), [True, True, True, False])
        self.assertTrue(torch.isfinite(total).all() and torch.isfinite(ratios).all())
        torch.testing.assert_close(torch.pow(10., total[:3]), torch.pow(10., self.values[:3, 1:]).sum(dim=-1))

    def test_readout_scales_only_use_actual_training_rows(self):
        cfg = self.config()
        model, _ = create_world_model(cfg)
        state = torch.cat([self.values, torch.full((1, 4), 99.)])
        rows = torch.tensor([True, True, True, True, False])
        ds = SimpleNamespace(split="train", state_cols=("m_tarsus", "b_fem", "b_tbt", "b_tmt"),
                             S=state, training_rows=rows, rows_mask=rows)
        configure_readout(model, ds)
        total, ratios, complete = anatomical_coordinates(self.values)
        head = model.get_level(1).anatomical_readout
        torch.testing.assert_close(head.total_mu, total[complete].mean())
        torch.testing.assert_close(head.ratio_mu, ratios[complete].mean(dim=0))
        ds.split = "val"
        with self.assertRaises(ValueError):
            configure_readout(model, ds)

    def test_ablations_same_registered_capacity_and_optimizer_covers_every_parameter(self):
        counts = []
        for mode in ("full", "none", "scalar"):
            _, harness = self.build(connection=mode)
            model = harness.model
            groups = [list(model.get_level(k).parameters()) for k in (1, 2)]
            ids = [id(p) for group in groups for p in group]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(set(ids), {id(p) for p in model.parameters()})
            optimizer = torch.optim.AdamW([{"params": g} for g in groups])
            self.assertEqual(set(ids), {id(p) for g in optimizer.param_groups for p in g["params"]})
            counts.append(sum(p.numel() for p in model.parameters()))
        self.assertEqual(len(set(counts)), 1)

    def test_supervised_has_no_latent_or_sigreg_objectives(self):
        cfg, harness = self.build(objective="supervised")
        out = self.run_forward(harness, self.batch(), cfg)
        self.assertFalse(any(k.startswith("latent_loss") for k in out))
        self.assertFalse(hasattr(harness, "sigreg_level1"))
        torch.testing.assert_close(out["loss"], out["readout_loss"] + .25 * out["total_coordinate_loss"] + .25 * out["ratio_coordinate_loss"])

    def test_saved_model_reuses_exact_inference_path(self):
        _, harness = self.build()
        model = harness.model.eval()
        context, actions = torch.randn(2, 3, 8), torch.randn(2, 3, 11)
        before = predict_from_context(model, context, actions)
        file = io.BytesIO(); torch.save(model, file); file.seek(0)
        loaded = torch.load(file, weights_only=False)
        after = predict_from_context(loaded, context, actions)
        torch.testing.assert_close(before["predicted_log_lengths"], after["predicted_log_lengths"])
        self.assertEqual(loaded.allen_anatomical_settings, model.allen_anatomical_settings)

    def test_reject_wrong_hierarchy_and_unconfigured_prediction(self):
        cfg = self.config(); cfg.num_levels = 1
        with self.assertRaises(ValueError):
            create_world_model(cfg)
        cfg = self.config(); cfg.level2.stride = 2
        with self.assertRaises(ValueError):
            create_world_model(cfg)
        cfg = self.config(); model, _ = create_world_model(cfg)
        with self.assertRaises(RuntimeError):
            predict_from_context(model, torch.randn(2, 3, 8), torch.randn(2, 3, 11))

    def test_actual_configs_residual_optimizer_and_capacity_match(self):
        from hydra import compose, initialize_config_dir
        from evowm.models.residual import enable_residual

        counts = {}
        config_names = ("full", "no_link", "scalar", "supervised", "single")
        with patch.dict(os.environ, {"EVOWM_ROOT": str(ROOT), "HJEPA_HOME": str(ROOT / "outputs")}):
            with initialize_config_dir(config_dir=str(ROOT / "configs/hjepa"), version_base=None):
                for name in config_names:
                    with self.subTest(config=name):
                        cfg = compose(config_name=f"allen_v3_{name}")
                        # Keep the actual architecture/weights; use fewer random
                        # projections only to shorten this CPU smoke test.
                        for level in range(1, int(cfg.num_levels) + 1):
                            OmegaConf.update(cfg, f"level{level}.loss.embed.sigreg.num_proj", 8, force_add=True)
                        model, losses = create_world_model(cfg)
                        enable_residual(model)
                        model.get_level(1).anatomical_readout.configure(self.values)
                        harness = Harness(model, losses)
                        groups = [list(model.get_level(k).parameters()) for k in range(1, int(cfg.num_levels) + 1)]
                        ids = [id(p) for group in groups for p in group]
                        self.assertEqual(len(ids), len(set(ids)))
                        self.assertEqual(set(ids), {id(p) for p in model.parameters()})
                        optimizer = torch.optim.AdamW([{"params": group} for group in groups], lr=3e-4)
                        counts[name] = sum(p.numel() for p in model.parameters())
                        batch = {"pixels_level1": torch.randn(2, 4, 8),
                                 "action_level1": torch.randn(2, 4, 117),
                                 "state_level1": self.values.repeat(2, 1, 1)}
                        output = self.run_forward(harness, batch, cfg)
                        output["loss"].backward()
                        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                        # Every registered module is reached, including the
                        # slots whose input is zeroed in channel ablations.
                        self.assertTrue(all(p.grad is not None for p in model.parameters()))
                        before = model.get_level(1).anatomical_readout.external[-1].weight.detach().clone()
                        optimizer.step()
                        self.assertFalse(torch.equal(before, model.get_level(1).anatomical_readout.external[-1].weight))
        self.assertEqual(len({counts[k] for k in config_names[:-1]}), 1)
        self.assertLess(abs(counts["single"] - counts["full"]) / counts["full"], .001)


if __name__ == "__main__":
    unittest.main()
