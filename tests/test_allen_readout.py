"""Anatomical readouts must not impose a temperature-response ordering."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evowm.models.allen_readout import (AnatomicalReadout, TARGET_COLUMNS, VARIANTS,
                                      configure_readout, masked_standardized_mse)


class ReadoutTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.values = torch.tensor([[1.1, 1.2, 1.4, 1.3], [1.3, 1.5, 1.6, 1.2],
                                    [1.5, 1.6, 1.8, 1.7], [1.7, float("nan"), 1.5, 1.8]])

    def head(self, variant):
        head = AnatomicalReadout(8, variant, hidden_dim=12)
        head.configure(self.values)
        return head

    def test_all_variants_have_identical_parameter_counts_and_shapes(self):
        counts = []
        for variant in VARIANTS:
            head = self.head(variant)
            counts.append(sum(p.numel() for p in head.parameters()))
            output = head(torch.randn(3, 2, 8))
            self.assertEqual(output.shape, (3, 2, 4))
            self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(len(set(counts)), 1)

    def test_allocation_bone_lengths_sum_to_its_predicted_total(self):
        head = self.head("allocation")
        embedding = torch.randn(5, 8)
        output = head(embedding)
        total_z = head._head(1, embedding)
        total = head.total_mu + total_z * head.total_sd
        torch.testing.assert_close(torch.pow(10., output[:, 1:]).sum(dim=1), torch.pow(10., total))
        self.assertTrue((torch.pow(10., output[:, 1:]) > 0).all())

    def test_allocation_allows_either_order_of_bone_response(self):
        # Hold total fixed and move only femur's allocation logit. Its response
        # can exceed or fall below distal response: no enforced distal gradient.
        head = self.head("allocation")
        class FirstCoordinate(nn.Module):
            def __init__(self, sign):
                super().__init__(); self.sign = sign
            def forward(self, x):
                return self.sign * x[..., :1]
        class Zero(nn.Module):
            def forward(self, x):
                return torch.zeros_like(x[..., :1])
        head.heads = nn.ModuleList([Zero(), Zero(), FirstCoordinate(1), Zero()])
        z = torch.zeros(2, 8); z[1, 0] = 1
        diff = head(z)[1] - head(z)[0]
        self.assertGreater(float(diff[1]), float(diff[3]))
        head.heads[2] = FirstCoordinate(-1)
        diff = head(z)[1] - head(z)[0]
        self.assertLess(float(diff[1]), float(diff[3]))

    def test_sequential_heads_receive_predictions_not_measured_targets(self):
        for variant, order in (("proximal", [1, 2, 3]), ("distal", [3, 2, 1])):
            head = self.head(variant)
            captured = {}
            hooks = [layer.register_forward_pre_hook(lambda m, args, i=i: captured.__setitem__(i, args[0].detach().clone()))
                     for i, layer in enumerate(head.heads)]
            z = torch.randn(2, 8)
            pred = head(z)
            standardized = (pred - head.morph_mu) / head.morph_sd
            torch.testing.assert_close(captured[order[0]][:, -2:], torch.zeros(2, 2))
            torch.testing.assert_close(captured[order[1]][:, -2], standardized[:, order[0]])
            torch.testing.assert_close(captured[order[2]][:, -2], standardized[:, order[0]])
            torch.testing.assert_close(captured[order[2]][:, -1], standardized[:, order[1]])
            for hook in hooks:
                hook.remove()

    def test_missing_targets_do_not_poison_gradients_and_traits_are_equal_weight(self):
        pred = torch.tensor([[1., 2., 3., 4.], [1., 2., 3., 4.]], requires_grad=True)
        target = torch.tensor([[0., float("nan"), 0., float("nan")],
                               [float("nan"), float("nan"), 0., float("nan")]])
        loss = masked_standardized_mse(pred, target, torch.ones(4))
        self.assertAlmostEqual(float(loss.detach()), 5.0)  # average trait MSE: (1 + 9)/2
        loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        torch.testing.assert_close(pred.grad[:, 1], torch.zeros(2))
        empty_prediction = torch.randn(2, 4, requires_grad=True)
        empty_loss = masked_standardized_mse(empty_prediction, torch.full((2, 4), float("nan")), torch.ones(4))
        self.assertEqual(float(empty_loss.detach()), 0.)
        empty_loss.backward()
        torch.testing.assert_close(empty_prediction.grad, torch.zeros_like(empty_prediction))

    def test_normalization_uses_only_actual_training_rows_and_is_serialized(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__(); self.jepa = nn.Module(); self.jepa.allen_readout = AnatomicalReadout(8, "allocation")
            def get_level(self, level):
                return self.jepa
        state = torch.cat([self.values, torch.full((1, 4), 9999.)])
        rows = torch.tensor([True, True, True, True, False])
        ds = SimpleNamespace(split="train", state_cols=TARGET_COLUMNS, S=state,
                             training_rows=rows, rows_mask=rows)
        model = Model()
        configure_readout(model, ds)
        expected = self.head("allocation")
        torch.testing.assert_close(model.jepa.allen_readout.morph_mu, expected.morph_mu)
        torch.testing.assert_close(model.jepa.allen_readout.total_mu, expected.total_mu)
        self.assertIn("jepa.allen_readout.configured", model.state_dict())
        optimizer = torch.optim.AdamW(model.get_level(1).parameters())
        actual = {id(p) for group in optimizer.param_groups for p in group["params"]}
        self.assertTrue({id(p) for p in model.jepa.allen_readout.parameters()} <= actual)
        ds.split = "val"
        with self.assertRaises(ValueError):
            configure_readout(model, ds)

    def test_unconfigured_or_invalid_inputs_fail(self):
        head = AnatomicalReadout(8, "parallel")
        with self.assertRaises(RuntimeError):
            head(torch.ones(2, 8))
        with self.assertRaises(ValueError):
            head.configure(torch.full((3, 4), float("nan")))
        with self.assertRaises(ValueError):
            masked_standardized_mse(torch.ones(2, 4), torch.ones(2, 4), torch.zeros(4))

    def test_original_jepa_adapter_backward_and_target_values_do_not_enter_predictions(self):
        from omegaconf import OmegaConf
        sys.path.insert(0, str(ROOT / "hjepa/h_jepa"))
        from evowm.models.allen_readout import create_world_model, allen_forward
        cfg = OmegaConf.load(ROOT / "configs/hjepa/base/evowm.yaml")
        cfg.num_levels = 1
        cfg.level1.wm.embed_dim = 16
        cfg.level1.wm.action_dim = 11
        cfg.level1.encoder.input_dim = 8
        cfg.level1.encoder.hidden_dim = 16
        cfg.level1.encoder.output_dim = 16
        cfg.level1.predictor.update({"depth": 1, "heads": 2, "dim_head": 8,
                                     "hidden_dim": 16, "mlp_dim": 32, "dropout": 0.})
        torch.set_num_threads(2)
        for variant in VARIANTS:
            cfg.evowm_allen_readout = {"variant": variant, "hidden_dim": 8, "weight": 1.0}
            model, losses = create_world_model(cfg)
            model.get_level(1).allen_readout.configure(self.values)
            class Harness(nn.Module):
                def __init__(self):
                    super().__init__(); self.model = model
                    for key, value in losses.items():
                        setattr(self, key, value)
                def log_dict(self, *args, **kwargs):
                    pass
            module = Harness()
            batch = {"pixels_level1": torch.randn(4, 4, 8),
                     "action_level1": torch.randn(4, 4, 11),
                     "state_level1": self.values.repeat(4, 1, 1)}
            before = allen_forward(module, batch, "train", cfg, normalize_batch=lambda b: None)
            changed = {k: v.clone() for k, v in batch.items()}
            changed["state_level1"][:, -1] += 99
            # The final species is a loss target, not an encoder input to its own
            # prediction. Changing its morphological pixels also must not leak.
            changed["pixels_level1"][:, -1] += 99
            after = allen_forward(module, changed, "train", cfg, normalize_batch=lambda b: None)
            torch.testing.assert_close(before["predicted_log_lengths"][:, -1],
                                       after["predicted_log_lengths"][:, -1])
            self.assertNotEqual(float(before["readout_loss"].detach()), float(after["readout_loss"].detach()))
            before["loss"].backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            self.assertTrue(gradients and all(torch.isfinite(g).all() for g in gradients))
            for head in model.get_level(1).allen_readout.heads:
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters()))


if __name__ == "__main__":
    unittest.main()
