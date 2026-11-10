"""Opt-in anatomical-scale JEPA, using upstream JEPA building blocks.

Both levels refer to the SAME species. They are not temporal strata or inferred
ancestors. Encoders run fine-to-global; predictions run global-to-fine. Inference
accepts only reference morphology and conditions, never target morphology.

All v3 modes use pointwise LayerNorm in place of upstream BatchNorm projectors:
flattening species windows into BatchNorm would break causal prefix isolation.
The replacement is local to newly constructed opt-in models; v2 is unchanged.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from .allen_readout import TARGET_COLUMNS, masked_standardized_mse

API_VERSION = 1


def _settings(cfg):
    raw = cfg.get("evowm_allen_anatomical")
    if not raw:
        raise ValueError("evowm_allen_anatomical settings are required")
    result = dict(raw)
    result.setdefault("mode", "hierarchical")
    result.setdefault("connection", "full")
    result.setdefault("objective", "jepa")
    result.setdefault("hidden_dim", 64)
    result.setdefault("observation_weight", 1.0)
    result.setdefault("total_weight", 0.25)
    result.setdefault("ratio_weight", 0.25)
    if result["mode"] not in ("hierarchical", "single"):
        raise ValueError("mode must be hierarchical or single")
    if result["connection"] not in ("full", "none", "scalar"):
        raise ValueError("connection must be full, none, or scalar")
    if result["objective"] not in ("jepa", "supervised"):
        raise ValueError("objective must be jepa or supervised")
    if int(result["hidden_dim"]) < 1:
        raise ValueError("hidden_dim must be positive")
    for key in ("observation_weight", "total_weight", "ratio_weight"):
        result[key] = float(result[key])
        if not math.isfinite(result[key]) or result[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if sum(result[k] for k in ("observation_weight", "total_weight", "ratio_weight")) <= 0:
        raise ValueError("At least one observation objective must be active")
    result.update(api_version=API_VERSION, normalization="pointwise_layernorm",
                  scale="raw_log10_mm", target_columns=list(TARGET_COLUMNS))
    return result


def _pointwise_normalization(module):
    """Replace only opt-in model's batch-coupled layers, preserving affine init."""
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            replacement = nn.LayerNorm(child.num_features, eps=child.eps,
                                       elementwise_affine=child.affine)
            if child.affine:
                with torch.no_grad():
                    replacement.weight.copy_(child.weight)
                    replacement.bias.copy_(child.bias)
            setattr(module, name, replacement)
            count += 1
        else:
            count += _pointwise_normalization(child)
    return count


def anatomical_coordinates(values):
    """Return log total, two log ratios vs distal bone, and complete-triplet mask.

    Missing triplets receive finite placeholder coordinates; callers MUST mask
    them. No partial sum or imputed bone becomes a total-length observation.
    """
    if values.shape[-1] != 4 or torch.isinf(values).any():
        raise ValueError("Expected four finite-or-NaN log lengths")
    bones = values[..., 1:]
    complete = torch.isfinite(bones).all(dim=-1)
    safe = torch.where(complete.unsqueeze(-1), bones, torch.zeros_like(bones))
    total = torch.logsumexp(safe * math.log(10), dim=-1) / math.log(10)
    ratios = safe[..., :2] - safe[..., 2:]
    return total, ratios, complete


def _mlp(inputs, outputs, hidden):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.GELU(), nn.Linear(hidden, outputs))


class AnatomicalReadout(nn.Module):
    """Fine readouts, top-down adapter and training-only coordinate scales."""

    def __init__(self, fine_dim, global_dim, hidden_dim=64):
        super().__init__()
        self.fine_dim, self.global_dim = int(fine_dim), int(global_dim)
        self.adapter = _mlp(fine_dim + global_dim + 1, fine_dim, hidden_dim)
        self.external = _mlp(fine_dim, 1, hidden_dim)
        self.ratios = _mlp(fine_dim, 2, hidden_dim)
        for name, size in (("morph_mu", 4), ("morph_sd", 4), ("ratio_mu", 2), ("ratio_sd", 2)):
            self.register_buffer(name, torch.ones(size) if name.endswith("sd") else torch.zeros(size))
        self.register_buffer("total_mu", torch.zeros(()))
        self.register_buffer("total_sd", torch.ones(()))
        self.register_buffer("configured", torch.tensor(False))

    @torch.no_grad()
    def configure(self, training_values):
        values = torch.as_tensor(training_values, dtype=torch.float64, device=self.total_mu.device)
        if values.ndim != 2 or values.shape[-1] != 4 or torch.isinf(values).any():
            raise ValueError("Expected N by four training log lengths")
        for j in range(4):
            observed = values[:, j][torch.isfinite(values[:, j])]
            if len(observed) < 2:
                raise ValueError(f"At least two training observations needed for {TARGET_COLUMNS[j]}")
            self.morph_mu[j] = observed.mean()
            sd = observed.std(unbiased=False)
            self.morph_sd[j] = sd if sd > 1e-8 else 1.
        total, ratios, complete = anatomical_coordinates(values)
        if int(complete.sum()) < 2:
            raise ValueError("At least two complete training bone triplets required")
        self.total_mu.copy_(total[complete].mean())
        sd = total[complete].std(unbiased=False)
        self.total_sd.copy_(sd if sd > 1e-8 else sd.new_tensor(1.))
        self.ratio_mu.copy_(ratios[complete].mean(dim=0))
        sd = ratios[complete].std(dim=0, unbiased=False)
        self.ratio_sd.copy_(torch.where(sd > 1e-8, sd, torch.ones_like(sd)))
        self.configured.fill_(True)

    def condition(self, fine, global_prediction, total_z, connection):
        if connection not in ("full", "scalar", "none"):
            raise ValueError("Invalid top-down connection")
        rich = global_prediction if connection == "full" else torch.zeros_like(global_prediction)
        scalar = total_z if connection != "none" else torch.zeros_like(total_z)
        return fine + self.adapter(torch.cat([fine, rich, scalar.unsqueeze(-1)], dim=-1))

    def decode(self, fine, log_total):
        external = self.morph_mu[0] + self.morph_sd[0] * self.external(fine).squeeze(-1)
        log_ratios = self.ratio_mu + self.ratio_sd * self.ratios(fine)
        logits = torch.cat([log_ratios * math.log(10), torch.zeros_like(log_ratios[..., :1])], dim=-1)
        log_p = F.log_softmax(logits, dim=-1) / math.log(10)
        lengths = torch.cat([external.unsqueeze(-1), log_total.unsqueeze(-1) + log_p], dim=-1)
        return {"predicted_log_lengths": lengths, "predicted_log_total": log_total,
                "predicted_log_proportions": log_p, "predicted_log_ratios": log_ratios}


def create_world_model(cfg):
    """Construct original per-level JEPAs, then attach anatomical-only modules."""
    from hjepa_utils import create_world_model as upstream_create
    options = _settings(cfg)
    expected = 1 if options["mode"] == "single" else 2
    if int(cfg.num_levels) != expected:
        raise ValueError(f"{options['mode']} requires num_levels={expected}")
    if cfg.get("evowm_units", False) or cfg.get("evowm_allen_readout", False):
        raise ValueError("Anatomical v3 cannot combine with legacy units/readouts")
    history = int(cfg.level1.wm.history_size)
    for level in range(1, expected + 1):
        setting = cfg[f"level{level}"]
        if (int(setting.wm.get("rollout_n", 1)) != 1 or setting.wm.get("nsteps") is not None
                or int(setting.wm.history_size) != history):
            raise ValueError("Anatomical levels require equal histories and one-step prediction")
        if int(setting.get("stride", 1)) != 1 or int(setting.get("window_size", 1)) != 1:
            raise ValueError("Anatomical levels do not use temporal pooling")
        if setting.get("action_encoder") or setting.wm.get("use_proprio", False):
            raise ValueError("Use direct Allen conditions and vector morphology at each level")
        if float(setting.get("idm_coeff", 0.)) or float(setting.get("action_sigreg_coeff", 0.)):
            raise ValueError("IDM/action regularization is not part of this matched experiment")
    model, losses = upstream_create(cfg)
    options["batchnorm_replacements"] = _pointwise_normalization(model)
    fine_dim = int(model.embed_dims[1])
    global_dim = int(model.embed_dims[expected])
    model.get_level(1).anatomical_readout = AnatomicalReadout(fine_dim, global_dim, int(options["hidden_dim"]))
    model.get_level(expected).anatomical_total = _mlp(global_dim, 1, int(options["hidden_dim"]))
    model.allen_anatomical_settings = options
    model.allen_anatomical_history = history
    model.allen_anatomical_action_dim = int(cfg.level1.wm.action_dim)
    if options["objective"] == "supervised":
        losses = {}
    return model, losses


def configure_readout(model, dataset):
    if dataset.split != "train" or tuple(dataset.state_cols) != TARGET_COLUMNS:
        raise ValueError("Readout configuration requires canonical training dataset")
    rows = torch.as_tensor(dataset.training_rows, dtype=torch.bool)
    actual = torch.as_tensor(dataset.rows_mask, dtype=torch.bool)
    if not torch.equal(rows, actual) or rows.ndim != 1:
        raise ValueError("Normalization must use exactly the training split")
    model.get_level(1).anatomical_readout.configure(torch.as_tensor(dataset.S)[rows])


def _encode(encoder, values):
    shape = values.shape
    return encoder(values.reshape(-1, shape[-1])).reshape(*shape[:-1], -1)


def predict_from_context(model, context_pixels, actions):
    """Shared training/inference path; outputs each causally valid next position."""
    if context_pixels.ndim != 3 or context_pixels.shape[-1] != 8:
        raise ValueError("Context must be B by H by eight morphology values/masks")
    if actions.ndim != 3 or actions.shape[:2] != context_pixels.shape[:2]:
        raise ValueError("Conditions and context must have matching B,H")
    if context_pixels.shape[1] != model.allen_anatomical_history:
        raise ValueError("Context must match saved history length")
    if actions.shape[-1] != model.allen_anatomical_action_dim:
        raise ValueError("Condition width does not match saved model")
    if not torch.isfinite(context_pixels).all() or not torch.isfinite(actions).all():
        raise ValueError("Predictor inputs must be finite; missingness uses masks")
    fine = model.get_level(1)
    head = fine.anatomical_readout
    if not bool(head.configured):
        raise RuntimeError("Configure readout from training observations first")
    z1 = _encode(fine.encoder, context_pixels)
    action1 = fine.action_embed(actions)
    base = fine.predict(z1, action1)
    options = model.allen_anatomical_settings
    output = {"context_embedding_1": z1}
    if options["mode"] == "hierarchical":
        global_model = model.get_level(2)
        z2 = _encode(global_model.encoder, z1)
        predicted2 = global_model.predict(z2, global_model.action_embed(actions))
        total_z = global_model.anatomical_total(predicted2).squeeze(-1)
        refined = head.condition(base, predicted2, total_z, options["connection"])
        output.update(context_embedding_2=z2, predicted_embedding_2=predicted2)
    else:
        total_z = fine.anatomical_total(base).squeeze(-1)
        # Same observation coordinates; no upper state or upper auxiliary loss.
        refined = head.condition(base, base, total_z, "none")
    output.update(head.decode(refined, head.total_mu + head.total_sd * total_z))
    output["predicted_embedding_1"] = refined
    return output


def _masked_mse(prediction, target, valid):
    if not valid.any():
        return prediction.sum() * 0.
    return F.mse_loss(prediction[valid], target[valid])


def _latent_loss(owner, cfg, level, prediction, target, valid):
    setting = cfg[f"level{level}"]
    component = setting.loss.get("embed", setting.loss)
    loss = prediction.sum() * 0.
    pred_cfg = component.get("pred", {})
    if bool(pred_cfg.get("enabled", True)):
        loss = loss + float(pred_cfg.get("weight", 1.)) * _masked_mse(
            prediction, target.detach() if setting.wm.get("detach_pred_target", False) else target, valid)
    sig_cfg = component.get("sigreg", {})
    if sig_cfg.get("enabled", False) and int(valid.sum()) >= 2:
        # A single valid-state population, without padded/missing triplets.
        regularizer = getattr(owner, f"sigreg_level{level}")
        loss = loss + float(sig_cfg.get("weight", 1.)) * regularizer(target[valid].unsqueeze(0))
    return loss


def anatomical_forward(self, batch, stage, cfg, *, normalize_batch):
    """Predictions precede loss-only target encoding; never teacher-force total."""
    normalize_batch(batch)
    history = int(cfg.level1.wm.history_size)
    pixels, actions, state = (batch[k] for k in ("pixels_level1", "action_level1", "state_level1"))
    if pixels.shape[1] != history + 1 or state.shape != (*pixels.shape[:2], 4):
        raise ValueError("Training requires H+1 reference/target species and four raw targets")
    output = predict_from_context(self.model, pixels[:, :history], actions[:, :history])
    targets = state[:, 1:history + 1]
    total, ratios, complete = anatomical_coordinates(targets)
    head = self.model.get_level(1).anatomical_readout
    observation = masked_standardized_mse(output["predicted_log_lengths"], targets, head.morph_sd)
    total_loss = _masked_mse(output["predicted_log_total"] / head.total_sd,
                            total / head.total_sd, complete)
    ratio_loss = _masked_mse(output["predicted_log_ratios"] / head.ratio_sd,
                            ratios / head.ratio_sd, complete)
    options = self.model.allen_anatomical_settings
    loss = (options["observation_weight"] * observation + options["total_weight"] * total_loss
            + options["ratio_weight"] * ratio_loss)
    output.update(readout_loss=observation, total_coordinate_loss=total_loss,
                  ratio_coordinate_loss=ratio_loss)
    if options["objective"] == "jepa":
        target1 = _encode(self.model.get_level(1).encoder, pixels[:, 1:history + 1])
        latent1 = _latent_loss(self, cfg, 1, output["predicted_embedding_1"], target1,
                              torch.isfinite(targets).any(dim=-1))
        output["latent_loss_level1"] = latent1
        loss = loss + latent1
        if options["mode"] == "hierarchical":
            target2 = _encode(self.model.get_level(2).encoder, target1)
            latent2 = _latent_loss(self, cfg, 2, output["predicted_embedding_2"], target2, complete)
            output["latent_loss_level2"] = latent2
            loss = loss + latent2
    output["loss"] = loss
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite anatomical training objective")
    self.log_dict({f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k},
                  on_step=True, sync_dist=True)
    return output
