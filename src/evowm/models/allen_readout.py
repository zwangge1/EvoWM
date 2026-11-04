"""Anatomical output organizations on the original single-level JEPA.

These are supervised OUTPUT readouts, not the temporal hierarchy of H-JEPA.
The encoder, action encoder, predictor and JEPA/SIGReg loss are built by the
upstream code. Four equally sized scalar heads share its predicted embedding:

* parallel: four independent length predictions;
* allocation: total leg length, then two free allocation logits (third fixed 0);
* proximal: femur -> tibiotarsus -> tarsometatarsus;
* distal: the same dependencies in reverse, an ordering control.

All variants have exactly the same trainable parameter count. Extra condition
slots are zero where unused. Sequential heads receive earlier PREDICTIONS,
never target measurements. No sign, monotonicity or response order is imposed.

Opt-in configuration:
  evowm_allen_readout: {variant: allocation, hidden_dim: 64, weight: 1.0}

Integration: create_world_model(cfg), then configure_readout(model, train_ds)
before constructing the optimizer. allen_forward wraps hjepa_forward. Because
the upstream function does not return its predicted embeddings, the supervised
term uses one additional predictor pass on the same encoded context. During
training this has its own dropout draw, identically for all candidate readouts.
The upstream loss is logged under its existing names; allen_total_loss logs the
actual combined training objective returned as output['loss'].
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


TARGET_COLUMNS = ("m_tarsus", "b_fem", "b_tbt", "b_tmt")
VARIANTS = ("parallel", "allocation", "proximal", "distal")


def _settings(cfg):
    setting = cfg.get("evowm_allen_readout")
    if isinstance(setting, str):
        result = {"variant": setting}
    elif setting is not None:
        result = dict(setting)
    else:
        raise ValueError("evowm_allen_readout is required")
    if result.get("variant") == "flat":
        result["variant"] = "parallel"
    if result.get("variant") not in VARIANTS:
        raise ValueError(f"Readout variant must be one of {VARIANTS}")
    result.setdefault("hidden_dim", 64)
    result.setdefault("weight", 1.0)
    weight = float(result["weight"])
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError("Readout loss weight must be finite and positive")
    return result


class AnatomicalReadout(nn.Module):
    """Map predicted latent states to four raw log10(mm) lengths."""

    def __init__(self, embed_dim: int, variant: str, hidden_dim: int = 64):
        super().__init__()
        variant = "parallel" if variant == "flat" else variant
        if variant not in VARIANTS or embed_dim < 1 or hidden_dim < 1:
            raise ValueError("Invalid readout variant or dimensions")
        self.variant = variant
        self.embed_dim = int(embed_dim)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(embed_dim + 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
            for _ in TARGET_COLUMNS
        ])
        self.register_buffer("morph_mu", torch.zeros(4))
        self.register_buffer("morph_sd", torch.ones(4))
        self.register_buffer("total_mu", torch.zeros(()))
        self.register_buffer("total_sd", torch.ones(()))
        self.register_buffer("allocation_center", torch.zeros(2))
        self.register_buffer("configured", torch.tensor(False))

    @torch.no_grad()
    def configure(self, training_values):
        """Fit observed-space scales from training rows only; caller selects rows."""
        values = torch.as_tensor(training_values, dtype=torch.float64, device=self.morph_mu.device)
        if values.ndim != 2 or values.shape[1] != 4 or torch.isinf(values).any():
            raise ValueError("Expected training log-length matrix (N, 4), finite or NaN")
        means, scales = [], []
        for j in range(4):
            observed = values[:, j][torch.isfinite(values[:, j])]
            if len(observed) < 2:
                raise ValueError(f"At least two training observations required for {TARGET_COLUMNS[j]}")
            means.append(observed.mean())
            scale = observed.std(unbiased=False)
            scales.append(scale if scale > 0 else scale.new_tensor(1.0))
        complete = values[torch.isfinite(values[:, 1:]).all(dim=1), 1:]
        if len(complete) < 2:
            raise ValueError("At least two complete training bone triplets are required")
        # Stable log10(sum(10**log_length)); no raw-length overflow.
        total = torch.logsumexp(complete * math.log(10), dim=-1) / math.log(10)
        total_sd = total.std(unbiased=False)
        self.morph_mu.copy_(torch.stack(means))
        self.morph_sd.copy_(torch.stack(scales))
        self.total_mu.copy_(total.mean())
        self.total_sd.copy_(total_sd if total_sd > 0 else total_sd.new_tensor(1.0))
        self.allocation_center.copy_((complete[:, :2] - complete[:, 2:]).mean(dim=0) * math.log(10))
        self.configured.fill_(True)

    def _head(self, index, embedding, first=None, second=None):
        zero = torch.zeros_like(embedding[..., :1])
        condition = [zero if first is None else first.unsqueeze(-1),
                     zero if second is None else second.unsqueeze(-1)]
        return self.heads[index](torch.cat([embedding] + condition, dim=-1)).squeeze(-1)

    def forward(self, embedding):
        if not bool(self.configured):
            raise RuntimeError("Configure readout from training rows before prediction")
        if embedding.ndim < 2 or embedding.shape[-1] != self.embed_dim:
            raise ValueError("Unexpected predicted-embedding shape")
        external = self._head(0, embedding)
        if self.variant == "allocation":
            total_z = self._head(1, embedding)
            first = self._head(2, embedding, first=total_z) + self.allocation_center[0]
            second = self._head(3, embedding, first=total_z) + self.allocation_center[1]
            logits = torch.stack([first, second, torch.zeros_like(first)], dim=-1)
            total = self.total_mu + self.total_sd * total_z
            bones = total.unsqueeze(-1) + F.log_softmax(logits, dim=-1) / math.log(10)
            external_log = self.morph_mu[0] + self.morph_sd[0] * external
            return torch.cat([external_log.unsqueeze(-1), bones], dim=-1)
        order = (3, 2, 1) if self.variant == "distal" else (1, 2, 3)
        predicted = {0: external}
        prior = []
        for index in order:
            if self.variant == "parallel":
                predicted[index] = self._head(index, embedding)
            else:
                predicted[index] = self._head(index, embedding,
                    first=prior[0] if prior else None, second=prior[1] if len(prior) > 1 else None)
                prior.append(predicted[index])
        normalized = torch.stack([predicted[i] for i in range(4)], dim=-1)
        return self.morph_mu + self.morph_sd * normalized


def masked_standardized_mse(prediction, target, scale):
    """Equal weight per measured trait, rather than per (species, trait) cell.

An entirely missing batch returns differentiable zero. NaNs are replaced before
arithmetic, so absent measurements cannot poison gradients through 0 * NaN.
"""
    if prediction.shape != target.shape or prediction.ndim < 2 or prediction.shape[-1] != 4:
        raise ValueError("Prediction and target must have matching (..., 4) shapes")
    if scale.shape != (4,) or not torch.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("Expected four finite positive training scales")
    if torch.isinf(target).any():
        raise ValueError("Infinite observed target")
    valid = torch.isfinite(target)
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    safe_prediction = torch.where(valid, prediction, torch.zeros_like(prediction))
    squared = ((safe_prediction - safe_target) / scale).square()
    squared = squared.reshape(-1, 4)
    count = valid.reshape(-1, 4).sum(dim=0)
    active = count > 0
    per_trait = squared.sum(dim=0) / count.clamp_min(1)
    return (per_trait * active).sum() / active.sum().clamp_min(1)


def create_world_model(cfg):
    """Original upstream JEPA with one registered observation readout attached."""
    from hjepa_utils import create_world_model as upstream_create
    if int(cfg.num_levels) != 1 or cfg.get("evowm_units", False):
        raise ValueError("Anatomical readout experiments use one JEPA level; not clade pooling")
    if int(cfg.level1.wm.get("rollout_n", 1)) != 1 or cfg.level1.wm.get("nsteps") is not None:
        raise ValueError("Anatomical readouts currently support teacher-forced one-step training")
    options = _settings(cfg)
    model, losses = upstream_create(cfg)
    model.get_level(1).allen_readout = AnatomicalReadout(
        int(model.embed_dims[1]), options["variant"], int(options["hidden_dim"]))
    return model, losses


def configure_readout(model, dataset):
    """Called with the actual training dataset before optimizer construction."""
    if dataset.split != "train" or tuple(dataset.state_cols) != TARGET_COLUMNS:
        raise ValueError("Readout configuration requires the Allen training dataset and fixed target order")
    rows = torch.as_tensor(dataset.training_rows, dtype=torch.bool)
    split_rows = torch.as_tensor(dataset.rows_mask, dtype=torch.bool)
    if rows.ndim != 1 or not torch.equal(rows, split_rows):
        raise ValueError("Training normalization rows must match the actual training split")
    state = torch.as_tensor(dataset.S)
    model.get_level(1).allen_readout.configure(state[rows])


def allen_forward(self, batch, stage, cfg, *, normalize_batch):
    """Upstream JEPA/SIGReg objective plus the explicitly supervised readout term."""
    from hjepa_utils import hjepa_forward
    options = _settings(cfg)
    output = hjepa_forward(self, batch, stage, cfg, normalize_batch=normalize_batch)
    history = int(cfg.level1.wm.history_size)
    jepa = self.model.get_level(1)
    predicted_embedding = jepa.predict(output["embed_1"][:, :history], output["action_1"][:, :history])
    prediction = jepa.allen_readout(predicted_embedding)
    # Measured targets are used only by the loss, never by sequential heads.
    target = batch["state_level1"][:, 1:history + 1]
    observation_loss = masked_standardized_mse(prediction, target, jepa.allen_readout.morph_sd)
    output["jepa_loss"] = output["loss"]
    output["readout_loss"] = observation_loss
    output["loss"] = output["jepa_loss"] + float(options["weight"]) * observation_loss
    output["predicted_log_lengths"] = prediction
    if not torch.isfinite(output["loss"]):
        raise ValueError("Nonfinite Allen training objective")
    self.log_dict({f"{stage}/readout_loss": observation_loss.detach(),
                   f"{stage}/allen_total_loss": output["loss"].detach()}, on_step=True, sync_dist=True)
    return output
