"""按习性尺度分层的 H-JEPA：第 1 层与上游完全相同，第 2、3 层的单位按习性事件分段
（设计草案 docs/design/第二层设计草案_2026-10-15.md 第 4 节；数据见 evowm.data.transect 的 units）。

- 每一层都是上游的 JEPA 对象（编码器、动作嵌入、预测器、SIGReg 都用上游的构建代码），容器是上游的 HJEPA，
  所以保存的模型对象里没有 EvoWM 自己的类。
- 第 2 层的状态：单位内各物种的第 1 层嵌入取掩码平均，再经第 2 层编码器；第 3 层：所含运动方式单位的第 2 层
  表示取掩码平均，再经第 3 层编码器。上层编码的是下层的表示，与 H-JEPA 相同。
- 动作：单位一级的习性变化（数据集直接给出，与第 1 层同构），经该层的动作嵌入。
- 损失：每层都是上游的教师强制一步预测 MSE 加 SIGReg（rollout_n = 1），总损失为各层之和；上层的梯度回传到下层编码器。

训练入口：上游 main_hjepa.py 在配置有 evowm_units: true 时改用本模块的 create_world_model 与 units_forward
（改动登记在 hjepa/UPSTREAM.md）。本模块需要上游 h_jepa 目录在 sys.path 上（训练时就是工作目录）。
"""
from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

UNIT_KEYS = {2: ("u2_pixels", "u2_mask", "u2_action"), 3: ("u3_pixels", "u3_mask", "u3_submask", "u3_action")}


def _plain(cfg) -> dict:
    from omegaconf import OmegaConf
    return OmegaConf.to_container(cfg, resolve=True) if not isinstance(cfg, dict) else copy.deepcopy(cfg)


def create_world_model(cfg):
    """第 1 层按上游原样构建（与只有第 1 层的模型完全一样）；第 2、3 层各用上游构建代码建一个单层 JEPA。"""
    from omegaconf import OmegaConf
    from hjepa_utils import create_world_model as upstream_create
    from models.hjepa import HJEPA

    base = _plain(cfg)
    n_levels = int(base["num_levels"])
    units = base["data"]["dataset"].get("units") or {}
    levels = list(units.get("levels", []))
    if n_levels != 1 + len(levels):
        raise ValueError(f"num_levels={n_levels} 与上层单位 {levels} 不符（应为 1 + 层数）")

    c1 = copy.deepcopy(base)
    c1["num_levels"] = 1
    m1, losses = upstream_create(OmegaConf.create(c1))
    jepas = [m1.get_level(1)]
    embed_dims = {1: m1.embed_dims[1]}
    for level in range(2, n_levels + 1):
        ck = copy.deepcopy(base)
        ck["num_levels"] = 1
        ck["level1"] = copy.deepcopy(base[f"level{level}"])
        ck["data"]["dataset"]["level1"]["frameskip"] = 1
        mk, lk = upstream_create(OmegaConf.create(ck))
        jepa = mk.get_level(1)
        jepa.level = level
        jepas.append(jepa)
        embed_dims[level] = mk.embed_dims[1]
        for name, module in lk.items():
            losses[name.replace("level1", f"level{level}")] = module

    model = HJEPA(jepas=jepas)
    model.embed_dims = embed_dims
    model.pixel_embed_dims = {}
    model.proprio_embed_dims = {}
    model.evowm_units = {"levels": levels, "max_members": int(units.get("max_members", 16)),
                         "max_subunits": int(units.get("max_subunits", 8)), "history": int(units.get("history", 3))}
    return model, losses


# ---------------------------------------------------------------------------
def masked_mean(z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """z (..., N, D)，mask (..., N) → (..., D)；没有有效成员的位置为 0。"""
    m = mask.to(z.dtype).unsqueeze(-1)
    return (z * m).sum(-2) / m.sum(-2).clamp_min(1.0)


def encode_members(encoder, pixels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """只对有效成员运行编码器，结果放回原位置（无效位置为 0）。pixels (..., P)，mask (...)。"""
    flat = pixels[mask]
    z = encoder(flat)
    out = z.new_zeros(*mask.shape, z.shape[-1])
    out[mask] = z
    return out


def unit_embed_l2(model, pixels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """(..., M, P) 的单位成员 → (..., D2) 的第 2 层表示。"""
    j1, j2 = model.get_level(1), model.get_level(2)
    z1 = masked_mean(encode_members(j1.encoder, pixels, mask), mask)
    lead = z1.shape[:-1]
    return j2.encoder(z1.reshape(-1, z1.shape[-1])).reshape(*lead, -1)


def unit_embed_l3(model, pixels: torch.Tensor, mask: torch.Tensor, submask: torch.Tensor) -> torch.Tensor:
    """(..., K, M, P) 的栖境单位（K 个运动方式单位，各 M 种）→ (..., D3) 的第 3 层表示。"""
    z2 = masked_mean(unit_embed_l2(model, pixels, mask), submask)
    lead = z2.shape[:-1]
    return model.get_level(3).encoder(z2.reshape(-1, z2.shape[-1])).reshape(*lead, -1)


def unit_action_embed(jepa, action: torch.Tensor) -> torch.Tensor:
    return jepa.action_embed(jepa.action_encoder(action))


# ---------------------------------------------------------------------------
def _level_loss(module, cfg, level: int, emb: torch.Tensor, act_emb: torch.Tensor, output: dict) -> torch.Tensor:
    """与上游 hjepa_forward 在 rollout_n = 1、只有 embed 分量时的计算完全相同。"""
    from hjepa_utils import _loss_components, _loss_term_enabled, _loss_term_weight, _sigreg_module_name

    level_cfg = cfg[f"level{level}"]
    suffix = f"_level{level}"
    H = int(level_cfg.wm.history_size)
    if int(level_cfg.wm.get("rollout_n", 1)) != 1 or level_cfg.wm.get("nsteps", None) is not None:
        raise ValueError("按习性分层的模型只支持 rollout_n = 1、不用 nsteps")
    comps = _loss_components(level_cfg.loss)
    if set(comps) != {"embed"}:
        raise ValueError("按习性分层的模型只支持 embed 损失分量")
    jepa = module.model.get_level(level)
    pred = jepa.predict(emb[:, :H], act_emb[:, :H])
    tgt = emb[:, 1:H + 1]
    with torch.no_grad():
        output[f"mse_loss{suffix}"] = F.mse_loss(pred, tgt)
        output[f"dim_mean_loss{suffix}"] = tgt.mean()
        output[f"dim_std_loss{suffix}"] = tgt.std()

    comp = comps["embed"]
    target = emb.detach() if level_cfg.wm.get("detach_pred_target") else emb
    loss = None
    if _loss_term_enabled(comp, "pred"):
        tf = F.mse_loss(pred, target[:, 1:H + 1])
        output[f"teacher_forcing_loss{suffix}"] = tf
        output[f"pred_loss{suffix}"] = tf
        loss = _loss_term_weight(comp, "pred") * tf
    if _loss_term_enabled(comp, "sigreg"):
        sig = getattr(module, _sigreg_module_name("embed", level))(emb.flatten(2).transpose(0, 1))
        output[f"sigreg_loss{suffix}"] = sig
        w = _loss_term_weight(comp, "sigreg") * sig
        loss = w if loss is None else loss + w
    output[f"embed_loss{suffix}"] = loss
    output[f"loss{suffix}"] = loss
    if loss.isnan():
        raise ValueError(f"NaN loss encountered at level {level}!")
    return loss


def units_forward(self, batch, stage, cfg, *, normalize_batch):
    """训练前向：第 1 层与上游相同；第 2、3 层用单位窗口。总损失为各层之和。"""
    normalize_batch(batch)
    model = self.model
    n_levels = int(cfg.num_levels)
    output = {}

    j1 = model.get_level(1)
    enc = j1.encode({"pixels": batch["pixels_level1"], "action": batch["action_level1"]}, key="pixels")
    output["embed_1"], output["action_1"] = enc["embed_0"], enc["action_0"]
    total = _level_loss(self, cfg, 1, enc["embed_0"], enc["action_0"], output)

    if n_levels >= 2:
        missing = [k for k in UNIT_KEYS[2] if k not in batch]
        if missing:
            raise KeyError(f"批次里缺少第 2 层单位的键 {missing}（数据集需配置 units.levels 含 locomotor）")
        emb2 = unit_embed_l2(model, batch["u2_pixels"], batch["u2_mask"])
        act2 = unit_action_embed(model.get_level(2), batch["u2_action"])
        output["embed_2"], output["action_2"] = emb2, act2
        total = total + _level_loss(self, cfg, 2, emb2, act2, output)
    if n_levels >= 3:
        missing = [k for k in UNIT_KEYS[3] if k not in batch]
        if missing:
            raise KeyError(f"批次里缺少第 3 层单位的键 {missing}（数据集需配置 units.levels 含 realm）")
        emb3 = unit_embed_l3(model, batch["u3_pixels"], batch["u3_mask"], batch["u3_submask"])
        act3 = unit_action_embed(model.get_level(3), batch["u3_action"])
        output["embed_3"], output["action_3"] = emb3, act3
        total = total + _level_loss(self, cfg, 3, emb3, act3, output)

    output["loss"] = total
    self.log_dict({f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}, on_step=True, sync_dist=True)
    return output
