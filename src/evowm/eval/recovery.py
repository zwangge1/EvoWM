"""模拟恢复检验 v1 的观测空间评估（判据见 docs/prereg/sim_recovery_v1.md）。

全部读数在观测空间（log10 单位）计算：
- 探针：训练物种上，岭回归把第 1 层编码器表征映射回状态列。
- E1 习性增益：留出科的样带上，有、无习性动作两种输入下预测下一种腿长的 RMSE 之差。
- E2 反事实体制对比：把目标物种的习性换成各体制原型，读出腿长相对"栖树"的变化。
- E3 半衰期：样带第 1 个预测（预测器只看前一种）对前一种腿长输入的敏感度随进化距离的衰减。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

LEG_TRAITS = ["m_tarsus", "k_tarsometatarsus", "k_tibiotarsus", "k_femur"]
BASE_REGIME = "insessorial"
DELTA = 0.02  # E3 的输入扰动（log10）
MIN_IDENT = 20  # E2 可识别组合：该体制与基准体制在该性状上各至少有多少个训练物种有观测


def load_world_model(ckpt: Path, hjepa_dir: Path):
    """加载上游保存的 HJEPA 对象（类路径如 models.hjepa.HJEPA，需把 h_jepa 目录放进 sys.path）。"""
    p = str(hjepa_dir / "h_jepa")
    if p not in sys.path:
        sys.path.insert(0, p)
    model = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.eval()
    return model


@dataclass
class Windows:
    rows: np.ndarray   # (N, L) 物种表行号
    dist: np.ndarray   # (N, L-1) 相邻物种的进化距离（Myr）


def sample_windows(ds, n_orders: int, length: int, seed: int) -> Windows:
    """从数据集的样带分布里取窗口：n_orders 个随机树与随机旋转，每个取全部起点。"""
    rng = np.random.default_rng(seed)
    rows, dist = [], []
    for _ in range(n_orders):
        r, d = ds._new_order(rng)
        n = len(r) - length + 1
        idx = np.arange(n)[:, None] + np.arange(length)[None, :]
        rows.append(r[idx])
        dist.append(d[idx[:, :-1]])
    return Windows(np.concatenate(rows), np.concatenate(dist))


class Level1:
    """第 1 层的编码、动作嵌入与预测。"""

    def __init__(self, model):
        self.jepa = model.get_level(1)

    @torch.no_grad()
    def encode(self, pix: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(pix.reshape(-1, pix.shape[-1])).float()
        z = self.jepa.encoder(x)
        return z.reshape(*pix.shape[:-1], -1).numpy()

    @torch.no_grad()
    def predict(self, emb: np.ndarray, act: np.ndarray, batch: int = 4096) -> np.ndarray:
        """emb (N, H, D)，act (N, H, A) → 预测 (N, H, D)，第 h 个输出对应第 h+1 个物种。"""
        out = []
        for i in range(0, len(emb), batch):
            e = torch.from_numpy(emb[i:i + batch]).float()
            a = self.jepa.action_embed(self.jepa.action_encoder(torch.from_numpy(act[i:i + batch]).float()))
            out.append(self.jepa.predict(e, a).numpy())
        return np.concatenate(out)


def fit_probe(z: np.ndarray, S: np.ndarray, rows_train: np.ndarray, alpha: float = 1.0):
    """每个状态列一个岭回归，只用该列有观测的训练物种。"""
    probes = []
    for j in range(S.shape[1]):
        ok = rows_train[np.isfinite(S[rows_train, j])]
        probes.append(Ridge(alpha=alpha).fit(z[ok], S[ok, j]))
    return probes


def apply_probe(probes, z: np.ndarray) -> np.ndarray:
    flat = z.reshape(-1, z.shape[-1])
    out = np.stack([p.predict(flat) for p in probes], axis=1)
    return out.reshape(*z.shape[:-1], len(probes))


def r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ok = np.isfinite(y)
    y, yhat = y[ok], yhat[ok]
    return float(1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2))


def family_bootstrap_rmse_diff(err_a, err_b, fam, n_boot=1000, seed=0):
    """按科重抽样的 RMSE 差（b − a）及 95% 置信区间。err_* 为逐预测误差（NaN 表示无观测）。"""
    ok = np.isfinite(err_a) & np.isfinite(err_b)
    ea, eb, f = err_a[ok], err_b[ok], fam[ok]
    fams, inv = np.unique(f, return_inverse=True)
    sa = np.bincount(inv, ea ** 2, len(fams))
    sb = np.bincount(inv, eb ** 2, len(fams))
    cnt = np.bincount(inv, minlength=len(fams)).astype(float)
    point = np.sqrt(sb.sum() / cnt.sum()) - np.sqrt(sa.sum() / cnt.sum())
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        k = rng.integers(0, len(fams), len(fams))
        boots.append(np.sqrt(sb[k].sum() / cnt[k].sum()) - np.sqrt(sa[k].sum() / cnt[k].sum()))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(point), float(lo), float(hi), int(ok.sum()), len(fams)


def evaluate_run(ds_train, ds_val, model, table: pd.DataFrame, truth: dict,
                 n_orders: int = 8, seed: int = 0) -> dict:
    L1 = Level1(model)
    H = int(L1.jepa.predictor.predictor.pos_embedding.shape[1]) if hasattr(L1.jepa.predictor, "predictor") else 3
    state_cols = ds_train.state_cols
    S = ds_train.S
    leg_idx = {c: state_cols.index(c) for c in LEG_TRAITS if c in state_cols}
    res: dict = {"history_size": H}

    # ---- 探针 ----
    z_all = L1.encode(ds_train.pix)
    rows_tr = np.flatnonzero(ds_train.rows_mask)
    rows_va = np.flatnonzero(ds_val.rows_mask)
    probes = fit_probe(z_all, S, rows_tr)
    s_hat = apply_probe(probes, z_all)
    res["probe_r2_val"] = {c: r2(S[rows_va, j], s_hat[rows_va, j]) for j, c in enumerate(state_cols)}

    # ---- E1：有、无习性的预测误差 ----
    W = sample_windows(ds_val, n_orders, H + 1, seed)
    act_eco = np.stack([ds_val.build_action(r, d, eco=True) for r, d in zip(W.rows, W.dist)])
    act_no = np.stack([ds_val.build_action(r, d, eco=False) for r, d in zip(W.rows, W.dist)])
    emb = L1.encode(ds_val.pix[W.rows])                         # (N, H+1, D)
    pred_eco = apply_probe(probes, L1.predict(emb[:, :H], act_eco[:, :H]))
    pred_no = apply_probe(probes, L1.predict(emb[:, :H], act_no[:, :H]))
    tgt_rows = W.rows[:, 1:H + 1]
    true = S[tgt_rows]                                          # (N, H, S)
    fam = table["family"].to_numpy()[tgt_rows].reshape(-1)
    e1 = {}
    for c, j in leg_idx.items():
        ea = (pred_eco[..., j] - true[..., j]).reshape(-1)
        en = (pred_no[..., j] - true[..., j]).reshape(-1)
        point, lo, hi, n, nf = family_bootstrap_rmse_diff(ea, en, fam, seed=seed)
        e1[c] = {"rmse_eco": float(np.sqrt(np.nanmean(ea ** 2))), "rmse_no_eco": float(np.sqrt(np.nanmean(en ** 2))),
                 "gain": point, "ci95": [lo, hi], "n_pred": n, "n_families": nf}
    res["E1"] = e1
    res["n_windows"] = int(len(W.rows))

    # ---- E1 补充：只看体制转变的相邻对（前一种与目标种体制不同），并与"已知真值"的理想预测比较 ----
    reg_all = table["sim_regime"].to_numpy()
    prev_rows = W.rows[:, :H]
    trans = (reg_all[prev_rows] != reg_all[tgt_rows]).reshape(-1)
    alpha = float(truth["alpha"])
    slope = float(truth.get("allometric_slope", 0.33))
    mass = table["m_mass"].to_numpy()
    k = np.exp(-alpha * W.dist[:, :H]).reshape(-1)
    e1t = {"share_transition_pairs": float(trans.mean())}
    for c, j in leg_idx.items():
        ea = (pred_eco[..., j] - true[..., j]).reshape(-1)
        en = (pred_no[..., j] - true[..., j]).reshape(-1)
        ea_t, en_t = np.where(trans, ea, np.nan), np.where(trans, en, np.nan)
        point, lo, hi, n, nf = family_bootstrap_rmse_diff(ea_t, en_t, fam, seed=seed)
        # 理想预测：已知 α、体制最优值、异速斜率，只用前一种（不知道目标种的体重，按前一种体重代入）
        off = truth["leg_offsets"][truth["leg_source"][c]]
        th = np.array([off.get(r, 0.0) if isinstance(r, str) else 0.0 for r in reg_all])
        tr_rows = np.flatnonzero(ds_train.rows_mask)
        tbar = float(np.mean(th[tr_rows]))
        xA = table[c].to_numpy()[prev_rows].reshape(-1)
        base = 1.0 + slope * mass[prev_rows].reshape(-1)
        rA = xA - base
        thA, thB = th[prev_rows].reshape(-1), th[tgt_rows].reshape(-1)
        yB = true[..., j].reshape(-1)
        o_w = base + thB + k * (rA - thA) - yB
        o_o = base + tbar + k * (rA - tbar) - yB
        o_w_t, o_o_t = np.where(trans, o_w, np.nan), np.where(trans, o_o, np.nan)
        opt, olo, ohi, _, _ = family_bootstrap_rmse_diff(o_w_t, o_o_t, fam, seed=seed)
        e1t[c] = {"gain": point, "ci95": [lo, hi], "n_pred": n, "n_families": nf,
                  "oracle_gain": opt, "oracle_ci95": [olo, ohi],
                  "ratio_to_oracle": float(point / opt) if opt > 0 else float("nan")}
    res["E1_transition"] = e1t

    # ---- E2：反事实体制对比（只用每个窗口的第 1 个预测）----
    regimes = truth["regimes"]
    reg = table["sim_regime"].to_numpy()
    E_all, Eg_all = ds_train.eco_rows(np.arange(len(table)))
    full_groups = Eg_all.min(axis=1) > 0
    protos = {}
    for r in regimes:
        sel = rows_tr[(reg[rows_tr] == r) & full_groups[rows_tr]]
        protos[r] = E_all[sel].mean(axis=0)
    emb0 = emb[:, :1]
    leg_pred = {}
    for r in regimes:
        acts = []
        for rr, d in zip(W.rows, W.dist):
            E, Eg = ds_val.eco_rows(rr[:2])
            E = E.copy(); Eg = Eg.copy()
            E[1] = protos[r]; Eg[1] = 1.0
            acts.append(ds_val.build_action(rr[:2], d[:1], eco=True, E=E, Eg=Eg))
        acts = np.stack(acts)
        leg_pred[r] = apply_probe(probes, L1.predict(emb0, acts[:, :1]))[:, 0, :]
    contrasts = {r: {c: float(np.mean(leg_pred[r][:, j] - leg_pred[BASE_REGIME][:, j])) for c, j in leg_idx.items()}
                 for r in regimes if r != BASE_REGIME}
    src = truth["leg_source"]
    off = truth["leg_offsets"]
    pairs = [(contrasts[r][c], off[src[c]][r] - off[src[c]][BASE_REGIME]) for r in contrasts for c in leg_idx]
    pv, tv = np.array(pairs).T
    rho = spearmanr(pv, tv).statistic if np.ptp(tv) > 0 else float("nan")
    res["E2"] = {"contrasts": contrasts, "spearman_rho": float(rho),
                 "median_abs_contrast": float(np.median(np.abs(pv)))}

    # ---- E2 相对腿长：扣除体重通路（腿长对 log 体重的 OLS 斜率由训练物种的观测值估计）----
    j_m = state_cols.index("m_mass")
    rel_contrasts, slopes = {}, {}
    for c, j in leg_idx.items():
        ok = rows_tr[np.isfinite(S[rows_tr, j]) & np.isfinite(S[rows_tr, j_m])]
        b = float(np.polyfit(S[ok, j_m], S[ok, j], 1)[0])
        slopes[c] = b
    for r in regimes:
        if r == BASE_REGIME:
            continue
        dm = leg_pred[r][:, j_m] - leg_pred[BASE_REGIME][:, j_m]
        rel_contrasts[r] = {c: float(np.mean(leg_pred[r][:, j] - leg_pred[BASE_REGIME][:, j] - slopes[c] * dm))
                            for c, j in leg_idx.items()}
    pairs_r = [(rel_contrasts[r][c], off[src[c]][r] - off[src[c]][BASE_REGIME]) for r in rel_contrasts for c in leg_idx]
    pr, tr_ = np.array(pairs_r).T
    rho_r = spearmanr(pr, tr_).statistic if np.ptp(tr_) > 0 else float("nan")
    # 可识别组合：训练物种中该体制与基准体制在该性状上都至少有 MIN_IDENT 种有观测
    ident = {}
    for c, j in leg_idx.items():
        obs = rows_tr[np.isfinite(S[rows_tr, j])]
        n_base = int(np.sum(reg[obs] == BASE_REGIME))
        ident[c] = [r for r in regimes if r != BASE_REGIME and n_base >= MIN_IDENT and int(np.sum(reg[obs] == r)) >= MIN_IDENT]
    pv_i = np.array([rel_contrasts[r][c] for c in leg_idx for r in ident[c]])
    tv_i = np.array([off[src[c]][r] - off[src[c]][BASE_REGIME] for c in leg_idx for r in ident[c]])
    rho_i = spearmanr(pv_i, tv_i).statistic if len(tv_i) > 2 and np.ptp(tv_i) > 0 else float("nan")
    mt = np.array([rel_contrasts[r]["m_tarsus"] for r in ident.get("m_tarsus", [])])
    res["E2_relative"] = {"contrasts": rel_contrasts, "allometric_slopes": slopes, "spearman_rho": float(rho_r),
                          "median_abs_contrast": float(np.median(np.abs(pr))),
                          "identifiable": ident, "n_identifiable": int(len(pv_i)),
                          "spearman_rho_identifiable": float(rho_i),
                          "m_tarsus_mean_abs": float(np.mean(np.abs(mt))) if len(mt) else float("nan"),
                          "mass_contrast": {r: float(np.mean(leg_pred[r][:, j_m] - leg_pred[BASE_REGIME][:, j_m]))
                                            for r in regimes if r != BASE_REGIME}}

    # ---- E3：对前一种腿长输入的敏感度随距离衰减 → t½ ----
    j_t = state_cols.index("m_tarsus")
    cols = ds_train.morph_cols
    m = len(cols)
    pert = np.zeros(ds_val.pix.shape[1], dtype=np.float32)
    for c in ("m_tarsus", "k_tarsometatarsus"):
        k = cols.index(c)
        pert[k] = DELTA / ds_train.morph_sd[k]
    pix0 = ds_val.pix[W.rows[:, 0]].copy()
    pix0_p = pix0 + pert[None, :] * np.concatenate([pix0[:, m:], pix0[:, m:]], axis=1)  # 只扰动有观测的列
    act0 = act_eco[:, :1]
    base = apply_probe(probes, L1.predict(L1.encode(pix0)[:, None], act0))[:, 0, j_t]
    moved = apply_probe(probes, L1.predict(L1.encode(pix0_p)[:, None], act0))[:, 0, j_t]
    s = (moved - base) / DELTA
    d0 = W.dist[:, 0]
    bins = np.quantile(d0, np.linspace(0, 1, 21))
    which = np.clip(np.digitize(d0, bins[1:-1]), 0, 19)
    bd = np.array([np.median(d0[which == b]) for b in range(20)])
    bs = np.array([np.median(s[which == b]) for b in range(20)])
    (a1,), _ = curve_fit(lambda x, a: np.exp(-a * x), bd, bs, p0=[0.05], bounds=(1e-5, 10))
    (c2, a2), _ = curve_fit(lambda x, c, a: c * np.exp(-a * x), bd, bs, p0=[1.0, 0.05], bounds=([0, 1e-5], [10, 10]))
    res["E3"] = {"t_half": float(np.log(2) / a1), "t_half_free_amplitude": float(np.log(2) / a2),
                 "amplitude": float(c2), "binned_distance": bd.round(3).tolist(), "binned_slope": bs.round(4).tolist()}
    return res
