"""多近亲 OU 对照（探索性；设计 docs/design/多近亲OU对照_2026-10-19.md）。只算第一层。

与 evowm.eval.ou（OU 式对照 v1）完全相同的窗口、读数、对照与同一批预测，另加两种用窗口里前面全部近亲（与世界模型这一步
看到的相同，1 到 3 种）的 OU 预测：
- m_no（多近亲·不告诉习性）：ŷ_B = μ + Σ_h w_h (y_h − μ)，w = (R + ηI)⁻¹ c，R_hk = exp(−α d_hk)，c_h = exp(−α d_hB)；
  缺测的近亲不用。α、η 逐项尺寸在训练科的窗口上用最小二乘拟合（全部位置的预测一起）。
- m_full（多近亲·全部 47 项）：上式 + 岭回归（残差 ~ [z, s1·z, s2·z, s1, s2]），z = Σ_h (w_h / Σw) x_hB 为各近亲与 B 的
  习性差按上式权重的平均（50 列，与动作的习性部分同样的算法），s1、s2 为 B 与紧挨着的前一种的两列距离；
  岭回归的行、抽样与惩罚选法同 evowm.eval.ou（按训练科分组 5 折交叉验证）。
窗口内两种的亲缘距离由叶到根的距离与相邻叶 LCA 深度算：d_ij = dtip_i + dtip_j − 2·min(LCA_i..LCA_{j−1})（深度优先叶序里
两叶的 LCA 是它们之间相邻 LCA 里最浅的）。随机数的用法与 recovery.sample_windows 相同，取到的窗口与相邻距离逐一核对。
汇总方式同 evowm.eval.ou：（科，类型 × 位置，性状），下标 = 类型 × 3 + 位置 − 1。
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .ou import KEYS as OU_KEYS
from .ou import MAX_ROWS, NPOS, features, l1_baselines, ridge_family_cv
from .wholebody import agg, datasets, rel, slopes_for, step_types

NEW_KEYS = ("m_no", "m_full")
KEYS = OU_KEYS + NEW_KEYS
HPARAMS = ("alpha", "eta")                 # 多近亲预测的拟合参数
ALPHA_RANGE = (1e-4, 2.0)
ETA_RANGE = (1e-4, 10.0)
L = NPOS + 1                               # 窗口长度


# ---------------------------------------------------------------------------
def windows_full(ds, n_orders: int, length: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """与 recovery.sample_windows 相同的窗口（同样的随机数用法：_new_order 就是 _order 去掉后两项），
    另返回窗口内两两亲缘距离 D (N, length, length)。"""
    rng = np.random.default_rng(seed)
    rows, dist, Ds = [], [], []
    for _ in range(n_orders):
        r, d, lca, dtip = ds._order(rng)
        n = len(r) - length + 1
        idx = np.arange(n)[:, None] + np.arange(length)[None, :]
        rows.append(r[idx])
        dist.append(d[idx[:, :-1]])
        lca, dtip = np.asarray(lca, dtype=np.float64), np.asarray(dtip, dtype=np.float64)
        D = np.zeros((n, length, length))
        for i in range(length):
            m = np.full(n, np.inf)
            for j in range(i + 1, length):
                m = np.minimum(m, lca[idx[:, j - 1]])
                D[:, i, j] = D[:, j, i] = dtip[idx[:, i]] + dtip[idx[:, j]] - 2.0 * m
        Ds.append(D)
    return np.concatenate(rows), np.concatenate(dist), np.concatenate(Ds)


def habit_pairs(ds, rows: np.ndarray) -> dict:
    """窗口内每一对 (h, j)（h < j）的习性差 (N, 50)：与 build_action 同样的算法，47 项习性差乘两种都有数据的掩码，
    再接 3 个习性组标记的乘积。相邻的一对与动作的习性部分完全相同。"""
    E, Eg = ds.eco_rows(rows.reshape(-1))
    E = E.reshape(rows.shape + (-1,))
    Eg = Eg.reshape(rows.shape + (-1,))
    cm = Eg[..., ds.col_group]
    out = {}
    for j in range(1, rows.shape[1]):
        for h in range(j):
            both = cm[:, j] * cm[:, h]
            out[(h, j)] = np.concatenate([(E[:, j] - E[:, h]) * both, Eg[:, j] * Eg[:, h]], axis=1).astype(np.float32)
    return out


def blup_weights(D: np.ndarray, m: np.ndarray, alpha: float, eta: float) -> dict:
    """三个位置的 OU 条件期望权重：位置 p 的目标为窗口第 p 种，近亲为第 0..p−1 种，返回 (N, p)。
    m (N, L) 为该项尺寸是否有实测；缺测的近亲在矩阵里换成单位行列、右端为 0，权重恰为 0。2×2、3×3 用显式公式。"""
    def ex(i, j):
        return np.exp(-alpha * D[:, i, j])

    mf = m.astype(np.float64)
    a = np.where(m, 1.0 + eta, 1.0)
    out = {1: (ex(0, 1) * mf[:, 0] / a[:, 0])[:, None]}
    r01 = ex(0, 1) * mf[:, 0] * mf[:, 1]
    c0, c1 = ex(0, 2) * mf[:, 0], ex(1, 2) * mf[:, 1]
    det = a[:, 0] * a[:, 1] - r01 ** 2
    out[2] = np.stack([(a[:, 1] * c0 - r01 * c1) / det, (a[:, 0] * c1 - r01 * c0) / det], axis=1)
    r02, r12 = ex(0, 2) * mf[:, 0] * mf[:, 2], ex(1, 2) * mf[:, 1] * mf[:, 2]
    c = (ex(0, 3) * mf[:, 0], ex(1, 3) * mf[:, 1], ex(2, 3) * mf[:, 2])
    a0, a1, a2 = a[:, 0], a[:, 1], a[:, 2]
    k00, k01, k02 = a1 * a2 - r12 ** 2, r02 * r12 - r01 * a2, r01 * r12 - r02 * a1     # 伴随矩阵（对称）
    k11, k12, k22 = a0 * a2 - r02 ** 2, r01 * r02 - a0 * r12, a0 * a1 - r01 ** 2
    det = a0 * k00 + r01 * k01 + r02 * k02
    out[3] = np.stack([(k00 * c[0] + k01 * c[1] + k02 * c[2]) / det,
                       (k01 * c[0] + k11 * c[1] + k12 * c[2]) / det,
                       (k02 * c[0] + k12 * c[1] + k22 * c[2]) / det], axis=1)
    return out


def blup_predict(v_w: np.ndarray, D: np.ndarray, mu: float, alpha: float, eta: float) -> tuple[np.ndarray, dict]:
    """窗口的实测 v_w (N, L)：返回三个位置的预测 (N, 3)（各自只用前面的近亲）与权重。"""
    m = np.isfinite(v_w)
    yc = np.where(m, v_w - mu, 0.0)
    W = blup_weights(D, m, alpha, eta)
    pred = np.stack([mu + (W[p] * yc[:, :p]).sum(axis=1) for p in range(1, L)], axis=1)
    return pred, W


def blup_fit(v_w: np.ndarray, D: np.ndarray, mu: float) -> tuple[float, float]:
    """α、η 的最小二乘：训练窗口全部位置的预测，目标与紧挨着的前一种都有实测的才计（与简单方法的训练对相同）。
    先在对数网格（9 × 5）上找，再从最好的格点用有界的 Nelder-Mead 细化。"""
    m = np.isfinite(v_w)
    valid = m[:, 1:] & m[:, :-1]
    tgt = v_w[:, 1:][valid]

    def f(th):
        p, _ = blup_predict(v_w, D, mu, float(np.exp(th[0])), float(np.exp(th[1])))
        return float(np.mean((p[valid] - tgt) ** 2))

    la = np.log(np.geomspace(ALPHA_RANGE[0], 1.0, 9))
    le = np.log(np.geomspace(1e-3, 1.0, 5))
    best = min(((f((x, y)), x, y) for x in la for y in le))
    bounds = [tuple(np.log(ALPHA_RANGE)), tuple(np.log(ETA_RANGE))]
    res = minimize(f, x0=np.array(best[1:]), method="Nelder-Mead", bounds=bounds,
                   options={"xatol": 1e-3, "fatol": 1e-12, "maxiter": 300})
    th = res.x if res.fun <= best[0] else np.array(best[1:])
    return float(np.exp(th[0])), float(np.exp(th[1]))


def _z(W: dict, pairs: dict) -> np.ndarray:
    """各位置的 z = Σ_h (w_h / Σw) x_hB，返回 (N, 3, 50)。权重和不为正时（极少）用紧挨着的前一种。"""
    out = []
    for p in range(1, L):
        w = W[p]
        s = w.sum(axis=1, keepdims=True)
        good = s[:, 0] > 1e-8
        wn = np.zeros_like(w)
        wn[good] = w[good] / s[good]
        wn[~good, p - 1] = 1.0
        out.append(sum(wn[:, [h]] * pairs[(h, p)] for h in range(p)))
    return np.stack(out, axis=1)


def hist_baselines(Rs: np.ndarray, rows_t, D_t, s_t, pairs_t, gt, rows_v, D_v, s_v, pairs_v,
                   seed: int = 0, chunk: int = 8192) -> tuple[dict, np.ndarray]:
    """两种多近亲对照的误差，按（窗口，位置）展开成 (N_v·3, C)，排法同 l1_baselines 的验证对。
    Rs (物种, C) 为相对值；s_t、s_v (N, 3, 2) 为各位置 A→B 的两列距离；gt (N_t·3,) 为训练预测目标的科。"""
    C = Rs.shape[1]
    n_v = len(rows_v) * NPOS
    E = {k: np.full((n_v, C), np.nan) for k in NEW_KEYS}
    params = np.full((C, len(HPARAMS)), np.nan)
    tr_rows = np.unique(rows_t)
    st, sv = s_t.reshape(-1, 2), s_v.reshape(-1, 2)
    for j in range(C):
        v = Rs[:, j]
        ok = tr_rows[np.isfinite(v[tr_rows])]
        if len(ok) < 10:
            continue
        mu = float(np.mean(v[ok]))                     # 与 wholebody._linear 的 μ 相同
        vt, vv = v[rows_t], v[rows_v]
        alpha, eta = blup_fit(vt, D_t, mu)
        params[j] = (alpha, eta)
        base_t, W_t = blup_predict(vt, D_t, mu, alpha, eta)
        base_v, W_v = blup_predict(vv, D_v, mu, alpha, eta)
        yb_v = vv[:, 1:].reshape(-1)
        E["m_no"][:, j] = base_v.reshape(-1) - yb_v
        mt = np.isfinite(vt)
        res = np.where(mt[:, 1:] & mt[:, :-1], vt[:, 1:] - base_t, np.nan).reshape(-1)
        ok_r = np.flatnonzero(np.isfinite(res))
        if len(ok_r) < 50:
            continue
        if len(ok_r) > MAX_ROWS:
            ok_r = np.random.default_rng(seed).choice(ok_r, MAX_ROWS, replace=False)
        zt = _z(W_t, pairs_t).reshape(-1, pairs_t[(0, 1)].shape[1])
        Wr, c, _ = ridge_family_cv(features(zt[ok_r], st[ok_r]), res[ok_r][:, None], gt[ok_r])
        zv = _z(W_v, pairs_v).reshape(-1, pairs_v[(0, 1)].shape[1])
        corr = np.concatenate([features(zv[i:i + chunk], sv[i:i + chunk]) @ Wr + c for i in range(0, n_v, chunk)])[:, 0]
        E["m_full"][:, j] = base_v.reshape(-1) + corr - yb_v
    return E, params


def _sums(fam: np.ndarray, typ: np.ndarray, pos: np.ndarray, errs: dict, F: int) -> dict:
    """同一批预测（全部读数、简单方法与对照都有限）上的平方误差和 (F, 3 × NPOS, C)；另返回个数 n。"""
    ok = np.logical_and.reduce([np.isfinite(errs[k]) for k in KEYS])
    tp = np.where((typ >= 0) & (pos >= 1) & (pos <= NPOS), typ * NPOS + (pos - 1), -1)
    out = {}
    n = None
    for k in KEYS:
        out[k], n = agg(fam, tp, np.where(ok, errs[k] ** 2, np.nan), F, 3 * NPOS)
    out["n"] = n
    return out


# ---------------------------------------------------------------------------
def eval_l1_hist(ds_tr, ds_va, model, table: pd.DataFrame, fam_codes: np.ndarray, F: int, n_orders: int,
                 seed: int) -> dict:
    """第一层：与 ou.eval_l1_ou 完全相同的窗口、读数与对照，另加两种多近亲对照。"""
    from ..data.units import habit_codes
    from .recovery import Level1, apply_probe, fit_probe, sample_windows
    L1 = Level1(model)
    H = int(L1.jepa.predictor.predictor.pos_embedding.shape[1]) if hasattr(L1.jepa.predictor, "predictor") else 3
    if H != NPOS:
        raise ValueError(f"历史长度为 {H}，本评估按 {NPOS} 个位置汇总")
    cols = list(ds_tr.state_cols)
    S = ds_tr.S.astype(float)
    jm = cols.index("m_mass")
    C = len(cols)
    rows_tr = np.flatnonzero(ds_tr.rows_mask)
    b = slopes_for(S, cols, rows_tr)
    probes = fit_probe(L1.encode(ds_tr.pix), S, rows_tr)
    rows_v, dist_v, D_v = windows_full(ds_va, n_orders, H + 1, seed)
    rows_t, dist_t, D_t = windows_full(ds_tr, n_orders, H + 1, seed)
    for ds, r, dd in ((ds_va, rows_v, dist_v), (ds_tr, rows_t, dist_t)):
        W0 = sample_windows(ds, n_orders, H + 1, seed)
        if not (np.array_equal(W0.rows, r) and np.array_equal(W0.dist, dd)):
            raise ValueError("windows_full 取到的窗口与 recovery.sample_windows 不一致")
    act_eco = np.stack([ds_va.build_action(r, d, eco=True) for r, d in zip(rows_v, dist_v)])
    act_no = np.stack([ds_va.build_action(r, d, eco=False) for r, d in zip(rows_v, dist_v)])
    emb = L1.encode(ds_va.pix[rows_v])
    pe = apply_probe(probes, L1.predict(emb[:, :H], act_eco[:, :H]))
    pn = apply_probe(probes, L1.predict(emb[:, :H], act_no[:, :H]))
    pz = apply_probe(probes, emb)
    prv, tgt = rows_v[:, :H], rows_v[:, 1:H + 1]
    yA, yB, pA, pB = S[prv], S[tgt], pz[:, :H], pz[:, 1:]

    def R(E):
        return rel(E, b, jm).reshape(-1, C)

    errs = {"copy": R(yA - yB), "a_eco": R(pe - yB), "a_no": R(pn - yB),
            "d_eco": R(yA + pe - pA - yB), "d_no": R(yA + pn - pA - yB),
            "floor": R(pB - yB), "zcopy": R(pA - yB)}

    loco = habit_codes(table)
    a_t, b_t, d_t = rows_t[:, :H].reshape(-1), rows_t[:, 1:H + 1].reshape(-1), dist_t[:, :H].reshape(-1)
    a, bb, d = prv.reshape(-1), tgt.reshape(-1), dist_v[:, :H].reshape(-1)
    A_dim = act_eco.shape[-1]
    act_t = np.stack([ds_tr.build_action(r, dd, eco=True) for r, dd in zip(rows_t, dist_t)])
    act_t2, act_v2 = act_t[:, :H].reshape(-1, A_dim), act_eco[:, :H].reshape(-1, A_dim)
    Rs = rel(S, b, jm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        E, params = l1_baselines(Rs, loco, a_t, b_t, d_t, a, bb, d, act_t2[:, :-2], act_v2[:, :-2],
                                 act_t2[:, -2:], act_v2[:, -2:], ds_tr.pix[a_t], ds_va.pix[a], fam_codes[b_t])
        E2, hparams = hist_baselines(Rs, rows_t, D_t, act_t[:, :H, -2:], habit_pairs(ds_tr, rows_t), fam_codes[b_t],
                                     rows_v, D_v, act_eco[:, :H, -2:], habit_pairs(ds_va, rows_v))
    errs.update(E)
    errs.update(E2)
    typ = step_types(loco[prv].reshape(-1), loco[tgt].reshape(-1))
    fam = fam_codes[tgt].reshape(-1)
    pos = np.tile(np.arange(1, H + 1), len(prv))
    out = _sums(fam, typ, pos, errs, F)
    out["params"] = params
    out["hparams"] = hparams
    return out


def run_hist_task(task: dict) -> dict:
    """task：root, table_rel, tree_file, fold, ckpt, seed, key, set。出错时返回 {"error": ...}，不中断整批运行。"""
    try:
        import json

        import torch
        torch.set_num_threads(1)
        from .recovery import load_world_model
        from .runs import ckpt_action_mode
        root = Path(task["root"])
        table = pd.read_csv(root / task["table_rel"])
        meta = json.loads((root / task["table_rel"]).with_suffix(".columns.json").read_text(encoding="utf-8"))
        cols = list(meta["morph"]) + list(meta["skeleton"])
        ckpt = Path(task["ckpt"])
        mode = ckpt_action_mode(ckpt)
        if mode != "change":
            raise ValueError(f"动作方式为 {mode}，只支持 change")
        model = load_world_model(ckpt, root / "hjepa")
        units = getattr(model, "evowm_units", None)
        ds_tr, ds_va = datasets(root, task["table_rel"], task["tree_file"], task["fold"], mode, units, cols)
        fam_codes, fam_names = pd.factorize(table["family"])
        out = {"key": task["key"], "fold": task["fold"], "set": task["set"], "cols": cols,
               "families": [str(x) for x in fam_names]}
        out["L1"] = eval_l1_hist(ds_tr, ds_va, model, table, fam_codes, len(fam_names), task.get("n_orders", 8),
                                 task["seed"])
        return out
    except Exception as e:  # noqa: BLE001
        import traceback
        return {"key": task.get("key"), "fold": task.get("fold"), "set": task.get("set"),
                "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]}
