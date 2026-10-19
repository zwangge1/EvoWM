"""OU 式对照（探索性；设计 docs/design/OU式对照_2026-10-19.md）。

与 evowm.eval.readout 完全相同的窗口、探针、读数、简单方法与同一批预测（同样的种子），另加三种 OU 式对照：
- o_ch（OU 式·只用类别）：ŷ_B = μ_c(A) + exp(−α′d)(y_A − μ_c(A)) + κ(1 − exp(−βd))(μ_c(B) − μ_c(A))。
  α′ 在全部训练对上、κ ∈ [0, 2] 与 β ∈ [0.001, 10]（每百万年）在类别不同的训练对上用最小二乘拟合；
- o_full（OU 式·全部 47 项）：不告诉习性的预测 + 岭回归（残差 ~ [u, s1·u, s2·u, s1, s2]），u 为动作的习性部分（50 列），
  s1、s2 为动作里的两列距离；
- o_same（OU 式·同样输入）：同上，u 再加 A 的编码器输入（23 项标准化实测与 23 个缺测标记；单位一级为成员平均）。
岭回归的行与 wholebody._ridge_correction 相同（有限的训练对，多于 6 万时同样抽 6 万），但惩罚改用按训练科分组的 5 折交叉验证选
（候选 1e-2 到 1e6）：同一支系里的训练对彼此不独立，留一会把惩罚选得太小（云端测试里空白模拟上明显过拟合）。
有限行完全相同的性状一起拟合，每个性状各自选惩罚；双精度。
逐个（模型，折）只返回按（科，类型 × 位置，性状）汇总的平方误差和：类型 0 不变、1 同环境、2 换环境；位置 1 到 3 为窗口里
第几个预测（世界模型看得到前面几个）；下标 = 类型 × 3 + 位置 − 1。对位置求和即得读数诊断的（科，类型，性状）。
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from .readout import KEYS as READOUT_KEYS
from .wholebody import _linear, _linear_units_full, agg, datasets, rel, slopes_for, step_types

OU_KEYS = ("o_ch", "o_full", "o_same")
KEYS = READOUT_KEYS + OU_KEYS
NPOS = 3                                   # 位置 1 到 3（各层的历史长度都是 3）
KAPPA_MAX = 2.0
BETA_RANGE = (1e-3, 10.0)                  # 每百万年
RIDGE_ALPHAS = np.logspace(-2, 6, 17)      # 惩罚候选（按科分组交叉验证）
CV_FOLDS = 5
MAX_ROWS = 60000
PARAMS = ("alpha_c", "kappa", "beta")      # OU 式·只用类别的拟合参数


# ---------------------------------------------------------------------------
def _alpha(m_a, ya, yb, d, nan: bool = False) -> float:
    """ŷ_B = m_A + exp(−αd)(y_A − m_A) 的最小二乘 α，搜索范围与方法同 wholebody._linear。
    nan=False 时调用方只传有限的对（同 _linear）；nan=True 时 NaN 的对不计（同 units.linear_units）。"""
    mean = np.nanmean if nan else np.mean

    def mse(la):
        return float(mean((m_a + np.exp(-np.exp(la) * d) * (ya - m_a) - yb) ** 2))
    return float(np.exp(minimize_scalar(mse, bounds=(np.log(1e-4), np.log(2.0)), method="bounded").x))


def shift_fit(r: np.ndarray, dl: np.ndarray, d: np.ndarray) -> tuple[float, float]:
    """残差 r ≈ κ(1 − exp(−βd))·dl 的最小二乘：给定 β 时 κ ∈ [0, KAPPA_MAX] 有闭式解；β 先在对数网格（41 点）上找，
    再在相邻两个格点之间细化。只用 dl ≠ 0 且都有限的对；不足 2 对时 κ = 0。返回 (κ, β)。"""
    ok = np.isfinite(r) & np.isfinite(dl) & np.isfinite(d) & (dl != 0)
    if ok.sum() < 2:
        return 0.0, BETA_RANGE[1]
    r, dl, d = r[ok], dl[ok], d[ok]

    def kappa_of(u):
        q = u * dl
        return float(np.clip(np.sum(r * q) / max(float(np.sum(q * q)), 1e-12), 0.0, KAPPA_MAX))

    def sse(lb):
        u = 1.0 - np.exp(-np.exp(lb) * d)
        return float(np.sum((r - kappa_of(u) * u * dl) ** 2))

    grid = np.linspace(np.log(BETA_RANGE[0]), np.log(BETA_RANGE[1]), 41)
    vals = np.array([sse(g) for g in grid])
    i = int(np.argmin(vals))
    lb = float(grid[i])
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    if hi > lo:
        res = minimize_scalar(sse, bounds=(lo, hi), method="bounded")
        if res.fun < vals[i]:
            lb = float(res.x)
    return kappa_of(1.0 - np.exp(-np.exp(lb) * d)), float(np.exp(lb))


def features(u: np.ndarray, s: np.ndarray) -> np.ndarray:
    """[u, s1·u, s2·u, s1, s2]；s 为动作里的两列距离（ln(1+d)/ln(1+220)、d/220）。用双精度（动作本身是单精度）。"""
    u, s = np.asarray(u, dtype=np.float64), np.asarray(s, dtype=np.float64)
    return np.concatenate([u, u * s[:, :1], u * s[:, 1:2], s], axis=1)


def ridge_family_cv(F: np.ndarray, Y: np.ndarray, groups: np.ndarray, alphas: np.ndarray = RIDGE_ALPHAS,
                    k: int = CV_FOLDS, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """带截距的岭回归 Y ~ F（多列 Y），惩罚按组（科）分 k 折交叉验证、逐列选：各组随机分到 k 折（种子 seed），
    每折用其余各折的格拉姆矩阵做特征分解，一次算完全部候选。选好以后在全部行上重拟合。
    返回 (系数 (p, m), 截距 (m,), 选中的惩罚 (m,))。组少于 2 个时用最大的惩罚。"""
    n = len(F)
    ug, inv = np.unique(groups, return_inverse=True)
    k = min(k, len(ug))
    sx, sy = F.sum(axis=0), Y.sum(axis=0)
    G, B = F.T @ F, F.T @ Y
    if k >= 2:
        perm = np.random.default_rng(seed).permutation(len(ug))
        fold_of = np.empty(len(ug), dtype=int)
        fold_of[perm] = np.arange(len(ug)) % k
        fold = fold_of[inv]
        err = np.zeros((len(alphas), Y.shape[1]))
        for f in range(k):
            te = fold == f
            n_tr = n - int(te.sum())
            if not te.any() or n_tr < 2:
                continue
            Ft, Yt = F[te], Y[te]
            mx, my = (sx - Ft.sum(axis=0)) / n_tr, (sy - Yt.sum(axis=0)) / n_tr
            e, V = np.linalg.eigh(G - Ft.T @ Ft - n_tr * np.outer(mx, mx))
            VB = V.T @ (B - Ft.T @ Yt - n_tr * np.outer(mx, my))
            Z = (Ft - mx) @ V
            for i, a in enumerate(alphas):
                err[i] += (((Z @ (VB / (np.maximum(e, 0.0) + a)[:, None])) + my - Yt) ** 2).sum(axis=0)
        best = alphas[np.argmin(err, axis=0)]
    else:
        best = np.full(Y.shape[1], alphas[-1])
    mx, my = sx / n, sy / n
    e, V = np.linalg.eigh(G - n * np.outer(mx, mx))
    VB = V.T @ (B - n * np.outer(mx, my))
    W = np.stack([V @ (VB[:, j] / (np.maximum(e, 0.0) + best[j])) for j in range(Y.shape[1])], axis=1)
    return W, my - mx @ W, best


def ridge_by_groups(Rt: np.ndarray, ut: list, st: np.ndarray, uv: list, sv: np.ndarray, gt: np.ndarray,
                    seed: int = 0, chunk: int = 8192) -> np.ndarray:
    """岭回归修正：训练对上 残差 ~ features(u, s)，预测验证对。Rt (n_t, C) 为各性状的残差（NaN 为缺）；
    ut、uv 为 u 的几块列（训练对、验证对），只在用到的行上拼起来，省内存；gt 为训练对的科（分组交叉验证用）。
    行的取法与抽样（种子 0，至多 6 万对）同 wholebody._ridge_correction；有限行完全相同的性状一起拟合（ridge_family_cv）。
    有限的训练对少于 50 个的性状返回 NaN。返回 (n_v, C)。"""
    n_v, C = len(sv), Rt.shape[1]
    out = np.full((n_v, C), np.nan)
    row_ok = np.isfinite(st).all(axis=1)
    for blk in ut:
        row_ok &= np.isfinite(blk).all(axis=1)
    fin = np.isfinite(Rt) & row_ok[:, None]

    def u_rows(blocks, rows):
        return np.concatenate([np.asarray(b[rows], dtype=np.float64) for b in blocks], axis=1)

    groups: dict = {}
    for j in range(C):
        groups.setdefault(np.ascontiguousarray(fin[:, j]).tobytes(), []).append(j)
    for key, js in groups.items():
        ok = np.flatnonzero(np.frombuffer(key, dtype=bool))
        if len(ok) < 50:
            continue
        if len(ok) > MAX_ROWS:
            ok = np.random.default_rng(seed).choice(ok, MAX_ROWS, replace=False)
        W, c, _ = ridge_family_cv(features(u_rows(ut, ok), st[ok]), Rt[np.ix_(ok, js)], gt[ok])
        parts = [features(u_rows(uv, slice(i, i + chunk)), sv[i:i + chunk]) @ W + c for i in range(0, n_v, chunk)]
        if parts:
            out[:, js] = np.concatenate(parts)
    return out


def _sums(fam: np.ndarray, typ: np.ndarray, pos: np.ndarray, errs: dict, F: int) -> dict:
    """同一批预测（全部读数、简单方法与对照都有限）上的平方误差和，形状 (F, 3 × NPOS, C)；另返回个数 n。"""
    ok = np.logical_and.reduce([np.isfinite(errs[k]) for k in KEYS])
    tp = np.where((typ >= 0) & (pos >= 1) & (pos <= NPOS), typ * NPOS + (pos - 1), -1)
    out = {}
    n = None
    for k in KEYS:
        out[k], n = agg(fam, tp, np.where(ok, errs[k] ** 2, np.nan), F, 3 * NPOS)
    out["n"] = n
    return out


def _pmat(Y: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """标准化实测（缺测记 0）与缺测标记，与编码器输入的排法相同。"""
    return np.concatenate([np.nan_to_num((Y - mean) / sd), np.isfinite(Y)], axis=1).astype(float)


# ---------------------------------------------------------------------------
def l1_baselines(Rs, cat, a_t, b_t, d_t, a, bb, d, Xt, Xv, st, sv, Pt, Pv, gt) -> tuple[dict, np.ndarray]:
    """第一层，逐项尺寸：已有简单方法（wholebody._linear，不变）与三种 OU 式对照的误差 (n, C)。
    Rs (物种, C) 为相对值；a_t, b_t, d_t 为训练窗口的相邻对，a, bb, d 为要预测的对；Xt, Xv 为动作的习性部分，st, sv 为
    两列距离，Pt, Pv 为前一种的编码器输入，gt 为训练对后一种的科。另返回 OU 式·只用类别的参数 (C, 3)：α′、κ、β。"""
    n, C = len(a), Rs.shape[1]
    E = {k: np.full((n, C), np.nan) for k in ("l_no", "l_ch", "l_full", "o_ch")}
    base_v = np.full((n, C), np.nan)
    res_t = np.full((len(a_t), C), np.nan)
    params = np.full((C, len(PARAMS)), np.nan)
    for j in range(C):
        v = Rs[:, j]
        E["l_no"][:, j], E["l_ch"][:, j], E["l_full"][:, j] = _linear(v, cat, a_t, b_t, d_t, a, bb, d, Xt, Xv)
        tr_rows = np.unique(np.concatenate([a_t, b_t]))
        ok = tr_rows[np.isfinite(v[tr_rows])]
        if len(ok) < 10:
            continue
        mu = float(np.mean(v[ok]))                                   # 以下 μ、μ_c、α 与 _linear 完全相同
        cats = np.unique(cat[ok][cat[ok] >= 0])
        mu_c = {int(c): float(np.mean(v[ok][cat[ok] == c])) for c in cats}
        m_c = np.array([mu_c.get(int(c), mu) for c in cat])
        g = np.isfinite(v[a_t]) & np.isfinite(v[b_t])
        at, bt, dt = a_t[g], b_t[g], d_t[g]
        alpha = _alpha(mu, v[at], v[bt], dt)
        res_t[:, j] = v[b_t] - (mu + np.exp(-alpha * d_t) * (v[a_t] - mu))
        base_v[:, j] = mu + np.exp(-alpha * d) * (v[a] - mu)
        ac = _alpha(m_c[at], v[at], v[bt], dt)                        # OU 式·只用类别
        rc = v[bt] - (m_c[at] + np.exp(-ac * dt) * (v[at] - m_c[at]))
        kappa, beta = shift_fit(rc, m_c[bt] - m_c[at], dt)
        pred = m_c[a] + np.exp(-ac * d) * (v[a] - m_c[a]) + kappa * (1.0 - np.exp(-beta * d)) * (m_c[bb] - m_c[a])
        E["o_ch"][:, j] = pred - v[bb]
        params[j] = (ac, kappa, beta)
    yb = Rs[bb]
    E["o_full"] = base_v + ridge_by_groups(res_t, [Xt], st, [Xv], sv, gt) - yb
    E["o_same"] = base_v + ridge_by_groups(res_t, [Xt, Pt], st, [Xv, Pv], sv, gt) - yb
    return E, params


def eval_l1_ou(ds_tr, ds_va, model, table: pd.DataFrame, fam_codes: np.ndarray, F: int, n_orders: int,
               seed: int) -> dict:
    """第一层（物种）：与 readout.eval_l1_readout 相同的窗口与读数，另加三种 OU 式对照，按位置拆分。"""
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
    W = sample_windows(ds_va, n_orders, H + 1, seed)
    act_eco = np.stack([ds_va.build_action(r, d, eco=True) for r, d in zip(W.rows, W.dist)])
    act_no = np.stack([ds_va.build_action(r, d, eco=False) for r, d in zip(W.rows, W.dist)])
    emb = L1.encode(ds_va.pix[W.rows])                                  # (N, H+1, D)
    pe = apply_probe(probes, L1.predict(emb[:, :H], act_eco[:, :H]))    # (N, H, C)
    pn = apply_probe(probes, L1.predict(emb[:, :H], act_no[:, :H]))
    pz = apply_probe(probes, emb)
    prv, tgt = W.rows[:, :H], W.rows[:, 1:H + 1]
    yA, yB, pA, pB = S[prv], S[tgt], pz[:, :H], pz[:, 1:]

    def R(E):
        return rel(E, b, jm).reshape(-1, C)

    errs = {"copy": R(yA - yB), "a_eco": R(pe - yB), "a_no": R(pn - yB),
            "d_eco": R(yA + pe - pA - yB), "d_no": R(yA + pn - pA - yB),
            "floor": R(pB - yB), "zcopy": R(pA - yB)}

    loco = habit_codes(table)
    Wt = sample_windows(ds_tr, n_orders, H + 1, seed)
    a_t, b_t, d_t = Wt.rows[:, :H].reshape(-1), Wt.rows[:, 1:H + 1].reshape(-1), Wt.dist[:, :H].reshape(-1)
    a, bb, d = prv.reshape(-1), tgt.reshape(-1), W.dist[:, :H].reshape(-1)
    A_dim = act_eco.shape[-1]
    act_t = np.stack([ds_tr.build_action(r, dd, eco=True) for r, dd in zip(Wt.rows, Wt.dist)])[:, :H].reshape(-1, A_dim)
    act_v = act_eco[:, :H].reshape(-1, A_dim)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        E, params = l1_baselines(rel(S, b, jm), loco, a_t, b_t, d_t, a, bb, d, act_t[:, :-2], act_v[:, :-2],
                                 act_t[:, -2:], act_v[:, -2:], ds_tr.pix[a_t], ds_va.pix[a], fam_codes[b_t])
    errs.update(E)
    typ = step_types(loco[prv].reshape(-1), loco[tgt].reshape(-1))
    fam = fam_codes[tgt].reshape(-1)
    pos = np.tile(np.arange(1, H + 1), len(prv))
    out = _sums(fam, typ, pos, errs, F)
    out["params"] = params
    return out


def eval_units_ou(ds_tr, ds_va, model, table: pd.DataFrame, fam_codes: np.ndarray, F: int, n_orders: int,
                  seed: int, b: np.ndarray) -> dict:
    """第二、三层（单位）：与 readout.eval_units_readout 相同的窗口、探针与读数，另加三种 OU 式对照，按位置拆分。"""
    from .units import LABELS, Level, _cat, apply_probes, e1_level, fill_units, fit_probes, linear_units, make_orders
    cols = list(ds_tr.state_cols)
    jm = cols.index("m_mass")
    V = ds_tr.S.astype(float)
    C = len(cols)
    z1_all = Level(model, 1).encode(ds_tr.pix)
    tr = make_orders(ds_tr, n_orders, seed)
    va = make_orders(ds_va, n_orders, seed + 1)
    fill_units(model, tr + va, z1_all, V, fam_codes)
    out = {}
    for name in model.evowm_units["levels"]:
        Ztr, _ = _cat(tr, lambda o: o.z[name])
        Ytr, _ = _cat(tr, lambda o: o.y[name])
        probes = fit_probes(Ztr, Ytr)
        Yva, off = _cat(va, lambda o: o.y[name])
        Zva, _ = _cat(va, lambda o: o.z[name])
        PZ = apply_probes(probes, Zva)
        lab, _ = _cat(va, lambda o: o.levels[name].label)
        famv, _ = _cat(va, lambda o: o.fam[name])
        e1 = e1_level(model, va, name, probes)
        oi, prev, target = e1["order"], e1["prev"], e1["target"]
        gp, gt = off[oi] + prev, off[oi] + target
        yA, yB, pA, pB = Yva[gp], Yva[gt], PZ[gp], PZ[gt]
        pe, pn = e1["pred_eco"], e1["pred_no"]

        def R(E):
            return rel(E, b, jm)

        errs = {"copy": R(yA - yB), "a_eco": R(pe - yB), "a_no": R(pn - yB),
                "d_eco": R(yA + pe - pA - yB), "d_no": R(yA + pn - pA - yB),
                "floor": R(pB - yB), "zcopy": R(pA - yB)}

        # 训练科叶序上的相邻单位对（与 units.linear_units 相同的排法）
        n_lab = len(LABELS[name])
        lab_tr = np.concatenate([o.levels[name].label for o in tr])
        la = np.concatenate([o.levels[name].label[:-1] for o in tr])
        lb = np.concatenate([o.levels[name].label[1:] for o in tr])
        d_t = np.concatenate([o.levels[name].dist for o in tr])
        act_t = np.concatenate([o.levels[name].action[:-1] for o in tr])
        fam_t = np.concatenate([o.fam[name][1:] for o in tr])          # 训练对后一个单位的主要科
        dv = np.concatenate([np.append(o.levels[name].dist, np.nan) for o in va])
        act_v = np.concatenate([o.levels[name].action for o in va])[gp]
        y_mean = np.nanmean(Ytr, axis=0)
        y_sd = np.nanstd(Ytr, axis=0)
        y_sd[~np.isfinite(y_sd) | (y_sd == 0)] = 1.0
        y_mean = np.nan_to_num(y_mean)
        Pt = _pmat(np.concatenate([o.y[name][:-1] for o in tr]), y_mean, y_sd)
        Pv = _pmat(Yva[gp], y_mean, y_sd)

        E = {k: np.full((len(gp), C), np.nan) for k in ("l_no", "l_ch", "l_full", "o_ch")}
        base_v = np.full((len(gp), C), np.nan)
        res_t = np.full((len(d_t), C), np.nan)
        params = np.full((C, len(PARAMS)), np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for j in range(C):
                r_tr = [rel(o.y[name], b, jm)[:, j] for o in tr]
                r_va = [rel(o.y[name], b, jm)[:, j] for o in va]
                rt_all = np.concatenate(r_tr)
                if np.isfinite(rt_all).sum() < 10:
                    continue
                lu = linear_units(tr, va, name, r_tr, r_va, oi, prev, target)
                E["l_no"][:, j], E["l_ch"][:, j] = lu["err_no"], lu["err_change"]
                _, E["l_full"][:, j] = _linear_units_full(tr, va, name, r_tr, r_va, oi, prev, target)
                mu = float(np.nanmean(rt_all))                        # μ、μ_c、α 与 linear_units 相同
                mu_c = np.array([np.nanmean(rt_all[lab_tr == c]) if np.any(lab_tr == c) else mu for c in range(n_lab)])
                at = np.concatenate([r[:-1] for r in r_tr])
                bt = np.concatenate([r[1:] for r in r_tr])
                alpha = lu["alpha"]
                res_t[:, j] = bt - (mu + np.exp(-alpha * d_t) * (at - mu))
                rv = np.concatenate(r_va)
                base_v[:, j] = mu + np.exp(-alpha * dv[gp]) * (rv[gp] - mu)
                ac = _alpha(mu_c[la], at, bt, d_t, nan=True)          # OU 式·只用类别
                rc = bt - (mu_c[la] + np.exp(-ac * d_t) * (at - mu_c[la]))
                kappa, beta = shift_fit(rc, mu_c[lb] - mu_c[la], d_t)
                ca, cb, dd = mu_c[lab[gp]], mu_c[lab[gt]], dv[gp]
                pred = ca + np.exp(-ac * dd) * (rv[gp] - ca) + kappa * (1.0 - np.exp(-beta * dd)) * (cb - ca)
                E["o_ch"][:, j] = pred - rv[gt]
                params[j] = (ac, kappa, beta)
            yb = R(yB)
            E["o_full"] = base_v + ridge_by_groups(res_t, [act_t[:, :-2]], act_t[:, -2:], [act_v[:, :-2]],
                                                   act_v[:, -2:], fam_t) - yb
            E["o_same"] = base_v + ridge_by_groups(res_t, [act_t[:, :-2], Pt], act_t[:, -2:], [act_v[:, :-2], Pv],
                                                   act_v[:, -2:], fam_t) - yb
        errs.update(E)
        typ = step_types(lab[gp], lab[gt]) if name == "locomotor" else np.where(lab[gp] == lab[gt], 0, 2)
        out[name] = _sums(famv[gt], typ, e1["position"], errs, F)
        out[name]["params"] = params
    return out


def run_ou_task(task: dict) -> dict:
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
        F = len(fam_names)
        n_orders = task.get("n_orders", 8)
        out = {"key": task["key"], "fold": task["fold"], "set": task["set"], "cols": cols,
               "families": [str(x) for x in fam_names]}
        out["L1"] = eval_l1_ou(ds_tr, ds_va, model, table, fam_codes, F, n_orders, task["seed"])
        if units:
            b = slopes_for(ds_tr.S.astype(float), cols, np.flatnonzero(ds_tr.rows_mask))
            lv = eval_units_ou(ds_tr, ds_va, model, table, fam_codes, F, n_orders, task["seed"], b)
            if "locomotor" in lv:
                out["L2"] = lv["locomotor"]
            if "realm" in lv:
                out["L3"] = lv["realm"]
        return out
    except Exception as e:  # noqa: BLE001
        import traceback
        return {"key": task.get("key"), "fold": task.get("fold"), "set": task.get("set"),
                "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]}
