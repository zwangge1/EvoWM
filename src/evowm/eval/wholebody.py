"""全身形态在各层的习性读数（探索性；设计 docs/design/探索_分层模块_2026-10-18.md）。

逐个（模型，折）计算，只返回按科汇总的和（平方误差和、个数、反事实对比之和），合并后按科自助重抽样：
- 第 1 层（物种）：留出科叶序上的窗口（与主分析 E1 相同，历史 1 到 3）。每个相邻对按运动方式分为三类：
  不变、栖境内改变（如栖树到地栖）、跨栖境改变（如陆到空、陆到水）。23 项形态的有、无习性误差。
  反事实：窗口第 1 个预测，把下一种的习性换成各运动方式的典型值，读出与栖树的差。
- 第 2 层（运动方式单位）、第 3 层（栖境单位）：与主分析相同的单位读数，换成 23 项形态；
  第 2 层的步分为栖境内、跨栖境两类。反事实与主分析 H4 相同（栖树、陆地为基准）。
- 线性基线：每项形态各自的"只用变化"线性基线（第 1、2 层按运动方式，第 3 层按栖境），与世界模型同一批预测。
  另有"全部习性列"的线性基线：在无习性的预测上，再加一个岭回归（残差 ~ 动作的习性部分，即世界模型拿到的同一组
  47 列习性差与 3 个习性组标记），惩罚用留一交叉验证在训练对上选。结果放在 lin2（五种预测用同一批）。
长度类形态扣除体重通路（相对值 = 值 − b × 体重，b 为该折训练物种上的 OLS 斜率）；体重与 HWI 不扣。
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from ..data.units import LOCOMOTOR, REALM_CODE, REALMS

NO_REL = ("m_mass", "m_hwi")
TYPES = ["none", "within", "cross"]              # 不变、栖境内改变、跨栖境改变
D_MAX = 220.0


def datasets(root: Path, table_rel: str, tree_file: str, fold: int, action_mode: str, units, state_cols: list):
    """与 evowm.eval.runs.fold_datasets 相同，只是状态列换成全部形态。"""
    from ..data.transect import TreeTransectDataset
    from .runs import N_FOLDS
    kw = dict(species_table=table_rel, tree_file=tree_file, level1={"frameskip": 1, "num_steps": 4}, evowm_root=str(root),
              n_trees=1, holdout={"level": "family", "frac": 0, "folds": N_FOLDS, "fold": fold, "seed": 0}, n_orders=4,
              action_mode=action_mode, units=units, state_cols=state_cols)
    return TreeTransectDataset(name="evowm_train", **kw), TreeTransectDataset(name="evowm_val", **kw)


def slopes_for(S: np.ndarray, cols: list, rows_tr: np.ndarray) -> np.ndarray:
    jm = cols.index("m_mass")
    b = np.zeros(len(cols))
    for j, c in enumerate(cols):
        if c in NO_REL:
            continue
        ok = rows_tr[np.isfinite(S[rows_tr, j]) & np.isfinite(S[rows_tr, jm])]
        b[j] = float(np.polyfit(S[ok, jm], S[ok, j], 1)[0])
    return b


def rel(Y: np.ndarray, b: np.ndarray, jm: int) -> np.ndarray:
    return Y - b * Y[..., [jm]]


def agg(fam: np.ndarray, typ: np.ndarray, V: np.ndarray, F: int, T: int) -> tuple[np.ndarray, np.ndarray]:
    """按（科，类型，性状）求和；V 为 NaN 或类型 < 0 的不计。返回 (和, 个数)，形状 (F, T, C)。"""
    N, C = V.shape
    ok = np.isfinite(V) & (typ >= 0)[:, None]
    idx = (fam * T + np.maximum(typ, 0))[:, None] * C + np.arange(C)[None, :]
    s = np.bincount(idx[ok], V[ok], F * T * C).reshape(F, T, C)
    n = np.bincount(idx[ok], minlength=F * T * C).reshape(F, T, C).astype(float)
    return s, n


def step_types(la: np.ndarray, lb: np.ndarray) -> np.ndarray:
    """运动方式编号（-1 为缺失）→ 0 不变、1 栖境内改变、2 跨栖境改变、-1 不计。"""
    ra = np.where(la >= 0, REALM_CODE[np.maximum(la, 0)], -1)
    rb = np.where(lb >= 0, REALM_CODE[np.maximum(lb, 0)], -1)
    t = np.where(la == lb, 0, np.where(ra == rb, 1, 2))
    return np.where((la < 0) | (lb < 0), -1, t)


def _ridge_correction(Xt: np.ndarray, rt: np.ndarray, Xv: np.ndarray, max_rows: int = 60000, seed: int = 0) -> np.ndarray:
    """训练对上岭回归 残差 ~ 习性变化（惩罚用留一交叉验证选），返回验证对的修正量；训练对太少时返回 NaN。"""
    from sklearn.linear_model import RidgeCV
    ok = np.flatnonzero(np.isfinite(rt) & np.isfinite(Xt).all(axis=1))
    if len(ok) < 50:
        return np.full(len(Xv), np.nan)
    if len(ok) > max_rows:
        ok = np.random.default_rng(seed).choice(ok, max_rows, replace=False)
    m = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(Xt[ok], rt[ok])
    return m.predict(Xv)


def _linear(v: np.ndarray, cat: np.ndarray, a_t, b_t, d_t, a, b, d, Xt=None, Xv=None):
    """一项形态的"只用变化"线性基线（evowm.eval.runs.linear_baseline 的推广）：
    无习性 v̂_B = μ + exp(−αd)(v_A − μ)；只用变化再加 κ(μ_c(B) − μ_c(A))。μ_c 为训练物种各类的均值（类别 -1 用 μ）。
    a_t, b_t, d_t 为训练窗口的相邻对；a, b, d 为要预测的对。返回 (无习性误差, 只用变化误差)。"""
    tr_rows = np.unique(np.concatenate([a_t, b_t]))
    ok = tr_rows[np.isfinite(v[tr_rows])]
    if len(ok) < 10:
        nan = np.full(len(a), np.nan)
        return (nan, nan) if Xt is None else (nan, nan, nan)
    mu = float(np.mean(v[ok]))
    cats = np.unique(cat[ok][cat[ok] >= 0])
    mu_c = {int(c): float(np.mean(v[ok][cat[ok] == c])) for c in cats}
    m_c = np.array([mu_c.get(int(c), mu) for c in cat])
    g = np.isfinite(v[a_t]) & np.isfinite(v[b_t])
    at, bt, dt = a_t[g], b_t[g], d_t[g]

    def mse(la):
        return float(np.mean((mu + np.exp(-np.exp(la) * dt) * (v[at] - mu) - v[bt]) ** 2))

    alpha = float(np.exp(minimize_scalar(mse, bounds=(np.log(1e-4), np.log(2.0)), method="bounded").x))
    res = v[bt] - (mu + np.exp(-alpha * dt) * (v[at] - mu))
    dl = m_c[bt] - m_c[at]
    nz = dl != 0
    kappa = float(np.clip(np.sum(res[nz] * dl[nz]) / max(float(np.sum(dl[nz] ** 2)), 1e-12), 0.0, 1.0))
    base = mu + np.exp(-alpha * d) * (v[a] - mu)
    if Xt is None:
        return base - v[b], base + kappa * (m_c[b] - m_c[a]) - v[b]
    rt = v[b_t] - (mu + np.exp(-alpha * d_t) * (v[a_t] - mu))
    full = base + _ridge_correction(Xt, rt, Xv)
    return base - v[b], base + kappa * (m_c[b] - m_c[a]) - v[b], full - v[b]


def _lin2(fam, typ, ee, en, l_no, l_ch, l_f, F: int) -> dict:
    """世界模型（有、无习性）与三种线性预测（无习性、只用类别、全部习性列）在同一批预测上的平方误差和。"""
    ok = np.isfinite(ee) & np.isfinite(en) & np.isfinite(l_no) & np.isfinite(l_ch) & np.isfinite(l_f)
    out = {}
    for k, e in (("w_eco", ee), ("w_no", en), ("l_no", l_no), ("l_ch", l_ch), ("l_full", l_f)):
        out[k], n = agg(fam, typ, np.where(ok, e ** 2, np.nan), F, 3)
    out["n"] = n
    return out


def _linear_units_full(tr, va, name: str, r_tr: list, r_va: list, oi, prev, target) -> tuple[np.ndarray, np.ndarray]:
    """单位一级的"全部习性列"线性基线：无习性部分与 evowm.eval.units.linear_units 相同（同样的 μ、α 拟合），
    再加岭回归（残差 ~ 单位动作的习性部分）。返回 (无习性误差, 全部习性列误差)，与 e1 的预测顺序一致。"""
    rt_all = np.concatenate(r_tr)
    mu = float(np.nanmean(rt_all))
    a_t = np.concatenate([r[:-1] for r in r_tr])
    b_t = np.concatenate([r[1:] for r in r_tr])
    d_t = np.concatenate([o.levels[name].dist for o in tr])
    Xt = np.concatenate([o.levels[name].action[:-1, :-2] for o in tr])

    def f(log_alpha):
        return float(np.nanmean((mu + np.exp(-np.exp(log_alpha) * d_t) * (a_t - mu) - b_t) ** 2))
    alpha = float(np.exp(minimize_scalar(f, bounds=(np.log(1e-4), np.log(2.0)), method="bounded").x))
    res = b_t - (mu + np.exp(-alpha * d_t) * (a_t - mu))
    rv = np.concatenate(r_va)
    off = np.concatenate([[0], np.cumsum([len(r) for r in r_va])])
    dv = np.concatenate([np.append(o.levels[name].dist, np.nan) for o in va])
    Xv_all = np.concatenate([o.levels[name].action[:, :-2] for o in va])
    ga, gb = off[oi] + prev, off[oi] + target
    base = mu + np.exp(-alpha * dv[ga]) * (rv[ga] - mu)
    full = base + _ridge_correction(Xt, res, Xv_all[ga])
    return base - rv[gb], full - rv[gb]


# ---------------------------------------------------------------------------
def eval_l1(ds_tr, ds_va, model, table: pd.DataFrame, fam_codes: np.ndarray, F: int, n_orders: int, seed: int) -> dict:
    from ..data.units import habit_codes
    from .recovery import Level1, apply_probe, fit_probe, sample_windows
    L1 = Level1(model)
    H = int(L1.jepa.predictor.predictor.pos_embedding.shape[1]) if hasattr(L1.jepa.predictor, "predictor") else 3
    cols = list(ds_tr.state_cols)
    S = ds_tr.S.astype(float)
    jm = cols.index("m_mass")
    rows_tr = np.flatnonzero(ds_tr.rows_mask)
    b = slopes_for(S, cols, rows_tr)
    probes = fit_probe(L1.encode(ds_tr.pix), S, rows_tr)
    W = sample_windows(ds_va, n_orders, H + 1, seed)
    act_eco = np.stack([ds_va.build_action(r, d, eco=True) for r, d in zip(W.rows, W.dist)])
    act_no = np.stack([ds_va.build_action(r, d, eco=False) for r, d in zip(W.rows, W.dist)])
    emb = L1.encode(ds_va.pix[W.rows])
    pe = apply_probe(probes, L1.predict(emb[:, :H], act_eco[:, :H]))
    pn = apply_probe(probes, L1.predict(emb[:, :H], act_no[:, :H]))
    prv, tgt = W.rows[:, :H], W.rows[:, 1:H + 1]
    true = S[tgt]
    C = len(cols)
    ee = rel(pe - true, b, jm).reshape(-1, C)
    en = rel(pn - true, b, jm).reshape(-1, C)
    loco = habit_codes(table)
    typ = step_types(loco[prv].reshape(-1), loco[tgt].reshape(-1))
    fam = fam_codes[tgt].reshape(-1)
    s_eco, n = agg(fam, typ, ee ** 2, F, 3)
    s_no, _ = agg(fam, typ, en ** 2, F, 3)

    # 线性基线：训练科窗口上拟合，逐项形态
    Wt = sample_windows(ds_tr, n_orders, H + 1, seed)
    a_t, b_t, d_t = Wt.rows[:, :H].reshape(-1), Wt.rows[:, 1:H + 1].reshape(-1), Wt.dist[:, :H].reshape(-1)
    a, bb, d = prv.reshape(-1), tgt.reshape(-1), W.dist[:, :H].reshape(-1)
    R = rel(S, b, jm)
    A_dim = act_eco.shape[-1]
    Xt = np.stack([ds_tr.build_action(r, dd, eco=True) for r, dd in zip(Wt.rows, Wt.dist)])[:, :H].reshape(-1, A_dim)[:, :-2]
    Xv = act_eco[:, :H].reshape(-1, A_dim)[:, :-2]
    lin_no, lin_ch, lin_f = (np.full((len(a), C), np.nan) for _ in range(3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for j in range(C):
            lin_no[:, j], lin_ch[:, j], lin_f[:, j] = _linear(R[:, j], loco, a_t, b_t, d_t, a, bb, d, Xt, Xv)
    lin2 = _lin2(fam, typ, ee, en, lin_no, lin_ch, lin_f, F)
    ok_both = np.isfinite(ee) & np.isfinite(lin_no) & np.isfinite(lin_ch)
    sl_no, nl = agg(fam, typ, np.where(ok_both, lin_no ** 2, np.nan), F, 3)
    sl_ch, _ = agg(fam, typ, np.where(ok_both, lin_ch ** 2, np.nan), F, 3)
    sw_eco, _ = agg(fam, typ, np.where(ok_both, ee ** 2, np.nan), F, 3)
    sw_no, _ = agg(fam, typ, np.where(ok_both, en ** 2, np.nan), F, 3)

    # 反事实：窗口第 1 个预测，下一种的习性换成各运动方式的典型值（训练物种、各习性组都有数据）
    E_all, Eg_all = ds_tr.eco_rows(np.arange(len(table)))
    full = Eg_all.min(axis=1) > 0
    a0 = W.rows[:, 0]
    Ea, Ega = ds_va.eco_rows(a0)
    cm = Ega[:, ds_va.col_group]
    d0 = W.dist[:, 0]
    dist_part = np.stack([np.log1p(d0) / np.log1p(D_MAX), d0 / D_MAX], axis=1)
    preds = {}
    for k, c in enumerate(LOCOMOTOR):
        sel = rows_tr[(loco[rows_tr] == k) & full[rows_tr]]
        if len(sel) < 5:
            continue
        p = E_all[sel].mean(axis=0)
        act = np.concatenate([(p[None, :] - Ea) * cm, Ega, dist_part], axis=1).astype(np.float32)
        preds[c] = rel(apply_probe(probes, L1.predict(emb[:, :1], act[:, None, :]))[:, 0, :], b, jm)
    K = len(LOCOMOTOR) - 1
    s_e2, n_e2 = np.zeros((F, K, C)), np.zeros((F, K, C))
    fam0 = fam_codes[W.rows[:, 1]]
    if "insessorial" in preds:
        for k, c in enumerate(LOCOMOTOR[1:]):
            if c in preds:
                s, nn = agg(fam0, np.zeros(len(fam0), int), preds[c] - preds["insessorial"], F, 1)
                s_e2[:, k], n_e2[:, k] = s[:, 0], nn[:, 0]
    sd = np.nanstd(R[rows_tr], axis=0)
    return {"s_eco": s_eco, "s_no": s_no, "n": n, "lin": {"s_no": sl_no, "s_ch": sl_ch, "w_eco": sw_eco, "w_no": sw_no, "n": nl},
            "lin2": lin2, "e2": {"s": s_e2, "n": n_e2, "labels": LOCOMOTOR[1:]}, "slopes": b, "sd": sd}


def eval_units_wb(ds_tr, ds_va, model, table: pd.DataFrame, fam_codes: np.ndarray, F: int, n_orders: int, seed: int,
                  b: np.ndarray) -> dict:
    from .units import BASE, LABELS, Level, _cat, apply_probes, e1_level, e2_level, fill_units, fit_probes, linear_units, \
        make_orders, prototypes
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
        lab, _ = _cat(va, lambda o: o.levels[name].label)
        famv, _ = _cat(va, lambda o: o.fam[name])
        e1 = e1_level(model, va, name, probes)
        gp, gt = off[e1["order"]] + e1["prev"], off[e1["order"]] + e1["target"]
        ee = rel(e1["pred_eco"] - Yva[gt], b, jm)
        en = rel(e1["pred_no"] - Yva[gt], b, jm)
        if name == "locomotor":
            typ = step_types(lab[gp], lab[gt])
        else:
            typ = np.where(lab[gp] == lab[gt], 0, 2)
        fam = famv[gt]
        s_eco, n = agg(fam, typ, ee ** 2, F, 3)
        s_no, _ = agg(fam, typ, en ** 2, F, 3)
        lin_no, lin_ch, lin_f = (np.full(ee.shape, np.nan) for _ in range(3))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for j in range(C):
                r_tr = [rel(o.y[name], b, jm)[:, j] for o in tr]
                r_va = [rel(o.y[name], b, jm)[:, j] for o in va]
                if np.isfinite(np.concatenate(r_tr)).sum() < 10:
                    continue
                lu = linear_units(tr, va, name, r_tr, r_va, e1["order"], e1["prev"], e1["target"])
                lin_no[:, j], lin_ch[:, j] = lu["err_no"], lu["err_change"]
                _, lin_f[:, j] = _linear_units_full(tr, va, name, r_tr, r_va, e1["order"], e1["prev"], e1["target"])
        lin2 = _lin2(fam, typ, ee, en, lin_no, lin_ch, lin_f, F)
        ok_both = np.isfinite(ee) & np.isfinite(lin_no) & np.isfinite(lin_ch)
        sl_no, nl = agg(fam, typ, np.where(ok_both, lin_no ** 2, np.nan), F, 3)
        sl_ch, _ = agg(fam, typ, np.where(ok_both, lin_ch ** 2, np.nan), F, 3)
        sw_eco, _ = agg(fam, typ, np.where(ok_both, ee ** 2, np.nan), F, 3)
        sw_no, _ = agg(fam, typ, np.where(ok_both, en ** 2, np.nan), F, 3)

        protos = prototypes(tr, name)
        e2 = e2_level(model, va, name, probes, protos)
        g2 = off[e2["order"]] + e2["target"]
        base_c = LABELS[name].index(BASE[name])
        labels = [c for c in LABELS[name] if c != BASE[name]]
        s_e2, n_e2 = np.zeros((F, len(labels), C)), np.zeros((F, len(labels), C))
        if base_c in e2["pred"]:
            for k, c in enumerate(labels):
                ci = LABELS[name].index(c)
                if ci in e2["pred"]:
                    s, nn = agg(famv[g2], np.zeros(len(g2), int), rel(e2["pred"][ci] - e2["pred"][base_c], b, jm), F, 1)
                    s_e2[:, k], n_e2[:, k] = s[:, 0], nn[:, 0]
        out[name] = {"s_eco": s_eco, "s_no": s_no, "n": n,
                     "lin": {"s_no": sl_no, "s_ch": sl_ch, "w_eco": sw_eco, "w_no": sw_no, "n": nl}, "lin2": lin2,
                     "e2": {"s": s_e2, "n": n_e2, "labels": labels},
                     "sd": np.nanstd(rel(Ytr, b, jm), axis=0)}
    return out


def run_wholebody_task(task: dict) -> dict:
    """task：root, table_rel, tree_file, fold, ckpt, seed, key, set。出错时返回 {"error": ...}，不中断整夜的运行。"""
    try:
        import torch
        torch.set_num_threads(1)
        from .recovery import load_world_model
        from .runs import ckpt_action_mode
        import json
        root = Path(task["root"])
        table = pd.read_csv(root / task["table_rel"])
        meta_path = (root / task["table_rel"]).with_suffix(".columns.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
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
        out = {"key": task["key"], "fold": task["fold"], "set": task["set"], "cols": cols,
               "families": [str(x) for x in fam_names]}
        out["L1"] = eval_l1(ds_tr, ds_va, model, table, fam_codes, F, task.get("n_orders", 8), task["seed"])
        if units:
            lv = eval_units_wb(ds_tr, ds_va, model, table, fam_codes, F, task.get("n_orders", 8), task["seed"],
                               out["L1"]["slopes"])
            if "locomotor" in lv:
                out["L2"] = lv["locomotor"]
            if "realm" in lv:
                out["L3"] = lv["realm"]
        return out
    except Exception as e:  # noqa: BLE001
        import traceback
        return {"key": task.get("key"), "fold": task.get("fold"), "set": task.get("set"),
                "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]}
