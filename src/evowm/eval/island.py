"""岛屿对的外部检验（设计草案 docs/design/外部检验_岛屿_草案_2026-10-18.md）。

一对 = 岛屿特有种 I 与 Ponti 等 2025 给它配的最近的大陆亲属 M。模型只看 M 的形态，加上从 M 到 I 的习性变化与两者在
该模型训练树上的进化距离，一步预测 I（与主分析 E2、E3 的第一步预测相同），再用该折训练物种上拟合的探针读回形态。

逐个（模型，折）计算，只用两种都在该折留出科里的对：
- 带习性（eco）、不带习性（no：动作的习性部分置零，保留距离）的预测；
- 打乱对照：L、G 组内把习性部分换成同组另一对的（保留自己的距离）；同一折的所有运行用同一组打乱；
- 对照预测：B1（M 的观测值加上两种都在训练科里的全部对的平均差）、B2（线性基线，相对跗跖按运动方式、HWI 按迁徙程度）。
  B0（I = M 的观测值）在合并时直接取 M 的观测值。
合并、自助重抽样与报告在 scripts/eval_island_v1.py。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

PONTI = "data/raw/ponti2025/morpho_db.csv"
PONTI_FILTERED = "data/raw/ponti2025/morpho_db_filtered.csv"
WRIGHT = "data/raw/wright2016/skeletal_data.xlsx"

TARGETS = ["m_tarsus", "m_mass", "m_hwi", "m_wing", "k_femur", "k_tibiotarsus", "k_tarsometatarsus"]
BONES = ["k_femur", "k_tibiotarsus", "k_tarsometatarsus"]
WRIGHT_COLS = {"k_femur": "femur.length", "k_tibiotarsus": "tibiotarsus.length",
               "k_tarsometatarsus": "tarsometatarsus.length"}
# 读数：相对值 = 长度 − b × 体重（b 为该折训练物种上的 OLS 斜率，与主分析的相对跗跖相同）
READOUTS = {"rtarsus": ("m_tarsus", True), "mass": ("m_mass", False), "hwi": ("m_hwi", False),
            "rwing": ("m_wing", True), "rfemur": ("k_femur", True), "rtibio": ("k_tibiotarsus", True),
            "rtmt": ("k_tarsometatarsus", True)}
PERM_READOUT = {"L": "rtarsus", "G": "hwi"}      # 打乱对照只算两个主要读数
PERM_SEED = 20261018


# ---------------------------------------------------------------------------
# 配对与分组（只用习性，不看形态）
def load_pairs(root: Path, table: pd.DataFrame) -> pd.DataFrame:
    """Ponti 的对：两种都在物种表里，且在 BirdTree 里不是同一种。每行一对；附 Ponti 自己的迁徙状态与标本测量的种均值。"""
    P = pd.read_csv(root / PONTI, sep=";", encoding="latin1")
    filt = set(pd.read_csv(root / PONTI_FILTERED, sep=";", encoding="latin1")["pairID"])
    pos = lambda s: s.where(s > 0)  # noqa: E731
    P = P.assign(pmigr=pd.to_numeric(P["Migratory/sedentary"], errors="coerce"),
                 p_tarsus=np.log10(pos(pd.to_numeric(P["Tarsus.Length"], errors="coerce"))),
                 p_mass=np.log10(pos(pd.to_numeric(P["BM"], errors="coerce"))),
                 p_hwi=pd.to_numeric(P["Hand-wing.Index"], errors="coerce") / 100.0,
                 p_wing=np.log10(pos(pd.to_numeric(P["Wing.Length"], errors="coerce"))))
    sp = (P.groupby(["pairID", "phy_sp", "IMb"])
          .agg(pmigr=("pmigr", "first"), p_tarsus=("p_tarsus", "mean"), p_mass=("p_mass", "mean"),
               p_hwi=("p_hwi", "mean"), p_wing=("p_wing", "mean"))
          .reset_index())
    isl = sp[sp.IMb == "1I"].drop(columns="IMb").set_index("pairID")
    mai = sp[sp.IMb == "0M"].drop(columns="IMb").set_index("pairID")
    pr = isl.join(mai, lsuffix="_I", rsuffix="_M", how="inner").reset_index()
    pr = pr.rename(columns={"phy_sp_I": "tip_I", "phy_sp_M": "tip_M"})
    if pr.pairID.duplicated().any():
        raise ValueError("Ponti 表里有一个 pairID 对应多个岛屿种或大陆种")
    tips = set(table["tip"])
    pr = pr[(pr.tip_I != pr.tip_M) & pr.tip_I.isin(tips) & pr.tip_M.isin(tips)].copy()
    pr["in_filtered"] = pr.pairID.isin(filt)
    return pr.reset_index(drop=True)


def annotate(pr: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """物种表行号、运动方式（主分析 H1 的 7 类）、迁徙程度、科，以及分组标记。"""
    from evowm.sim.regimes import regime_locomotor
    row = pd.Series(np.arange(len(table)), index=table["tip"].to_numpy())
    loco = regime_locomotor(table).fillna("").to_numpy(dtype=object)
    migr = table["e_migr"].to_numpy(float)
    fam = table["family"].to_numpy(dtype=object)
    order = table["order"].to_numpy(dtype=object)
    pr = pr.copy()
    for s in ("I", "M"):
        r = row.loc[pr[f"tip_{s}"]].to_numpy()
        pr[f"row_{s}"] = r
        pr[f"loco_{s}"] = loco[r]
        pr[f"migr_{s}"] = migr[r]
        pr[f"family_{s}"] = fam[r]
    pr["order_I"] = order[pr.row_I.to_numpy()]
    known_l = (pr.loco_I != "") & (pr.loco_M != "")
    known_g = np.isfinite(pr.migr_I) & np.isfinite(pr.migr_M)
    pr["L"] = known_l & (pr.loco_I != pr.loco_M)
    pr["G"] = known_g & (pr.migr_I != pr.migr_M)
    pr["N"] = known_l & known_g & ~pr.L & ~pr.G
    pr["toward_T"] = pr.L & (pr.loco_I == "terrestrial")
    pr["away_T"] = pr.L & (pr.loco_M == "terrestrial")
    pr["mig_down"] = pr.G & (pr.migr_I < pr.migr_M)
    pr["G_ponti"] = np.isfinite(pr.pmigr_I) & np.isfinite(pr.pmigr_M) & (pr.pmigr_I != pr.pmigr_M)
    pr["cross_family"] = pr.family_I != pr.family_M
    return pr


# ---------------------------------------------------------------------------
# 观测值：物种表；真实数据时三段腿骨可用 Wright 等 2016 补缺
def wright_bones(root: Path, table: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Wright 等 2016 的三段腿骨：只用三段都有的标本，按种取 log10 均值，再减去与物种表（Skelevision）共有种上的平均差。"""
    W = pd.read_excel(root / WRIGHT, sheet_name="skeletal_data.csv")
    cols = list(WRIGHT_COLS.values())
    W = W.dropna(subset=cols)
    W = W[(W[cols] > 0).all(axis=1)]
    wm = np.log10(W[cols].astype(float)).groupby(W["species"]).mean()
    wm.columns = list(WRIGHT_COLS)
    tab = table.set_index("tip")
    wm = wm[wm.index.isin(tab.index)]
    both = wm.index[tab.loc[wm.index, BONES].notna().all(axis=1).to_numpy()]
    off = {b: float((wm.loc[both, b] - tab.loc[both, b]).mean()) for b in BONES}
    return wm - pd.Series(off), {"offsets": off, "n_overlap": int(len(both)), "n_species": int(len(wm))}


def observed(root: Path, table: pd.DataFrame, use_wright: bool) -> tuple[np.ndarray, dict]:
    """观测值矩阵（物种 × TARGETS）。use_wright 时三段腿骨缺失的物种用校准后的 Wright 值补上（只作目标值，不进探针）。"""
    Y = table[TARGETS].to_numpy(float).copy()
    if not use_wright:
        return Y, {}
    cal, info = wright_bones(root, table)
    row = pd.Series(np.arange(len(table)), index=table["tip"].to_numpy())
    r = row.loc[cal.index].to_numpy()
    filled = 0
    for b in BONES:
        j = TARGETS.index(b)
        miss = ~np.isfinite(Y[r, j])
        Y[r[miss], j] = cal[b].to_numpy()[miss]
        filled += int(miss.sum())
    info["n_filled_values"] = filled
    return Y, info


def derived(Y: np.ndarray, slopes: dict) -> dict:
    """TARGETS 列的数组 (..., T) → 读数字典；相对值扣除体重通路。"""
    jm = TARGETS.index("m_mass")
    out = {}
    for name, (col, rel) in READOUTS.items():
        j = TARGETS.index(col)
        out[name] = Y[..., j] - slopes[col] * Y[..., jm] if rel else Y[..., j].copy()
    return out


# ---------------------------------------------------------------------------
def pair_distances(tree, idx_a, idx_b) -> np.ndarray:
    """两片叶之间沿树的路径长度（Myr）；idx 为叶在 tree.tip_names 里的下标。"""
    parent, depth, nodes = tree.parent, tree.depth, tree.tip_nodes
    out = np.empty(len(idx_a))
    for k, (a, b) in enumerate(zip(idx_a, idx_b)):
        u, w = int(nodes[a]), int(nodes[b])
        anc, x = set(), u
        while x >= 0:
            anc.add(x)
            x = int(parent[x])
        x = w
        while x not in anc:
            x = int(parent[x])
        out[k] = depth[u] + depth[w] - 2.0 * depth[x]
    return out


def linear_fit(ds_tr, v: np.ndarray, cats: np.ndarray, H: int, n_orders: int, seed: int) -> dict:
    """主分析只用变化的线性基线（evowm.eval.runs.linear_baseline）推广到任意数值 v 与类别 cats：
    v̂_B = μ + exp(−αd)(v_A − μ) + κ(μ_c(B) − μ_c(A))。μ、μ_c 为训练物种的均值；α、κ 在训练科叶序上的相邻对上
    最小二乘拟合（κ 限于 [0, 1]）。类别为空字符串的物种用 μ。"""
    from evowm.eval.recovery import sample_windows
    tr = np.flatnonzero(ds_tr.rows_mask)
    ok = tr[np.isfinite(v[tr])]
    mu = float(np.mean(v[ok]))
    cats = np.asarray(cats, dtype=object)
    mu_c = {c: float(np.mean(v[ok][cats[ok] == c])) for c in set(cats[ok]) if c != ""}
    m_c = np.array([mu_c.get(c, mu) for c in cats])
    W = sample_windows(ds_tr, n_orders, H + 1, seed)
    a, b, d = W.rows[:, :H].reshape(-1), W.rows[:, 1:H + 1].reshape(-1), W.dist[:, :H].reshape(-1)
    good = np.isfinite(v[a]) & np.isfinite(v[b])
    a, b, d = a[good], b[good], d[good]

    def mse(la):
        return float(np.mean((mu + np.exp(-np.exp(la) * d) * (v[a] - mu) - v[b]) ** 2))

    alpha = float(np.exp(minimize_scalar(mse, bounds=(np.log(1e-4), np.log(2.0)), method="bounded").x))
    res = v[b] - (mu + np.exp(-alpha * d) * (v[a] - mu))
    dl = m_c[b] - m_c[a]
    nz = dl != 0
    kappa = float(np.clip(np.sum(res[nz] * dl[nz]) / max(float(np.sum(dl[nz] ** 2)), 1e-12), 0.0, 1.0))
    return {"mu": mu, "m_c": m_c, "alpha": alpha, "kappa": kappa}


def linear_predict(fit: dict, v_a: np.ndarray, rows_a: np.ndarray, rows_b: np.ndarray, d: np.ndarray) -> np.ndarray:
    m_c = fit["m_c"]
    return fit["mu"] + np.exp(-fit["alpha"] * d) * (v_a - fit["mu"]) + fit["kappa"] * (m_c[rows_b] - m_c[rows_a])


# ---------------------------------------------------------------------------
def evaluate_fold(ds_tr, ds_va, model, table: pd.DataFrame, pairs: pd.DataFrame, fold: int, seed: int,
                  n_perm: int = 1000, Y: np.ndarray | None = None, n_orders: int = 8) -> dict:
    """一个（模型，折）：两种都在该折留出科里的对上的预测与对照。Y 为观测值矩阵（物种 × TARGETS，默认物种表）。"""
    from evowm.eval.recovery import Level1, apply_probe, fit_probe
    L1 = Level1(model)
    H = int(L1.jepa.predictor.predictor.pos_embedding.shape[1]) if hasattr(L1.jepa.predictor, "predictor") else 3
    V = table[TARGETS].to_numpy(float)                 # 探针与斜率只用物种表的训练物种
    Y = V if Y is None else Y
    rows_tr = np.flatnonzero(ds_tr.rows_mask)
    held = ds_va.rows_mask
    sel = held[pairs.row_I.to_numpy()] & held[pairs.row_M.to_numpy()]
    P = pairs[sel].reset_index(drop=True)
    rI, rM = P.row_I.to_numpy(), P.row_M.to_numpy()

    jm = TARGETS.index("m_mass")
    slopes = {}
    for c in ("m_tarsus", "m_wing", *BONES):
        j = TARGETS.index(c)
        ok = rows_tr[np.isfinite(V[rows_tr, j]) & np.isfinite(V[rows_tr, jm])]
        slopes[c] = float(np.polyfit(V[ok, jm], V[ok, j], 1)[0])

    probes = fit_probe(L1.encode(ds_tr.pix), V, rows_tr)
    tree = ds_tr.trees[0]
    pos = {n: i for i, n in enumerate(tree.tip_names)}
    tips = table["tip"].to_numpy()
    d = pair_distances(tree, [pos[t] for t in tips[rM]], [pos[t] for t in tips[rI]])

    def acts(eco: bool) -> np.ndarray:
        return np.stack([ds_va.build_action(np.array([a, b]), np.array([dd], np.float32), eco=eco)[0]
                         for a, b, dd in zip(rM, rI, d)]).astype(np.float32)

    act_eco, act_no = acts(True), acts(False)
    emb = L1.encode(ds_va.pix[rM])                    # (N, D)

    def predict(e: np.ndarray, a: np.ndarray) -> np.ndarray:
        return apply_probe(probes, L1.predict(e[:, None, :], a[:, None, :]))[:, 0, :]

    pred_eco, pred_no = predict(emb, act_eco), predict(emb, act_no)

    perm = {}
    for code, (grp, readout) in enumerate(PERM_READOUT.items()):
        idx = np.flatnonzero(P[grp].to_numpy())
        if n_perm <= 0 or len(idx) < 2:
            perm[grp] = {"pair_index": idx, readout: np.zeros((0, len(idx)))}
            continue
        rng = np.random.default_rng([PERM_SEED, int(fold), code])
        order = np.stack([rng.permutation(len(idx)) for _ in range(n_perm)])
        A = np.repeat(act_eco[idx][None], n_perm, axis=0)
        A[:, :, :-2] = act_eco[idx][order][:, :, :-2]          # 习性部分换成同组另一对的，保留自己的距离
        E = np.repeat(emb[idx][None], n_perm, axis=0).reshape(-1, emb.shape[-1])
        pp = predict(E, A.reshape(-1, A.shape[-1])).reshape(n_perm, len(idx), len(TARGETS))
        perm[grp] = {"pair_index": idx, readout: derived(pp, slopes)[readout]}

    # 对照预测
    obs_I, obs_M = derived(Y[rI], slopes), derived(Y[rM], slopes)
    tr_set = np.zeros(len(table), bool)
    tr_set[rows_tr] = True
    tp = pairs[tr_set[pairs.row_I.to_numpy()] & tr_set[pairs.row_M.to_numpy()]]
    dI, dM = derived(Y[tp.row_I.to_numpy()], slopes), derived(Y[tp.row_M.to_numpy()], slopes)
    delta = {k: float(np.nanmean(dI[k] - dM[k])) if np.isfinite(dI[k] - dM[k]).any() else float("nan") for k in dI}
    b1 = {k: obs_M[k] + delta[k] for k in obs_M}

    v_rt = V[:, TARGETS.index("m_tarsus")] - slopes["m_tarsus"] * V[:, jm]
    v_hwi = V[:, TARGETS.index("m_hwi")]
    from evowm.sim.regimes import regime_locomotor
    loco = regime_locomotor(table).fillna("").to_numpy(dtype=object)
    mig = np.array(["" if not np.isfinite(x) else f"{x:.2f}" for x in table["e_migr"].to_numpy(float)], dtype=object)
    fit_rt = linear_fit(ds_tr, v_rt, loco, H, n_orders, seed)
    fit_hwi = linear_fit(ds_tr, v_hwi, mig, H, n_orders, seed)
    b2 = {"rtarsus": linear_predict(fit_rt, obs_M["rtarsus"], rM, rI, d),
          "hwi": linear_predict(fit_hwi, obs_M["hwi"], rM, rI, d)}

    return {"fold": int(fold), "pairID": P.pairID.to_numpy(), "distance": d, "slopes": slopes,
            "obs_I": obs_I, "obs_M": obs_M, "eco": derived(pred_eco, slopes), "no": derived(pred_no, slopes),
            "B1": b1, "B1_delta": delta, "B2": b2, "perm": perm,
            "linear": {"rtarsus": {"alpha": fit_rt["alpha"], "kappa": fit_rt["kappa"]},
                       "hwi": {"alpha": fit_hwi["alpha"], "kappa": fit_hwi["kappa"]}},
            "n_train_pairs": int(len(tp))}


def run_island_task(task: dict) -> dict:
    """task：root, table_rel, tree_file, fold, ckpt, seed, key, analysis, n_perm, wright（真实数据时为 True）。"""
    import torch
    torch.set_num_threads(1)
    from evowm.eval.recovery import load_world_model
    from evowm.eval.runs import ckpt_action_mode, fold_datasets
    root = Path(task["root"])
    table = pd.read_csv(root / task["table_rel"])
    pairs = annotate(load_pairs(root, table), table)
    Y, _ = observed(root, table, bool(task.get("wright")))
    ckpt = Path(task["ckpt"])
    mode = ckpt_action_mode(ckpt)
    if mode != "change":
        raise ValueError(f"{ckpt} 的动作方式为 {mode}，岛屿检验只用 change")
    ds_tr, ds_va = fold_datasets(root, task["table_rel"], task["tree_file"], task["fold"], mode)
    model = load_world_model(ckpt, root / "hjepa")
    out = evaluate_fold(ds_tr, ds_va, model, table, pairs, task["fold"], task["seed"], task.get("n_perm", 1000), Y)
    out.update(key=task["key"], analysis=task["analysis"])
    return out
