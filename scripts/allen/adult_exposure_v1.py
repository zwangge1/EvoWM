"""成鸟暴露说与鸭科复核（设计 docs/design/成鸟暴露与鸭科复核_设计_2026-10-27.md）。

A：P 组内 T × X（X = 地面 + 水面 + 水下觅食比例）的系数，AVONET 跗跖与 Eliason 跗跖骨。
B：全部非雀形目，v1 主模型加 T × X，看发育方式的交互项是否被吸收。
C：Watanabe 鸭科三段骨（species_traits 里 src_skeleton == watanabe2017）在 P 组的分段复核。
探索：雀形目内 T × X。
在 EvoWM 根目录运行：OMP_NUM_THREADS=2 python scripts/allen/adult_exposure_v1.py；结果 outputs/allen_v1/adult_exposure.md。
"""
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import cholesky, solve_triangular

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/allen"))
from devmode_v2 import LIFE, base, ci, load_eliason  # noqa: E402
from evowm.data.tree import read_trees  # noqa: E402
from evowm.eval.pgls import pgls_lambda, tip_vcv  # noqa: E402

OUT = ROOT / "outputs/allen_v1"


def gls_fit(d, C_all, pos, resp, cols_fn):
    d = d.dropna(subset=[resp]).reset_index(drop=True)
    rows = np.array([pos[x] for x in d["tip"]])
    C = C_all[np.ix_(rows, rows)]
    cols = cols_fn(d)
    names = [k for k in cols if k == "intercept" or np.asarray(cols[k]).std() > 0]
    X = np.column_stack([cols[k] for k in names])
    while np.linalg.matrix_rank(X) < X.shape[1]:          # 小样本里哑变量共线时，从后往前丢协变量
        drop = [k for k in names if k not in ("intercept", "M", "T", "TxX", "X", "TxE", "E")][-1]
        names.remove(drop)
        X = np.column_stack([cols[k] for k in names])
    y = d[resp].to_numpy(float)
    res = pgls_lambda(y, X, C, names)
    n = len(d)
    V = C * res["lambda"]
    V[np.arange(n), np.arange(n)] = np.diag(C)
    L = cholesky(V, lower=True, check_finite=False)
    Z = solve_triangular(L, X, lower=True, check_finite=False)
    z = solve_triangular(L, y, lower=True, check_finite=False)
    beta, *_ = np.linalg.lstsq(Z, z, rcond=None)
    r = z - Z @ beta
    s2 = float(r @ r) / (n - X.shape[1])
    cov = np.linalg.inv(Z.T @ Z) * s2
    return {"n": n, "lambda": res["lambda"], "beta": dict(zip(names, beta)), "se": {k: np.sqrt(cov[i, i]) for i, k in enumerate(names)}}


def covars(d, cols):
    cols.update({"habdens": d["e_habdens"].to_numpy(float), "migr": d["e_migr"].to_numpy(float), "lat_span": d["lat_span"].to_numpy(float) / 10.0})
    for k in LIFE:
        v = d[f"e_life_{k}"].to_numpy(float)
        if v.std() > 0:
            cols[f"life_{k}"] = v
    lk = [k for k in cols if k.startswith("life_")]
    if lk and np.all(sum(cols[k] for k in lk) == 1):
        cols.pop(lk[-1])
    return cols


def cols_TX(d):
    T = d["bio1_c"].to_numpy(float) / 10.0
    X = d["X"].to_numpy(float)
    return covars(d, {"intercept": np.ones(len(d)), "M": d["m_mass"].to_numpy(float), "T": T, "TxX": T * X, "X": X})


def cols_TX_E(d):
    T = d["bio1_c"].to_numpy(float) / 10.0
    X = d["X"].to_numpy(float)
    E = d["exposed"].to_numpy(float)
    return covars(d, {"intercept": np.ones(len(d)), "M": d["m_mass"].to_numpy(float), "T": T, "TxX": T * X, "X": X, "TxE": T * E, "E": E})


def cols_T(d):
    T = d["bio1_c"].to_numpy(float) / 10.0
    return covars(d, {"intercept": np.ones(len(d)), "M": d["m_mass"].to_numpy(float), "T": T})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = base()
    tree = read_trees(ROOT / "data/processed/trees/hackett_t021.nwk", max_trees=1)[0]
    pos = {n: i for i, n in enumerate(tree.tip_names)}
    C_all = tip_vcv(tree)
    d = d[d["tip"].isin(pos)].reset_index(drop=True)
    d = d.merge(load_eliason(set(d["tip"])), on="tip", how="left")
    d["X"] = d[["e_fs_ground", "e_fs_wataroundsurf", "e_fs_watbelowsurf"]].sum(axis=1, min_count=1)
    d.loc[d[["e_fs_ground", "e_fs_wataroundsurf", "e_fs_watbelowsurf"]].isna().any(axis=1), "X"] = np.nan
    L = ["# 成鸟暴露说与鸭科复核", "", "脚本 `scripts/allen/adult_exposure_v1.py`。判据见设计文档。系数：温度每升高 10 °C 的 log10 变化，0.0043 = 1%，方括号为 95% 区间。", ""]

    # A
    P = d[(d["mode"] == "P") & d["X"].notna()].reset_index(drop=True)
    L += ["## A 早成鸟内部：成鸟暴露 X（地面 + 水面 + 水下觅食比例）是否预测温度斜率", "",
          f"P 组有 X 的物种 {len(P)}；X 的分布：均值 {P.X.mean():.2f}，四分位 {P.X.quantile(.25):.2f} 到 {P.X.quantile(.75):.2f}。", "",
          "| 响应 | n | λ | T（X = 0 时） | T × X | X = 1 时的斜率 |", "|---|---|---|---|---|---|"]
    a_recs = {}
    for lab, resp, sub in [("AVONET 跗跖", "m_tarsus", P[P["avonet_inferred"] == 0]), ("Eliason 跗跖骨", "b_tmt", P), ("Eliason r", "b_r", P)]:
        r = gls_fit(sub.reset_index(drop=True), C_all, pos, resp, cols_TX)
        a_recs[lab] = r
        b, s = r["beta"], r["se"]
        L.append(f"| {lab} | {r['n']} | {r['lambda']:.2f} | {ci(b['T'], s['T'])} | {ci(b['TxX'], s['TxX'])} | {b['T'] + b['TxX']:+.4f} |")
    ra = a_recs["AVONET 跗跖"]
    a_ok = ra["beta"]["TxX"] - 1.96 * ra["se"]["TxX"] > 0 and a_recs["Eliason 跗跖骨"]["beta"]["TxX"] > 0
    a_against = ra["beta"]["TxX"] - 1.96 * ra["se"]["TxX"] <= 0
    L += ["", f"判定：{'A 支持成鸟暴露说。' if a_ok else ('A 反对成鸟暴露说：早成鸟内部，露天觅食比例不改变温度斜率。' if a_against else 'A 介于两者之间。')}", ""]

    # B
    NP = d[(d["order"] != "Passeriformes") & d["X"].notna() & (d["avonet_inferred"] == 0) & d["m_tarsus"].notna()].reset_index(drop=True)
    rb = gls_fit(NP, C_all, pos, "m_tarsus", cols_TX_E)
    rb0 = gls_fit(NP, C_all, pos, "m_tarsus", lambda dd: covars(dd, {"intercept": np.ones(len(dd)), "M": dd["m_mass"].to_numpy(float), "T": dd["bio1_c"].to_numpy(float) / 10.0,
                                                                        "TxE": dd["bio1_c"].to_numpy(float) / 10.0 * dd["exposed"].to_numpy(float), "E": dd["exposed"].to_numpy(float)}))
    L += ["## B 全部非雀形目：成鸟暴露能否吸收发育方式的差别", "",
          f"物种 {rb['n']}。只放发育方式时 T × 暴露组 = {ci(rb0['beta']['TxE'], rb0['se']['TxE'])}；",
          f"同时放成鸟暴露 X 后：T × 暴露组 = {ci(rb['beta']['TxE'], rb['se']['TxE'])}，T × X = {ci(rb['beta']['TxX'], rb['se']['TxX'])}。", ""]

    # C
    W = d[(d["src_skeleton"] == "watanabe2017")].reset_index(drop=True)
    W["w_r"] = W["k_tarsometatarsus"] - W["k_femur"]
    L += ["## C 鸭科 90 种（Watanabe 2017 三段骨，独立于 Eliason）", "",
          "| 响应 | n | λ | 温度系数 |", "|---|---|---|---|"]
    c_recs = {}
    for lab, resp in [("股骨", "k_femur"), ("胫跗骨", "k_tibiotarsus"), ("跗跖骨", "k_tarsometatarsus"), ("r = 跗跖骨 − 股骨", "w_r")]:
        r = gls_fit(W, C_all, pos, resp, cols_T)
        c_recs[lab] = r
        L.append(f"| {lab} | {r['n']} | {r['lambda']:.2f} | {ci(r['beta']['T'], r['se']['T'])} |")
    ct, cr = c_recs["跗跖骨"], c_recs["r = 跗跖骨 − 股骨"]
    c_ok = ct["beta"]["T"] - 1.96 * ct["se"]["T"] > 0 and (cr["beta"]["T"] - 1.96 * cr["se"]["T"] > 0)
    L += ["", f"判定：{'C 复核通过。' if c_ok else 'C 未复核（90 种功效有限，不推翻 Eliason）。'}", ""]

    # 探索：雀形目内 T × X
    PA = d[(d["order"] == "Passeriformes") & d["X"].notna() & (d["avonet_inferred"] == 0) & d["m_tarsus"].notna()].reset_index(drop=True)
    rp = gls_fit(PA, C_all, pos, "m_tarsus", cols_TX)
    L += ["## 探索：雀形目内 T × X", "",
          f"物种 {rp['n']}；T（X = 0）{ci(rp['beta']['T'], rp['se']['T'])}，T × X {ci(rp['beta']['TxX'], rp['se']['TxX'])}。", ""]
    (OUT / "adult_exposure.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
