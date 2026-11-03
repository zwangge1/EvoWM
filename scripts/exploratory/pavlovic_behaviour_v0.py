"""探索（不进判据）：Pavlovic 等 2019 的保温行为（坐下盖腿 Sit、单腿站立 S1L、喙插背羽 BR，物种级有无 0/1，852 种）
与跗跖温度斜率的关系。

数据是"照片里出现过这种行为"的有无，不是腿暴露的时间份额，而且水鸟与地栖鸟的 Sit 更高（与 X 正相关），
所以它不能当"暴露"用，只能问另一个问题：有行为缓冲（能把腿盖起来）的物种，跗跖对温度的响应是否更弱或更强。
在 EvoWM 根目录运行：OMP_NUM_THREADS=1 python scripts/exploratory/pavlovic_behaviour_v0.py
输出 outputs/allen_v1/pavlovic_behaviour.md。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/allen"))
from devmode_v2 import base  # noqa: E402
from evowm.data.tree import read_trees  # noqa: E402
from evowm.eval.pgls import pgls_lambda, tip_vcv  # noqa: E402
from mass_vs_exposure_v1 import ci, covars  # noqa: E402

XLSX = ROOT / "data/raw/Pavlovic/extracted/Pavlovic_etal_bird_thermoregulatory_behaviour_data_Oct2018.xlsx"
OUT = ROOT / "outputs/allen_v1/pavlovic_behaviour.md"


def load_pavlovic():
    raw = pd.read_excel(XLSX, sheet_name=0, header=None)
    hdr = raw.index[raw.iloc[:, 0].astype(str).str.strip() == "Order"][0]
    x = raw.iloc[hdr + 1:].copy()
    x.columns = raw.iloc[hdr].tolist()
    x = x.dropna(subset=["Scientific name"])
    for c in ["BR", "S1L", "Sit", "BR-weight", "S1L-weight", "Sit-weight"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x["tip"] = x["Scientific name"].astype(str).str.strip().str.replace(" ", "_")
    return x[["tip", "BR", "S1L", "Sit", "BR-weight", "S1L-weight", "Sit-weight"]]


def fit(d, C_all, pos, terms):
    rows = np.array([pos[t] for t in d["tip"]])
    C = C_all[np.ix_(rows, rows)]
    T = d["bio1_c"].to_numpy(float) / 10.0
    cols = {"intercept": np.ones(len(d)), "M": d["m_mass"].to_numpy(float) - d["m_mass"].mean(), "T": T}
    for k in terms:
        v = d[k].to_numpy(float)
        cols[k] = v
        cols[f"Tx{k}"] = T * v
    cols.update(covars(d))
    names = list(cols)
    res = pgls_lambda(d["m_tarsus"].to_numpy(float), np.column_stack([cols[k] for k in names]), C, names)
    k = len(names) + 2
    return {"n": res["n"], "aic": 2 * k - 2 * res["loglik"], "coef": {n: (res["coef"][n]["beta"], res["coef"][n]["se"]) for n in names if n == "T" or n.startswith("Tx")}}


def main():
    d = base()
    d["X"] = d[["e_fs_ground", "e_fs_wataroundsurf", "e_fs_watbelowsurf"]].sum(axis=1, min_count=3)
    d = d[d["avonet_inferred"] == 0].merge(load_pavlovic(), on="tip", how="inner").dropna(subset=["m_tarsus", "X", "bio1_c", "Sit", "S1L"]).reset_index(drop=True)
    tree = read_trees(ROOT / "data/processed/trees/hackett_t021.nwk", max_trees=1)[0]
    pos = {n: i for i, n in enumerate(tree.tip_names)}
    C_all = tip_vcv(tree)
    d = d[d["tip"].isin(pos)].reset_index(drop=True)
    L = ["# 探索：Pavlovic 等 2019 的保温行为与跗跖温度斜率", "",
         f"匹配到 {len(d)} 种（雀形目 {(d.order == 'Passeriformes').sum()}）。行为为物种级有无（0/1）。Sit 与 X 的 Spearman 相关 "
         f"{d[['Sit', 'X']].corr(method='spearman').iloc[0, 1]:+.2f}，S1L 与 X {d[['S1L', 'X']].corr(method='spearman').iloc[0, 1]:+.2f}；"
         f"非雀形目 Sit 均值 {d.loc[d.order != 'Passeriformes', 'Sit'].mean():.2f}，雀形目 {d.loc[d.order == 'Passeriformes', 'Sit'].mean():.2f}。",
         "坐下盖腿的行为在地栖与水生鸟里更常见，所以它不是腿暴露时间的反向指标，这里只问行为缓冲是否改变温度斜率。", "",
         "| 模型 | n | AIC | T | 交互项 |", "|---|---|---|---|---|"]
    specs = [("E0 基线", []), ("E1 + X", ["X"]), ("E2 + Sit", ["Sit"]), ("E3 + S1L", ["S1L"]), ("E4 + X + Sit", ["X", "Sit"]), ("E5 + X + Sit + S1L", ["X", "Sit", "S1L"])]
    for lab, terms in specs:
        r = fit(d, C_all, pos, terms)
        inter = "，".join(f"T×{k} {ci(*r['coef'][f'Tx{k}'])}" for k in terms)
        L.append(f"| {lab} | {r['n']} | {r['aic']:.1f} | {ci(*r['coef']['T'])} | {inter} |")
    sub = d[d["Sit-weight"] >= 0.5].reset_index(drop=True)
    r = fit(sub, C_all, pos, ["X", "Sit"])
    L += ["", f"只用 Sit 判定权重 ≥ 0.5 的 {len(sub)} 种：T×X {ci(*r['coef']['TxX'])}，T×Sit {ci(*r['coef']['TxSit'])}。"]
    OUT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
