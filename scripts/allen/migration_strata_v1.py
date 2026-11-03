"""迁徙分层的稳健性（设计 docs/design/迁徙分层_稳健性_设计_2026-10-30.md）。
在 EvoWM 根目录运行：OMP_NUM_THREADS=2 python scripts/allen/migration_strata_v1.py → outputs/allen_v1/migration_strata.md
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/allen"))
from devmode_v2 import LIFE, base  # noqa: E402
from evowm.data.tree import read_trees  # noqa: E402
from evowm.eval.pgls import pgls_lambda, tip_vcv  # noqa: E402

DESIGN = ROOT / "docs/design/迁徙分层_稳健性_设计_2026-10-30.md"
OUT = ROOT / "outputs/allen_v1/migration_strata.md"
LAYERS = [("全部", None), ("留鸟", 0.0), ("部分迁徙", 0.5), ("完全迁徙", 1.0)]


def fit(d, C_all, pos, terms, m_center):
    """terms ⊆ {'G', 'M', 'X'}：G 为暴露组哑变量（发育方式），M 为体重交互，X 为成鸟暴露交互。零方差列自动去掉。"""
    rows = np.array([pos[t] for t in d["tip"]])
    C = C_all[np.ix_(rows, rows)]
    T = d["bio1_c"].to_numpy(float) / 10.0
    M = d["m_mass"].to_numpy(float) - m_center
    cols = {"intercept": np.ones(len(d)), "M": M, "T": T}
    if "G" in terms:
        G = d["exposed"].to_numpy(float); cols["G"] = G; cols["TxG"] = T * G
    if "M" in terms:
        cols["TxM"] = T * M
    if "X" in terms:
        X = d["X"].to_numpy(float); cols["X"] = X; cols["TxX"] = T * X
    cols.update({"habdens": d["e_habdens"].to_numpy(float), "migr": d["e_migr"].to_numpy(float), "lat_span": d["lat_span"].to_numpy(float) / 10.0})
    for k in LIFE:
        cols[f"life_{k}"] = d[f"e_life_{k}"].to_numpy(float)
    names = [k for k in cols if k == "intercept" or np.std(cols[k]) > 0]
    lk = [k for k in names if k.startswith("life_")]
    if lk and np.allclose(sum(cols[k] for k in lk), 1):
        names.remove(lk[-1])
    Xd = np.column_stack([cols[k] for k in names])
    while np.linalg.matrix_rank(Xd) < Xd.shape[1]:
        names.remove([k for k in names if k.startswith("life_")][-1]); Xd = np.column_stack([cols[k] for k in names])
    res = pgls_lambda(d["m_tarsus"].to_numpy(float), Xd, C, names)
    return res["n"], {k: (res["coef"][k]["beta"], res["coef"][k]["se"]) for k in ("T", "TxG", "TxM", "TxX") if k in res["coef"]}


def ci(c):
    b, se = c
    return f"{b:+.4f} [{b - 1.96 * se:+.4f}, {b + 1.96 * se:+.4f}]"


def main():
    if not DESIGN.exists():
        sys.exit("缺少设计文档")
    d = base()
    d["X"] = d[["e_fs_ground", "e_fs_wataroundsurf", "e_fs_watbelowsurf"]].sum(axis=1, min_count=3)
    tree = read_trees(ROOT / "data/processed/trees/hackett_t021.nwk", max_trees=1)[0]
    pos = {n: i for i, n in enumerate(tree.tip_names)}
    C_all = tip_vcv(tree)
    d = d[d["tip"].isin(pos) & (d["avonet_inferred"] == 0)].dropna(subset=["m_tarsus", "e_migr"]).reset_index(drop=True)
    mc = float(d.dropna(subset=["X"])["m_mass"].mean())
    nonp = d[d["order"] != "Passeriformes"]
    sall = d.dropna(subset=["X"])
    L = ["# 迁徙分层的稳健性", "", f"设计 `{DESIGN.relative_to(ROOT)}`。系数为每升高 10 °C 的 log10 变化，方括号为 95% 区间。", "",
         "## 1 非雀形目，发育方式之差（暴露组 − 巢内组）", "", "| 层 | n | 暴露组斜率之差 |", "|---|---|---|"]
    for lab, v in LAYERS:
        sub = nonp if v is None else nonp[nonp["e_migr"] == v]
        n, c = fit(sub.reset_index(drop=True), C_all, pos, {"G"}, mc)
        L.append(f"| {lab} | {n} | {ci(c['TxG'])} |"); print(lab, "G", n, ci(c["TxG"]), flush=True)
    L += ["", "## 2 S_all，两轴同放（M3）", "", "| 层 | n | 体型轴 d | 暴露轴 f |", "|---|---|---|---|"]
    for lab, v in LAYERS:
        sub = sall if v is None else sall[sall["e_migr"] == v]
        n, c = fit(sub.reset_index(drop=True), C_all, pos, {"M", "X"}, mc)
        L.append(f"| {lab} | {n} | {ci(c['TxM'])} | {ci(c['TxX'])} |"); print(lab, "M3", n, ci(c["TxM"]), ci(c["TxX"]), flush=True)
    L += ["", "## 3 留鸟层，只有暴露轴（M2）", "", "| 样本 | n | 暴露轴 f |", "|---|---|---|"]
    for lab, sub in (("非雀形目", sall[(sall["order"] != "Passeriformes") & (sall["e_migr"] == 0)]), ("雀形目", sall[(sall["order"] == "Passeriformes") & (sall["e_migr"] == 0)])):
        n, c = fit(sub.reset_index(drop=True), C_all, pos, {"X"}, mc)
        L.append(f"| {lab} | {n} | {ci(c['TxX'])} |"); print(lab, "M2 resident", n, ci(c["TxX"]), flush=True)
    OUT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
