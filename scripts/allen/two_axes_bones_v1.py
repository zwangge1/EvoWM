"""两轴与分段：骨长上的体型与暴露同场比较（设计 docs/design/两轴与分段_骨长_设计_2026-10-30.md）。
在 EvoWM 根目录运行：OMP_NUM_THREADS=2 python scripts/allen/two_axes_bones_v1.py → outputs/allen_v1/two_axes_bones.md
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/allen"))
from devmode_v2 import LIFE, base, load_eliason  # noqa: E402
from evowm.data.tree import read_trees  # noqa: E402
from evowm.eval.pgls import pgls_lambda, tip_vcv  # noqa: E402

DESIGN = ROOT / "docs/design/两轴与分段_骨长_设计_2026-10-30.md"
OUT = ROOT / "outputs/allen_v1/two_axes_bones.md"


def gfit(d, C_all, pos, resp, inter, temp="bio1_c", mc=None):
    """inter：交互项列表，元素为 (温度列, 调节列)，调节列为 'M' 或 'X'。返回 n、AIC、交互系数。"""
    d = d.dropna(subset=[resp]).reset_index(drop=True)
    rows = np.array([pos[t] for t in d["tip"]])
    C = C_all[np.ix_(rows, rows)]
    M = d["m_mass"].to_numpy(float) - (mc if mc is not None else d["m_mass"].mean())
    X = d["X"].to_numpy(float)
    temps = sorted({t for t, _ in inter} | ({temp} if temp else set()))
    cols = {"intercept": np.ones(len(d)), "M": M, "X": X}
    for t in temps:
        cols[t] = d[t].to_numpy(float) / 10.0
    for t, z in inter:
        cols[f"{t}x{z}"] = cols[t] * (M if z == "M" else X)
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
    res = pgls_lambda(d[resp].to_numpy(float), Xd, C, names)
    k = len(names) + 2
    return {"n": res["n"], "aic": 2 * k - 2 * res["loglik"], "lambda": res["lambda"],
            "coef": {n: (res["coef"][n]["beta"], res["coef"][n]["se"]) for n in names if "x" in n and n.split("x")[-1] in ("M", "X") or n in temps}}


def ci(v):
    return f"{v[0]:+.4f} [{v[0] - 1.96 * v[1]:+.4f}, {v[0] + 1.96 * v[1]:+.4f}]"


def excl(v):
    return v[0] - 1.96 * v[1] > 0 or v[0] + 1.96 * v[1] < 0


def judge(f):
    tm, fe, r = f["tmt"], f["fem"], f["r"]
    if tm[0] > 0 and excl(tm) and (not excl(fe) or fe[0] < 0.5 * tm[0]) and r[0] > 0 and excl(r):
        return "暴露轴分段：只有跗跖骨随暴露变化，股骨不变"
    if tm[0] > 0 and excl(tm) and fe[0] > 0 and excl(fe) and not excl(r):
        return "暴露轴整腿"
    if not excl(tm):
        return "未检出"
    return "介于之间，按实描述"


def main():
    if not DESIGN.exists():
        sys.exit("缺少设计文档")
    d = base()
    d["X"] = d[["e_fs_ground", "e_fs_wataroundsurf", "e_fs_watbelowsurf"]].sum(axis=1, min_count=3)
    tree = read_trees(ROOT / "data/processed/trees/hackett_t021.nwk", max_trees=1)[0]
    pos = {n: i for i, n in enumerate(tree.tip_names)}
    C_all = tip_vcv(tree)
    d = d[d["tip"].isin(pos)].dropna(subset=["X", "bio1_c"]).reset_index(drop=True)
    A = d.merge(load_eliason(set(d["tip"])), on="tip", how="inner")
    A = A.rename(columns={"b_fem": "fem", "b_tbt": "tbt", "b_tmt": "tmt", "b_r": "r"})
    B = d.copy()
    B["fem"], B["tbt"], B["tmt"] = B["k_femur"], B["k_tibiotarsus"], B["k_tarsometatarsus"]
    B["r"] = B["tmt"] - B["fem"]
    B = B.dropna(subset=["fem", "tmt"]).reset_index(drop=True)
    sets = {"A Eliason 全部": A.dropna(subset=["fem", "tmt"]).reset_index(drop=True)}
    sets["A Eliason 非雀形目"] = sets["A Eliason 全部"][sets["A Eliason 全部"]["order"] != "Passeriformes"].reset_index(drop=True)
    sets["B Skelevision + 鸭科"] = B
    L = ["# 两轴与分段：骨长上的体型与暴露同场比较", "", f"设计 `{DESIGN.relative_to(ROOT)}`。系数为每升高 10 °C 的 log10 变化，方括号为 95% 区间。", ""]
    verdicts = {}
    for sname, S in sets.items():
        mc = float(S["m_mass"].mean())
        L += [f"## {sname}（{len(S)} 种，雀形目 {(S['order'] == 'Passeriformes').sum()}，X 均值 {S['X'].mean():.2f}）", "",
              "| 响应 | AIC M2 − M1 | M1 体型轴 d | M2 暴露轴 f | M3 体型轴 d | M3 暴露轴 f |", "|---|---|---|---|---|---|"]
        f3 = {}
        for resp, zh in (("fem", "股骨"), ("tbt", "胫跗骨"), ("tmt", "跗跖骨"), ("r", "比例 r")):
            m1 = gfit(S, C_all, pos, resp, [("bio1_c", "M")], mc=mc)
            m2 = gfit(S, C_all, pos, resp, [("bio1_c", "X")], mc=mc)
            m3 = gfit(S, C_all, pos, resp, [("bio1_c", "M"), ("bio1_c", "X")], mc=mc)
            f3[resp] = m3["coef"]["bio1_cxX"]
            L.append(f"| {zh} | {m2['aic'] - m1['aic']:+.1f} | {ci(m1['coef']['bio1_cxM'])} | {ci(m2['coef']['bio1_cxX'])} | "
                     f"{ci(m3['coef']['bio1_cxM'])} | {ci(m3['coef']['bio1_cxX'])} |")
            print(sname, zh, ci(m3["coef"]["bio1_cxM"]), ci(m3["coef"]["bio1_cxX"]), flush=True)
        verdicts[sname] = judge(f3)
        L += ["", f"判定（M3 暴露轴）：{verdicts[sname]}", ""]
    L.insert(4, f"## 主判定（数据集 A Eliason 全部）：{verdicts['A Eliason 全部']}\n")
    OUT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
