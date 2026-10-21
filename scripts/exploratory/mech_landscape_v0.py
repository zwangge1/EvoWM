"""探索 v0：三段腿骨比例的几何力学量，与生活方式有没有干净的关系（只用 CPU，几分钟）。探索性分析，不判定。

数据：Skelevision 三段腿骨（股骨 l1、胫跗骨 l2、跗跖骨 l3，log10 mm）与体重都有的物种（约 1,767 种，几乎全是雀形目加雁形目）。
腿当作平面三连杆：趾尖触地点在原点，髋在触地点正上方（静态平衡），膝、踝的内角给定，踝在趾后、膝在踝前、髋在膝后（鸟腿的折线）。
几何量（肌肉只以"各关节扭转刚度相同"这一条假设进来）：
- 相对腿长 rel_leg = log10(l1+l2+l3) − b·log10(体重)，b 为全部物种上的 OLS 斜率；
- 比例 p_f、p_t、p_m：三段各占总长的比例；
- 站姿高度 H/L：膝、踝内角 120° 时髋高 ÷ 总长；
- 下蹲行程 crouch = (H(150°,150°) − H(90°,90°)) / L：从伸展到深蹲髋能下降多少（吸能的位移余量）；
- 力臂 R_knee/L、R_ankle/L（120° 姿态）：地反力对膝、踝的力臂 ÷ 总长（Biewener 的 EMA 分母部分）；
- 几何刚度 log10 G，G = L² / (R_knee² + R_ankle²)：各关节扭转刚度相同时，腿的竖直刚度 ∝ G（大 = 腿硬、省力矩；小 = 腿软、吸能行程大）。
分析：PGLS（Pagel λ，Hackett 第 21 棵树）：几何量 ~ 生活方式类别（运动方式 7 类；觅食层）；再加相对腿长作协变量，看比例是否带有总长之外的信号。
似然比检验给类别整体的 p 值。另给各类的原始均值。
在 EvoWM 根目录运行：python scripts/exploratory/mech_landscape_v0.py；结果 outputs/mech_v0/report.md、species_indices.csv。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from evowm.data.tree import read_trees  # noqa: E402
from evowm.eval.pgls import pgls_lambda, tip_vcv  # noqa: E402
from evowm.sim.regimes import regime_forstrat, regime_locomotor  # noqa: E402

OUT = ROOT / "outputs/mech_v0"
BONES = ["k_femur", "k_tibiotarsus", "k_tarsometatarsus"]
POSTURES = {"ext": (150.0, 150.0), "std": (120.0, 120.0), "crouch": (90.0, 90.0)}
INDICES = ["rel_leg", "p_f", "p_t", "p_m", "H_std", "crouch", "R_knee", "R_ankle", "logG"]
IDX_ZH = {"rel_leg": "相对腿长", "p_f": "股骨比例", "p_t": "胫跗骨比例", "p_m": "跗跖骨比例", "H_std": "站姿高度 H/L",
          "crouch": "下蹲行程 /L", "R_knee": "膝力臂 /L", "R_ankle": "踝力臂 /L", "logG": "几何刚度 log10 G"}


def posture(l1, l2, l3, knee_deg, ankle_deg):
    """髋在趾尖正上方时的各关节位置。返回 (H, x_ankle, x_knee)，x 为前后位置（前为正），单位同 l。
    方向角从竖直量起：跗跖骨 a3 < 0（踝在趾后），胫跗骨 a2 = a3 + (π − 踝角) > a3（膝在踝前），股骨 a1 = a2 − (π − 膝角)。
    在 a3 ∈ (−π/2, 0] 上用网格找 x_hip = 0 的根（取离 0 最近的变号点），再线性插值。"""
    tk, ta = np.deg2rad(knee_deg), np.deg2rad(ankle_deg)
    a3 = np.linspace(-np.pi / 2 + 1e-3, 0.0, 1501)
    a2 = a3 + (np.pi - ta)
    a1 = a2 - (np.pi - tk)
    l1, l2, l3 = (np.asarray(v)[:, None] for v in (l1, l2, l3))
    x_hip = l3 * np.sin(a3) + l2 * np.sin(a2) + l1 * np.sin(a1)             # (n, grid)
    s = np.sign(x_hip)
    n = x_hip.shape[0]
    H = np.full(n, np.nan)
    xa = np.full(n, np.nan)
    xk = np.full(n, np.nan)
    for i in range(n):
        ch = np.flatnonzero(s[i, :-1] * s[i, 1:] <= 0)
        if len(ch) == 0:
            continue
        j = ch[-1]                                                           # 离 a3 = 0 最近的根
        w = x_hip[i, j] / (x_hip[i, j] - x_hip[i, j + 1]) if x_hip[i, j] != x_hip[i, j + 1] else 0.0
        r3 = a3[j] + w * (a3[j + 1] - a3[j])
        r2, r1 = r3 + (np.pi - ta), r3 + (np.pi - ta) - (np.pi - tk)
        H[i] = l3[i, 0] * np.cos(r3) + l2[i, 0] * np.cos(r2) + l1[i, 0] * np.cos(r1)
        xa[i] = l3[i, 0] * np.sin(r3)
        xk[i] = xa[i] + l2[i, 0] * np.sin(r2)
    return H, xa, xk


def indices(df: pd.DataFrame) -> pd.DataFrame:
    l1, l2, l3 = (10 ** df[b].to_numpy(float) for b in BONES)
    L = l1 + l2 + l3
    out = pd.DataFrame(index=df.index)
    logL = np.log10(L)
    m = df["m_mass"].to_numpy(float)
    b = np.polyfit(m, logL, 1)[0]
    out["rel_leg"] = logL - b * m
    out["p_f"], out["p_t"], out["p_m"] = l1 / L, l2 / L, l3 / L
    H = {}
    for k, (kd, ad) in POSTURES.items():
        H[k], xa, xk = posture(l1, l2, l3, kd, ad)
        if k == "std":
            out["R_ankle"], out["R_knee"] = np.abs(xa) / L, np.abs(xk) / L
    out["H_std"] = H["std"] / L
    out["crouch"] = (H["ext"] - H["crouch"]) / L
    out["logG"] = np.log10(1.0 / (out["R_knee"] ** 2 + out["R_ankle"] ** 2))
    out.attrs["slope"] = float(b)
    return out


def pgls_class(y, groups: pd.Series, C, base: str, covar=None):
    """y ~ 类别哑变量（+ 协变量）；返回系数表、λ、与只含截距（+协变量）模型的似然比检验 p。"""
    cats = [c for c in groups.unique() if c != base]
    X = [np.ones(len(y))] + [(groups == c).to_numpy(float) for c in cats]
    names = ["intercept"] + list(cats)
    X0 = [np.ones(len(y))]
    if covar is not None:
        X.append(covar)
        X0.append(covar)
        names.append("rel_leg")
    full = pgls_lambda(y, np.column_stack(X), C, names)
    null = pgls_lambda(y, np.column_stack(X0), C)
    lr = 2 * (full["loglik"] - null["loglik"])
    return full, 1 - chi2.cdf(max(lr, 0.0), df=len(cats))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t = pd.read_csv(ROOT / "data/processed/species_traits.csv")
    ok = t[BONES + ["m_mass"]].notna().all(axis=1)
    d = t.loc[ok].copy()
    d["loco"] = regime_locomotor(t)[ok].astype(str)
    d["fs"] = regime_forstrat(t)[ok].astype(str)
    d["is_passerine"] = d["order"] == "Passeriformes"
    idx = indices(d)
    d = pd.concat([d[["tip", "family", "order", "m_mass", "m_tarsus", "loco", "fs", "is_passerine"] + BONES], idx], axis=1)
    d.to_csv(OUT / "species_indices.csv", index=False)

    tree = read_trees(ROOT / "data/processed/trees/hackett_t021.nwk", max_trees=1)[0]
    pos = {n: i for i, n in enumerate(tree.tip_names)}
    rows = np.array([pos[x] for x in d["tip"]])
    C = tip_vcv(tree)[np.ix_(rows, rows)]

    L = ["# 探索 v0：三段腿骨比例的几何力学量与生活方式", "",
         "脚本 `scripts/exploratory/mech_landscape_v0.py`。探索性分析，不判定。", "",
         f"- 物种 {len(d)}（雀形目 {int(d.is_passerine.sum())}，其余 {int((~d.is_passerine).sum())}，几乎都是雁形目）。"
         f"腿长对体重的斜率 b = {idx.attrs['slope']:.3f}。",
         "- 腿当作平面三连杆，髋在趾尖正上方；几何量的定义见脚本开头。几何刚度 G 大表示同样的关节刚度下腿更硬、更省力矩；"
         "小表示腿更软、吸能位移余量更大。", ""]

    def section(title, groups, base, sub):
        nonlocal L
        dd = d[sub]
        g = groups[sub]
        keep = g.isin(g.value_counts()[g.value_counts() >= 15].index)
        dd, g = dd[keep], g[keep]
        Cs = C[np.ix_(np.flatnonzero(sub)[keep.to_numpy()], np.flatnonzero(sub)[keep.to_numpy()])]
        L += [f"## {title}", "", f"物种 {len(dd)}；类别（≥ 15 种）：" +
              "、".join(f"{c} {n}" for c, n in g.value_counts().items()) + f"；基准类 {base}。", ""]
        L += ["**各类原始均值**", "", "| 类别 | n | " + " | ".join(IDX_ZH[k] for k in INDICES) + " |",
              "|---|---|" + "---|" * len(INDICES)]
        for c, sub_d in dd.groupby(g):
            L.append(f"| {c} | {len(sub_d)} | " + " | ".join(f"{sub_d[k].mean():.3f}" for k in INDICES) + " |")
        L += ["", "**PGLS：几何量 ~ 类别**（系数为与基准类的差，括号为标准误；λ 为系统发育信号；p 为类别整体的似然比检验）", "",
              "| 几何量 | λ | p | " + " | ".join(c for c in g.unique() if c != base) + " |",
              "|---|---|---|" + "---|" * (g.nunique() - 1)]
        cats = [c for c in g.unique() if c != base]
        for k in INDICES:
            y = dd[k].to_numpy(float)
            full, p = pgls_class(y, g, Cs, base)
            cells = [f"{full['coef'][c]['beta']:+.3f} ({full['coef'][c]['se']:.3f})" for c in cats]
            L.append(f"| {IDX_ZH[k]} | {full['lambda']:.2f} | {p:.2g} | " + " | ".join(cells) + " |")
        L += ["", "**PGLS：几何量 ~ 类别 + 相对腿长**（比例是否带有总长之外的信号）", "",
              "| 几何量 | λ | p（类别） | " + " | ".join(cats) + " | 相对腿长 |", "|---|---|---|" + "---|" * (len(cats) + 1)]
        for k in INDICES:
            if k == "rel_leg":
                continue
            y = dd[k].to_numpy(float)
            full, p = pgls_class(y, g, Cs, base, covar=dd["rel_leg"].to_numpy(float))
            cells = [f"{full['coef'][c]['beta']:+.3f} ({full['coef'][c]['se']:.3f})" for c in cats]
            L.append(f"| {IDX_ZH[k]} | {full['lambda']:.2f} | {p:.2g} | " + " | ".join(cells) +
                     f" | {full['coef']['rel_leg']['beta']:+.3f} ({full['coef']['rel_leg']['se']:.3f}) |")
        L.append("")

    allm = np.ones(len(d), bool)
    section("1. 全部物种，运动方式", d["loco"], "insessorial", allm)
    pas = d["is_passerine"].to_numpy()
    section("2. 雀形目，运动方式", d["loco"], "insessorial", pas)
    fs_ok = pas & (d["fs"] != "<NA>").to_numpy() & (d["fs"] != "mixed").to_numpy()
    section("3. 雀形目，主导觅食层", d["fs"], "understory", fs_ok)
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"写入 {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
