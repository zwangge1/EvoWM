"""种内层面：成鸟暴露 X 是否预测跗跖骨对温度的种内响应（设计 docs/design/种内成鸟暴露_设计_2026-10-27.md）。

沿用 mixed_specimens_v1 的清洗与样本；模型加 T × X（X 为物种级，地面 + 水面 + 水下觅食比例）。
判据：跗跖骨 γ > 0、区间不含 0、≥ 0.0043；股骨 γ 区间含 0 或 < 跗跖骨一半。
在 EvoWM 根目录运行：python scripts/allen/within_exposure_v1.py；结果 outputs/allen_v1/within_exposure.md。
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/allen"))
from mixed_specimens_v1 import load, subset  # noqa: E402

OUT = ROOT / "outputs/allen_v1"
MIN_EFFECT = 0.0043
RESP_ZH = {"f": "股骨", "t": "胫跗骨", "m": "跗跖骨", "r": "r = 跗跖骨 − 股骨"}


def add_X(d):
    sp = pd.read_csv(ROOT / "data/processed/species_traits.csv", usecols=["tip", "e_fs_ground", "e_fs_wataroundsurf", "e_fs_watbelowsurf"])
    sp["X"] = sp[["e_fs_ground", "e_fs_wataroundsurf", "e_fs_watbelowsurf"]].sum(axis=1, min_count=3)
    sp["Xg"] = sp["e_fs_ground"]
    return d.merge(sp[["tip", "X", "Xg"]], on="tip", how="left")


def fit(s, resp, temp="bio1", xcol="X", with_mass=False):
    dd = s.dropna(subset=[xcol] + (["M_c"] if with_mass else [])).copy()
    dd["TX"] = dd[f"T_{temp}"] * dd[xcol]
    f = f"{resp} ~ T_{temp} + TX + male + year_c" + (" + M_c" if with_mass else "")
    rec = {"resp": resp, "n": len(dd), "n_species": dd["tip"].nunique()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = smf.mixedlm(f, dd, groups=dd["tip"], re_formula=f"~T_{temp}").fit(reml=True, method=["lbfgs"])
            ok = res.converged and np.isfinite(res.bse["TX"])
            rec["re"] = "slope"
        except Exception:
            ok = False
        if not ok:
            res = smf.mixedlm(f, dd, groups=dd["tip"]).fit(reml=True, method=["lbfgs"])
            rec["re"] = "intercept"
    rec.update({"beta": float(res.params[f"T_{temp}"]), "beta_se": float(res.bse[f"T_{temp}"]),
                "gamma": float(res.params["TX"]), "gamma_se": float(res.bse["TX"])})
    return rec


def ci(b, se):
    return f"{b:+.4f} [{b - 1.96 * se:+.4f}, {b + 1.96 * se:+.4f}]"


def row(label, recs):
    g = recs["m"]
    return (f"| {label} | {g['n']}, {g['n_species']} | " + " | ".join(ci(recs[k]["gamma"], recs[k]["gamma_se"]) + ("" if recs[k]["re"] == "slope" else "（随机截距）") for k in ["f", "m", "r"]) +
            f" | {ci(g['beta'], g['beta_se'])} | {g['beta'] + g['gamma']:+.4f} |")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = add_X(load())
    tables = {}
    for sample, min_n in [("main", 5), ("main", 10), ("all_months", 5)]:
        s = subset(d, sample, min_n)
        for with_mass in [False, True]:
            for temp in (["bio1", "bio5", "bio6"] if (sample, min_n, with_mass) == ("main", 5, False) else ["bio1"]):
                tables[(sample, min_n, with_mass, temp, "X")] = {k: fit(s, k, temp, "X", with_mass) for k in ["f", "m", "r"]}
        tables[(sample, min_n, False, "bio1", "Xg")] = {k: fit(s, k, "bio1", "Xg", False) for k in ["f", "m", "r"]}
    main_t = tables[("main", 5, False, "bio1", "X")]
    gm, gf = main_t["m"], main_t["f"]
    ok_m = gm["gamma"] - 1.96 * gm["gamma_se"] > 0 and gm["gamma"] >= MIN_EFFECT
    f_lo, f_hi = gf["gamma"] - 1.96 * gf["gamma_se"], gf["gamma"] + 1.96 * gf["gamma_se"]
    ok_f = (f_lo <= 0 <= f_hi) or gf["gamma"] < 0.5 * gm["gamma"]
    if ok_m and ok_f:
        verdict = "通过：地面活动多的物种，种内跗跖骨的温度斜率更正；股骨没有这个效应。"
    elif ok_m and f_lo > 0:
        verdict = "整腿版：跗跖骨与股骨的露天效应都为正，种内不分段。"
    elif gm["gamma"] - 1.96 * gm["gamma_se"] <= 0:
        verdict = "不通过：跗跖骨的 γ 区间含 0 或为负。"
    else:
        verdict = "介于两者之间，按实描述。"
    s_main = subset(d, "main", 5)
    L = ["# 种内层面：成鸟暴露与跗跖骨的温度响应", "",
         "脚本 `scripts/allen/within_exposure_v1.py`。判据见设计文档，运行前定下。", "",
         "## 符号", "",
         "- γ：成鸟暴露 X 每增加 1（纯树栖到纯地面），种内温度斜率（每 10 °C 的 log10 变化）增加多少；0.0043 = 1%。方括号为 1.96 倍标准误。",
         "- β（X = 0）：纯树栖物种的种内斜率；X = 1 的斜率 = β + γ。协变量：性别、年份（种内中心化）；含体重版再加 log10 个体体重。",
         f"- 主样本 {s_main.tip.nunique()} 种 {len(s_main)} 件；X 的物种间分布：均值 {s_main.groupby('tip').X.first().mean():.2f}，SD {s_main.groupby('tip').X.first().std():.2f}，"
         f"X ≥ 0.5 的物种 {int((s_main.groupby('tip').X.first() >= 0.5).sum())} 种。", "",
         f"## 判定：{verdict}", "",
         "## 结果", "",
         "| 样本 | n, 种 | γ 股骨 | γ 跗跖骨 | γ r | β 跗跖骨（X = 0） | 跗跖骨 X = 1 斜率 |", "|---|---|---|---|---|---|---|",
         row("主样本，年均温", tables[("main", 5, False, "bio1", "X")]),
         row("主样本，含体重", tables[("main", 5, True, "bio1", "X")]),
         row("主样本，最热月", tables[("main", 5, False, "bio5", "X")]),
         row("主样本，最冷月", tables[("main", 5, False, "bio6", "X")]),
         row("主样本，X 只用地面觅食", tables[("main", 5, False, "bio1", "Xg")]),
         row("主样本，每种 ≥ 10 件", tables[("main", 10, False, "bio1", "X")]),
         row("全部标本不限月份", tables[("all_months", 5, False, "bio1", "X")]),
         "", "## 读法", "",
         "- 假说要的是 γ 跗跖骨 > 0、股骨接近 0。β 为负、β + γ 接近 0 或为正，就是跨物种在雀形目里看到的那条梯度在种内重现。", ""]
    (OUT / "within_exposure.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
