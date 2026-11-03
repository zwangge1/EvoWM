"""体型还是暴露：T × M 与 T × X 的同场比较（设计 docs/design/体型还是暴露_同场比较_设计_2026-10-30.md）。

模型族 M0 到 M3，样本 S_all、S_nonpass、S_pass，PGLS（Pagel λ 最大似然，Hackett 第 21 棵树），AIC 比较；
逐目留一；稳健性换温度变量。判据见设计第 3、4 节，运行前定下。
在 EvoWM 根目录运行：OMP_NUM_THREADS=2 python scripts/allen/mass_vs_exposure_v1.py
输出 outputs/allen_v1/mass_vs_exposure.md、mass_vs_exposure.json。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/allen"))
from devmode_v2 import LIFE, base  # noqa: E402
from evowm.data.tree import read_trees  # noqa: E402
from evowm.eval.pgls import pgls_lambda, tip_vcv  # noqa: E402

DESIGN = ROOT / "docs/design/体型还是暴露_同场比较_设计_2026-10-30.md"
OUT = ROOT / "outputs/allen_v1"
MIN_EFFECT = 0.0043
MIN_ORDER = 40
MASS_Q = {"10 g": 1.0, "35 g": np.log10(35), "150 g": np.log10(150), "1 kg": 3.0}


def covars(d):
    cols = {"habdens": d["e_habdens"].to_numpy(float), "migr": d["e_migr"].to_numpy(float), "lat_span": d["lat_span"].to_numpy(float) / 10.0}
    for k in LIFE:
        v = d[f"e_life_{k}"].to_numpy(float)
        if v.std() > 0:
            cols[f"life_{k}"] = v
    lk = [k for k in cols if k.startswith("life_")]
    if lk and np.all(sum(cols[k] for k in lk) == 1):
        cols.pop(lk[-1])
    return cols


def design(d, model, temp, m_center):
    T = d[temp].to_numpy(float) / 10.0
    M = d["m_mass"].to_numpy(float) - m_center
    X = d["X"].to_numpy(float)
    cols = {"intercept": np.ones(len(d)), "M": M, "T": T}
    if model in ("M1", "M3"):
        cols["TxM"] = T * M
    if model in ("M2", "M3"):
        cols["X"] = X
        cols["TxX"] = T * X
    cols.update(covars(d))
    names = list(cols)
    return np.column_stack([cols[k] for k in names]), names


def fit(d, C_all, pos, model, temp, m_center):
    rows = np.array([pos[x] for x in d["tip"]])
    C = C_all[np.ix_(rows, rows)]
    Xd, names = design(d, model, temp, m_center)
    res = pgls_lambda(d["m_tarsus"].to_numpy(float), Xd, C, names)
    k = Xd.shape[1] + 2  # 系数 + σ² + λ
    out = {"n": res["n"], "lambda": res["lambda"], "loglik": res["loglik"], "aic": 2 * k - 2 * res["loglik"],
           "coef": {n: [res["coef"][n]["beta"], res["coef"][n]["se"]] for n in ("T", "TxM", "X", "TxX") if n in res["coef"]}}
    return out


def ci(b, se):
    return f"{b:+.4f} [{b - 1.96 * se:+.4f}, {b + 1.96 * se:+.4f}]"


def excl0(b, se):
    return (b - 1.96 * se > 0) or (b + 1.96 * se < 0)


def slopes_at(fitres, m_center):
    """M1 与 M3 下，体重四分位处、X = 0 与 1 的温度斜率（点估计）。"""
    c = {k: v[0] for k, v in fitres["coef"].items()}
    out = {}
    for lab, lm in MASS_Q.items():
        base_ = c["T"] + c.get("TxM", 0.0) * (lm - m_center)
        out[lab] = {"X=0": base_, "X=1": base_ + c.get("TxX", 0.0)}
    return out


def main():
    if not DESIGN.exists():
        sys.exit(f"缺少设计文档 {DESIGN.relative_to(ROOT)}")
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    d = base()
    d["X"] = d[["e_fs_ground", "e_fs_wataroundsurf", "e_fs_watbelowsurf"]].sum(axis=1, min_count=3)
    d = d[(d["avonet_inferred"] == 0)].dropna(subset=["m_tarsus", "X", "bio1_c", "bio5_c", "bio6_c"]).reset_index(drop=True)
    tree = read_trees(ROOT / "data/processed/trees/hackett_t021.nwk", max_trees=1)[0]
    pos = {n: i for i, n in enumerate(tree.tip_names)}
    C_all = tip_vcv(tree)
    d = d[d["tip"].isin(pos)].reset_index(drop=True)
    is_pass = d["order"] == "Passeriformes"
    samples = {"S_all": d, "S_nonpass": d[~is_pass].reset_index(drop=True), "S_pass": d[is_pass].reset_index(drop=True)}
    m_center = float(d["m_mass"].mean())
    res = {"n": {k: int(len(v)) for k, v in samples.items()}, "m_center_log10": m_center, "models": {}, "loo": {}, "robust": {}}

    # ---- 四个模型 × 三个样本 ----
    for s, sub in samples.items():
        res["models"][s] = {}
        for m in ("M0", "M1", "M2", "M3"):
            res["models"][s][m] = fit(sub, C_all, pos, m, "bio1_c", m_center)
            print(f"{s} {m} n={res['models'][s][m]['n']} λ={res['models'][s][m]['lambda']:.2f} AIC={res['models'][s][m]['aic']:.1f} {time.time() - t0:.0f}s", flush=True)

    A = res["models"]["S_all"]
    d1 = A["M1"]["coef"]["TxM"]; d3 = A["M3"]["coef"]["TxM"]; f2 = A["M2"]["coef"]["TxX"]; f3 = A["M3"]["coef"]["TxX"]
    dAIC_21 = A["M2"]["aic"] - A["M1"]["aic"]
    dAIC_32 = A["M3"]["aic"] - A["M2"]["aic"]
    dAIC_31 = A["M3"]["aic"] - A["M1"]["aic"]
    f_ok = f3[0] > 0 and excl0(*f3) and f3[0] >= MIN_EFFECT
    d_shrunk = (abs(d3[0]) <= 0.5 * abs(d1[0])) or not excl0(*d3)
    d_ok = excl0(*d3)
    f_shrunk = (abs(f3[0]) <= 0.5 * abs(f2[0])) or not excl0(*f3)
    if dAIC_21 <= -2 and f_ok and d_shrunk:
        verdict = "暴露优先：成鸟暴露吸收了体型轴"
    elif dAIC_21 >= 2 and d_ok and f_shrunk:
        verdict = "体型优先：体型轴吸收了成鸟暴露"
    elif d_ok and excl0(*f3) and dAIC_31 <= -2 and dAIC_32 <= -2:
        verdict = "两个轴：体型与成鸟暴露各自独立起作用"
    else:
        verdict = "都不成立，按实描述"
    res["verdict"] = {"text": verdict, "dAIC_M2_minus_M1": dAIC_21, "dAIC_M3_minus_M2": dAIC_32, "dAIC_M3_minus_M1": dAIC_31,
                      "f_ok": bool(f_ok), "d_shrunk": bool(d_shrunk), "d_ratio_M3_over_M1": d3[0] / d1[0] if d1[0] else float("nan")}
    res["slopes"] = {m: slopes_at(A[m], m_center) for m in ("M1", "M3")}

    # ---- 逐目留一（S_all，M3）----
    counts = d["order"].value_counts()
    for o in counts[counts >= MIN_ORDER].index:
        sub = d[d["order"] != o].reset_index(drop=True)
        r = fit(sub, C_all, pos, "M3", "bio1_c", m_center)
        res["loo"][o] = {"n_dropped": int(counts[o]), "n": r["n"], "TxX": r["coef"]["TxX"], "TxM": r["coef"]["TxM"]}
        print(f"留一 {o}: TxX {ci(*r['coef']['TxX'])} {time.time() - t0:.0f}s", flush=True)
    fs = [v["TxX"][0] for v in res["loo"].values()]
    loo_ok = all(excl0(*v["TxX"]) for v in res["loo"].values()) and min(fs) >= 0.5 * f3[0] and max(fs) <= 1.5 * f3[0]
    res["loo_verdict"] = {"robust": bool(loo_ok), "min": min(fs), "max": max(fs), "full": f3[0],
                          "driver": max(res["loo"], key=lambda o: abs(res["loo"][o]["TxX"][0] - f3[0]))}

    # ---- 稳健性：换温度 ----
    for temp in ("bio5_c", "bio6_c", "bio1_box"):
        sub = d.dropna(subset=[temp]).reset_index(drop=True)
        res["robust"][temp] = {m: fit(sub, C_all, pos, m, temp, m_center) for m in ("M1", "M2", "M3")}
        print(f"稳健 {temp} 完成 {time.time() - t0:.0f}s", flush=True)

    (OUT / "mass_vs_exposure.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- 报告 ----
    L = ["# 体型还是暴露：T × M 与 T × X 的同场比较", "",
         f"设计 `{DESIGN.relative_to(ROOT)}`。S_all {res['n']['S_all']} 种（非雀形目 {res['n']['S_nonpass']}，雀形目 {res['n']['S_pass']}），"
         f"体重中心 $\\log_{{10}}$ = {m_center:.2f}（约 {10 ** m_center:.0f} g）。系数为每升高 10 °C 的 log10 变化，方括号为 95% 区间。", "",
         f"## 判定（S_all）：{verdict}", "",
         f"- AIC：M2 − M1 = {dAIC_21:+.1f}，M3 − M2 = {dAIC_32:+.1f}，M3 − M1 = {dAIC_31:+.1f}（负值表示前者更好）。",
         f"- 体型轴 d：M1 中 {ci(*d1)}，M3 中 {ci(*d3)}（保留 {100 * res['verdict']['d_ratio_M3_over_M1']:.0f}%）。",
         f"- 暴露轴 f：M2 中 {ci(*f2)}，M3 中 {ci(*f3)}。", "",
         "## 四个模型 × 三个样本", "",
         "| 样本 | 模型 | n | λ | AIC | T（X = 0，平均体型） | T × M | T × X |", "|---|---|---|---|---|---|---|---|"]
    for s in samples:
        for m in ("M0", "M1", "M2", "M3"):
            r = res["models"][s][m]
            L.append(f"| {s} | {m} | {r['n']} | {r['lambda']:.2f} | {r['aic']:.1f} | {ci(*r['coef']['T'])} | "
                     f"{ci(*r['coef']['TxM']) if 'TxM' in r['coef'] else ''} | {ci(*r['coef']['TxX']) if 'TxX' in r['coef'] else ''} |")
    L += ["", "## 不同体型处的温度斜率（S_all，点估计）", "", "| 体重 | M1（只有体型轴） | M3，X = 0 | M3，X = 1 |", "|---|---|---|---|"]
    for lab in MASS_Q:
        L.append(f"| {lab} | {res['slopes']['M1'][lab]['X=0']:+.4f} | {res['slopes']['M3'][lab]['X=0']:+.4f} | {res['slopes']['M3'][lab]['X=1']:+.4f} |")
    lv = res["loo_verdict"]
    L += ["", f"## 逐目留一（S_all，M3 的 T × X）：{'稳健' if lv['robust'] else '不稳健'}，全样本 {lv['full']:+.4f}，留一范围 {lv['min']:+.4f} 到 {lv['max']:+.4f}，偏离最大的是去掉 {lv['driver']}", "",
          "| 去掉的目 | 种数 | T × X | T × M |", "|---|---|---|---|"]
    for o, v in sorted(res["loo"].items(), key=lambda kv: kv[1]["TxX"][0]):
        L.append(f"| {o} | {v['n_dropped']} | {ci(*v['TxX'])} | {ci(*v['TxM'])} |")
    L += ["", "## 稳健性：换温度变量（S_all）", "", "| 温度 | 模型 | AIC | T × M | T × X |", "|---|---|---|---|---|"]
    for temp, mm in res["robust"].items():
        for m, r in mm.items():
            L.append(f"| {temp} | {m} | {r['aic']:.1f} | {ci(*r['coef']['TxM']) if 'TxM' in r['coef'] else ''} | {ci(*r['coef']['TxX']) if 'TxX' in r['coef'] else ''} |")
    (OUT / "mass_vs_exposure.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
