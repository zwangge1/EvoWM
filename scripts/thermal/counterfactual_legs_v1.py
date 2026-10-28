"""热阀反事实 v1 第二步：真实腿 R 与去散热腿 C（设计 docs/design/热阀与落震_反事实闭环_设计_2026-10-28.md 第 2 节）。

两条真实腿：露天组 = Eliason 早成鸟 P（非雀形目）三段与体重都有的物种，各列 log10 的中位数；
树栖组 = Eliason 雀形目里成鸟暴露 X = 0（地面、水面、水下觅食比例都为 0）的物种，同样取中位数。
去散热腿：每段按设计第 2.2 节的斜率回推 ΔT = 10、20、40 °C（主口径三段都回推；次口径只回推跗跖骨，ΔT = 20）。
体重固定。命名例子：绿头鸭（Watanabe 骨长，species_traits 的 k_ 列），只用于画图。
在 EvoWM 根目录运行：python scripts/thermal/counterfactual_legs_v1.py
输出 outputs/thermal_v1/legs.json。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/allen"))
from devmode_v2 import base, load_eliason  # noqa: E402

DESIGN = ROOT / "docs/design/热阀与落震_反事实闭环_设计_2026-10-28.md"
OUT = ROOT / "outputs/thermal_v1/legs.json"
SEG = ["b_tmt", "b_tbt", "b_fem"]          # Leg(tmt, tbt, fem) 的顺序
DT = [10, 20, 40]
# 设计第 2.2 节：每 10 °C 的 log10 斜率（跗跖骨、胫跗骨、股骨）
SLOPES = {"exposed": {"b_tmt": 0.0161, "b_tbt": 0.0084, "b_fem": 0.0002},
          "arboreal": {"b_tmt": -0.0051, "b_tbt": -0.0042, "b_fem": 0.0021}}
T_REAL = 25.0


def medians(d):
    return {c: float(d[c].median()) for c in SEG + ["m_mass"]}


def shift(log10, slopes, dT, only_tmt=False):
    out = dict(log10)
    for c in SEG:
        if only_tmt and c != "b_tmt":
            continue
        out[c] = log10[c] - slopes[c] * dT / 10.0
    return out


def mm(log10):
    return {c: float(10 ** log10[c]) for c in SEG}


def main():
    if not DESIGN.exists():
        sys.exit(f"缺少设计文档 {DESIGN.relative_to(ROOT)}")
    d = base()
    e = load_eliason(set(d["tip"]))
    d = d.merge(e, on="tip", how="inner").dropna(subset=SEG + ["m_mass"])
    d["X"] = d[["e_fs_ground", "e_fs_wataroundsurf", "e_fs_watbelowsurf"]].sum(axis=1, min_count=3)
    groups = {
        "exposed": d[(d["order"] != "Passeriformes") & (d["mode"] == "P")],
        "arboreal": d[(d["order"] == "Passeriformes") & (d["X"] == 0)],
    }
    res = {"design": str(DESIGN.relative_to(ROOT)), "T_real": T_REAL, "deltas": DT, "slopes_per_10C": SLOPES, "groups": {}}
    for g, sub in groups.items():
        R = medians(sub)
        legs = {"R": R}
        for dT in DT:
            legs[f"C{dT}"] = shift(R, SLOPES[g], dT)
        legs["Ct20"] = shift(R, SLOPES[g], 20, only_tmt=True)
        res["groups"][g] = {
            "n_species": int(len(sub)), "body_mass_g": float(10 ** R["m_mass"]),
            "legs_log10": legs, "legs_mm": {k: mm(v) for k, v in legs.items()},
            "change_pct_vs_R": {k: {c: float(100 * (10 ** (v[c] - R[c]) - 1)) for c in SEG} for k, v in legs.items() if k != "R"},
        }
    t = pd.read_csv(ROOT / "data/processed/species_traits.csv")
    ex = t[t["tip"] == "Anas_platyrhynchos"]
    if len(ex) and ex[["k_tarsometatarsus", "k_tibiotarsus", "k_femur", "m_mass"]].notna().all(axis=1).item():
        r = ex.iloc[0]
        log10 = {"b_tmt": float(r["k_tarsometatarsus"]), "b_tbt": float(r["k_tibiotarsus"]), "b_fem": float(r["k_femur"]),
                 "m_mass": float(r["m_mass"])}
        res["example_mallard"] = {"tip": "Anas_platyrhynchos", "legs_log10": {"R": log10, "C20": shift(log10, SLOPES["exposed"], 20)},
                                  "legs_mm": {"R": mm(log10), "C20": mm(shift(log10, SLOPES["exposed"], 20))},
                                  "body_mass_g": float(10 ** log10["m_mass"])}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for g, v in res["groups"].items():
        print(g, v["n_species"], "种，体重", round(v["body_mass_g"], 1), "g")
        for k, m in v["legs_mm"].items():
            print(f"  {k}: 跗跖骨 {m['b_tmt']:.1f} 胫跗骨 {m['b_tbt']:.1f} 股骨 {m['b_fem']:.1f} mm")
    if "example_mallard" in res:
        print("绿头鸭", res["example_mallard"]["legs_mm"])


if __name__ == "__main__":
    main()
