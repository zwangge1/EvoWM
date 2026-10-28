"""事后探索（M1、M2 跑完之后才写，不进判据）：腿对温度的关系在 AVONET 外测跗跖上长什么样。

问题：M1 在 Skelevision 的 1,727 种（几乎全是雀形目）里没看到跗跖骨随温度变长。这和 Fröhlich 2023 用 AVONET 跗跖报告的
Allen 定律是否矛盾？分三步看：同一批物种的外测跗跖；全部 AVONET 实测种；按雀形目与非雀形目拆开。
协变量与 M1 相同（log10 体重、生活方式、栖境密度、迁徙等级、纬度跨度），可选加 EltonTraits 的地面觅食与林下觅食比例。
在 EvoWM 根目录运行：OMP_NUM_THREADS=2 python scripts/exploratory/allen_posthoc_v0.py；结果 outputs/allen_v1/posthoc.md。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from evowm.data.tree import read_trees  # noqa: E402
from evowm.eval.pgls import pgls_lambda, tip_vcv  # noqa: E402

LIFE = ["aerial", "aquatic", "generalist", "terrestrial"]
OUT = ROOT / "outputs/allen_v1"


def X_of(d, temp, extra):
    cols = {"i": np.ones(len(d)), "M": d["m_mass"].values, "T": d[temp].values / 10.0, "habdens": d["e_habdens"].values,
            "migr": d["e_migr"].values, "span": d["lat_span"].values / 10.0}
    for k in LIFE:
        v = d[f"e_life_{k}"].values.astype(float)
        if v.std() > 0:
            cols[k] = v
    for k in extra:
        cols[k] = d[k].values.astype(float)
    return np.column_stack(list(cols.values())), list(cols)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t = pd.read_csv(ROOT / "data/processed/species_traits.csv")
    c = pd.read_csv(ROOT / "data/processed/species_climate.csv")
    tree = read_trees(ROOT / "data/processed/trees/hackett_t021.nwk", max_trees=1)[0]
    pos = {n: i for i, n in enumerate(tree.tip_names)}
    C_all = tip_vcv(tree)
    a = t.merge(c, on="tip", how="inner")
    a["lat_span"] = a["lat_max"] - a["lat_min"]
    a["abslat"] = a["lat_c"].abs()
    a = a[(a["avonet_inferred"] == 0) & a["tip"].isin(pos)]
    skel = a[a[["k_femur", "k_tibiotarsus", "k_tarsometatarsus"]].notna().all(axis=1)]
    P, N = a[a["order"] == "Passeriformes"], a[a["order"] != "Passeriformes"]
    rows_out = []

    def run(d, label, resp, temp="bio1_c", extra=()):
        need = [resp, "m_mass", "e_habdens", "e_migr", "lat_span", temp] + list(extra)
        d = d.dropna(subset=need).reset_index(drop=True)
        rows = np.array([pos[x] for x in d["tip"]])
        X, names = X_of(d, temp, extra)
        res = pgls_lambda(d[resp].to_numpy(float), X, C_all[np.ix_(rows, rows)], names)
        b, se = res["coef"]["T"]["beta"], res["coef"]["T"]["se"]
        ex = "；".join(f"{k} {res['coef'][k]['beta']:+.3f} ({res['coef'][k]['se']:.3f})" for k in extra)
        rows_out.append(f"| {label} | {len(d)} | {res['lambda']:.2f} | {b:+.4f} [{b - 1.96 * se:+.4f}, {b + 1.96 * se:+.4f}] | {ex} |")
        print(rows_out[-1])

    run(skel, "Skelevision 物种，外测跗跖（AVONET）", "m_tarsus")
    run(skel, "Skelevision 物种，骨骼跗跖骨", "k_tarsometatarsus")
    run(a, "全部 AVONET 实测种，外测跗跖", "m_tarsus")
    run(P, "雀形目，外测跗跖", "m_tarsus")
    run(P, "雀形目，外测跗跖 + 地面觅食 + 林下觅食比例", "m_tarsus", extra=("e_fs_ground", "e_fs_understory"))
    run(P, "雀形目，外测跗跖，最热月最高温", "m_tarsus", temp="bio5_c")
    run(P, "雀形目，外测跗跖，最冷月最低温", "m_tarsus", temp="bio6_c")
    run(P, "雀形目，外测跗跖，质心纬度绝对值代替温度（每 10°）", "m_tarsus", temp="abslat")
    run(P, "雀形目，翅长（羽，对照）", "m_wing")
    run(P, "雀形目，第一枚次级飞羽长（对照）", "m_secondary1")
    run(N, "非雀形目，外测跗跖", "m_tarsus")
    run(N, "非雀形目，外测跗跖 + 地面觅食比例", "m_tarsus", extra=("e_fs_ground",))
    run(N, "非雀形目，翅长（羽，对照）", "m_wing")

    L = ["# 事后探索：外测跗跖与温度，雀形目与非雀形目拆开", "",
         "脚本 `scripts/exploratory/allen_posthoc_v0.py`。M1、M2 之后才做，不进判据。系数是温度每升高 10 °C 的 log10 变化（0.0043 = 1%），"
         "方括号为 PGLS 标准误的 95% 区间；协变量同 M1。", "",
         "| 模型 | n | λ | 温度系数 | 额外协变量 |", "|---|---|---|---|---|"] + rows_out + [""]
    (OUT / "posthoc.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
