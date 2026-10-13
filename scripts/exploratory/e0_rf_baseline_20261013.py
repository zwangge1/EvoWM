"""探索性基线：形态预测习性（随机森林，平衡准确率；随机 5 折 vs 科分块 5 折）。

2026-10-13 选方向用，未预注册，结果不进论文。数字见 docs/context/数据盘点_A还是B_2026-10-13.md。
原运行环境：scikit-learn 1.9.1，pandas 3.0.5，numpy 2.5.3。
先运行 e0_export_avonet_sheets.py 生成 avonet3_birdtree.csv。
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[2]
a = pd.read_csv(ROOT / "data/interim/exploratory_20261013/avonet3_birdtree.csv")
fd = pd.read_csv(ROOT / "data/raw/eltontraits_derived/forstrat_dom.csv")

a = a[(a["Total.individuals"] > 0) & (a.Inference == "NO")].copy()
a["lm"] = np.log10(a["Mass"])


def res(c):
    """对 log 体重做 OLS 后的残差，即相对性状。"""
    y = np.log10(a[c])
    b = np.polyfit(a.lm, y, 1)
    return y - np.polyval(b, a.lm)


for c in ["Tarsus.Length", "Wing.Length", "Tail.Length",
          "Beak.Length_Culmen", "Beak.Depth", "Beak.Width"]:
    a["r_" + c] = res(c)
a["HWI"] = a["Hand-Wing.Index"]

sets = {
    "size": ["lm"],
    "leg": ["lm", "r_Tarsus.Length"],
    "leg+wing": ["lm", "r_Tarsus.Length", "HWI", "r_Wing.Length", "r_Tail.Length"],
    "all": ["lm", "r_Tarsus.Length", "HWI", "r_Wing.Length", "r_Tail.Length",
            "r_Beak.Length_Culmen", "r_Beak.Depth", "r_Beak.Width"],
}

a["sp"] = a.Species3
fd["sp"] = fd.scientificNameStd
a = a.merge(fd[["sp", "dom_strat"]], on="sp", how="left")

for tgt in ["Primary.Lifestyle", "Trophic.Niche", "dom_strat"]:
    d = a.dropna(subset=[tgt] + sets["all"])
    d = d[d[tgt] != "mixed"]
    vc = d[tgt].value_counts()
    d = d[d[tgt].isin(vc[vc >= 30].index)]
    print(f"## {tgt} n={len(d)} k={d[tgt].nunique()}")
    for nm, cols in sets.items():
        out = []
        for cvn, cv, g in [("random", StratifiedKFold(5, shuffle=True, random_state=0), None),
                           ("family-block", GroupKFold(5), d.Family3)]:
            clf = RandomForestClassifier(300, min_samples_leaf=3, class_weight="balanced",
                                         n_jobs=-1, random_state=0)
            p = cross_val_predict(clf, d[cols], d[tgt], cv=cv, groups=g)
            out.append(f"{cvn} {balanced_accuracy_score(d[tgt], p):.3f}")
        print(f"  {nm:9s}", " | ".join(out))
