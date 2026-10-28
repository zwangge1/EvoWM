"""事后探索（不进判据）：发育方式检验之后，逐目看温度斜率（跗跖与翅长），生活方式子集内与目内（鸻形目、夜鹰目）的暴露对巢内对比。
在 EvoWM 根目录运行：OMP_NUM_THREADS=2 python scripts/exploratory/devmode_posthoc_v0.py；输出打印到终端。
"""

import sys, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/allen')); sys.path.insert(0, str(ROOT / 'src'))
from devmode_pgls_v1 import load, fit, ci, ROOT
from evowm.data.tree import read_trees
from evowm.eval.pgls import pgls_lambda, tip_vcv
d = load()
tree = read_trees(ROOT / "data/processed/trees/hackett_t021.nwk", max_trees=1)[0]
pos = {n: i for i, n in enumerate(tree.tip_names)}; C_all = tip_vcv(tree)
d = d[d.tip.isin(pos)].reset_index(drop=True)
NP = d[d.order != "Passeriformes"].reset_index(drop=True)
for life in ["aquatic", "terrestrial", "generalist", "insessorial", "aerial"]:
    sub = NP[NP[f"e_life_{life}"] == 1].reset_index(drop=True)
    if sub.exposed.nunique() < 2 or sub.exposed.sum() < 30 or (sub.exposed == 0).sum() < 30:
        print(f"{life}: n={len(sub)} 暴露 {int(sub.exposed.sum())}，跳过"); continue
    r = fit(sub, C_all, pos)
    print(f"{life}: n={r['n']} 暴露 {r['n_exposed']} λ={r['lambda']:.2f} 暴露组 {ci(r['exp_b'], r['exp_se'])} 巢内组 {ci(r['nest_b'], r['nest_se'])} 之差 {ci(r['diff_b'], r['diff_se'])}")
print("\n各目单独（≥ 40 种）：温度斜率")
rows = []
for o, sub in NP.groupby("order"):
    if len(sub) < 40: continue
    sub = sub.reset_index(drop=True)
    rr = np.array([pos[x] for x in sub.tip]); C = C_all[np.ix_(rr, rr)]
    T = sub.bio1_c.values / 10
    cols = {"i": np.ones(len(sub)), "M": sub.m_mass.values, "T": T, "habdens": sub.e_habdens.values, "migr": sub.e_migr.values, "span": (sub.lat_max - sub.lat_min).values / 10}
    lk = []
    for k in ["aerial", "aquatic", "generalist", "terrestrial"]:
        v = sub[f"e_life_{k}"].values.astype(float)
        if v.std() > 0: cols[k] = v; lk.append(k)
    if lk and np.all(sum(cols[k] for k in lk) == 1): cols.pop(lk[-1])
    X = np.column_stack(list(cols.values()))
    keep = [i for i in range(X.shape[1]) if i == 0 or X[:, i].std() > 0]
    X = X[:, keep]; names = [list(cols)[i] for i in keep]
    if np.linalg.matrix_rank(X) < X.shape[1]:
        X = np.column_stack([cols["i"], cols["M"], cols["T"], cols["span"]]); names = ["i", "M", "T", "span"]
    res = pgls_lambda(sub.m_tarsus.values.astype(float), X, C, names)
    b, se = res["coef"]["T"]["beta"], res["coef"]["T"]["se"]
    rows.append((o, len(sub), sub["mode"].mode()[0], b, se))
for o, n, m, b, se in sorted(rows, key=lambda x: -x[3]):
    print(f"{o:20s} n={n:4d} {m:3s} {ci(b, se)}")

ch = d[d.order == "Charadriiformes"].reset_index(drop=True)
print("鸻形目 各模式物种数", ch["mode"].value_counts().to_dict())
r = fit(ch, C_all, pos); print(f"鸻形目 暴露(P+SP) 对 巢内(SA): n={r['n']} 暴露 {r['n_exposed']} 暴露组 {ci(r['exp_b'], r['exp_se'])} 巢内组 {ci(r['nest_b'], r['nest_se'])} 之差 {ci(r['diff_b'], r['diff_se'])}")
ch2 = ch.copy(); ch2["exposed"] = (ch2["mode"] == "P").astype(int)
r = fit(ch2, C_all, pos); print(f"鸻形目 P 对 SP+SA: 暴露组 {ci(r['exp_b'], r['exp_se'])} 其余 {ci(r['nest_b'], r['nest_se'])} 之差 {ci(r['diff_b'], r['diff_se'])}")
cap = d[d.order == "Caprimulgiformes"].reset_index(drop=True)
print("夜鹰目 各模式", cap["mode"].value_counts().to_dict())
r = fit(cap, C_all, pos); print(f"夜鹰目 SP 对 SA+A: n={r['n']} 暴露 {r['n_exposed']} 暴露组 {ci(r['exp_b'], r['exp_se'])} 巢内组 {ci(r['nest_b'], r['nest_se'])} 之差 {ci(r['diff_b'], r['diff_se'])}")
# 翅长各目
print("\n翅长（羽）各目温度斜率，与跗跖对照：")
from evowm.eval.pgls import pgls_lambda
NP = d[d.order != "Passeriformes"]
for o, sub in NP.groupby("order"):
    if len(sub) < 40: continue
    sub = sub.reset_index(drop=True); rr = np.array([pos[x] for x in sub.tip]); C = C_all[np.ix_(rr, rr)]
    X = np.column_stack([np.ones(len(sub)), sub.m_mass.values, sub.bio1_c.values / 10, (sub.lat_max - sub.lat_min).values / 10])
    out = []
    for resp in ["m_tarsus", "m_wing"]:
        res = pgls_lambda(sub[resp].values.astype(float), X, C, ["i", "M", "T", "span"]); b, se = res["coef"]["T"]["beta"], res["coef"]["T"]["se"]
        out.append(f"{b:+.4f}±{se:.4f}")
    print(f"{o:20s} n={len(sub):4d} {sub['mode'].mode()[0]:3s} 跗跖 {out[0]}  翅长 {out[1]}")
