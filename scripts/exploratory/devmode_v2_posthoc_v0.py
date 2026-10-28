"""事后探索：按四档发育方式分别看 AVONET 跗跖与 Eliason 三段骨的温度斜率；留巢期剂量在 A 与 SA 内。"""
import sys, numpy as np, pandas as pd
from pathlib import Path
ROOT0 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT0 / "scripts/allen")); sys.path.insert(0, str(ROOT0 / "src"))
from devmode_v2 import base, load_cooney, load_eliason, fit, ci, ROOT
from evowm.data.tree import read_trees
from evowm.eval.pgls import tip_vcv
d = base(); tree = read_trees(ROOT / "data/processed/trees/hackett_t021.nwk", max_trees=1)[0]
pos = {n: i for i, n in enumerate(tree.tip_names)}; C_all = tip_vcv(tree)
d = d[d.tip.isin(pos)].reset_index(drop=True); tips = set(d.tip)
d = d.merge(load_cooney(tips), on="tip", how="left").merge(load_eliason(tips), on="tip", how="left")
d["mode4"] = d["mode"]; d.loc[d.order == "Passeriformes", "mode4"] = "A_pass"
print("Eliason 非雀形目各模式物种数", d[(d.order!="Passeriformes") & d.b_tmt.notna()]["mode"].value_counts().to_dict())
print("\n四档发育方式（Eliason 骨长，非雀形目 + 雀形目）：")
for m in ["P", "SP", "SA", "A", "A_pass"]:
    sub = d[(d.mode4 == m) & d.b_tmt.notna()].reset_index(drop=True)
    if len(sub) < 30: print(m, "n =", len(sub), "跳过"); continue
    out = {k: fit(sub, C_all, pos, k) for k in ["b_fem", "b_tbt", "b_tmt", "b_r", "b_hum"]}
    print(f"{m:7s} n={len(sub):4d} 股骨 {ci(out['b_fem']['T_b'], out['b_fem']['T_se'])} 胫跗 {ci(out['b_tbt']['T_b'], out['b_tbt']['T_se'])} 跗跖骨 {ci(out['b_tmt']['T_b'], out['b_tmt']['T_se'])} r {ci(out['b_r']['T_b'], out['b_r']['T_se'])} 肱骨 {ci(out['b_hum']['T_b'], out['b_hum']['T_se'])}")
print("\n留巢期剂量（T × log10 留巢期，中心化），AVONET 跗跖：")
for m in ["A", "SA", "A_pass"]:
    sub = d[(d.mode4 == m) & d.log_fledging_d.notna() & d.m_tarsus.notna() & (d.avonet_inferred == 0)].reset_index(drop=True)
    sub["lfl"] = sub.log_fledging_d - sub.log_fledging_d.mean()
    r = fit(sub, C_all, pos, "m_tarsus", group_col="lfl")
    print(f"{m:7s} n={r['n']:4d} T×留巢期 {ci(r['diff_b'], r['diff_se'])}；留巢期中位 {10**sub.log_fledging_d.median():.0f} 天")
print("\n巢型在非雀形目 A 与 SA 内（开放 对 洞巢+封闭）：")
for m in ["A", "SA"]:
    sub = d[(d["mode"] == m) & (d.order != "Passeriformes") & d.open_nest.notna() & d.m_tarsus.notna() & (d.avonet_inferred == 0)].reset_index(drop=True) if "open_nest" in d else None
    if sub is None:
        d["open_nest"] = np.where(d.nest_type == "open", 1.0, np.where(d.nest_type.isin(["cavity", "closed"]), 0.0, np.nan))
        sub = d[(d["mode"] == m) & (d.order != "Passeriformes") & d.open_nest.notna() & d.m_tarsus.notna() & (d.avonet_inferred == 0)].reset_index(drop=True)
    if sub.open_nest.nunique() < 2 or len(sub) < 60: print(m, "n =", len(sub), "跳过"); continue
    r = fit(sub, C_all, pos, "m_tarsus", group_col="open_nest")
    print(f"{m:3s} n={r['n']}（开放 {r['n_g1']}） 封闭 {ci(r['g0_b'], r['g0_se'])} 开放 {ci(r['g1_b'], r['g1_se'])} 之差 {ci(r['diff_b'], r['diff_se'])}")
