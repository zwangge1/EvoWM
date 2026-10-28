"""发育方式与腿的 Allen 定律 v2（设计 docs/design/发育方式与腿的Allen定律_v2_巢型与骨长_设计_2026-10-27.md）。

T1 巢型：巢内组里开放巢对洞巢 + 封闭巢，T × 开放巢 的系数（合并、雀形目、非雀形目巢内组）。
T2 骨长复现：Eliason 跗跖骨长，非雀形目，暴露组对巢内组。
T3 分段：暴露组（Eliason）三段各自的温度系数与 r = 跗跖骨 − 股骨；巢内组作对照。
探索：留巢期连续变量；半早成单独；肱骨。
在 EvoWM 根目录运行：OMP_NUM_THREADS=2 python scripts/allen/devmode_v2.py；结果 outputs/allen_v1/devmode_v2.md。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cholesky, solve_triangular

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from evowm.data.species_table import AVONET_XLSX, CROSSWALK_SHEET, _tip  # noqa: E402
from evowm.data.tree import read_trees  # noqa: E402
from evowm.eval.pgls import pgls_lambda, tip_vcv  # noqa: E402

OUT = ROOT / "outputs/allen_v1"
MIN_EFFECT = 0.0043
LIFE = ["aerial", "aquatic", "generalist", "terrestrial"]


def base():
    t = pd.read_csv(ROOT / "data/processed/species_traits.csv")
    c = pd.read_csv(ROOT / "data/processed/species_climate.csv")
    dm = pd.read_csv(ROOT / "data/raw/devmode/devmode_families.csv")
    d = t.merge(c, on="tip", how="inner").merge(dm[["family", "mode", "exposed"]], on="family", how="left")
    d.loc[d["order"] == "Passeriformes", ["mode", "exposed"]] = ["A", 0]
    d["lat_span"] = d["lat_max"] - d["lat_min"]
    d = d.dropna(subset=["m_mass", "e_habdens", "e_migr", "lat_span", "bio1_c", "exposed"] + [f"e_life_{k}" for k in LIFE])
    d["exposed"] = d["exposed"].astype(int)
    return d.reset_index(drop=True)


def crosswalk():
    cw = pd.read_excel(ROOT / AVONET_XLSX, sheet_name=CROSSWALK_SHEET)
    cw = cw[cw["Match.type"] == "1BL to 1BT"]
    return dict(zip(cw["Species1"].map(_tip), cw["Species3"].map(_tip)))


def load_cooney(tips):
    x = pd.read_excel(ROOT / "data/raw/cooney2020/41467_2020_16257_MOESM3_ESM.xlsx", sheet_name="Supplementary Figures 4-7")
    x["tip"] = x["binomial"].map(_tip)
    cw = crosswalk()
    miss = ~x["tip"].isin(tips)
    x.loc[miss, "tip"] = x.loc[miss, "tip"].map(cw)
    x = x.dropna(subset=["tip"]).drop_duplicates("tip")
    return x[["tip", "nest_type", "dev_mod", "log_fledging_d", "log_incubation_d"]]


def load_eliason(tips):
    e = pd.read_csv(ROOT / "data/raw/eliason2023/S1_limb_data.csv")
    e["tip"] = e["Taxon"].str.strip().map(_tip)
    cw = crosswalk()
    miss = ~e["tip"].isin(tips)
    e.loc[miss, "tip"] = e.loc[miss, "tip"].map(cw)
    e = e.dropna(subset=["tip"])
    for c in ["fem_leng_mm", "tbt_leng_mm", "tmt_leng_mm", "hum_leng_mm"]:
        e[c] = pd.to_numeric(e[c], errors="coerce")
    g = e.groupby("tip").agg(b_fem=("fem_leng_mm", "mean"), b_tbt=("tbt_leng_mm", "mean"), b_tmt=("tmt_leng_mm", "mean"),
                             b_hum=("hum_leng_mm", "mean"), n_spec=("tip", "size")).reset_index()
    for c in ["b_fem", "b_tbt", "b_tmt", "b_hum"]:
        g[c] = np.log10(g[c].where(g[c] > 0))
    g["b_r"] = g["b_tmt"] - g["b_fem"]
    return g


def design(d, group_col=None, temp="bio1_c"):
    T = d[temp].to_numpy(float) / 10.0
    cols = {"intercept": np.ones(len(d)), "M": d["m_mass"].to_numpy(float), "T": T}
    if group_col is not None:
        G = d[group_col].to_numpy(float)
        cols["TxG"] = T * G
        cols["G"] = G
    cols.update({"habdens": d["e_habdens"].to_numpy(float), "migr": d["e_migr"].to_numpy(float), "lat_span": d["lat_span"].to_numpy(float) / 10.0})
    for k in LIFE:
        v = d[f"e_life_{k}"].to_numpy(float)
        if v.std() > 0:
            cols[f"life_{k}"] = v
    lk = [k for k in cols if k.startswith("life_")]
    if lk and np.all(sum(cols[k] for k in lk) == 1):
        cols.pop(lk[-1])
    return np.column_stack(list(cols.values())), list(cols)


def fit(d, C_all, pos, resp, group_col=None, temp="bio1_c"):
    d = d.dropna(subset=[resp, temp]).reset_index(drop=True)
    rows = np.array([pos[x] for x in d["tip"]])
    C = C_all[np.ix_(rows, rows)]
    X, names = design(d, group_col, temp)
    y = d[resp].to_numpy(float)
    res = pgls_lambda(y, X, C, names)
    n = len(d)
    V = C * res["lambda"]
    V[np.arange(n), np.arange(n)] = np.diag(C)
    L = cholesky(V, lower=True, check_finite=False)
    Z = solve_triangular(L, X, lower=True, check_finite=False)
    z = solve_triangular(L, y, lower=True, check_finite=False)
    beta, *_ = np.linalg.lstsq(Z, z, rcond=None)
    r = z - Z @ beta
    s2 = float(r @ r) / (n - X.shape[1])
    cov = np.linalg.inv(Z.T @ Z) * s2
    iT = names.index("T")
    rec = {"n": n, "lambda": res["lambda"], "T_b": beta[iT], "T_se": np.sqrt(cov[iT, iT])}
    if group_col is not None:
        iG = names.index("TxG")
        rec.update({"n_g1": int(d[group_col].sum()), "g0_b": beta[iT], "g0_se": np.sqrt(cov[iT, iT]),
                    "g1_b": beta[iT] + beta[iG], "g1_se": np.sqrt(cov[iT, iT] + cov[iG, iG] + 2 * cov[iT, iG]),
                    "diff_b": beta[iG], "diff_se": np.sqrt(cov[iG, iG])})
    return rec


def ci(b, se):
    return f"{b:+.4f} [{b - 1.96 * se:+.4f}, {b + 1.96 * se:+.4f}]"


def sig_pos(b, se):
    return b - 1.96 * se > 0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = base()
    tree = read_trees(ROOT / "data/processed/trees/hackett_t021.nwk", max_trees=1)[0]
    pos = {n: i for i, n in enumerate(tree.tip_names)}
    C_all = tip_vcv(tree)
    d = d[d["tip"].isin(pos)].reset_index(drop=True)
    tips = set(d["tip"])
    co = load_cooney(tips)
    el = load_eliason(tips)
    d = d.merge(co, on="tip", how="left").merge(el, on="tip", how="left")
    d["open_nest"] = np.where(d["nest_type"] == "open", 1.0, np.where(d["nest_type"].isin(["cavity", "closed"]), 0.0, np.nan))
    L = ["# 发育方式与腿的 Allen 定律 v2：巢型、骨长复现、分段", "",
         "脚本 `scripts/allen/devmode_v2.py`。判据见设计文档第 1、2 节，运行前定下。系数：温度每升高 10 °C 的 log10 变化，0.0043 = 1%，方括号为 95% 区间。", ""]

    # ---------- T1 ----------
    nest = d[(d["exposed"] == 0) & d["open_nest"].notna() & d["m_tarsus"].notna() & (d["avonet_inferred"] == 0)].reset_index(drop=True)
    t1 = {}
    for lab, sub in [("合并", nest), ("雀形目", nest[nest["order"] == "Passeriformes"]), ("非雀形目巢内组", nest[nest["order"] != "Passeriformes"])]:
        sub = sub.reset_index(drop=True)
        t1[lab] = fit(sub, C_all, pos, "m_tarsus", group_col="open_nest")
    m = t1["合并"]
    t1_support = sig_pos(m["diff_b"], m["diff_se"]) and m["diff_b"] >= MIN_EFFECT and t1["雀形目"]["diff_b"] > 0
    t1_against = (m["diff_b"] - 1.96 * m["diff_se"] <= 0)
    L += ["## T1 巢的热缓冲（巢内组：开放巢 对 洞巢 + 封闭巢；AVONET 跗跖）", "",
          "| 样本 | n（开放巢） | λ | 洞巢与封闭巢斜率 | 开放巢斜率 | 之差（开放 − 封闭） |", "|---|---|---|---|---|---|"]
    for lab, r in t1.items():
        L.append(f"| {lab} | {r['n']}（{r['n_g1']}） | {r['lambda']:.2f} | {ci(r['g0_b'], r['g0_se'])} | {ci(r['g1_b'], r['g1_se'])} | {ci(r['diff_b'], r['diff_se'])} |")
    L += ["", f"判定：{'T1 支持机制。' if t1_support else ('T1 反对机制：开放巢不比封闭巢更接近 Allen。' if t1_against else 'T1 介于两者之间。')}", ""]

    # ---------- T2 ----------
    npb = d[(d["order"] != "Passeriformes") & d["b_tmt"].notna()].reset_index(drop=True)
    t2 = fit(npb, C_all, pos, "b_tmt", group_col="exposed")
    t2_rep = sig_pos(t2["g1_b"], t2["g1_se"]) and (t2["diff_b"] - 1.96 * t2["diff_se"] > 0)
    L += ["## T2 用 Eliason 跗跖骨长复现两组（非雀形目）", "",
          f"物种 {t2['n']}（暴露组 {t2['n_g1']}），λ = {t2['lambda']:.2f}。暴露组 {ci(t2['g1_b'], t2['g1_se'])}，巢内组 {ci(t2['g0_b'], t2['g0_se'])}，之差 {ci(t2['diff_b'], t2['diff_se'])}。", "",
          f"判定：{'T2 复现。' if t2_rep else 'T2 未复现。'}", ""]

    # ---------- T3 ----------
    L += ["## T3 分段（Eliason 三段骨长，非雀形目）", "",
          "| 组 | n | 股骨 | 胫跗骨 | 跗跖骨 | r = 跗跖骨 − 股骨 | 肱骨（探索） |", "|---|---|---|---|---|---|---|"]
    t3 = {}
    for lab, sub in [("暴露组", npb[npb["exposed"] == 1]), ("巢内组", npb[npb["exposed"] == 0])]:
        sub = sub.reset_index(drop=True)
        recs = {k: fit(sub, C_all, pos, k) for k in ["b_fem", "b_tbt", "b_tmt", "b_r", "b_hum"]}
        t3[lab] = recs
        L.append(f"| {lab} | {recs['b_tmt']['n']} | " + " | ".join(ci(recs[k]["T_b"], recs[k]["T_se"]) for k in ["b_fem", "b_tbt", "b_tmt", "b_r", "b_hum"]) + " |")
    e = t3["暴露组"]
    seg_ok = sig_pos(e["b_tmt"]["T_b"], e["b_tmt"]["T_se"]) and ((e["b_fem"]["T_b"] - 1.96 * e["b_fem"]["T_se"] <= 0 <= e["b_fem"]["T_b"] + 1.96 * e["b_fem"]["T_se"]) or e["b_fem"]["T_b"] < 0.5 * e["b_tmt"]["T_b"]) and sig_pos(e["b_r"]["T_b"], e["b_r"]["T_se"])
    whole = all(sig_pos(e[k]["T_b"], e[k]["T_se"]) for k in ["b_fem", "b_tbt", "b_tmt"]) and not sig_pos(e["b_r"]["T_b"], e["b_r"]["T_se"])
    L += ["", f"判定：{'T3 支持分段：暴露组里只有跗跖骨随温度变长。' if seg_ok else ('T3 整腿 Allen：暴露组三段都变长，比例不变。' if whole else 'T3 未检出分段。')}", ""]

    # ---------- 探索 ----------
    L += ["## 探索（不进判据）", ""]
    nb = d[(d["exposed"] == 0) & d["log_fledging_d"].notna() & d["m_tarsus"].notna() & (d["avonet_inferred"] == 0)].reset_index(drop=True)
    nb["lfl"] = nb["log_fledging_d"] - nb["log_fledging_d"].mean()
    r = fit(nb, C_all, pos, "m_tarsus", group_col="lfl")
    L.append(f"- 巢内组，T × log10 留巢期（中心化）：{ci(r['diff_b'], r['diff_se'])}（n = {r['n']}）。为正表示留巢越久温度斜率越正。")
    eb = d[(d["exposed"] == 1) & d["log_fledging_d"].notna() & d["m_tarsus"].notna() & (d["avonet_inferred"] == 0)].reset_index(drop=True)
    eb["lfl"] = eb["log_fledging_d"] - eb["log_fledging_d"].mean()
    r = fit(eb, C_all, pos, "m_tarsus", group_col="lfl")
    L.append(f"- 暴露组，T × log10 到首飞天数（中心化）：{ci(r['diff_b'], r['diff_se'])}（n = {r['n']}）。")
    for lab, sub in [("半早成 SP", d[d["mode"] == "SP"]), ("早成 P", d[d["mode"] == "P"]), ("半晚成 SA", d[d["mode"] == "SA"]), ("晚成 A（非雀形目）", d[(d["mode"] == "A") & (d["order"] != "Passeriformes")])]:
        sub = sub[sub["m_tarsus"].notna() & (sub["avonet_inferred"] == 0)].reset_index(drop=True)
        r = fit(sub, C_all, pos, "m_tarsus")
        L.append(f"- {lab}单独（AVONET 跗跖，n = {r['n']}）：{ci(r['T_b'], r['T_se'])}")
    L += ["", "## 覆盖", "",
          f"- Cooney 对上 BirdTree {int(d['nest_type'].notna().sum() + d['nest_type'].isna().sum() - d['nest_type'].isna().sum())} 种有巢型；巢内组有巢型且非 mixed 的 {len(nest)} 种。",
          f"- Eliason 对上 BirdTree 且三段齐全的非雀形目 {len(npb)} 种。", ""]
    (OUT / "devmode_v2.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
