"""发育方式与腿的 Allen 定律（设计 docs/design/发育方式与腿的Allen定律_设计_2026-10-26.md）。

主模型（非雀形目，PGLS，Pagel λ）：log10 跗跖 ~ M + T + T×E + E + 协变量；E = 暴露组（早成 P + 半早成 SP）。
判据（运行前定）：暴露组斜率 > 0、区间不含 0、≥ 0.0043；巢内组斜率区间含 0 或 < 暴露组一半；两组之差区间不含 0。
稳健性：P 对 A 严格对比；去掉 uncertain 科；加 T×生活方式；bio5、bio6、盒均值；体重三档；雀形目参照。探索：翅长对照；鸭科三段腿骨。
在 EvoWM 根目录运行：OMP_NUM_THREADS=2 python scripts/allen/devmode_pgls_v1.py；结果 outputs/allen_v1/devmode.md、devmode_coefs.csv。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cholesky, solve_triangular

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from evowm.data.tree import read_trees  # noqa: E402
from evowm.eval.pgls import pgls_lambda, tip_vcv  # noqa: E402

OUT = ROOT / "outputs/allen_v1"
MIN_EFFECT = 0.0043
LIFE = ["aerial", "aquatic", "generalist", "terrestrial"]


def load():
    t = pd.read_csv(ROOT / "data/processed/species_traits.csv")
    c = pd.read_csv(ROOT / "data/processed/species_climate.csv")
    dm = pd.read_csv(ROOT / "data/raw/devmode/devmode_families.csv")
    d = t.merge(c, on="tip", how="inner").merge(dm[["family", "mode", "exposed", "note"]], on="family", how="left")
    d.loc[d["order"] == "Passeriformes", ["mode", "exposed"]] = ["A", 0]
    d["lat_span"] = d["lat_max"] - d["lat_min"]
    d["uncertain"] = d["note"].fillna("").str.contains("uncertain").astype(int)
    need = ["m_tarsus", "m_mass", "e_habdens", "e_migr", "lat_span", "bio1_c", "exposed"] + [f"e_life_{k}" for k in LIFE]
    d = d[d["avonet_inferred"] == 0].dropna(subset=need).reset_index(drop=True)
    d["exposed"] = d["exposed"].astype(int)
    return d


def design(d, temp, life_x=False):
    T = d[temp].to_numpy(float) / 10.0
    E = d["exposed"].to_numpy(float)
    cols = {"intercept": np.ones(len(d)), "M": d["m_mass"].to_numpy(float), "T": T, "TxE": T * E, "E": E,
            "habdens": d["e_habdens"].to_numpy(float), "migr": d["e_migr"].to_numpy(float), "lat_span": d["lat_span"].to_numpy(float) / 10.0}
    for k in LIFE:
        v = d[f"e_life_{k}"].to_numpy(float)
        if v.std() > 0:
            cols[f"life_{k}"] = v
            if life_x:
                cols[f"Tx{k}"] = T * v
    return np.column_stack(list(cols.values())), list(cols)


def fit(d, C_all, pos, temp="bio1_c", life_x=False, resp="m_tarsus", boot=0, seed=0):
    d = d.dropna(subset=[resp, temp]).reset_index(drop=True)
    rows = np.array([pos[x] for x in d["tip"]])
    C = C_all[np.ix_(rows, rows)]
    X, names = design(d, temp, life_x)
    y = d[resp].to_numpy(float)
    res = pgls_lambda(y, X, C, names)
    # 系数协方差（固定 λ）
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
    i0, i1 = names.index("T"), names.index("TxE")
    nest = (beta[i0], np.sqrt(cov[i0, i0]))
    diff = (beta[i1], np.sqrt(cov[i1, i1]))
    exp_b = beta[i0] + beta[i1]
    exp_se = np.sqrt(cov[i0, i0] + cov[i1, i1] + 2 * cov[i0, i1])
    rec = {"n": n, "n_exposed": int(d["exposed"].sum()), "lambda": res["lambda"], "temp": temp, "resp": resp, "life_x": life_x,
           "nest_b": nest[0], "nest_se": nest[1], "exp_b": exp_b, "exp_se": exp_se, "diff_b": diff[0], "diff_se": diff[1],
           "beta_M": beta[names.index("M")]}
    if boot:
        rng = np.random.default_rng(seed)
        fam = pd.factorize(d["family"])[0]
        nf = fam.max() + 1
        out = np.empty((boot, 3))
        for b in range(boot):
            w = rng.multinomial(nf, np.full(nf, 1.0 / nf))[fam].astype(float)
            sw = np.sqrt(w)
            bb, *_ = np.linalg.lstsq(Z * sw[:, None], z * sw, rcond=None)
            out[b] = [bb[i0], bb[i0] + bb[i1], bb[i1]]
        q = np.percentile(out, [2.5, 97.5], axis=0)
        rec.update({"nest_blo": q[0, 0], "nest_bhi": q[1, 0], "exp_blo": q[0, 1], "exp_bhi": q[1, 1], "diff_blo": q[0, 2], "diff_bhi": q[1, 2]})
    return rec


def ci(b, se):
    return f"{b:+.4f} [{b - 1.96 * se:+.4f}, {b + 1.96 * se:+.4f}]"


def row(label, r):
    return (f"| {label} | {r['n']}（暴露 {r['n_exposed']}） | {r['lambda']:.2f} | {ci(r['exp_b'], r['exp_se'])} | "
            f"{ci(r['nest_b'], r['nest_se'])} | {ci(r['diff_b'], r['diff_se'])} |")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = load()
    tree = read_trees(ROOT / "data/processed/trees/hackett_t021.nwk", max_trees=1)[0]
    pos = {n: i for i, n in enumerate(tree.tip_names)}
    C_all = tip_vcv(tree)
    d = d[d["tip"].isin(pos)].reset_index(drop=True)
    NP = d[d["order"] != "Passeriformes"].reset_index(drop=True)
    PA = d[d["order"] == "Passeriformes"].reset_index(drop=True)
    recs = {}

    recs["main"] = fit(NP, C_all, pos, boot=500)
    recs["strict"] = fit(NP[NP["mode"].isin(["P", "A"])].reset_index(drop=True), C_all, pos)
    recs["no_uncertain"] = fit(NP[NP["uncertain"] == 0].reset_index(drop=True), C_all, pos)
    recs["life_x"] = fit(NP, C_all, pos, life_x=True)
    recs["bio5"] = fit(NP, C_all, pos, temp="bio5_c")
    recs["bio6"] = fit(NP, C_all, pos, temp="bio6_c")
    recs["box"] = fit(NP, C_all, pos, temp="bio1_box")
    mass = 10 ** NP["m_mass"]
    for lab, sel in [("lt100", mass < 100), ("100to1000", (mass >= 100) & (mass < 1000)), ("gt1000", mass >= 1000)]:
        sub = NP[sel].reset_index(drop=True)
        if sub["exposed"].nunique() == 2 and len(sub) > 100:
            recs[f"mass_{lab}"] = fit(sub, C_all, pos)
    recs["wing"] = fit(NP, C_all, pos, resp="m_wing")
    # 雀形目参照：只有 T（全部 E = 0），用同一函数但 TxE 列全 0 会奇异，故单独拟合
    pa_rows = np.array([pos[x] for x in PA["tip"]])
    Xp, names_p = design(PA.assign(exposed=0), "bio1_c")
    keep = [i for i, nm in enumerate(names_p) if nm not in ("TxE", "E")]
    res_p = pgls_lambda(PA["m_tarsus"].to_numpy(float), Xp[:, keep], C_all[np.ix_(pa_rows, pa_rows)], [names_p[i] for i in keep])
    pa_T = res_p["coef"]["T"]

    pd.DataFrame(recs).T.to_csv(OUT / "devmode_coefs.csv", float_format="%.5f")

    m = recs["main"]
    exp_lo, exp_hi = m["exp_b"] - 1.96 * m["exp_se"], m["exp_b"] + 1.96 * m["exp_se"]
    nest_lo, nest_hi = m["nest_b"] - 1.96 * m["nest_se"], m["nest_b"] + 1.96 * m["nest_se"]
    diff_lo, diff_hi = m["diff_b"] - 1.96 * m["diff_se"], m["diff_b"] + 1.96 * m["diff_se"]
    ok_exp = exp_lo > 0 and m["exp_b"] >= MIN_EFFECT
    ok_nest = (nest_lo <= 0 <= nest_hi) or (m["nest_b"] < 0.5 * m["exp_b"])
    ok_diff = diff_lo > 0 or diff_hi < 0
    lx = recs["life_x"]
    lx_ok = (lx["diff_b"] - 1.96 * lx["diff_se"] > 0) or (lx["diff_b"] + 1.96 * lx["diff_se"] < 0)
    if ok_exp and ok_nest and ok_diff:
        verdict = "成立：暴露组斜率为正且不小于最小关心效应，巢内组接近 0，两组之差区间不含 0。" + ("" if lx_ok else " 但加入温度 × 生活方式后两组之差区间含 0：无法与生活方式区分。")
    elif (exp_lo <= 0 <= exp_hi) or (diff_lo <= 0 <= diff_hi):
        verdict = "不成立：暴露组斜率或两组之差的区间含 0。"
    else:
        verdict = "介于两者之间，按实描述。"

    L = ["# 发育方式与腿的 Allen 定律", "",
         "脚本 `scripts/allen/devmode_pgls_v1.py`。判据见设计文档第 4 节，运行前定下。", "",
         "## 符号", "",
         "- 系数：温度每升高 10 °C，log10 跗跖长（AVONET 外测）的变化；0.0043 = 1%。方括号为 1.96 倍标准误（固定 λ 的系数协方差）。",
         "- 暴露组 = 早成 + 半早成（腿在巢外长）；巢内组 = 半晚成 + 晚成。『之差』= 暴露组斜率 − 巢内组斜率。",
         f"- 非雀形目 {len(NP)} 种，暴露组 {int(NP.exposed.sum())} 种（P {int((NP['mode']=='P').sum())}，SP {int((NP['mode']=='SP').sum())}），"
         f"巢内组 {int((NP.exposed==0).sum())} 种（SA {int((NP['mode']=='SA').sum())}，A {int((NP['mode']=='A').sum())}）；uncertain 科内物种 {int(NP.uncertain.sum())}。", "",
         f"## 判定：{verdict}", "",
         "## 1. 主模型与稳健性（非雀形目，年均温除注明外）", "",
         "| 模型 | n | λ | 暴露组斜率 | 巢内组斜率 | 之差 |", "|---|---|---|---|---|---|",
         row("主模型（bio1 质心）", recs["main"]),
         f"| 　科重抽样 95% 区间 | | | [{m['exp_blo']:+.4f}, {m['exp_bhi']:+.4f}] | [{m['nest_blo']:+.4f}, {m['nest_bhi']:+.4f}] | [{m['diff_blo']:+.4f}, {m['diff_bhi']:+.4f}] |",
         row("严格：只用 P 对 A", recs["strict"]),
         row("去掉 uncertain 科", recs["no_uncertain"]),
         row("加 温度 × 生活方式", recs["life_x"]),
         row("最热月最高温 bio5", recs["bio5"]),
         row("最冷月最低温 bio6", recs["bio6"]),
         row("年均温盒均值", recs["box"])]
    for lab, zh in [("mass_lt100", "体重 < 100 g"), ("mass_100to1000", "体重 100 到 1,000 g"), ("mass_gt1000", "体重 > 1,000 g")]:
        if lab in recs:
            L.append(row(zh, recs[lab]))
    L += ["", "## 2. 参照与对照", "",
          f"- 雀形目（全部晚成，n = {res_p['n']}，λ = {res_p['lambda']:.2f}）：温度斜率 {ci(pa_T['beta'], pa_T['se'])}。",
          f"- 翅长（非雀形目，阴性对照，翅不该随发育方式变）：暴露组 {ci(recs['wing']['exp_b'], recs['wing']['exp_se'])}，巢内组 {ci(recs['wing']['nest_b'], recs['wing']['nest_se'])}，之差 {ci(recs['wing']['diff_b'], recs['wing']['diff_se'])}。", "",
          "## 3. 读法", "",
          "- 假说要的是：暴露组为正（≥ 0.0043），巢内组接近 0，之差区间不含 0；加温度 × 生活方式后之差仍在。",
          "- 之差为正但暴露组本身不显著，或两组都为正，都不算成立。", ""]
    (OUT / "devmode.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
