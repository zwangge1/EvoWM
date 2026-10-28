"""方向 5，M1：跨物种 PGLS，分段 Allen 定律（设计文档 docs/design/标本级变异_分段Allen定律_设计_2026-10-21.md 第 3.1、4 节）。

响应（都是 log10）：股骨 f、胫跗骨 t、跗跖骨 m、核心量 r = m − f（跗跖骨相对股骨的比例），另加整腿 L = log10(10^f + 10^t + 10^m)。
温度：T 以 10 °C 为单位，系数直接是"每升高 10 °C 的 log10 变化"；最小关心效应 0.0043（r 变 1%）。
模型：y ~ M + T + 生活方式哑变量（栖树为基准）+ 栖境密度 + 迁徙等级 + 纬度跨度；PGLS，Pagel λ 逐模型最大似然。
温度版本：主 bio1 质心年均温；bio5 最热月最高、bio6 最冷月最低；盒均值三版作稳健性。
区间：PGLS 标准误的 1.96 倍；主模型另按科加权重抽样 500 次（白化后按科的多项式权重做加权最小二乘）。
稳健性：只雀形目；去掉纬度跨度协变量。探索性：肱骨、尺骨（翅骨对照，Weeks 2025 的方向）。
判据（设计第 4 节）：r 对 bio1 的系数 > 0、区间不含 0、点估计 ≥ 0.0043 为"跨物种成立"；三段都正而 r 区间含 0 为"整腿 Allen"。
在 EvoWM 根目录运行：OMP_NUM_THREADS=2 python scripts/allen/pgls_species_v1.py；结果 outputs/allen_v1/species_pgls.md、species_coefs.csv。
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
LIFE = ["aerial", "aquatic", "generalist", "terrestrial"]   # 基准 insessorial
RESP_ZH = {"f": "股骨", "t": "胫跗骨", "m": "跗跖骨", "r": "r = 跗跖骨 − 股骨", "L": "整腿", "h": "肱骨（探索）", "u": "尺骨（探索）"}
TEMP_ZH = {"bio1_c": "年均温（质心）", "bio5_c": "最热月最高温（质心）", "bio6_c": "最冷月最低温（质心）",
           "bio1_box": "年均温（盒均值）", "bio5_box": "最热月最高温（盒均值）", "bio6_box": "最冷月最低温（盒均值）"}


def load():
    t = pd.read_csv(ROOT / "data/processed/species_traits.csv")
    c = pd.read_csv(ROOT / "data/processed/species_climate.csv")
    d = t.merge(c, on="tip", how="inner")
    d["f"], d["t"], d["m"] = d["k_femur"], d["k_tibiotarsus"], d["k_tarsometatarsus"]
    d["r"] = d["m"] - d["f"]
    d["L"] = np.log10(10 ** d["f"] + 10 ** d["t"] + 10 ** d["m"])
    d["h"], d["u"] = d["k_humerus"], d["k_ulna"]
    d["lat_span"] = d["lat_max"] - d["lat_min"]
    need = ["f", "t", "m", "m_mass", "e_habdens", "e_migr", "lat_span", "bio1_c"] + [f"e_life_{k}" for k in LIFE]
    d = d.dropna(subset=need).reset_index(drop=True)
    return d


def design(d, temp, with_span=True):
    cols = {"intercept": np.ones(len(d)), "M": d["m_mass"].to_numpy(float), "T": d[temp].to_numpy(float) / 10.0,
            "habdens": d["e_habdens"].to_numpy(float), "migr": d["e_migr"].to_numpy(float)}
    if with_span:
        cols["lat_span"] = d["lat_span"].to_numpy(float) / 10.0
    for k in LIFE:
        v = d[f"e_life_{k}"].to_numpy(float)
        if v.std() > 0:            # 子样本里没有的类别（如雀形目里的水生）去掉，免得设计矩阵奇异
            cols[f"life_{k}"] = v
    return np.column_stack(list(cols.values())), list(cols.keys())


class GLS:
    """固定 λ 的 PGLS：白化一次，之后重复拟合与科重抽样都便宜。"""

    def __init__(self, C, lam):
        n = len(C)
        V = C * lam
        V[np.arange(n), np.arange(n)] = np.diag(C)
        self.L = cholesky(V, lower=True, check_finite=False)

    def whiten(self, A):
        return solve_triangular(self.L, A, lower=True, check_finite=False)


def family_boot(d, gls, y, X, n_boot=500, seed=0):
    """科加权重抽样：白化后按科的多项式权重做加权最小二乘，返回 T 系数的 2.5%、97.5% 分位。"""
    rng = np.random.default_rng(seed)
    z, Z = gls.whiten(y), gls.whiten(X)
    fam = pd.factorize(d["family"])[0]
    nf = fam.max() + 1
    out = np.empty(n_boot)
    for b in range(n_boot):
        w = rng.multinomial(nf, np.full(nf, 1.0 / nf))[fam].astype(float)
        sw = np.sqrt(w)
        beta, *_ = np.linalg.lstsq(Z * sw[:, None], z * sw, rcond=None)
        out[b] = beta[2]   # T 在第 3 列
    return np.percentile(out, [2.5, 97.5])


def fit_one(d, C, resp, temp, with_span=True, boot=False):
    sub = d.dropna(subset=[resp, temp]).reset_index(drop=True)
    rows = sub["_row"].to_numpy()
    Cs = C[np.ix_(rows, rows)]
    X, names = design(sub, temp, with_span)
    y = sub[resp].to_numpy(float)
    res = pgls_lambda(y, X, Cs, names)
    cT = res["coef"]["T"]
    rec = {"resp": resp, "temp": temp, "with_span": with_span, "n": len(sub), "lambda": res["lambda"],
           "beta_T": cT["beta"], "se_T": cT["se"], "lo_T": cT["beta"] - 1.96 * cT["se"], "hi_T": cT["beta"] + 1.96 * cT["se"],
           "beta_M": res["coef"]["M"]["beta"], "se_M": res["coef"]["M"]["se"]}
    if boot:
        lo, hi = family_boot(sub, GLS(Cs, res["lambda"]), y, X)
        rec["boot_lo_T"], rec["boot_hi_T"] = lo, hi
    return rec


def fmt(rec, boot=False):
    s = f"{rec['beta_T']:+.4f} [{rec['lo_T']:+.4f}, {rec['hi_T']:+.4f}]"
    if boot and "boot_lo_T" in rec:
        s += f" 科抽样 [{rec['boot_lo_T']:+.4f}, {rec['boot_hi_T']:+.4f}]"
    return s


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = load()
    tree = read_trees(ROOT / "data/processed/trees/hackett_t021.nwk", max_trees=1)[0]
    pos = {n: i for i, n in enumerate(tree.tip_names)}
    d = d[d["tip"].isin(pos)].reset_index(drop=True)
    d["_row"] = np.arange(len(d))
    C_all = tip_vcv(tree)
    rows = np.array([pos[x] for x in d["tip"]])
    C = C_all[np.ix_(rows, rows)]
    pas = d["order"] == "Passeriformes"
    recs = []

    # 1. 主分析：bio1 质心，四个响应 + 整腿，科重抽样
    main = {}
    for resp in ["f", "t", "m", "r", "L"]:
        rec = fit_one(d, C, resp, "bio1_c", boot=True)
        rec["block"] = "main"
        main[resp] = rec
        recs.append(rec)
    # 2. 其他温度版本
    for temp in ["bio5_c", "bio6_c", "bio1_box", "bio5_box", "bio6_box"]:
        for resp in ["f", "t", "m", "r"]:
            rec = fit_one(d, C, resp, temp)
            rec["block"] = "temp"
            recs.append(rec)
    # 3. 稳健性：只雀形目；去掉纬度跨度
    dp = d[pas].reset_index(drop=True)
    dp["_row"] = np.arange(len(dp))
    Cp = C[np.ix_(np.flatnonzero(pas), np.flatnonzero(pas))]
    rob = {}
    for resp in ["f", "m", "r"]:
        rec = fit_one(dp, Cp, resp, "bio1_c")
        rec["block"] = "passerines"
        rob[("pas", resp)] = rec
        recs.append(rec)
        rec = fit_one(d, C, resp, "bio1_c", with_span=False)
        rec["block"] = "no_span"
        rob[("nospan", resp)] = rec
        recs.append(rec)
    # 4. 探索：翅骨
    expl = {}
    for resp in ["h", "u"]:
        rec = fit_one(d, C, resp, "bio1_c")
        rec["block"] = "wing"
        expl[resp] = rec
        recs.append(rec)
    pd.DataFrame(recs).to_csv(OUT / "species_coefs.csv", index=False, float_format="%.5f")

    # 判定
    r = main["r"]
    seg_pos = all(main[k]["lo_T"] > 0 for k in ["f", "t", "m"])
    if r["lo_T"] > 0 and r["beta_T"] >= MIN_EFFECT:
        verdict = "跨物种成立：r 对年均温的系数为正、区间不含 0、点估计不小于最小关心效应。"
    elif seg_pos and r["lo_T"] <= 0 <= r["hi_T"]:
        verdict = "整腿 Allen：三段各自都随温度变长，但跗跖骨相对股骨的比例不随温度变。"
    elif r["hi_T"] < 0:
        verdict = "方向相反：温度越高跗跖骨相对股骨越短，与分段假说相反。"
    else:
        verdict = "跨物种这一层未检出：r 的区间含 0 或点估计小于最小关心效应。"

    L = ["# 方向 5，M1：跨物种 PGLS，分段 Allen 定律", "",
         "脚本 `scripts/allen/pgls_species_v1.py`。判据与模型写在设计文档第 3.1、4 节，运行前定下。", "",
         "## 符号", "",
         "- 系数都是『温度每升高 10 °C，响应（log10 骨长）的变化』；0.0043 对应骨长变 1%。方括号是 PGLS 标准误的 95% 区间，"
         "『科抽样』是按科加权重抽样 500 次的 95% 区间。",
         "- 响应：股骨、胫跗骨、跗跖骨、r = log10(跗跖骨 / 股骨)、整腿 = log10 三段之和。协变量：log10 体重、生活方式、栖境密度、迁徙等级、纬度跨度。",
         f"- 物种 {len(d)}（雀形目 {int(pas.sum())}，其余 {int((~pas).sum())}，基本是雁形目）。年均温（质心）范围 "
         f"{d.bio1_c.min():.1f} 到 {d.bio1_c.max():.1f} °C，四分位 {d.bio1_c.quantile(.25):.1f} 到 {d.bio1_c.quantile(.75):.1f}。", "",
         f"## 判定：{verdict}", "",
         "## 1. 主分析（年均温，质心）", "",
         "| 响应 | n | λ | 温度系数（每 10 °C） | 体重系数 |", "|---|---|---|---|---|"]
    for k in ["f", "t", "m", "r", "L"]:
        rec = main[k]
        L.append(f"| {RESP_ZH[k]} | {rec['n']} | {rec['lambda']:.2f} | {fmt(rec, boot=True)} | {rec['beta_M']:+.3f} ({rec['se_M']:.3f}) |")
    L += ["", "## 2. 其他温度版本（温度系数，每 10 °C）", "",
          "| 温度 | 股骨 | 胫跗骨 | 跗跖骨 | r |", "|---|---|---|---|---|"]
    by = {(x["temp"], x["resp"]): x for x in recs if x["block"] in ("main", "temp")}
    for temp in TEMP_ZH:
        L.append(f"| {TEMP_ZH[temp]} | " + " | ".join(fmt(by[(temp, k)]) for k in ["f", "t", "m", "r"]) + " |")
    L += ["", "## 3. 稳健性（年均温，质心）", "",
          "| 版本 | 股骨 | 跗跖骨 | r |", "|---|---|---|---|",
          f"| 只雀形目（n = {rob[('pas', 'r')]['n']}） | " + " | ".join(fmt(rob[("pas", k)]) for k in ["f", "m", "r"]) + " |",
          "| 去掉纬度跨度协变量 | " + " | ".join(fmt(rob[("nospan", k)]) for k in ["f", "m", "r"]) + " |", "",
          "## 4. 探索：翅骨（不进判据）", "",
          "| 响应 | n | λ | 温度系数（每 10 °C） |", "|---|---|---|---|"]
    for k in ["h", "u"]:
        rec = expl[k]
        L.append(f"| {RESP_ZH[k]} | {rec['n']} | {rec['lambda']:.2f} | {fmt(rec)} |")
    L += ["", "## 5. 读法", "",
          "- 分段假说要的是：跗跖骨的温度系数为正，股骨的接近 0，于是 r 为正。",
          "- 整腿 Allen 是：三段系数都为正、大小接近，r 接近 0。",
          "- 系数在 log10 尺度，0.0043 = 1%，0.0086 = 2%。", ""]
    (OUT / "species_pgls.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
