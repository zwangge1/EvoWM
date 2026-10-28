"""方向 5，M2：种内混合模型，分段 Allen 定律（设计文档第 2.2、3.2、3.3、4 节）。

数据：data/processed/specimens.csv（UMMZ，三段腿骨齐全）+ species_traits.csv（迁徙等级、物种均值）。
清洗（3.3）：成体标记、性别雌或雄、有温度、纬度不在分布范围外、无圈养字样；种内 |z| > 4 的骨长去掉；
体重解析成功且在该种中位数的 1/3 到 3 倍之间才算有体重。
样本：
- main：留鸟全年 + 迁徙种（迁徙等级 2、3）繁殖季（北半球纬度 > 23.5° 取 5 到 8 月，南半球 < −23.5° 取 11 到 2 月，热带全年）；
- resident：只留鸟；all_months：全部物种不限月份。
纳入门槛：每种 ≥ 5 件且纬度跨度 ≥ 10°（主）；≥ 10 件（稳健性）。
模型（3.2）：y_ij = α_j + β_j T_ij + γ M_ij + δ sex_ij + θ year_ij + e_ij，物种随机截距与随机斜率（T），T、M、year 在种内中心化，
T 以 10 °C 为单位；REML。随机斜率不收敛时退回随机截距并注明。两版：不含体重（全部标本，主判据）与含体重（子集）。
响应：股骨 f、胫跗骨 t、跗跖骨 m、r = m − f（log10）。温度：bio1 主；bio5、bio6 探索。
判据（第 4 节）：main 样本、不含体重版、r 对 bio1 的 β > 0、95% 区间不含 0、点估计 ≥ 0.0043（每 10 °C 变 1%）。
在 EvoWM 根目录运行：python scripts/allen/mixed_specimens_v1.py；结果 outputs/allen_v1/specimens_mixed.md、specimens_coefs.csv。
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "outputs/allen_v1"
MIN_EFFECT = 0.0043
RESP_ZH = {"f": "股骨", "t": "胫跗骨", "m": "跗跖骨", "r": "r = 跗跖骨 − 股骨"}
TEMP_ZH = {"bio1": "年均温", "bio5": "最热月最高温", "bio6": "最冷月最低温"}
SAMPLE_ZH = {"main": "主样本（留鸟全年 + 迁徙种繁殖季）", "resident": "只留鸟", "all_months": "全部物种不限月份"}


def load():
    d = pd.read_csv(ROOT / "data/processed/specimens.csv", low_memory=False)
    sp = pd.read_csv(ROOT / "data/processed/species_traits.csv", usecols=["tip", "e_migr", "family"])
    d = d.merge(sp, on="tip", how="left")
    d = d[(d["inst"] == "UMMZ") & d[["femur", "tibiotarsus", "tarsometatarsus"]].notna().all(axis=1)]
    d = d[(d["adult_ok"] == 1) & d["sex"].isin(["Male", "Female"]) & d["bio1"].notna()
          & (d["lat_in_range"] != 0) & (d["captive_hint"] == 0) & d["e_migr"].notna()].copy()
    d["f"], d["t"], d["m"] = np.log10(d["femur"]), np.log10(d["tibiotarsus"]), np.log10(d["tarsometatarsus"])
    d["r"] = d["m"] - d["f"]
    d["male"] = (d["sex"] == "Male").astype(float)
    # 种内离群
    for k in ["f", "t", "m"]:
        z = d.groupby("tip")[k].transform(lambda x: (x - x.mean()) / x.std(ddof=1) if len(x) > 2 else 0.0)
        d = d[z.abs().fillna(0) <= 4]
    # 体重：相对该种中位数 1/3 到 3 倍之外置缺
    med = d.groupby("tip")["weight_g"].transform("median")
    bad = (d["weight_g"] < med / 3) | (d["weight_g"] > med * 3)
    d.loc[bad, "weight_g"] = np.nan
    d["M"] = np.log10(d["weight_g"])
    # 繁殖季
    north, south = d["lat"] > 23.5, d["lat"] < -23.5
    breed = np.where(north, d["month"].isin([5, 6, 7, 8]), np.where(south, d["month"].isin([11, 12, 1, 2]), True))
    d["in_main"] = (d["e_migr"] == 0) | breed
    d["in_resident"] = d["e_migr"] == 0
    d["in_all_months"] = True
    return d.reset_index(drop=True)


def subset(d, sample, min_n):
    s = d[d[f"in_{sample}"]].copy()
    g = s.groupby("tip")
    n, span = g.size(), g["lat"].agg(lambda x: x.max() - x.min())
    keep = n.index[(n >= min_n) & (span >= 10)]
    s = s[s["tip"].isin(keep)].copy()
    for temp in ["bio1", "bio5", "bio6"]:
        s[f"T_{temp}"] = (s[temp] - s.groupby("tip")[temp].transform("mean")) / 10.0
    s["year_c"] = (s["year"] - s.groupby("tip")["year"].transform("mean")).fillna(0.0) / 10.0
    s["M_c"] = s["M"] - s.groupby("tip")["M"].transform("mean")
    return s


def fit(s, resp, temp, with_mass):
    dd = s.dropna(subset=["M_c"]) if with_mass else s
    f = f"{resp} ~ T_{temp} + male + year_c" + (" + M_c" if with_mass else "")
    rec = {"resp": resp, "temp": temp, "with_mass": with_mass, "n": len(dd), "n_species": dd["tip"].nunique()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = smf.mixedlm(f, dd, groups=dd["tip"], re_formula=f"~T_{temp}").fit(reml=True, method=["lbfgs"])
            ok = res.converged and np.isfinite(res.bse[f"T_{temp}"])
            rec["re"] = "slope"
        except Exception:
            ok = False
        if not ok:
            res = smf.mixedlm(f, dd, groups=dd["tip"]).fit(reml=True, method=["lbfgs"])
            rec["re"] = "intercept"
    b, se = float(res.params[f"T_{temp}"]), float(res.bse[f"T_{temp}"])
    rec.update({"beta_T": b, "se_T": se, "lo_T": b - 1.96 * se, "hi_T": b + 1.96 * se,
                "beta_male": float(res.params["male"]), "se_male": float(res.bse["male"]),
                "beta_year": float(res.params["year_c"]), "se_year": float(res.bse["year_c"]),
                "resid_sd": float(np.sqrt(res.scale))})
    if with_mass:
        rec["beta_M"], rec["se_M"] = float(res.params["M_c"]), float(res.bse["M_c"])
    if rec["re"] == "slope":
        try:
            rec["sd_slope"] = float(np.sqrt(res.cov_re.iloc[1, 1]))
        except Exception:
            rec["sd_slope"] = np.nan
    return rec


def fmt(rec):
    tag = "" if rec["re"] == "slope" else "（随机截距）"
    return f"{rec['beta_T']:+.4f} [{rec['lo_T']:+.4f}, {rec['hi_T']:+.4f}]{tag}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = load()
    recs = []
    tables = {}
    for sample in ["main", "resident", "all_months"]:
        for min_n in [5, 10]:
            s = subset(d, sample, min_n)
            for with_mass in [False, True]:
                temps = ["bio1", "bio5", "bio6"] if (sample == "main" and min_n == 5) else ["bio1"]
                for temp in temps:
                    for resp in ["f", "t", "m", "r"]:
                        rec = fit(s, resp, temp, with_mass)
                        rec.update({"sample": sample, "min_n": min_n})
                        recs.append(rec)
                        tables[(sample, min_n, with_mass, temp, resp)] = rec
    pd.DataFrame(recs).to_csv(OUT / "specimens_coefs.csv", index=False, float_format="%.5f")

    key = tables[("main", 5, False, "bio1", "r")]
    segs = [tables[("main", 5, False, "bio1", k)] for k in ["f", "t", "m"]]
    if key["lo_T"] > 0 and key["beta_T"] >= MIN_EFFECT:
        verdict = "种内成立：r 对年均温的种内斜率为正、区间不含 0、点估计不小于最小关心效应。"
    elif all(x["lo_T"] > 0 for x in segs) and key["lo_T"] <= 0 <= key["hi_T"]:
        verdict = "整腿 Allen（种内）：三段都随温度变长，比例不变。"
    elif key["hi_T"] < 0:
        verdict = "方向相反（种内）：越热的地方跗跖骨相对股骨越短。"
    else:
        verdict = "种内这一层未检出：r 的区间含 0 或点估计小于最小关心效应。"

    s_main = subset(d, "main", 5)
    L = ["# 方向 5，M2：种内混合模型，分段 Allen 定律", "",
         "脚本 `scripts/allen/mixed_specimens_v1.py`。清洗、样本、模型与判据见设计文档第 2.2、3.2、3.3、4 节，运行前定下。", "",
         "## 符号", "",
         "- 系数是『同一物种内，采集点年均温每高 10 °C，log10 骨长的变化』（物种随机截距与随机斜率的总平均）；0.0043 对应 1%。方括号为 1.96 倍标准误。",
         "- 协变量：性别（雄 = 1）、采集年份（种内中心化，每 10 年）；含体重版再加 log10 野外体重（种内中心化）。",
         f"- 主样本（≥ 5 件、纬度跨度 ≥ 10°）：{s_main.tip.nunique()} 种，{len(s_main)} 件，其中有体重 {int(s_main.M.notna().sum())} 件；"
         f"种内年均温 SD 中位 {s_main.groupby('tip').bio1.std().median():.2f} °C。", "",
         f"## 判定：{verdict}", "",
         "## 1. 主样本，年均温", "",
         "| 响应 | 不含体重（n, 种） | 温度系数 | 性别（雄 − 雌） | 年份（每 10 年） | 含体重（n） | 温度系数 | 体重系数 |",
         "|---|---|---|---|---|---|---|---|"]
    for k in ["f", "t", "m", "r"]:
        a, b = tables[("main", 5, False, "bio1", k)], tables[("main", 5, True, "bio1", k)]
        L.append(f"| {RESP_ZH[k]} | {a['n']}, {a['n_species']} | {fmt(a)} | {a['beta_male']:+.4f} ({a['se_male']:.4f}) | "
                 f"{a['beta_year']:+.4f} ({a['se_year']:.4f}) | {b['n']} | {fmt(b)} | {b['beta_M']:+.3f} ({b['se_M']:.3f}) |")
    L += ["", "## 2. 主样本，其他温度（不含体重）", "", "| 温度 | 股骨 | 胫跗骨 | 跗跖骨 | r |", "|---|---|---|---|---|"]
    for temp in ["bio1", "bio5", "bio6"]:
        L.append(f"| {TEMP_ZH[temp]} | " + " | ".join(fmt(tables[("main", 5, False, temp, k)]) for k in ["f", "t", "m", "r"]) + " |")
    L += ["", "## 3. 样本与门槛（年均温，不含体重）", "", "| 样本 | 门槛 | n, 种 | 股骨 | 胫跗骨 | 跗跖骨 | r |", "|---|---|---|---|---|---|---|"]
    for sample in ["main", "resident", "all_months"]:
        for min_n in [5, 10]:
            a = tables[(sample, min_n, False, "bio1", "r")]
            L.append(f"| {SAMPLE_ZH[sample]} | ≥ {min_n} 件 | {a['n']}, {a['n_species']} | " +
                     " | ".join(fmt(tables[(sample, min_n, False, "bio1", k)]) for k in ["f", "t", "m", "r"]) + " |")
    L += ["", "## 4. 残差与随机斜率（主样本，不含体重，年均温）", "", "| 响应 | 残差 SD（log10） | 物种间斜率 SD |", "|---|---|---|"]
    for k in ["f", "t", "m", "r"]:
        a = tables[("main", 5, False, "bio1", k)]
        L.append(f"| {RESP_ZH[k]} | {a['resid_sd']:.4f} | {a.get('sd_slope', float('nan')):.4f} |")
    L += ["", "## 5. 读法", "",
          "- 残差 SD 含 Skelevision 测量误差（跗跖骨约 0.018）；性别系数是控制温度与年份后的雌雄差（log10），0.0043 = 1%。",
          "- 年份系数只是种内年代趋势的附带估计，不进判据。", ""]
    (OUT / "specimens_mixed.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
