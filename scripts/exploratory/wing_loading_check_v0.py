"""翼载荷核对：用实测翅面积检验里程碑 1 里的翼载荷代理，并在实测子集上重看翼载荷与跗跖长的关系（探索性）。

实测来源：Alerstam et al. 2007 Protocol S1（体重 kg、翼展 m、翅面积 m²，约 130 种）；Tanaka 2018 Dryad 表（约 110 行、按个体，
有翅面积、翼展、体重）。两表按种取均值，Alerstam 优先。名字按 BirdTree（下划线）匹配，少量同义名手工对照。
做四件事：
1. 代理 logWL_proxy = M − W − S1（AVONET，log10）与实测 logWL = log10(体重 / 翅面积) 的相关与回归斜率。
2. 翅面积能否由 AVONET 预测：log10 面积 ~ 翅长 + 次级飞羽长（+ 体重），给 R² 与系数；用它给全部 AVONET 物种一个校准后的翼载荷。
3. 实测子集上的 PGLS：跗跖 ~ 体重 + 实测 logWL + HWI（+ 生活方式），看实测翼载荷的系数符号。
4. 全体 AVONET 上用校准翼载荷重跑里程碑 1 的模型 C。
结果 outputs/wing_loading_check_v0/report.md、matched.csv。在 EvoWM 根目录运行：python scripts/exploratory/wing_loading_check_v0.py
"""
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/exploratory"))
from evowm.data.tree import read_trees  # noqa: E402
from evowm.eval.pgls import pgls_lambda, tip_vcv  # noqa: E402
from evowm.sim.regimes import LIFESTYLE  # noqa: E402
from flight_service_v0 import GLS, design, load  # noqa: E402

OUT = ROOT / "outputs/wing_loading_check_v0"
ALERSTAM = ROOT / "data/raw/alerstam2007/pbio.0050197.sd001.pdf"
TANAKA = ROOT / "data/raw/tanaka2018/online+data+package.xlsx"
SYN = {  # 旧名 → BirdTree 名（只列表里出现、且 BirdTree 用另一个名字的）
    "Larus_ridibundus": "Chroicocephalus_ridibundus", "Larus_minutus": "Hydrocoloeus_minutus",
    "Sterna_caspia": "Hydroprogne_caspia", "Corvus_pica": "Pica_pica", "Chrysotis_amazonica": "Amazona_amazonica",
    "Phoenicopterus_ruber": "Phoenicopterus_roseus", "Parus_caeruleus": "Cyanistes_caeruleus",
    "Parus_ater": "Periparus_ater", "Parus_palustris": "Poecile_palustris", "Carduelis_chloris": "Chloris_chloris",
    "Carduelis_spinus": "Spinus_spinus", "Carduelis_flammea": "Acanthis_flammea", "Miliaria_calandra": "Emberiza_calandra",
    "Delichon_urbica": "Delichon_urbicum", "Hirundo_rustica": "Hirundo_rustica", "Sterna_sandvicensis": "Thalasseus_sandvicensis",
    "Casmerodius_albus": "Ardea_alba", "Egretta_alba": "Ardea_alba", "Philomachus_pugnax": "Philomachus_pugnax",
    "Tetrao_tetrix": "Lyrurus_tetrix", "Anas_penelope": "Anas_penelope", "Regulus_regulus": "Regulus_regulus",
}


def parse_alerstam() -> pd.DataFrame:
    txt = subprocess.run(["pdftotext", "-layout", str(ALERSTAM), "-"], capture_output=True, text=True, check=True).stdout
    rows = []
    num = re.compile(r"-?\d+\.?\d*")
    for line in txt.splitlines():
        m = re.search(r"([A-Z][a-z]+ [a-z]+)( •)?\s+(-?\d.*)$", line)
        if not m:
            continue
        sp = m.group(1).replace(" ", "_")
        v = [float(x) for x in num.findall(m.group(3))]
        rec = {"species": sp, "mass_kg": np.nan, "span_m": np.nan, "area_m2": np.nan}

        def is_int(x):
            return abs(x - round(x)) < 1e-9

        ok = False
        if len(v) >= 6 and is_int(v[-2]) and is_int(v[-4]) and is_int(v[-6]) and 0.004 <= v[-1] <= 1.5 \
                and 0.1 <= v[-3] <= 3.6 and 0.003 <= v[-5] <= 15:
            rec.update(mass_kg=v[-5], span_m=v[-3], area_m2=v[-1])
            ok = True
        elif len(v) >= 4 and is_int(v[-2]) and is_int(v[-4]) and 0.1 <= v[-1] <= 3.6 and 0.003 <= v[-3] <= 15:
            rec.update(mass_kg=v[-3], span_m=v[-1])
            ok = True
        elif len(v) >= 2 and is_int(v[-2]) and 0.003 <= v[-1] <= 15:
            rec.update(mass_kg=v[-1])
            ok = True
        if ok:
            rows.append(rec)
    d = pd.DataFrame(rows)
    # 同一种出现两行（本组与 • 组）时取有面积的那行，否则取第一行
    d = d.sort_values("area_m2", na_position="last").drop_duplicates("species", keep="first")
    d["source"] = "alerstam2007"
    return d


def parse_tanaka() -> pd.DataFrame:
    d = pd.read_excel(TANAKA, sheet_name="Sheet1", header=None)
    d = d.iloc[7:, [1, 2, 3, 4]].copy()
    d.columns = ["species", "mass_kg", "area_m2", "span_m"]
    d = d.dropna(subset=["species"])
    for c in ("mass_kg", "area_m2", "span_m"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["species"] = d["species"].astype(str).str.strip().str.replace(" ", "_")
    g = d.groupby("species", as_index=False)[["mass_kg", "area_m2", "span_m"]].mean()
    g["source"] = "tanaka2018"
    return g


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    a, t = parse_alerstam(), parse_tanaka()
    real = pd.concat([a, t]).sort_values("source").drop_duplicates("species", keep="first")   # alerstam 优先
    real = real.dropna(subset=["area_m2", "mass_kg"])
    real["tip"] = real["species"].map(lambda s: SYN.get(s, s))
    d = load()                                                   # AVONET 实测物种（与里程碑 1 相同）
    m = real.merge(d, on="tip", how="inner")
    unmatched = sorted(set(real["tip"]) - set(d["tip"]))
    m["logWL_real"] = np.log10(m["mass_kg"] * 9.81 / m["area_m2"])     # N/m²
    m["logArea"] = np.log10(m["area_m2"] * 1e6)                        # mm²
    m["logSpan"] = np.log10(m["span_m"] * 1000)
    m.to_csv(OUT / "matched.csv", index=False)

    L = ["# 翼载荷核对：实测翅面积对代理", "", "脚本 `scripts/exploratory/wing_loading_check_v0.py`。探索性分析。", "",
         f"- Alerstam 2007 解析出 {len(a)} 种（有翅面积 {int(a.area_m2.notna().sum())}），Tanaka 2018 {len(t)} 种；"
         f"合并有翅面积与体重 {len(real)} 种，与 AVONET 实测物种匹配上 {len(m)} 种。未匹配 {len(unmatched)} 种：" +
         "、".join(unmatched[:40]) + ("…" if len(unmatched) > 40 else ""), ""]

    # 1. 代理与实测
    r = np.corrcoef(m["logWL"], m["logWL_real"])[0, 1]
    b = np.polyfit(m["logWL"], m["logWL_real"], 1)
    L += ["## 1. 代理翼载荷（体重 ÷ 翅长 × 次级飞羽长）与实测翼载荷", "",
          f"- 相关系数 {r:.3f}；实测对代理的斜率 {b[0]:.3f}（代理若正确应接近 1）。",
          f"- 两边体重差异（AVONET 与实测源，log10）：均值 {np.mean(m['m_mass'] - np.log10(m['mass_kg'] * 1000)):+.3f}，"
          f"标准差 {np.std(m['m_mass'] - np.log10(m['mass_kg'] * 1000)):.3f}。", ""]

    # 2. 翅面积能否由 AVONET 预测
    def ols(y, X, names):
        X = np.column_stack([np.ones(len(y))] + X)
        beta, res, *_ = np.linalg.lstsq(X, y, rcond=None)
        yhat = X @ beta
        r2 = 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
        return dict(zip(["intercept"] + names, beta)), r2, np.std(y - yhat)

    f1, r2_1, s1 = ols(m["logArea"].to_numpy(), [m["m_wing"].to_numpy(), m["m_secondary1"].to_numpy()], ["W", "S1"])
    f2, r2_2, s2 = ols(m["logArea"].to_numpy(), [m["m_wing"].to_numpy(), m["m_secondary1"].to_numpy(), m["m_mass"].to_numpy()],
                       ["W", "S1", "M"])
    f3, r2_3, s3 = ols(m["logSpan"].to_numpy(), [m["m_wing"].to_numpy()], ["W"])
    L += ["## 2. 由 AVONET 预测翅面积与翼展", "",
          f"- log 面积 ~ 翅长 + 次级飞羽长：W {f1['W']:.2f}，S1 {f1['S1']:.2f}，R² {r2_1:.3f}，残差标准差 {s1:.3f}（代理假设两系数都为 1）。",
          f"- 再加体重：W {f2['W']:.2f}，S1 {f2['S1']:.2f}，M {f2['M']:.2f}，R² {r2_2:.3f}，残差标准差 {s2:.3f}。",
          f"- log 翼展 ~ 翅长：斜率 {f3['W']:.2f}，R² {r2_3:.3f}。", ""]

    # 3. 实测子集的 PGLS
    tree = read_trees(ROOT / "data/processed/trees/hackett_t021.nwk", max_trees=1)[0]
    pos = {n: i for i, n in enumerate(tree.tip_names)}
    rows = np.array([pos[x] for x in m["tip"]])
    C = tip_vcv(tree)[np.ix_(rows, rows)]
    T = m["m_tarsus"].to_numpy()
    fits = {}
    for name, cols, nm in (("体重 + 实测翼载荷", [m["m_mass"], m["logWL_real"]], ["M", "logWL_real"]),
                           ("体重 + 实测翼载荷 + HWI", [m["m_mass"], m["logWL_real"], m["m_hwi"]], ["M", "logWL_real", "HWI"]),
                           ("体重 + 代理翼载荷 + HWI", [m["m_mass"], m["logWL"], m["m_hwi"]], ["M", "logWL_proxy", "HWI"])):
        X = np.column_stack([np.ones(len(m))] + [c.to_numpy(float) for c in cols])
        names = ["intercept"] + nm
        life = [(m["life"] == c).to_numpy(float) for c in LIFESTYLE if c != "insessorial" and (m["life"] == c).sum() >= 5]
        lnames = [f"life_{c}" for c in LIFESTYLE if c != "insessorial" and (m["life"] == c).sum() >= 5]
        X = np.column_stack([X] + life) if life else X
        fits[name] = pgls_lambda(T, X, C, names + lnames)
    L += [f"## 3. 实测子集（{len(m)} 种）的 PGLS：跗跖 ~ …（生活方式类别 ≥ 5 种的作协变量，栖树为基准）", "",
          "| 模型 | λ | " + " | ".join(["M", "logWL_real", "logWL_proxy", "HWI"]) + " |", "|---|---|---|---|---|---|"]
    for name, f in fits.items():
        cells = [f"{f['coef'][k]['beta']:+.3f} ({f['coef'][k]['se']:.3f})" if k in f["coef"] else "·"
                 for k in ["M", "logWL_real", "logWL_proxy", "HWI"]]
        L.append(f"| {name} | {f['lambda']:.2f} | " + " | ".join(cells) + " |")
    L.append("")

    # 4. 全体 AVONET 用校准翼载荷重跑模型 C
    d["logWL_cal"] = d["m_mass"] - (f2["intercept"] + f2["W"] * d["m_wing"] + f2["S1"] * d["m_secondary1"] + f2["M"] * d["m_mass"])
    rows_all = np.array([pos[x] for x in d["tip"]])
    Call = tip_vcv(tree)[np.ix_(rows_all, rows_all)]
    XC, nC = design(d, with_wl=True)
    j = nC.index("logWL")
    XC[:, j] = d["logWL_cal"].to_numpy()
    nC[j] = "logWL_cal"
    gls = GLS(Call, 0.929)                                   # 里程碑 1 模型 A 的 λ
    fC = gls.fit(d["m_tarsus"].to_numpy(), XC, nC)
    L += [f"## 4. 全体 AVONET（{len(d)} 种）：模型 C 换成校准翼载荷（面积由翅长、次级飞羽长、体重预测）", "",
          f"- logWL_cal 系数 {fC['logWL_cal'][0]:+.4f} ({fC['logWL_cal'][1]:.4f})；HWI {fC['HWI'][0]:+.4f} ({fC['HWI'][1]:.4f})；"
          f"M {fC['M'][0]:+.4f}。里程碑 1 里代理翼载荷的系数是 −0.229。", "",
          "## 怎么看", "",
          "- 第 1、2 节回答代理行不行：相关低或斜率远离 1，说明里程碑 1 的负号可能是代理造成的。",
          "- 第 3 节是决定性的：实测翼载荷的系数若仍为负或不显著，起跳要求不决定腿长，本线关闭。",
          "- 第 4 节只是把校准后的代理推到全体，供参考。"]
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
