"""热阀反事实 v1 第三步：真实腿 R 与去散热腿 C 的落震（设计 docs/design/热阀与落震_反事实闭环_设计_2026-10-28.md 第 3、5 节）。

两组各五条腿（R、C10、C20、C40、Ct20，来自 outputs/thermal_v1/legs.json）× 256 组关节参数（所有腿共用，配对）
× 6 个工况（触地速度两档 × 硬地、草地、湿沙）。工况由 R 定：载荷 = 体重的一半，触地速度按 Froude 数从 FFY 的 1.2 与 2.0 m/s
换到 R 的腿长，并对所有腿固定不变（同一只鸟、同样的落法）。稳健性：C20 的触地速度按它自己的腿长缩放。
主指标 p_F 可行率；判定按设计第 5 节（权衡说 / 免费阀说 / 反向，剂量单调性）。
在 EvoWM 根目录运行：python scripts/thermal/drop_thermal_v1.py [--workers 2] [--n-param 256] [--legs outputs/thermal_v1/legs.json]
输出 outputs/thermal_v1/：drop_runs.csv.gz、drop_decision.json、drop_report.md。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from evowm.engineering.drop import G, Criteria, Leg, bird_scenario, evaluate, froude_v0, param_box  # noqa: E402

DESIGN = ROOT / "docs/design/热阀与落震_反事实闭环_设计_2026-10-28.md"
SEED, N_PARAM, N_BOOT = 20261028, 256, 10000
V_FFY = (1.2, 2.0)
TERRAINS = ("concrete", "turf", "wetsand")
LEG_NAMES = ["R", "C10", "C20", "C40", "Ct20"]
DOSE = ["C10", "C20", "C40"]
MAIN = Criteria(g_cap=10.0, s_max_frac=0.15)
SENS = {"行程上限 10%": Criteria(g_cap=10.0, s_max_frac=0.10), "行程上限 20%": Criteria(g_cap=10.0, s_max_frac=0.20),
        "峰值上限 5 g": Criteria(g_cap=5.0, s_max_frac=0.15)}
MEANINGFUL = 0.05
GROUP_ZH = {"exposed": "露天组（早成鸟 P）", "arboreal": "树栖雀形目（X = 0）"}
LEG_ZH = {"R": "R 真实腿", "C10": "C 去散热 10 °C", "C20": "C 去散热 20 °C（主）", "C40": "C 去散热 40 °C", "Ct20": "C 只回推跗跖骨 20 °C"}


def leg_of(mm: dict) -> Leg:
    return Leg(mm["b_tmt"] / 1000.0, mm["b_tbt"] / 1000.0, mm["b_fem"] / 1000.0)


def pf_stats(cube: dict, names: list, K: np.ndarray, base: str = "R"):
    """cube[name] 形状 (工况, 参数)；配对差按参数组自助重抽样。"""
    pf = {n: float(cube[n].mean()) for n in names}
    boot = {n: cube[n][:, K].mean(axis=(0, 2)) for n in names}
    d = {n: {"diff": pf[n] - pf[base], "ci95": [float(v) for v in np.percentile(boot[n] - boot[base], [2.5, 97.5])]}
         for n in names if n != base}
    per_s = {n: [float(x) for x in cube[n].mean(axis=1)] for n in names}
    return pf, d, per_s


def verdict_q1(pf: dict, d: dict) -> tuple[str, dict]:
    """设计第 5 节：权衡说 / 免费阀说 / 反向 / 介于之间。"""
    d20, ci20 = d["C20"]["diff"], d["C20"]["ci95"]
    seq = [pf["R"]] + [pf[n] for n in DOSE]
    rho = float(spearmanr([0, 10, 20, 40], seq).statistic) if np.ptp(seq) > 0 else float("nan")
    mono_up = rho >= 0.9
    mono_down = rho <= -0.9
    small = all(abs(d[n]["diff"]) < MEANINGFUL for n in DOSE)
    if d20 >= MEANINGFUL and ci20[0] > 0 and mono_up:
        v = "权衡说成立：去散热的腿落震更好，且随剂量单调"
    elif small:
        v = "免费阀说成立：三个剂量下差异都小于 5 个百分点"
    elif d20 <= -MEANINGFUL and ci20[1] < 0:
        v = "反向：去散热的腿落震更差（更长的跗跖骨落地更好）" + ("，随剂量单调" if mono_down else "")
    else:
        v = "介于之间，按实描述"
    return v, {"rho_dose": rho, "monotone_up": bool(mono_up), "monotone_down": bool(mono_down), "all_small": bool(small)}


def run_group(g: str, info: dict, n_param: int, workers: int, out: Path):
    legs = {n: leg_of(info["legs_mm"][n]) for n in LEG_NAMES}
    R = legs["R"]
    body = info["body_mass_g"] / 1000.0
    scen = [bird_scenario(body, R, froude_v0(v, R.total), ter) for v in V_FFY for ter in TERRAINS]
    params = param_box(n_param, SEED, scen[0])
    t0 = time.time()
    res = evaluate(legs, params, scen, workers=workers)
    # 稳健性：C20 的触地速度按自己的腿长缩放（其余工况量不变，仍由 R 定）
    scen_fr = [bird_scenario(body, R, froude_v0(v, legs["C20"].total), ter) for v in V_FFY for ter in TERRAINS]
    res_fr = evaluate({"C20_froude": legs["C20"]}, params, scen_fr, workers=workers)
    print(f"{g}: {len(res) + len(res_fr)} 次落震，{time.time() - t0:.0f} 秒", flush=True)
    rows = [{k: v for k, v in r.items() if not k.startswith("sc_")} | {"group": g, "v0": r["sc_v0"], "terrain": r["sc_terrain"]}
            for r in res + res_fr]
    nS, nP = len(scen), len(params)
    idx = {(r["leg"], r["scenario"], r["j"]): r for r in res + res_fr}
    legs_all = dict(legs, C20_froude=legs["C20"])
    scen_of = {n: scen for n in LEG_NAMES} | {"C20_froude": scen_fr}

    def ok_cube(cr, names):
        return {n: np.array([[cr.ok(idx[(n, s, j)], legs_all[n], scen_of[n][s]) for j in range(nP)] for s in range(nS)])
                for n in names}

    rng = np.random.default_rng(0)
    K = rng.integers(0, nP, size=(N_BOOT, nP))
    names = LEG_NAMES + ["C20_froude"]
    cube = ok_cube(MAIN, names)
    pf, d, per_s = pf_stats(cube, names, K)
    verdict, dose = verdict_q1(pf, d)
    sens = {}
    for name, cr in SENS.items():
        p2, d2, _ = pf_stats(ok_cube(cr, names), names, K)
        sens[name] = {"pF": p2, "diff": d2}

    def per_j(metric, n):
        v = np.array([[idx[(n, s, j)].get(metric, np.nan) for j in range(nP)] for s in range(nS)], float)
        if metric == "a_res":
            v = v / G
        if metric == "leg_stroke":
            v = v / legs_all[n].total
        return np.nanmean(v, axis=0)

    second = {}
    for metric in ("eta", "a_res", "leg_stroke", "peak_grf_bw"):
        base_v = per_j(metric, "R")
        second[metric] = {}
        for n in names:
            if n == "R":
                continue
            x = per_j(metric, n) - base_v
            ok = np.isfinite(x)
            if ok.sum() == 0:
                second[metric][n] = {"median": float("nan"), "ci95": [float("nan")] * 2, "n": 0}
                continue
            xb = x[ok]
            med_b = [float(np.median(xb[rng.integers(0, len(xb), len(xb))])) for _ in range(2000)]
            second[metric][n] = {"median": float(np.median(xb)), "ci95": [float(v) for v in np.percentile(med_b, [2.5, 97.5])],
                                 "n": int(ok.sum())}
    fails = {n: float(np.mean([bool(idx[(n, s, j)].get("fail")) for s in range(nS) for j in range(nP)])) for n in names}
    dec = {"verdict": verdict, "dose": dose, "pF": pf, "diff": d, "pF_per_scenario": per_s, "sensitivity": sens,
           "secondary": second, "fail_rate": fails, "legs_mm": info["legs_mm"], "body_mass_g": info["body_mass_g"],
           "n_species": info["n_species"], "scenarios": [dict(v0=s.v0, terrain=s.terrain, m=s.m, r_foot=s.r_foot) for s in scen],
           "n_param": nP}
    return rows, dec


def report(decs: dict, out: Path, n_param: int):
    pct = lambda x: f"{100 * x:.1f}%"  # noqa: E731
    pp = lambda x: f"{100 * x['diff']:+.1f} [{100 * x['ci95'][0]:+.1f}, {100 * x['ci95'][1]:+.1f}]"  # noqa: E731
    L = ["# 热阀反事实 v1：去散热腿的落震", "",
         f"设计 `{DESIGN.relative_to(ROOT)}` 第 3、5 节。{n_param} 组关节参数（所有腿共用，配对）× 6 个工况"
         "（触地速度两档 × 硬地、草地、湿沙），载荷与速度由各组的真实腿 R 定。p_F = 四道判据都满足的参数组比例；"
         "方括号为按参数组自助重抽样 10,000 次的 95% 区间；有意义的差 = 5 个百分点。", ""]
    for g, dec in decs.items():
        sc = dec["scenarios"]
        L += [f"## {GROUP_ZH[g]}：{dec['verdict']}", "",
              f"{dec['n_species']} 种的中位数，体重 {dec['body_mass_g']:.0f} g（一条腿载荷 {1e3 * sc[0]['m']:.0f} g），"
              f"触地速度 {sc[0]['v0']:.2f} 与 {sc[-1]['v0']:.2f} m/s。剂量单调性 ρ = {dec['dose']['rho_dose']:.2f}。", "",
              "| 腿 | 三段 mm（跗跖、胫跗、股） | 可行率 | 与 R 之差（百分点） |", "|---|---|---|---|"]
        for n in LEG_NAMES + ["C20_froude"]:
            m = dec["legs_mm"]["C20" if n == "C20_froude" else n]
            lab = LEG_ZH.get(n, "C 20 °C，速度按自身腿长缩放（稳健性）")
            L.append(f"| {lab} | {m['b_tmt']:.1f}、{m['b_tbt']:.1f}、{m['b_fem']:.1f} | {pct(dec['pF'][n])} | "
                     f"{'' if n == 'R' else pp(dec['diff'][n])} |")
        L += ["", "各工况的可行率（" + "、".join(f"{s['v0']:.2f} m/s {s['terrain']}" for s in sc) + "）：", ""]
        for n in LEG_NAMES:
            L.append(f"- {LEG_ZH[n]}：" + "、".join(pct(x) for x in dec["pF_per_scenario"][n]))
        L += ["", "敏感性（与 R 之差，百分点）：", "", "| 判据 | " + " | ".join(LEG_ZH[n] for n in LEG_NAMES if n != "R") + " |",
              "|---" * len(LEG_NAMES) + "|"]
        for name, v in dec["sensitivity"].items():
            L.append(f"| {name} | " + " | ".join(pp(v["diff"][n]) for n in LEG_NAMES if n != "R") + " |")
        L += ["", "次要读数（每组参数 6 个工况平均，与 R 配对差的中位数）：", "",
              "| 腿 | 缓冲效率 η | 峰值合加速度（g） | 腿行程（占腿长） | 峰值地面反力（倍体重） |", "|---|---|---|---|---|"]
        for n in LEG_NAMES:
            if n == "R":
                continue
            L.append(f"| {LEG_ZH[n]} | " + " | ".join(
                f"{dec['secondary'][m][n]['median']:+.4f} [{dec['secondary'][m][n]['ci95'][0]:+.4f}, {dec['secondary'][m][n]['ci95'][1]:+.4f}]"
                for m in ("eta", "a_res", "leg_stroke", "peak_grf_bw")) + " |")
        L += ["", "失败率（求解、塌陷、深陷）：" + "，".join(f"{LEG_ZH.get(n, n)} {pct(v)}" for n, v in dec["fail_rate"].items()), ""]
    if {"exposed", "arboreal"} <= set(decs):
        de, da = decs["exposed"]["diff"]["C20"]["diff"], decs["arboreal"]["diff"]["C20"]["diff"]
        L += ["## Q2 分工", "",
              f"露天组 C20 − R = {100 * de:+.1f} 个百分点；树栖雀形目 C20 − R = {100 * da:+.1f} 个百分点。"
              + ("两组方向相反。" if de * da < 0 else "两组方向相同或有一组为零：落震只看长度不看类群，按设计第 5 节写明。")]
    (out / "drop_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legs", default=str(ROOT / "outputs/thermal_v1/legs.json"))
    ap.add_argument("--n-param", type=int, default=N_PARAM, help="只在测试时缩减")
    ap.add_argument("--groups", nargs="+", default=["exposed", "arboreal"])
    ap.add_argument("--out", default=str(ROOT / "outputs/thermal_v1"))
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    if not DESIGN.exists():
        sys.exit(f"缺少设计文档 {DESIGN.relative_to(ROOT)}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lj = json.loads(Path(args.legs).read_text(encoding="utf-8"))
    rows, decs = [], {}
    for g in args.groups:
        r, dec = run_group(g, lj["groups"][g], args.n_param, args.workers, out)
        rows += r
        decs[g] = dec
        (out / "drop_decision.json").write_text(json.dumps(decs, ensure_ascii=False, indent=1), encoding="utf-8")
    pd.DataFrame(rows).to_csv(out / "drop_runs.csv.gz", index=False)
    report(decs, out, args.n_param)


if __name__ == "__main__":
    main()
