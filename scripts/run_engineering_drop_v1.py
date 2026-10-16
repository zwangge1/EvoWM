"""工程验证 v1 第二步：原腿 O、推测腿 P_k、观测陆栖腿 T 的落震（预注册 docs/prereg/engineering_drop_v1.md 第 2 到 4 节）。

O、T：雀形目里栖树、地栖物种的三段 log10 长度与 log10 体重的逐列中位数（三段与体重都有的物种）。
P_k：O 加上 k 倍的模型变化量 Δ（outputs/engineering_v1/delta.json），k = 1、2、4、8。
口径 A（主）：三段按比例缩放到 O 的总长，只比比例；口径 B：体重固定在 O，扣除体重变化带来的长度变化（异速斜率 b_j 在
雀形目上拟合），不缩放。每条腿 × 256 组参数（所有腿共用）× 6 个工况各跑一次。
在 EvoWM 根目录运行：python scripts/run_engineering_drop_v1.py [--workers 16]
输出 outputs/engineering_v1/：legs.json、runs.csv.gz、decision.json、report.md。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evowm.engineering.drop import G, Criteria, Leg, bird_scenario, evaluate, froude_v0, param_box  # noqa: E402
from evowm.sim.regimes import regime_locomotor  # noqa: E402

PREREG = ROOT / "docs/prereg/engineering_drop_v1.md"
TABLE = "data/processed/species_traits.csv"
SEG = ["k_tarsometatarsus", "k_tibiotarsus", "k_femur"]      # Leg(tmt, tbt, fem) 的顺序
SEED, N_PARAM, KS, N_BOOT = 20261016, 256, (1, 2, 4, 8), 10000
V_FFY = (1.2, 2.0)
TERRAINS = ("concrete", "turf", "wetsand")
MAIN = Criteria(g_cap=10.0, s_max_frac=0.15)
SENS = {"行程上限 10%": Criteria(g_cap=10.0, s_max_frac=0.10), "行程上限 20%": Criteria(g_cap=10.0, s_max_frac=0.20),
        "峰值上限 5 g": Criteria(g_cap=5.0, s_max_frac=0.15)}
MEANINGFUL = 0.05


def group_medians(table: pd.DataFrame):
    t = table.copy()
    t["loco"] = regime_locomotor(t).to_numpy()
    ok = (t["order"] == "Passeriformes") & t[SEG + ["m_mass"]].notna().all(axis=1)
    p = t[ok]
    med = {lab: {c: float(p.loc[p.loco == lab, c].median()) for c in SEG + ["m_mass"]}
           for lab in ("insessorial", "terrestrial")}
    n = {lab: int((p.loco == lab).sum()) for lab in ("insessorial", "terrestrial")}
    b = {c: float(np.polyfit(p["m_mass"], p[c], 1)[0]) for c in SEG}
    return med, n, b, int(len(p))


def leg_of(log10_mm: dict) -> Leg:
    return Leg(*(10.0 ** log10_mm[c] / 1000.0 for c in SEG))


def build_legs(med, b, delta):
    o, t = med["insessorial"], med["terrestrial"]
    O = leg_of(o)
    A, B = {"O": O}, {}
    for k in KS:
        A[f"P{k}"] = leg_of({c: o[c] + k * delta[c] for c in SEG}).scaled(O.total)
        B[f"P{k}"] = leg_of({c: o[c] + k * (delta[c] - b[c] * delta["m_mass"]) for c in SEG})
    A["T"] = leg_of(t).scaled(O.total)
    dm = t["m_mass"] - o["m_mass"]
    B["T"] = leg_of({c: o[c] + (t[c] - o[c]) - b[c] * dm for c in SEG})
    return O, A, B, 10.0 ** o["m_mass"] / 1000.0


def verdict(d: float, ci) -> str:
    if ci[0] > 0 and d >= MEANINGFUL:
        return "更好"
    if ci[1] < 0 and d <= -MEANINGFUL:
        return "更差"
    return "未检出有意义的差异"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=TABLE, help="只在测试时换成模拟表")
    ap.add_argument("--delta", default=str(ROOT / "outputs/engineering_v1/delta.json"))
    ap.add_argument("--n-param", type=int, default=N_PARAM, help="只在测试时缩减")
    ap.add_argument("--out", default=str(ROOT / "outputs/engineering_v1"))
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    if not PREREG.exists():
        sys.exit(f"缺少预注册 {PREREG.relative_to(ROOT)}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dj = json.loads(Path(args.delta).read_text(encoding="utf-8"))
    if dj.get("table") != args.table:
        sys.exit(f"delta.json 来自 {dj.get('table')}，与 --table {args.table} 不同")
    delta = {c: float(dj["delta"][c]["mean"]) for c in SEG + ["m_mass"]}

    table = pd.read_csv(ROOT / args.table)
    med, n_sp, b, n_pas = group_medians(table)
    O, A, B, body = build_legs(med, b, delta)
    legs = {**A, **{f"B_{k}": v for k, v in B.items()}}
    scen = [bird_scenario(body, O, froude_v0(v, O.total), ter) for v in V_FFY for ter in TERRAINS]
    params = param_box(args.n_param, SEED, scen[0])
    (out / "legs.json").write_text(json.dumps({
        "species": n_sp, "n_passerines": n_pas, "medians_log10": med, "allometric_slopes": b, "delta": delta,
        "body_mass_kg": body,
        "legs_mm": {k: [1e3 * v.tmt, 1e3 * v.tbt, 1e3 * v.fem] for k, v in legs.items()},
        "scenarios": [dict(v0=s.v0, terrain=s.terrain, m=s.m, kc=s.kc, r_foot=s.r_foot, T=s.T, h=s.h) for s in scen],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    t0 = time.time()
    res = evaluate(legs, params, scen, workers=args.workers)
    print(f"{len(res)} 次落震，{time.time() - t0:.0f} 秒", flush=True)
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("sc_")} | {"v0": r["sc_v0"], "terrain": r["sc_terrain"]}
                       for r in res])
    df.to_csv(out / "runs.csv.gz", index=False)

    names = list(legs)
    nS, nP = len(scen), len(params)
    idx = {(r["leg"], r["scenario"], r["j"]): r for r in res}

    def ok_cube(cr):
        return {n: np.array([[cr.ok(idx[(n, s, j)], legs[n], scen[s]) for j in range(nP)] for s in range(nS)])
                for n in names}

    rng = np.random.default_rng(0)
    K = rng.integers(0, nP, size=(N_BOOT, nP))

    def pf_table(cube):
        pf = {n: float(cube[n].mean()) for n in names}
        boot = {n: cube[n][:, K].mean(axis=(0, 2)) for n in names}
        d = {}
        for n in names:
            if n == "O":
                continue
            x = boot[n] - boot["O"]
            d[n] = {"diff": pf[n] - pf["O"], "ci95": [float(v) for v in np.percentile(x, [2.5, 97.5])]}
        per_s = {n: [float(cube[n][s].mean()) for s in range(nS)] for n in names}
        return pf, d, per_s

    cube = ok_cube(MAIN)
    pf, d, per_s = pf_table(cube)
    e2 = verdict(d["P1"]["diff"], d["P1"]["ci95"])
    e1 = verdict(d["T"]["diff"], d["T"]["ci95"])
    seq = [pf["O"]] + [pf[f"P{k}"] for k in KS]
    rho = float(spearmanr([0, *KS], seq).statistic) if np.ptp(seq) > 0 else float("nan")
    e3 = "单调变好" if rho >= 0.9 else ("单调变差" if rho <= -0.9 else "非单调")
    same_dir = bool(np.sign(d["T"]["diff"]) == np.sign(d["P8"]["diff"])) if d["T"]["diff"] and d["P8"]["diff"] else False
    sens = {}
    for name, cr in SENS.items():
        p2, d2, _ = pf_table(ok_cube(cr))
        sens[name] = {"pF": p2, "diff": d2}

    # 次要读数：每组参数下 6 个工况平均，与 O 配对差的中位数与区间
    def per_j(metric, n):
        v = np.array([[idx[(n, s, j)].get(metric, np.nan) for j in range(nP)] for s in range(nS)], float)
        if metric == "a_res":
            v = v / G
        if metric == "leg_stroke":
            v = v / legs[n].total
        return np.nanmean(v, axis=0)

    second = {}
    for metric in ("eta", "a_res", "leg_stroke", "peak_grf_bw"):
        base = per_j(metric, "O")
        second[metric] = {}
        for n in names:
            if n == "O":
                continue
            x = per_j(metric, n) - base
            ok = np.isfinite(x)
            med_b = [float(np.median(x[ok][rng.integers(0, ok.sum(), ok.sum())])) for _ in range(2000)] if ok.sum() else [np.nan]
            second[metric][n] = {"median": float(np.median(x[ok])) if ok.sum() else float("nan"),
                                 "ci95": [float(v) for v in np.nanpercentile(med_b, [2.5, 97.5])], "n": int(ok.sum())}
    fails = {n: float(df.loc[df.leg == n, "fail"].fillna("").astype(str).ne("").mean()) for n in names}

    decision = {"E1_T": e1, "E2_P1": e2, "E3": e3, "E3_rho": rho, "T_same_direction_as_P8": same_dir,
                "E2_descriptive_only": e1 != "更好", "pF": pf, "diff": d, "pF_per_scenario": per_s,
                "sensitivity": sens, "secondary": second, "fail_rate": fails,
                "n_param": nP, "n_scenarios": nS, "table": args.table}
    (out / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 报告 ----
    pct = lambda x: f"{100 * x:.1f}%"  # noqa: E731
    pp = lambda x: f"{100 * x['diff']:+.1f} [{100 * x['ci95'][0]:+.1f}, {100 * x['ci95'][1]:+.1f}]"  # noqa: E731
    lab = {"O": "O 原腿", "P1": "P×1", "P2": "P×2", "P4": "P×4", "P8": "P×8", "T": "T 观测陆栖腿"}
    L = ["# 工程验证 v1：推测腿的落震", "",
         f"预注册 `docs/prereg/engineering_drop_v1.md`。雀形目三段与体重都有的 {n_pas} 种（栖树 {n_sp['insessorial']}、"
         f"地栖 {n_sp['terrestrial']}）。体重 {1e3 * body:.1f} g（一条腿载荷 {1e3 * scen[0].m:.1f} g），原腿总长 "
         f"{1e3 * O.total:.1f} mm。{nP} 组参数 × {nS} 个工况（触地速度 {scen[0].v0:.2f}、{scen[-1].v0:.2f} m/s × 硬地、草地、湿沙）。", "",
         f"**结论**：E2（主，P×1 对 O，口径 A）{e2}；E1（正对照，T 对 O）{e1}；E3（倍数扫描）{e3}（ρ = {rho:.2f}）。"
         + ("E1 不是更好，E2 只作描述。" if e1 != "更好" else ""), "",
         "Δ（log10，模型推测的栖树 → 地栖变化，雀形目窗口）：" + "，".join(f"{c} {delta[c]:+.4f}" for c in SEG + ["m_mass"]), "",
         "## 口径 A（总长相同，只比比例）", "", "| 腿 | 三段 mm（跗跖、胫跗、股） | 可行率 | 与 O 之差（百分点）|", "|---|---|---|---|"]
    for n in A:
        g = legs[n]
        L.append(f"| {lab[n]} | {1e3 * g.tmt:.1f}、{1e3 * g.tbt:.1f}、{1e3 * g.fem:.1f} | {pct(pf[n])} | "
                 f"{'' if n == 'O' else pp(d[n])} |")
    L += ["", "各工况的可行率（" + "、".join(f"{s.v0:.2f} m/s {s.terrain}" for s in scen) + "）：", ""]
    for n in A:
        L.append(f"- {lab[n]}：" + "、".join(pct(x) for x in per_s[n]))
    L += ["", "## 敏感性（口径 A，与 O 之差，百分点）", "", "| 判据 | " + " | ".join(lab[n] for n in A if n != "O") + " |",
          "|---" * len(A) + "|"]
    for name, v in sens.items():
        L.append(f"| {name} | " + " | ".join(pp(v["diff"][n]) for n in A if n != "O") + " |")
    L += ["", "## 次要读数（口径 A，每组参数 6 个工况平均，与 O 配对差的中位数）", "",
          "| 腿 | 缓冲效率 η | 峰值合加速度（g）| 腿行程（占腿长）| 峰值地面反力（倍体重）|", "|---|---|---|---|---|"]
    for n in A:
        if n == "O":
            continue
        L.append(f"| {lab[n]} | " + " | ".join(
            f"{second[m][n]['median']:+.4f} [{second[m][n]['ci95'][0]:+.4f}, {second[m][n]['ci95'][1]:+.4f}]"
            for m in ("eta", "a_res", "leg_stroke", "peak_grf_bw")) + " |")
    L += ["", "## 口径 B（体重固定在 O，不约束总长，只作描述）", "", "| 腿 | 三段 mm | 可行率 | 与 O 之差（百分点）|", "|---|---|---|---|"]
    for k in B:
        n = f"B_{k}"
        g = legs[n]
        L.append(f"| {lab[k]} | {1e3 * g.tmt:.1f}、{1e3 * g.tbt:.1f}、{1e3 * g.fem:.1f} | {pct(pf[n])} | {pp(d[n])} |")
    L += ["", "失败率（求解、塌陷、深陷）：" + "，".join(f"{n} {pct(v)}" for n, v in fails.items())]
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(json.dumps({"E1_T": e1, "E2_P1": e2, "E3": e3, "rho": rho}, ensure_ascii=False))
    print(f"报告：{out / 'report.md'}")


if __name__ == "__main__":
    main()
