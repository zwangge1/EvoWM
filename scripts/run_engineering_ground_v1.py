"""工程验证 v2（预注册 docs/prereg/engineering_ground_v1.md）：方向检验。地面任务筛选（第一步）与推测方向检验（第二步）。

比例平面：总长固定时三段比例的二维平面（shape.py），原腿 O 在原点。
  T 方向：观测陆栖腿 T（地栖雀形目物种中位数）相对 O 的方向；P 方向：模型推测的变化量 Δ（v1 已算）的方向。
  r_T：T 离 O 的距离（"栖树到地栖"的比例改动幅度）。沿一个方向走 m·r_T（m = 1、2、4、8）得到 T_m、P_m；
  m = 8 为"工程尺度"的放大。方向环：半径 r_T 与 8·r_T 上各 24 个方向，代表幅度相同、方向随机的改动。
读数（direction.py）：F 峰值地面反力（相对变化）、E 能量损失、V 前向速度保持；改善量越大越好。
  只问方向：改善量的区间下限 > 0 为"上坡"，上限 < 0 为"下坡"，其余为"不确定"；幅度如实报告，不设门槛。
第一步（筛选，参数种子 SEED1）：只跑 O、T_1、T_8，任务 R、L、H、S。某任务的某读数通过：T_8 为上坡。
第二步（检验，参数种子 SEED2）：只对有读数通过的任务。对通过的读数判定：
  E1（复现）：T_8 仍为上坡；E2（主）：P_8 上坡、下坡或不确定；E3：P_m 的改善随 m 单调累积（ρ ≥ 0.8）；
  E4：8·r_T 方向环上 P_8 的百分位（≥ 0.9 优于随机，≤ 0.1 劣于随机）、与最陡方向的夹角、效率；
  E5（描述）：落震（v1 引擎、工况、判据与参数）上的同样方向环。
在 EvoWM 根目录运行：python scripts/run_engineering_ground_v1.py [--workers 32]
输出 outputs/engineering_ground_v1/：legs.json、stage1_runs.csv.gz、stage1.json、stage2_<任务>_runs.csv.gz、
drop_ring_runs.csv.gz、decision.json、report.md。
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evowm.engineering import drop as D  # noqa: E402
from evowm.engineering.direction import METRICS, boot_means, gain_table, ring_summary  # noqa: E402
from evowm.engineering.drop import Leg  # noqa: E402
from evowm.engineering.run import RunCriteria, evaluate_run, ground_scenario, run_box  # noqa: E402
from evowm.engineering.shape import leg_at, ring, shape_xy  # noqa: E402
from evowm.sim.regimes import regime_locomotor  # noqa: E402

PREREG = ROOT / "docs/prereg/engineering_ground_v1.md"
TABLE = "data/processed/species_traits.csv"
DELTA = "outputs/engineering_v1/delta.json"
DELTA_MD5 = "75e16acc751d8918d4d4f75a247bb0c1"      # v1 的 delta.json（预注册第 3 节）
SEG = ["k_tarsometatarsus", "k_tibiotarsus", "k_femur"]      # Leg(tmt, tbt, fem) 的顺序
KS = (1, 2, 4, 8)                          # v1 的倍数（只用于核对与口径 B）
MULT = (1, 2, 4, 8)                        # 沿方向走 m·r_T
SEED1, SEED2, N_PARAM, N_BOOT, N_DIR = 20261017, 20261018, 256, 10000, 24
SPEEDS = (0.8, 1.2)                        # 前向速度，单位 √(gL)
DHS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)       # 踩空下落高度，单位腿长
TASKS = {                                  # 任务 → 每条腿的载荷（体重的倍数）与地面
    "R": dict(loads=(1.0,), terrain="turf", name="奔跑踩空"),
    "L": dict(loads=(1.5, 2.0), terrain="turf", name="负重奔跑踩空"),
    "H": dict(loads=(0.5,), terrain="turf", name="跳跃踩空"),
    "S": dict(loads=(1.0,), terrain="wetsand", name="软地奔跑踩空"),
}
MNAME = {"F": "峰值地面反力", "E": "能量损失", "V": "速度保持"}
CRIT = RunCriteria(vx_min=0.5, slip_frac=0.5)
REF = 0.05                                # 只作描述：幅度是否达到 5%（F）或 0.05（E、V）
PCT_HI, PCT_LO, RHO_MIN = 0.90, 0.10, 0.8


# ---------------------------------------------------------------- 腿
def group_medians(table: pd.DataFrame):
    """与 v1（scripts/run_engineering_drop_v1.py）相同。"""
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


def build(med, b, delta, n_dir=N_DIR):
    o, t = med["insessorial"], med["terrestrial"]
    O = leg_of(o)
    v1 = {"O": O, "T": leg_of(t).scaled(O.total),
          **{f"P{k}": leg_of({c: o[c] + k * delta[c] for c in SEG}).scaled(O.total) for k in KS}}
    xy_t, xy_p = shape_xy(v1["T"], O), shape_xy(v1["P1"], O)
    r_t, r_p = float(np.hypot(*xy_t)), float(np.hypot(*xy_p))
    e_t, e_p = xy_t / r_t, xy_p / r_p
    legs = {"O": O, "Pk1": v1["P1"]}
    for m in MULT:
        legs[f"T{m}"] = leg_at(O, m * r_t * e_t)
        legs[f"P{m}"] = leg_at(O, m * r_t * e_p)
    rings = {m: ring(O, m * r_t, n_dir) for m in (1, 8)}
    B = {f"B_P{k}": leg_of({c: o[c] + k * (delta[c] - b[c] * delta["m_mass"]) for c in SEG}) for k in KS}
    dm = t["m_mass"] - o["m_mass"]
    B["B_T"] = leg_of({c: o[c] + (t[c] - o[c]) - b[c] * dm for c in SEG})
    geo = dict(xy_T=xy_t.tolist(), xy_P1=xy_p.tolist(), r_T=r_t, r_P1=r_p, k_of_P_at_rT=r_t / r_p,
               theta_T=float(np.arctan2(xy_t[1], xy_t[0])), theta_P=float(np.arctan2(xy_p[1], xy_p[0])))
    return O, v1, legs, rings, B, geo, 10.0 ** o["m_mass"] / 1000.0


def ring_angles(names) -> np.ndarray:
    return np.radians([int(n.rsplit("_", 1)[1]) for n in names])


# ---------------------------------------------------------------- 仿真
def task_scenarios(task: str, body: float, O: Leg):
    scen, tags = [], []
    for load in TASKS[task]["loads"]:
        for s in SPEEDS:
            for dh in DHS:
                scen.append(ground_scenario(body, O, s, dh, TASKS[task]["terrain"], load))
                tags.append(dict(load=load, speed=s, dh=dh))
    return scen, tags


def run_task(legs: dict, task: str, body: float, O: Leg, seed: int, n_param: int, workers: int) -> pd.DataFrame:
    scen, tags = task_scenarios(task, body, O)
    params = run_box(n_param, seed, scen[0])
    res = evaluate_run(legs, params, scen, workers=workers)
    rows = []
    for r in res:
        row = {k: v for k, v in r.items() if not k.startswith("sc_")}
        row.update(task=task, **tags[r["scenario"]])
        row["ok"] = bool(CRIT.ok(r, legs[r["leg"]]))
        rows.append(row)
    return pd.DataFrame(rows)


def pass_rates(df: pd.DataFrame, names, base="O", seed=0) -> dict:
    """通过率与 O 之差（描述）；区间：对参数组自助重抽样，同一次抽样用于所有腿。"""
    piv = df.groupby(["leg", "j"]).ok.mean().unstack("j")
    n = piv.shape[1]
    K = np.random.default_rng(seed).integers(0, n, size=(N_BOOT, n))
    out = {}
    for name in names:
        x = piv.loc[name].to_numpy() - piv.loc[base].to_numpy()
        out[name] = {"p": float(piv.loc[name].mean()), "diff": float(x.mean()),
                     "ci95": [float(v) for v in np.percentile(x[K].mean(axis=1), [2.5, 97.5])]}
    return out


def strip(B: dict) -> dict:
    return {k: {kk: vv for kk, vv in v.items() if kk != "boot"} for k, v in B.items()}


def uphill(g: dict) -> bool:
    return bool(g["ci95"][0] > 0)


def slope(g: dict) -> str:
    return "上坡" if g["ci95"][0] > 0 else "下坡" if g["ci95"][1] < 0 else "不确定"


def zscore(g: dict) -> float:
    sd = float(np.std(g["boot"])) if "boot" in g else float("nan")
    return g["mean"] / sd if sd > 0 else float("nan")


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- 主流程
def main():
    global N_BOOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=TABLE, help="只在测试时换成模拟表")
    ap.add_argument("--delta", default=str(ROOT / DELTA))
    ap.add_argument("--check-legs", default=str(ROOT / "outputs/engineering_v1/legs.json"),
                    help="v1 的 legs.json，核对重算的 O、T、P×k 与 v1 一致；测试时给空串跳过")
    ap.add_argument("--skip-md5", action="store_true", help="只在测试时使用")
    ap.add_argument("--n-param", type=int, default=N_PARAM, help="只在测试时缩减")
    ap.add_argument("--n-dir", type=int, default=N_DIR, help="只在测试时缩减")
    ap.add_argument("--n-boot", type=int, default=N_BOOT, help="只在测试时缩减")
    ap.add_argument("--tasks", default=",".join(TASKS), help="只在测试时缩减")
    ap.add_argument("--force-pass", default="", help="只在测试时使用：强制某些任务进入第二步，如 R:F")
    ap.add_argument("--out", default=str(ROOT / "outputs/engineering_ground_v1"))
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    N_BOOT = args.n_boot
    if not PREREG.exists():
        sys.exit(f"缺少预注册 {PREREG.relative_to(ROOT)}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dpath = Path(args.delta)
    if not args.skip_md5 and md5(dpath) != DELTA_MD5:
        sys.exit(f"{dpath} 的 md5 与预注册不同，停止")
    dj = json.loads(dpath.read_text(encoding="utf-8"))
    if dj.get("table") != args.table:
        sys.exit(f"delta.json 来自 {dj.get('table')}，与 --table {args.table} 不同")
    delta = {c: float(dj["delta"][c]["mean"]) for c in SEG + ["m_mass"]}
    med, n_sp, b, n_pas = group_medians(pd.read_csv(ROOT / args.table))
    O, v1, legs, rings, B, geo, body = build(med, b, delta, args.n_dir)
    if args.check_legs:
        old = json.loads(Path(args.check_legs).read_text(encoding="utf-8"))["legs_mm"]
        for k, g in v1.items():
            if np.max(np.abs(1e3 * np.array([g.tmt, g.tbt, g.fem]) - np.array(old[k]))) > 1e-9:
                sys.exit(f"重算的 {k} 与 v1 的 legs.json 不一致，停止")
        if np.max(np.abs(1e3 * np.array([legs["T1"].tmt - v1["T"].tmt, legs["T1"].tbt - v1["T"].tbt,
                                         legs["T1"].fem - v1["T"].fem]))) > 1e-9:
            sys.exit("T_1 与 T 不一致，停止")
        print("自检：O、T、P×k 与 v1 一致；T_1 即 T", flush=True)
    ring_legs = {n: g for r in rings.values() for n, g in r.items()}
    every = {**legs, **ring_legs, **B}
    (out / "legs.json").write_text(json.dumps({
        "species": n_sp, "n_passerines": n_pas, "medians_log10": med, "allometric_slopes": b, "delta": delta,
        "body_mass_kg": body, "geometry": geo,
        "legs_mm": {k: [1e3 * v.tmt, 1e3 * v.tbt, 1e3 * v.fem] for k, v in every.items()},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    tasks = args.tasks.split(",")
    dec = {"tasks": tasks, "geometry": geo, "seeds": [SEED1, SEED2], "n_param": args.n_param, "n_dir": args.n_dir}

    # ---- 第一步：筛选（只跑 O、T_1、T_8）
    t0 = time.time()
    s1, frames = {}, []
    for task in tasks:
        df = run_task({"O": O, "T1": legs["T1"], "T8": legs["T8"]}, task, body, O, SEED1, args.n_param, args.workers)
        frames.append(df)
        s1[task] = {"pass_rate": pass_rates(df, ["T1", "T8"])}
        for m in METRICS:
            Bm = boot_means(gain_table(df, ["T1", "T8"], "O", m), N_BOOT)
            s1[task][m] = {**strip(Bm), "pass": uphill(Bm["T8"]), "z_T8": zscore(Bm["T8"])}
            print(f"筛选 {task} {m}：T_8 {Bm['T8']['mean']:+.4f} [{Bm['T8']['ci95'][0]:+.4f}, {Bm['T8']['ci95'][1]:+.4f}]，"
                  f"T_1 {Bm['T1']['mean']:+.4f}，{'上坡，通过' if s1[task][m]['pass'] else '未通过'}"
                  f"（{time.time() - t0:.0f} 秒）", flush=True)
    pd.concat(frames).to_csv(out / "stage1_runs.csv.gz", index=False)
    passed = [(t, m) for t in tasks for m in METRICS if s1[t][m]["pass"]]
    for item in filter(None, args.force_pass.split(",")):
        t, m = item.split(":")
        if (t, m) not in passed:
            passed.append((t, m))
            print(f"测试：强制 {t} {m} 进入第二步", flush=True)
    primary = max(passed, key=lambda tm: s1[tm[0]][tm[1]]["z_T8"]) if passed else None
    (out / "stage1.json").write_text(json.dumps({"screen": s1, "passed": passed, "primary": primary},
                                                ensure_ascii=False, indent=2), encoding="utf-8")
    dec.update(screen=s1, passed=passed, primary=primary, stage2={})
    if not passed:
        print("没有任务与读数通过筛选，到此为止", flush=True)

    # ---- 第二步：检验推测方向（换一批参数）
    for task in dict.fromkeys(t for t, _ in passed):
        L2 = {**legs, **ring_legs, **B}
        df = run_task(L2, task, body, O, SEED2, args.n_param, args.workers)
        df.to_csv(out / f"stage2_{task}_runs.csv.gz", index=False)
        main_names = ["Pk1", *(f"T{m}" for m in MULT), *(f"P{m}" for m in MULT)]
        res = {"pass_rate": pass_rates(df, main_names + list(B))}
        for m in METRICS:
            Bm = boot_means(gain_table(df, [n for n in L2 if n != "O"], "O", m), N_BOOT)
            rr = {}
            for mult, rg in rings.items():
                names = list(rg)
                rr[f"x{mult}"] = ring_summary(Bm, names, ring_angles(names),
                                              {"P": (f"P{mult}", geo["theta_P"]), "T": (f"T{mult}", geo["theta_T"])})
            pg = [Bm[f"P{mm}"]["mean"] for mm in MULT]
            rho = float(spearmanr(MULT, pg).statistic) if np.ptp(pg) > 0 else float("nan")
            r8 = rr["x8"]
            item = {"gains": {k: v for k, v in strip(Bm).items() if k in main_names + list(B)}, "rings": rr,
                    "E3_rho": rho, "tested": (task, m) in passed}
            e1 = uphill(Bm["T8"])
            item["E1"] = "复现" if e1 else "未复现"
            item["E2"] = slope(Bm["P8"]) if e1 else f"{slope(Bm['P8'])}（E1 未复现，只作描述）"
            item["P1_biological"] = slope(Bm["P1"])
            item["reaches_ref"] = {"T8": bool(Bm["T8"]["mean"] >= REF), "P8": bool(Bm["P8"]["mean"] >= REF)}
            item["E3"] = ("单调累积" if rho >= RHO_MIN else "不单调") + ("" if e1 else "（只作描述）")
            pct = r8["P"]["percentile"]
            e4 = "优于随机" if pct >= PCT_HI else "劣于随机" if pct <= PCT_LO else "与随机无异"
            item["E4"] = e4 if e1 else f"{e4}（只作描述）"
            if (task, m) not in passed:
                for k in ("E1", "E2", "E3", "E4"):
                    item[k] = "未通过筛选，只作描述"
            res[m] = item
            print(f"检验 {task} {m}：E1 {item['E1']}，E2 {item['E2']}（P_8 {Bm['P8']['mean']:+.4f} "
                  f"[{Bm['P8']['ci95'][0]:+.4f}, {Bm['P8']['ci95'][1]:+.4f}]），E3 {item['E3']}（ρ = {rho:.2f}），"
                  f"E4 {item['E4']}（百分位 {pct:.2f}，夹角 {r8['P']['angle_to_steepest_deg']:.0f}°，"
                  f"效率 {r8['P']['efficiency']:.2f}）", flush=True)
        dec["stage2"][task] = res

    # ---- E5：落震上的同样方向环（v1 引擎、工况、判据与参数），只作描述
    if passed:
        L5 = {"O": O, **{f"{d}{m}": legs[f"{d}{m}"] for d in "TP" for m in (1, 8)}, **ring_legs}
        scen = [D.bird_scenario(body, O, D.froude_v0(v, O.total), ter) for v in (1.2, 2.0)
                for ter in ("concrete", "turf", "wetsand")]
        params = D.param_box(args.n_param, 20261016, scen[0])
        res = D.evaluate(L5, params, scen, workers=args.workers)
        crit = D.Criteria(g_cap=10.0, s_max_frac=0.15)
        dfd = pd.DataFrame([dict(leg=r["leg"], j=r["j"], scenario=r["scenario"], a_res=r["a_res"],
                                 leg_stroke=r["leg_stroke"], fail=r["fail"],
                                 ok=bool(crit.ok(r, L5[r["leg"]], scen[r["scenario"]]))) for r in res])
        dfd.to_csv(out / "drop_ring_runs.csv.gz", index=False)
        piv = dfd.groupby(["leg", "j"]).ok.mean().unstack("j")
        G = {n: piv.loc[n].to_numpy() - piv.loc["O"].to_numpy() for n in L5 if n != "O"}
        Bd = boot_means(G, N_BOOT)
        e5 = {}
        for mult, rg in rings.items():
            names = list(rg)
            e5[f"x{mult}"] = ring_summary(Bd, names, ring_angles(names),
                                          {"P": (f"P{mult}", geo["theta_P"]), "T": (f"T{mult}", geo["theta_T"])})
        dec["E5_drop"] = e5
    (out / "decision.json").write_text(json.dumps(dec, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(out, dec, legs, B, O, body, n_sp)
    print(f"完成（{time.time() - t0:.0f} 秒）；报告：{out / 'report.md'}", flush=True)


# ---------------------------------------------------------------- 报告
def write_report(out: Path, dec: dict, legs: dict, B: dict, O: Leg, body: float, n_sp: dict):
    f4 = lambda g: f"{g['mean']:+.4f} [{g['ci95'][0]:+.4f}, {g['ci95'][1]:+.4f}]"  # noqa: E731
    mm = lambda g: f"{1e3 * g.tmt:.1f}、{1e3 * g.tbt:.1f}、{1e3 * g.fem:.1f}"  # noqa: E731
    geo = dec["geometry"]
    L = ["# 工程验证 v2：地面任务上的方向检验", "", "预注册 `docs/prereg/engineering_ground_v1.md`。", "",
         "## 符号与术语", "",
         f"- **O（原腿）**：栖树雀形目（{n_sp['insessorial']} 种）的物种中位数；跗跖、胫跗、股 {mm(O)} mm，体重 "
         f"{1e3 * body:.1f} g。",
         f"- **T 方向**：观测陆栖腿 T（地栖雀形目 {n_sp['terrestrial']} 种的中位数，缩放到 O 的总长）相对 O 的比例改动方向。"
         f"**P 方向**：模型推测的变化量 Δ 的方向。两者在比例平面上相差 "
         f"{np.degrees(abs(geo['theta_T'] - geo['theta_P'])):.0f}°。",
         f"- **r_T**：T 离 O 的距离 {geo['r_T']:.4f}（"
         f"\"栖树到地栖\"的比例改动幅度）。**T_m、P_m**：沿 T 或 P 方向走 m·r_T（m = 1、2、4、8；m = 8 为工程尺度的放大）。"
         f"T_1 就是 T；模型的原始预测 P×1 的幅度只有 r_T 的 1/{geo['k_of_P_at_rT']:.2f}。",
         "- **比例平面、方向环、最陡方向**：总长固定时三段比例的二维平面；以 O 为圆心、半径 r_T 或 8·r_T 的圆上均匀取的 "
         "24 条腿，代表幅度相同、方向随机的改动；方向环上改善最大的方向（余弦拟合）。",
         "- **读数**：F 峰值地面反力（相对变化）、E 能量损失、V 前向速度保持；改善量越大越好。区间下限 > 0 为**上坡**，"
         "上限 < 0 为**下坡**，其余为**不确定**；幅度如实报告（F 的 0.01 即峰值力降低 1%）。**百分位**：某方向的改善在方向环 24 个方向中的位置。**效率**：某方向的改善占方向环上最大改善的比例。",
         "- **任务**：R 奔跑踩空、L 负重奔跑踩空（1.5、2 倍体重）、H 跳跃踩空（每条腿半体重）、S 软地奔跑踩空；"
         "每个任务 2 种速度 × 6 个下落高度（0 到 0.5 倍腿长）。",
         "- **E1 到 E5**：第二步的判定（预注册第 6 节）。", "",
         "## 第一步：筛选（只跑 O、T_1、T_8）", "", "| 任务 | 读数 | T_1 改善 | T_8 改善 | 结果 |", "|---|---|---|---|---|"]
    for t, s in dec["screen"].items():
        for m in METRICS:
            L.append(f"| {t} {TASKS[t]['name']} | {m} {MNAME[m]} | {f4(s[m]['T1'])} | {f4(s[m]['T8'])} | "
                     f"{'通过' if s[m]['pass'] else '未通过'} |")
    L += ["", f"通过：{'、'.join(f'{t} {m}' for t, m in dec['passed']) or '无'}。", ""]
    for t, r in dec.get("stage2", {}).items():
        L += [f"## 第二步：{t} {TASKS[t]['name']}", ""]
        for m in METRICS:
            it = r[m]
            r8 = it["rings"]["x8"]
            g = it["gains"]
            L += [f"**{m} {MNAME[m]}**：E1 {it['E1']}；E2 {it['E2']}（生物尺度 P_1：{it.get('P1_biological', '')}）；"
                  f"E3 {it['E3']}（ρ = {it['E3_rho']:.2f}）；E4 {it['E4']}。", "",
                  "| 方向 | m = 1 | m = 2 | m = 4 | m = 8 |", "|---|---|---|---|---|",
                  "| T | " + " | ".join(f4(g[f"T{x}"]) for x in MULT) + " |",
                  "| P | " + " | ".join(f4(g[f"P{x}"]) for x in MULT) + " |", "",
                  f"方向环 8·r_T：最陡方向 {r8['steepest_deg']:.0f}°（区间半宽 {r8['steepest_ci_halfwidth_deg']:.0f}°，"
                  f"R² {r8['r2']:.2f}），最大改善 {r8['max_gain']:+.4f}；P 百分位 {r8['P']['percentile']:.2f}、夹角 "
                  f"{r8['P']['angle_to_steepest_deg']:.0f}°、效率 {r8['P']['efficiency']:.2f}；T 百分位 "
                  f"{r8['T']['percentile']:.2f}、夹角 {r8['T']['angle_to_steepest_deg']:.0f}°、效率 {r8['T']['efficiency']:.2f}。", ""]
    if "E5_drop" in dec:
        e = dec["E5_drop"]["x8"]
        L += ["## E5：落震上的方向环（描述，读数为 v1 的可行率）", "",
              f"8·r_T：最陡方向 {e['steepest_deg']:.0f}°；P 百分位 {e['P']['percentile']:.2f}，T 百分位 {e['T']['percentile']:.2f}。", ""]
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
