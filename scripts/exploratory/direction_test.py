"""方向检验（方案 1）的测试：只用合成腿，不是计划中的比较腿。

圆心为天鹅比例的合成腿（总长 6.4 cm，体重 20 g）。两个测试方向取比例平面上 −19.4° 与 24.0°（与 v1 中推测方向、
观测方向的角度相同，夹角 43°；圆心不同，所以只是两个普通方向）。半径取 0.0177（与"栖树到地栖"的比例改动同大小）
及其 2、4、8 倍；方向环取 12 个方向，放在 0.0177 与 8 倍半径上。任务：奔跑踩空（R）与跳跃踩空（H），
各 2 种速度 × 3 个下落高度。读数：峰值地面反力（F）、能量损失（E）、速度保持（V）、通过率。
看四件事：方向环上改善量是否呈清楚的余弦形（拟合 R²）；最陡方向能估得多准；单个方向的改善在生物尺度上
能否与零区分；沿一个方向放大时改善是否单调累积，8 倍时能否达到 5%。
在 EvoWM 根目录运行：python scripts/exploratory/direction_test.py [--n-param 48] [--workers 2]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from evowm.engineering.direction import METRICS, boot_means, gain_table, ring_summary  # noqa: E402
from evowm.engineering.drop import Leg  # noqa: E402
from evowm.engineering.run import RunCriteria, evaluate_run, ground_scenario, run_box  # noqa: E402
from evowm.engineering.shape import leg_at, ring  # noqa: E402

TASKS = {"R": (1.0, "turf"), "H": (0.5, "turf"), "L": (1.5, "turf"), "S": (1.0, "wetsand")}
DIRS = {"甲": np.radians(-19.4), "乙": np.radians(24.0)}
R1 = 0.0177
MULT = (1, 2, 4, 8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-param", type=int, default=48)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tasks", default="R,H")
    ap.add_argument("--n-dir", type=int, default=12)
    ap.add_argument("--out", default=str(ROOT / "outputs/exploratory/direction_test"))
    args = ap.parse_args()
    L, body = 0.064, 0.020
    c = Leg(0.10, 0.211, 0.097).scaled(L)
    legs = {"C": c}
    rings = {m: ring(c, m * R1, args.n_dir) for m in (1, 8)}
    for r in rings.values():
        legs.update(r)
    for name, th in DIRS.items():
        for m in MULT:
            legs[f"{name}x{m}"] = leg_at(c, m * R1 * np.array([np.cos(th), np.sin(th)]))
    scen, tags = [], []
    for task in args.tasks.split(","):
        load, ter = TASKS[task]
        for s in (0.8, 1.2):
            for dh in (0.0, 0.2, 0.4):
                scen.append(ground_scenario(body, c, s, dh, ter, load))
                tags.append(dict(task=task, speed=s, dh=dh))
    P = run_box(args.n_param, 0, scen[0])
    t0 = time.time()
    res = evaluate_run(legs, P, scen, workers=args.workers)
    print(len(res), "次", f"{time.time() - t0:.0f} 秒", flush=True)
    cr = RunCriteria()
    rows = []
    for r in res:
        row = {k: v for k, v in r.items() if not k.startswith("sc_")}
        row.update(tags[r["scenario"]])
        row["ok"] = bool(cr.ok(r, legs[r["leg"]]))
        rows.append(row)
    d = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_pickle(str(out) + ".pkl")
    summary = {}
    for task in d.task.unique():
        dt = d[d.task == task].copy()
        summary[task] = {"pass_rate_center": float(dt[dt.leg == "C"].ok.mean())}
        for metric in METRICS:
            G = gain_table(dt, legs, "C", metric)
            B = boot_means(G)
            res_m = {}
            for m, rg in rings.items():
                names = list(rg)
                ang = np.radians([int(n.rsplit("_", 1)[1]) for n in names])
                probes = {k: (f"{k}x{m}", th) for k, th in DIRS.items()}
                rs = ring_summary(B, names, ang, probes)
                res_m[f"ring_x{m}"] = {k: v for k, v in rs.items() if k != "gains"} | {"gains": rs["gains"]}
            res_m["accumulation"] = {k: {m: {"gain": B[f"{k}x{m}"]["mean"], "ci95": B[f"{k}x{m}"]["ci95"]}
                                         for m in MULT} for k in DIRS}
            summary[task][metric] = res_m
            r1, r8 = res_m["ring_x1"], res_m["ring_x8"]
            print(f"[{task} {metric}] 环 ×1：最陡 {r1['steepest_deg']:.0f}°（±{r1['steepest_ci_halfwidth_deg']:.0f}°），"
                  f"幅度 {r1['amplitude']:.4f}，R² {r1['r2']:.2f}；环 ×8：最陡 {r8['steepest_deg']:.0f}°"
                  f"（±{r8['steepest_ci_halfwidth_deg']:.0f}°），幅度 {r8['amplitude']:.4f}，R² {r8['r2']:.2f}，"
                  f"最大改善 {r8['max_gain']:+.4f}", flush=True)
            for k in DIRS:
                acc = res_m["accumulation"][k]
                print(f"    方向{k}：" + "；".join(f"×{m} {acc[m]['gain']:+.4f} [{acc[m]['ci95'][0]:+.4f}, {acc[m]['ci95'][1]:+.4f}]"
                                                 for m in MULT)
                      + f"；×1 百分位 {r1[k]['percentile']:.2f}、夹角 {r1[k]['angle_to_steepest_deg']:.0f}°、效率 {r1[k]['efficiency']:.2f}"
                      + f" [{r1[k]['efficiency_ci95'][0]:.2f}, {r1[k]['efficiency_ci95'][1]:.2f}]", flush=True)
        pr = dt.groupby("leg").ok.mean()
        print(f"[{task}] 通过率：圆心 {pr['C']:.3f}；环 ×8 与圆心之差（百分点）最小 {100 * (pr[list(rings[8])] - pr['C']).min():+.1f}，"
              f"最大 {100 * (pr[list(rings[8])] - pr['C']).max():+.1f}", flush=True)
    Path(str(out) + ".json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
