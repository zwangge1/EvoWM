"""奔跑一步的灵敏度（工程验证 v2 开发用，只用合成腿，不是计划中的比较腿）。

1. 三条合成腿（三段等长、天鹅比例、跗跖骨加长，总长 6.4 cm，体重 20 g）在各任务下的通过率与失败类型；
2. 方向环：以天鹅比例腿为圆心、半径 0.0177（与计划中 T 的比例改动同大小）的 12 个方向，看比例改动这么大时
   通过率能差多少；
3. 只改大小：天鹅比例腿整体加长 3%，任务的绝对速度与下落高度不变。
在 EvoWM 根目录运行：python scripts/exploratory/run_sensitivity.py [--n-param 64] [--workers 2]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from evowm.engineering.drop import Leg  # noqa: E402
from evowm.engineering.run import RunCriteria, evaluate_run, ground_scenario, run_box  # noqa: E402
from evowm.engineering.shape import ring, shape_xy  # noqa: E402

TASKS = {  # 任务 → (载荷, 地面)
    "R": (1.0, "turf"), "L1.5": (1.5, "turf"), "L2": (2.0, "turf"), "H": (0.5, "turf"), "S": (1.0, "wetsand")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-param", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tasks", default="R")
    ap.add_argument("--speeds", default="0.8,1.2")
    ap.add_argument("--dh", default="0,0.2,0.4")
    ap.add_argument("--ring", type=int, default=12)
    ap.add_argument("--out", default=str(ROOT / "outputs/exploratory/run_sensitivity.pkl"))
    args = ap.parse_args()
    L, body = 0.064, 0.020
    legs = {"equal": Leg(L / 3, L / 3, L / 3), "swan_like": Leg(0.10, 0.211, 0.097).scaled(L),
            "long_tmt": Leg(0.16, 0.11, 0.06).scaled(L)}
    c = legs["swan_like"]
    if args.ring:
        legs.update(ring(c, 0.0177, args.ring))
    legs["swan_x1.03"] = Leg(c.tmt * 1.03, c.tbt * 1.03, c.fem * 1.03)
    speeds = [float(v) for v in args.speeds.split(",")]
    dhs = [float(v) for v in args.dh.split(",")]
    scen, tags = [], []
    for task in args.tasks.split(","):
        load, ter = TASKS[task]
        for s in speeds:
            for dh in dhs:
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
        row["ok"] = cr.ok(r, legs[r["leg"]])
        rows.append(row)
    d = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    d.to_pickle(args.out)
    pd.set_option("display.width", 200)
    base = ["equal", "swan_like", "long_tmt", "swan_x1.03"]
    print(d[d.leg.isin(base)].groupby(["task", "speed", "dh", "leg"]).ok.mean().unstack("leg").round(3))
    print(d[d.leg.isin(base)].groupby(["leg"]).fail.value_counts().unstack(fill_value=0))
    ok = d[(d.fail == "") & d.leg.isin(base)]
    print(ok.groupby(["task", "speed", "dh"])[["peak_f", "compression", "t_stance", "vx_ratio", "e_loss"]].median().round(3))
    pc = d[d.leg == "swan_like"].groupby("task").ok.mean()
    for task in d.task.unique():
        dd = d[d.task == task]
        p = dd.groupby("leg").ok.mean()
        rg = p[[k for k in p.index if k.startswith("ring")]] - p["swan_like"]
        ang = [int(k.split("_")[-1]) for k in rg.index]
        print(f"[{task}] 圆心通过率 {pc[task]:.3f}；方向环与圆心之差（百分点）：" +
              "，".join(f"{a}°:{100 * v:+.1f}" for a, v in zip(ang, rg.values)))
        print(f"[{task}] 整体加长 3% 与圆心之差：{100 * (p['swan_x1.03'] - p['swan_like']):+.1f} 个百分点")
    print("比例坐标（相对天鹅比例腿）：", {k: shape_xy(v, c).round(3).tolist() for k, v in legs.items() if not k.startswith("ring")})


if __name__ == "__main__":
    main()
