"""落震引擎的灵敏度（真实鸟类尺度）：三条合成腿在同一批 FFY v2.5 参数盒参数下的配对比较，判据取几档。
只用合成腿（不是工程验证计划中的比较腿）。在 EvoWM 根目录运行：python scripts/exploratory/drop_sensitivity.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from evowm.engineering.drop import G, Criteria, Leg, bird_scenario, evaluate, froude_v0, param_box  # noqa: E402


def main():
    L = 0.064
    legs = {"equal": Leg(L / 3, L / 3, L / 3), "swan_like": Leg(0.10, 0.211, 0.097).scaled(L),
            "long_tmt": Leg(0.16, 0.11, 0.06).scaled(L)}
    sc = bird_scenario(0.020, legs["equal"], froude_v0(1.2, L), "turf")
    P = param_box(128, seed=0, sc=sc)
    t0 = time.time()
    res = evaluate(legs, P, [sc], workers=2)
    print("runs", len(res), f"{time.time() - t0:.1f}s")
    d = pd.DataFrame(res)
    d["peak_g"] = d.a_res / G
    d["stroke_frac"] = d.leg_stroke / L
    print("fail", d.groupby("leg")["fail"].apply(lambda s: (s != "").mean()).round(3).to_dict())
    print(d.groupby("leg")[["peak_g", "stroke_frac", "eta", "peak_grf_bw", "slip"]].median().round(4))
    rows = []
    for g in (3.0, 5.0, 10.0):
        for f in (0.10, 0.15, 0.20):
            cr = Criteria(g_cap=g, s_max_frac=f)
            rows.append(dict(g_cap=g, s_max_frac=f, **{n: np.mean([cr.ok(r, legs[n], sc) for r in res if r["leg"] == n])
                                                      for n in legs}))
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    p = d.pivot(index="j", columns="leg", values="eta")
    for n in ("swan_like", "long_tmt"):
        x = (p[n] - p["equal"]).dropna()
        print("η 配对差", n, round(x.median(), 4), np.percentile(x, [25, 75]).round(4))


if __name__ == "__main__":
    main()
