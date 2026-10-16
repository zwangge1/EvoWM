"""奔跑一步引擎的检查（只用合成腿）：
1. 无阻尼（关节阻尼与地面阻尼为零、摩擦足够大）时，离地时机身的机械能损失应很小（腿的动能与弹簧余能另算）；
2. 步长缩到 1/4，主要读数不变；
3. 摆腿回收：触地时足端水平速度为零。
在 EvoWM 根目录运行：python scripts/exploratory/run_smoke.py
"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from evowm.engineering import drop as D  # noqa: E402
from evowm.engineering.drop import G, Leg  # noqa: E402
from evowm.engineering.run import RunParams, ground_scenario, step  # noqa: E402


def main():
    L, body = 0.064, 0.020
    leg = Leg(0.10, 0.211, 0.097).scaled(L)
    D.TERRAIN["elastic"] = dict(D.TERRAIN["turf"], zeta_c=0.0, mu=2.0)
    keys = ("fail", "peak_f", "compression", "t_stance", "vx_ratio", "vz_to", "e_loss", "slip")
    for name, ter, tau in (("无阻尼", "elastic", 0.0), ("有阻尼", "turf", 0.05)):
        sc = ground_scenario(body, leg, 1.2, 0.2, ter)
        unit = sc.m_unit * G * L
        k = 2.5 * unit
        c = tau * np.sqrt(L / G) * k
        p = RunParams(k, k, 0.0, c, c, 0.0, 110.0, 105.0, 62.0)
        r = step(leg, p, sc, trace=True)
        tr = r.pop("trace")
        vxf = (tr["xf"][2] - tr["xf"][0]) / (tr["t"][2] - tr["t"][0])
        print(name, {k_: (round(r[k_], 4) if isinstance(r[k_], float) else r[k_]) for k_ in keys},
              f"触地前足端水平速度 {vxf:.4f} m/s")
        r4 = step(leg, p, replace(sc, h=sc.h / 4))
        rel = {k_: round(r4[k_] / r[k_] - 1, 4) for k_ in keys[1:] if np.isfinite(r[k_]) and r[k_] != 0}
        print("  步长 1/4 的相对变化：", rel)


if __name__ == "__main__":
    main()
