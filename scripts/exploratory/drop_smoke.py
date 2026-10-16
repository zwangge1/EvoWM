"""落震引擎的冒烟与物理检查（真实鸟类尺度）：自由落体、能量、步长收敛、地面与速度。
只用合成腿（不是工程验证计划中的比较腿）。在 EvoWM 根目录运行：python scripts/exploratory/drop_smoke.py
"""
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from evowm.engineering.drop import G, Leg, Params, bird_scenario, drop, froude_v0  # noqa: E402

L = 0.064                                   # 合成腿总长（m），雀形目量级
legs = {"equal": Leg(L / 3, L / 3, L / 3), "swan_like": Leg(0.10, 0.211, 0.097).scaled(L)}
v_norm = froude_v0(1.2, L)                  # FFY 的 1.2 m/s 按 Froude 数换到这条腿
sc = bird_scenario(0.020, legs["equal"], v_norm, "turf")
u = sc.m * G * sc.l_ref
tau = 0.05 * np.sqrt(sc.l_ref / G)
P = Params(2.4 * u, 2.4 * u, 10.9 * u, tau * 2.4 * u, tau * 2.4 * u, tau * 10.9 * u, 144.0, 133.0, 41.0)
print(f"工况：载荷 {sc.m * 1e3:.1f} g，v0 {sc.v0:.3f} m/s，kc {sc.kc:.0f} N/m，足端 {sc.r_foot * 1e3:.2f} mm，"
      f"T {sc.T} s，h {sc.h:.2e} s")
for name, leg in legs.items():
    t0 = time.time()
    r = drop(leg, P, sc)
    print(name, f"{time.time() - t0:.2f}s", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})
# 1) 自由落体：触地前机身加速度 ≈ -g
r = drop(legs["equal"], P, sc, trace=True)
tr = r["trace"]
i_c = int(np.flatnonzero(tr["z_foot"] <= sc.r_foot)[0])
print("触地前加速度均值", round(float(np.mean(tr["az"][2:max(3, i_c - 2)])), 3), "应为", -G)
# 2) 能量：压缩段吸能 vs 机身可用能量（动能 + 下沉段势能，不含杆件）
avail = 0.5 * sc.m * sc.v0 ** 2 + sc.m * G * r["stroke"]
print("吸能 / 可用能量", round(r["E_abs"] / avail, 4), "；峰值地面反力", round(r["peak_grf_bw"], 2), "倍体重")
# 3) 步长收敛
for f in (2.0, 1.0, 0.5, 0.25):
    r = drop(legs["equal"], P, replace(sc, h=sc.h * f))
    print(f"h × {f}", "峰值", round(r["peak_a"] / G, 3), "g；腿行程", round(1e3 * r["leg_stroke"], 3), "mm；η", round(r["eta"], 4), r["fail"])
# 4) 地面与速度
for ter in ("concrete", "turf", "wetsand"):
    for v0 in (v_norm, froude_v0(2.0, L)):
        r = drop(legs["equal"], P, replace(sc, terrain=ter, v0=v0))
        print(f"{ter:9s} v0 {v0:.2f}", "峰值", round(r["peak_a"] / G, 2), "g；腿行程", round(1e3 * r["leg_stroke"], 2),
              "mm；下陷", round(1e3 * r["sink"], 3), "mm；打滑", round(1e3 * r["slip"], 3), "mm", r["fail"])
