"""探索（不进判据）：同样的"去散热"缩短量放在哪一段，落震可行率变不变。

热阀反事实 v1 的主结果是免费阀（去散热的腿与真实腿可行率无差）。这里问：这个"免费"是跗跖骨特有的，还是哪一段都免费？
露天组真实腿 R 上，把 20 °C 的跗跖骨缩短量（log10 −0.0322，−7.2%）分别只放在跗跖骨（Ct20，主分析已跑）、只放在胫跗骨（Cb20）、
只放在股骨（Cf20），同样的 256 组参数与 6 个工况。在 EvoWM 根目录运行：python scripts/exploratory/drop_segment_swap_v0.py
输出 outputs/thermal_v1/segment_swap.json。
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/thermal"))
from drop_thermal_v1 import MAIN, SEED, TERRAINS, V_FFY, leg_of, pf_stats  # noqa: E402
from evowm.engineering.drop import Leg, bird_scenario, evaluate, froude_v0, param_box  # noqa: E402

DLOG = -0.0161 * 2  # 20 °C 的跗跖骨缩短量（log10）


def main():
    lj = json.loads((ROOT / "outputs/thermal_v1/legs.json").read_text(encoding="utf-8"))
    info = lj["groups"]["exposed"]
    R = leg_of(info["legs_mm"]["R"])
    f = 10 ** DLOG
    legs = {"R": R, "Ct20": Leg(R.tmt * f, R.tbt, R.fem), "Cb20": Leg(R.tmt, R.tbt * f, R.fem), "Cf20": Leg(R.tmt, R.tbt, R.fem * f)}
    body = info["body_mass_g"] / 1000.0
    scen = [bird_scenario(body, R, froude_v0(v, R.total), ter) for v in V_FFY for ter in TERRAINS]
    params = param_box(256, SEED, scen[0])
    res = evaluate(legs, params, scen, workers=2)
    idx = {(r["leg"], r["scenario"], r["j"]): r for r in res}
    nS, nP = len(scen), len(params)
    cube = {n: np.array([[MAIN.ok(idx[(n, s, j)], legs[n], scen[s]) for j in range(nP)] for s in range(nS)]) for n in legs}
    K = np.random.default_rng(0).integers(0, nP, size=(10000, nP))
    pf, d, per_s = pf_stats(cube, list(legs), K)
    out = {"legs_mm": {n: [1e3 * g.tmt, 1e3 * g.tbt, 1e3 * g.fem] for n, g in legs.items()}, "pF": pf, "diff": d, "per_scenario": per_s}
    (ROOT / "outputs/thermal_v1/segment_swap.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for n in legs:
        print(n, [round(x, 1) for x in out["legs_mm"][n]], round(100 * pf[n], 1), "" if n == "R" else f"{100 * d[n]['diff']:+.1f} [{100 * d[n]['ci95'][0]:+.1f}, {100 * d[n]['ci95'][1]:+.1f}]")


if __name__ == "__main__":
    main()
