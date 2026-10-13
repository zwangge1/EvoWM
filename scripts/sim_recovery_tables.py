"""恢复检验的模拟表：效应 / 零模型 × 若干种子。设定见 docs/prereg/sim_recovery_v1.md（v2 沿用同一模拟设定）。

所有形态列都换成模拟值，不含任何真实形态数值；习性列、科目、缺失位置沿用真实物种表。
输出 data/processed/sim/recovery_<版本>/<情景>_s<种子>/ 下的 species_traits.csv、.columns.json、truth.json。
在 EvoWM 根目录运行：python scripts/sim_recovery_tables.py [--version v2 --seeds 3 4]
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evowm.data.tree import read_trees  # noqa: E402
from evowm.sim.ou import simulate_bm_nodes, simulate_ou_regimes  # noqa: E402
from evowm.sim.regimes import fitch, regime_locomotor  # noqa: E402

REGIMES = ["insessorial", "terrestrial", "generalist", "aerial", "wading", "diving", "surface"]
T_HALF = 12.0                      # Myr
ALPHA = np.log(2) / T_HALF
SD_STAT = 0.06                     # OU 残差平稳 SD（log10）
SD_ME = 0.01                       # 测量误差 SD
MASS_RATE, MASS_ROOT = 0.0033, 1.5
SLOPE = 0.33                       # 长度对体重的异速斜率（log10）
FITCH_SEED = 0

# 效应情景：腿部潜变量的体制最优值偏移（相对"栖树"，log10）。这是模拟的真值设定，不是对真实鸟类的断言。
LEG_OFFSETS = {
    "L_tmt": {"insessorial": 0.00, "terrestrial": 0.08, "generalist": 0.03, "aerial": -0.15,
              "wading": 0.20, "diving": -0.08, "surface": -0.10},
    "L_tib": {"insessorial": 0.00, "terrestrial": 0.05, "generalist": 0.02, "aerial": -0.08,
              "wading": 0.15, "diving": 0.05, "surface": -0.04},
    "L_fem": {"insessorial": 0.00, "terrestrial": 0.01, "generalist": 0.00, "aerial": -0.03,
              "wading": 0.02, "diving": -0.04, "surface": -0.03},
}
LEG_SOURCE = {"m_tarsus": "L_tmt", "k_tarsometatarsus": "L_tmt", "k_tibiotarsus": "L_tib", "k_femur": "L_fem"}


def build(table: pd.DataFrame, meta: dict, tree, scenario: str, seed: int):
    cols = meta["morph"] + meta["skeleton"]
    others = [c for c in cols if c not in LEG_SOURCE and c != "m_mass"]
    latents = list(LEG_OFFSETS) + others

    # 体制：真实习性 → 运动方式；祖先体制用 Fitch 重建（固定种子，两个情景共用同一体制历史）
    lab = regime_locomotor(table)
    code = lab.map({r: i for i, r in enumerate(REGIMES)}).fillna(-1).astype(int).to_numpy()
    pos = {n: i for i, n in enumerate(tree.tip_names)}
    order = np.array([pos[t] for t in table["tip"]])
    tip_states = np.full(tree.n_tips, -1)
    tip_states[order] = code
    node_regime, changes = fitch(tree, tip_states, len(REGIMES), np.random.default_rng(FITCH_SEED))

    theta = np.zeros((len(REGIMES), len(latents)))
    if scenario == "effect":
        for j, lat in enumerate(latents):
            if lat in LEG_OFFSETS:
                theta[:, j] = [LEG_OFFSETS[lat][r] for r in REGIMES]

    rng = np.random.default_rng(seed)
    ou = simulate_ou_regimes(tree, node_regime, theta, ALPHA, np.full(len(latents), SD_STAT), rng)
    mass = simulate_bm_nodes(tree, MASS_RATE, MASS_ROOT, rng)
    tips = tree.tip_nodes[order]                      # 按物种表行序
    lat = {name: ou[tips, j] for j, name in enumerate(latents)}
    logm = mass[tips]

    n = len(table)
    sim = {}
    for c in cols:
        eps = rng.normal(size=n) * SD_ME
        if c == "m_mass":
            sim[c] = logm + eps
        elif c == "m_hwi":
            sim[c] = 0.3 + lat[c] + eps
        elif c in LEG_SOURCE:
            sim[c] = 1.0 + SLOPE * logm + lat[LEG_SOURCE[c]] + eps
        else:
            sim[c] = 1.0 + SLOPE * logm + lat[c] + eps
    out = table.copy()
    real_nan = table[cols].isna()
    for c in cols:
        out[c] = np.where(real_nan[c].to_numpy(), np.nan, sim[c])
    out["sim_regime"] = lab.to_numpy()

    truth = {
        "scenario": scenario, "seed": seed, "tree": "hackett_tree1", "regime_variable": "locomotor",
        "regimes": REGIMES, "regime_counts": lab.value_counts().to_dict(), "fitch_changes": int(changes),
        "t_half_myr": T_HALF, "alpha": ALPHA, "sd_stat": SD_STAT, "sd_measurement": SD_ME,
        "mass_bm_rate": MASS_RATE, "allometric_slope": SLOPE,
        "leg_offsets": LEG_OFFSETS if scenario == "effect" else {k: {r: 0.0 for r in REGIMES} for k in LEG_OFFSETS},
        "leg_source": LEG_SOURCE,
    }
    return out, truth


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1", help="输出到 data/processed/sim/recovery_<version>/")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    args = ap.parse_args()
    src = ROOT / "data/processed/species_traits.csv"
    meta = json.loads(src.with_suffix(".columns.json").read_text(encoding="utf-8"))
    table = pd.read_csv(src)
    tree = read_trees(ROOT / "data/raw/birdtree/hackett_tree1.nwk", max_trees=1)[0]
    for scenario in ("effect", "null"):
        for seed in args.seeds:
            out, truth = build(table, meta, tree, scenario, seed)
            dst = ROOT / f"data/processed/sim/recovery_{args.version}/{scenario}_s{seed}"
            dst.mkdir(parents=True, exist_ok=True)
            out.to_csv(dst / "species_traits.csv", index=False, float_format="%.6g")
            shutil.copy(src.with_suffix(".columns.json"), dst / "species_traits.columns.json")
            (dst / "truth.json").write_text(json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"写入 {dst}（Fitch 变化 {truth['fitch_changes']} 次）")


if __name__ == "__main__":
    main()
