"""探索用的全身模块模拟表（设计 docs/design/探索_分层模块_2026-10-18.md）。

与 sim_recovery_tables.py 的 v5（主分析第 0 阶段）同一设定：真实习性 → 运动方式体制，Fitch 重建，OU（t½ 12 Myr，
平稳 SD 0.06），早期爆发布朗背景，数据在 Hackett 第 1 棵树上生成；腿的体制偏移与 v5 效应情景相同（随运动方式变）。
另加：飞行器官（翅长、Kipp 距离、HWI、肱骨、尺骨、桡骨、腕掌骨、龙骨）的最优值只随栖境（陆、水、空）变，
同一栖境内各运动方式相同。其余形态（喙 4 项、尾、次级飞羽、第二指、叉骨、头骨、巩膜环）不受习性影响，作阴性对照。
这些偏移是模拟的真值设定，不是对真实鸟类的断言。随机数的用法与 v5 的 build 相同。
输出 data/processed/sim/explore_v6/modular_s<种子>/（species_traits.csv、.columns.json、truth.json）。
在 EvoWM 根目录运行：python scripts/sim_modular_tables.py --seeds 15 16 --bg-rate 7.8e-4 --bg-eb -0.025
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import sim_recovery_tables as v5  # noqa: E402
from evowm.data.tree import read_trees  # noqa: E402
from evowm.data.units import REALM_OF  # noqa: E402
from evowm.sim.ou import simulate_bm_nodes, simulate_eb_nodes, simulate_ou_regimes  # noqa: E402
from evowm.sim.regimes import fitch, regime_locomotor  # noqa: E402

# 栖境对飞行器官潜变量最优值的偏移（相对"陆"；长度为 log10，HWI 为 /100 后的单位）
FLIGHT_OFFSETS = {
    "m_wing": {"land": 0.0, "water": -0.03, "air": 0.06},
    "m_kipps": {"land": 0.0, "water": -0.04, "air": 0.10},
    "m_hwi": {"land": 0.0, "water": -0.04, "air": 0.10},
    "k_humerus": {"land": 0.0, "water": -0.03, "air": 0.05},
    "k_ulna": {"land": 0.0, "water": -0.03, "air": 0.06},
    "k_radius": {"land": 0.0, "water": -0.03, "air": 0.06},
    "k_carpometacarpus": {"land": 0.0, "water": -0.03, "air": 0.06},
    "k_keel": {"land": 0.0, "water": -0.04, "air": 0.08},
}


def build(table: pd.DataFrame, meta: dict, tree, seed: int, bg_rate: float, bg_eb: float):
    cols = meta["morph"] + meta["skeleton"]
    others = [c for c in cols if c not in v5.LEG_SOURCE and c != "m_mass"]
    latents = list(v5.LEG_OFFSETS) + others
    lab = regime_locomotor(table)
    code = lab.map({r: i for i, r in enumerate(v5.REGIMES)}).fillna(-1).astype(int).to_numpy()
    pos = {n: i for i, n in enumerate(tree.tip_names)}
    order = np.array([pos[t] for t in table["tip"]])
    tip_states = np.full(tree.n_tips, -1)
    tip_states[order] = code
    node_regime, changes = fitch(tree, tip_states, len(v5.REGIMES), np.random.default_rng(v5.FITCH_SEED))

    theta = np.zeros((len(v5.REGIMES), len(latents)))
    for j, lat in enumerate(latents):
        if lat in v5.LEG_OFFSETS:
            theta[:, j] = [v5.LEG_OFFSETS[lat][r] for r in v5.REGIMES]
        elif lat in FLIGHT_OFFSETS:
            theta[:, j] = [FLIGHT_OFFSETS[lat][REALM_OF[r]] for r in v5.REGIMES]

    rng = np.random.default_rng(seed)
    ou = simulate_ou_regimes(tree, node_regime, theta, v5.ALPHA, np.full(len(latents), v5.SD_STAT), rng)
    mass = simulate_bm_nodes(tree, v5.MASS_RATE, v5.MASS_ROOT, rng)
    if bg_rate > 0:
        ou = ou + np.column_stack([simulate_eb_nodes(tree, bg_rate, bg_eb, rng) for _ in latents])
    tips = tree.tip_nodes[order]
    lat = {name: ou[tips, j] for j, name in enumerate(latents)}
    logm = mass[tips]
    n = len(table)
    sim = {}
    for c in cols:
        eps = rng.normal(size=n) * v5.SD_ME
        if c == "m_mass":
            sim[c] = logm + eps
        elif c == "m_hwi":
            sim[c] = 0.3 + lat[c] + eps
        elif c in v5.LEG_SOURCE:
            sim[c] = 1.0 + v5.SLOPE * logm + lat[v5.LEG_SOURCE[c]] + eps
        else:
            sim[c] = 1.0 + v5.SLOPE * logm + lat[c] + eps
    out = table.copy()
    real_nan = table[cols].isna()
    for c in cols:
        out[c] = np.where(real_nan[c].to_numpy(), np.nan, sim[c])
    out["sim_regime"] = lab.to_numpy()
    truth = {
        "scenario": "modular", "seed": seed, "tree": "hackett_tree1", "regime_variable": "locomotor",
        "regimes": v5.REGIMES, "realm_of": REALM_OF, "fitch_changes": int(changes),
        "t_half_myr": v5.T_HALF, "alpha": v5.ALPHA, "sd_stat": v5.SD_STAT, "sd_measurement": v5.SD_ME,
        "mass_bm_rate": v5.MASS_RATE, "allometric_slope": v5.SLOPE, "background_rate0": bg_rate, "background_eb": bg_eb,
        "leg_offsets": v5.LEG_OFFSETS, "leg_source": v5.LEG_SOURCE, "flight_offsets": FLIGHT_OFFSETS,
        "null_traits": [c for c in others if c not in FLIGHT_OFFSETS],
    }
    return out, truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[15, 16])
    ap.add_argument("--bg-rate", type=float, default=7.8e-4)
    ap.add_argument("--bg-eb", type=float, default=-0.025)
    args = ap.parse_args()
    src = ROOT / "data/processed/species_traits.csv"
    meta = json.loads(src.with_suffix(".columns.json").read_text(encoding="utf-8"))
    table = pd.read_csv(src)
    tree = read_trees(ROOT / "data/raw/birdtree/hackett_tree1.nwk", max_trees=1)[0]
    for seed in args.seeds:
        out, truth = build(table, meta, tree, seed, args.bg_rate, args.bg_eb)
        dst = ROOT / f"data/processed/sim/explore_v6/modular_s{seed}"
        dst.mkdir(parents=True, exist_ok=True)
        out.to_csv(dst / "species_traits.csv", index=False, float_format="%.6g")
        shutil.copy(src.with_suffix(".columns.json"), dst / "species_traits.columns.json")
        (dst / "truth.json").write_text(json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"写入 {dst}（Fitch 变化 {truth['fitch_changes']} 次）")


if __name__ == "__main__":
    main()
