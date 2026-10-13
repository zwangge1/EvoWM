"""冒烟测试用的模拟物种表：形态列全部换成树上布朗运动模拟值，不含任何真实形态数值。

- 习性列、科目、缺失位置沿用真实物种表（缺失位置保持一致，用来测试掩码）。
- 腿部三段骨和跗跖额外加上已知的习性效应（地面觅食比例），作为可被找回的信号。
输出 data/processed/sim/smoke/species_traits.csv、.columns.json、truth.json。
在 EvoWM 根目录运行：python scripts/sim_smoke_table.py
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
from evowm.sim.bm import simulate_bm  # noqa: E402

SEED = 20261013
RATE = 0.002          # log10 单位方差/百万年
EFFECT = 0.15         # 地面觅食比例从 0 到 1 时，腿部列（log10）增加的量
LEG_COLS = ["m_tarsus", "k_femur", "k_tibiotarsus", "k_tarsometatarsus"]


def main():
    src = ROOT / "data/processed/species_traits.csv"
    meta = json.loads(src.with_suffix(".columns.json").read_text(encoding="utf-8"))
    table = pd.read_csv(src)
    tree = read_trees(ROOT / "data/raw/birdtree/hackett_tree1.nwk", max_trees=1)[0]
    rng = np.random.default_rng(SEED)

    cols = meta["morph"] + meta["skeleton"]
    sim = simulate_bm(tree, len(cols), RATE, rng)
    pos = {t: i for i, t in enumerate(tree.tip_names)}
    sim = sim[[pos[t] for t in table["tip"]]]

    out = table.copy()
    real_nan = table[cols].isna().to_numpy()
    out[cols] = np.where(real_nan, np.nan, sim)
    ground = table["e_fs_ground"].fillna(0).to_numpy()
    for c in LEG_COLS:
        out[c] = out[c] + EFFECT * ground

    dst = ROOT / "data/processed/sim/smoke"
    dst.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst / "species_traits.csv", index=False, float_format="%.6g")
    shutil.copy(src.with_suffix(".columns.json"), dst / "species_traits.columns.json")
    truth = {"seed": SEED, "process": "BM on hackett_tree1", "rate": RATE,
             "effect": {c: {"e_fs_ground": EFFECT} for c in LEG_COLS}}
    (dst / "truth.json").write_text(json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写入 {dst}")


if __name__ == "__main__":
    main()
