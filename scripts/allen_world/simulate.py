"""Generate known conditional effects + phylogenetic noise, retaining real missingness.

This calibration generator is not a claim about the biological evolutionary process.
It is independent of JEPA and permits null, temperature-only, motion-only and joint cases.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from evowm.data.tree import read_trees
from evowm.sim.bm import simulate_bm


def generate(table_path, tree_path, output, scenario, seed):
    table_path, tree_path, output = Path(table_path), Path(tree_path), Path(output)
    table = pd.read_csv(table_path)
    meta = json.loads(table_path.with_suffix(".columns.json").read_text(encoding="utf-8"))
    cols = meta["morph"] + meta["skeleton"]
    if cols != ["m_tarsus", "b_fem", "b_tbt", "b_tmt"]:
        raise ValueError("Unexpected target schema")
    tree = read_trees(tree_path, max_trees=1)[0]
    rng = np.random.default_rng(seed)
    noise = simulate_bm(tree, 4, rate=0.000015, rng=rng)
    noise += rng.normal(0, 0.015, noise.shape)
    row = {t: i for i, t in enumerate(tree.tip_names)}
    noise = noise[[row[t] for t in table.tip]]
    measured = np.isfinite(table[cols].to_numpy(float))
    mass = table.m_mass.fillna(1).to_numpy(float)
    early = (table.e_dev_P.fillna(0) + table.e_dev_SP.fillna(0)).to_numpy(float)
    temp = table.bio1_c.fillna(0).to_numpy(float) / 10
    motion = table.e_fs_ground.fillna(0).to_numpy(float)
    temp_b = np.array([0.012, 0.0, 0.006, 0.015]) if scenario in ("temperature", "joint") else np.zeros(4)
    motion_b = np.array([0.04, 0.03, 0.03, 0.03]) if scenario in ("motion", "joint") else np.zeros(4)
    y = (np.array([0.8, 0.7, 0.9, 0.7])[None, :] + mass[:, None] * 0.3 +
         (temp * (0.3 + 0.7 * early))[:, None] * temp_b[None, :] + motion[:, None] * motion_b + noise)
    table[cols] = np.where(measured, y, np.nan)
    output.mkdir(parents=True, exist_ok=False)
    dest = output / "species_traits.csv"
    table.to_csv(dest, index=False)
    meta["simulation"] = {"scenario": scenario, "seed": seed, "generator": "conditional mean + tree BM residual",
                          "temperature_coefficients_per_10C_early": dict(zip(cols, temp_b.tolist())),
                          "temperature_late_multiplier": 0.3, "motion_ground_coefficients": dict(zip(cols, motion_b.tolist())),
                          "mass_slope": 0.3, "preserves_observation_masks": True,
                          "input_sha256": hashlib.sha256(table_path.read_bytes()).hexdigest(),
                          "tree_sha256": hashlib.sha256(tree_path.read_bytes()).hexdigest()}
    meta["table_sha256"] = hashlib.sha256(dest.read_bytes()).hexdigest()
    meta["simulation_generator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    dest.with_suffix(".columns.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "truth.json").write_text(json.dumps(meta["simulation"], indent=2), encoding="utf-8")
    return dest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--table", type=Path, default=ROOT / "outputs/allen_world_v1/input/species_traits.csv")
    p.add_argument("--tree", type=Path, default=ROOT / "data/processed/trees/hackett_t021.nwk")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--scenario", choices=["null", "temperature", "motion", "joint"], required=True)
    p.add_argument("--seed", type=int, default=13)
    a = p.parse_args()
    print(generate(a.table, a.tree, a.out_dir, a.scenario, a.seed))


if __name__ == "__main__":
    main()
