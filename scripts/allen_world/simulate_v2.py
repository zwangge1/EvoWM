"""Known-gradient, equal-gradient, reversed-gradient and null calibration tables.

This generative model is independent of JEPA. These are diagnostic simulations,
not a reconstruction of biological evolution or an automatic approval gate.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(Path(__file__).resolve().parent)]
from evowm.data.tree import read_trees
from evowm.sim.bm import simulate_bm
from prepare_v2 import GROUPS, TARGETS, sha

SCENARIOS = ("null", "motion", "temperature", "joint", "equal", "reverse")
POSITIVE = np.array([0.012, 0.0, 0.006, 0.015])
EQUAL = np.array([0.012, 0.009, 0.009, 0.009])
REVERSE = np.array([0.012, 0.015, 0.006, 0.0])
NON_P = np.array([0.0036, 0.003, 0.003, 0.003])
MOTION_COEFF = np.array([0.04, 0.03, 0.03, 0.03])


def conditional_truth(table, scenario):
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown simulation scenario: {scenario}")
    p_exact = table.e_dev_P.fillna(0).to_numpy(float) == 1
    slope = np.zeros((len(table), len(TARGETS)))
    if scenario in ("temperature", "joint", "equal", "reverse"):
        profile = EQUAL if scenario == "equal" else REVERSE if scenario == "reverse" else POSITIVE
        slope[:] = NON_P
        slope[p_exact] = profile
    exposure_cols = ["e_fs_ground", "e_fs_wataroundsurf", "e_fs_watbelowsurf"]
    exposure = table[exposure_cols].sum(axis=1, min_count=3)
    if exposure.dropna().lt(0).any() or exposure.dropna().gt(1.000001).any():
        raise ValueError("Expected exposure proportions summing to [0,1]")
    factor = 0.5 + exposure.fillna(0.5).to_numpy(float) if scenario == "joint" else np.ones(len(table))
    slope *= factor[:, None]
    motion = MOTION_COEFF if scenario in ("motion", "joint") else np.zeros(4)
    return slope, motion, p_exact, factor


def generate(table_path, tree_path, output, scenario, seed):
    table_path, tree_path, output = Path(table_path).resolve(), Path(tree_path).resolve(), Path(output).resolve()
    if output / "species_traits.csv" == table_path:
        raise ValueError("Simulation cannot overwrite its source table")
    table = pd.read_csv(table_path)
    meta = json.loads(table_path.with_suffix(".columns.json").read_text(encoding="utf-8"))
    if meta.get("eco_groups") != GROUPS or meta.get("morph", []) + meta.get("skeleton", []) != TARGETS:
        raise ValueError("Expected prepared Allen v2 five-group input")
    if table.tip.isna().any() or table.tip.duplicated().any():
        raise ValueError("Input tips must be unique and present")
    if meta.get("table_sha256") != sha(table_path):
        raise ValueError("Source table hash mismatch")
    slope, motion_coeff, p_exact, exposure_factor = conditional_truth(table, scenario)
    tree = read_trees(tree_path, max_trees=1)[0]
    lookup = {tip: i for i, tip in enumerate(tree.tip_names)}
    missing_tips = set(table.tip) - set(lookup)
    if missing_tips:
        raise ValueError(f"{len(missing_tips)} input species are absent from simulation tree")
    rng = np.random.default_rng(seed)
    noise = simulate_bm(tree, n_traits=4, rate=0.000015, rng=rng)
    noise += rng.normal(0., 0.015, size=noise.shape)
    noise = noise[[lookup[t] for t in table.tip]]
    measured = np.isfinite(table[TARGETS].to_numpy(float))
    mass = table.m_mass.fillna(1.).to_numpy(float)
    temperature = table.bio1_c.fillna(0.).to_numpy(float) / 10.
    motion = table.e_life_terrestrial.fillna(0.).to_numpy(float)
    mean = np.array([0.8, 0.7, 0.9, 0.7])[None] + 0.3 * mass[:, None]
    mean = mean + temperature[:, None] * slope + motion[:, None] * motion_coeff
    simulated = table.copy()
    simulated[TARGETS] = np.where(measured, mean + noise, np.nan)
    output.mkdir(parents=True, exist_ok=False)
    dest = output / "species_traits.csv"
    simulated.to_csv(dest, index=False, float_format="%.12g")
    per_tip = table[["tip", "family", "order", "mode"]].copy()
    per_tip["P_exact"] = p_exact
    per_tip["exposure_factor"] = exposure_factor
    for j, col in enumerate(TARGETS):
        per_tip[f"true_slope_{col}"] = slope[:, j]
        per_tip[f"conditional_mean_{col}"] = mean[:, j]
        per_tip[f"observed_mask_{col}"] = measured[:, j]
    per_tip["slope_contrast_tmt_minus_tbt"] = slope[:, 3] - slope[:, 2]
    per_tip["slope_contrast_tbt_minus_fem"] = slope[:, 2] - slope[:, 1]
    per_tip["slope_contrast_tmt_minus_fem"] = slope[:, 3] - slope[:, 1]
    truth_path = output / "true_slopes.csv"
    per_tip.to_csv(truth_path, index=False, float_format="%.12g")
    truth = {
        "scenario": scenario, "seed": int(seed), "targets": TARGETS,
        "generator": "Known conditional mean plus independent tree BM residuals and tip noise; not a biological reconstruction",
        "equation": "intercept + 0.3*log10_mass + (bio1_C/10)*per_tip_thermal_slope + life_terrestrial*motion_coefficient + BM + tip_noise",
        "temperature_predictor": "bio1_c", "temperature_derivative": "log10 length per +10C, all other observed conditions fixed",
        "joint_shift_derivative": "Also equals parallel bio1/bio5/bio6 shift derivative because this generator depends on bio1 only",
        "P_definition": "e_dev_P == 1 exactly; SP is not P",
        "P_profiles_per_10C_before_exposure": {"positive": dict(zip(TARGETS, POSITIVE.tolist())),
                                              "equal": dict(zip(TARGETS, EQUAL.tolist())),
                                              "reverse": dict(zip(TARGETS, REVERSE.tolist()))},
        "non_P_profile_per_10C_before_exposure": dict(zip(TARGETS, NON_P.tolist())),
        "profile_application": "null/motion: thermal slope is zero; temperature/joint: positive P profile; equal: equal P bone profile; reverse: reversed P profile. Non-P skeletal slopes equal each other in every thermal scenario.",
        "exposure_modulation": "joint only: multiply species thermal slopes by 0.5 + X, where X=ground+water-surface+underwater; missing X=0.5 for generation only",
        "motion_predictor": "e_life_terrestrial", "motion_coefficients": dict(zip(TARGETS, motion_coeff.tolist())),
        "missing_covariate_generation_fallbacks": {"m_mass": 1.0, "bio1_c": 0.0, "e_life_terrestrial": 0.0, "missing_development": "non-P"},
        "BM_rate": 0.000015, "tip_noise_sd": 0.015, "preserves_target_observation_masks": True,
        "preserves_conditions_and_frozen_folds": True, "source_table_sha256": sha(table_path),
        "source_metadata_sha256": sha(table_path.with_suffix(".columns.json")), "tree_sha256": sha(tree_path),
        "generator_sha256": sha(__file__), "true_slopes_sha256": sha(truth_path),
        "scientific_pass": None,
    }
    meta = copy.deepcopy(meta)
    meta["simulation"] = truth
    meta["table_sha256"] = sha(dest)
    meta["simulation_generator_sha256"] = sha(__file__)
    dest.with_suffix(".columns.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (output / "truth.json").write_text(json.dumps(truth, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return dest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--table", type=Path, default=ROOT / "outputs/allen_world_v2/input/species_traits.csv")
    p.add_argument("--tree", type=Path, default=ROOT / "data/processed/trees/hackett_t021.nwk")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--scenario", required=True, choices=SCENARIOS)
    p.add_argument("--seed", type=int, default=13)
    args = p.parse_args()
    print(generate(args.table, args.tree, args.out_dir, args.scenario, args.seed))


if __name__ == "__main__":
    main()
