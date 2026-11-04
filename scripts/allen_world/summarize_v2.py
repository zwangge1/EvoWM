"""Retain the full experiment matrix and audit known-simulation recovery.

The calibration flag is a fixed development heuristic, not a significance test,
formal error-rate calibration, or automatic permission to make paper claims.
Partial runs are valid summaries and return success while recording incompleteness.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BONES = ("b_fem", "b_tbt", "b_tmt")
SCENARIOS = {"null", "motion", "temperature", "joint", "equal", "reverse"}
FIELDS = ("name", "stage", "scenario", "variant", "seed", "fold")


def boolean(series, name):
    result = series.astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
    if result.isna().any():
        raise ValueError(f"Missing/invalid boolean flag {name}")
    return result.to_numpy(bool)


def paired_rows(windows, truth, fold):
    required = ["target_tip", "family", "development_mode", "order", "temperature_axis",
                "within_training_temperature_marginals", "complete_three_bones"]
    required += [f"response_per_10C_{b}" for b in BONES] + [f"measured_{b}" for b in BONES]
    absent = set(required) - set(windows)
    needed_truth = ["tip", "family", "mode", "order"] + [f"true_slope_{b}" for b in BONES] + [f"observed_mask_{b}" for b in BONES]
    if absent or set(needed_truth) - set(truth):
        raise ValueError(f"Missing required response/truth fields: {sorted(absent | (set(needed_truth) - set(truth)))}")
    if windows.temperature_axis.isna().any() or not windows.temperature_axis.eq("joint").all():
        raise ValueError("Primary calibration requires joint-temperature response windows")
    if truth.tip.isna().any() or truth.tip.duplicated().any():
        raise ValueError("Truth tips must be present and unique")
    data = windows[required].merge(truth[needed_truth], left_on="target_tip", right_on="tip", how="left",
                                  validate="many_to_one", suffixes=("", "_truth"), indicator=True)
    if not data._merge.eq("both").all():
        raise ValueError("A prediction target is absent from per-species truth")
    if not data.family.eq(data.family_truth).all() or not data.order.eq(data.order_truth).all():
        raise ValueError("Prediction and truth taxonomy disagree")
    if not data.development_mode.eq(data["mode"]).all():
        raise ValueError("Prediction and truth developmental modes disagree")
    complete = boolean(data.complete_three_bones, "complete_three_bones")
    measured = np.column_stack([boolean(data[f"measured_{b}"], f"measured_{b}") for b in BONES]).all(axis=1)
    truth_measured = np.column_stack([boolean(data[f"observed_mask_{b}"], f"observed_mask_{b}") for b in BONES]).all(axis=1)
    if not np.array_equal(complete, measured) or not np.array_equal(measured, truth_measured):
        raise ValueError("Prediction and truth complete-bone masks disagree")
    selected = complete & boolean(data.within_training_temperature_marginals, "temperature support")
    selected &= data["mode"].eq("P").to_numpy() & data.order.ne("Passeriformes").to_numpy()
    value_columns = [f"response_per_10C_{b}" for b in BONES] + [f"true_slope_{b}" for b in BONES]
    if not np.isfinite(data.loc[selected, value_columns].to_numpy(float)).all():
        raise ValueError("Supported complete P targets have nonfinite responses/truth")
    result = data.loc[selected, ["target_tip", "family"] + value_columns].copy()
    result["fold"] = int(fold)
    return result


def aggregate_recovery(frames, *, min_species=30, min_families=5):
    if not frames:
        return {"pass": False, "n_windows": 0, "n_species": 0, "n_families": 0,
                "reasons": ["no_supported_prediction_windows"], "metrics": {}}
    data = pd.concat(frames, ignore_index=True)
    if data.empty:
        return {"pass": False, "n_windows": 0, "n_species": 0, "n_families": 0,
                "reasons": ["no_supported_prediction_windows"], "metrics": {}}
    if data.groupby("target_tip").fold.nunique().gt(1).any():
        raise ValueError("A target species occurs in multiple held-out folds")
    if data.groupby("target_tip").family.nunique().gt(1).any():
        raise ValueError("A target species has conflicting family labels")
    columns = [f"response_per_10C_{b}" for b in BONES] + [f"true_slope_{b}" for b in BONES]
    species = data.groupby(["target_tip", "family"], sort=True)[columns].mean()
    for a, b in (("b_tmt", "b_tbt"), ("b_tbt", "b_fem")):
        species[f"response_per_10C_{a}_minus_{b}"] = species[f"response_per_10C_{a}"] - species[f"response_per_10C_{b}"]
        species[f"true_slope_{a}_minus_{b}"] = species[f"true_slope_{a}"] - species[f"true_slope_{b}"]
    family_means = species.groupby(level="family").mean()
    reasons = []
    if len(species) < min_species:
        reasons.append(f"fewer_than_{min_species}_supported_species")
    if len(family_means) < min_families:
        reasons.append(f"fewer_than_{min_families}_supported_families")
    metrics = {}
    for name in list(BONES) + ["b_tmt_minus_b_tbt", "b_tbt_minus_b_fem"]:
        prediction = float(family_means[f"response_per_10C_{name}"].mean())
        truth = float(family_means[f"true_slope_{name}"].mean())
        error = prediction - truth
        tolerance = max(0.003, 0.5 * abs(truth))
        passed = bool(abs(error) <= tolerance + 1e-12)
        metrics[name] = {"prediction": prediction, "truth": truth, "error": error,
                         "absolute_error": abs(error), "tolerance": tolerance, "pass": passed}
        if not passed:
            reasons.append("recovery_error:" + name)
    return {"pass": not reasons, "n_windows": len(data), "n_species": len(species),
            "n_families": len(family_means), "reasons": reasons, "metrics": metrics,
            "weighting": "windows averaged within species; species averaged within family; equal family weights for both prediction and truth"}


def summarize(run, root=ROOT):
    run, root = Path(run).resolve(), Path(root).resolve()
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    jobs = manifest["jobs"]
    if not jobs or len({j["name"] for j in jobs}) != len(jobs):
        raise ValueError("Manifest must contain uniquely named planned jobs")
    status_path = run / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
    inventory, metrics, loaded = [], [], {}
    for job in jobs:
        item = {k: job.get(k) for k in FIELDS}
        state = status.get(job["name"], {})
        item.update(training=state.get("training", "pending"), evaluation=state.get("evaluation", "pending"))
        path = run / "evaluation" / job["name"] / "pilot_results.json"
        item["result_present"] = path.is_file()
        item["summary_error"] = None
        item["complete"] = item["training"] == "done" and item["evaluation"] == "done" and path.is_file()
        if item["complete"]:
            try:
                result = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(result.get("metrics"), dict) or not result["metrics"]:
                    raise ValueError("Completed evaluator output has no metrics")
                loaded[job["name"]] = result
                for support, outcomes in result["metrics"].items():
                    for target, methods in outcomes.items():
                        for method, values in methods.items():
                            metrics.append({k: job.get(k) for k in FIELDS} | {"support": support, "target": target,
                                "method": method, **values})
            except (ValueError, TypeError, KeyError, OSError) as exc:
                item["complete"] = False
                item["summary_error"] = str(exc)
        inventory.append(item)
    by_name = {r["name"]: r for r in inventory}
    calibration = {"scope": "Development recovery heuristic only; no formal false-positive/power calibration, no automatic paper claims, no gradient-based winner selection",
                   "criterion": "Each bone slope and both adjacent slope contrasts must have absolute error <= max(0.003,0.5*abs(truth)); matched supported P/nonpasserine complete-bone targets, species then equal-family weighting",
                   "completeness": "All six scenarios, at least two seeds, all five folds and every configured simulation job must finish; each scenario/seed needs >=30 supported species and >=5 families",
                   "variants": {}}
    for variant in sorted({j["variant"] for j in jobs}):
        simjobs = [j for j in jobs if j["variant"] == variant and j["stage"] == "simulation"]
        reasons = []
        scenarios = {j["scenario"] for j in simjobs}
        seeds = {j["seed"] for j in simjobs}
        if scenarios != SCENARIOS:
            reasons.append("missing_calibration_scenarios:" + ",".join(sorted(SCENARIOS - scenarios)))
        if len(seeds) < 2:
            reasons.append("fewer_than_two_simulation_seeds")
        missing = [j["name"] for j in simjobs if not by_name[j["name"]]["complete"]]
        if missing:
            reasons.append("incomplete_configured_simulation_jobs")
        reports = []
        for scenario in sorted(scenarios):
            for seed in sorted(seeds):
                subset = [j for j in simjobs if j["scenario"] == scenario and j["seed"] == seed]
                group_reasons = []
                if len(subset) != 5 or {j["fold"] for j in subset} != set(range(5)):
                    group_reasons.append("five_folds_not_configured")
                frames = []
                for job in subset:
                    if not by_name[job["name"]]["complete"]:
                        group_reasons.append("incomplete:" + job["name"])
                        continue
                    try:
                        window_file = run / "evaluation" / job["name"] / "temperature_windows.csv.gz"
                        truth_file = (root / job["table"]).parent / "true_slopes.csv"
                        frames.append(paired_rows(pd.read_csv(window_file), pd.read_csv(truth_file), job["fold"]))
                    except (ValueError, TypeError, KeyError, OSError, pd.errors.ParserError) as exc:
                        group_reasons.append(f"invalid_response:{job['name']}:{exc}")
                try:
                    report = aggregate_recovery(frames)
                except (ValueError, KeyError) as exc:
                    report = {"pass": False, "n_windows": 0, "n_species": 0, "n_families": 0,
                              "metrics": {}, "reasons": [str(exc)]}
                report["reasons"] = group_reasons + report["reasons"]
                report["pass"] = not report["reasons"]
                report.update(scenario=scenario, seed=seed)
                reports.append(report)
        passed = bool(simjobs) and not reasons and bool(reports) and all(r["pass"] for r in reports)
        calibration["variants"][variant] = {"pass": passed, "planned_simulation_jobs": len(simjobs),
            "complete_simulation_jobs": len(simjobs) - len(missing), "incomplete_jobs": missing,
            "reasons": reasons, "scenario_seed_reports": reports}
    for row in metrics:
        row["development_calibration_pass"] = calibration["variants"][row["variant"]]["pass"]
    for row in inventory:
        row["development_calibration_pass"] = calibration["variants"][row["variant"]]["pass"]
    summary_columns = list(FIELDS) + ["support", "target", "method", "n_windows", "n_species", "n_families",
                                     "family_weighted_rmse", "family_weighted_mean_error", "development_calibration_pass"]
    pd.DataFrame(metrics, columns=summary_columns).to_csv(run / "summary.csv", index=False)
    pd.DataFrame(inventory).to_csv(run / "jobs.csv", index=False)
    (run / "calibration.json").write_text(json.dumps(calibration, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return {"planned": len(jobs), "complete": sum(bool(r["complete"]) for r in inventory),
            "calibration": {v: r["pass"] for v, r in calibration["variants"].items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--root", type=Path, default=ROOT)
    args = p.parse_args()
    print(json.dumps(summarize(args.run, args.root), ensure_ascii=False))


if __name__ == "__main__":
    main()
