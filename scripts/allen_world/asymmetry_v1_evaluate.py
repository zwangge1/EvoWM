"""Matched CPU evaluation of sparse-label training and incomplete references.

Every predictor sees the same three reference species and the same available
measurements. Target measurements enter only the output/scoring table. BM and
OU are fitted anew on each cell's masked training table; no full-data fit is
reused. This module deliberately does not perturb ecological covariates.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
STATE_COLUMNS = ("m_tarsus", "b_fem", "b_tbt", "b_tmt")
REFERENCE_CASES = ("observed", "hide_half_bones", "hide_all_bones", "hide_distal")


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(value), ensure_ascii=False,
                                    indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def visible_references(cell, reference_case):
    """Use the same species-stable masking contract as input preparation."""
    from asymmetry_v1_plan import make_reference_states
    if reference_case not in REFERENCE_CASES:
        raise ValueError("Unknown reference case: " + str(reference_case))
    return make_reference_states(cell.states, cell.reference_rows, cell.table.tip.to_numpy(),
                                 reference_case, cell.job["mask_seed"], cell.job["fold"])


def mask_reference_pixels(pixels, available):
    """Hide both standardized values and visibility flags, without mutation."""
    pixels, available = np.asarray(pixels), np.asarray(available)
    if pixels.shape != (*available.shape[:-1], 8) or available.shape[-1] != 4:
        raise ValueError("Expected four values followed by their four masks")
    if available.dtype != bool or not np.isfinite(pixels).all():
        raise ValueError("Reference pixels must be finite and availability boolean")
    original = pixels[..., 4:] > 0
    if np.any(available & ~original):
        raise ValueError("Cannot expose a naturally missing reference measurement")
    result = pixels.copy()
    result[..., :4] = np.where(available, result[..., :4], 0)
    result[..., 4:] = available.astype(result.dtype)
    return result


def validate_architecture(options, variant, num_levels):
    expected = {"full": ("hierarchical", "full", "jepa", 2),
                "single": ("single", "none", "jepa", 1),
                "supervised": ("hierarchical", "full", "supervised", 2)}
    actual = (options.get("mode"), options.get("connection"), options.get("objective"), int(num_levels))
    if variant not in expected or actual != expected[variant]:
        raise ValueError("Checkpoint architecture differs from the declared job")


def _indices(values, n, label, ndim=1):
    values = np.asarray(values)
    if (values.ndim != ndim or not np.issubdtype(values.dtype, np.integer)
            or np.any(values < 0) or np.any(values >= n)):
        raise ValueError("Invalid " + label)
    return values


@dataclass
class CellData:
    run: Path
    root: Path
    job: dict
    entry: dict
    table: pd.DataFrame
    states: np.ndarray
    train_rows: np.ndarray
    validation_rows: np.ndarray
    reference_rows: np.ndarray
    target_rows: np.ndarray
    distances: np.ndarray
    tip_ids_of_rows: np.ndarray
    table_path: Path
    tree_path: Path
    npz_path: Path
    hashes: dict


def load_cell(run, job_name, root=ROOT):
    root = Path(root).resolve()
    run = Path(run)
    run = (root / run).resolve() if not run.is_absolute() else run.resolve()
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    jobs = [j for j in manifest["jobs"] if j["name"] == job_name]
    if len(jobs) != 1:
        raise ValueError("Expected exactly one manifest job named " + job_name)
    job = jobs[0]
    entry = manifest["inputs"]["tables"][job["input_key"]]
    for field in ("condition", "fold", "replicate", "mask_seed"):
        if field in entry and entry[field] != job[field]:
            raise ValueError("Job/input identity mismatch: " + field)
    paths = {}
    for field in ("table", "tree", "npz", "metadata", "mask_table", "audit"):
        if field not in entry or (field in ("mask_table", "audit") and not entry.get("hashes")):
            continue
        value = Path(entry[field])
        path = (root / value).resolve() if not value.is_absolute() else value.resolve()
        if not path.is_relative_to(run) or not path.is_file():
            raise ValueError("Expected a frozen input within this run: " + str(path))
        paths[field] = path
    hashes = {("table_metadata_sha256" if key == "metadata" else key + "_sha256"): sha(value)
              for key, value in paths.items()}
    for key, digest in entry.get("hashes", {}).items():
        if hashes.get(key) != digest:
            raise ValueError("Frozen input hash mismatch: " + key)
    table = pd.read_csv(paths["table"])
    if table.tip.isna().any() or table.tip.duplicated().any() or table.family.isna().any():
        raise ValueError("Species must have unique tips and nonmissing families")
    states = table[list(STATE_COLUMNS)].to_numpy(np.float32).astype(float)
    if np.isinf(states).any():
        raise ValueError("Infinite morphology is not a missing observation")
    with np.load(paths["npz"], allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    train = _indices(data["train_rows"], len(table), "training rows")
    val = _indices(data["validation_rows"], len(table), "validation rows")
    refs = _indices(data["reference_rows"], len(table), "reference rows", ndim=2)
    target = _indices(data["target_rows"], len(table), "target rows")
    if (not len(train) or not len(val) or not len(target) or refs.shape != (len(target), 3)
            or len(np.unique(train)) != len(train) or len(np.unique(val)) != len(val)):
        raise ValueError("Expected nonempty frozen splits and three-reference windows")
    if (np.intersect1d(train, val).size or not np.isin(refs, val).all()
            or not np.isin(target, val).all() or np.any(refs == target[:, None])
            or any(len(set(row)) != 3 for row in refs)):
        raise ValueError("Train/test or target/reference leakage in frozen inputs")
    if set(table.family.iloc[train]) & set(table.family.iloc[val]):
        raise ValueError("Training and validation families overlap")
    distances = np.asarray(data["distances"], dtype=float)
    if distances.shape != refs.shape or not np.isfinite(distances).all() or (distances < 0).any():
        raise ValueError("Invalid frozen adjacent distances")
    tips = np.asarray(data["tip_ids_of_rows"])
    if tips.shape != (len(table),) or not np.issubdtype(tips.dtype, np.integer):
        raise ValueError("Invalid tree tip mapping")
    required = np.unique(np.r_[train, refs.ravel(), target])
    if (tips[required] < 0).any():
        raise ValueError("Required species missing from tree")
    kept = train[np.isfinite(states[train, 1:]).all(axis=1)]
    if (not np.isfinite(states[train, 0]).all()
            or not np.array_equal(np.isfinite(states[train, 1:]), np.isfinite(states[train, 1:2]).repeat(3, axis=1))):
        raise ValueError("Training bones must have the nested triplet observation pattern")
    if "kept_train_bone_rows" in data and not np.array_equal(np.sort(kept), np.sort(data["kept_train_bone_rows"])):
        raise ValueError("Saved kept-bone rows disagree with the masked table")
    if "hidden_train_bone_rows" in data:
        hidden = _indices(data["hidden_train_bone_rows"], len(table), "hidden training bone rows")
        if not np.isin(hidden, train).all() or np.isfinite(states[hidden, 1:]).any():
            raise ValueError("Hidden training bone labels remain visible")
    return CellData(run, root, job, entry, table, states, train, val, refs, target,
                    distances, tips, paths["table"], paths["tree"], paths["npz"],
                    hashes)


def prediction_frame(cell, predicted, model, reference_case, available):
    predicted = np.asarray(predicted, dtype=float)
    if predicted.shape != (len(cell.target_rows), 4) or not np.isfinite(predicted).all():
        raise ValueError("Expected finite predictions for four target traits")
    rows = cell.target_rows
    baseline_available = np.isfinite(cell.states[cell.reference_rows])
    frame = pd.DataFrame({"model": model, "condition": cell.job["condition"],
        "reference_case": reference_case, "fold": int(cell.job["fold"]),
        "replicate": int(cell.job["replicate"]), "seed": int(cell.job["seed"]),
        "mask_seed": int(cell.job["mask_seed"]), "window": np.arange(len(rows)),
        "target_row": rows, "target_tip": cell.table.tip.to_numpy()[rows],
        "family": cell.table.family.to_numpy()[rows],
        "has_reference_bones": baseline_available[..., 1:].any(axis=(1, 2)),
        "visible_reference_bone_values": available[..., 1:].sum(axis=(1, 2)),
        "complete_three_bones": np.isfinite(cell.states[rows, 1:]).all(axis=1)})
    for j, col in enumerate(STATE_COLUMNS):
        frame["observed_" + col] = cell.states[rows, j]
        frame["predicted_" + col] = predicted[:, j]
    for index in range(3):
        frame[f"context_{index}_tip"] = cell.table.tip.to_numpy()[cell.reference_rows[:, index]]
    return frame


def _output_dir(cell):
    out = (cell.run / "evaluation" / cell.job["name"]).resolve()
    if out.parent != (cell.run / "evaluation").resolve():
        raise ValueError("Job name must be a single directory component")
    if (out / "result.json").exists():
        raise FileExistsError("Completed evaluation already exists: " + str(out))
    out.mkdir(parents=True, exist_ok=True)
    return out


def _finish(cell, out, frames, details, started):
    frame = pd.concat(frames, ignore_index=True)
    if frame.duplicated(["model", "reference_case", "window"]).any():
        raise ValueError("Duplicate evaluation model/case/window")
    temporary = out / "predictions.csv.gz.tmp"
    frame.to_csv(temporary, index=False, compression="gzip")
    os.replace(temporary, out / "predictions.csv.gz")
    result = {"status": "complete", "job": cell.job, "input_sha256": cell.hashes,
        "reference_cases": list(REFERENCE_CASES), "n_windows_per_case": len(cell.target_rows),
        "n_training_species": len(cell.train_rows),
        "n_training_bone_species": int(np.isfinite(cell.states[cell.train_rows, 1:]).all(axis=1).sum()),
        "n_prediction_rows": len(frame), "seconds": time.monotonic() - started,
        "target_measurements": "Scoring only; never passed to either inference backend.",
        "reference_mask": "Shared plan helper: species-stable SHA256 keyed by mask seed/fold; half removes floor(N/2) complete reference species, identically for every predictor.",
        **details}
    result["output_sha256"] = {p.name: sha(p) for p in out.iterdir()
                               if p.is_file() and not p.name.endswith(".tmp") and p.name != "result.json"}
    save_json(out / "result.json", result)
    print("EVALUATED " + cell.job["name"], flush=True)
    return result


def evaluate_classic(run, job_name, root=ROOT):
    from evowm.data.tree import read_trees
    from phylo_baseline_core import PhyloKernel, PhyloOptimizationError, fit_nested, predict_joint
    from phylo_baseline_data import build_condition_design
    started = time.monotonic()
    cell = load_cell(run, job_name, root)
    if cell.job["kind"] != "classic":
        raise ValueError("Expected a classic job")
    out = _output_dir(cell)
    trees = read_trees(cell.tree_path)
    if len(trees) != 1:
        raise ValueError("Expected exactly one frozen tree")
    tree = trees[0]
    required = np.unique(np.r_[cell.train_rows, cell.reference_rows.ravel(), cell.target_rows])
    if not np.array_equal(np.asarray(tree.tip_names)[cell.tip_ids_of_rows[required]],
                          cell.table.tip.to_numpy()[required]):
        raise ValueError("Frozen tree tip identities disagree with the table")
    train, states, tips = cell.train_rows, cell.states, cell.tip_ids_of_rows
    bone_rows = train[np.isfinite(states[train, 1:]).all(axis=1)]
    X, design = build_condition_design(cell.table, train, rank_support_rows=bone_rows)
    save_json(out / "design.json", design)
    kernel = PhyloKernel(tree)
    frames, fits = [], {}
    for model in ("BM", "OU"):
        print(f"FIT_BEGIN {cell.job['name']} {model} species={len(train)} bone_species={len(bone_rows)}", flush=True)
        try:
            fit = fit_nested(kernel, tips[train], X[train], states[train], model=model, estimate_eta=True)
        except PhyloOptimizationError as exc:
            save_json(out / (model.lower() + "_optimization_failure.json"), exc.diagnostics)
            raise
        if model == "OU" and fit["loglik"] < fits["BM"]["loglik"] - 1e-6:
            raise ValueError("OU fit is worse than its included BM boundary")
        save_json(out / (model.lower() + "_fit.json"), fit)
        fits[model] = {key: fit[key] for key in ("alpha", "eta", "loglik")}
        for case in REFERENCE_CASES:
            visible = visible_references(cell, case)
            available = np.isfinite(visible)
            prediction = np.empty((len(cell.target_rows), 4))
            for i, (refs, target) in enumerate(zip(cell.reference_rows, cell.target_rows)):
                # Hide the values as well as passing an explicit observation mask.
                prediction[i] = predict_joint(kernel, fit, tips[refs], X[refs], visible[i],
                    tips[target:target + 1], X[target:target + 1], ref_mask=available[i])[0]
            frames.append(prediction_frame(cell, prediction, model, case, available))
    return _finish(cell, out, frames, {"fits": fits, "n_predictors": X.shape[1],
        "fit_scope": "All training external observations and retained training bone triplets; no validation labels.",
        "classic_form": "Four-trait BM/fixed-root scalar-alpha OU conditional regression with proportional tip error."}, started)


def evaluate_neural(run, job_name, checkpoint, config=None, batch=512, root=ROOT):
    from evaluate import load_config, datasets, action_windows, AnatomicalPredictor
    from evowm.eval.recovery import Windows, load_world_model
    started = time.monotonic()
    cell = load_cell(run, job_name, root)
    if cell.job["kind"] != "neural":
        raise ValueError("Expected a neural job")
    checkpoint = Path(checkpoint).resolve()
    cfg, config_path = load_config(checkpoint, config, cell.root)
    params = cfg["data"]["dataset"]
    for field, expected in (("species_table", cell.table_path), ("tree_file", cell.tree_path)):
        actual = (cell.root / params[field]).resolve()
        if actual != expected:
            raise ValueError("Checkpoint configuration uses another " + field)
    if int(params["holdout"]["fold"]) != int(cell.job["fold"]) or int(cfg["seed"]) != int(cell.job["seed"]):
        raise ValueError("Checkpoint configuration has another fold or seed")
    train, validation = datasets(cfg, cell.root)
    if (not np.array_equal(np.flatnonzero(train.rows_mask), cell.train_rows)
            or not np.array_equal(np.flatnonzero(validation.rows_mask), cell.validation_rows)):
        raise ValueError("Checkpoint dataset split differs from frozen input")
    if not np.array_equal(train.S.astype(float), cell.states, equal_nan=True):
        raise ValueError("Checkpoint dataset morphology differs from frozen table")
    if list(train.morph_cols) != list(STATE_COLUMNS):
        raise ValueError("Expected canonical reference morphology order")
    model = load_world_model(checkpoint, cell.root / "hjepa")
    options = getattr(model, "allen_anatomical_settings", {})
    validate_architecture(options, cell.job["variant"], model.num_levels)
    history = int(model.allen_anatomical_history)
    if history != 3:
        raise ValueError("Expected exactly three references")
    out = _output_dir(cell)
    predictor = AnatomicalPredictor(model)
    windows = Windows(np.column_stack([cell.reference_rows, cell.target_rows]), cell.distances)
    actions = action_windows(validation, windows, history, mode="full")
    pixels = validation.pix[cell.reference_rows]
    frames = []
    for case in REFERENCE_CASES:
        available = np.isfinite(visible_references(cell, case))
        context = mask_reference_pixels(pixels, available)
        prediction = predictor.predict(context, actions, batch=batch)[:, -1]
        frames.append(prediction_frame(cell, prediction, cell.job["variant"], case, available))
    return _finish(cell, out, frames, {"checkpoint": str(checkpoint),
        "checkpoint_sha256": sha(checkpoint), "config": str(config_path),
        "config_sha256": sha(config_path), "train_audit": train.audit(),
        "architecture": options, "inference_device": "cpu"}, started)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("neural", "classic"):
        command = commands.add_parser(name)
        command.add_argument("--run", type=Path, required=True)
        command.add_argument("--job", required=True)
        command.add_argument("--threads", type=int, default=1)
        if name == "neural":
            command.add_argument("--checkpoint", type=Path, required=True)
            command.add_argument("--config", type=Path)
            command.add_argument("--batch", type=int, default=512)
    args = parser.parse_args(argv)
    if args.threads < 1 or (args.command == "neural" and args.batch < 1):
        parser.error("Threads and batch size must be positive")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(args.threads)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=args.threads):
        if args.command == "classic":
            evaluate_classic(args.run, args.job)
        else:
            import torch
            torch.set_num_threads(args.threads)
            evaluate_neural(args.run, args.job, args.checkpoint, args.config, args.batch)


if __name__ == "__main__":
    main()
