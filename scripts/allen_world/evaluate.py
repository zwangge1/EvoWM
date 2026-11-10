"""Pilot diagnostics for a saved Allen JEPA object; no publication inference.

Only the final target in each H+1-species window is scored. Its morphology is
never passed to the predictor or linear baseline. Repeated windows are averaged
within species before equal weighting of families; no window-level confidence
intervals are produced. All four masks share one trained JEPA and one fitted
masked linear baseline per outcome.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import operator
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
MODES = ("full", "no_temperature", "no_motion", "neither")
TEMPERATURE_AXES = ("joint", "bio1_c", "bio5_c", "bio6_c")
BONES = ("b_fem", "b_tbt", "b_tmt")
ANATOMICAL_OUTCOMES = ("log10_total_leg", "log10_fem_over_tmt", "log10_tbt_over_tmt",
                      "log10_proportion_b_fem", "log10_proportion_b_tbt", "log10_proportion_b_tmt")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arithmetic(expression):
    """Resolve the upstream numeric ${eval:...} expressions without Python eval."""
    operators = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                 ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv}
    def walk(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in operators:
            return operators[type(node.op)](walk(node.left), walk(node.right))
        raise ValueError(f"Unsupported numeric config expression: {expression}")
    return walk(ast.parse(expression, mode="eval").body)


def load_config(checkpoint, config, root):
    from omegaconf import OmegaConf
    os.environ["EVOWM_ROOT"] = str(root)
    os.environ.setdefault("HJEPA_HOME", str(root / "outputs/hjepa"))
    OmegaConf.register_new_resolver("eval", arithmetic, replace=True)
    if config is None:
        candidates = [checkpoint.parent / "config.yaml", checkpoint.parent / "hydra/.hydra/config.yaml",
                      checkpoint.parent / ".hydra/config.yaml"]
        config = next((p for p in candidates if p.exists()), None)
        if config is None:
            raise FileNotFoundError("No saved training config next to checkpoint; supply --config")
    config = Path(config).resolve()
    cfg = OmegaConf.load(config)
    if "defaults" in cfg:
        from hydra import compose, initialize_config_dir
        with initialize_config_dir(config_dir=str(config.parent), version_base=None):
            cfg = compose(config_name=config.stem)
    return OmegaConf.to_container(cfg, resolve=True), config


def datasets(cfg, root):
    from evowm.data.allen import AllenDataset
    params = dict(cfg["data"]["dataset"])
    if params.pop("type", None) != "evowm_allen":
        raise ValueError("Evaluator accepts Allen configurations only")
    name = params.pop("name", "allen_train")
    val_name = params.pop("val_name", "allen_val")
    params["evowm_root"] = str(root)
    # Dataset construction remains identical to training; refresh does not affect
    # sample_windows, which uses explicit evaluation seeds.
    tr = AllenDataset(name=name, **params)
    va = AllenDataset(name=val_name, **params)
    if np.any(tr.rows_mask & va.rows_mask):
        raise AssertionError("Training and validation species overlap")
    if not tr.rows_mask.any() or not va.rows_mask.any():
        raise ValueError("Both training and held-out splits must be nonempty")
    if not np.allclose(tr.pix, va.pix) or not np.allclose(tr.condition_mu, va.condition_mu):
        raise AssertionError("Train and validation normalization differs")
    return tr, va


def temperature_indices(ds, axis="joint"):
    if axis not in TEMPERATURE_AXES:
        raise ValueError(f"Unknown temperature axis: {axis}")
    indices = np.flatnonzero(ds.col_group == ds.eco_group_names.index("temperature"))
    if axis != "joint":
        index = ds.eco_cols.index(axis)
        if index not in indices:
            raise ValueError(f"{axis} is not a temperature condition")
        indices = np.array([index])
    return indices


def action_windows(ds, windows, history, mode="full", temperature_delta=0., temperature_axis="joint"):
    out = []
    tidx = temperature_indices(ds, temperature_axis)
    for rows, dist in zip(windows.rows, windows.dist):
        if temperature_delta:
            raw = ds.condition_raw[rows].copy()
            raw[-1, tidx] += temperature_delta
            action = ds.build_action(rows, dist, mode=mode, raw_conditions=raw)
        else:
            action = ds.build_action(rows, dist, mode=mode)
        out.append(action[:history])
    return np.asarray(out, dtype=np.float32)


def with_ratio(values, columns):
    j_t, j_f = columns.index("b_tmt"), columns.index("b_fem")
    return np.column_stack([values, values[:, j_t] - values[:, j_f]])


def anatomical_coordinates(values, columns):
    """Derive global size and allocation from the same three measured bones.

    Partial triplets give no total/proportion labels. The two independent
    allocation coordinates use tarsometatarsus as their common denominator.
    """
    bones = np.asarray(values, dtype=float)[:, [columns.index(b) for b in BONES]]
    complete = np.isfinite(bones).all(axis=1)
    safe = np.where(complete[:, None], bones, 0.)
    total = np.logaddexp.reduce(safe * np.log(10.), axis=1) / np.log(10.)
    result = {"log10_total_leg": total,
              "log10_fem_over_tmt": safe[:, 0] - safe[:, 2],
              "log10_tbt_over_tmt": safe[:, 1] - safe[:, 2]}
    result.update({f"log10_proportion_{bone}": safe[:, j] - total for j, bone in enumerate(BONES)})
    return {name: np.where(complete, value, np.nan) for name, value in result.items()}


def outcome_values(values, columns, outcomes):
    """Use one coordinate transform for observations and every comparator."""
    values = np.asarray(values)
    basic = dict(zip(columns + ["log10_tmt_over_fem"], with_ratio(values, columns).T))
    if any(name in ANATOMICAL_OUTCOMES for name in outcomes):
        basic.update(anatomical_coordinates(values, columns))
    return np.column_stack([basic[name] for name in outcomes])


class AnatomicalPredictor:
    """NumPy batching bridge to the exact train-time hierarchical inference API.

    Unlike Level1, this receives raw reference pixels, never target morphology
    or precomputed target encodings. Its output is already in log10 mm.
    """
    def __init__(self, model):
        self.model = model.eval()

    def predict(self, context, actions, batch=512):
        import torch
        from evowm.models.allen_anatomical import predict_from_context
        context, actions = np.asarray(context), np.asarray(actions)
        if (batch < 1 or context.ndim != 3 or actions.ndim != 3
                or context.shape[:2] != actions.shape[:2] or not len(context)):
            raise ValueError("Expected nonempty matching (N,H,D) reference context and actions")
        if not np.isfinite(context).all() or not np.isfinite(actions).all():
            raise ValueError("Reference pixels/actions must be finite, with missingness encoded by masks")
        device = next(self.model.parameters(), torch.empty(0)).device
        chunks = []
        with torch.no_grad():
            for start in range(0, len(context), batch):
                pixels = torch.as_tensor(context[start:start + batch], dtype=torch.float32, device=device)
                conditions = torch.as_tensor(actions[start:start + batch], dtype=torch.float32, device=device)
                prediction = predict_from_context(self.model, pixels, conditions)["predicted_log_lengths"]
                if prediction.shape != (*pixels.shape[:2], 4) or not torch.isfinite(prediction).all():
                    raise ValueError("Hierarchical inference returned invalid log lengths")
                chunks.append(prediction.detach().cpu().numpy())
        return np.concatenate(chunks)


def prediction_backend(model, pixels, state, train_rows, columns, batch=512):
    """Select a complete inference path; never fit a new probe for v3 models."""
    settings = getattr(model, "allen_anatomical_settings", None)
    if settings is not None:
        if columns != ["m_tarsus", "b_fem", "b_tbt", "b_tmt"]:
            raise ValueError("Hierarchical readout requires canonical target ordering")
        settings = dict(settings)
        info = {"kind": "trained_anatomical_hierarchy", "levels": int(model.num_levels),
                "mode": settings.get("mode"), "connection": settings.get("connection"),
                "objective": settings.get("objective"), "settings": settings,
                "detail": "Saved anatomical model evaluated through predict_from_context, identical to training inference; only reference morphology supplied; no probe fit."}
        return AnatomicalPredictor(model), np.asarray(pixels), lambda prediction: prediction, info
    from evowm.eval.recovery import Level1
    predictor = Level1(model)
    embeddings = predictor.encode(pixels)
    decode, info = prediction_decoder(model, embeddings, state, train_rows, columns, batch=batch)
    return predictor, embeddings, decode, info


def prediction_decoder(model, embeddings, state, train_rows, columns, batch=512):
    """Select a trained anatomical readout or fit a train-only legacy probe.

    The returned callable receives predicted embeddings only. Supervised heads
    are never replaced by a probe fitted after training.
    """
    import torch
    from evowm.eval.recovery import apply_probe, fit_probe
    head = getattr(model.get_level(1), "allen_readout", None)
    if head is None:
        probes = fit_probe(embeddings, state, train_rows)
        return lambda z: apply_probe(probes, z), {
            "kind": "train_only_ridge_probe", "detail": "Ridge alpha=1 on measured training species only."}
    if columns != ["m_tarsus", "b_fem", "b_tbt", "b_tmt"]:
        raise ValueError("Anatomical readout requires canonical target ordering")
    head.eval()
    def decode(z):
        flat = np.asarray(z).reshape(-1, z.shape[-1])
        chunks = []
        with torch.no_grad():
            for start in range(0, len(flat), batch):
                values = head(torch.from_numpy(flat[start:start + batch]).float())
                chunks.append(values.detach().cpu().numpy())
        result = np.concatenate(chunks)
        if result.shape != (len(flat), 4) or not np.isfinite(result).all():
            raise ValueError("Anatomical readout produced invalid lengths")
        return result.reshape(*z.shape[:-1], 4)
    return decode, {"kind": "trained_anatomical_readout", "variant": str(getattr(head, "variant", getattr(head, "kind", "unknown"))),
                    "detail": "Saved end-to-end trained readout, raw log10 mm; no held-out or post-training probe fit."}


def biological_groups(table, rows):
    mode = table["mode"].to_numpy()[rows]
    passerine = table["order"].to_numpy()[rows] == "Passeriformes"
    return {
        "P_nonpasserine": (mode == "P") & ~passerine,
        "SP_nonpasserine": (mode == "SP") & ~passerine,
        "early_P_SP": np.isin(mode, ["P", "SP"]),
        "late_SA_A": np.isin(mode, ["SA", "A"]),
        "Passeriformes": passerine,
        "nonPasseriformes": ~passerine,
    }


def family_metric(errors, target_rows, table):
    """Mean squared error within species, then families; sqrt only at the end."""
    ok = np.isfinite(errors)
    if not ok.any():
        return {"n_windows": 0, "n_species": 0, "n_families": 0, "family_weighted_rmse": None,
                "family_weighted_mean_error": None}
    d = pd.DataFrame({"row": target_rows[ok], "error": errors[ok], "sq_error": errors[ok] ** 2})
    per_species = d.groupby("row", sort=True)[["error", "sq_error"]].mean()
    per_species["family"] = table.family.to_numpy()[per_species.index.to_numpy()]
    per_family = per_species.groupby("family")[["error", "sq_error"]].mean()
    return {"n_windows": int(ok.sum()), "n_species": len(per_species), "n_families": len(per_family),
            "family_weighted_rmse": float(np.sqrt(per_family.sq_error.mean())),
            "family_weighted_mean_error": float(per_family.error.mean())}


def allocation_metric(errors, target_rows, table):
    """Joint RMSE of two log-ratio coordinates, with no fictitious signed bias."""
    errors = np.asarray(errors, dtype=float)
    if errors.ndim != 2 or errors.shape[1] != 2:
        raise ValueError("Joint allocation errors must contain exactly two coordinates")
    finite = np.isfinite(errors).all(axis=1)
    distance = np.sqrt(np.mean(np.where(finite[:, None], errors, 0.) ** 2, axis=1))
    result = family_metric(np.where(finite, distance, np.nan), target_rows, table)
    result["family_weighted_mean_error"] = None
    result["coordinate_weighting"] = "Mean squared error across fem/tmt and tbt/tmt log10 coordinates before species/family aggregation; no signed multivariate bias defined."
    return result


def response_metric(values, rows, table):
    ok = np.isfinite(values)
    if not ok.any():
        return {"n_windows": 0, "n_species": 0, "n_families": 0, "family_weighted_mean": None}
    per_species = pd.DataFrame({"row": rows[ok], "value": values[ok]}).groupby("row").value.mean()
    families = table.family.to_numpy()[per_species.index.to_numpy()]
    return {"n_windows": int(ok.sum()), "n_species": len(per_species), "n_families": len(set(families)),
            "family_weighted_mean": float(pd.DataFrame({"value": per_species.to_numpy(), "family": families}).groupby("family").value.mean().mean())}


def temperature_diagnostic(ds_tr, ds_va, l1, decode, context_emb, windows, observed, outcomes,
                           raw, out, axis, step, batch, history, primary=False):
    """Temperature sensitivity with all three bones on a common measured support."""
    target = windows.rows[:, -1]
    table = ds_va.table
    tr_rows = np.flatnonzero(ds_tr.rows_mask)
    tidx = temperature_indices(ds_va, axis)
    # Support checks include all temperature coordinates, even when only one moves.
    all_tidx = temperature_indices(ds_va, "joint")
    train_temp = ds_tr.condition_raw[tr_rows][:, all_tidx]
    low, high = np.nanmin(train_temp, axis=0), np.nanmax(train_temp, axis=0)
    target_temp = ds_va.condition_raw[target][:, all_tidx]
    delta = np.array([step if idx in tidx else 0. for idx in all_tidx])
    support = (np.isfinite(target_temp).all(axis=1) & (target_temp - delta >= low).all(axis=1)
               & (target_temp + delta <= high).all(axis=1))
    perturbed = []
    cols = list(ds_tr.state_cols)
    for signed_step in (-step, step):
        actions = action_windows(ds_va, windows, history, temperature_delta=signed_step, temperature_axis=axis)
        perturbed.append(outcome_values(decode(l1.predict(context_emb, actions, batch=batch))[:, -1], cols, outcomes))
    slopes = (perturbed[1] - perturbed[0]) * (10. / (2 * step))
    slopes[~support] = np.nan
    bone_indices = [outcomes.index(b) for b in BONES]
    complete_bones = np.isfinite(observed[:, bone_indices]).all(axis=1)
    groups = biological_groups(table, target)
    thermal = raw[["window", "target_row", "target_tip", "family"]].copy()
    thermal["within_training_temperature_marginals"] = support
    thermal["complete_three_bones"] = complete_bones
    thermal["development_mode"] = table["mode"].to_numpy()[target]
    thermal["order"] = table["order"].to_numpy()[target]
    thermal["temperature_axis"] = axis
    diagnostics, by_group = {}, {name: {} for name in groups}
    for j, col in enumerate(outcomes):
        thermal[f"response_per_10C_{col}"] = slopes[:, j]
        thermal[f"measured_{col}"] = np.isfinite(observed[:, j])
        measured = np.isfinite(observed[:, j]) if col == "m_tarsus" else complete_bones
        diagnostics[col] = response_metric(np.where(measured, slopes[:, j], np.nan), target, table)
        for group, selected in groups.items():
            by_group[group][col] = response_metric(np.where(selected & measured, slopes[:, j], np.nan), target, table)
    for first, second in (("b_tmt", "b_tbt"), ("b_tbt", "b_fem"), ("b_tmt", "b_fem")):
        contrast = slopes[:, outcomes.index(first)] - slopes[:, outcomes.index(second)]
        name = f"{first}_minus_{second}"
        thermal[f"response_contrast_{name}"] = contrast
        diagnostics[name] = response_metric(np.where(complete_bones, contrast, np.nan), target, table)
        for group, selected in groups.items():
            by_group[group][name] = response_metric(np.where(selected & complete_bones, contrast, np.nan), target, table)
    # One coarse coordinate, not a sum of log lengths: log10(sum of physical lengths).
    def log_total(values):
        z = values[:, bone_indices] * np.log(10.)
        return np.logaddexp.reduce(z, axis=1) / np.log(10.)
    total_slope = (log_total(perturbed[1]) - log_total(perturbed[0])) * (10. / (2 * step))
    total_slope[~support] = np.nan
    thermal["response_per_10C_log10_total_leg"] = total_slope
    diagnostics["log10_total_leg"] = response_metric(np.where(complete_bones, total_slope, np.nan), target, table)
    for group, selected in groups.items():
        by_group[group]["log10_total_leg"] = response_metric(np.where(selected & complete_bones, total_slope, np.nan), target, table)
    filename = "temperature_windows.csv.gz" if primary else f"temperature_{axis}_windows.csv.gz"
    thermal.to_csv(out / filename, index=False, compression="gzip")
    moved = [ds_va.eco_cols[j] for j in tidx]
    return {
        "axis": axis, "scope": "Only final-target selected temperature coordinates are perturbed; all other conditions and source morphology fixed. This is conditional model sensitivity, not a causal/evolutionary estimate. A single-axis conditional sensitivity is not the legacy bio1-only PGLS slope; joint sensitivity moves all three temperatures together. Marginal range checks do not establish joint support.",
        "step_C": step, "columns": moved, "support_columns": [ds_va.eco_cols[j] for j in all_tidx],
        "training_min_C": low.tolist(), "training_max_C": high.tolist(),
        "within_marginal_support_windows": int(support.sum()), "raw_file": filename,
        "support_rule": "External tarsus uses observed external tarsus; all bone, ratio, total and contrast summaries share measured complete-three-bone targets and marginal temperature support.",
        "summary_measured_outcomes_only": diagnostics,
        "summary_by_development_measured_outcomes_only": by_group,
        "development_scope": "Family-assigned developmental modes. P_nonpasserine is the subgroup corresponding to the existing full-gradient finding; P+SP is reported separately, not substituted.",
    }


def observational_gradient(observed, predicted, rows, table, columns):
    """Matched descriptive OLS on species-averaged held-out P predictions.

    This is a pattern diagnostic; it is not the earlier PGLS and provides no
    phylogenetically calibrated significance test.
    """
    idx = [columns.index(c) for c in BONES]
    obs, pred = observed[:, idx], predicted[:, idx]
    selected = biological_groups(table, rows)["P_nonpasserine"]
    valid = selected & np.isfinite(obs).all(axis=1) & np.isfinite(pred).all(axis=1)
    detail = {"scope": "Descriptive same-sample OLS of held-out P/nonpasserine observed versus predicted log lengths. Predictions averaged per species. Not PGLS, not a replacement for the published gradient test, no significance claim.",
              "subgroup": "P_nonpasserine"}
    if not valid.any():
        return detail | {"status": "no_complete_targets", "n_species": 0}
    data = pd.DataFrame({"row": rows[valid]})
    for j, name in enumerate(BONES):
        data[f"observed_{name}"] = obs[valid, j]
        data[f"predicted_{name}"] = pred[valid, j]
    data = data.groupby("row").mean()
    info = table.iloc[data.index.to_numpy()]
    # P is fixed in this stratum, so developmental dummy columns add no variation.
    covariates = ["m_mass", "bio1_c", "e_habdens", "e_migr"] + [
        "e_life_" + s for s in ("aerial", "aquatic", "generalist", "terrestrial")]
    if any(c not in info for c in covariates):
        return detail | {"status": "missing_covariates", "n_species": len(data)}
    finite = np.isfinite(info[covariates].to_numpy(float)).all(axis=1)
    info, data = info.loc[finite], data.loc[finite]
    if len(data) < 12:
        return detail | {"status": "insufficient_species", "n_species": len(data)}
    candidates = {"intercept": np.ones(len(data))}
    candidates.update({c: info[c].to_numpy(float) for c in covariates if c != "bio1_c"})
    # Nuisance space first: if temperature is fully determined by that space,
    # do not invent a uniquely interpretable partial temperature coefficient.
    candidates["bio1_per_10C"] = info.bio1_c.to_numpy(float) / 10.
    names, design = [], np.empty((len(data), 0))
    for name, values in candidates.items():
        new = np.column_stack([design, values])
        if np.linalg.matrix_rank(new) > len(names):
            names.append(name); design = new
    if "bio1_per_10C" not in names or len(data) <= len(names) + 3:
        return detail | {"status": "unidentifiable_or_insufficient", "n_species": len(data)}
    t = names.index("bio1_per_10C")
    result = detail | {"status": "ok", "n_species": len(data), "n_families": int(info.family.nunique()),
                       "covariates": names, "observed": {}, "predicted": {}}
    for kind in ("observed", "predicted"):
        beta = np.linalg.lstsq(design, data[[f"{kind}_{b}" for b in BONES]].to_numpy(), rcond=None)[0][t]
        result[kind] = dict(zip(BONES, beta.tolist()))
        result[kind].update(b_tmt_minus_b_tbt=float(beta[2] - beta[1]), b_tbt_minus_b_fem=float(beta[1] - beta[0]),
                            b_tmt_minus_b_fem=float(beta[2] - beta[0]))
    return result


def evaluate(ds_tr, ds_va, model, cfg, out, *, n_orders=2, seed=0, batch=512, temperature_step=1.,
             temperature_axis="joint", all_temperature_axes=False):
    from sklearn.linear_model import Ridge
    from evowm.eval.recovery import sample_windows
    out.mkdir(parents=True, exist_ok=True)
    history = int(cfg["level1"]["wm"]["history_size"])
    if int(cfg["level1"]["wm"]["rollout_n"]) != 1:
        raise ValueError("Pilot evaluator supports one-step prediction only")
    cols = list(ds_tr.state_cols)
    if set(cols) != {"m_tarsus", "b_fem", "b_tbt", "b_tmt"}:
        raise ValueError(f"Expected four Allen length outcomes, received {cols}")
    state = ds_tr.S.astype(float)
    tr_rows = np.flatnonzero(ds_tr.rows_mask)
    if min(np.isfinite(state[tr_rows]).sum(axis=0)) < 2:
        raise ValueError("At least two training observations are required for each outcome probe")
    # v3 uses reference pixels directly; legacy paths encode rows before selecting
    # the reference window. No branch supplies target morphology to inference.
    l1, all_emb, decode, readout_info = prediction_backend(model, ds_tr.pix, state, tr_rows, cols, batch=batch)
    wt = sample_windows(ds_tr, n_orders, history + 1, seed)
    wv = sample_windows(ds_va, n_orders, history + 1, seed + 1)
    target = wv.rows[:, -1]
    source = wv.rows[:, -2]
    context_emb = all_emb[wv.rows[:, :history]]
    modes = tuple(getattr(ds_tr, "mask_modes", MODES))
    acts_tr = {m: action_windows(ds_tr, wt, history, m) for m in modes}
    acts_va = {m: action_windows(ds_va, wv, history, m) for m in modes}
    predictions = {}
    for m in modes:
        predictions[f"jepa_{m}"] = decode(l1.predict(context_emb, acts_va[m], batch=batch))[:, -1]

    # Same source morphology (with missing masks), H-step conditions and distances
    # as JEPA; no morphology of the final target is included.
    x_context_tr = ds_tr.pix[wt.rows[:, :history]].reshape(len(wt.rows), -1)
    x_context_va = ds_va.pix[wv.rows[:, :history]].reshape(len(wv.rows), -1)
    xtr = np.concatenate([np.concatenate([x_context_tr, acts_tr[m].reshape(len(wt.rows), -1)], axis=1) for m in modes])
    target_tr = np.tile(wt.rows[:, -1], len(modes))
    ytr = state[target_tr]
    counts = np.bincount(target_tr, minlength=len(state))
    weights = 1. / counts[target_tr]
    linear = []
    for j in range(len(cols)):
        ok = np.isfinite(ytr[:, j])
        linear.append(Ridge(alpha=1.).fit(xtr[ok], ytr[ok, j], sample_weight=weights[ok]))
    for m in modes:
        xv = np.concatenate([x_context_va, acts_va[m].reshape(len(wv.rows), -1)], axis=1)
        predictions[f"ridge_{m}"] = np.column_stack([reg.predict(xv) for reg in linear])
    predictions["copy_previous"] = state[source]
    outcomes = cols + ["log10_tmt_over_fem"]
    hierarchical = getattr(model, "allen_anatomical_settings", None) is not None
    if hierarchical:
        outcomes += list(ANATOMICAL_OUTCOMES)
    predictions = {name: outcome_values(pred, cols, outcomes) for name, pred in predictions.items()}
    observed = outcome_values(state[target], cols, outcomes)
    table = ds_va.table
    raw = pd.DataFrame({"window": np.arange(len(target)), "target_row": target, "source_row": source,
                        "target_tip": table.tip.to_numpy()[target], "source_tip": table.tip.to_numpy()[source],
                        "family": table.family.to_numpy()[target]})
    for j in range(history):
        raw[f"context_{j}_tip"] = table.tip.to_numpy()[wv.rows[:, j]]
    for col in ("order", "mode", "m_mass", "bio1_c", "bio5_c", "bio6_c", "X"):
        if col in table:
            raw[f"target_{col}"] = table[col].to_numpy()[target]
    groups = biological_groups(table, target)
    metrics = {"available": {}, "common_support_all_methods": {}, "by_subgroup_available": {g: {} for g in groups}}
    complete_bones = np.isfinite(state[target][:, [cols.index(b) for b in BONES]]).all(axis=1)
    if hierarchical:
        metrics["complete_three_bones"] = {}
        raw["complete_three_bones"] = complete_bones
    for j, c in enumerate(outcomes):
        raw[f"observed_{c}"] = observed[:, j]
        raw[f"observed_mask_{c}"] = np.isfinite(observed[:, j])
        common = np.isfinite(observed[:, j]) & np.logical_and.reduce([np.isfinite(p[:, j]) for p in predictions.values()])
        for name, pred in predictions.items():
            raw[f"{name}_{c}"] = pred[:, j]
            errors = pred[:, j] - observed[:, j]
            metrics["available"].setdefault(c, {})[name] = family_metric(errors, target, table)
            metrics["common_support_all_methods"].setdefault(c, {})[name] = family_metric(np.where(common, errors, np.nan), target, table)
            if hierarchical:
                metrics["complete_three_bones"].setdefault(c, {})[name] = family_metric(
                    np.where(complete_bones, errors, np.nan), target, table)
            for group, selected in groups.items():
                metrics["by_subgroup_available"][group].setdefault(c, {})[name] = family_metric(
                    np.where(selected, errors, np.nan), target, table)
    raw.to_csv(out / "prediction_windows.csv.gz", index=False, compression="gzip")
    result = {
        "scope": ("Single-checkpoint pilot diagnostics, not calibrated evolutionary or causal evidence. "
                  + ("Full anatomical inference path scored, including the configured upper-to-lower connection."
                     if hierarchical else "L1 scored even when checkpoint has an auxiliary legacy L2.")),
        "history": history, "scored_step": "final target only", "n_orders": n_orders, "seed": seed,
        "n_windows": len(wv.rows), "n_distinct_target_species": len(set(target)),
        "n_target_families": int(table.iloc[target].family.nunique()),
        "prediction_readout": readout_info, "mask_modes": list(modes),
        "probe_fit": readout_info["detail"],
        "linear_baseline": "One Ridge alpha=1 per outcome, jointly trained on all configured masks, same H source states and actions; total fit weight one per target species.",
        "aggregation": "Mean squared errors over windows within target species, mean within family, then equal family mean and square root. Common support includes copy, which needs measured previous-species outcome.",
        "metrics": metrics, "train_audit": ds_tr.audit(), "val_audit": ds_va.audit(),
        "observational_gradient": observational_gradient(observed, predictions["jepa_full"], target, table, outcomes),
    }
    if hierarchical:
        idx = [outcomes.index(c) for c in ("log10_fem_over_tmt", "log10_tbt_over_tmt")]
        common = complete_bones & np.logical_and.reduce([
            np.isfinite(pred[:, idx]).all(axis=1) for pred in predictions.values()])
        for name, pred in predictions.items():
            error = pred[:, idx] - observed[:, idx]
            for support, selected in (("available", complete_bones), ("complete_three_bones", complete_bones),
                                      ("common_support_all_methods", common)):
                metrics[support].setdefault("allocation_logratio_joint", {})[name] = allocation_metric(
                    np.where(selected[:, None], error, np.nan), target, table)
            for group, selected in groups.items():
                metrics["by_subgroup_available"][group].setdefault("allocation_logratio_joint", {})[name] = allocation_metric(
                    np.where((selected & complete_bones)[:, None], error, np.nan), target, table)
        result["anatomical_evaluation"] = {
            "coordinate_units": "log10 mm for lengths/total, log10 for proportions and ratios",
            "independent_allocation_coordinates": ["log10_fem_over_tmt", "log10_tbt_over_tmt"],
            "joint_allocation_metric": "allocation_logratio_joint: equal coordinate MSE, then existing species/family weighting; derived from the saved paired coordinate errors",
            "primary_bone_support": "complete_three_bones; external tarsus also reported on available observations",
            "derived_outcomes": list(ANATOMICAL_OUTCOMES),
            "derived_baseline_rule": "Totals and allocation coordinates are derived from each method's paired bone predictions, not fitted separately. Partial observed/copy triplets do not supply derived labels.",
        }
    if temperature_axis not in TEMPERATURE_AXES:
        raise ValueError(f"Unknown temperature axis: {temperature_axis}")
    if temperature_step > 0:
        axes = [temperature_axis] + ([a for a in TEMPERATURE_AXES if a != temperature_axis] if all_temperature_axes else [])
        result["temperature_diagnostics"] = {}
        for axis in axes:
            diagnostics = temperature_diagnostic(ds_tr, ds_va, l1, decode, context_emb, wv, observed, outcomes,
                                                 raw, out, axis, temperature_step, batch, history,
                                                 primary=axis == temperature_axis)
            result["temperature_diagnostics"][axis] = diagnostics
        result["temperature_diagnostic"] = result["temperature_diagnostics"][temperature_axis]
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path, help="Trusted local *_object.ckpt saved by training")
    p.add_argument("--config", type=Path, help="Prefer the exact saved training config.yaml")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--n-orders", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--temperature-step", type=float, default=1., help="Symmetric Celsius perturbation; 0 disables")
    p.add_argument("--temperature-axis", choices=TEMPERATURE_AXES, default="joint")
    p.add_argument("--all-temperature-axes", action="store_true", help="Evaluate joint and three single-axis shifts using the same windows")
    args = p.parse_args()
    if min(args.n_orders, args.batch, args.threads) < 1 or args.temperature_step < 0 or not np.isfinite(args.temperature_step):
        p.error("Counts must be positive and temperature-step finite/nonnegative")
    root, checkpoint = args.root.resolve(), args.checkpoint.resolve()
    import torch
    from threadpoolctl import threadpool_limits
    from evowm.eval.recovery import load_world_model
    torch.set_num_threads(args.threads)
    cfg, config_path = load_config(checkpoint, args.config, root)
    ds_tr, ds_va = datasets(cfg, root)
    model = load_world_model(checkpoint, root / "hjepa")
    if not hasattr(model, "get_level"):
        raise TypeError("Need the saved model object checkpoint, not a Lightning state dict")
    with threadpool_limits(limits=args.threads):
        result = evaluate(ds_tr, ds_va, model, cfg, args.out_dir, n_orders=args.n_orders, seed=args.seed,
                          batch=args.batch, temperature_step=args.temperature_step,
                          temperature_axis=args.temperature_axis, all_temperature_axes=args.all_temperature_axes)
    table_path = root / cfg["data"]["dataset"]["species_table"]
    tree_path = root / cfg["data"]["dataset"]["tree_file"]
    result["provenance"] = {
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
        "config": str(config_path), "config_sha256": sha(config_path),
        "resolved_config": cfg, "table": str(table_path), "table_sha256": sha(table_path),
        "table_metadata_sha256": sha(table_path.with_suffix(".columns.json")),
        "tree_sha256": sha(tree_path), "evaluator_sha256": sha(__file__), "threads": args.threads,
    }
    (args.out_dir / "pilot_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Pilot diagnostics written to {args.out_dir.resolve()}; {result['n_distinct_target_species']} distinct held-out targets")


if __name__ == "__main__":
    main()
