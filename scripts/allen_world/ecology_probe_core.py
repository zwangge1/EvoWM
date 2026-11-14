"""Pure-array conditional probes for frozen Allen predictors.

These helpers never train a model or use target morphology to construct a
condition. Reallocation means equal transfer between two observed foraging
fractions; it is not a causal intervention or a share of evolutionary causes.
An invalid proposed transfer is flagged and left unchanged, never clipped.
Callers must exclude invalid rows and check joint training-data support before
inference. Numeric feasibility alone is not an empirical support check.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


TEMPERATURE_COLUMNS = ("bio1_c", "bio5_c", "bio6_c")
FORAGING_COLUMNS = tuple("e_fs_" + s for s in
    ("watbelowsurf", "wataroundsurf", "ground", "understory", "midhigh", "canopy", "aerial"))
MOTION_COLUMNS = tuple("e_life_" + s for s in
    ("aerial", "aquatic", "generalist", "insessorial", "terrestrial"))
DEVELOPMENT_COLUMNS = tuple("e_dev_" + s for s in ("P", "SP", "SA", "A"))
CONDITION_COLUMNS = (TEMPERATURE_COLUMNS + FORAGING_COLUMNS + MOTION_COLUMNS
                     + DEVELOPMENT_COLUMNS + ("m_mass", "e_habdens", "e_migr"))
STATE_COLUMNS = ("m_tarsus", "b_fem", "b_tbt", "b_tmt")
OUTCOME_NAMES = ("m_tarsus", "log10_total_leg", "b_fem", "b_tbt", "b_tmt",
                 "log10_proportion_b_fem", "log10_proportion_b_tbt",
                 "log10_proportion_b_tmt", "log10_fem_over_tmt", "log10_tbt_over_tmt")
SHARE_INDICES = (5, 6, 7)
SCENARIO_NAMES = ("observed", "shifted", "cold", "cold_shifted", "warm", "warm_shifted")
COMPARISON_NAMES = ("observed_ecology", "cold_ecology", "warm_ecology",
                    "temperature_original", "temperature_shifted", "temperature_by_ecology")
METRIC_NAMES = ("delta_log10", "percent_change", "share_pp",
               "response_factor_change_percent", "interaction_per10C_log10")
METRIC_DESCRIPTIONS = {
    "delta_log10": "After minus before in log10 outcome; interaction is warm ecology contrast minus cold ecology contrast.",
    "percent_change": "100*(10**delta_log10-1), relative change of length, share or ratio; not a causal contribution. Not reported for the interaction row.",
    "share_pp": "100*(share_after-share_before); only three share columns, not relative percent. Not reported for the interaction row.",
    "response_factor_change_percent": "Interaction only: 100*(warm ecological response factor/cold ecological response factor-1). Not warm percent minus cold percent.",
    "interaction_per10C_log10": "Interaction log difference divided by cold-to-warm Celsius span and multiplied by 10; ecological transfer magnitude is unchanged.",
}


def _readonly(values):
    arr = np.array(values, copy=True)
    arr.setflags(write=False)
    return arr


def _number(value, name, *, positive=False, unit_interval=False):
    value = float(value)
    if not np.isfinite(value) or (positive and value <= 0) or (unit_interval and not 0 <= value <= 1):
        raise ValueError("Invalid " + name)
    return value


def _condition_matrix(values, columns, ndim):
    columns = tuple(columns)
    if len(columns) != len(set(columns)) or set(columns) != set(CONDITION_COLUMNS):
        raise ValueError("Expected the 22 distinct original Allen condition columns")
    arr = np.asarray(values, dtype=float)
    if arr.ndim != ndim or not len(arr) or arr.shape[-1] != len(columns):
        raise ValueError("Condition dimensions do not match columns")
    if np.isinf(arr).any():
        raise ValueError("Infinite condition values are not missing observations")
    return arr, columns


def _foraging_name(name):
    name = str(name)
    if not name.startswith("e_fs_"):
        name = "e_fs_" + name
    if name not in FORAGING_COLUMNS:
        raise ValueError("Unknown foraging component: " + name)
    return name


@dataclass(frozen=True)
class ConditionScenarios:
    """Read-only snapshots. Masks index target rows (or windows), not predictors."""
    values: Dict[str, np.ndarray]
    ecology_feasible: np.ndarray
    temperature_feasible: np.ndarray
    joint_feasible: np.ndarray
    ecology_reason: np.ndarray
    temperature_reason: np.ndarray
    metadata: dict

    def valid_for(self, scenario):
        if scenario not in SCENARIO_NAMES:
            raise ValueError("Unknown scenario")
        if scenario == "observed":
            return _readonly(np.ones(len(self.ecology_feasible), dtype=bool))
        if scenario == "shifted":
            return self.ecology_feasible
        if scenario in ("cold", "warm"):
            return self.temperature_feasible
        return self.joint_feasible


def target_condition_scenarios(raw_targets, columns=CONDITION_COLUMNS, *,
                               recipient, donor, fraction_shift=.1,
                               temperature_step=1., temperature_axis="joint",
                               total_atol=1e-7):
    """Build six target-only scenarios from raw N x 22 conditions.

    recipient gains fraction_shift and donor loses it. Complete fractions must
    be in [0,1] and sum approximately to 0.99 or 1 (the original source values).
    No closure, pseudocounts or clipping is performed. Infeasible ecology rows
    retain the original composition, with ecology_feasible=False.

    Temperature shifts preserve NaNs and every unselected value. All three
    climate coordinates must be observed for temperature_feasible=True; this
    flag does not establish joint environmental support. Joint shifts preserve
    existing climate offsets. Single-axis shifts additionally reject resulting
    bio6 > bio1 or bio1 > bio5 on either cold/warm endpoint.
    """
    raw, columns = _condition_matrix(raw_targets, columns, 2)
    recipient, donor = _foraging_name(recipient), _foraging_name(donor)
    if recipient == donor:
        raise ValueError("Recipient and donor must differ")
    fraction_shift = _number(fraction_shift, "fraction_shift", unit_interval=True)
    temperature_step = _number(temperature_step, "temperature_step", positive=True)
    if temperature_step > np.finfo(float).max / 2:
        raise ValueError("Cold-to-warm temperature span overflows")
    total_atol = _number(total_atol, "total_atol", positive=True)
    if temperature_axis not in ("joint", *TEMPERATURE_COLUMNS):
        raise ValueError("Unknown temperature axis")
    fi = [columns.index(c) for c in FORAGING_COLUMNS]
    ti = [columns.index(c) for c in TEMPERATURE_COLUMNS]
    ai, bi = columns.index(recipient), columns.index(donor)
    fractions = raw[:, fi]
    complete = np.isfinite(fractions).all(axis=1)
    bounded = complete & ((fractions >= 0) & (fractions <= 1)).all(axis=1)
    total = fractions.sum(axis=1)
    closed = (np.isclose(total, .99, atol=total_atol, rtol=0)
              | np.isclose(total, 1., atol=total_atol, rtol=0))
    transferable = ((raw[:, bi] >= fraction_shift)
                    & (raw[:, ai] + fraction_shift <= 1.))
    feasible = bounded & closed & transferable
    reason = np.full(len(raw), "ok", dtype="U40")
    reason[~complete] = "missing_foraging_component"
    reason[complete & ~bounded] = "fraction_outside_unit_interval"
    reason[bounded & ~closed] = "unsupported_fraction_sum"
    reason[bounded & closed & ~transferable] = "infeasible_requested_transfer"
    shifted = raw.copy()
    shifted[feasible, ai] += fraction_shift
    shifted[feasible, bi] -= fraction_shift
    if not np.allclose(shifted[feasible][:, fi].sum(axis=1), total[feasible],
                       atol=1e-14, rtol=0):
        raise AssertionError("Transfer changed a row's original fraction sum")
    changed = [columns.index(temperature_axis)] if temperature_axis != "joint" else ti
    cold, warm = raw.copy(), raw.copy()
    cold_shifted, warm_shifted = shifted.copy(), shifted.copy()
    try:
        with np.errstate(over="raise", invalid="raise"):
            for array, sign in ((cold, -1), (cold_shifted, -1), (warm, 1), (warm_shifted, 1)):
                array[:, changed] += sign * temperature_step
    except FloatingPointError as exc:
        raise ValueError("Temperature shift overflows; inspect raw conditions") from exc
    temp_complete = np.isfinite(raw[:, ti]).all(axis=1)
    order_ok = np.ones(len(raw), dtype=bool)
    for array in (raw, cold, warm):
        # ti order is annual mean, warmest-month maximum, coldest-month minimum.
        order_ok &= (array[:, ti[2]] <= array[:, ti[0]]) & (array[:, ti[0]] <= array[:, ti[1]])
    temperature_feasible = temp_complete & order_ok
    temp_reason = np.full(len(raw), "ok", dtype="U40")
    temp_reason[~temp_complete] = "missing_temperature_coordinate"
    temp_reason[temp_complete & ~order_ok] = "invalid_temperature_order"
    values = dict(zip(SCENARIO_NAMES, (raw, shifted, cold, cold_shifted, warm, warm_shifted)))
    metadata = {
        "recipient": recipient, "donor": donor, "fraction_shift": fraction_shift,
        "fraction_shift_percentage_points": 100 * fraction_shift,
        "temperature_axis": temperature_axis, "temperature_half_step_C": temperature_step,
        "temperature_span_C": 2 * temperature_step, "condition_columns": list(columns),
        "scenario_names": list(SCENARIO_NAMES),
        "fraction_rule": "Retain the original observed 0.99/1 sum; transfer equal amounts; never close or clip.",
        "invalid_ecology_rule": "Original composition is retained; caller must subset by feasibility before changed-condition inference.",
        "temperature_rule": "Shift selected observed values equally; NaNs stay missing. Validity requires complete three-temperature coordinates and ordered endpoints.",
        "support_note": "Feasibility is numeric only; empirical joint support must be checked separately.",
        "scope": "Frozen-model conditional sensitivity, not an observed intervention or causal contribution.",
    }
    return ConditionScenarios({k: _readonly(v) for k, v in values.items()},
                              _readonly(feasible), _readonly(temperature_feasible),
                              _readonly(feasible & temperature_feasible), _readonly(reason),
                              _readonly(temp_reason), metadata)


def window_condition_scenarios(raw_windows, columns=CONDITION_COLUMNS, **kwargs):
    """Build scenarios for N x (references+target) x 22; change final row only."""
    windows, columns = _condition_matrix(raw_windows, columns, 3)
    if windows.shape[1] < 2:
        raise ValueError("Expected at least one reference and one final target")
    target = target_condition_scenarios(windows[:, -1], columns, **kwargs)
    values = {}
    for name in SCENARIO_NAMES:
        changed = windows.copy()
        changed[:, -1] = target.values[name]
        values[name] = _readonly(changed)
    meta = dict(target.metadata, reference_rows_unchanged=True,
                n_reference_rows=int(windows.shape[1] - 1))
    return ConditionScenarios(values, target.ecology_feasible, target.temperature_feasible,
                              target.joint_feasible, target.ecology_reason,
                              target.temperature_reason, meta)


def target_scenarios_for_pairs(raw_targets, pairs, columns=CONDITION_COLUMNS, **kwargs):
    """Return a dictionary keyed by (recipient, donor), preserving pair order."""
    pairs = [(_foraging_name(a), _foraging_name(b)) for a, b in pairs]
    if not pairs or len(pairs) != len(set(pairs)):
        raise ValueError("Expected nonempty distinct directed component pairs")
    return {pair: target_condition_scenarios(raw_targets, columns,
                                            recipient=pair[0], donor=pair[1], **kwargs)
            for pair in pairs}


def outcome_coordinates(log_lengths, columns=STATE_COLUMNS):
    """N x 4 log10 lengths -> N x 10 outcomes, ordered by OUTCOME_NAMES.

    Three-bone coordinates require all three bones. Individual observed lengths
    remain available when other measurements are missing. No exponentiation of
    absolute log lengths is needed, so large finite log lengths are supported.
    """
    columns = tuple(columns)
    if len(columns) != len(set(columns)) or set(columns) != set(STATE_COLUMNS):
        raise ValueError("Expected the four Allen length states")
    values = np.asarray(log_lengths, dtype=float)
    if values.ndim != 2 or not len(values) or values.shape[1] != 4 or np.isinf(values).any():
        raise ValueError("Expected nonempty N x 4 log lengths; NaN is allowed, infinity is not")
    v = values[:, [columns.index(c) for c in STATE_COLUMNS]]
    bones = v[:, 1:]
    complete = np.isfinite(bones).all(axis=1)
    safe = np.where(complete[:, None], bones, 0.)
    # Shift before multiplication to avoid avoidable overflow at large offsets.
    anchor = np.max(safe, axis=1)
    try:
        with np.errstate(over="raise", invalid="raise"):
            centered = safe - anchor[:, None]
            log_sum = np.log10(np.exp(centered * np.log(10.)).sum(axis=1))
            total = anchor + log_sum
            shares = centered - log_sum[:, None]
            derived = np.column_stack([total, shares, safe[:, 0] - safe[:, 2], safe[:, 1] - safe[:, 2]])
    except FloatingPointError as exc:
        raise ValueError("Anatomical coordinate overflow; inspect extreme log lengths") from exc
    derived[~complete] = np.nan
    return np.column_stack([v[:, 0], derived[:, 0], v[:, 1:], derived[:, 1:]])


def percent_change_from_log(delta_log10):
    """Convert a log10 difference to relative percent; preserve missing NaNs."""
    values = np.asarray(delta_log10, dtype=float)
    if np.isinf(values).any():
        raise ValueError("Infinite log response difference")
    try:
        with np.errstate(over="raise", invalid="raise"):
            result = 100 * np.expm1(np.log(10.) * values)
    except FloatingPointError as exc:
        raise ValueError("Response percentage overflow; inspect extreme model predictions") from exc
    return result


def _mask(mask, n, name):
    if mask is None:
        return np.ones(n, dtype=bool)
    mask = np.asarray(mask)
    if mask.dtype.kind != "b" or mask.shape != (n,):
        raise ValueError(name + " must be a boolean mask with one entry per window")
    return mask


def summarize_corner_effects(predictions, *, columns=STATE_COLUMNS, temperature_step=1.,
                             ecology_feasible=None, temperature_feasible=None):
    """Return result[comparison][metric] arrays N x len(OUTCOME_NAMES).

    Input predictions maps all six SCENARIO_NAMES to paired N x 4 log lengths.
    Rows must refer to the same windows in the same order. The caller ensures
    identity and training-support checks; masks here enforce feasibility only.
    Comparison temperature_by_ecology is a difference of log response factors:
      (warm_shifted - warm) - (cold_shifted - cold).
    It has response_factor_change_percent and interaction_per10C_log10, NOT a
    percent_change or share_pp. No metric is a causal or evolutionary share.
    """
    if set(predictions) != set(SCENARIO_NAMES):
        raise ValueError("Expected all six paired scenario predictions")
    step = _number(temperature_step, "temperature_step", positive=True)
    coords = {name: outcome_coordinates(predictions[name], columns) for name in SCENARIO_NAMES}
    shape = coords["observed"].shape
    if any(v.shape != shape for v in coords.values()):
        raise ValueError("All scenarios must use the same windows and outcomes")
    eco = _mask(ecology_feasible, shape[0], "ecology_feasible")
    temp = _mask(temperature_feasible, shape[0], "temperature_feasible")
    pairs = {"observed_ecology": ("observed", "shifted", eco),
             "cold_ecology": ("cold", "cold_shifted", eco & temp),
             "warm_ecology": ("warm", "warm_shifted", eco & temp),
             "temperature_original": ("cold", "warm", temp),
             "temperature_shifted": ("cold_shifted", "warm_shifted", eco & temp)}
    result = {}
    for name, (before, after, feasible) in pairs.items():
        row = {m: np.full(shape, np.nan) for m in METRIC_NAMES}
        delta = coords[after] - coords[before]
        delta[~feasible] = np.nan
        row["delta_log10"] = delta
        row["percent_change"] = percent_change_from_log(delta)
        for index in SHARE_INDICES:
            with np.errstate(over="raise", invalid="raise"):
                pp = 100 * (np.exp(np.log(10.) * coords[after][:, index])
                            - np.exp(np.log(10.) * coords[before][:, index]))
            pp[~feasible] = np.nan
            row["share_pp"][:, index] = pp
        result[name] = row
    interaction = result["warm_ecology"]["delta_log10"] - result["cold_ecology"]["delta_log10"]
    row = {m: np.full(shape, np.nan) for m in METRIC_NAMES}
    row["delta_log10"] = interaction
    row["response_factor_change_percent"] = percent_change_from_log(interaction)
    row["interaction_per10C_log10"] = interaction / (2 * step) * 10
    result["temperature_by_ecology"] = row
    return result
