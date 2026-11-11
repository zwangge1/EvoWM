"""Read-only inputs for checkpoint-free phylogenetic comparators of Allen v3.

Saved reference identities are the authority. The frozen single tree supplies
covariances; a NumPy replay additionally checks every saved reference window.
Neither checkpoints nor the PyTorch/Hydra training environment are required.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/allen_world"))
sys.path.insert(0, str(ROOT / "src"))
from diagnose_v3 import contained, inspect_source, sha
from evowm.data.tree import Tree, leaf_order, pair_distance, read_trees, restrict

STATE_COLUMNS = ("m_tarsus", "b_fem", "b_tbt", "b_tmt")
TEMPERATURE_COLUMNS = ("bio1_c", "bio5_c", "bio6_c")
FORAGING_COLUMNS = tuple("e_fs_" + s for s in
    ("watbelowsurf", "wataroundsurf", "ground", "understory", "midhigh", "canopy", "aerial"))
MOTION_COLUMNS = tuple("e_life_" + s for s in
    ("aerial", "aquatic", "generalist", "insessorial", "terrestrial"))
DEVELOPMENT_COLUMNS = tuple("e_dev_" + s for s in ("P", "SP", "SA", "A"))
CONDITION_COLUMNS = (TEMPERATURE_COLUMNS + FORAGING_COLUMNS + MOTION_COLUMNS
    + DEVELOPMENT_COLUMNS + ("m_mass", "e_habdens", "e_migr"))
REQUIRE_COLUMNS = ("m_tarsus", "m_mass", "e_dev_A", *TEMPERATURE_COLUMNS)
FOLD_COLUMN = "allen_frozen_family_fold"


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def identity_columns(history):
    return ["window", "target_row", "source_row", "target_tip", "source_tip", "family",
            *[f"context_{i}_tip" for i in range(history)]]


def _saved_columns(history):
    return identity_columns(history) + ["complete_three_bones", *["observed_" + c for c in STATE_COLUMNS]]


def reconstruct_saved_rows(table, saved, history, validation_mask):
    """Resolve all H reference tips, checking saved rows, split and target exclusion."""
    if table.tip.isna().any() or table.tip.duplicated().any():
        raise ValueError("Species tips must be unique and nonmissing")
    index = pd.Index(table.tip)
    reference = np.column_stack([index.get_indexer(saved[f"context_{i}_tip"]) for i in range(history)])
    target = index.get_indexer(saved.target_tip)
    if (reference < 0).any() or (target < 0).any():
        raise ValueError("A saved target/reference tip is absent from the frozen table")
    if not np.array_equal(saved.window.to_numpy(), np.arange(len(saved))):
        raise ValueError("Saved windows must retain their original contiguous order")
    if not np.array_equal(target, saved.target_row.to_numpy()):
        raise ValueError("Saved target row and tip disagree")
    if (not np.array_equal(reference[:, -1], saved.source_row.to_numpy())
            or not np.array_equal(table.tip.to_numpy()[reference[:, -1]], saved.source_tip.to_numpy())):
        raise ValueError("Saved source must be the final reference")
    if not np.array_equal(table.family.to_numpy()[target], saved.family.to_numpy()):
        raise ValueError("Saved target family differs from the frozen table")
    if not validation_mask[target].all() or not validation_mask[reference].all():
        raise ValueError("Saved references and target must belong to the held-out fold")
    if (reference == target[:, None]).any():
        raise ValueError("Target morphology would leak through a reference")
    if any(len(set(row)) != history for row in reference):
        raise ValueError("A reference window repeats a species")
    return reference, target


def replay_windows(tree, row_of_tip, rows_mask, history, n_orders, seed):
    """Exact original _order/sample_windows RNG use, retaining tree identity 0."""
    rng = np.random.default_rng(seed)
    rows, distances, order_ids = [], [], []
    for order_id in range(n_orders):
        # Keep even this one-tree draw: it is part of the original algorithm.
        rng.integers(1)
        tips, lca = leaf_order(tree, rng)
        tips, lca = restrict(tips, lca, rows_mask[row_of_tip])
        order = row_of_tip[tips]
        dist = pair_distance(tree, tips, lca).astype(np.float32)
        count = len(order) - history
        if count <= 0:
            raise ValueError("Too few held-out tips for the saved reference history")
        idx = np.arange(count)[:, None] + np.arange(history + 1)[None, :]
        rows.append(order[idx])
        distances.append(dist[idx[:, :-1]])
        order_ids.append(np.full(count, order_id, dtype=np.int64))
    return np.concatenate(rows), np.concatenate(distances), np.concatenate(order_ids)


def _frozen_masks(table, metadata, dataset):
    holdout = dataset["holdout"]
    if (holdout.get("level") != "family" or holdout.get("folds") != 5
            or holdout.get("seed") != 0 or holdout.get("frac", 0) != 0
            or dataset.get("frozen_fold_column") != FOLD_COLUMN):
        raise ValueError("Expected the original frozen five family folds (seed=0)")
    if tuple(dataset.get("require_cols", ())) != REQUIRE_COLUMNS:
        raise ValueError("Original v3 eligibility columns changed")
    declaration = metadata["frozen_split"]
    if (declaration.get("level") != "family" or declaration.get("folds") != 5
            or declaration.get("seed") != 0
            or tuple(declaration.get("require_cols", ())) != REQUIRE_COLUMNS):
        raise ValueError("Frozen split metadata differs from saved eligibility")
    eligible = table[list(REQUIRE_COLUMNS)].notna().all(axis=1).to_numpy()
    groups = set(table.loc[eligible, "family"])
    mapping = declaration["family_to_fold"]
    if not groups <= set(mapping):
        raise ValueError("Eligible family missing from the recorded fold mapping")
    for group in groups:
        value = mapping[group]
        if type(value) is not int or value not in range(5):
            raise ValueError("Invalid recorded family fold")
    selected = table.family.isin(groups)
    recorded = table.loc[selected, FOLD_COLUMN].to_numpy()
    expected = table.loc[selected, "family"].map(mapping).to_numpy()
    if not np.array_equal(recorded, expected):
        raise ValueError("Frozen fold column and family mapping disagree")
    fold = int(holdout["fold"])
    if fold not in range(5):
        raise ValueError("Fold must be 0..4")
    val = eligible & table[FOLD_COLUMN].eq(fold).to_numpy()
    train = eligible & ~table[FOLD_COLUMN].eq(fold).to_numpy()
    if not train.any() or not val.any():
        raise ValueError("Both frozen training and validation splits must be nonempty")
    return train, val


def _rank_columns(values, train_rows, support_rows, names, tolerance):
    """Keep one ordered basis estimable both on training and bone support."""
    train = values[train_rows]
    support = values[support_rows]
    basis, support_basis, kept, dropped = [], [], [], []
    for j, name in enumerate(names):
        vector = train[:, j].copy()
        support_vector = support[:, j].copy()
        original_norm = float(np.linalg.norm(vector))
        support_norm = float(np.linalg.norm(support_vector))
        for _ in range(2):
            for q in basis:
                vector -= q * np.dot(q, vector)
            for q in support_basis:
                support_vector -= q * np.dot(q, support_vector)
        residual_norm = float(np.linalg.norm(vector))
        support_residual = float(np.linalg.norm(support_vector))
        threshold = tolerance * max(original_norm, np.sqrt(len(train)))
        support_threshold = tolerance * max(support_norm, np.sqrt(len(support)))
        if residual_norm <= threshold:
            dropped.append({"column": name, "reason": "zero_on_train" if original_norm == 0 else "train_linear_dependency",
                            "residual_norm": residual_norm, "threshold": float(threshold)})
        elif support_residual <= support_threshold:
            dropped.append({"column": name,
                            "reason": "zero_on_bone_support" if support_norm == 0 else "bone_support_linear_dependency",
                            "residual_norm": support_residual, "threshold": float(support_threshold)})
        else:
            kept.append(j)
            basis.append(vector / residual_norm)
            support_basis.append(support_vector / support_residual)
    return kept, dropped


def build_condition_design(table, train_rows, condition_columns=CONDITION_COLUMNS,
                           mode="interactions", rank_tolerance=1e-10, rank_support_rows=None):
    """Train-standardized conditions, missing indicators, and prespecified products.

    The same design/rank selection is used for all outcomes. Source foraging
    fractions (including .99 totals) remain untouched; no composite X is used.
    """
    columns = list(condition_columns)
    if set(columns) & set(STATE_COLUMNS):
        raise ValueError("Target morphology cannot enter a condition design")
    if tuple(columns) != CONDITION_COLUMNS:
        raise ValueError("Expected the exact 22 original v3 condition columns and order")
    if mode not in ("interactions", "main"):
        raise ValueError("Design mode must be interactions or main")
    train_rows = np.asarray(train_rows)
    if train_rows.dtype == bool:
        train_rows = np.flatnonzero(train_rows)
    if (train_rows.ndim != 1 or len(train_rows) < 2 or not np.issubdtype(train_rows.dtype, np.integer)
            or len(np.unique(train_rows)) != len(train_rows)
            or np.any(train_rows < 0) or np.any(train_rows >= len(table))):
        raise ValueError("Expected distinct valid training row indices")
    if not np.isfinite(rank_tolerance) or not 0 < rank_tolerance < 1:
        raise ValueError("Rank tolerance must be between zero and one")
    support_rows = train_rows if rank_support_rows is None else np.asarray(rank_support_rows)
    if support_rows.dtype == bool:
        support_rows = np.flatnonzero(support_rows)
    if (support_rows.ndim != 1 or len(support_rows) < 2
            or not np.issubdtype(support_rows.dtype, np.integer)
            or len(np.unique(support_rows)) != len(support_rows)
            or not np.isin(support_rows, train_rows).all()):
        raise ValueError("Rank support must be distinct training rows with at least two observations")
    raw = table[columns].to_numpy(float)
    finite = np.isfinite(raw)
    if not finite[train_rows].any(axis=0).all():
        raise ValueError("A condition has no finite training observation")
    observed_train = np.where(finite[train_rows], raw[train_rows], np.nan)
    mu, sd = np.nanmean(observed_train, axis=0), np.nanstd(observed_train, axis=0)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.
    z = np.where(finite, (raw - mu) / sd, 0.)
    parts = [np.ones((len(table), 1)), z, (~finite).astype(float)]
    names = ["intercept", *columns, *["missing_" + c for c in columns]]
    pairs = []
    if mode == "interactions":
        partners = ("m_mass", *FORAGING_COLUMNS, *DEVELOPMENT_COLUMNS, *MOTION_COLUMNS)
        for temp in TEMPERATURE_COLUMNS:
            for partner in partners:
                parts.append((z[:, columns.index(temp)] * z[:, columns.index(partner)])[:, None])
                names.append(temp + ":" + partner)
                pairs.append([temp, partner])
    candidates = np.column_stack(parts)
    if not np.isfinite(candidates).all():
        raise ValueError("Condition design contains nonfinite values")
    kept, dropped = _rank_columns(candidates, train_rows, support_rows, names, rank_tolerance)
    info = {"mode": mode, "condition_columns": columns,
            "columns": [names[j] for j in kept], "candidate_columns": names,
            "kept_indices": kept, "dropped_columns": dropped,
            "n_candidates": len(names), "rank": len(kept), "n_train_species": len(train_rows),
            "n_rank_support_species": len(support_rows),
            "scaling": {c: {"mean": float(mu[j]), "sd": float(sd[j]),
                            "missing_train": int((~finite[train_rows, j]).sum())} for j, c in enumerate(columns)},
            "interaction_pairs": pairs, "rank_tolerance": rank_tolerance,
            "condition_support": {c: {"train_min": float(np.nanmin(observed_train[:, j])),
                "train_max": float(np.nanmax(observed_train[:, j])),
                "train_observed": int(finite[train_rows, j].sum()),
                "bone_observed": int(finite[support_rows, j].sum()),
                "bone_min": float(raw[support_rows, j][finite[support_rows, j]].min()) if finite[support_rows, j].any() else None,
                "bone_max": float(raw[support_rows, j][finite[support_rows, j]].max()) if finite[support_rows, j].any() else None}
                for j, c in enumerate(columns)},
            "rank_rule": "Prespecified column order; twice-reorthogonalized Gram-Schmidt on eligible training species and training complete-bone support. Both must retain rank; support uses observation masks, never outcome values.",
            "missing_rule": "Missing/nonfinite condition is train-mean imputed (z=0), with a separate missing flag; products multiply imputed standardized main effects.",
            "foraging_rule": "Seven original fractions unchanged, including .99 totals; no closure and no X aggregation.",
            "scaling_scope": "Training species only; validation rows and outcomes do not select columns or scaling."}
    return candidates[:, kept], info


@dataclass
class SourceRun:
    root: Path
    path: Path
    manifest: dict
    jobs: list
    audit: dict
    table: pd.DataFrame
    metadata: dict
    tree: Tree
    tree_row_indices: np.ndarray
    states: np.ndarray
    results: dict
    canonical_windows: dict
    evaluation_paths: dict


@dataclass
class FoldData:
    fold: int
    table: pd.DataFrame
    tree: Tree
    tip_ids_of_rows: np.ndarray
    states: np.ndarray
    train_rows: np.ndarray
    validation_rows: np.ndarray
    reference_rows: np.ndarray
    target_rows: np.ndarray
    tree_ids: np.ndarray
    order_ids: np.ndarray
    distances: np.ndarray
    X: np.ndarray
    design_info: dict
    source_refs: dict
    saved_results: dict
    saved_prediction_paths: dict
    saved_windows: pd.DataFrame

    @property
    def ref_rows(self):
        return self.reference_rows

    def saved_predictions(self, job=None):
        """Original predictions for one variant/seed; no inference or refitting."""
        job = job or self.source_refs["canonical_job"]
        return pd.read_csv(self.saved_prediction_paths[job])


def load_source_run(root=ROOT, run=None):
    """Verify all frozen inputs/snapshots and pair the full 50 real-data jobs."""
    if run is None:
        raise ValueError("An explicit completed source run is required")
    root = Path(root).resolve()
    run, manifest, jobs, audit = inspect_source(run, "full", require_checkpoints=False, root=root)
    completion = _read_json(run / "completion.json")
    if (completion.get("planned") != len(manifest["jobs"])
            or completion.get("completed") != len(manifest["jobs"])):
        raise ValueError("Completion counts do not match the full source manifest")
    results = {j["name"]: _read_json(run / "evaluation" / j["name"] / "pilot_results.json") for j in jobs}
    canonical_name = "real_development_observed_full_s13_f0"
    first = results[canonical_name]
    dataset = first["provenance"]["resolved_config"]["data"]["dataset"]
    table_path = contained(root / dataset["species_table"], run)
    tree_path = contained(root / dataset["tree_file"], run)
    metadata_path = table_path.with_suffix(".columns.json")
    table, metadata = pd.read_csv(table_path), _read_json(metadata_path)
    if table.tip.isna().any() or table.tip.duplicated().any():
        raise ValueError("Frozen table tips must be unique and nonmissing")
    condition_columns = tuple(c for group in metadata["eco_groups"].values() for c in group)
    if condition_columns != CONDITION_COLUMNS or tuple(metadata["state_default"]) != STATE_COLUMNS:
        raise ValueError("Frozen state or condition schema changed")
    trees = read_trees(tree_path)
    if len(trees) != 1 or dataset["n_trees"] != 1:
        raise ValueError("Missing saved tree IDs can only be resolved for the original single-tree run")
    tree = trees[0]
    tree_rows = pd.Index(table.tip).get_indexer(tree.tip_names)
    if (tree_rows < 0).any() or len(np.unique(tree_rows)) != len(tree_rows):
        raise ValueError("Frozen tree tips do not map uniquely into the input table")
    states = table[list(STATE_COLUMNS)].to_numpy(np.float32).astype(float)
    hashes = {"table_sha256": sha(table_path), "table_metadata_sha256": sha(metadata_path), "tree_sha256": sha(tree_path)}
    canonical, paths, evaluation_hashes = {}, {}, {}
    for fold in range(5):
        name = f"real_development_observed_full_s13_f{fold}"
        old = results[name]
        if (int(old["history"]), int(old["n_orders"]), int(old["seed"])) != (3, 2, 0):
            raise ValueError("Expected original H=3, n_orders=2, evaluation seed=0")
        saved_path = run / "evaluation" / name / "prediction_windows.csv.gz"
        header = pd.read_csv(saved_path, nrows=0).columns
        verification_columns = identity_columns(3) + ["complete_three_bones"] + [c for c in header if c.startswith("observed_")]
        if not set(_saved_columns(3)) <= set(verification_columns):
            raise ValueError("Original saved observation columns are incomplete")
        saved = pd.read_csv(saved_path, usecols=verification_columns)
        train, val = _frozen_masks(table, metadata, old["provenance"]["resolved_config"]["data"]["dataset"])
        refs, target = reconstruct_saved_rows(table, saved, 3, val)
        replayed, distances, order_ids = replay_windows(tree, tree_rows, val, 3, 2, 1)
        if not np.array_equal(replayed, np.column_stack([refs, target])):
            raise ValueError("Saved reference windows differ from exact frozen-tree RNG replay")
        np.testing.assert_allclose(saved[["observed_" + c for c in STATE_COLUMNS]], states[target], atol=1e-12, rtol=0, equal_nan=True)
        if not np.array_equal(saved.complete_three_bones.to_numpy(bool), np.isfinite(states[target, 1:]).all(axis=1)):
            raise ValueError("Saved complete-bone support changed")
        canonical[fold] = {"saved": saved, "reference_rows": refs, "target_rows": target,
                           "train_mask": train, "val_mask": val, "distances": distances,
                           "order_ids": order_ids, "verification_columns": verification_columns}
    for job in jobs:
        name, fold = job["name"], int(job["fold"])
        old = results[name]
        prov, saved_cfg = old["provenance"], old["provenance"]["resolved_config"]
        ds = saved_cfg["data"]["dataset"]
        if (int(old["history"]), int(old["n_orders"]), int(old["seed"])) != (3, 2, 0):
            raise ValueError("Evaluation windows were not shared across all jobs")
        if (saved_cfg["seed"] != job["seed"] or ds["holdout"]["fold"] != fold
                or ds["species_table"] != dataset["species_table"] or ds["tree_file"] != dataset["tree_file"]
                or ds["n_trees"] != 1):
            raise ValueError("Job identity and saved data configuration disagree: " + name)
        config = run / "training/ckpts" / name / "config.yaml"
        if not config.is_file() or sha(config) != prov["config_sha256"]:
            raise ValueError("Original training configuration missing or changed: " + name)
        if any(prov.get(key) != value for key, value in hashes.items()):
            raise ValueError("Saved evaluation input hashes disagree: " + name)
        train, val = _frozen_masks(table, metadata, ds)
        if not np.array_equal(train, canonical[fold]["train_mask"]) or not np.array_equal(val, canonical[fold]["val_mask"]):
            raise ValueError("Job split differs from canonical fold: " + name)
        path = run / "evaluation" / name / "prediction_windows.csv.gz"
        check_columns = canonical[fold]["verification_columns"]
        saved = pd.read_csv(path, usecols=check_columns)
        pd.testing.assert_frame_equal(saved[check_columns], canonical[fold]["saved"][check_columns],
                                      check_dtype=False, check_exact=True)
        paths[name] = path
        evaluation_hashes[name] = {"prediction_windows_sha256": sha(path),
                                  "pilot_results_sha256": sha(path.with_name("pilot_results.json")),
                                  "config_sha256": prov["config_sha256"]}
    audit = dict(audit, real_jobs_paired=len(jobs), frozen_tree_count=1,
                 tree_identity="tree_id=0, uniquely identified by frozen file SHA256",
                 evaluation_seed=0, validation_window_seed=1, training_window_seed=0,
                 training_seeds=[13, 14], seed_interpretation="13/14 are training random seeds; all jobs used evaluation seed 0.",
                 canonical_replay_verified_folds=list(range(5)), state_cast="Original dataset float32 log10 measurements, promoted to float64; no second log transform.",
                 observation_pairing="All saved observed_* values and masks, complete-bone flags, and H-reference/target/source identities match within each fold across all 10 jobs.",
                 evaluation_files_sha256=evaluation_hashes, checkpoint_required=False, **hashes)
    return SourceRun(root, run, manifest, jobs, audit, table, metadata, tree, tree_rows,
                     states, results, canonical, paths)


def load_fold(source, fold, design="interactions"):
    """Return full-row conditions and original held-out H-reference windows."""
    if fold not in range(5):
        raise ValueError("Fold must be 0..4")
    canonical = source.canonical_windows[fold]
    train_rows = np.flatnonzero(canonical["train_mask"])
    support_rows = train_rows[np.isfinite(source.states[train_rows, 1:]).all(axis=1)]
    X, info = build_condition_design(source.table, train_rows, mode=design, rank_support_rows=support_rows)
    name = f"real_development_observed_full_s13_f{fold}"
    result = source.results[name]
    # Cross-check the reconstruction's train-only condition scaling against the
    # exact audit emitted by the original evaluator.
    for key, field in (("condition_mu", "mean"), ("condition_sd", "sd")):
        actual = [info["scaling"][c][field] for c in CONDITION_COLUMNS]
        np.testing.assert_allclose(actual, result["train_audit"][key], atol=1e-12, rtol=0)
    names = [j["name"] for j in source.jobs if j["fold"] == fold]
    refs = dict(source.audit, canonical_job=name, source_run=str(source.path),
                source_manifest=str(source.path / "manifest.json"),
                canonical_results=str(source.path / "evaluation" / name / "pilot_results.json"),
                canonical_windows=str(source.evaluation_paths[name]), fold=int(fold),
                history=3, n_orders=2, n_windows=len(canonical["target_rows"]))
    tip_ids = np.full(len(source.table), -1, dtype=np.int64)
    tip_ids[source.tree_row_indices] = np.arange(source.tree.n_tips)
    if (tip_ids[np.r_[train_rows, canonical["target_rows"], canonical["reference_rows"].ravel()]] < 0).any():
        raise ValueError("A required species is missing from the frozen tree")
    return FoldData(int(fold), source.table, source.tree, tip_ids, source.states, train_rows,
                    np.flatnonzero(canonical["val_mask"]), canonical["reference_rows"],
                    canonical["target_rows"], np.zeros(len(canonical["target_rows"]), dtype=np.int64),
                    canonical["order_ids"], canonical["distances"], X, info, refs,
                    {n: source.results[n] for n in names}, {n: source.evaluation_paths[n] for n in names},
                    pd.read_csv(source.evaluation_paths[name]))
