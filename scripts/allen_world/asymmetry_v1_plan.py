"""Prespecified Allen label-budget plans and immutable, role-safe inputs.

Only training-species skeletal measurements are removed.  Held-out references
and targets retain their original table values; reference-only missingness is
applied separately at prediction time.  No split, model, or outcome-driven
selection is fitted here.  NumPy/Pandas are sufficient; no torch/Hydra import.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

CONDITIONS = ("complete100", "uniform50", "uniform25", "family50", "family25")
VARIANTS = ("full", "single", "supervised")
REF_CASES = ("observed", "hide_half_bones", "hide_all_bones", "hide_distal")
REPLICATES = [{"replicate": 0, "mask_seed": 101, "seed": 13},
              {"replicate": 1, "mask_seed": 202, "seed": 14}]
STATE_COLUMNS = ("m_tarsus", "b_fem", "b_tbt", "b_tmt")
BONE_COLUMNS = STATE_COLUMNS[1:]
FOLD_COLUMN = "allen_frozen_family_fold"
REQUIRE_COLUMNS = ("m_tarsus", "m_mass", "e_dev_A", "bio1_c", "bio5_c", "bio6_c")
HASH_PROTOCOL = "SHA256 of compact UTF-8 JSON [allen-asymmetry-v1, scope, mask_seed, fold, identity]; ascending digest then identity."


def _cells(mode):
    if mode not in ("full", "smoke"):
        raise ValueError("mode must be full or smoke")
    folds = range(5) if mode == "full" else (0,)
    replicates = REPLICATES if mode == "full" else REPLICATES[:1]
    conditions = CONDITIONS if mode == "full" else ("complete100", "uniform25", "family25")
    return [dict(fold=fold, **rep, condition=condition,
                 input_key=f"f{fold}_r{rep['replicate']}_{condition}")
            for fold in folds for rep in replicates for condition in conditions]


def make_plan(mode="full"):
    """Full: 150 neural + 50 classic fits; smoke: 9 neural + 3 classic fits."""
    cells, jobs = _cells(mode), []
    for cell in cells:
        for variant in (*VARIANTS, "classic"):
            kind = "classic" if variant == "classic" else "neural"
            jobs.append(dict(cell, name=f"{variant}_{cell['input_key']}", kind=kind,
                             variant=variant, config="allen_v3_" + variant,
                             steps=(64 if mode == "smoke" else 4000) if kind == "neural" else 0,
                             overrides=[]))
    return dict(mode=mode, jobs=jobs, conditions=list(CONDITIONS), variants=list(VARIANTS),
                reference_cases=list(REF_CASES), replicates=copy.deepcopy(REPLICATES),
                counts=dict(inputs=len(cells), neural=len(cells) * len(VARIANTS),
                            classic=len(cells), total=len(jobs)),
                budget_rule="floor(original complete-three-bone eligible training species * retained percentage / 100)",
                seed_interpretation="mask_seed controls deterministic species/family deletion; seed is the training seed; original held-out evaluation windows are frozen.",
                selection_rule="No model, budget, architecture, reference case, or endpoint is selected using held-out performance.")


def stable_hash(scope, mask_seed, fold, identity):
    """Portable ordering key; independent of Python hash and table/window order."""
    payload = json.dumps(["allen-asymmetry-v1", str(scope), int(mask_seed), int(fold), str(identity)],
                         ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sorted_identities(identities, scope, mask_seed, fold):
    return sorted(identities, key=lambda value: (stable_hash(scope, mask_seed, fold, value), str(value)))


def choose_train_bone_rows(table, train_rows, condition, mask_seed, fold):
    """Whole-triplet budgets depend only on training identity/availability.

    Family deletion takes a prefix of hash-ordered families and, only when
    needed, a prefix of one boundary family's hash-ordered species.  The 25%
    sets are nested in the 50% sets within each mechanism and replicate.
    """
    if condition not in CONDITIONS:
        raise ValueError("Unknown condition")
    rows = np.asarray(train_rows)
    if (rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer)
            or len(np.unique(rows)) != len(rows) or np.any(rows < 0) or np.any(rows >= len(table))):
        raise ValueError("train_rows must contain distinct valid integer indices")
    observed = np.isfinite(table[list(BONE_COLUMNS)].to_numpy(float))[rows]
    if np.any(observed.any(axis=1) != observed.all(axis=1)):
        raise ValueError("Original training bones must have whole-triplet observation support")
    support = rows[observed.all(axis=1)]
    if len(support) == 0:
        raise ValueError("No complete training bones")
    tips = table.tip.to_numpy()
    if table.tip.isna().any() or table.tip.duplicated().any():
        raise ValueError("Species tips must be unique and nonmissing")
    families = table.family.to_numpy()
    if pd.isna(families[support]).any():
        raise ValueError("Training bone families must be recorded")
    row_of_tip = {str(tips[row]): int(row) for row in support}
    tip_order = _sorted_identities(row_of_tip, "train-tip", mask_seed, fold)
    if condition.startswith("family"):
        family_order = _sorted_identities(set(map(str, families[support])), "train-family", mask_seed, fold)
        ordered = [row_of_tip[tip] for family in family_order for tip in tip_order
                   if str(families[row_of_tip[tip]]) == family]
    else:
        family_order = []
        ordered = [row_of_tip[tip] for tip in tip_order]
    fraction = 100 if condition == "complete100" else int(condition[-2:])
    keep_count = len(support) * fraction // 100
    kept, hidden = np.sort(ordered[:keep_count]), np.sort(ordered[keep_count:])
    kept, hidden = kept.astype(np.int64), hidden.astype(np.int64)
    family_counts = []
    for family in sorted(set(map(str, families[support]))):
        before = int(sum(str(families[row]) == family for row in support))
        after = int(sum(str(families[row]) == family for row in kept))
        family_counts.append(dict(family=family, original=before, retained=after, hidden=before - after))
    boundary = [item["family"] for item in family_counts if 0 < item["retained"] < item["original"]]
    if condition.startswith("family") and len(boundary) > 1:
        raise AssertionError("Family budget may split at most one boundary family")
    return dict(kept=kept, hidden=hidden, ordered=np.asarray(ordered, dtype=np.int64),
                audit=dict(original_train_bone_species=len(support), kept_train_bone_species=len(kept),
                           hidden_train_bone_species=len(hidden), retained_percent=fraction,
                           retained_fraction=len(kept) / len(support), family_order=family_order,
                           partial_families=boundary, family_counts=family_counts,
                           original_train_bone_families=len(family_counts),
                           retained_train_bone_families=sum(item["retained"] > 0 for item in family_counts),
                           hash_protocol=HASH_PROTOCOL,
                           selection_inputs="Training tip/family identities and original finite bone mask only; never measurement values, ecology or held-out outcomes."))


def bone_related_columns(table, metadata):
    """Identify canonical, legacy, count/provenance and named derived bone fields.

    The original frozen source is additionally hash-verified by load_source_run.
    This scan includes fields excluded from current model metadata, so a hidden
    row does not retain a legacy skeleton or a measurement-derived proxy.
    """
    declared = set(metadata.get("skeleton", ())) | set(BONE_COLUMNS)
    found = []
    for column in table.columns:
        name = str(column).lower()
        related = (column in declared or name.startswith(("b_", "k_", "n_eliason_"))
                   or any(token in name for token in ("femur", "tibiotars", "tarsometatars", "skeleton",
                       "bone", "anatomical", "allocation", "logratio", "log_ratio", "total_leg", "leg_total"))
                   or re.search(r"(?:^|_)(?:fem|tbt|tmt)(?:_|$)", name) is not None)
        if related and column != "m_tarsus":
            found.append(column)
    forbidden = set(metadata.get("eco", ())) | {FOLD_COLUMN, "tip", "family", "m_tarsus"}
    forbidden |= {column for group in metadata.get("eco_groups", {}).values() for column in group}
    if set(found) & forbidden:
        raise ValueError("A bone-related field overlaps a protected identity/ecology field")
    return found


def make_reference_states(states, reference_rows, tips, case, mask_seed, fold):
    """Return reference states [window,H,4], with missing coordinates set NaN.

    Callers derive both values and masks from this result: value=nan_to_num,
    mask=isfinite.  In particular, they must not reuse an unmasked pixel tensor.
    Half deletion hides floor(N/2) of originally complete-bone species in the
    unique reference pool, using the same selection in every repeated window.
    Target states are never modified or used to choose reference deletions.
    """
    if case not in REF_CASES:
        raise ValueError("Unknown reference case")
    states, rows, tips = np.asarray(states), np.asarray(reference_rows), np.asarray(tips)
    if states.ndim != 2 or states.shape != (len(tips), 4):
        raise ValueError("Expected one four-trait state per tip")
    if (rows.ndim != 2 or not np.issubdtype(rows.dtype, np.integer)
            or np.any(rows < 0) or np.any(rows >= len(states))):
        raise ValueError("Invalid reference rows")
    if pd.isna(tips).any() or len(set(map(str, tips))) != len(tips):
        raise ValueError("Species tips must be unique and nonmissing")
    result = np.array(states[rows], dtype=float, copy=True)
    if case == "hide_all_bones":
        result[:, :, 1:] = np.nan
    elif case == "hide_distal":
        result[:, :, 3] = np.nan
    elif case == "hide_half_bones":
        pool = np.unique(rows)
        pool = pool[np.isfinite(states[pool, 1:]).all(axis=1)]
        row_of_tip = {str(tips[row]): int(row) for row in pool}
        ordered = _sorted_identities(row_of_tip, "reference-tip", mask_seed, fold)
        hidden = [row_of_tip[tip] for tip in ordered[:len(ordered) // 2]]
        result[:, :, 1:][np.isin(rows, hidden)] = np.nan
    return result


def _sha_bytes(content):
    return hashlib.sha256(content).hexdigest()


def _json_bytes(value):
    # eco_groups insertion order defines the action tensor schema.  Recursive
    # key sorting would silently change column order and invalidate AllenDataset.
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _relative(path, root):
    path = Path(path).resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("Frozen input path must remain within root") from exc


def _freeze(path, content):
    """An existing frozen artifact is reusable only if its bytes match."""
    path = Path(path)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError("Refusing to replace different frozen input: " + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return _sha_bytes(content)


def _masked_csv_bytes(original, columns, hidden, mask_columns):
    if not len(hidden):
        return original
    rows = list(csv.reader(io.StringIO(original.decode("utf-8-sig"), newline="")))
    if rows[0] != list(columns):
        raise ValueError("Source CSV header differs from loaded source table")
    indices = [rows[0].index(column) for column in mask_columns]
    for row in hidden:
        for column in indices:
            rows[int(row) + 1][column] = ""
    stream = io.StringIO(newline="")
    csv.writer(stream, lineterminator="\n").writerows(rows)
    return stream.getvalue().encode("utf-8")


def prepare_inputs(source, destination, root, mode="full"):
    """Freeze per-cell tables, exact saved windows, masks, hashes and audit.

    ``source`` is the validated SourceRun returned by load_source_run.  Every
    returned artifact path is root-relative POSIX.  The npz has only numerical
    arrays and is safe to load with allow_pickle=False.  It contains no old model
    predictions or teacher latents.  Training must construct normalization,
    reference pixels, teacher pixels, labels and derived anatomy from *table*.
    """
    cells = _cells(mode)
    root, destination = Path(root).resolve(), Path(destination).resolve()
    _relative(destination, root)
    source_path = Path(source.path).resolve()
    if destination == source_path or source_path in destination.parents:
        raise ValueError("Destination cannot modify the original source run")
    table, metadata = source.table, source.metadata
    if not isinstance(table.index, pd.RangeIndex) or list(table.index) != list(range(len(table))):
        raise ValueError("Source row positions must equal the frozen CSV row indices")
    if tuple(metadata.get("state_default", ())) != STATE_COLUMNS:
        raise ValueError("Original four-trait state schema is required")
    first = source.results["real_development_observed_full_s13_f0"]
    dataset = first["provenance"]["resolved_config"]["data"]["dataset"]
    table_path, tree_path = (root / dataset[key] for key in ("species_table", "tree_file"))
    metadata_path = table_path.with_suffix(".columns.json")
    original = table_path.read_bytes()
    tree_bytes, metadata_bytes = tree_path.read_bytes(), metadata_path.read_bytes()
    originals = {"table_sha256": _sha_bytes(original), "tree_sha256": _sha_bytes(tree_bytes),
                 "table_metadata_sha256": _sha_bytes(metadata_bytes)}
    for key, digest in originals.items():
        if source.audit.get(key) != digest:
            raise ValueError("Source artifact hash differs from the validated SourceRun: " + key)
    pd.testing.assert_frame_equal(pd.read_csv(io.BytesIO(original)), table, check_exact=True)
    if json.loads(metadata_bytes) != metadata:
        raise ValueError("Source metadata was altered after validation")
    source_states = table[list(STATE_COLUMNS)].to_numpy(np.float32).astype(float)
    np.testing.assert_array_equal(source_states, source.states)
    mask_columns = bone_related_columns(table, metadata)
    shared_tree = destination / "shared/tree.nwk"
    shared_metadata = destination / "shared/source.columns.json"
    _freeze(shared_tree, tree_bytes)
    _freeze(shared_metadata, metadata_bytes)
    tip_ids = np.full(len(table), -1, dtype=np.int64)
    tree_rows = np.asarray(source.tree_row_indices, dtype=np.int64)
    if (len(tree_rows) != source.tree.n_tips or len(np.unique(tree_rows)) != len(tree_rows)
            or (tree_rows < 0).any() or (tree_rows >= len(table)).any()):
        raise ValueError("Invalid frozen tree-to-table row mapping")
    tip_ids[tree_rows] = np.arange(source.tree.n_tips)
    eligible = table[list(REQUIRE_COLUMNS)].notna().all(axis=1).to_numpy()
    entries = {}
    for cell in cells:
        fold, key = cell["fold"], cell["input_key"]
        canonical = source.canonical_windows[fold]
        train_mask, val_mask = np.asarray(canonical["train_mask"]), np.asarray(canonical["val_mask"])
        expected_val = eligible & table[FOLD_COLUMN].eq(fold).to_numpy()
        expected_train = eligible & ~table[FOLD_COLUMN].eq(fold).to_numpy()
        if (not np.array_equal(train_mask, expected_train) or not np.array_equal(val_mask, expected_val)):
            raise ValueError("Canonical windows and original frozen folds disagree")
        train, validation = np.flatnonzero(train_mask), np.flatnonzero(val_mask)
        refs, targets = np.asarray(canonical["reference_rows"]), np.asarray(canonical["target_rows"])
        if (refs.ndim != 2 or targets.shape != (len(refs),) or not val_mask[refs].all()
                or not val_mask[targets].all() or (refs == targets[:, None]).any()
                or (tip_ids[np.r_[train, targets, refs.ravel()]] < 0).any()):
            raise ValueError("Frozen windows must use held-out references excluding their target")
        if (np.asarray(canonical["distances"]).shape != refs.shape
                or np.asarray(canonical["order_ids"]).shape != targets.shape):
            raise ValueError("Frozen distance/order arrays do not match the saved windows")
        choice = choose_train_bone_rows(table, train, cell["condition"], cell["mask_seed"], fold)
        kept, hidden = choice["kept"], choice["hidden"]
        masked = table.copy(deep=True)
        masked.loc[hidden, mask_columns] = np.nan
        content = _masked_csv_bytes(original, table.columns, hidden, mask_columns)
        reloaded = pd.read_csv(io.BytesIO(content))
        pd.testing.assert_frame_equal(reloaded, masked, check_dtype=False, check_exact=True)
        np.testing.assert_array_equal(reloaded.loc[validation, list(STATE_COLUMNS)].to_numpy(np.float32),
                                      source_states[validation].astype(np.float32))
        cell_dir = destination / key
        output_table = cell_dir / "species_traits.csv"
        table_hash = _freeze(output_table, content)
        audit = dict(cell, **choice["audit"], masked_columns=mask_columns,
                     n_train_species=len(train), n_validation_species=len(validation), n_windows=len(targets),
                     source_inputs=originals, source_table=_relative(table_path, root),
                     source_metadata=_relative(metadata_path, root),
                     source_tree=_relative(tree_path, root),
                     fold_policy="Original source table fold column and canonical train/validation masks; no repartitioning.",
                     role_policy="Only selected eligible training rows are erased, including legacy skeleton/count/provenance columns. Held-out table morphology stays intact. All training labels, teachers, references, normalization and anatomical transforms must be rebuilt from this masked table.",
                     reference_policy="Prediction-only reference missingness via make_reference_states; target truth is unchanged.",
                     source_metadata_policy="Original schema/frozen_split retained; original coverage and source-builder metadata describe the original source, not the deleted training subset.")
        derived_metadata = copy.deepcopy(metadata)
        derived_metadata["table_sha256"] = table_hash
        derived_metadata["asymmetry_v1"] = audit
        output_metadata = output_table.with_suffix(".columns.json")
        metadata_hash = _freeze(output_metadata, _json_bytes(derived_metadata))
        arrays = dict(train_rows=train, validation_rows=validation, reference_rows=refs,
                      target_rows=targets, distances=np.asarray(canonical["distances"]),
                      tip_ids_of_rows=tip_ids, order_ids=np.asarray(canonical["order_ids"]),
                      kept_train_bone_rows=kept, hidden_train_bone_rows=hidden)
        stream = io.BytesIO()
        np.savez_compressed(stream, **arrays)
        npz_path = cell_dir / "windows.npz"
        npz_hash = _freeze(npz_path, stream.getvalue())
        rank = np.full(len(table), -1, dtype=np.int64)
        rank[choice["ordered"]] = np.arange(len(choice["ordered"]))
        mask_table = pd.DataFrame(dict(row=np.arange(len(table)), tip=table.tip, family=table.family,
                                      train=train_mask, validation=val_mask,
                                      original_complete_bones=np.isfinite(source_states[:, 1:]).all(axis=1),
                                      kept_train_bones=np.isin(np.arange(len(table)), kept),
                                      hidden_train_bones=np.isin(np.arange(len(table)), hidden),
                                      selection_rank=rank))
        mask_path = cell_dir / "training_mask.csv"
        mask_hash = _freeze(mask_path, mask_table.to_csv(index=False, lineterminator="\n").encode("utf-8"))
        audit_path = cell_dir / "audit.json"
        audit_hash = _freeze(audit_path, _json_bytes(audit))
        entries[key] = dict(cell, table=_relative(output_table, root), tree=_relative(shared_tree, root),
                            metadata=_relative(output_metadata, root), npz=_relative(npz_path, root),
                            mask_table=_relative(mask_path, root), audit=_relative(audit_path, root),
                            hashes=dict(table_sha256=table_hash, table_metadata_sha256=metadata_hash,
                                        tree_sha256=originals["tree_sha256"], npz_sha256=npz_hash,
                                        mask_table_sha256=mask_hash, audit_sha256=audit_hash),
                            n_train_species=len(train), n_validation_species=len(validation),
                            n_original_train_bones=choice["audit"]["original_train_bone_species"],
                            n_kept_train_bones=len(kept), n_hidden_train_bones=len(hidden))
    result = dict(mode=mode, tables=entries, source_run=_relative(source_path, root),
                  source_inputs=originals, original_metadata=_relative(shared_metadata, root),
                  masked_columns=mask_columns, hash_protocol=HASH_PROTOCOL,
                  evaluation_seed=0, validation_window_seed=1,
                  plan=make_plan(mode),
                  reference_policy="Half deletes floor(N/2) unique originally complete reference species by stable hash; all deletes three bones; distal deletes b_tmt only. State NaNs must become zero values and zero masks together.")
    manifest_path = destination / "input_audit.json"
    digest = _freeze(manifest_path, _json_bytes(result))
    return dict(result, input_audit=_relative(manifest_path, root), input_audit_sha256=digest)
