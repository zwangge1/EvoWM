"""Frozen configuration-only Allen v4 search and leakage-safe nested inputs.

This module changes no model or upstream training implementation.  Inner tables
keep every original row/tree tip but erase all model values of the outer held-out
families before repartitioning the remaining families.  The copied outer table
and metadata retain their original bytes and frozen outer split.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from evowm.data.transect import family_folds
from evowm.data.tree import read_trees

FOLD_COLUMN = "allen_frozen_family_fold"
OUTER_FOLD_COLUMN = "allen_v4_original_outer_fold"
STATE_COLUMNS = ("m_tarsus", "b_fem", "b_tbt", "b_tmt")
REFERENCE_CANDIDATE = "reference"
INNER_FOLDS = (0, 1)
TRAINING_SEEDS = (13, 14)
OUTER_VARIANTS = ("full", "single", "supervised")


def _candidate(name, steps=4000, *, gentle=False, light=False, total=.25, ratio=.25):
    masks = [.75] + [.25 / 7] * 7 if gentle else [.125] * 8
    pred = .25 if light else 1.0
    overrides = [
        "data.dataset.condition_mask_probs=" + json.dumps(masks, separators=(",", ":")),
        "evowm_allen_anatomical.observation_weight=1.0",
        "evowm_allen_anatomical.total_weight=" + str(float(total)),
        "evowm_allen_anatomical.ratio_weight=" + str(float(ratio)),
        "++level1.loss.embed.pred.enabled=true",
        "++level1.loss.embed.pred.weight=" + str(pred),
        "++level2.loss.embed.pred.enabled=true",
        "++level2.loss.embed.pred.weight=" + str(pred),
    ]
    return dict(name=name, steps=steps, mask_probs=masks, pred_weight_level1=pred,
                pred_weight_level2=pred, observation_weight=1., total_weight=float(total),
                ratio_weight=float(ratio), sigreg_policy="unchanged_original", overrides=overrides)


CANDIDATES = [
    _candidate("reference"),
    _candidate("gentle_mask", gentle=True),
    _candidate("light_latent", gentle=True, light=True),
    _candidate("size_balance", gentle=True, light=True, total=1.),
    _candidate("bone_balance", gentle=True, light=True, total=0., ratio=0.),
    _candidate("long", steps=8000, gentle=True, light=True),
    _candidate("size_long", steps=8000, gentle=True, light=True, total=1.),
]


def _scope(mode):
    if mode not in ("smoke", "full"):
        raise ValueError("mode must be smoke or full")
    return ([0], [0], [13]) if mode == "smoke" else (list(range(5)), list(INNER_FOLDS), list(TRAINING_SEEDS))


def _job(stage, outer_fold, fold, seed, candidate, variant, mode):
    recipe = next(c for c in CANDIDATES if c["name"] == candidate)
    steps = 64 if mode == "smoke" else recipe["steps"]
    name = (f"inner_o{outer_fold}_i{fold}_{candidate}_s{seed}" if stage == "inner"
            else f"outer_o{outer_fold}_{variant}_{candidate}_s{seed}")
    return dict(name=name, stage=stage, outer_fold=outer_fold,
                inner_fold=fold if stage == "inner" else None, fold=fold, seed=seed,
                candidate=candidate, variant=variant, config="allen_v3_" + variant,
                steps=steps, recipe_steps=recipe["steps"], overrides=list(recipe["overrides"]),
                input_key=f"inner_{outer_fold}" if stage == "inner" else "outer")


def make_plan(mode="full"):
    """Return the predeclared 140+30 full or 7+3 smoke job matrix.

    Outer templates deliberately have no selected recipe or steps.  Fill them
    only after inner selection using make_outer_jobs.  tune_jobs aliases the
    inner job contents for readers using tuning terminology.
    """
    outer_folds, inner_folds, seeds = _scope(mode)
    jobs = [_job("inner", outer, inner, seed, c["name"], "full", mode)
            for outer in outer_folds for inner in inner_folds for seed in seeds for c in CANDIDATES]
    templates = [dict(stage="outer", outer_fold=outer, fold=outer, seed=seed,
                      variant=variant, config="allen_v3_" + variant, input_key="outer")
                 for outer in outer_folds for seed in seeds for variant in OUTER_VARIANTS]
    return dict(mode=mode, candidates=copy.deepcopy(list(CANDIDATES)),
                reference_candidate=REFERENCE_CANDIDATE, outer_folds=outer_folds,
                inner_folds=inner_folds, seeds=seeds, inner_jobs=jobs,
                tune_jobs=copy.deepcopy(jobs), outer_templates=templates,
                counts=dict(inner=len(jobs), outer=len(templates), total=len(jobs) + len(templates)),
                selection_scope="Outer-family values excluded before all inner preparation and selection.",
                outer_comparison="Chosen full-model recipe reused for full/single/supervised; controls are retrained, not independently tuned.",
                interpretation="Development study: previously viewed outer results are not a new untouched test set.")


def make_outer_jobs(mode, selections):
    """Expand outer-fold -> selected candidate name after inner-only selection."""
    folds, _, seeds = _scope(mode)
    if not isinstance(selections, dict):
        raise ValueError("selections must map each outer fold to a candidate name")
    try:
        chosen = {int(key): value for key, value in selections.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid outer fold selection keys") from exc
    if len(chosen) != len(selections) or set(chosen) != set(folds):
        raise ValueError("one selected candidate is required for every configured outer fold")
    names = {c["name"] for c in CANDIDATES}
    if any(not isinstance(value, str) or value not in names for value in chosen.values()):
        raise ValueError("unknown selected candidate")
    return [_job("outer", fold, fold, seed, chosen[fold], variant, mode)
            for fold in folds for seed in seeds for variant in OUTER_VARIANTS]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path(path, root):
    return (root / path).resolve()


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare_inputs(source_table, source_meta, tree, destination, root=ROOT, *, outer_folds=None):
    """Freeze original outer inputs and one scrubbed inner table per outer fold.

    Return tables[input_key] with root-relative table/metadata/tree paths suitable
    for job.update(...), together with hashes and explicit split counts.  This
    function never overwrites a destination and never imports Hydra.
    """
    root = Path(root).resolve()
    table_path, meta_path, tree_path = (_path(p, root) for p in (source_table, source_meta, tree))
    destination = _path(destination, root)
    if destination == root or not destination.is_relative_to(root):
        raise ValueError("destination must be a new directory strictly inside root")
    if destination.exists():
        raise FileExistsError("refusing to overwrite prepared inputs: " + str(destination))
    folds = list(range(5)) if outer_folds is None else list(outer_folds)
    if (not folds or any(type(fold) is not int or fold not in range(5) for fold in folds)
            or len(set(folds)) != len(folds)):
        raise ValueError("outer_folds must be distinct integers from zero through four")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    source_digest = sha256(table_path)
    if meta.get("table_sha256") != source_digest:
        raise ValueError("source table SHA256 disagrees with its frozen metadata")
    table = pd.read_csv(table_path)
    if (table.tip.isna().any() or table.tip.duplicated().any()
            or table.family.isna().any() or "order" not in table):
        raise ValueError("unique tips and complete family/order identity columns are required")
    if OUTER_FOLD_COLUMN in table:
        raise ValueError("source already contains the reserved original-outer-fold column")
    declaration = meta.get("frozen_split", {})
    if (declaration.get("level") != "family" or declaration.get("folds") != 5
            or declaration.get("seed") != 0
            or declaration.get("audit_column", FOLD_COLUMN) != FOLD_COLUMN):
        raise ValueError("source must declare the frozen five family folds with seed zero")
    require = list(declaration.get("require_cols", []))
    if not require or FOLD_COLUMN not in table:
        raise ValueError("source eligibility and frozen fold column are required")
    groups = meta.get("eco_groups", {})
    model_columns = list(dict.fromkeys(list(meta.get("morph", [])) + list(meta.get("skeleton", []))
                      + list(meta.get("state_default", [])) + list(meta.get("eco", []))
                      + [c for columns in groups.values() for c in columns] + require))
    if tuple(meta.get("state_default", [])) != STATE_COLUMNS or not set(require + model_columns) <= set(table):
        raise ValueError("source model/eligibility columns do not match the Allen state schema")
    if np.isinf(table[model_columns].to_numpy(float)).any():
        raise ValueError("source model values must be finite or missing")
    eligible = table[require].notna().all(axis=1).to_numpy()
    mapping = declaration.get("family_to_fold", {})
    if (not isinstance(mapping, dict) or not set(table.loc[eligible, "family"]) <= set(mapping)
            or any(type(value) is not int or value not in range(5) for value in mapping.values())):
        raise ValueError("source family mapping is incomplete or invalid")
    source_fold = table.family.map(mapping).fillna(-1).astype(int).to_numpy()
    if not np.array_equal(source_fold, table[FOLD_COLUMN].to_numpy()):
        raise ValueError("source frozen fold column and family mapping disagree")
    trees = read_trees(tree_path)
    if len(trees) != 1 or not set(trees[0].tip_names) <= set(table.tip):
        raise ValueError("one frozen tree is required and all tree tips must remain in the table")
    if not set(table.loc[eligible, "tip"]) <= set(trees[0].tip_names):
        raise ValueError("an eligible species is absent from the frozen tree")
    # Also erase legacy/unselected morphology and derived ecological audit
    # columns: retaining only identity prevents a later lookup from exposing an
    # outer observation through a column absent from today's model metadata.
    erased_columns = [c for c in table if c not in ("tip", "family", "order", FOLD_COLUMN)]
    prepared = []
    for outer in folds:
        held_families = {family for family, value in mapping.items() if value == outer}
        excluded = table.family.isin(held_families).to_numpy()
        inner = table.copy(deep=True)
        inner[OUTER_FOLD_COLUMN] = source_fold
        inner.loc[excluded, erased_columns] = np.nan
        inner_eligible = inner[require].notna().all(axis=1).to_numpy()
        if not np.array_equal(inner_eligible, eligible & ~excluded):
            raise ValueError("outer erasure did not isolate eligible outer-training species")
        if not excluded.any() or len(set(inner.loc[inner_eligible, "family"])) < 5:
            raise ValueError("outer split requires held-out species and at least five remaining eligible families")
        inner_mapping = family_folds(inner, inner_eligible, n_folds=5, seed=0, level="family")
        inner[FOLD_COLUMN] = inner.family.map(inner_mapping).fillna(-1).astype(int)
        info = dict(outer_fold=outer, n_rows=len(inner), n_tree_tips=trees[0].n_tips,
                    n_outer_excluded_rows=int(excluded.sum()), n_outer_excluded_families=len(held_families),
                    n_inner_eligible=int(inner_eligible.sum()), n_inner_families=len(inner_mapping),
                    inner_folds={str(k): {"n_species": int((inner_eligible & inner[FOLD_COLUMN].eq(k)).sum()),
                                         "n_families": sum(v == k for v in inner_mapping.values())}
                                 for k in range(5)},
                    erased_columns=erased_columns, model_columns=model_columns, outer_fold_audit_column=OUTER_FOLD_COLUMN,
                    erased_values="All nonidentity data, including legacy morphology and ecological audit columns, for every member of outer held-out families are NaN.",
                    identity_policy="All rows, tip names, family names, orders and the complete frozen tree are retained.")
        inner_meta = copy.deepcopy(meta)
        inner_meta["frozen_split"] = dict(level="family", folds=5, seed=0, require_cols=require,
            eligible_species=int(inner_eligible.sum()), family_to_fold=inner_mapping, audit_column=FOLD_COLUMN,
            rule="Original family_folds(seed=0,n_folds=5) applied only after complete outer-family value erasure.")
        inner_meta["audit_only_columns"] = list(dict.fromkeys(meta.get("audit_only_columns", []) + [OUTER_FOLD_COLUMN]))
        inner_meta["nested_inner_preparation"] = info
        prepared.append((outer, inner, inner_meta, info))
    destination.mkdir(parents=True)
    copied_tree = destination / "tree.nwk"
    shutil.copyfile(tree_path, copied_tree)
    outer_dir = destination / "outer"
    outer_dir.mkdir()
    outer_table, outer_meta = outer_dir / "species_traits.csv", outer_dir / "species_traits.columns.json"
    shutil.copyfile(table_path, outer_table)
    shutil.copyfile(meta_path, outer_meta)

    def spec(table, metadata):
        return dict(table=table.relative_to(root).as_posix(), metadata=metadata.relative_to(root).as_posix(),
                    tree=copied_tree.relative_to(root).as_posix(), table_sha256=sha256(table),
                    metadata_sha256=sha256(metadata), tree_sha256=sha256(copied_tree))

    tables, audits = {"outer": spec(outer_table, outer_meta)}, {}
    for outer, inner, inner_meta, info in prepared:
        folder = destination / f"inner_{outer}"
        folder.mkdir()
        new_table, new_meta = folder / "species_traits.csv", folder / "species_traits.columns.json"
        inner.to_csv(new_table, index=False, float_format="%.17g")
        inner_meta["table_sha256"] = sha256(new_table)
        _json(new_meta, inner_meta)
        _json(folder / "family_folds.json", inner_meta["frozen_split"])
        _json(folder / "split_audit.json", info)
        tables[f"inner_{outer}"] = spec(new_table, new_meta)
        audits[str(outer)] = info
    result = dict(tables=tables, audit=audits, outer_folds=folds,
                  source_sha256=dict(table=source_digest, metadata=sha256(meta_path), tree=sha256(tree_path)),
                  source_paths=dict(table=str(table_path), metadata=str(meta_path), tree=str(tree_path)),
                  original_outer_bytes_preserved=True)
    _json(destination / "input_manifest.json", result)
    return result
