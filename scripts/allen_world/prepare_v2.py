"""Separate ecological information groups without changing observations or folds.

Foraging strata are exposure proxies, not measured bare skin or thermal dose.
AVONET life categories are locomotor proxies; correlations between groups remain.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

TARGETS = ["m_tarsus", "b_fem", "b_tbt", "b_tmt"]
REQUIRE_COLS = ["m_tarsus", "m_mass", "e_dev_A", "bio1_c", "bio5_c", "bio6_c"]
GROUPS = {
    "temperature": ["bio1_c", "bio5_c", "bio6_c"],
    "exposure": [f"e_fs_{s}" for s in ("watbelowsurf", "wataroundsurf", "ground", "understory", "midhigh", "canopy", "aerial")],
    "motion": [f"e_life_{s}" for s in ("aerial", "aquatic", "generalist", "insessorial", "terrestrial")],
    "development": [f"e_dev_{s}" for s in ("P", "SP", "SA", "A")],
    "background": ["m_mass", "e_habdens", "e_migr"],
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(table_path, out_dir):
    from evowm.data.transect import family_folds
    table_path, out_dir = Path(table_path).resolve(), Path(out_dir).resolve()
    destination = out_dir / "species_traits.csv"
    if destination == table_path:
        raise ValueError("v2 preparation must not overwrite its input")
    for protected in (ROOT / "data/raw", ROOT / "data/processed"):
        if out_dir == protected or protected in out_dir.parents:
            raise ValueError("Write derived v2 inputs under outputs, not source data")
    meta_path = table_path.with_suffix(".columns.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    table = pd.read_csv(table_path)
    columns = sum(GROUPS.values(), [])
    required = ["tip", "family", "order", "mode"] + TARGETS + columns
    missing = [c for c in required if c not in table]
    if missing:
        raise ValueError(f"Missing v2 input columns: {missing}")
    if table.tip.isna().any() or table.tip.duplicated().any():
        raise ValueError("Input tip names must be present and unique")
    if table.family.isna().any():
        raise ValueError("Input families must be present")
    if len(columns) != len(set(columns)) or set(columns) & set(TARGETS):
        raise AssertionError("Condition groups overlap or contain an outcome")
    if np.isinf(table[TARGETS + columns].to_numpy(float)).any():
        raise ValueError("Infinite values in input")
    if meta.get("table_sha256") and meta["table_sha256"] != sha(table_path):
        raise ValueError("Source table does not match its recorded SHA256")
    eligible = table[REQUIRE_COLS].notna().all(axis=1).to_numpy()
    fold_of = family_folds(table, eligible, n_folds=5, seed=0, level="family")
    # Extra audit column only; never included in model condition/state metadata.
    table["allen_frozen_family_fold"] = table.family.map(fold_of).fillna(-1).astype(int)
    meta = copy.deepcopy(meta)
    meta.update(schema_version="allen_world_v2", eco_groups=copy.deepcopy(GROUPS), eco=columns,
                morph=[TARGETS[0]], skeleton=TARGETS[1:], state_default=TARGETS,
                action_dim=5 * len(columns) + len(GROUPS) + 2)
    meta["condition_interpretation"] = {
        "exposure": "Foraging-stratum ecological proxy; neither bare-skin fraction nor measured thermal dose.",
        "motion": "AVONET life categories; not measured gait, force, or mechanical work.",
        "development": "Family-assigned P/SP/SA/A, with the existing Passeriformes=A convention; not individual development measurements.",
        "shared_information": "Columns do not overlap across groups, but biological and statistical correlations remain. Incremental prediction does not identify unique causal effects.",
        "X": "Audit-only sum of ground/water-surface/underwater foraging; not a duplicate model input.",
    }
    meta["frozen_split"] = {"level": "family", "folds": 5, "seed": 0,
                            "require_cols": REQUIRE_COLS, "eligible_species": int(eligible.sum()),
                            "family_to_fold": fold_of, "audit_column": "allen_frozen_family_fold",
                            "rule": "Same eligibility and family_folds algorithm as allen_world_v1; never repartition by architecture or skeletal availability."}
    meta["v2_preparation"] = {"source_table": str(table_path), "source_table_sha256": sha(table_path),
                               "source_metadata_sha256": sha(meta_path), "script_sha256": sha(__file__),
                               "observations_changed": False}
    meta["audit_only_columns"] = list(dict.fromkeys(meta.get("audit_only_columns", []) + ["allen_frozen_family_fold"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination, index=False, float_format="%.12g")
    meta["table_sha256"] = sha(destination)
    destination.with_suffix(".columns.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (out_dir / "family_folds.json").write_text(json.dumps(meta["frozen_split"], indent=2), encoding="utf-8")
    return destination


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--table", type=Path, default=ROOT / "outputs/allen_world_v1/input/species_traits.csv")
    p.add_argument("--out-dir", type=Path, default=ROOT / "outputs/allen_world_v2/input")
    args = p.parse_args()
    print(prepare(args.table, args.out_dir))


if __name__ == "__main__":
    main()
