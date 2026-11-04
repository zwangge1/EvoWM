"""Assemble the Allen JEPA input without changing legacy tables or imputing bones.

Eliason bone lengths remain distinct from the legacy Skelevision/Watanabe k_*
columns. Temperatures stay in degrees Celsius; train-fold normalization belongs
to the dataset. X is an audit column, not a second copy of the motion inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from evowm.data.species_table import AVONET_XLSX, CROSSWALK_SHEET, _tip

MODES = ("P", "SP", "SA", "A")
TEMPERATURE = ["bio1_c", "bio5_c", "bio6_c"]
MOTION = [f"e_life_{s}" for s in ("aerial", "aquatic", "generalist", "insessorial", "terrestrial")] + [
    f"e_fs_{s}" for s in ("watbelowsurf", "wataroundsurf", "ground", "understory", "midhigh", "canopy", "aerial")]
BACKGROUND = ["m_mass"] + [f"e_dev_{s}" for s in MODES] + ["e_habdens", "e_migr"]
TARGETS = ["m_tarsus", "b_fem", "b_tbt", "b_tmt"]
ELIASON_FILE = "data/raw/eliason2023/S1_limb_data.csv"
SOURCES = {
    "traits": "data/processed/species_traits.csv",
    "climate": "data/processed/species_climate.csv",
    "development": "data/raw/devmode/devmode_families.csv",
    "eliason": ELIASON_FILE,
    "taxonomic_crosswalk": AVONET_XLSX,
}


def _unique(df: pd.DataFrame, key: str, label: str) -> None:
    if key not in df or df[key].isna().any() or df[key].duplicated().any():
        raise ValueError(f"{label}: {key} must be nonmissing and unique")


def eliason_means(raw: pd.DataFrame, tips: set[str], crosswalk: dict[str, str]) -> pd.DataFrame:
    """Follow the existing Allen loader: arithmetic specimen mean, then log10 mm."""
    e = raw.copy()
    e["tip"] = e["Taxon"].where(e["Taxon"].notna()).map(lambda x: _tip(x) if pd.notna(x) else np.nan)
    miss = ~e["tip"].isin(tips)
    e.loc[miss, "tip"] = e.loc[miss, "tip"].map(crosswalk)
    e = e[e["tip"].isin(tips)].copy()
    columns = {"fem_leng_mm": "b_fem", "tbt_leng_mm": "b_tbt", "tmt_leng_mm": "b_tmt"}
    for col in columns:
        e[col] = pd.to_numeric(e[col], errors="coerce")
        # Nonpositive and infinite values cannot be physical length observations.
        e[col] = e[col].where(np.isfinite(e[col]) & (e[col] > 0))
    g = e.groupby("tip", sort=True)
    result = np.log10(g[list(columns)].mean()).rename(columns=columns)
    counts = g[list(columns)].count().rename(columns={k: "n_eliason_" + v[2:] for k, v in columns.items()})
    result = result.join(counts)
    result["n_eliason_specimens"] = g.size()
    return result.reset_index()


def load_eliason(root: Path, tips: set[str]) -> pd.DataFrame:
    cw = pd.read_excel(root / AVONET_XLSX, sheet_name=CROSSWALK_SHEET)
    cw = cw[cw["Match.type"] == "1BL to 1BT"]
    mapping = dict(zip(cw["Species1"].map(_tip), cw["Species3"].map(_tip)))
    return eliason_means(pd.read_csv(root / ELIASON_FILE), tips, mapping)


def assemble_table(traits: pd.DataFrame, climate: pd.DataFrame, development: pd.DataFrame,
                   bones: pd.DataFrame) -> pd.DataFrame:
    """Left joins retain the master-table row order, including missing-data rows."""
    _unique(traits, "tip", "traits")
    _unique(climate, "tip", "climate")
    _unique(development, "family", "development")
    _unique(bones, "tip", "Eliason")
    bad = development["mode"].dropna().loc[lambda s: ~s.isin(MODES)]
    if len(bad):
        raise ValueError(f"Unknown developmental mode(s): {sorted(bad.unique())}")
    required = ["family", "order", "m_tarsus", "m_mass", "e_habdens", "e_migr"] + MOTION
    missing = [x for x in required if x not in traits]
    if missing:
        raise ValueError(f"Traits table missing columns: {missing}")
    new = set(TEMPERATURE + ["mode", "exposed", "X"] + [f"e_dev_{m}" for m in MODES]) | (set(bones) - {"tip"})
    collision = new & set(traits)
    if collision:
        raise ValueError(f"Refusing ambiguous/previously augmented trait columns: {sorted(collision)}")
    d = traits.merge(climate[["tip"] + TEMPERATURE], on="tip", how="left", validate="one_to_one", sort=False)
    d = d.merge(development[["family", "mode"]], on="family", how="left", validate="many_to_one", sort=False)
    # Same family-based coding convention as scripts/allen/devmode_v2.py.
    d.loc[d["order"] == "Passeriformes", "mode"] = "A"
    for m in MODES:
        d[f"e_dev_{m}"] = (d["mode"] == m).astype(float).where(d["mode"].notna())
    d["exposed"] = d["mode"].isin(["P", "SP"]).astype(float).where(d["mode"].notna())
    d["X"] = d[["e_fs_ground", "e_fs_wataroundsurf", "e_fs_watbelowsurf"]].sum(axis=1, min_count=3)
    d = d.merge(bones, on="tip", how="left", validate="one_to_one", sort=False)
    if len(d) != len(traits) or d["tip"].tolist() != traits["tip"].tolist():
        raise AssertionError("Master-table rows or order changed during assembly")
    numeric = TARGETS + TEMPERATURE + MOTION + BACKGROUND
    for col in numeric:
        d[col] = pd.to_numeric(d[col], errors="raise")
        if np.isinf(d[col].to_numpy(float)).any():
            raise ValueError(f"Infinite value in {col}")
    return d


def table_metadata(d: pd.DataFrame) -> dict:
    eligible = d[["m_tarsus", "m_mass"]].notna().all(axis=1)
    full_temp = d[TEMPERATURE].notna().all(axis=1)
    full_bones = d[TARGETS[1:]].notna().all(axis=1)
    broad = eligible & full_temp & d["mode"].notna()
    return {
        "schema_version": "allen_world_v1",
        "n_species": len(d),
        "morph": ["m_tarsus"],
        "skeleton": TARGETS[1:],
        "state_default": TARGETS,
        "eco_groups": {"temperature": TEMPERATURE, "motion": MOTION, "background": BACKGROUND},
        "eco": TEMPERATURE + MOTION + BACKGROUND,
        "units": {**{c: "log10(mm)" for c in TARGETS}, "m_mass": "log10(g)",
                  **{c: "degrees Celsius" for c in TEMPERATURE}},
        "preprocessing": "No fitted normalization in builder; fit normalization on train-fold rows only.",
        "development_coding": "Family-level mode from devmode_families.csv; all Passeriformes assigned A as in legacy Allen analysis. Unknown modes stay missing, never all-zero dummy codes.",
        "bone_source": "Eliason S1_limb_data.csv only: mean positive measured length by species, then log10. Legacy k_* columns retained for provenance but excluded from model metadata.",
        "validation_scope": "Eliason is a training target source here, not an independent external validation dataset. Evaluate on held-out families and report source/species overlap for any additional dataset.",
        "audit_only_columns": ["X", "exposed", "mode", "src_skeleton"],
        "coverage": {
            "mass_and_external_tarsus": int(eligible.sum()),
            "also_temperature_and_development": int(broad.sum()),
            "also_complete_eliason_bones": int((broad & full_bones).sum()),
            "eliason_bone_families": int(d.loc[broad & full_bones, "family"].nunique()),
            "eliason_bone_orders": int(d.loc[broad & full_bones, "order"].nunique()),
            "eliason_bones_by_mode": d.loc[broad & full_bones, "mode"].value_counts().to_dict(),
            "observations_by_target": {c: int(d[c].notna().sum()) for c in TARGETS},
            "complete_motion": int(d[MOTION].notna().all(axis=1).sum()),
        },
    }


def build(root: Path, out_dir: Path) -> dict:
    root, out_dir = Path(root).resolve(), Path(out_dir).resolve()
    for protected in (root / "data/raw", root / "data/processed"):
        if out_dir == protected or protected in out_dir.parents:
            raise ValueError("Use an output directory outside data/raw and data/processed")
    traits = pd.read_csv(root / SOURCES["traits"])
    bones = load_eliason(root, set(traits["tip"]))
    d = assemble_table(traits, pd.read_csv(root / SOURCES["climate"]),
                       pd.read_csv(root / SOURCES["development"]), bones)
    meta = table_metadata(d)
    meta["sources"] = {name: {"path": path, "sha256": hashlib.sha256((root / path).read_bytes()).hexdigest()}
                       for name, path in SOURCES.items()}
    meta["builder_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "species_traits.csv"
    d.to_csv(dst, index=False, float_format="%.12g")
    meta["table_sha256"] = hashlib.sha256(dst.read_bytes()).hexdigest()
    dst.with_suffix(".columns.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--out-dir", type=Path, default=ROOT / "outputs/allen_world_v1/input")
    args = p.parse_args()
    meta = build(args.root, args.out_dir)
    print(json.dumps({"out_dir": str(args.out_dir.resolve()), "coverage": meta["coverage"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
