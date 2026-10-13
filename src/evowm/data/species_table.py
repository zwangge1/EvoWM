"""物种主表：每个 BirdTree 物种一行，含形态（m_）、骨段（k_）、习性（e_）三组列。

输入全部来自 data/raw（只读）；输出 data/processed/species_traits.csv 和同名 .columns.json。

口径：
- AVONET 推断值（Inference == YES，由近缘种推出）置为缺失，不当观测用。
- Skelevision 只用个体实测表 Skelevision_Only，按种对 log10 长度取均值；
  不用其物种均值表（已沿树插补）。列名：tarsus 是胫跗骨，metatarsus 是跗跖骨。
- 鸭科骨段（Watanabe 2017）按同物异名映射到 BirdTree 名；同一种多条（亚种）按样本量加权平均。
- EltonTraits 派生表先按名直配，配不上的再经 AVONET 自带的 BirdLife 到 BirdTree 一对一对照补配。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

AVONET_XLSX = "data/raw/avonet/AVONET_Supplementary_dataset1.xlsx"
SKEL_ONLY = "data/raw/skelevision/Skelevision_Only_Dataset_v1.csv"
WATANABE = "data/raw/watanabe2017/watanabe2017_anatidae.csv"
ELTON_FS = "data/raw/eltontraits_derived/avonet_joined_all.csv"
ELTON_DIET = "data/raw/eltontraits_derived/avonet_hwi.csv"
CROSSWALK_SHEET = "BirdLife–BirdTree crosswalk"  # 表名原文含 en dash，不能改

AVONET_MORPH = {  # 取 log10
    "Beak.Length_Culmen": "m_beak_culmen",
    "Beak.Length_Nares": "m_beak_nares",
    "Beak.Width": "m_beak_width",
    "Beak.Depth": "m_beak_depth",
    "Tarsus.Length": "m_tarsus",
    "Wing.Length": "m_wing",
    "Kipps.Distance": "m_kipps",
    "Secondary1": "m_secondary1",
    "Tail.Length": "m_tail",
    "Mass": "m_mass",
}

SKEL_COLS = {  # Skelevision 原列名 → 输出列（log10 mm）
    "femur": "k_femur",
    "tarsus": "k_tibiotarsus",
    "metatarsus": "k_tarsometatarsus",
    "humerus": "k_humerus",
    "ulna": "k_ulna",
    "radius": "k_radius",
    "carpometacarpus": "k_carpometacarpus",
    "second_digit": "k_digit2",
    "cv.keel.1": "k_keel",
    "cv.furcula.1": "k_furcula",
    "cv.skull.1": "k_skull",
    "sclerotic_ring": "k_sclerotic",
}
LEG = ["k_femur", "k_tibiotarsus", "k_tarsometatarsus"]

# Watanabe 名 → BirdTree 名（BirdTree 采用 2012 年 BirdLife 分类，部分属与种被合并）。
# 映射目标不在树上的条目会被丢弃并打印。
WATANABE_SYNONYMS = {
    "Spatula clypeata": "Anas clypeata",
    "Mareca strepera": "Anas strepera",
    "Mareca falcata": "Anas falcata",
    "Mareca penelope": "Anas penelope",
    "Sibirionetta formosa": "Anas formosa",
    "Anser caerulescens": "Chen caerulescens",
    "Anser rossii": "Chen rossii",
    "Anser canagicus": "Chen canagica",
    "Anser cygnoid": "Anser cygnoides",
    "Rhodonessa caryophyllacea": "Rhodonessa caryophyllacea",
    "Anas nesiotis a": "Anas nesiotis",
    "Anas zonorhyncha": "Anas poecilorhyncha",
    "Melanitta americana": "Melanitta nigra",
    "Melanitta deglandi": "Melanitta fusca",
    "Melanitta stejnegeri": "Melanitta fusca",
    "Branta h. minima": "Branta hutchinsii",
    "Branta h. taverneri": "Branta hutchinsii",
    "Branta canadensis moffitti": "Branta canadensis",
    "Branta c.canadensis": "Branta canadensis",
}

FORSTRAT = ["watbelowsurf", "wataroundsurf", "ground", "understory", "midhigh", "canopy", "aerial"]
DIET = ["Inv", "Vend", "Vect", "Vfish", "Vunk", "Scav", "Fruit", "Nect", "Seed", "PlantO"]


def _tip(name: str) -> str:
    return str(name).strip().replace(" ", "_")


def _onehot(series: pd.Series, prefix: str) -> pd.DataFrame:
    s = series.astype("string").str.strip()
    cats = sorted(s.dropna().unique())
    out = pd.DataFrame(
        {f"{prefix}{c.lower().replace(' ', '_')}": (s == c).astype(float) for c in cats},
        index=series.index,
    )
    out[s.isna().to_numpy()] = np.nan
    return out


def _log10_pos(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    return np.log10(x.where(x > 0))


def build(root: Path) -> tuple[pd.DataFrame, dict]:
    root = Path(root)
    log = []

    # ---- AVONET（BirdTree 分类）----
    av = pd.read_excel(root / AVONET_XLSX, sheet_name="AVONET3_BirdTree")
    av["tip"] = av["Species3"].map(_tip)
    df = pd.DataFrame({
        "tip": av["tip"],
        "family": av["Family3"].astype(str).str.strip(),
        "order": av["Order3"].astype(str).str.strip(),
        "avonet_inferred": (av["Inference"].astype(str).str.upper() == "YES").astype(int),
    })
    for src, dst in AVONET_MORPH.items():
        df[dst] = _log10_pos(av[src])
    df["m_hwi"] = pd.to_numeric(av["Hand-Wing.Index"], errors="coerce") / 100.0
    morph_cols = list(AVONET_MORPH.values()) + ["m_hwi"]
    df.loc[df["avonet_inferred"] == 1, morph_cols] = np.nan
    log.append(f"AVONET: {len(df)} 种，推断值置缺 {int(df['avonet_inferred'].sum())} 种")

    eco_av = pd.concat([
        _onehot(av["Primary.Lifestyle"], "e_life_"),
        _onehot(av["Trophic.Niche"], "e_niche_"),
        _onehot(av["Habitat"], "e_hab_"),
    ], axis=1)
    eco_av["e_habdens"] = (pd.to_numeric(av["Habitat.Density"], errors="coerce") - 1) / 2
    eco_av["e_migr"] = (pd.to_numeric(av["Migration"], errors="coerce") - 1) / 2
    df = pd.concat([df, eco_av], axis=1)

    # ---- Skelevision：个体实测，按种取 log10 均值 ----
    sk = pd.read_csv(root / SKEL_ONLY)
    sk_log = pd.DataFrame({dst: _log10_pos(sk[src]) for src, dst in SKEL_COLS.items()})
    sk_log["tip"] = sk["species"].map(_tip)
    g = sk_log.groupby("tip")
    sk_mean = g.mean()
    sk_n = g.count()[LEG].rename(columns=lambda c: "n_" + c[2:])
    skel = sk_mean.join(sk_n)
    skel["src_skeleton"] = "skelevision"
    log.append(f"Skelevision: {len(sk)} 个标本，{len(skel)} 种；"
               f"三段腿骨均有实测的种 {int((skel[[c for c in skel if c.startswith('n_')]] > 0).all(axis=1).sum())}")

    # ---- Watanabe 鸭科 ----
    wa = pd.read_csv(root / WATANABE)
    tips = set(df["tip"])
    wa["target"] = wa["species"].map(lambda s: WATANABE_SYNONYMS.get(s, s)).map(_tip)
    dropped = wa.loc[~wa["target"].isin(tips), "species"].tolist()
    wa = wa[wa["target"].isin(tips)].copy()
    for src, dst in [("fem_mm", "k_femur"), ("tib_mm", "k_tibiotarsus"), ("tmt_mm", "k_tarsometatarsus")]:
        wa[dst] = np.log10(wa[src])
    wa["w"] = wa["n_min"].clip(lower=1)
    rows = []
    for tip, grp in wa.groupby("target"):
        w = grp["w"].to_numpy(float)
        rec = {c: float(np.average(grp[c], weights=w)) for c in LEG}
        rec.update({"n_" + c[2:]: int(grp["n_min"].sum()) for c in LEG})
        rec["tip"] = tip
        rows.append(rec)
    wat = pd.DataFrame(rows).set_index("tip")
    wat["src_skeleton"] = "watanabe2017"
    overlap = set(wat.index) & set(skel.index)
    if overlap:
        raise ValueError(f"鸭科与 Skelevision 物种重叠：{sorted(overlap)[:5]}")
    log.append(f"Watanabe: 映射后 {len(wat)} 种；丢弃 {len(dropped)} 条（树上无对应）：{dropped}")

    bones = pd.concat([skel, wat], axis=0, sort=False)
    df = df.merge(bones, left_on="tip", right_index=True, how="left")
    df["src_skeleton"] = df["src_skeleton"].fillna("")
    for c in [c for c in df if c.startswith("n_")]:
        df[c] = df[c].fillna(0).astype(int)

    # ---- EltonTraits 派生表 ----
    fs = pd.read_csv(root / ELTON_FS).drop_duplicates("scientificNameStd")
    di = pd.read_csv(root / ELTON_DIET).drop_duplicates("scientificNameStd")
    el = fs[["scientificNameStd"] + [f"ForStrat.{c}" for c in FORSTRAT]].merge(
        di[["scientificNameStd", "Nocturnal", "PelagicSpecialist"] + [f"Diet.{c}" for c in DIET]],
        on="scientificNameStd", how="outer",
    )
    el["tip"] = el["scientificNameStd"].map(_tip)
    direct = el["tip"].isin(tips)
    cw = pd.read_excel(root / AVONET_XLSX, sheet_name=CROSSWALK_SHEET)
    cw = cw[cw["Match.type"] == "1BL to 1BT"]
    cw_map = dict(zip(cw["Species1"].map(_tip), cw["Species3"].map(_tip)))
    el.loc[~direct, "tip"] = el.loc[~direct, "tip"].map(cw_map)
    el = el.dropna(subset=["tip"])
    el = el[el["tip"].isin(tips)].drop_duplicates("tip")
    log.append(f"EltonTraits: 直配 {int(direct.sum())} 名，对照表补配后共 {len(el)} 种")

    eco_el = pd.DataFrame({"tip": el["tip"]})
    for c in FORSTRAT:
        eco_el[f"e_fs_{c.lower()}"] = el[f"ForStrat.{c}"].to_numpy(float) / 100.0
    for c in DIET:
        eco_el[f"e_diet_{c.lower()}"] = el[f"Diet.{c}"].to_numpy(float) / 100.0
    eco_el["e_noct"] = el["Nocturnal"].to_numpy(float)
    eco_el["e_pelagic"] = el["PelagicSpecialist"].to_numpy(float)
    df = df.merge(eco_el, on="tip", how="left")

    # ---- 列清单 ----
    m_cols = [c for c in df if c.startswith("m_")]
    k_cols = [c for c in df if c.startswith("k_")]
    eco_groups = {
        "avonet": [c for c in df if c.startswith(("e_life_", "e_niche_", "e_hab_")) or c in ("e_habdens", "e_migr")],
        "elton_forstrat": [c for c in df if c.startswith("e_fs_")],
        "elton_diet": [c for c in df if c.startswith("e_diet_")] + ["e_noct", "e_pelagic"],
    }
    meta = {
        "n_species": int(len(df)),
        "morph": m_cols,
        "skeleton": k_cols,
        "eco_groups": eco_groups,
        "eco": [c for grp in eco_groups.values() for c in grp],
        "state_default": ["m_tarsus", "k_femur", "k_tibiotarsus", "k_tarsometatarsus", "m_mass"],
        "log": log,
    }
    return df, meta


def main(root: str | Path = ".") -> None:
    root = Path(root).resolve()
    df, meta = build(root)
    out = root / "data/processed/species_traits.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, float_format="%.6g")
    out.with_suffix(".columns.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    for line in meta["log"]:
        print(line)
    print(f"写入 {out}（{len(df)} 行，形态 {len(meta['morph'])} 列，骨段 {len(meta['skeleton'])} 列，习性 {len(meta['eco'])} 列）")
