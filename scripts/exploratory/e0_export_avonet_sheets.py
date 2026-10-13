"""探索性：把 AVONET 补充数据 xlsx 的三张表导出为 csv，并做 Skelevision x AVONET3 盘点表。

2026-10-13 数据盘点用，只用于选方向。输出到 data/interim/exploratory_20261013/。
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
XLSX = ROOT / "data/raw/avonet/AVONET_Supplementary_dataset1.xlsx"
OUT = ROOT / "data/interim/exploratory_20261013"

SHEETS = {
    "AVONET3_BirdTree": "avonet3_birdtree.csv",
    "AVONET1_BirdLife": "avonet1_birdlife.csv",
    "BirdLife–BirdTree crosswalk": "cw_birdlife_birdtree.csv",  # 表名原文含 en dash，不能改
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for sheet, name in SHEETS.items():
        pd.read_excel(XLSX, sheet_name=sheet).to_csv(OUT / name, index=False)

    sp = pd.read_csv(ROOT / "data/raw/skelevision/Skelevision_species_complete_v1.1.csv")
    a3 = pd.read_csv(OUT / "avonet3_birdtree.csv")
    a3["sp"] = a3["Species3"].str.replace(" ", "_")
    fd = pd.read_csv(ROOT / "data/raw/eltontraits_derived/forstrat_dom.csv")
    fd["sp"] = fd["scientificNameStd"].str.replace(" ", "_")
    x = sp.merge(a3, left_on="Species", right_on="sp").merge(
        fd[["sp", "dom_strat"]], on="sp", how="left"
    )
    x.to_csv(OUT / "skelevision_x_avonet3.csv", index=False)
    print(f"skelevision x avonet3: {len(x)} 种")


if __name__ == "__main__":
    main()
