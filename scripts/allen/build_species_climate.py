"""物种级温度表：每个 BirdTree 物种的分布区温度（WorldClim 2.1，bio1 年均温、bio5 最热月最高、bio6 最冷月最低，°C）。

两种估计（见设计文档 2.1）：
- 质心值 *_c：AVONET 分布质心处的像元值，质心在海里时取 1° 内最近的陆地像元（dist_c 记录回退距离，像元数）。
- 盒均值 *_box：纬度取 AVONET 的 Min/Max.Latitude，经度取质心 ± 半宽，半宽由 Range.Size（km²）当作正方形换算：
  half = sqrt(Range.Size) / 2 / (111 km · cos(质心纬度))，至少 0.5°。n_box 为盒内陆地像元数。
输出 data/processed/species_climate.csv，只含 species_traits.csv 里的物种。
在 EvoWM 根目录运行：python scripts/allen/build_species_climate.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from evowm.data.climate import WorldClim  # noqa: E402
from evowm.data.species_table import AVONET_XLSX, _tip  # noqa: E402


def main():
    av = pd.read_excel(ROOT / AVONET_XLSX, sheet_name="AVONET3_BirdTree")
    av["tip"] = av["Species3"].map(_tip)
    cols = ["Centroid.Latitude", "Centroid.Longitude", "Min.Latitude", "Max.Latitude", "Range.Size"]
    d = av[["tip"] + cols].copy()
    d.columns = ["tip", "lat_c", "lon_c", "lat_min", "lat_max", "range_km2"]
    for c in d.columns[1:]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    traits = pd.read_csv(ROOT / "data/processed/species_traits.csv", usecols=["tip"])
    d = d[d["tip"].isin(traits["tip"])].drop_duplicates("tip").reset_index(drop=True)

    wc = WorldClim(ROOT)
    vals, dist = wc.sample(d["lat_c"], d["lon_c"], fallback_deg=1.0)
    for k, v in vals.items():
        d[f"{k}_c"] = v
    d["dist_c"] = dist

    half = np.sqrt(d["range_km2"]) / 2.0 / (111.0 * np.cos(np.deg2rad(d["lat_c"].clip(-89, 89))))
    d["half_width_deg"] = half.clip(lower=0.5)
    box = {f"{k}_box": np.full(len(d), np.nan) for k in vals}
    n_box = np.zeros(len(d), int)
    for i, row in d.iterrows():
        m, n = wc.box_mean(row["lat_min"], row["lat_max"], row["lon_c"], row["half_width_deg"])
        for k, v in m.items():
            box[f"{k}_box"][i] = v
        n_box[i] = n
    for k, v in box.items():
        d[k] = v
    d["n_box"] = n_box

    out = ROOT / "data/processed/species_climate.csv"
    d.to_csv(out, index=False, float_format="%.4f")
    print(f"写入 {out}：{len(d)} 种；质心直接命中 {(d.dist_c == 0).sum()}，回退 {((d.dist_c > 0)).sum()}，"
          f"无值 {d.dist_c.isna().sum()}；盒均值无值 {(d.n_box == 0).sum()}")
    print("bio1 质心与盒均值相关", round(d[["bio1_c", "bio1_box"]].corr().iloc[0, 1], 3))


if __name__ == "__main__":
    main()
