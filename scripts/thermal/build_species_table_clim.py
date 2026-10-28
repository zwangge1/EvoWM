"""给物种表加气候列，作为第四个习性组 climate（热阀反事实 v1，设计 docs/design/热阀与落震_反事实闭环_设计_2026-10-28.md 第 4 节）。

输入 data/processed/species_traits.csv（主分析的表，不改动）与 data/processed/species_climate.csv（分布质心的 WorldClim 值）。
输出 data/processed/species_traits_clim.csv 与 species_traits_clim.columns.json：
  e_clim_bio1 = (bio1_c − 物种均值) / 10，bio5、bio6 同样；单位 10 °C，所以动作里的气候差直接是"每 10 °C"。
  columns.json 的 eco_groups 多一组 climate，climate_scale 记下均值，反事实时按它换算。
在 EvoWM 根目录运行：python scripts/thermal/build_species_table_clim.py
"""
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/processed/species_traits.csv"
CLIM = ROOT / "data/processed/species_climate.csv"
DST = ROOT / "data/processed/species_traits_clim.csv"
BIOS = (1, 5, 6)
DIV = 10.0


def main():
    t = pd.read_csv(SRC)
    meta = json.loads(SRC.with_suffix(".columns.json").read_text(encoding="utf-8"))
    c = pd.read_csv(CLIM, usecols=["tip"] + [f"bio{b}_c" for b in BIOS])
    assert t["tip"].is_unique and c["tip"].is_unique
    d = t.merge(c, on="tip", how="left")
    assert len(d) == len(t)
    scale = {}
    cols = []
    for b in BIOS:
        col = f"e_clim_bio{b}"
        mean = float(d[f"bio{b}_c"].mean())
        d[col] = (d[f"bio{b}_c"] - mean) / DIV
        scale[col] = {"source": f"bio{b}_c", "mean": mean, "div": DIV}
        cols.append(col)
    d = d.drop(columns=[f"bio{b}_c" for b in BIOS])
    meta = dict(meta)
    meta["eco_groups"] = {**meta["eco_groups"], "climate": cols}
    meta["eco"] = [x for grp in meta["eco_groups"].values() for x in grp]
    meta["climate_scale"] = scale
    meta["source_table_md5"] = hashlib.md5(SRC.read_bytes()).hexdigest()
    d.to_csv(DST, index=False)
    DST.with_suffix(".columns.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    n_clim = int(d[cols].notna().all(axis=1).sum())
    print(f"{DST.relative_to(ROOT)}: {len(d)} 种，有气候 {n_clim} 种，习性列 {len(meta['eco'])}，习性组 {list(meta['eco_groups'])}")
    print(f"action_dim（change）应为 {len(meta['eco']) + len(meta['eco_groups']) + 2}")
    print(f"md5 {hashlib.md5(DST.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
