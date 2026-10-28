"""WorldClim 2.1 生物气候栅格的点取值与盒均值（只依赖 numpy 与 Pillow，不需要 GDAL）。

栅格：data/raw/worldclim/wc2.1_2.5m_bio_{k}.tif，全球等经纬网格，左上角 (-180, 90)，像元 2.5 角分（1/24 度），
8640 列 × 4320 行，float32，无数据值 -3.4e38（海洋）。bio1 年均温、bio5 最热月最高温、bio6 最冷月最低温，单位 °C。

- sample(lat, lon)：点取值；落在海里（无数据）时在 fallback_deg 范围内找最近的有效像元（海岸采集点、小岛常见）。
- box_mean(lat_min, lat_max, lon_c, half_width_deg)：纬度带 × 经度窗内有效像元的均值，用作物种分布区温度的粗估。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

RES = 2.5 / 60.0           # 像元边长（度）
NODATA_BELOW = -1e30       # 小于此值视为无数据
BIO_NAMES = {1: "bio1", 5: "bio5", 6: "bio6"}


class WorldClim:
    def __init__(self, root: Path, bios=(1, 5, 6)):
        self.root = Path(root)
        self.bios = tuple(bios)
        self.grid = {}
        for k in self.bios:
            path = self.root / f"data/raw/worldclim/wc2.1_2.5m_bio_{k}.tif"
            a = np.array(Image.open(path), dtype=np.float32)
            if a.shape != (4320, 8640):
                raise ValueError(f"{path} 形状 {a.shape}，不是 2.5 角分全球网格")
            a[a < NODATA_BELOW] = np.nan
            self.grid[k] = a
        self.valid = ~np.isnan(self.grid[self.bios[0]])

    @staticmethod
    def rowcol(lat, lon):
        lat = np.nan_to_num(np.asarray(lat, float), nan=0.0)   # 缺坐标的行在 sample 里按 ok 掩码跳过
        lon = np.nan_to_num(np.asarray(lon, float), nan=0.0)
        r = np.clip(np.floor((90.0 - lat) / RES).astype(int), 0, 4319)
        c = np.clip(np.floor((lon + 180.0) / RES).astype(int), 0, 8639)
        return r, c

    def sample(self, lat, lon, fallback_deg: float = 0.25):
        """返回 dict{bio: 值数组} 与 dist 数组（用了回退时到最近有效像元的距离，像元数；0 为直接命中；nan 为找不到）。"""
        lat = np.asarray(lat, float)
        lon = np.asarray(lon, float)
        n = len(lat)
        r, c = self.rowcol(lat, lon)
        ok = np.isfinite(lat) & np.isfinite(lon)
        dist = np.full(n, np.nan)
        rr, cc = r.copy(), c.copy()
        w = int(round(fallback_deg / RES))
        for i in np.flatnonzero(ok):
            if self.valid[r[i], c[i]]:
                dist[i] = 0.0
                continue
            best = None
            for radius in range(1, w + 1):
                r0, r1 = max(r[i] - radius, 0), min(r[i] + radius, 4319)
                c0, c1 = c[i] - radius, c[i] + radius
                cols = np.arange(c0, c1 + 1) % 8640       # 经度回绕
                sub = self.valid[r0:r1 + 1][:, cols]
                if sub.any():
                    ii, jj = np.nonzero(sub)
                    d = np.hypot(ii + r0 - r[i], jj + c0 - c[i])
                    k = int(np.argmin(d))
                    best = (ii[k] + r0, cols[jj[k]], float(d[k]))
                    break
            if best is not None:
                rr[i], cc[i], dist[i] = best
        out = {}
        for k in self.bios:
            v = np.full(n, np.nan, dtype=np.float32)
            hit = np.isfinite(dist)
            v[hit] = self.grid[k][rr[hit], cc[hit]]
            out[BIO_NAMES[k]] = v
        return out, dist

    def box_mean(self, lat_min, lat_max, lon_c, half_width_deg):
        """纬度 [lat_min, lat_max]、经度 lon_c ± half_width_deg 内有效像元均值；返回 dict{bio: 值} 与像元数。"""
        if not all(np.isfinite([lat_min, lat_max, lon_c, half_width_deg])):
            return {BIO_NAMES[k]: np.nan for k in self.bios}, 0
        r_top, _ = self.rowcol(max(lat_min, lat_max), 0.0)
        r_bot, _ = self.rowcol(min(lat_min, lat_max), 0.0)
        half = int(np.ceil(min(half_width_deg, 180.0) / RES))
        _, c0 = self.rowcol(0.0, lon_c)
        cols = np.arange(c0 - half, c0 + half + 1) % 8640
        m = self.valid[r_top:r_bot + 1][:, cols]
        n = int(m.sum())
        if n == 0:
            return {BIO_NAMES[k]: np.nan for k in self.bios}, 0
        return {BIO_NAMES[k]: float(np.nanmean(self.grid[k][r_top:r_bot + 1][:, cols][m])) for k in self.bios}, n
