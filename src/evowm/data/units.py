"""按习性事件分段的上层单位（设计草案 docs/design/第二层设计草案_2026-10-15.md 第 2 到 4 节）。

一条深度优先叶序上，连续的、习性类别相同的物种组成一个单位：
- locomotor（第 2 层）：运动方式 7 类（evowm.sim.regimes.regime_locomotor，只用习性列），与 H1 的"运动方式变化对"同一定义；
- realm（第 3 层）：栖境 3 类（陆、水、空），由运动方式合并而来，所以每个栖境单位由完整的运动方式单位组成。
叶序上缺失类别的物种沿用前一种的类别（叶序开头的沿用后一种），不打断单位。

单位一级的动作与第 1 层"只含变化"的动作同构（列的顺序也相同）：两单位平均习性之差（只在两边都有数据的列上）、
各习性组"两边都有数据"的标记、两单位成员之间的平均进化距离（对数与线性）。第 k 行是第 k 个单位到第 k+1 个单位的转变，
最后一行没有下一个单位，填 0。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LOCOMOTOR = ["insessorial", "terrestrial", "generalist", "aerial", "wading", "diving", "surface"]
REALM_OF = {"insessorial": "land", "terrestrial": "land", "generalist": "land",
            "wading": "water", "diving": "water", "surface": "water", "aerial": "air"}
REALMS = ["land", "water", "air"]
UNIT_LEVELS = ("locomotor", "realm")
D_MAX = 220.0  # 与第 1 层相同的距离缩放（transect.D_MAX）


def habit_codes(table: pd.DataFrame) -> np.ndarray:
    """每个物种的运动方式编号（LOCOMOTOR 的下标），缺失为 -1。只用习性列。"""
    from ..sim.regimes import regime_locomotor
    lab = regime_locomotor(table)
    return lab.map({r: i for i, r in enumerate(LOCOMOTOR)}).fillna(-1).astype(int).to_numpy()


REALM_CODE = np.array([REALMS.index(REALM_OF[r]) for r in LOCOMOTOR])


def fill_missing(codes: np.ndarray) -> np.ndarray:
    """叶序上的缺失（-1）沿用前一种的类别；开头的缺失沿用第一个有类别的物种；全缺失时记为 0。"""
    c = pd.Series(np.where(codes < 0, np.nan, codes)).ffill().bfill()
    return c.fillna(0).astype(int).to_numpy()


def runs(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """连续相同类别的段：返回各段在叶序上的起点与终点（不含）。"""
    brk = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    return np.r_[0, brk].astype(np.int64), np.r_[brk, len(codes)].astype(np.int64)


def _mean_min(x: np.ndarray, y: np.ndarray) -> float:
    """所有 (p, q) 组合上 min(x_p, y_q) 的平均（y 可含 inf）。排序加前缀和，O((n+m) log m)。"""
    ys = np.sort(y)
    cs = np.concatenate([[0.0], np.cumsum(np.where(np.isfinite(ys), ys, 0.0))])
    k = np.searchsorted(ys, x, side="left")          # 小于 x_p 的 y 个数（inf 不会被计入）
    total = cs[k] + (len(ys) - k) * x
    return float(total.sum() / (len(x) * len(ys)))


def adjacent_unit_distance(starts: np.ndarray, ends: np.ndarray, lca: np.ndarray, dtip: np.ndarray) -> np.ndarray:
    """相邻两单位全部成员两两之间的平均进化距离（Myr），长度 n_units - 1。

    叶序上 p < q 两叶的 LCA 深度 = 二者之间相邻 LCA 深度的最小值。单位 A = [a0, a1) 与紧随其后的 B = [a1, b1)：
    x_p = min(lca[p..a1-1])（含两单位交界），y_q = min(lca[a1..q-1])（q = a1 时为 inf），LCA 深度 = min(x_p, y_q)。
    距离 = 两叶到根的距离之和 − 2 × LCA 深度。"""
    out = np.empty(len(starts) - 1)
    for k in range(len(starts) - 1):
        a0, a1, b1 = starts[k], ends[k], ends[k + 1]
        x = np.minimum.accumulate(lca[a0:a1][::-1])[::-1]
        y = np.concatenate([[np.inf], np.minimum.accumulate(lca[a1:b1 - 1])]) if b1 - a1 > 1 else np.array([np.inf])
        out[k] = dtip[a0:a1].mean() + dtip[a1:b1].mean() - 2.0 * _mean_min(x, y)
    return out


def unit_habits(rows: np.ndarray, starts: np.ndarray, E: np.ndarray, cmask: np.ndarray, Eg: np.ndarray):
    """各单位的平均习性（每列只在有数据的成员上平均）、各列是否有数据、各习性组是否有成员有数据。

    rows：叶序上的物种表行号；E：物种习性（缺失置 0）；cmask：各列是否有数据（按习性组展开）；Eg：各习性组是否有数据。"""
    s = np.add.reduceat(E[rows] * cmask[rows], starts, axis=0)
    n = np.add.reduceat(cmask[rows], starts, axis=0)
    mean = s / np.maximum(n, 1.0)
    g_any = np.add.reduceat(Eg[rows], starts, axis=0) > 0
    return mean.astype(np.float32), (n > 0).astype(np.float32), g_any.astype(np.float32)


def unit_actions(mean: np.ndarray, col_ok: np.ndarray, g_any: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """单位一级的动作（只含变化），列的顺序与第 1 层 change 动作相同。最后一行填 0。"""
    both = col_ok[1:] * col_ok[:-1]
    act = np.concatenate([
        (mean[1:] - mean[:-1]) * both,
        g_any[1:] * g_any[:-1],
        np.log1p(dist)[:, None] / np.log1p(D_MAX),
        dist[:, None] / D_MAX,
    ], axis=1)
    return np.concatenate([act, np.zeros((1, act.shape[1]))], axis=0).astype(np.float32)


@dataclass
class UnitLevel:
    starts: np.ndarray        # (n_units,) 单位在叶序上的起点
    ends: np.ndarray          # (n_units,) 终点（不含）
    label: np.ndarray         # (n_units,) 类别编号（运动方式为 LOCOMOTOR 下标，栖境为 REALMS 下标）
    action: np.ndarray        # (n_units, action_dim) 第 k 行为第 k → k+1 个单位的转变，最后一行为 0
    dist: np.ndarray          # (n_units - 1,) 相邻单位成员间的平均进化距离（Myr）
    habit: np.ndarray | None = None   # (n_units, n_eco) 平均习性（评估的反事实用）
    col_ok: np.ndarray | None = None  # (n_units, n_eco) 各列是否有成员有数据
    g_any: np.ndarray | None = None   # (n_units, n_groups) 各习性组是否有成员有数据
    sub: np.ndarray | None = None   # 栖境层：(n_units, 2) 所含运动方式单位的下标范围 [k0, k1)

    def __len__(self) -> int:
        return len(self.starts)


def build_unit_levels(rows: np.ndarray, lca: np.ndarray, dtip: np.ndarray, loco: np.ndarray,
                      E: np.ndarray, cmask: np.ndarray, Eg: np.ndarray, levels) -> dict[str, UnitLevel]:
    """一条叶序上各层的单位。rows：叶序上的物种表行号；lca：相邻叶 LCA 深度；dtip：各叶到根的距离；
    loco：物种表每行的运动方式编号（-1 为缺失）。"""
    unknown = [lv for lv in levels if lv not in UNIT_LEVELS]
    if unknown:
        raise ValueError(f"未知的单位层：{unknown}，只能是 {UNIT_LEVELS}")
    code_l = fill_missing(loco[rows])
    out = {}
    for name in ("locomotor", "realm"):
        if name not in levels:
            continue
        codes = code_l if name == "locomotor" else REALM_CODE[code_l]
        s, e = runs(codes)
        mean, col_ok, g_any = unit_habits(rows, s, E, cmask, Eg)
        dist = adjacent_unit_distance(s, e, lca, dtip)
        out[name] = UnitLevel(s, e, codes[s], unit_actions(mean, col_ok, g_any, dist), dist, mean, col_ok, g_any)
    if "realm" in out:
        ls = runs(code_l)[0]
        r = out["realm"]
        r.sub = np.stack([np.searchsorted(ls, r.starts), np.searchsorted(ls, r.ends)], axis=1)
    return out
