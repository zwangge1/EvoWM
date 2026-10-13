"""离散"体制"（习性类别）的定义与沿树重建，供 OU 模拟使用。

- 体制变量从物种主表的习性列派生（只用习性，不看任何形态值）。
- 祖先体制用 Fitch 简约法重建：先自下而上求候选集合，再自上而下定状态；
  平局按固定随机种子打破，保证可复现。缺失物种的叶候选集为全集。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.tree import Tree

LIFESTYLE = ["aerial", "aquatic", "generalist", "insessorial", "terrestrial"]
FORSTRAT = ["watbelowsurf", "wataroundsurf", "ground", "understory", "midhigh", "canopy", "aerial"]


def regime_lifestyle(table: pd.DataFrame) -> pd.Series:
    """AVONET Primary.Lifestyle（觅食时的主要运动方式）。"""
    cols = [f"e_life_{c}" for c in LIFESTYLE]
    v = table[cols].to_numpy(float)
    out = pd.Series(pd.NA, index=table.index, dtype="object")
    ok = np.isfinite(v).all(axis=1) & (np.nansum(v, axis=1) > 0)
    out[ok] = np.array(LIFESTYLE)[np.nanargmax(v[ok], axis=1)]
    return out


def regime_forstrat(table: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
    """EltonTraits 主导觅食层：占比 ≥ threshold 的层，否则为 mixed。"""
    cols = [f"e_fs_{c}" for c in FORSTRAT]
    v = table[cols].to_numpy(float)
    out = pd.Series(pd.NA, index=table.index, dtype="object")
    ok = np.isfinite(v).all(axis=1)
    top = np.nanargmax(np.where(np.isfinite(v), v, -1), axis=1)
    lab = np.array(FORSTRAT)[top]
    lab = np.where(v[np.arange(len(v)), top] >= threshold, lab, "mixed")
    out[ok] = lab[ok]
    return out


def regime_locomotor(table: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
    """运动方式细分：生活方式为主，用觅食层把涉水与地面、潜水与水面分开。"""
    life = regime_lifestyle(table)
    fs = table[[f"e_fs_{c}" for c in FORSTRAT]]
    water = fs["e_fs_watbelowsurf"].fillna(0) + fs["e_fs_wataroundsurf"].fillna(0)
    out = life.copy()
    out[(life == "terrestrial") & (water >= threshold)] = "wading"
    out[(life == "aquatic") & (fs["e_fs_watbelowsurf"].fillna(0) >= threshold)] = "diving"
    out[(life == "aquatic") & (out == "aquatic")] = "surface"
    return out


REGIMES = {"lifestyle": regime_lifestyle, "forstrat": regime_forstrat, "locomotor": regime_locomotor}


def fitch(tree: Tree, tip_states: np.ndarray, n_states: int, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """Fitch 简约重建。tip_states 按 tree.tip_names 顺序，-1 表示缺失。

    返回 (node_state, n_changes)：每个节点的状态（0..n_states-1）与最少状态变化次数。
    """
    n = len(tree.parent)
    full = (1 << n_states) - 1
    sets = np.zeros(n, dtype=np.int64)
    for k, v in enumerate(tree.tip_nodes):
        s = tip_states[k]
        sets[v] = full if s < 0 else (1 << int(s))
    changes = 0
    for v in range(n - 1, -1, -1):          # 先序编号的逆序 = 子节点先于父节点
        a, b = tree.child_ptr[v], tree.child_ptr[v + 1]
        if a == b:
            continue
        ch = tree.child_idx[a:b]
        inter, union = full, 0
        for c in ch:
            inter &= int(sets[c])
            union |= int(sets[c])
        if inter:
            sets[v] = inter
        else:
            sets[v] = union
            changes += 1                     # 多叉时为近似下界，足够用来比较变量
    state = np.full(n, -1, dtype=np.int64)
    for v in range(n):                       # 先序：父节点先定
        s = int(sets[v])
        p = tree.parent[v]
        if p >= 0 and (s >> int(state[p])) & 1:
            state[v] = state[p]
        else:
            opts = [i for i in range(n_states) if (s >> i) & 1]
            state[v] = opts[0] if len(opts) == 1 else int(rng.choice(opts))
    return state, changes
