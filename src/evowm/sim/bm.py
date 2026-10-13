"""树上的布朗运动模拟（冒烟测试用）。恢复检验用的 OU 加习性体制模拟另写。"""
from __future__ import annotations

import numpy as np

from ..data.tree import Tree


def simulate_bm(tree: Tree, n_traits: int, rate: float, rng: np.random.Generator) -> np.ndarray:
    """沿树做布朗运动，返回 (n_tips, n_traits)，行序同 tree.tip_names。

    每条枝上的增量 ~ N(0, rate * 枝长)。节点按先序编号，父节点先算。
    """
    n = len(tree.parent)
    blen = np.zeros(n)
    nonroot = tree.parent >= 0
    blen[nonroot] = tree.depth[nonroot] - tree.depth[tree.parent[nonroot]]
    noise = rng.normal(size=(n, n_traits)) * np.sqrt(rate * blen)[:, None]
    x = np.zeros((n, n_traits))
    for v in range(1, n):
        x[v] = x[tree.parent[v]] + noise[v]
    return x[tree.tip_nodes]
