"""树上的多性状 OU 模拟，最优值随离散体制变化（恢复检验用）。

每条枝取子节点的体制 r；沿长度 t 的枝：
    x_child = θ_r + (x_parent − θ_r)·e^{−αt} + N(0, σ²_stat·(1 − e^{−2αt}))
根节点从根体制的平稳分布抽取。α 与平稳方差对所有性状相同，最优值 θ 按性状与体制给定。
"""
from __future__ import annotations

import numpy as np

from ..data.tree import Tree


def simulate_ou_regimes(
    tree: Tree,
    node_regime: np.ndarray,
    theta: np.ndarray,
    alpha: float,
    sd_stat: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """返回所有节点的性状值 (n_nodes, n_traits)。

    node_regime: (n_nodes,) 体制编号；theta: (n_regimes, n_traits)；sd_stat: (n_traits,)。
    """
    n = len(tree.parent)
    k = theta.shape[1]
    x = np.zeros((n, k))
    r0 = node_regime[tree.root]
    x[tree.root] = theta[r0] + rng.normal(size=k) * sd_stat
    blen = np.zeros(n)
    nonroot = tree.parent >= 0
    blen[nonroot] = tree.depth[nonroot] - tree.depth[tree.parent[nonroot]]
    decay = np.exp(-alpha * blen)
    sd_step = np.sqrt(1.0 - decay ** 2)[:, None] * sd_stat[None, :]
    noise = rng.normal(size=(n, k)) * sd_step
    for v in range(1, n):                  # 先序编号：父节点先算
        th = theta[node_regime[v]]
        x[v] = th + (x[tree.parent[v]] - th) * decay[v] + noise[v]
    return x


def simulate_bm_nodes(tree: Tree, rate: float, root_value: float, rng: np.random.Generator) -> np.ndarray:
    """单性状布朗运动，返回所有节点的值 (n_nodes,)。"""
    n = len(tree.parent)
    blen = np.zeros(n)
    nonroot = tree.parent >= 0
    blen[nonroot] = tree.depth[nonroot] - tree.depth[tree.parent[nonroot]]
    steps = rng.normal(size=n) * np.sqrt(rate * blen)
    x = np.zeros(n)
    x[tree.root] = root_value
    for v in range(1, n):
        x[v] = x[tree.parent[v]] + steps[v]
    return x


def simulate_eb_nodes(tree: Tree, rate0: float, b: float, rng: np.random.Generator) -> np.ndarray:
    """单性状"早期爆发"布朗运动（根处速率 rate0，速率随距根时间 t 按 e^{b t} 变化；b = 0 即普通布朗运动），根值 0。
    返回所有节点的值 (n_nodes,)。一条从 t1 到 t2 的枝的方差为 rate0 (e^{b t2} − e^{b t1}) / b。"""
    n = len(tree.parent)
    var = np.zeros(n)
    nonroot = tree.parent >= 0
    t1, t2 = tree.depth[tree.parent[nonroot]], tree.depth[nonroot]
    var[nonroot] = rate0 * ((np.exp(b * t2) - np.exp(b * t1)) / b if b != 0 else (t2 - t1))
    steps = rng.normal(size=n) * np.sqrt(var)
    x = np.zeros(n)
    for v in range(1, n):
        x[v] = x[tree.parent[v]] + steps[v]
    return x
