"""系统发育协方差矩阵、Pagel λ 的 PGLS 与平稳 OU 回归（最大似然），真实数据主分析 v1 的基线用。"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize_scalar

from evowm.data.tree import Tree


def tip_vcv(tree: Tree) -> np.ndarray:
    """叶间共享枝长矩阵 C（行列按 Newick 叶序，即 tree.tip_names 的顺序）：C[i, j] = 根到 LCA(i, j) 的距离。

    Newick 是先序序列，所以每个类群的叶在这个顺序里是连续的一段；每对叶只在它们的 LCA 处写一次。
    """
    n_nodes, n = len(tree.parent), tree.n_tips
    pos = np.full(n_nodes, -1)
    pos[tree.tip_nodes] = np.arange(n)
    lo = np.full(n_nodes, np.iinfo(np.int64).max)
    hi = np.full(n_nodes, -1)
    lo[tree.tip_nodes] = pos[tree.tip_nodes]
    hi[tree.tip_nodes] = pos[tree.tip_nodes] + 1
    for v in range(n_nodes - 1, 0, -1):   # 子节点编号大于父节点
        p = tree.parent[v]
        lo[p] = min(lo[p], lo[v])
        hi[p] = max(hi[p], hi[v])
    C = np.empty((n, n))
    for v in range(n_nodes):
        ch = tree.children(v)
        if len(ch) == 0:
            continue
        d, a, b = tree.depth[v], lo[v], hi[v]
        for c in ch:
            C[lo[c]:hi[c], a:lo[c]] = d
            C[lo[c]:hi[c], hi[c]:b] = d
    C[np.arange(n), np.arange(n)] = tree.depth[tree.tip_nodes]
    return C


def pgls_lambda(y: np.ndarray, X: np.ndarray, C: np.ndarray, names: list[str] | None = None) -> dict:
    """y = Xβ + e，e ~ N(0, σ² V(λ))，V(λ) = λ C + (1 − λ) diag(C)。λ 在 [0, 1] 上取最大似然。"""
    n, p = X.shape
    diag = np.diag(C).copy()

    def fit(lam: float):
        V = C * lam
        V[np.arange(n), np.arange(n)] = diag
        L = cholesky(V, lower=True, overwrite_a=True, check_finite=False)
        z = solve_triangular(L, y, lower=True, check_finite=False)
        Z = solve_triangular(L, X, lower=True, check_finite=False)
        beta, *_ = np.linalg.lstsq(Z, z, rcond=None)
        r = z - Z @ beta
        s2 = float(r @ r) / n
        loglik = -0.5 * n * (np.log(2 * np.pi * s2) + 1) - float(np.sum(np.log(np.diag(L))))
        return loglik, beta, Z, r

    opt = minimize_scalar(lambda l: -fit(l)[0], bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-3})
    lam = float(opt.x)
    loglik, beta, Z, r = fit(lam)
    s2 = float(r @ r) / (n - p)
    se = np.sqrt(np.diag(np.linalg.inv(Z.T @ Z)) * s2)
    names = names or [f"x{i}" for i in range(p)]
    return {"lambda": lam, "loglik": float(loglik), "n": int(n),
            "coef": {k: {"beta": float(b), "se": float(e)} for k, b, e in zip(names, beta, se)}}


def ou_stationary(y: np.ndarray, X: np.ndarray, C: np.ndarray, names: list[str] | None = None,
                  q_grid: tuple = (0.0, 0.02, 0.05, 0.1, 0.2)) -> dict:
    """y = Xβ + e，e ~ N(0, s² [(1 − q) exp(−α D) + q I])，D 为叶间距离（由 C 算出，超度量树上即平稳 OU 的协方差）。
    q 为测量误差占比，在网格上取最大似然；α 在 [1e-4, 1] 上取最大似然（对数尺度）。半衰期 = ln 2 / α。"""
    n, p = X.shape
    dg = np.diag(C)
    D = dg[:, None] + dg[None, :] - 2 * C

    def fit(alpha: float, q: float):
        V = np.exp(-alpha * D) * (1 - q)
        V[np.arange(n), np.arange(n)] += q
        L = cholesky(V, lower=True, overwrite_a=True, check_finite=False)
        z = solve_triangular(L, y, lower=True, check_finite=False)
        Z = solve_triangular(L, X, lower=True, check_finite=False)
        beta, *_ = np.linalg.lstsq(Z, z, rcond=None)
        r = z - Z @ beta
        s2 = float(r @ r) / n
        loglik = -0.5 * n * (np.log(2 * np.pi * s2) + 1) - float(np.sum(np.log(np.diag(L))))
        return loglik, beta, s2

    best = None
    for q in q_grid:
        opt = minimize_scalar(lambda la: -fit(np.exp(la), q)[0], bounds=(np.log(1e-4), 0.0), method="bounded",
                              options={"xatol": 1e-3})
        if best is None or -opt.fun > best[0]:
            best = (-opt.fun, float(np.exp(opt.x)), q)
    loglik, alpha, q = best
    _, beta, s2 = fit(alpha, q)
    names = names or [f"x{i}" for i in range(p)]
    return {"alpha": alpha, "t_half": float(np.log(2) / alpha), "sd_stat": float(np.sqrt(s2 * (1 - q))), "q_me": q,
            "loglik": float(loglik), "n": int(n), "coef": {k: float(b) for k, b in zip(names, beta)}}
