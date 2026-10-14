"""多次运行（多棵树、多个训练种子、多折）合并后的簇自助重抽样、单侧 p 值与 Holm 校正。

真实数据主分析 v1 与第 0 阶段校准用（预注册 docs/prereg/real_main_v1.md）。簇通常是科或目；同一个簇在所有运行里的
预测一起被抽中或不抽中，因此运行之间的相关不会被当成独立信息。
"""
from __future__ import annotations

import numpy as np


def _index(clusters: np.ndarray, ok: np.ndarray):
    uniq, inv = np.unique(np.asarray(clusters, dtype=object)[ok].astype(str), return_inverse=True)
    return uniq, inv


def _draws(n_clusters: int, n_boot: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n_clusters, size=(n_boot, n_clusters))


def boot_gain(err_eco, err_no, clusters, n_boot: int = 10000, seed: int = 0) -> dict:
    """增益 = RMSE(无习性) − RMSE(有习性)，按簇自助重抽样。NaN 的预测不计入。"""
    ea, en = np.asarray(err_eco, float), np.asarray(err_no, float)
    ok = np.isfinite(ea) & np.isfinite(en)
    uniq, inv = _index(clusters, ok)
    sa = np.bincount(inv, ea[ok] ** 2, len(uniq))
    sn = np.bincount(inv, en[ok] ** 2, len(uniq))
    cnt = np.bincount(inv, minlength=len(uniq)).astype(float)
    point = float(np.sqrt(sn.sum() / cnt.sum()) - np.sqrt(sa.sum() / cnt.sum()))
    K = _draws(len(uniq), n_boot, seed)
    c = cnt[K].sum(1)
    boot = np.sqrt(sn[K].sum(1) / c) - np.sqrt(sa[K].sum(1) / c)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"gain": point, "ci95": [float(lo), float(hi)], "n_pred": int(ok.sum()), "n_clusters": int(len(uniq)),
            "boot": boot}


def boot_mean(values, clusters, n_boot: int = 10000, seed: int = 0) -> dict:
    """逐窗口数值的均值（例如 E2 的逐窗口对比），按簇自助重抽样。"""
    v = np.asarray(values, float)
    ok = np.isfinite(v)
    uniq, inv = _index(clusters, ok)
    s = np.bincount(inv, v[ok], len(uniq))
    cnt = np.bincount(inv, minlength=len(uniq)).astype(float)
    point = float(s.sum() / cnt.sum())
    K = _draws(len(uniq), n_boot, seed)
    boot = s[K].sum(1) / cnt[K].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean": point, "ci95": [float(lo), float(hi)], "n": int(ok.sum()), "n_clusters": int(len(uniq)), "boot": boot}


def boot_gain_diff(a_eco, a_no, b_eco, b_no, clusters, n_boot: int = 10000, seed: int = 0) -> dict:
    """两组预测（同一批窗口）的增益之差（A − B），同一次重抽样里一起计算。"""
    arrs = [np.asarray(x, float) for x in (a_eco, a_no, b_eco, b_no)]
    ok = np.logical_and.reduce([np.isfinite(x) for x in arrs])
    uniq, inv = _index(clusters, ok)
    sums = [np.bincount(inv, x[ok] ** 2, len(uniq)) for x in arrs]
    cnt = np.bincount(inv, minlength=len(uniq)).astype(float)

    def gain(s_eco, s_no, c):
        return np.sqrt(s_no / c) - np.sqrt(s_eco / c)

    point = float(gain(sums[0].sum(), sums[1].sum(), cnt.sum()) - gain(sums[2].sum(), sums[3].sum(), cnt.sum()))
    K = _draws(len(uniq), n_boot, seed)
    c = cnt[K].sum(1)
    boot = gain(sums[0][K].sum(1), sums[1][K].sum(1), c) - gain(sums[2][K].sum(1), sums[3][K].sum(1), c)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"diff": point, "ci95": [float(lo), float(hi)], "n_pred": int(ok.sum()), "n_clusters": int(len(uniq))}


def p_one_sided(boot: np.ndarray, direction: int) -> float:
    """预期方向上的单侧自助 p 值：direction=+1 时为抽样值 ≤ 0 的比例，−1 时为 ≥ 0 的比例。"""
    boot = np.asarray(boot, float)
    frac = np.mean(boot <= 0) if direction > 0 else np.mean(boot >= 0)
    return float(max(frac, 1.0 / (len(boot) + 1)))


def holm(pvals: dict, alpha: float) -> dict:
    """Holm 逐步校正，返回每个假设是否拒绝零假设。"""
    names = sorted(pvals, key=lambda k: pvals[k])
    m = len(names)
    out, still = {}, True
    for i, k in enumerate(names):
        still = still and pvals[k] <= alpha / (m - i)
        out[k] = bool(still)
    return out


def strip_boot(d):
    """去掉结果里的自助抽样数组，便于写 json。"""
    if isinstance(d, dict):
        return {k: strip_boot(v) for k, v in d.items() if k != "boot"}
    if isinstance(d, list):
        return [strip_boot(x) for x in d]
    return d
