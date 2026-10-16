"""方向检验的读数（工程验证 v2，方案 1）。

对每条腿，与圆心腿（O）逐个（参数组，工况）配对，两者都完成这一步时算差值：
  F  峰值地面反力的相对变化（越低越好，"腿的安全"）；
  E  支撑期能量损失的变化（能量损失占触地动能的比例，越低越好，"经济性"）；
  V  前向速度保持的变化（离地 / 触地前向速度，越高越好，"稳定性"）。
先对每组参数在全部工况上取平均，再对参数组取平均；改善量 gain 统一取"越大越好"的符号。
区间：对参数组自助重抽样，同一次抽样用于所有腿（配对）。
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from evowm.engineering.shape import angle_between, percentile_of, steepest

METRICS = {  # 读数 → (列名, 变化的算法, 改善的符号)
    "F": ("peak_f", "rel", -1.0),
    "E": ("e_loss", "abs", -1.0),
    "V": ("vx_ratio", "abs", +1.0),
}


def per_param_gain(df: pd.DataFrame, leg: str, base: str, metric: str) -> np.ndarray:
    """每组参数的改善量（全部工况上平均，只用两条腿都完成这一步的配对）。返回按参数组编号排序的数组。"""
    col, how, sign = METRICS[metric]
    ok = df.fail.eq("") if "fail" in df else np.ones(len(df), bool)
    a = df[(df.leg == leg) & ok].set_index(["j", "scenario"])[col]
    b = df[(df.leg == base) & ok].set_index(["j", "scenario"])[col]
    a, b = a.align(b, join="inner")
    x = (a / b - 1.0) if how == "rel" else (a - b)
    g = sign * x.groupby(level="j").mean()
    n_param = int(df.j.max()) + 1
    out = np.full(n_param, np.nan)
    out[g.index.to_numpy()] = g.to_numpy()
    return out


def gain_table(df: pd.DataFrame, legs, base: str, metric: str) -> dict[str, np.ndarray]:
    return {n: per_param_gain(df, n, base, metric) for n in legs if n != base}


def boot_means(G: dict[str, np.ndarray], n_boot: int = 2000, seed: int = 0) -> dict:
    """每条腿改善量的平均与自助区间；同一次抽样用于所有腿。缺失（该组参数没有有效配对）的参数组跳过。"""
    n = len(next(iter(G.values())))
    rng = np.random.default_rng(seed)
    K = rng.integers(0, n, size=(n_boot, n))
    out = {}
    for k, v in G.items():
        with warnings.catch_warnings():       # 某次抽样全是缺失的参数组时，nanmean 给 NaN 并告警；NaN 由 nanpercentile 跳过
            warnings.simplefilter("ignore", RuntimeWarning)
            bs = np.nanmean(v[K], axis=1)
        out[k] = {"mean": float(np.nanmean(v)), "ci95": [float(x) for x in np.nanpercentile(bs, [2.5, 97.5])],
                  "boot": bs}
    return out


def ring_summary(B: dict, ring_names: list[str], ring_angles: np.ndarray, probes: dict) -> dict:
    """方向环：余弦拟合的最陡方向与幅度（点估计与自助区间），探针方向（P、T）的改善、百分位、
    与最陡方向的夹角、占环上最大改善的比例（效率）及其区间。probes: 名字 → (腿名, 方向角)。"""
    gains = np.array([B[n]["mean"] for n in ring_names])
    st = steepest(ring_angles, gains)
    boot = np.array([B[n]["boot"] for n in ring_names])          # [方向, 抽样]
    th_b = np.array([steepest(ring_angles, boot[:, i])["theta"] for i in range(boot.shape[1])])
    dth = np.array([angle_between(t, st["theta"]) for t in th_b])
    out = {"steepest_deg": float(np.degrees(st["theta"])), "steepest_ci_halfwidth_deg": float(np.degrees(np.percentile(dth, 95))),
           "amplitude": st["amplitude"], "offset": st["offset"], "r2": st["r2"],
           "max_gain": float(gains.max()), "min_gain": float(gains.min()),
           "gains": {int(round(np.degrees(a))) % 360: float(g) for a, g in zip(ring_angles, gains)}}
    mx_b = boot.max(axis=0)
    for key, (leg, theta) in probes.items():
        g, gb = B[leg]["mean"], B[leg]["boot"]
        eff_b = gb / mx_b
        out[key] = {"gain": g, "ci95": B[leg]["ci95"], "percentile": percentile_of(g, gains),
                    "angle_to_steepest_deg": float(np.degrees(angle_between(theta, st["theta"]))),
                    "efficiency": float(g / gains.max()) if gains.max() > 0 else float("nan"),
                    "efficiency_ci95": [float(x) for x in np.nanpercentile(eff_b, [2.5, 97.5])]}
    return out
