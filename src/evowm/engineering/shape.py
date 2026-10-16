"""三段腿的比例平面、方向环与最陡方向（工程验证 v2）。

比例平面：三段 log10 长度去掉共同部分后的二维平面。坐标
  x（远端对近端）= (log10 跗跖 − log10 股) / √2，
  y（外侧对中段）= (log10 跗跖 + log10 股 − 2·log10 胫跗) / √6。
总长相同的腿只在这个平面上不同；三段同乘一个数，坐标不变。
"""
from __future__ import annotations

import numpy as np

from evowm.engineering.drop import Leg

# Leg(tmt, tbt, fem) 顺序下的两个单位向量，都与 (1, 1, 1) 正交
U1 = np.array([1.0, 0.0, -1.0]) / np.sqrt(2.0)
U2 = np.array([1.0, -2.0, 1.0]) / np.sqrt(6.0)


def log_lengths(leg: Leg) -> np.ndarray:
    return np.log10([leg.tmt, leg.tbt, leg.fem])


def shape_xy(leg: Leg, center: Leg | None = None) -> np.ndarray:
    """腿在比例平面上的坐标；给 center 时为相对 center 的坐标。"""
    v = log_lengths(leg) - (log_lengths(center) if center is not None else 0.0)
    return np.array([v @ U1, v @ U2])


def leg_at(center: Leg, xy) -> Leg:
    """相对 center 偏移 xy（比例平面坐标）的腿，总长与 center 相同。"""
    v = log_lengths(center) + xy[0] * U1 + xy[1] * U2
    return Leg(*(10.0 ** v)).scaled(center.total)


def ring(center: Leg, radius: float, n_dir: int = 24) -> dict[str, Leg]:
    """以 center 为圆心、半径 radius 的方向环：n_dir 个方向均匀分布，从远端对近端的正方向起逆时针。"""
    out = {}
    for i in range(n_dir):
        a = 2 * np.pi * i / n_dir
        out[f"ring{radius:.4f}_{i * 360 // n_dir:03d}"] = leg_at(center, radius * np.array([np.cos(a), np.sin(a)]))
    return out


def steepest(angles_rad: np.ndarray, gains: np.ndarray) -> dict:
    """对方向环上的改善量拟合 g(θ) = c + a·cos θ + b·sin θ（最小二乘），返回最陡方向与幅度。"""
    A = np.column_stack([np.ones_like(angles_rad), np.cos(angles_rad), np.sin(angles_rad)])
    coef, *_ = np.linalg.lstsq(A, gains, rcond=None)
    c, a, b = coef
    resid = gains - A @ coef
    ss_tot = float(np.sum((gains - gains.mean()) ** 2))
    return dict(theta=float(np.arctan2(b, a)), amplitude=float(np.hypot(a, b)), offset=float(c),
                r2=float(1 - np.sum(resid ** 2) / ss_tot) if ss_tot > 0 else float("nan"))


def angle_between(a: float, b: float) -> float:
    """两个方向的夹角（0 到 π）。"""
    d = (a - b + np.pi) % (2 * np.pi) - np.pi
    return float(abs(d))


def percentile_of(value: float, ring_values: np.ndarray) -> float:
    """value 在方向环取值中的百分位：严格小于它的比例，相等的算一半。"""
    r = np.asarray(ring_values, float)
    return float((np.sum(r < value) + 0.5 * np.sum(r == value)) / len(r))
