"""探索性诊断（事后，不属于任何预注册）：真实数据主分析 v1 第 0 阶段的模拟表（v5，有早期爆发背景）上，线性预测器的 E1。

三种"有习性"的用法，都与无习性的线性预测 r̂ = μ + exp(−αd)(r_A − μ) 比较：
  hab  ：现有线性基线 r̂ = μ_c(B) + exp(−αd)(r_A − μ_c(A))，即把习性当作"水平"（训练科的类别均值）；
  diff ：无习性预测 + κ(μ_c(B) − μ_c(A))，只在近亲之间习性变化时起作用；
  odiff：同上，但用真值偏移 κ(θ_B − θ_A)，即习性变化信息的上限。
α、κ 在训练科的窗口上用最小二乘拟合（κ 限制在 [0, 1]）。第 21 棵树，按科 5 折，留出科窗口取法与世界模型的 E1 相同（H = 3）。
在 EvoWM 根目录运行（需要 data/processed/sim/recovery_v5/ 的表）：python scripts/exploratory/diag_stage0_linear_v1.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from evowm.eval.pooled import boot_gain  # noqa: E402
from evowm.eval.recovery import sample_windows  # noqa: E402
from evowm.eval.runs import fold_datasets  # noqa: E402

H = 3


def flat(W):
    return W.rows[:, :H].reshape(-1), W.rows[:, 1:H + 1].reshape(-1), W.dist[:, :H].reshape(-1)


def run(sc, s, tree):
    rel = f"data/processed/sim/recovery_v5/{sc}_s{s}/species_traits.csv"
    table = pd.read_csv(ROOT / rel)
    truth = json.loads((ROOT / rel).with_name("truth.json").read_text(encoding="utf-8"))
    off = truth["leg_offsets"]["L_tmt"]
    reg = table["sim_regime"].to_numpy()
    theta = np.array([off.get(x, 0.0) if isinstance(x, str) else 0.0 for x in reg])
    fam = table["family"].to_numpy()
    out = {k: [] for k in ("no", "hab", "diff", "odiff", "fam", "trans")}
    for k in range(5):
        ds_tr, ds_va = fold_datasets(ROOT, rel, f"data/processed/trees/hackett_t{tree:03d}.nwk", k)
        S, cols = ds_tr.S, ds_tr.state_cols
        t_, m_ = S[:, cols.index("m_tarsus")].astype(float), S[:, cols.index("m_mass")].astype(float)
        tr = np.flatnonzero(ds_tr.rows_mask)
        ok = tr[np.isfinite(t_[tr]) & np.isfinite(m_[tr])]
        slope = np.polyfit(m_[ok], t_[ok], 1)[0]
        r = t_ - slope * m_
        mu = float(np.nanmean(r[ok]))
        mu_c = {c: float(np.nanmean(r[ok][reg[ok] == c])) for c in np.unique(reg[ok].astype(str))}
        mc = np.array([mu_c.get(str(x), mu) for x in reg])

        def pred_no(a, d, al):
            return mu + np.exp(-al * d) * (r[a] - mu)

        def pred_hab(a, b, d, al):
            return mc[b] + np.exp(-al * d) * (r[a] - mc[a])

        a_t, b_t, d_t = flat(sample_windows(ds_tr, 8, H + 1, s))

        def fit(f):
            o = minimize_scalar(lambda la: np.nanmean((f(np.exp(la)) - r[b_t]) ** 2), bounds=(np.log(1e-4), np.log(2.0)),
                                method="bounded")
            return float(np.exp(o.x))

        al_no = fit(lambda al: pred_no(a_t, d_t, al))
        al_h = fit(lambda al: pred_hab(a_t, b_t, d_t, al))
        res_t = r[b_t] - pred_no(a_t, d_t, al_no)

        def kappa(delta):
            m = np.isfinite(res_t) & (delta != 0)
            return float(np.clip(np.sum(res_t[m] * delta[m]) / max(np.sum(delta[m] ** 2), 1e-12), 0, 1))

        k_d = kappa(mc[b_t] - mc[a_t])
        k_o = kappa(theta[b_t] - theta[a_t])
        a, b, d = flat(sample_windows(ds_va, 8, H + 1, s))
        base = pred_no(a, d, al_no)
        out["no"].append(base - r[b])
        out["hab"].append(pred_hab(a, b, d, al_h) - r[b])
        out["diff"].append(base + k_d * (mc[b] - mc[a]) - r[b])
        out["odiff"].append(base + k_o * (theta[b] - theta[a]) - r[b])
        out["fam"].append(fam[b])
        out["trans"].append(reg[a] != reg[b])
    o = {k: np.concatenate(v) for k, v in out.items()}
    print(f"{sc}_s{s} 树 {tree}，习性变化对占 {np.mean(o['trans']):.1%}", flush=True)
    for name in ("hab", "diff", "odiff"):
        g = boot_gain(o[name], o["no"], o["fam"], 2000, 0)
        tm = o["trans"]
        gt = boot_gain(np.where(tm, o[name], np.nan), np.where(tm, o["no"], np.nan), o["fam"], 2000, 0)
        print(f"    {name:5s} {g['gain']:+.4f} [{g['ci95'][0]:+.4f}, {g['ci95'][1]:+.4f}]，变化对 {gt['gain']:+.4f}", flush=True)


if __name__ == "__main__":
    for s in (9, 10):
        for sc in ("effect", "null"):
            run(sc, s, 21)
