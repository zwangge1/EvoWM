"""事后诊断（看到 v2 种子 3 的 E1 结果后写，不参与判定）：E1 在每个模拟种子上有多少可检出的信号。

用真值参数构造只看前一种的理想预测器（与 evaluate_run 里 E1_transition 的理想预测相同），
在与 E1 完全相同的留出科窗口上，计算它在全部相邻对上的习性增益及按科自助 95% 置信区间。
理想预测器的增益置信区间若含 0，说明该种子下 E1 本身没有检验力，模型增益不显著不能归因于模型。

在 EvoWM 根目录运行：python scripts/exploratory/diag_e1_power_v2.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_recovery import datasets  # noqa: E402
from evowm.eval.recovery import family_bootstrap_rmse_diff, sample_windows  # noqa: E402

H = 3          # 与 recovery_v2_l1 的 history_size 相同
N_ORDERS = 8   # 与 eval_recovery_v2.py 相同


def oracle_all_pairs(sc: str, s: int, c: str = "m_tarsus") -> dict:
    rel = f"data/processed/sim/recovery_v2/{sc}_s{s}/species_traits.csv"
    table = pd.read_csv(ROOT / rel)
    truth = json.loads((ROOT / rel).with_name("truth.json").read_text(encoding="utf-8"))
    ds_tr, ds_va = datasets(rel, "data/raw/birdtree/hackett_tree1.nwk", 1)
    W = sample_windows(ds_va, N_ORDERS, H + 1, s)
    prev_rows, tgt_rows = W.rows[:, :H], W.rows[:, 1:H + 1]
    reg = table["sim_regime"].to_numpy()
    fam = table["family"].to_numpy()[tgt_rows].reshape(-1)
    trans = (reg[prev_rows] != reg[tgt_rows]).reshape(-1)
    alpha, slope = float(truth["alpha"]), float(truth.get("allometric_slope", 0.33))
    mass = table["m_mass"].to_numpy()
    k = np.exp(-alpha * W.dist[:, :H]).reshape(-1)
    off = truth["leg_offsets"][truth["leg_source"][c]]
    th = np.array([off.get(r, 0.0) if isinstance(r, str) else 0.0 for r in reg])
    tbar = float(np.mean(th[np.flatnonzero(ds_tr.rows_mask)]))
    x = table[c].to_numpy()
    xA, yB = x[prev_rows].reshape(-1), x[tgt_rows].reshape(-1)
    base = 1.0 + slope * mass[prev_rows].reshape(-1)
    rA = xA - base
    thA, thB = th[prev_rows].reshape(-1), th[tgt_rows].reshape(-1)
    err_w = base + thB + k * (rA - thA) - yB
    err_o = base + tbar + k * (rA - tbar) - yB
    g, lo, hi, n, nf = family_bootstrap_rmse_diff(err_w, err_o, fam, seed=s)
    ok = np.isfinite(err_w) & np.isfinite(err_o)
    fam_trans = len(np.unique(fam[ok & trans]))
    return {"oracle_gain_all": g, "oracle_ci95_all": [lo, hi], "n_pred": n, "n_families": nf,
            "share_transition": float(trans[ok].mean()), "families_with_transition": int(fam_trans)}


def main():
    out = {f"{sc}_s{s}": oracle_all_pairs(sc, s) for sc in ("effect", "null") for s in (3, 4)}
    p = ROOT / "outputs/recovery_v2/diag_e1_power.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    for k, v in out.items():
        print(k, f"{v['oracle_gain_all']:+.4f} [{v['oracle_ci95_all'][0]:+.4f}, {v['oracle_ci95_all'][1]:+.4f}]",
              f"转变对占比 {v['share_transition']:.3f}，含转变对的科 {v['families_with_transition']}/{v['n_families']}")


if __name__ == "__main__":
    main()
