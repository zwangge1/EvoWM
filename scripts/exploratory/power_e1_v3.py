"""v3 设计用的检验力估计（不训练模型）：理想预测器的 E1 增益，在多少套模拟数据里能被检出。

比较两种留出设计，窗口取法与评估相同（每份留出集 8 个随机叶序，窗口长 4，history 3）：
- 现行：按科留出 15%（38 个科，与 v1、v2 相同）。
- 5 折：按科 5 折交叉验证，5 份的预测合并后，按全部科自助重抽样求置信区间。
用额外的模拟种子 101 到 120，只看效应情景（零模型下理想预测器的增益恒为 0）。
检出 = m_tarsus 增益的 95% 置信区间下限大于 0。
这些种子不训练任何模型，只用来估计设计的检验力，不参与 v3 的判定。

在 EvoWM 根目录运行：python scripts/exploratory/power_e1_v3.py
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from evowm.data.transect import TreeTransectDataset  # noqa: E402
from evowm.eval.recovery import family_bootstrap_rmse_diff, oracle_errors, sample_windows  # noqa: E402

SEEDS = list(range(101, 121))
H, N_ORDERS = 3, 8
TREE = "data/raw/birdtree/hackett_tree1.nwk"
DESIGNS = {
    "holdout15": [{"level": "family", "frac": 0.15, "seed": 0}],
    "cv5": [{"level": "family", "frac": 0, "folds": 5, "fold": k, "seed": 0} for k in range(5)],
}


def oracle_pooled(rel: str, table: pd.DataFrame, truth: dict, holdouts: list, seed: int, c: str = "m_tarsus") -> dict:
    ew, eo, fam = [], [], []
    for h in holdouts:
        kw = dict(species_table=rel, tree_file=TREE, level1={"frameskip": 1, "num_steps": 4},
                  evowm_root=str(ROOT), n_trees=1, holdout=h, n_orders=2)
        tr, va = TreeTransectDataset(name="evowm_train", **kw), TreeTransectDataset(name="evowm_val", **kw)
        W = sample_windows(va, N_ORDERS, H + 1, seed)
        tgt = W.rows[:, 1:H + 1]
        y = tr.S[tgt][..., tr.state_cols.index(c)]
        w, o = oracle_errors(table, truth, np.flatnonzero(tr.rows_mask), W, H, c, y)
        ew.append(w)
        eo.append(o)
        fam.append(table["family"].to_numpy()[tgt].reshape(-1))
    g, lo, hi, n, nf = family_bootstrap_rmse_diff(np.concatenate(ew), np.concatenate(eo), np.concatenate(fam), seed=seed)
    return {"gain": g, "ci95": [lo, hi], "n_pred": n, "n_families": nf}


def main():
    subprocess.run([sys.executable, str(ROOT / "scripts/sim_recovery_tables.py"), "--version", "power",
                    "--seeds", *map(str, SEEDS)], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    out = {}
    for s in SEEDS:
        rel = f"data/processed/sim/recovery_power/effect_s{s}/species_traits.csv"
        table = pd.read_csv(ROOT / rel)
        truth = json.loads((ROOT / rel).with_name("truth.json").read_text(encoding="utf-8"))
        out[s] = {d: oracle_pooled(rel, table, truth, h, s) for d, h in DESIGNS.items()}
        print(s, "  ".join(f"{d} {r['gain']:+.4f} [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] 科 {r['n_families']}"
                           for d, r in out[s].items()), flush=True)
    summ = {}
    for d in DESIGNS:
        g = np.array([out[s][d]["gain"] for s in SEEDS])
        lo = np.array([out[s][d]["ci95"][0] for s in SEEDS])
        hi = np.array([out[s][d]["ci95"][1] for s in SEEDS])
        summ[d] = {"power": float(np.mean(lo > 0)), "gain_median": float(np.median(g)),
                   "ci_width_median": float(np.median(hi - lo)), "n_seeds": len(SEEDS)}
    p = ROOT / "outputs/recovery_v3/power_e1.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"summary": summ, "per_seed": out}, ensure_ascii=False, indent=2), encoding="utf-8")
    for d, v in summ.items():
        print(f"{d}: 检出比例 {v['power']:.2f}，增益中位数 {v['gain_median']:+.4f}，置信区间宽度中位数 {v['ci_width_median']:.4f}")


if __name__ == "__main__":
    main()
