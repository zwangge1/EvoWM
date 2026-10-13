"""开发用：评估单个检查点（模拟表 + 第 1 层读数），打印 E1/E2/E3 摘要并存 JSON。

用法：python scripts/eval_dev.py <检查点路径> <模拟表目录名，如 effect_s1> [输出 JSON 路径]
只用于开发集（模拟种子 1、2），不作为任何正式判据。
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_recovery import datasets  # noqa: E402
from evowm.eval.recovery import evaluate_run, load_world_model  # noqa: E402


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    tree1 = "--tree1" in sys.argv          # 只用真树（Hackett 第 1 棵）取样带
    ckpt, name = Path(args[0]), args[1]
    out = Path(args[2]) if len(args) > 2 else None
    torch.set_num_threads(1)
    t0 = time.time()
    ver = "v2" if name.endswith(("_s3", "_s4")) else "v1"
    rel = f"data/processed/sim/recovery_{ver}/{name}/species_traits.csv"
    table = pd.read_csv(ROOT / rel)
    truth = json.loads((ROOT / rel).with_name("truth.json").read_text(encoding="utf-8"))
    ds_tr, ds_va = datasets(rel, "data/raw/birdtree/hackett_tree1.nwk", 1) if tree1 else datasets(rel)
    model = load_world_model(ckpt, ROOT / "hjepa")
    seed = int(name.split("_s")[-1])
    r = evaluate_run(ds_tr, ds_va, model, table, truth, n_orders=8, seed=seed)
    e1 = " ".join(f"{c}:{v['gain']:+.4f}[{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]" for c, v in r["E1"].items())
    print(f"{ckpt.name} {name} ({time.time() - t0:.0f}s)")
    print(f"  E1 增益 {e1}")
    et = r["E1_transition"]["m_tarsus"]
    print(f"  E1 转变对 m_tarsus {et['gain']:+.4f}[{et['ci95'][0]:+.4f},{et['ci95'][1]:+.4f}] 理想 {et['oracle_gain']:+.4f} 比值 {et['ratio_to_oracle']:.2f}")
    contr = ", ".join("%s:%+.3f" % (k, v["m_tarsus"]) for k, v in r["E2"]["contrasts"].items())
    print(f"  E2 ρ={r['E2']['spearman_rho']:.3f} |c|中位={r['E2']['median_abs_contrast']:.4f}  m_tarsus 对比 {{{contr}}}")
    print(f"  E3 t½={r['E3']['t_half_free_amplitude']:.1f} 振幅={r['E3']['amplitude']:.2f} "
          f"斜率首末={r['E3']['binned_slope'][0]:.2f}/{r['E3']['binned_slope'][-1]:.2f}")
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
