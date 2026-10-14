"""评估模拟恢复检验 v1 的 4 个模型，对照 docs/prereg/sim_recovery_v1.md 的判据出报告。

在 EvoWM 根目录运行：python scripts/eval_recovery.py
读取 outputs/hjepa/ckpts/recovery_v1/<情景>_s<种子>/evowm_l2_object.ckpt；
逐个结果写到 outputs/recovery_v1/*.json，报告写到 outputs/recovery_v1/sim_recovery_v1.md（入库版本在 docs/results/，由本地复制）。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evowm.data.transect import TreeTransectDataset  # noqa: E402
from evowm.eval.recovery import evaluate_run, load_world_model  # noqa: E402

RUNS = [("effect", 1), ("effect", 2), ("null", 1), ("null", 2)]
TRAITS_C = ["m_tarsus", "k_tarsometatarsus"]
CKPT_ROOT = ROOT / "outputs/hjepa/ckpts/recovery_v1"


def datasets(table_rel: str, tree_file: str = "data/raw/birdtree/hackett_100.nwk", n_trees: int = 100):
    kw = dict(species_table=table_rel, tree_file=tree_file,
              level1={"frameskip": 1, "num_steps": 4}, level2={"frameskip": 3, "window_size": 1, "num_steps": 4},
              evowm_root=str(ROOT), n_trees=n_trees, holdout={"level": "family", "frac": 0.15, "seed": 0}, n_orders=4)
    return TreeTransectDataset(name="evowm_train", **kw), TreeTransectDataset(name="evowm_val", **kw)


def main():
    out_dir = ROOT / "outputs/recovery_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for sc, s in RUNS:
        name = f"{sc}_s{s}"
        rel = f"data/processed/sim/recovery_v1/{name}/species_traits.csv"
        table = pd.read_csv(ROOT / rel)
        truth = json.loads((ROOT / rel).with_name("truth.json").read_text(encoding="utf-8"))
        ds_tr, ds_va = datasets(rel)
        model = load_world_model(CKPT_ROOT / name / "evowm_l2_object.ckpt", ROOT / "hjepa")
        res = evaluate_run(ds_tr, ds_va, model, table, truth, n_orders=8, seed=s)
        (out_dir / f"{name}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        results[name] = res
        print(name, {c: round(res["E1"][c]["gain"], 4) for c in TRAITS_C}, "rho", round(res["E2"]["spearman_rho"], 3),
              "t½", round(res["E3"]["t_half_free_amplitude"], 2))

    # ---- 判据 ----
    crit = {}
    c1 = all(results[f"effect_s{s}"]["E1"][c]["ci95"][0] > 0 for s in (1, 2) for c in TRAITS_C)
    c2_detail = {}
    for s in (1, 2):
        for c in TRAITS_C:
            n = results[f"null_s{s}"]["E1"][c]
            e = results[f"effect_s{s}"]["E1"][c]
            c2_detail[f"s{s}/{c}"] = (n["ci95"][0] <= 0 <= n["ci95"][1]) or (n["gain"] < 0.1 * e["gain"])
    c2 = all(c2_detail.values())
    c3a = all(results[f"effect_s{s}"]["E2"]["spearman_rho"] >= 0.7 for s in (1, 2))
    c3b = all(results[f"null_s{s}"]["E2"]["median_abs_contrast"] < 0.25 * results[f"effect_s{s}"]["E2"]["median_abs_contrast"]
              for s in (1, 2))
    c4 = all(6 <= results[n]["E3"]["t_half_free_amplitude"] <= 24 for n in results)  # 预注册修订 1
    crit = {"C1": c1, "C2": c2, "C3": c3a and c3b, "C4": c4}

    # ---- 报告 ----
    yn = lambda b: "通过" if b else "未通过"  # noqa: E731
    lines = ["# 模拟恢复检验 v1 结果", "",
             "判据见 `docs/prereg/sim_recovery_v1.md`。数值单位均为 log10；习性增益 = 无习性 RMSE − 有习性 RMSE，"
             "95% 置信区间按留出科自助重抽样 1,000 次。", "",
             f"**总体：{'通过' if all(crit.values()) else '未通过'}**（C1 {yn(c1)}，C2 {yn(c2)}，C3 {yn(c3a and c3b)}，C4 {yn(c4)}）", "",
             "## E1 习性增益", "",
             "| 模型 | 性状 | 有习性 RMSE | 无习性 RMSE | 增益 | 95% CI | 预测数 / 科数 |", "|---|---|---|---|---|---|---|"]
    for name, res in results.items():
        for c, v in res["E1"].items():
            lines.append(f"| {name} | {c} | {v['rmse_eco']:.4f} | {v['rmse_no_eco']:.4f} | {v['gain']:.4f} | "
                         f"[{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}] | {v['n_pred']} / {v['n_families']} |")
    lines += ["", "## E2 反事实体制对比（相对栖树）", "",
              "| 模型 | Spearman ρ（与真值） | 对比绝对值中位数 |", "|---|---|---|"]
    for name, res in results.items():
        lines.append(f"| {name} | {res['E2']['spearman_rho']:.3f} | {res['E2']['median_abs_contrast']:.4f} |")
    lines += ["", "## E3 半衰期（真值 12 Myr）", "", "| 模型 | t½（带振幅，C4 用） | t½（不带振幅，原定） | 振幅 |", "|---|---|---|---|"]
    for name, res in results.items():
        e3 = res["E3"]
        lines.append(f"| {name} | {e3['t_half_free_amplitude']:.2f} | {e3['t_half']:.2f} | {e3['amplitude']:.3f} |")
    lines += ["", "## 探针 R²（留出物种，编码器表征 → 观测值）", "", "| 模型 | " + " | ".join(results["effect_s1"]["probe_r2_val"]) + " |",
              "|---" * (1 + len(results["effect_s1"]["probe_r2_val"])) + "|"]
    for name, res in results.items():
        lines.append(f"| {name} | " + " | ".join(f"{v:.3f}" for v in res["probe_r2_val"].values()) + " |")
    lines += ["", "## C2 明细", ""] + [f"- {k}: {yn(v)}" for k, v in c2_detail.items()]
    rep = ROOT / "outputs/recovery_v1/sim_recovery_v1.md"
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "criteria.json").write_text(json.dumps({"criteria": crit, "C2_detail": c2_detail}, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print("判据：", crit)
    print(f"报告：{rep}")


if __name__ == "__main__":
    main()
