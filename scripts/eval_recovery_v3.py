"""评估模拟恢复检验 v3（按科 5 折交叉验证）。

每个情景、每个种子有 5 个模型（outputs/hjepa/ckpts/recovery_v3/<情景>_s<种子>_f<折>/），各自只在本折留出的科上评估，
再把 5 折合并：
- E1：逐预测误差合并后，按全部科（约 194 个）自助重抽样 1,000 次求增益的 95% 置信区间；同一批窗口上算理想预测器的增益，
  报告模型与理想预测器之比。另报只看体制转变对的增益。
- E2：相对腿长对比取 5 折平均；可识别组合取 5 折都可识别的；Spearman ρ 与零模型平均绝对值同 v2。
- E3：5 折的（距离，敏感度）合并后按 v2 的方法拟合 t½，另列各折单独拟合的值。

在 EvoWM 根目录运行：
  python scripts/eval_recovery_v3.py                 # 开发种子 1、2，报告写到 docs/results/sim_recovery_v3_dev.md
  python scripts/eval_recovery_v3.py --seeds 5 6     # 检验种子：docs/prereg/sim_recovery_v3.md 写定前拒绝运行
逐个结果写到 outputs/recovery_v3/<情景>_s<种子>.json。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evowm.data.transect import TreeTransectDataset  # noqa: E402
from evowm.eval.recovery import (BASE_REGIME, LEG_TRAITS, evaluate_run, family_bootstrap_rmse_diff,  # noqa: E402
                                 fit_half_life, load_world_model)

DEV_SEEDS, TEST_SEEDS = (1, 2), (5, 6)
N_FOLDS = 5
TREE = "data/raw/birdtree/hackett_tree1.nwk"
CKPT_ROOT = ROOT / "outputs/hjepa/ckpts/recovery_v3"
PREREG = ROOT / "docs/prereg/sim_recovery_v3.md"
REG_ORDER = ["terrestrial", "aerial", "wading", "diving", "surface", "generalist"]


def fold_datasets(rel: str, fold: int):
    kw = dict(species_table=rel, tree_file=TREE, level1={"frameskip": 1, "num_steps": 4}, evowm_root=str(ROOT),
              n_trees=1, holdout={"level": "family", "frac": 0, "folds": N_FOLDS, "fold": fold, "seed": 0}, n_orders=4)
    return TreeTransectDataset(name="evowm_train", **kw), TreeTransectDataset(name="evowm_val", **kw)


def gain(a: np.ndarray, b: np.ndarray, fam: np.ndarray, seed: int) -> dict:
    g, lo, hi, n, nf = family_bootstrap_rmse_diff(a, b, fam, seed=seed)
    return {"gain": g, "ci95": [lo, hi], "n_pred": n, "n_families": nf}


def evaluate_cv(sc: str, s: int) -> dict:
    rel = f"data/processed/sim/recovery_v3/{sc}_s{s}/species_traits.csv"
    table = pd.read_csv(ROOT / rel)
    truth = json.loads((ROOT / rel).with_name("truth.json").read_text(encoding="utf-8"))
    folds, raws = [], []
    for k in range(N_FOLDS):
        ds_tr, ds_va = fold_datasets(rel, k)
        model = load_world_model(CKPT_ROOT / f"{sc}_s{s}_f{k}" / "evowm_l1_object.ckpt", ROOT / "hjepa")
        raw = {}
        folds.append(evaluate_run(ds_tr, ds_va, model, table, truth, n_orders=8, seed=s, raw=raw))
        raws.append(raw)

    out = {"folds": folds}
    # ---- E1 合并 ----
    e1 = {}
    for c in LEG_TRAITS:
        if c not in raws[0]["E1"]:
            continue
        cat = {key: np.concatenate([r["E1"][c][key] for r in raws]) for key in raws[0]["E1"][c]}
        fam, tr = cat["family"], cat["transition"]
        m = gain(cat["err_eco"], cat["err_no_eco"], fam, s)
        o = gain(cat["oracle_eco"], cat["oracle_no_eco"], fam, s)
        nan = np.full_like(cat["err_eco"], np.nan)
        mt = gain(np.where(tr, cat["err_eco"], nan), np.where(tr, cat["err_no_eco"], nan), fam, s)
        ot = gain(np.where(tr, cat["oracle_eco"], nan), np.where(tr, cat["oracle_no_eco"], nan), fam, s)
        e1[c] = {"model": m, "oracle": o, "ratio_to_oracle": m["gain"] / o["gain"] if o["gain"] > 0 else float("nan"),
                 "model_transition": mt, "oracle_transition": ot, "share_transition": float(tr.mean()),
                 "rmse_eco": float(np.sqrt(np.nanmean(cat["err_eco"] ** 2))),
                 "rmse_no_eco": float(np.sqrt(np.nanmean(cat["err_no_eco"] ** 2)))}
    out["E1"] = e1

    # ---- E2 合并：相对对比取 5 折平均，可识别组合取 5 折都可识别的 ----
    regimes = [r for r in truth["regimes"] if r != BASE_REGIME]
    traits = list(folds[0]["E2_relative"]["contrasts"][regimes[0]])
    mean_c = {r: {c: float(np.mean([f["E2_relative"]["contrasts"][r][c] for f in folds])) for c in traits} for r in regimes}
    ident = {c: [r for r in regimes if all(r in f["E2_relative"]["identifiable"][c] for f in folds)] for c in traits}
    off, src = truth["leg_offsets"], truth["leg_source"]
    pv = np.array([mean_c[r][c] for c in traits for r in ident[c]])
    tv = np.array([off[src[c]][r] - off[src[c]][BASE_REGIME] for c in traits for r in ident[c]])
    rho = spearmanr(pv, tv).statistic if len(tv) > 2 and np.ptp(tv) > 0 else float("nan")
    mt_abs = [abs(mean_c[r]["m_tarsus"]) for r in ident.get("m_tarsus", [])]
    out["E2_relative"] = {"contrasts_mean": mean_c, "identifiable_all_folds": ident, "n_identifiable": int(len(pv)),
                          "spearman_rho_identifiable": float(rho),
                          "m_tarsus_mean_abs": float(np.mean(mt_abs)) if mt_abs else float("nan"),
                          "rho_per_fold": [f["E2_relative"]["spearman_rho_identifiable"] for f in folds]}

    # ---- E3 合并 ----
    d0 = np.concatenate([r["E3"]["distance"] for r in raws])
    sens = np.concatenate([r["E3"]["sensitivity"] for r in raws])
    out["E3"] = fit_half_life(d0, sens)
    out["E3"]["t_half_free_amplitude_per_fold"] = [f["E3"]["t_half_free_amplitude"] for f in folds]

    out["probe_r2_val_mean"] = {c: float(np.mean([f["probe_r2_val"][c] for f in folds])) for c in folds[0]["probe_r2_val"]}
    return out


def ci(d: dict) -> str:
    return f"{d['gain']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"


def report(res: dict, seeds, path: Path):
    L = ["# 模拟恢复检验 v3：开发种子结果（按科 5 折交叉验证）", "",
         "只用于确定 v3 的判据，不参与判定。数值单位均为 log10；增益 = 无习性 RMSE − 有习性 RMSE，"
         "95% 置信区间按全部科自助重抽样 1,000 次。理想预测器在同一批窗口上计算。", "",
         "## E1 m_tarsus 习性增益（5 折合并）", "",
         "| 模型 | 模型增益 | 理想预测器增益 | 模型 / 理想 | 转变对：模型 | 转变对：理想 | 科数 |", "|---|---|---|---|---|---|---|"]
    for k, r in res.items():
        e = r["E1"]["m_tarsus"]
        L.append(f"| {k} | {ci(e['model'])} | {ci(e['oracle'])} | {e['ratio_to_oracle']:.2f} | "
                 f"{ci(e['model_transition'])} | {ci(e['oracle_transition'])} | {e['model']['n_families']} |")
    L += ["", "## E1 各折单独的 m_tarsus 增益（只作参考）", "", "| 模型 | 折 0 | 折 1 | 折 2 | 折 3 | 折 4 |", "|---|---|---|---|---|---|"]
    for k, r in res.items():
        L.append(f"| {k} | " + " | ".join(f"{f['E1']['m_tarsus']['gain']:+.4f}" for f in r["folds"]) + " |")
    L += ["", "## E1 骨段性状（5 折合并，只作描述）", "", "| 模型 | 性状 | 模型增益 | 理想预测器增益 |", "|---|---|---|---|"]
    for k, r in res.items():
        for c, e in r["E1"].items():
            if c != "m_tarsus":
                L.append(f"| {k} | {c} | {ci(e['model'])} | {ci(e['oracle'])} |")
    L += ["", "## E2 相对腿长对比（5 折平均）", "",
          "| 模型 | 可识别组合数 | ρ | 各折 ρ | m_tarsus 平均绝对值 | m_tarsus（地栖 / 空中 / 涉水 / 潜水 / 水面 / 泛化） |",
          "|---|---|---|---|---|---|"]
    for k, r in res.items():
        e = r["E2_relative"]
        rho = "不适用" if e["spearman_rho_identifiable"] != e["spearman_rho_identifiable"] else f"{e['spearman_rho_identifiable']:.3f}"
        per = " / ".join("无" if x != x else f"{x:.2f}" for x in e["rho_per_fold"])
        mt = " / ".join(f"{e['contrasts_mean'][x]['m_tarsus']:+.3f}" for x in REG_ORDER)
        L.append(f"| {k} | {e['n_identifiable']} | {rho} | {per} | {e['m_tarsus_mean_abs']:.4f} | {mt} |")
    L += ["", "## E3 半衰期（真值 12 Myr）", "", "| 模型 | t½ 合并拟合（带振幅） | 振幅 | t½（不带振幅） | 各折 t½（带振幅） |", "|---|---|---|---|---|"]
    for k, r in res.items():
        e = r["E3"]
        per = " / ".join(f"{x:.1f}" for x in e["t_half_free_amplitude_per_fold"])
        L.append(f"| {k} | {e['t_half_free_amplitude']:.1f} | {e['amplitude']:.2f} | {e['t_half']:.1f} | {per} |")
    L += ["", "## 探针 R²（留出物种，5 折平均）", ""]
    cols = list(next(iter(res.values()))["probe_r2_val_mean"])
    L += ["| 模型 | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for k, r in res.items():
        L.append(f"| {k} | " + " | ".join(f"{r['probe_r2_val_mean'][c]:.3f}" for c in cols) + " |")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEV_SEEDS))
    args = ap.parse_args()
    if any(s in TEST_SEEDS for s in args.seeds):
        if not PREREG.exists():
            sys.exit(f"检验种子 {TEST_SEEDS} 要等 {PREREG.relative_to(ROOT)} 写定并提交后才能评估。")
        sys.exit("检验种子的判定部分在预注册写定后补到本脚本里。")
    out_dir = ROOT / "outputs/recovery_v3"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {}
    for s in args.seeds:
        for sc in ("effect", "null"):
            key = f"{sc}_s{s}"
            missing = [k for k in range(N_FOLDS) if not (CKPT_ROOT / f"{key}_f{k}" / "evowm_l1_object.ckpt").exists()]
            if missing:
                print(f"跳过 {key}：缺少第 {missing} 折的模型")
                continue
            res[key] = evaluate_cv(sc, s)
            (out_dir / f"{key}.json").write_text(json.dumps(res[key], ensure_ascii=False, indent=2), encoding="utf-8")
            e = res[key]["E1"]["m_tarsus"]
            print(f"{key}: E1 模型 {ci(e['model'])}，理想 {ci(e['oracle'])}，比 {e['ratio_to_oracle']:.2f}；"
                  f"E2 ρ {res[key]['E2_relative']['spearman_rho_identifiable']:.3f}，"
                  f"m_tarsus |对比| {res[key]['E2_relative']['m_tarsus_mean_abs']:.4f}；"
                  f"E3 t½ {res[key]['E3']['t_half_free_amplitude']:.1f}", flush=True)
    if res:
        rep = ROOT / "docs/results/sim_recovery_v3_dev.md"
        report(res, args.seeds, rep)
        print(f"报告：{rep}")


if __name__ == "__main__":
    main()
