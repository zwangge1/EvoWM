"""评估模拟恢复检验 v4（预注册 docs/prereg/sim_recovery_v4.md），检验种子 7、8。

v4 与 v3 的唯一区别：E1 改用相对跗跖。相对跗跖 = m_tarsus − b × m_mass，b 为各折训练物种上 m_tarsus 对 m_mass
的 OLS 斜率（即 evaluate_run 里 E2 用的异速斜率）；逐预测误差 = 跗跖误差 − b × 体重误差。
训练、5 折留出、窗口、E2、E3 的算法与 v3 相同。预注册文件不存在时拒绝运行。

在 EvoWM 根目录运行：python scripts/eval_recovery_v4.py
输出只写到 outputs/recovery_v4/：<情景>_s<种子>.json、criteria.json、sim_recovery_v4.md。
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
sys.path.insert(0, str(ROOT / "scripts"))

from eval_recovery_v3 import REG_ORDER, ci, fold_datasets, gain  # noqa: E402  留出方式与置信区间同 v3
from evowm.eval.recovery import BASE_REGIME, evaluate_run, fit_half_life, load_world_model  # noqa: E402

SEEDS = (7, 8)
N_FOLDS = 5
PREREG = ROOT / "docs/prereg/sim_recovery_v4.md"


def evaluate_cv(sc: str, s: int, ckpt_root: Path) -> dict:
    rel = f"data/processed/sim/recovery_v4/{sc}_s{s}/species_traits.csv"
    table = pd.read_csv(ROOT / rel)
    truth = json.loads((ROOT / rel).with_name("truth.json").read_text(encoding="utf-8"))
    folds, raws = [], []
    for k in range(N_FOLDS):
        ds_tr, ds_va = fold_datasets(rel, k)
        model = load_world_model(ckpt_root / f"{sc}_s{s}_f{k}" / "evowm_l1_object.ckpt", ROOT / "hjepa")
        raw = {}
        folds.append(evaluate_run(ds_tr, ds_va, model, table, truth, n_orders=8, seed=s, raw=raw))
        raws.append(raw)

    # ---- E1：相对跗跖（判定用）、绝对跗跖与体重（描述），5 折合并 ----
    cols = raws[0]["E1_state"]["state_cols"]
    jt, jm = cols.index("m_tarsus"), cols.index("m_mass")
    acc = {key: [] for key in ("rel_eco", "rel_no", "abs_eco", "abs_no", "mass_eco", "mass_no",
                               "or_eco", "or_no", "family", "transition")}
    slopes = []
    for f, raw in zip(folds, raws):
        b = float(f["E2_relative"]["allometric_slopes"]["m_tarsus"])
        slopes.append(b)
        ee, en = raw["E1_state"]["err_eco"], raw["E1_state"]["err_no_eco"]
        acc["rel_eco"].append(ee[:, jt] - b * ee[:, jm])
        acc["rel_no"].append(en[:, jt] - b * en[:, jm])
        acc["abs_eco"].append(ee[:, jt])
        acc["abs_no"].append(en[:, jt])
        acc["mass_eco"].append(ee[:, jm])
        acc["mass_no"].append(en[:, jm])
        mt = raw["E1"]["m_tarsus"]
        acc["or_eco"].append(mt["oracle_eco"])
        acc["or_no"].append(mt["oracle_no_eco"])
        acc["family"].append(mt["family"])
        acc["transition"].append(mt["transition"])
    a = {key: np.concatenate(v) for key, v in acc.items()}
    fam, tr = a["family"], a["transition"]
    nan = np.full(len(fam), np.nan)
    T = lambda x: np.where(tr, x, nan)  # noqa: E731
    out = {"folds": folds, "allometric_slope_per_fold": slopes, "share_transition": float(tr.mean())}
    out["E1_rel"] = {"all": gain(a["rel_eco"], a["rel_no"], fam, s), "transition": gain(T(a["rel_eco"]), T(a["rel_no"]), fam, s)}
    out["E1_abs"] = {"all": gain(a["abs_eco"], a["abs_no"], fam, s), "transition": gain(T(a["abs_eco"]), T(a["abs_no"]), fam, s),
                     "oracle_all": gain(a["or_eco"], a["or_no"], fam, s)}
    out["E1_mass"] = {"all": gain(a["mass_eco"], a["mass_no"], fam, s)}

    # ---- E2：同 v3（相对对比取 5 折平均，可识别组合取 5 折都可识别的）----
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
                          "m_tarsus_mean_abs": float(np.mean(mt_abs)) if mt_abs else float("nan")}

    # ---- E3：同 v3（5 折的距离与敏感度合并后拟合）----
    d0 = np.concatenate([r["E3"]["distance"] for r in raws])
    sens = np.concatenate([r["E3"]["sensitivity"] for r in raws])
    out["E3"] = fit_half_life(d0, sens)
    out["E3"]["t_half_free_amplitude_per_fold"] = [f["E3"]["t_half_free_amplitude"] for f in folds]
    out["probe_r2_val_mean"] = {c: float(np.mean([f["probe_r2_val"][c] for f in folds])) for c in folds[0]["probe_r2_val"]}
    return out


def criteria(res: dict) -> dict:
    """docs/prereg/sim_recovery_v4.md 的 C1 到 C4，均针对种子 7、8。"""
    c1, c2, c3 = {}, {}, {}
    for s in SEEDS:
        e, n = res[f"effect_s{s}"], res[f"null_s{s}"]
        c1[s] = bool(e["E1_rel"]["all"]["ci95"][0] > 0)
        lo, hi = n["E1_rel"]["all"]["ci95"]
        c2[s] = bool((lo <= 0 <= hi) or (n["E1_rel"]["all"]["gain"] < 0.1 * e["E1_rel"]["all"]["gain"]))
        c3[s] = bool(e["E2_relative"]["spearman_rho_identifiable"] >= 0.7
                     and n["E2_relative"]["m_tarsus_mean_abs"] < 0.25 * e["E2_relative"]["m_tarsus_mean_abs"])
    c4 = {k: bool(6 <= r["E3"]["t_half_free_amplitude"] <= 24) for k, r in res.items()}
    return {"C1": all(c1.values()), "C2": all(c2.values()), "C3": all(c3.values()), "C4": all(c4.values()),
            "C1_per_seed": c1, "C2_per_seed": c2, "C3_per_seed": c3, "C4_per_model": c4}


def report(res: dict, crit: dict, path: Path):
    yn = lambda b: "通过" if b else "未通过"  # noqa: E731
    ok = all(crit[k] for k in ("C1", "C2", "C3", "C4"))
    L = ["# 模拟恢复检验 v4 结果（相对跗跖 E1，按科 5 折交叉验证，检验种子 7、8）", "",
         "判据见 `docs/prereg/sim_recovery_v4.md`。数值单位均为 log10；增益 = 无习性 RMSE − 有习性 RMSE，"
         "95% 置信区间按全部科自助重抽样 1,000 次。相对跗跖 = 跗跖 − b × 体重，b 为各折训练物种上的异速斜率。", "",
         f"**总体：{yn(ok)}**（C1 {yn(crit['C1'])}，C2 {yn(crit['C2'])}，C3 {yn(crit['C3'])}，C4 {yn(crit['C4'])}）", "",
         "## 判据明细", "",
         "- C1（效应情景相对跗跖增益下限 > 0）：" + "，".join(f"种子 {s} {yn(v)}" for s, v in crit["C1_per_seed"].items()),
         "- C2（零模型区间含 0，或小于效应的 10%）：" + "，".join(f"种子 {s} {yn(v)}" for s, v in crit["C2_per_seed"].items()),
         "- C3（ρ ≥ 0.7 且零模型对比 < 效应的 25%）：" + "，".join(f"种子 {s} {yn(v)}" for s, v in crit["C3_per_seed"].items()),
         "- C4（合并拟合 t½ 在 6 到 24 Myr）：" + "，".join(f"{k} {yn(v)}" for k, v in crit["C4_per_model"].items()), "",
         "## E1 习性增益（5 折合并）", "",
         "| 模型 | 相对跗跖（判定） | 相对跗跖：习性变化对 | 绝对跗跖 | 体重 | 理想预测器（绝对跗跖） | 科数 |",
         "|---|---|---|---|---|---|---|"]
    for k, r in res.items():
        L.append(f"| {k} | {ci(r['E1_rel']['all'])} | {ci(r['E1_rel']['transition'])} | {ci(r['E1_abs']['all'])} | "
                 f"{ci(r['E1_mass']['all'])} | {ci(r['E1_abs']['oracle_all'])} | {r['E1_rel']['all']['n_families']} |")
    L += ["", "## E2 相对腿长对比（5 折平均）", "",
          "| 模型 | 可识别组合数 | ρ | m_tarsus 平均绝对值 | m_tarsus（地栖 / 空中 / 涉水 / 潜水 / 水面 / 泛化） |", "|---|---|---|---|---|"]
    for k, r in res.items():
        e = r["E2_relative"]
        rho = "不适用" if e["spearman_rho_identifiable"] != e["spearman_rho_identifiable"] else f"{e['spearman_rho_identifiable']:.3f}"
        mt = " / ".join(f"{e['contrasts_mean'][x]['m_tarsus']:+.3f}" for x in REG_ORDER)
        L.append(f"| {k} | {e['n_identifiable']} | {rho} | {e['m_tarsus_mean_abs']:.4f} | {mt} |")
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
    ap.add_argument("--ckpt-root", default=str(ROOT / "outputs/hjepa/ckpts/recovery_v4"))
    ap.add_argument("--out", default=str(ROOT / "outputs/recovery_v4"))
    args = ap.parse_args()
    if not PREREG.exists():
        sys.exit(f"要等 {PREREG.relative_to(ROOT)} 写定并提交后才能评估。")
    ckpt_root, out = Path(args.ckpt_root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for s in SEEDS:
        for sc in ("effect", "null"):
            missing = [k for k in range(N_FOLDS) if not (ckpt_root / f"{sc}_s{s}_f{k}" / "evowm_l1_object.ckpt").exists()]
            if missing:
                sys.exit(f"{sc}_s{s} 缺少第 {missing} 折的模型，检验不能进行")
    res = {}
    for s in SEEDS:
        for sc in ("effect", "null"):
            key = f"{sc}_s{s}"
            res[key] = evaluate_cv(sc, s, ckpt_root)
            (out / f"{key}.json").write_text(json.dumps(res[key], ensure_ascii=False, indent=2), encoding="utf-8")
            r = res[key]
            print(f"{key}: E1 相对跗跖 {ci(r['E1_rel']['all'])}（绝对 {ci(r['E1_abs']['all'])}）；"
                  f"E2 ρ {r['E2_relative']['spearman_rho_identifiable']:.3f}；E3 t½ {r['E3']['t_half_free_amplitude']:.1f}", flush=True)
    crit = criteria(res)
    (out / "criteria.json").write_text(json.dumps(crit, ensure_ascii=False, indent=2), encoding="utf-8")
    report(res, crit, out / "sim_recovery_v4.md")
    print("判据：", {k: crit[k] for k in ("C1", "C2", "C3", "C4")})
    print(f"报告：{out / 'sim_recovery_v4.md'}")


if __name__ == "__main__":
    main()
