"""开发检查（探索性，不属于任何预注册）：动作只含习性变化（配置 recovery_v5_l1）。

用真实数据主分析 v1 第 0 阶段的模拟表（种子 9、10，已看过结果，所以只作开发，不作判定），第 21 棵树，
效应与零模型各 5 折。同一棵树上第 0 阶段的完整动作模型（outputs/hjepa/ckpts/stage0_v1/）如果在，就并列比较。
用来决定是否值得写预注册 v2，参照第 0 阶段的三条判据（只看第 21 棵树）：
  效应的相对跗跖 E1 两个种子区间下限都大于 0；零模型区间含 0 或小于效应的 10%；
  E2 的 ρ ≥ 0.7，零模型 m_tarsus 对比平均绝对值小于效应的 25%。
在 EvoWM 根目录运行：python scripts/exploratory/eval_dev_change_v1.py
输出只写到 outputs/dev_change_v1/：summary.json、report.md。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from evowm.eval.pooled import boot_gain, boot_gain_diff, strip_boot  # noqa: E402
from evowm.eval.recovery import BASE_REGIME, fit_half_life  # noqa: E402
from evowm.eval.runs import N_FOLDS, relative_errors, run_all  # noqa: E402

N_BOOT = 10000
REG6 = ["terrestrial", "aerial", "wading", "diving", "surface", "generalist"]
ZH = {"terrestrial": "地栖", "aerial": "空中", "wading": "涉水", "diving": "潜水", "surface": "水面", "generalist": "泛化"}
VARIANT_ZH = {"change": "只含习性变化", "full": "完整动作（第 0 阶段）"}


def table_rel(sc, s):
    return f"data/processed/sim/recovery_v5/{sc}_s{s}/species_traits.csv"


def summarize(grp, truth):
    fam = np.concatenate([r["family"] for r in grp])
    trans = np.concatenate([r["transition"] for r in grp])
    rel = [relative_errors(r) for r in grp]
    re_, rn_ = np.concatenate([a for a, _ in rel]), np.concatenate([b for _, b in rel])
    cols = grp[0]["E1_state"]["state_cols"]
    jt, jm = cols.index("m_tarsus"), cols.index("m_mass")
    ee = np.concatenate([r["E1_state"]["err_eco"] for r in grp])
    en = np.concatenate([r["E1_state"]["err_no_eco"] for r in grp])
    nan = np.full(len(fam), np.nan)
    out = {"E1_rel": boot_gain(re_, rn_, fam, N_BOOT, 0),
           "E1_rel_transition": boot_gain(np.where(trans, re_, nan), np.where(trans, rn_, nan), fam, N_BOOT, 0),
           "E1_abs": boot_gain(ee[:, jt], en[:, jt], fam, N_BOOT, 0),
           "E1_mass": boot_gain(ee[:, jm], en[:, jm], fam, N_BOOT, 0),
           "share_transition": float(np.mean(trans)), "_fam": fam, "_re": re_, "_rn": rn_}
    regs = [x for x in truth["regimes"] if x != BASE_REGIME]
    traits = list(grp[0]["res"]["E2_relative"]["contrasts"][regs[0]])
    mean_c = {x: {c: float(np.mean([r["res"]["E2_relative"]["contrasts"][x][c] for r in grp])) for c in traits} for x in regs}
    ident = {c: [x for x in regs if all(x in r["res"]["E2_relative"]["identifiable"][c] for r in grp)] for c in traits}
    off, src = truth["leg_offsets"], truth["leg_source"]
    pv = [mean_c[x][c] for c in traits for x in ident[c]]
    tv = [off[src[c]][x] - off[src[c]][BASE_REGIME] for c in traits for x in ident[c]]
    out["E2_mean"] = mean_c
    out["E2_rho"] = float(spearmanr(pv, tv).statistic) if len(tv) > 2 and np.ptp(tv) > 0 else float("nan")
    out["E2_mtarsus_mean_abs"] = float(np.mean([abs(mean_c[x]["m_tarsus"]) for x in ident["m_tarsus"]]))
    fit = fit_half_life(np.concatenate([r["E3"]["distance"] for r in grp]), np.concatenate([r["E3"]["sensitivity"] for r in grp]))
    out["t_half"] = fit["t_half_free_amplitude"]
    if all("linear" in r for r in grp):
        le = np.concatenate([r["linear"]["err_eco"] for r in grp])
        lc = np.concatenate([r["linear"]["err_change"] for r in grp])
        ln = np.concatenate([r["linear"]["err_no"] for r in grp])
        out["linear_level"] = boot_gain(le, ln, fam, N_BOOT, 0)
        out["linear_change"] = boot_gain(lc, ln, fam, N_BOOT, 0)
        out["linear_kappa"] = float(np.median([r["linear"]["kappa"] for r in grp]))
    return out


def fmt(d):
    return f"{d['gain']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[9, 10])
    ap.add_argument("--tree", type=int, default=21)
    ap.add_argument("--ckpt-root", default=str(ROOT / "outputs/hjepa/ckpts/dev_change_v1"))
    ap.add_argument("--ref-root", default=str(ROOT / "outputs/hjepa/ckpts/stage0_v1"))
    ap.add_argument("--out", default=str(ROOT / "outputs/dev_change_v1"))
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tt = f"t{args.tree:03d}"

    tasks, variants = [], []
    for variant, root in (("change", Path(args.ckpt_root)), ("full", Path(args.ref_root))):
        cks = {(sc, s, k): root / f"{sc}_s{s}_{tt}_f{k}" / "evowm_l1_object.ckpt"
               for s in args.seeds for sc in ("effect", "null") for k in range(N_FOLDS)}
        if not all(c.exists() for c in cks.values()):
            if variant == "change":
                sys.exit(f"缺少 {root} 下的模型")
            print(f"没有找到 {root} 下第 0 阶段的完整动作模型，只报告新模型")
            continue
        variants.append(variant)
        for (sc, s, k), ck in cks.items():
            tasks.append({"root": str(ROOT), "table_rel": table_rel(sc, s), "fold": k, "seed": s,
                          "tree_file": f"data/processed/trees/hackett_{tt}.nwk", "ckpt": str(ck),
                          "truth": str(ROOT / table_rel(sc, s).replace("species_traits.csv", "truth.json")),
                          "linear": variant == "change", "key": f"{variant}_{sc}_s{s}"})
    results = run_all(tasks, args.workers)

    summ = {}
    for variant in variants:
        for s in args.seeds:
            for sc in ("effect", "null"):
                truth = json.loads((ROOT / table_rel(sc, s)).with_name("truth.json").read_text(encoding="utf-8"))
                summ[f"{variant}_{sc}_s{s}"] = summarize([r for r in results if r["key"] == f"{variant}_{sc}_s{s}"], truth)
            e, n = summ[f"{variant}_effect_s{s}"], summ[f"{variant}_null_s{s}"]
            if np.array_equal(e["_fam"], n["_fam"]):  # 同一批窗口：效应减零模型的配对差
                summ[f"{variant}_paired_s{s}"] = boot_gain_diff(e["_re"], e["_rn"], n["_re"], n["_rn"], e["_fam"], N_BOOT, 0)

    checks = {}
    for s in args.seeds:
        e, n = summ[f"change_effect_s{s}"], summ[f"change_null_s{s}"]
        lo, hi = n["E1_rel"]["ci95"]
        checks[s] = {"G1": bool(e["E1_rel"]["ci95"][0] > 0),
                     "G2": bool((lo <= 0 <= hi) or n["E1_rel"]["gain"] < 0.1 * e["E1_rel"]["gain"]),
                     "G3": bool(e["E2_rho"] >= 0.7 and n["E2_mtarsus_mean_abs"] < 0.25 * e["E2_mtarsus_mean_abs"])}
    passed = all(all(c.values()) for c in checks.values())
    clean = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in summ.items()}
    (out / "summary.json").write_text(json.dumps({"checks": checks, "passed": passed, "summary": strip_boot(clean)},
                                                 ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    yn = lambda b: "通过" if b else "未通过"  # noqa: E731
    L = ["# 开发检查：动作只含习性变化（探索性，不属于预注册）", "",
         f"第 0 阶段模拟表（种子 {'、'.join(map(str, args.seeds))}，有早期爆发背景），第 {args.tree} 棵树，按科 5 折。"
         f"数值单位 log10；增益 = 无习性 RMSE − 有习性 RMSE；按科自助重抽样 {N_BOOT} 次。", "",
         f"**参照第 0 阶段判据（只看这一棵树）：{yn(passed)}**（"
         + "；".join(f"种子 {s}：G1 {yn(c['G1'])}，G2 {yn(c['G2'])}，G3 {yn(c['G3'])}" for s, c in checks.items()) + "）", "",
         "## E1 相对跗跖增益", "",
         "| 动作 | 模型 | 全部相邻对 | 习性变化对 | 绝对跗跖 | 体重 |", "|---|---|---|---|---|---|"]
    for variant in variants:
        for s in args.seeds:
            for sc in ("effect", "null"):
                v = summ[f"{variant}_{sc}_s{s}"]
                L.append(f"| {VARIANT_ZH[variant]} | {sc}_s{s} | {fmt(v['E1_rel'])} | {fmt(v['E1_rel_transition'])} | "
                         f"{v['E1_abs']['gain']:+.4f} | {v['E1_mass']['gain']:+.4f} |")
    L += ["", "## 效应减零模型（同一批窗口的配对差）", "", "| 动作 | 种子 | 差 [95% CI] |", "|---|---|---|"]
    for variant in variants:
        for s in args.seeds:
            p = summ.get(f"{variant}_paired_s{s}")
            if p:
                L.append(f"| {VARIANT_ZH[variant]} | {s} | {p['diff']:+.4f} [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}] |")
    L += ["", "## 线性预测器（同一批窗口）", "", "| 模型 | 习性当水平用 | 习性只当变化用 | κ 中位数 |", "|---|---|---|---|"]
    for s in args.seeds:
        for sc in ("effect", "null"):
            v = summ[f"change_{sc}_s{s}"]
            L.append(f"| {sc}_s{s} | {fmt(v['linear_level'])} | {fmt(v['linear_change'])} | {v['linear_kappa']:.2f} |")
    L += ["", "## E2 相对跗跖对比（5 折平均）与 E3", "",
          "| 动作 | 模型 | ρ | m_tarsus 平均绝对值 | " + " / ".join(ZH[x] for x in REG6) + " | t½（Myr） |", "|---|---|---|---|---|---|"]
    for variant in variants:
        for s in args.seeds:
            for sc in ("effect", "null"):
                v = summ[f"{variant}_{sc}_s{s}"]
                rho = "不适用" if v["E2_rho"] != v["E2_rho"] else f"{v['E2_rho']:.3f}"
                L.append(f"| {VARIANT_ZH[variant]} | {sc}_s{s} | {rho} | {v['E2_mtarsus_mean_abs']:.4f} | "
                         + " / ".join(f"{v['E2_mean'][x]['m_tarsus']:+.3f}" for x in REG6) + f" | {v['t_half']:.1f} |")
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(json.dumps({"checks": checks, "passed": passed}, ensure_ascii=False))
    print(f"报告：{out / 'report.md'}")


if __name__ == "__main__":
    main()
