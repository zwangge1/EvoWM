"""真实数据主分析 v1 的第 0 阶段：树不对时的模拟校准（预注册 docs/prereg/real_main_v1.md 第 4 节）。

模拟数据在 Hackett 第 1 棵树上生成（种子 9、10，效应与零模型，sim_recovery_tables.py --version v5，
习性的 OU 响应同 v4，另加按真实数据残差结构设定的早期爆发布朗背景），
模型在第 21、41、61、81、100 棵树上训练（各 5 折），评估用同一棵训练树取窗口。
判定 G1 到 G3；并按固定规则确定 E2 的判定方式（按科、按目重抽样或零模型阈值），写进 decision.json；
E3 半衰期只作真实数据的参照（有布朗背景时没有单一真值）。
在 EvoWM 根目录运行：python scripts/eval_stage0_v1.py
输出只写到 outputs/stage0_v1/：decision.json、stage0.json、report.md。
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

from evowm.eval.pooled import boot_gain, boot_mean, strip_boot  # noqa: E402
from evowm.eval.recovery import BASE_REGIME, fit_half_life  # noqa: E402
from evowm.eval.runs import N_FOLDS, relative_errors, run_all  # noqa: E402

PREREG = ROOT / "docs/prereg/real_main_v1.md"
N_BOOT = 10000
REG6 = ["terrestrial", "aerial", "wading", "diving", "surface", "generalist"]
ZH = {"terrestrial": "地栖", "aerial": "空中", "wading": "涉水", "diving": "潜水", "surface": "水面", "generalist": "泛化"}


def table_rel(sc, s):
    return f"data/processed/sim/recovery_v5/{sc}_s{s}/species_traits.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[9, 10])
    ap.add_argument("--trees", type=int, nargs="+", default=[21, 41, 61, 81, 100])
    ap.add_argument("--ckpt-root", default=str(ROOT / "outputs/hjepa/ckpts/stage0_v1"))
    ap.add_argument("--out", default=str(ROOT / "outputs/stage0_v1"))
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    if not PREREG.exists():
        sys.exit(f"要等 {PREREG.relative_to(ROOT)} 提交后才能运行。")
    out, ckpt_root = Path(args.out), Path(args.ckpt_root)
    out.mkdir(parents=True, exist_ok=True)

    tasks = []
    for s in args.seeds:
        for sc in ("effect", "null"):
            truth = ROOT / table_rel(sc, s).replace("species_traits.csv", "truth.json")
            for t in args.trees:
                for k in range(N_FOLDS):
                    ck = ckpt_root / f"{sc}_s{s}_t{t:03d}_f{k}" / "evowm_l1_object.ckpt"
                    if not ck.exists():
                        sys.exit(f"缺少模型 {ck}")
                    tasks.append({"root": str(ROOT), "table_rel": table_rel(sc, s), "fold": k, "seed": s,
                                  "tree_file": f"data/processed/trees/hackett_t{t:03d}.nwk", "ckpt": str(ck),
                                  "truth": str(truth), "key": f"{sc}_s{s}_t{t:03d}"})
    results = run_all(tasks, args.workers)
    truth0 = json.loads((ROOT / table_rel("effect", args.seeds[0])).with_name("truth.json").read_text(encoding="utf-8"))

    summ = {}
    for s in args.seeds:
        for sc in ("effect", "null"):
            grp = [r for r in results if r["key"].startswith(f"{sc}_s{s}_")]
            table = pd.read_csv(ROOT / table_rel(sc, s))
            truth = json.loads((ROOT / table_rel(sc, s)).with_name("truth.json").read_text(encoding="utf-8"))
            rel = [relative_errors(r) for r in grp]
            fam = np.concatenate([r["family"] for r in grp])
            e1 = boot_gain(np.concatenate([a for a, _ in rel]), np.concatenate([b for _, b in rel]), fam, N_BOOT, 0)
            # E2：树与折平均；可识别组合取所有运行都可识别的
            regs = [x for x in truth["regimes"] if x != BASE_REGIME]
            traits = list(grp[0]["res"]["E2_relative"]["contrasts"][regs[0]])
            mean_c = {x: {c: float(np.mean([r["res"]["E2_relative"]["contrasts"][x][c] for r in grp])) for c in traits} for x in regs}
            ident = {c: [x for x in regs if all(x in r["res"]["E2_relative"]["identifiable"][c] for r in grp)] for c in traits}
            off, src = truth["leg_offsets"], truth["leg_source"]
            pv = [mean_c[x][c] for c in traits for x in ident[c]]
            tv = [off[src[c]][x] - off[src[c]][BASE_REGIME] for c in traits for x in ident[c]]
            rho = float(spearmanr(pv, tv).statistic) if len(tv) > 2 and np.ptp(tv) > 0 else float("nan")
            mt_abs = float(np.mean([abs(mean_c[x]["m_tarsus"]) for x in ident["m_tarsus"]]))
            # E2 逐窗口相对对比（m_tarsus）按科、按目自助重抽样
            tgt = np.concatenate([r["E2_window"]["target_rows"] for r in grp])
            ci = {}
            for level in ("family", "order"):
                cl = table[level].to_numpy()[tgt]
                ci[level] = {x: boot_mean(np.concatenate([r["E2_window"]["rel"][x]["m_tarsus"] for r in grp]), cl, N_BOOT, 0)
                             for x in REG6}
            # 每次运行（每棵树）的 E2 对比：5 折窗口合并后的均值，用于零模型阈值
            run_c = {t: {x: float(np.mean(np.concatenate([r["E2_window"]["rel"][x]["m_tarsus"] for r in grp
                                                            if r["key"].endswith(f"_t{t:03d}")]))) for x in REG6}
                     for t in args.trees}
            # E3：每棵树合并 5 折拟合
            th = {}
            for t in args.trees:
                gt = [r for r in grp if r["key"].endswith(f"_t{t:03d}")]
                fit = fit_half_life(np.concatenate([r["E3"]["distance"] for r in gt]),
                                    np.concatenate([r["E3"]["sensitivity"] for r in gt]))
                th[t] = fit["t_half_free_amplitude"]
            summ[f"{sc}_s{s}"] = {"E1_rel": e1, "E2_mean": mean_c, "E2_identifiable": ident, "E2_rho": rho,
                                  "E2_mtarsus_mean_abs": mt_abs, "E2_ci": ci, "E2_run": run_c, "t_half": th}

    # ---- 判定 ----
    g1, g2, g3 = {}, {}, {}
    for s in args.seeds:
        e, n = summ[f"effect_s{s}"], summ[f"null_s{s}"]
        g1[s] = bool(e["E1_rel"]["ci95"][0] > 0)
        lo, hi = n["E1_rel"]["ci95"]
        g2[s] = bool((lo <= 0 <= hi) or (n["E1_rel"]["gain"] < 0.1 * e["E1_rel"]["gain"]))
        g3[s] = bool(e["E2_rho"] >= 0.7 and n["E2_mtarsus_mean_abs"] < 0.25 * e["E2_mtarsus_mean_abs"])
    passed = all(g1.values()) and all(g2.values()) and all(g3.values())

    def n_sig(level, sc):
        return sum(int(not (v["ci95"][0] <= 0 <= v["ci95"][1])) for s in args.seeds
                   for v in summ[f"{sc}_s{s}"]["E2_ci"][level].values())

    n_null = {lv: n_sig(lv, "null") for lv in ("family", "order")}
    n_tests = len(args.seeds) * len(REG6)
    level = "family" if n_null["family"] <= 0.1 * n_tests else ("order" if n_null["order"] <= 0.1 * n_tests else "threshold")
    # 零模型阈值：零模型每次运行（种子 × 树）各运动方式对比绝对值的 95% 分位数
    null_runs = [abs(v) for s in args.seeds for t in args.trees for v in summ[f"null_s{s}"]["E2_run"][t].values()]
    thr = float(np.percentile(null_runs, 95))
    check = {}
    for s in args.seeds:
        if level == "threshold":  # 与真实数据相同的统计量：全部运行的逐窗口对比合并后的均值
            m = summ[f"effect_s{s}"]["E2_ci"]["family"]
            check[s] = {"wading": bool(m["wading"]["mean"] > thr), "aerial": bool(m["aerial"]["mean"] < -thr)}
        else:
            c = summ[f"effect_s{s}"]["E2_ci"][level]
            check[s] = {"wading": bool(c["wading"]["ci95"][0] > 0), "aerial": bool(c["aerial"]["ci95"][1] < 0)}
    th_all = {sc: [v for s in args.seeds for v in summ[f"{sc}_s{s}"]["t_half"].values()] for sc in ("effect", "null")}
    e3_ref = {sc: float(np.nanmedian(v)) for sc, v in th_all.items()}
    bg = {k: truth0.get(k) for k in ("t_half_myr", "background_rate0", "background_eb")}
    decision = {"pass": bool(passed), "G1": g1, "G2": g2, "G3": g3, "e2_ci_level": level, "e2_null_threshold": thr,
                "null_sig_count": n_null, "n_null_tests": n_tests, "effect_check": check,
                "e3_sim_t_half_median": e3_ref, "sim_process": bg, "trees": args.trees, "seeds": args.seeds}
    (out / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "stage0.json").write_text(json.dumps(strip_boot(summ), ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    yn = lambda b: "通过" if b else "未通过"  # noqa: E731
    level_zh = {"family": "按科重抽样", "order": "按目重抽样", "threshold": f"零模型阈值 {thr:.4f}"}[level]
    L = ["# 真实数据主分析 v1 第 0 阶段：树不对时的模拟校准", "",
         "预注册 `docs/prereg/real_main_v1.md` 第 4 节。数据在 Hackett 第 1 棵树上生成（习性的 OU 响应 t½ = "
         f"{bg['t_half_myr']} Myr，同 v4；另加早期爆发布朗背景，根处速率 {bg['background_rate0']}，b = {bg['background_eb']}），"
         "模型在第 " + "、".join(map(str, args.trees)) + " 棵树上训练（各 5 折）。数值单位 log10；"
         f"E1 增益 = 无习性 RMSE − 有习性 RMSE；置信区间按科自助重抽样 {N_BOOT} 次。", "",
         f"**结论：{yn(passed)}**（G1 {yn(all(g1.values()))}，G2 {yn(all(g2.values()))}，G3 {yn(all(g3.values()))}）", "",
         f"- E2 的判定方式：**{level_zh}**"
         f"（零模型 {n_tests} 个对比中区间不含 0 的个数：按科 {n_null['family']}，按目 {n_null['order']}）",
         f"- E3 参照（每棵树合并拟合的 t½ 中位数）：效应 {e3_ref['effect']:.1f} Myr，零模型 {e3_ref['null']:.1f} Myr", "",
         "## E1 相对跗跖增益（5 棵树、5 折合并）", "", "| 模型 | 增益 | 95% CI | 科数 |", "|---|---|---|---|"]
    for k, v in summ.items():
        e = v["E1_rel"]
        L.append(f"| {k} | {e['gain']:+.4f} | [{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}] | {e['n_clusters']} |")
    L += ["", "## E2 相对跗跖对比（树与折平均）", "",
          "| 模型 | ρ | m_tarsus 平均绝对值 | 地栖 / 空中 / 涉水 / 潜水 / 水面 / 泛化 |", "|---|---|---|---|"]
    for k, v in summ.items():
        rho = "不适用" if v["E2_rho"] != v["E2_rho"] else f"{v['E2_rho']:.3f}"
        L.append(f"| {k} | {rho} | {v['E2_mtarsus_mean_abs']:.4f} | "
                 + " / ".join(f"{v['E2_mean'][x]['m_tarsus']:+.3f}" for x in REG6) + " |")
    for lv, zh in (("family", "科"), ("order", "目")):
        L += ["", f"## E2 逐窗口对比的 95% 区间（按{zh}重抽样）", "", "| 模型 | " + " | ".join(ZH[x] for x in REG6) + " |",
              "|---" * 7 + "|"]
        for k, v in summ.items():
            L.append(f"| {k} | " + " | ".join(f"[{v['E2_ci'][lv][x]['ci95'][0]:+.3f}, {v['E2_ci'][lv][x]['ci95'][1]:+.3f}]"
                                            for x in REG6) + " |")
    L += ["", f"## 零模型阈值：每次运行的对比绝对值的 95% 分位数 = {thr:.4f}", "",
          "| 模型 | 树 | " + " | ".join(ZH[x] for x in REG6) + " |", "|---" * 8 + "|"]
    for k, v in summ.items():
        for t in args.trees:
            L.append(f"| {k} | {t} | " + " | ".join(f"{v['E2_run'][t][x]:+.4f}" for x in REG6) + " |")
    L += ["", "## E3 半衰期（每棵树合并 5 折拟合；有布朗背景，没有单一真值，只作真实数据的参照）", "", "| 模型 | " + " | ".join(f"树 {t}" for t in args.trees) + " |",
          "|---" * (len(args.trees) + 1) + "|"]
    for k, v in summ.items():
        L.append(f"| {k} | " + " | ".join(f"{v['t_half'][t]:.1f}" for t in args.trees) + " |")
    L += ["", "## 判据明细", "",
          "- G1（效应相对跗跖增益下限 > 0）：" + "，".join(f"种子 {s} {yn(v)}" for s, v in g1.items()),
          "- G2（零模型区间含 0 或小于效应的 10%）：" + "，".join(f"种子 {s} {yn(v)}" for s, v in g2.items()),
          "- G3（ρ ≥ 0.7 且零模型对比 < 效应的 25%）：" + "，".join(f"种子 {s} {yn(v)}" for s, v in g3.items())]
    if check:
        L.append("- 检查（效应情景涉水 > 0、空中 < 0 按所选方式判定为显著）：" + "；".join(
            f"种子 {s} 涉水 {yn(c['wading'])}、空中 {yn(c['aerial'])}" for s, c in check.items()))
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False))
    print(f"报告：{out / 'report.md'}")


if __name__ == "__main__":
    main()
