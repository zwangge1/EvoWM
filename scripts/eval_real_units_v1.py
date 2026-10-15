"""按习性分层的真实数据主分析（预注册 docs/prereg/units_main_v1.md 第 5、6 节）。

三层模型（配置 recovery_v5_hab3）。主分析：Hackett 第 21、41、61、81、100 棵树 × 训练种子 1、2 × 5 折；稳健性：Ericson 第 21、41、61
棵树 × 种子 1 × 5 折。S1、S2 另评估 v2.1 只训第一层的模型（检查点子目录 real_v2，同树、同种子、同折，同一套窗口）。
进入条件：第 0 阶段（outputs/stage0_units_v1/decision.json）的 G1-L2、G2-L2、G-L1 都通过；G3-L2 决定 H4 的地位。
所有读数在观测空间（log10），各运行的预测合并后按簇自助重抽样 10,000 次。
在 EvoWM 根目录运行：python scripts/eval_real_units_v1.py
输出只写到 outputs/real_units_v1/：results.json、report.md。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evowm.eval.pooled import boot_gain, boot_gain_diff, holm, p_one_sided, strip_boot  # noqa: E402
from evowm.eval.runs import N_FOLDS, ckpt_action_mode, run_all, run_unit_task  # noqa: E402
from evowm.eval.units_pool import GENERALIST, run_means, summarize_l1, summarize_level  # noqa: E402

PREREG = ROOT / "docs/prereg/units_main_v1.md"
TABLE = "data/processed/species_traits.csv"
N_BOOT = 10000
ALPHA = 0.025
LEVELS = ["locomotor", "realm"]
MODES = ["terrestrial", "generalist", "aerial", "wading", "diving", "surface"]
H4 = {"H4a": ("wading", +1), "H4b": ("terrestrial", +1), "H4c": ("aerial", -1)}
ZH = {"insessorial": "栖树", "terrestrial": "地栖", "aerial": "空中", "wading": "涉水", "diving": "潜水",
      "surface": "水面", "generalist": "泛化", "water": "水中", "air": "空中（栖境）"}


def ckpt_units(ck: Path) -> list:
    cfg = yaml.safe_load((ck.parent / "config.yaml").read_text(encoding="utf-8")) or {}
    return list(((cfg.get("data") or {}).get("dataset") or {}).get("units", {}).get("levels", []) or [])


def tasks_for(prefix, backbone, trees, seeds, ckpt_root, ckpt_name, required, table_rel, units):
    tasks = []
    for t in trees:
        tree_file = f"data/processed/trees/{backbone}_t{t:03d}.nwk"
        for s in seeds:
            cks = [ckpt_root / f"{backbone}_t{t:03d}_s{s}_f{k}" / f"{ckpt_name}_object.ckpt" for k in range(N_FOLDS)]
            if not (ROOT / tree_file).exists() or not all(c.exists() for c in cks):
                if required:
                    sys.exit(f"缺少 {tree_file} 或 {backbone}_t{t:03d}_s{s} 的模型（{ckpt_root}）")
                print(f"跳过 {prefix} {backbone}_t{t:03d}_s{s}（树或模型缺失）")
                continue
            for k, ck in enumerate(cks):
                if ckpt_action_mode(ck) != "change":
                    sys.exit(f"{ck} 不是用只含习性变化的动作训练的")
                if ckpt_units(ck) != units:
                    sys.exit(f"{ck} 的上层单位是 {ckpt_units(ck)}，应为 {units}")
                tasks.append({"root": str(ROOT), "table_rel": table_rel, "tree_file": tree_file, "fold": k, "seed": s,
                              "ckpt": str(ck), "real": True, "key": f"{prefix}_{backbone}_t{t:03d}_s{s}"})
    return tasks


def subset_gain(r_list, mask_fn):
    """若干运行（5 折）合并后的点估计增益（不重抽样），用于每次运行的结果。"""
    lv = [r["units"]["levels"]["locomotor"]["E1"] for r in r_list]
    m = np.concatenate([mask_fn(e) for e in lv])
    eco = np.concatenate([e["rel_eco"] for e in lv])[m]
    no = np.concatenate([e["rel_no"] for e in lv])[m]
    return float(np.sqrt(np.mean(no ** 2)) - np.sqrt(np.mean(eco ** 2)))


def no_generalist(e):
    return (e["prev_label"] != GENERALIST) & (e["target_label"] != GENERALIST)


def paired_l1(main, l1):
    """S1：三层模型与只训第一层的模型在同一套第一层窗口上的配对比较（运动方式变化对）。"""
    idx = {(r["key"].split("_", 1)[1], r["fold"]): r for r in l1}
    a_e, a_n, b_e, b_n, fam = [], [], [], [], []
    for r in sorted(main, key=lambda r: (r["key"], r["fold"])):
        q = idx.get((r["key"].split("_", 1)[1], r["fold"]))
        if q is None:
            continue
        if len(q["L1"]["family"]) != len(r["L1"]["family"]) or np.any(q["L1"]["family"] != r["L1"]["family"]):
            sys.exit(f"S1 配对失败：{r['key']} 折 {r['fold']} 的第一层窗口与只训第一层的模型不同")
        tr = r["L1"]["transition"]
        for dst, src in ((a_e, r["L1"]["rel_eco"]), (a_n, r["L1"]["rel_no"]), (b_e, q["L1"]["rel_eco"]),
                         (b_n, q["L1"]["rel_no"])):
            dst.append(np.where(tr, src, np.nan))
        fam.append(r["L1"]["family"])
    if not fam:
        return None
    a_e, a_n, b_e, b_n, fam = map(np.concatenate, (a_e, a_n, b_e, b_n, fam))
    return {"hab3": boot_gain(a_e, a_n, fam, N_BOOT, 0), "l1_only": boot_gain(b_e, b_n, fam, N_BOOT, 0),
            "diff": boot_gain_diff(a_e, a_n, b_e, b_n, fam, N_BOOT, 0), "n_pairs": int(np.isfinite(a_e).sum())}


def fmt(d, key="gain"):
    return f"{d[key]:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=TABLE, help="只在测试时换成模拟表")
    ap.add_argument("--hackett-trees", type=int, nargs="+", default=[21, 41, 61, 81, 100])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--ericson-trees", type=int, nargs="*", default=[21, 41, 61])
    ap.add_argument("--ckpt-root", default=str(ROOT / "outputs/hjepa/ckpts/real_units_v1"))
    ap.add_argument("--l1-root", default=str(ROOT / "outputs/hjepa/ckpts/real_v2"))
    ap.add_argument("--stage0", default=str(ROOT / "outputs/stage0_units_v1/decision.json"))
    ap.add_argument("--out", default=str(ROOT / "outputs/real_units_v1"))
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    if not PREREG.exists():
        sys.exit(f"缺少预注册 {PREREG.relative_to(ROOT)}")
    dpath = Path(args.stage0)
    if not dpath.exists():
        sys.exit("缺少第 0 阶段结果，真实数据不能评估")
    decision = json.loads(dpath.read_text(encoding="utf-8"))
    gates = {g: decision.get(g) or {} for g in ("G1_L2", "G2_L2", "G_L1")}
    if not (decision.get("pass") and all(v and all(v.values()) for v in gates.values())):
        sys.exit("第 0 阶段 G1-L2、G2-L2、G-L1 未全部通过，真实数据不能评估（预注册第 4 节）")
    if decision.get("config") != "hab3" or decision.get("primary") != "no_generalist":
        sys.exit("第 0 阶段的配置或 H3 读数与预注册不符（应为 hab3、no_generalist）")
    h4_confirm = bool(all((decision.get("G3_L2") or {"x": False}).values()))
    level = decision.get("e2_ci_level")
    boot_level = level if level in ("family", "order") else None
    thr = float(decision["e2_null_threshold"]) if level == "threshold" else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_root, l1_root = Path(args.ckpt_root), Path(args.l1_root)

    tasks = tasks_for("hab3", "hackett", args.hackett_trees, args.seeds, ckpt_root, "evowm_hab3", True, args.table, LEVELS) \
        + tasks_for("hab3", "ericson", args.ericson_trees, [1], ckpt_root, "evowm_hab3", False, args.table, LEVELS) \
        + tasks_for("l1", "hackett", args.hackett_trees, args.seeds, l1_root, "evowm_l1", False, args.table, [])
    results = run_all(tasks, args.workers, fn=run_unit_task)
    table = pd.read_csv(ROOT / args.table, usecols=["family", "order"])
    f2o = dict(zip(table["family"], table["order"]))

    main_ = [r for r in results if r["key"].startswith("hab3_hackett_")]
    eric = [r for r in results if r["key"].startswith("hab3_ericson_")]
    l1 = [r for r in results if r["key"].startswith("l1_hackett_")]
    S = summarize_l1(main_, N_BOOT)
    S["locomotor"] = summarize_level(main_, "locomotor", N_BOOT, None, f2o)
    S["realm"] = summarize_level(main_, "realm", N_BOOT, None, f2o)
    runs = sorted({r["key"] for r in main_})
    S["n_runs"] = len(runs)
    lo = S["locomotor"]
    h3 = lo["subsets"]["no_generalist"]
    e2 = lo["E2_order"] if boot_level == "order" else lo["E2"]
    per_run_e2 = run_means(main_, "locomotor", MODES)
    per_run_h3 = {k: {"no_generalist": subset_gain([r for r in main_ if r["key"] == k], no_generalist),
                      "all": subset_gain([r for r in main_ if r["key"] == k], lambda e: np.ones(len(e["rel_eco"]), bool))}
                  for k in runs}

    # ---- 判定用检验（预注册第 5 节）----
    pvals = {"H3": p_one_sided(h3["boot"], +1)}
    if h4_confirm and boot_level:
        for h, (x, d) in H4.items():
            pvals[h] = p_one_sided(e2[x]["boot"], d)
    rej = holm(pvals, ALPHA)
    verdict = {"H3": "支持" if rej["H3"] else "未检出"}
    consistency = {}
    for h, (x, d) in H4.items():
        consistency[h] = float(np.mean([np.sign(per_run_e2[k][x]) == d for k in runs]))
        est, (lo_, hi_) = e2[x]["mean"], e2[x]["ci95"]
        if not h4_confirm:
            if thr is not None:
                desc = ("超过零模型阈值，方向与预期一致" if d * est > thr else
                        ("超过零模型阈值，方向与预期相反" if -d * est > thr else "未超过零模型阈值"))
            else:
                desc = "方向与预期一致" if np.sign(est) == d else "方向与预期相反"
            verdict[h] = "描述（不判定）：" + desc
        elif thr is not None:
            verdict[h] = "支持" if d * est > thr else ("相反" if -d * est > thr else "不确定")
        elif rej[h]:
            verdict[h] = "支持"
        elif (d > 0 and hi_ < 0) or (d < 0 and lo_ > 0):
            verdict[h] = "相反"
        else:
            verdict[h] = "不确定"

    s1 = paired_l1(main_, l1) if l1 else None
    s2_l1 = summarize_l1(l1, N_BOOT) if l1 else None
    robust = None
    if eric:
        rl = summarize_level(eric, "locomotor", N_BOOT, None, f2o)
        robust = {"n_runs": len({r["key"] for r in eric}), "H3": rl["subsets"]["no_generalist"], "all": rl["E1_rel"],
                  "E2": {m: rl["E2"][m]["mean"] for m in MODES if m in rl["E2"]}}

    res = {"stage0": decision, "h4_confirmatory": h4_confirm,
           "tests": {"p_one_sided": pvals, "holm_reject": rej, "verdict": verdict, "direction_consistency": consistency},
           "main": S, "per_run": {"H3": per_run_h3, "E2": per_run_e2}, "S1": s1, "S2_l1_only": s2_l1,
           "robust_ericson": robust}
    (out / "results.json").write_text(json.dumps(strip_boot(res), ensure_ascii=False, indent=2, default=float),
                                      encoding="utf-8")
    write_report(out, res, S, args, level, thr, boot_level, h4_confirm, pvals, verdict, consistency)
    print("结论：", verdict)
    print(f"报告：{out / 'report.md'}")


def write_report(out, res, S, args, level, thr, boot_level, h4_confirm, pvals, verdict, consistency):
    lo, re_ = S["locomotor"], S["realm"]
    h3 = lo["subsets"]["no_generalist"]
    e2 = lo["E2_order"] if boot_level == "order" else lo["E2"]
    lvzh = {"family": "按科重抽样", "order": "按目重抽样", "threshold": f"零模型阈值 ±{thr or 0:.4f}（不进 Holm）"}.get(level, "未知")
    L = ["# 按习性分层的真实数据主分析 v1 结果（三层模型）", "",
         f"预注册 `docs/prereg/units_main_v1.md`。Hackett 第 {'、'.join(map(str, args.hackett_trees))} 棵树 × 训练种子 "
         f"{'、'.join(map(str, args.seeds))} × 5 折，共 {S['n_runs']} 次运行合并。数值单位 log10。第 0 阶段：G1-L2、G2-L2、G-L1 通过；G3-L2 "
         + ("通过，H4 为确认性检验" if h4_confirm else "未通过，H4 只作描述，H3 单独检验") + f"；E2 判定方式为{lvzh}。", "",
         ("## 判定用检验（Holm，总体 α = 0.025 单侧）" if (h4_confirm and boot_level) else
          "## 判定用检验（H3 单独检验，α = 0.025 单侧）"), "",
         "| 假设 | 读数 | 估计 | 95% CI | 单侧 p | 结论 | 各运行方向一致比例 |", "|---|---|---|---|---|---|---|",
         f"| H3 运动方式转变的可迁移预测价值（第二层） | 不涉及泛化的第二层步上的相对跗跖增益 | {h3['gain']:+.4f} | "
         f"[{h3['ci95'][0]:+.4f}, {h3['ci95'][1]:+.4f}] | {pvals['H3']:.4f} | {verdict['H3']} | |"]
    for h, (x, d) in H4.items():
        e = e2[x]
        p = f"{pvals[h]:.4f}" if h in pvals else "无"
        L.append(f"| {h} {ZH[x]}{'＞' if d > 0 else '＜'}栖树 | 第二层反事实相对跗跖对比 | {e['mean']:+.4f} | "
                 f"[{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}] | {p} | {verdict[h]} | {consistency[h]:.0%} |")
    flag = [h for h, c in consistency.items() if c < 0.8]
    if flag:
        L.append(f"\n各运行方向一致比例低于 80%：{'、'.join(flag)}，结论注明随树或种子不稳定。")
    chk = res["stage0"].get("effect_check") or {}
    if chk and not all(all(v.values()) for v in chk.values()):
        L.append("\n第 0 阶段的检查（效应情景涉水 > 0、空中 < 0 按所选方式判为显著）没有全部通过：H4 的\"不确定\""
                 "注明\"模拟中这个检验功效不足\"（预注册第 4 节）。")
    L += ["", "## 第二层明细", "", "| 读数 | 增益 [95% CI] |", "|---|---|",
          f"| 不涉及泛化的转变（H3，占第二层步的 {1 - lo['share_generalist']:.0%}） | {fmt(h3)} |",
          f"| 全部转变（次要） | {fmt(lo['E1_rel'])} |",
          f"| 体重（诊断） | {fmt(lo['E1_m_mass'])} |",
          f"| S4 目标段只有 1 种 | {fmt(lo['subsets']['single'])} |", f"| S4 目标段多于 1 种 | {fmt(lo['subsets']['multi'])} |",
          f"| S4 目标段 2 到 4 种 | {fmt(lo['subsets']['len_2_4'])} |", f"| S4 目标段 5 到 16 种 | {fmt(lo['subsets']['len_5_16'])} |",
          f"| S4 目标段超过 16 种 | {fmt(lo['subsets']['len_gt16'])} |",
          f"| S3 线性基线，全部转变 | {fmt(lo['linear_gain'])} |",
          f"| S3 线性基线，不涉及泛化 | {fmt(lo['linear_gain_no_generalist'])} |",
          f"| S3 世界模型 − 线性基线，全部转变 | {fmt(lo['wm_minus_linear'], 'diff')} |",
          f"| S3 世界模型 − 线性基线，不涉及泛化 | {fmt(lo['wm_minus_linear_no_generalist'], 'diff')} |"]
    L += ["", "每次运行（5 折合并的点估计）：", "", "| 运行 | 不涉及泛化 | 全部转变 |", "|---|---|---|"]
    for k, v in res["per_run"]["H3"].items():
        L.append(f"| {k} | {v['no_generalist']:+.4f} | {v['all']:+.4f} |")
    L += ["", f"## 第二层反事实对比（相对跗跖，相对栖树；区间按{'目' if boot_level == 'order' else '科'}重抽样"
          + (f"，判定用零模型阈值 ±{thr:.4f}" if thr is not None else "") + "）", "",
          "| 运动方式 | 对比 | 95% CI | 各运行估计的范围 |", "|---|---|---|---|"]
    for x in MODES:
        e = e2[x]
        pr = [v[x] for v in res["per_run"]["E2"].values()]
        L.append(f"| {ZH[x]} | {e['mean']:+.4f} | [{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}] | {min(pr):+.4f} 到 {max(pr):+.4f} |")
    L += ["", "## 第三层（栖境，只作描述）", "", "| 读数 | 增益 [95% CI] |", "|---|---|",
          f"| 相对跗跖 | {fmt(re_['E1_rel'])} |", f"| 体重 | {fmt(re_['E1_m_mass'])} |", f"| HWI | {fmt(re_['E1_m_hwi'])} |",
          "", "| 反事实（相对陆地） | 相对跗跖 | 体重 | HWI |", "|---|---|---|---|"]
    for m in ("water", "air"):
        if m in re_["E2"]:
            L.append(f"| {ZH[m]} | {fmt(re_['E2'][m], 'mean')} | {fmt(re_['E2_m_mass'][m], 'mean')} | {fmt(re_['E2_m_hwi'][m], 'mean')} |")
    L += ["", "## S1、S2：第一层", ""]
    if res["S1"]:
        s1 = res["S1"]
        L += [f"- S1 运动方式变化对上的相对跗跖增益：三层模型 {fmt(s1['hab3'])}，只训第一层（v2.1）{fmt(s1['l1_only'])}，"
              f"配对差 {fmt(s1['diff'], 'diff')}（{s1['n_pairs']} 个预测）"]
    else:
        L.append("- S1：没有找到 v2.1 只训第一层的模型，跳过")
    L.append(f"- S2 运动方式相同的相邻对：三层模型 {fmt(S['L1_within'])}"
             + (f"，只训第一层（v2.1）{fmt(res['S2_l1_only']['L1_within'])}" if res["S2_l1_only"] else ""))
    L.append(f"- 三层模型第一层的全部相邻对 {fmt(S['L1_all'])}")
    s5 = lo["S5"]
    L += ["", "## S5：第二层对第一层（预测留出单位的平均相对跗跖）", "",
          f"RMSE：第二层 {s5['rmse_l2']:.4f}，第一层直跳 {s5['rmse_jump']:.4f}，第一层连推 {s5['rmse_roll']:.4f}（{s5['n_units']} 个单位）；"
          f"直跳 − 第二层 {fmt(s5['jump_minus_l2'])}，连推 − 第二层 {fmt(s5['roll_minus_l2'])}（正值表示第二层更准）。"]
    rb = res["robust_ericson"]
    if rb:
        L += ["", f"## 稳健性：Ericson 树（{rb['n_runs']} 次运行，不判定）", "",
              f"- H3 读数 {fmt(rb['H3'])}；全部转变 {fmt(rb['all'])}",
              "- 第二层反事实：" + "，".join(f"{ZH[m]} {v:+.4f}" for m, v in rb["E2"].items())]
    L += ["", "## 探针（留出单位上的 R²，各运行平均）", "", "| 层 | m_tarsus | m_mass | m_hwi |", "|---|---|---|---|"]
    for name, v in (("第二层", lo), ("第三层", re_)):
        L.append(f"| {name} | " + " | ".join(f"{v['probe_r2'].get(c, float('nan')):.3f}" for c in ("m_tarsus", "m_mass", "m_hwi")) + " |")
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
