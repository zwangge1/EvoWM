"""按习性分层的第 0 阶段：树不对时的模拟校准（预注册 docs/prereg/units_main_v1.md 第 4 节）。

与第一层 v2 的第 0 阶段（scripts/eval_stage0_v2.py）同样的做法：数据在 Hackett 第 1 棵树上生成
（sim_recovery_tables.py --version v5，新种子 13、14，带早期爆发布朗背景），模型在第 21、41、61、81、100 棵树上训练（各 5 折）。
判据：
- G1-L2：效应情景两个种子，H3 读数（第 2 层相对跗跖增益；--primary 选全部第 2 层步或不涉及泛化的子集）的 95% 区间下限 > 0；
- G2-L2：零模型同一读数区间含 0，或小于效应的 10%；
- G-L1：联合训练后第一层仍满足 v2 的 G1、G2（运动方式变化对上的相对跗跖增益）；
- G3-L2（只决定 H4 的地位）：第 2 层反事实对比与真值的 ρ ≥ 0.7，且零模型对比平均绝对值 < 效应的 25%。
G1-L2、G2-L2、G-L1 都通过才进入真实数据。E2 的判定方式按 v2 的固定规则（按科、按目重抽样或零模型阈值）写进 decision.json。
三层模型（--config hab3），H3 读数为不涉及泛化的第二层步（--primary no_generalist），都按预注册固定为默认值。
在 EvoWM 根目录运行：python scripts/eval_stage0_units_v1.py
输出只写到 outputs/stage0_units_v1/：decision.json、stage0.json、report.md。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from evowm.eval.pooled import strip_boot  # noqa: E402
from evowm.eval.runs import N_FOLDS, ckpt_action_mode, run_all, run_unit_task  # noqa: E402
from evowm.eval.units_pool import PRIMARY, g2_ok, h3_readout, run_means, summarize_l1, summarize_level  # noqa: E402

PREREG = ROOT / "docs/prereg/units_main_v1.md"
N_BOOT = 10000
CKPT_NAME = {"hab2": "evowm_hab2", "hab3": "evowm_hab3"}
UNITS = {"hab2": ["locomotor"], "hab3": ["locomotor", "realm"]}
MODES = ["terrestrial", "generalist", "aerial", "wading", "diving", "surface"]
ZH = {"terrestrial": "地栖", "aerial": "空中", "wading": "涉水", "diving": "潜水", "surface": "水面", "generalist": "泛化"}


def table_rel(sc, s):
    return f"data/processed/sim/recovery_v5/{sc}_s{s}/species_traits.csv"


def fmt(g, k="gain"):
    return f"{g[k]:+.4f} [{g['ci95'][0]:+.4f}, {g['ci95'][1]:+.4f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=list(CKPT_NAME), default="hab3")
    ap.add_argument("--primary", choices=list(PRIMARY), default="no_generalist")
    ap.add_argument("--seeds", type=int, nargs="+", default=[13, 14])
    ap.add_argument("--trees", type=int, nargs="+", default=[21, 41, 61, 81, 100])
    ap.add_argument("--ckpt-root", default=str(ROOT / "outputs/hjepa/ckpts/stage0_units_v1"))
    ap.add_argument("--out", default=str(ROOT / "outputs/stage0_units_v1"))
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    if not PREREG.exists():
        sys.exit(f"缺少预注册 {PREREG.relative_to(ROOT)}")
    out, ckpt_root = Path(args.out), Path(args.ckpt_root)
    out.mkdir(parents=True, exist_ok=True)

    tasks = []
    for s in args.seeds:
        for sc in ("effect", "null"):
            for t in args.trees:
                for k in range(N_FOLDS):
                    ck = ckpt_root / f"{sc}_s{s}_t{t:03d}_f{k}" / f"{CKPT_NAME[args.config]}_object.ckpt"
                    if not ck.exists():
                        sys.exit(f"缺少模型 {ck}")
                    cfg = yaml.safe_load((ck.parent / "config.yaml").read_text(encoding="utf-8")) or {}
                    lv = (((cfg.get("data") or {}).get("dataset") or {}).get("units") or {}).get("levels")
                    if ckpt_action_mode(ck) != "change" or list(lv or []) != UNITS[args.config]:
                        sys.exit(f"{ck} 的动作方式或上层单位与配置 {args.config} 不符")
                    tasks.append({"root": str(ROOT), "table_rel": table_rel(sc, s), "fold": k, "seed": s,
                                  "tree_file": f"data/processed/trees/hackett_t{t:03d}.nwk", "ckpt": str(ck),
                                  "key": f"{sc}_s{s}_t{t:03d}"})
    results = run_all(tasks, args.workers, fn=run_unit_task)
    has_l3 = "realm" in results[0]["units"]["levels"]

    summ, runs = {}, {}
    for s in args.seeds:
        for sc in ("effect", "null"):
            grp = [r for r in results if r["key"].startswith(f"{sc}_s{s}_")]
            table = pd.read_csv(ROOT / table_rel(sc, s), usecols=["family", "order"])
            f2o = dict(zip(table["family"], table["order"]))
            truth = json.loads((ROOT / table_rel(sc, s)).with_name("truth.json").read_text(encoding="utf-8"))
            v = summarize_l1(grp, N_BOOT)
            v["locomotor"] = summarize_level(grp, "locomotor", N_BOOT, truth, f2o)
            if has_l3:
                v["realm"] = summarize_level(grp, "realm", N_BOOT, None, f2o)
            summ[f"{sc}_s{s}"] = v
            runs[f"{sc}_s{s}"] = run_means(grp, "locomotor", MODES)

    # ---- 判定 ----
    g1, g2, gl1, g3, h3v = {}, {}, {}, {}, {}
    for s in args.seeds:
        e, n = summ[f"effect_s{s}"], summ[f"null_s{s}"]
        he, hn = h3_readout(e["locomotor"], args.primary), h3_readout(n["locomotor"], args.primary)
        h3v[s] = {sc: [x["gain"], *x["ci95"]] for sc, x in (("effect", he), ("null", hn))}
        g1[s] = bool(he["ci95"][0] > 0)
        g2[s] = g2_ok(hn, he)
        gl1[s] = bool(e["L1_transition"]["ci95"][0] > 0 and g2_ok(n["L1_transition"], e["L1_transition"]))
        g3[s] = bool(e["locomotor"]["E2_rho"] >= 0.7
                     and n["locomotor"]["E2_mean_abs"] < 0.25 * e["locomotor"]["E2_mean_abs"])
    passed = all(g1.values()) and all(g2.values()) and all(gl1.values())

    def n_sig(key, sc):
        return sum(int(not (x["ci95"][0] <= 0 <= x["ci95"][1])) for s in args.seeds
                   for x in summ[f"{sc}_s{s}"]["locomotor"][key].values())
    n_null = {"family": n_sig("E2", "null"), "order": n_sig("E2_order", "null")}
    n_tests = len(args.seeds) * len(MODES)
    level = "family" if n_null["family"] <= 0.1 * n_tests else ("order" if n_null["order"] <= 0.1 * n_tests else "threshold")
    null_runs = [abs(x) for s in args.seeds for rm in runs[f"null_s{s}"].values() for x in rm.values()]
    thr = float(np.percentile(null_runs, 95))
    check = {}
    for s in args.seeds:
        if level == "threshold":
            m = summ[f"effect_s{s}"]["locomotor"]["E2"]
            check[s] = {"wading": bool(m["wading"]["mean"] > thr), "aerial": bool(m["aerial"]["mean"] < -thr)}
        else:
            c = summ[f"effect_s{s}"]["locomotor"]["E2" if level == "family" else "E2_order"]
            check[s] = {"wading": bool(c["wading"]["ci95"][0] > 0), "aerial": bool(c["aerial"]["ci95"][1] < 0)}
    decision = {"pass": bool(passed), "config": args.config, "primary": args.primary,
                "G1_L2": g1, "G2_L2": g2, "G_L1": gl1, "G3_L2": g3, "H3_readout": h3v,
                "e2_ci_level": level, "e2_null_threshold": thr,
                "null_sig_count": n_null, "n_null_tests": n_tests, "effect_check": check,
                "trees": args.trees, "seeds": args.seeds, "has_L3": has_l3}
    (out / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "stage0.json").write_text(json.dumps(strip_boot({"summary": summ, "runs": runs}), ensure_ascii=False,
                                                indent=2, default=float), encoding="utf-8")

    yn = lambda b: "通过" if b else "未通过"  # noqa: E731
    L = ["# 按习性分层的第 0 阶段：树不对时的模拟校准", "",
         f"配置 {args.config}；H3 读数：{PRIMARY[args.primary]}。数据在 Hackett 第 1 棵树上生成，模型在第 "
         + "、".join(map(str, args.trees)) + f" 棵树上训练（各 5 折）。数值单位 log10；区间按科自助重抽样 {N_BOOT} 次。", "",
         f"**结论：{yn(passed)}**（G1-L2 {yn(all(g1.values()))}，G2-L2 {yn(all(g2.values()))}，G-L1 {yn(all(gl1.values()))}；"
         f"G3-L2 {yn(all(g3.values()))}，只决定 H4 的地位）", "",
         f"- E2 的判定方式：{ {'family': '按科重抽样', 'order': '按目重抽样', 'threshold': '零模型阈值'}[level] }（零模型 {n_tests} 个对比中区间不含 0 的个数：按科 {n_null['family']}，按目 {n_null['order']}；"
         f"零模型阈值 {thr:.4f}）", "",
         "| 模型 | H3 读数 | 全部第 2 层步 | 不涉及泛化 | 第 1 层变化对（G-L1） | E2 ρ | E2 平均绝对值 |", "|---|---|---|---|---|---|---|"]
    for k, v in summ.items():
        lo = v["locomotor"]
        rho = lo.get("E2_rho")
        L.append(f"| {k} | {fmt(h3_readout(lo, args.primary))} | {fmt(lo['E1_rel'])} | {fmt(lo['subsets']['no_generalist'])} | "
                 f"{fmt(v['L1_transition'])} | {'' if rho is None else f'{rho:.3f}'} | {lo['E2_mean_abs']:.4f} |")
    L += ["", "## 第 2 层反事实对比（相对跗跖，相对栖树，树与折合并）", "", "| 模型 | " + " | ".join(ZH[m] for m in MODES) + " |",
          "|---" * (len(MODES) + 1) + "|"]
    for k, v in summ.items():
        L.append(f"| {k} | " + " | ".join(f"{fmt(v['locomotor']['E2'][m], 'mean')}" for m in MODES) + " |")
    if has_l3:
        L += ["", "## 第 3 层（只作描述）", "", "| 模型 | 相对跗跖增益 | 体重增益 | HWI 增益 |", "|---|---|---|---|"]
        for k, v in summ.items():
            r = v["realm"]
            L.append(f"| {k} | {fmt(r['E1_rel'])} | {fmt(r['E1_m_mass'])} | {fmt(r['E1_m_hwi'])} |")
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False))
    print(f"报告：{out / 'report.md'}")


if __name__ == "__main__":
    main()
