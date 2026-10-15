"""开发检查（探索性，不属于任何预注册）：按习性尺度分层（设计草案 docs/design/第二层设计草案_2026-10-15.md 第 5 到 7 节）。

第 0 阶段的模拟表（种子 9、10，带早期爆发背景，已在开发检查 dev_change_v1 用过），第 21 棵树，效应与零模型各 5 折。
三组模型：两层（recovery_v5_hab2）、三层（recovery_v5_hab3）、只训第 1 层（recovery_v5_l1，dev_change_v1 的模型）。
报告（只作开发，不作判定）：
- H3 的读数：第 2 层在运动方式转变处的相对跗跖增益（全部第 2 层步；不涉及"泛化"的子集）；
- 第 1 层读数（运动方式变化对上的相对跗跖增益）在联合训练前后的变化（G-L1、S1），运动方式相同的相邻对（S2）；
- 第 2 层的反事实对比（H4 的读数）、线性基线（S3）、子集（S4）、第 2 层对第 1 层（S5）；
- 第 3 层的读数（只作描述）与第 3 层的去留规则（设计草案第 6 节）。
在 EvoWM 根目录运行：python scripts/exploratory/eval_dev_units_v1.py
输出只写到 outputs/dev_units_v1/：decision.json、summary.json、report.md。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from evowm.eval.pooled import boot_gain, boot_gain_diff, boot_mean, strip_boot  # noqa: E402
from evowm.eval.runs import N_FOLDS, run_all, run_unit_task  # noqa: E402

N_BOOT = 10000
CONFIGS = {"hab2": ("dev_units_v1/hab2", "evowm_hab2"), "hab3": ("dev_units_v1/hab3", "evowm_hab3"),
           "l1": ("dev_change_v1", "evowm_l1")}
CFG_ZH = {"hab2": "两层", "hab3": "三层", "l1": "只训第 1 层"}
MODES = ["terrestrial", "generalist", "aerial", "wading", "diving", "surface"]
ZH = {"terrestrial": "地栖", "aerial": "空中", "wading": "涉水", "diving": "潜水", "surface": "水面", "generalist": "泛化",
      "water": "水中", "air": "空中（栖境）"}
GENERALIST = 2       # evowm.data.units.LOCOMOTOR 里"泛化"的下标
KEEP_RATIO = 0.75    # 设计草案第 6 节：加入第 3 层后，第 1、2 层读数不低于两层模型的 75%


def table_rel(sc, s):
    return f"data/processed/sim/recovery_v5/{sc}_s{s}/species_traits.csv"


def g2_ok(null, effect):
    """G2 的写法：零模型区间含 0，或小于效应的 10%。"""
    lo, hi = null["ci95"]
    return bool((lo <= 0 <= hi) or (null["gain"] < 0.1 * effect["gain"]))


def cat(grp, get):
    return np.concatenate([get(r) for r in grp])


def summarize(grp, truth):
    """一组（配置 × 情景 × 种子，或合并种子）的 5 折读数。"""
    out = {}
    tr = cat(grp, lambda r: r["L1"]["transition"])
    fam1 = cat(grp, lambda r: r["L1"]["family"])
    re1, rn1 = cat(grp, lambda r: r["L1"]["rel_eco"]), cat(grp, lambda r: r["L1"]["rel_no"])
    nan = np.full(len(fam1), np.nan)
    out["L1_transition"] = boot_gain(np.where(tr, re1, nan), np.where(tr, rn1, nan), fam1, N_BOOT, 0)
    out["L1_within"] = boot_gain(np.where(~tr, re1, nan), np.where(~tr, rn1, nan), fam1, N_BOOT, 0)
    out["L1_all"] = boot_gain(re1, rn1, fam1, N_BOOT, 0)
    if "units" not in grp[0]:
        return out
    for name in grp[0]["units"]["levels"]:
        lv = [r["units"]["levels"][name] for r in grp]
        e1 = {k: np.concatenate([v["E1"][k] for v in lv]) for k in ("rel_eco", "rel_no", "family", "prev_label",
                                                                     "target_label", "target_len", "err_eco", "err_no")}
        targets = grp[0]["units"]["targets"]
        f = e1["family"]
        s = {"E1_rel": boot_gain(e1["rel_eco"], e1["rel_no"], f, N_BOOT, 0), "n_pred": int(len(f)),
             "probe_r2": {c: float(np.nanmean([v["probe_r2_val_units"][c] for v in lv])) for c in targets}}
        for c in ("m_mass", "m_hwi"):
            if c in targets:
                j = targets.index(c)
                s[f"E1_{c}"] = boot_gain(e1["err_eco"][:, j], e1["err_no"][:, j], f, N_BOOT, 0)
        nan = np.full(len(f), np.nan)
        subsets = {"single": e1["target_len"] == 1, "multi": e1["target_len"] > 1,
                   "len_2_4": (e1["target_len"] >= 2) & (e1["target_len"] <= 4), "len_5_16": (e1["target_len"] >= 5) & (e1["target_len"] <= 16),
                   "len_gt16": e1["target_len"] > 16}
        if name == "locomotor":
            subsets["no_generalist"] = (e1["prev_label"] != GENERALIST) & (e1["target_label"] != GENERALIST)
            s["share_generalist"] = float(np.mean(~subsets["no_generalist"]))
        s["subsets"] = {k: boot_gain(np.where(m, e1["rel_eco"], nan), np.where(m, e1["rel_no"], nan), f, N_BOOT, 0)
                        for k, m in subsets.items()}
        lin_no = np.concatenate([v["linear"]["err_no"] for v in lv])
        lin_ch = np.concatenate([v["linear"]["err_change"] for v in lv])
        s["linear_gain"] = boot_gain(lin_ch, lin_no, f, N_BOOT, 0)
        s["wm_minus_linear"] = boot_gain_diff(e1["rel_eco"], e1["rel_no"], lin_ch, lin_no, f, N_BOOT, 0)
        f2 = np.concatenate([v["E2"]["family"] for v in lv])
        modes = list(lv[0]["E2"]["rel"])
        s["E2"] = {m: boot_mean(np.concatenate([v["E2"]["rel"][m] for v in lv]), f2, N_BOOT, 0) for m in modes}
        s["E2_mean_abs"] = float(np.mean([abs(s["E2"][m]["mean"]) for m in modes]))
        if name == "locomotor" and truth is not None:
            off = truth["leg_offsets"]["L_tmt"]
            tv = [off[m] - off["insessorial"] for m in modes]
            s["E2_rho"] = float(spearmanr([s["E2"][m]["mean"] for m in modes], tv).statistic) if np.ptp(tv) > 0 else float("nan")
        if name == "realm":
            for c in ("m_mass", "m_hwi"):
                if c in targets:
                    j = targets.index(c)
                    s[f"E2_{c}"] = {m: boot_mean(np.concatenate([v["E2"]["targets"][m][:, j] for v in lv]), f2, N_BOOT, 0)
                                    for m in modes}
        if name == "locomotor":
            s5 = {k: np.concatenate([v["S5"][k] for v in lv]) for k in ("err_l2", "err_jump", "err_roll", "family")}
            s["S5"] = {"rmse_l2": float(np.sqrt(np.mean(s5["err_l2"] ** 2))),
                       "rmse_jump": float(np.sqrt(np.mean(s5["err_jump"] ** 2))),
                       "rmse_roll": float(np.sqrt(np.mean(s5["err_roll"] ** 2))),
                       "jump_minus_l2": boot_gain(s5["err_l2"], s5["err_jump"], s5["family"], N_BOOT, 0),
                       "roll_minus_l2": boot_gain(s5["err_l2"], s5["err_roll"], s5["family"], N_BOOT, 0),
                       "n_units": int(len(s5["family"]))}
        out[name] = s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[9, 10])
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(N_FOLDS)))
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--ckpt-root", default=str(ROOT / "outputs/hjepa/ckpts"))
    ap.add_argument("--out", default=str(ROOT / "outputs/dev_units_v1"))
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tasks, missing = [], []
    for cfg in args.configs:
        sub, name = CONFIGS[cfg]
        for s in args.seeds:
            for sc in ("effect", "null"):
                for k in args.folds:
                    ck = Path(args.ckpt_root) / sub / f"{sc}_s{s}_t021_f{k}" / f"{name}_object.ckpt"
                    if not ck.exists():
                        missing.append(str(ck))
                        continue
                    tasks.append({"root": str(ROOT), "table_rel": table_rel(sc, s), "fold": k, "seed": s,
                                  "tree_file": "data/processed/trees/hackett_t021.nwk", "ckpt": str(ck),
                                  "key": f"{cfg}_{sc}_s{s}"})
    if missing:
        print(f"缺少 {len(missing)} 个模型（相应的比较跳过）：", *missing[:5], sep="\n  ")
    results = run_all(tasks, args.workers, fn=run_unit_task)

    summ = {}
    for cfg in args.configs:
        for sc in ("effect", "null"):
            truth = json.loads((ROOT / table_rel(sc, args.seeds[0])).with_name("truth.json").read_text(encoding="utf-8"))
            for s in args.seeds + ["pooled"]:
                grp = [r for r in results if r["key"] == f"{cfg}_{sc}_s{s}"] if s != "pooled" else \
                      [r for r in results if r["key"].startswith(f"{cfg}_{sc}_s")]
                if grp:
                    summ[f"{cfg}_{sc}_s{s}"] = summarize(grp, truth if sc == "effect" else None)

    # ---- 开发判断（不作判定）----
    def get(cfg, sc, s, *path):
        v = summ.get(f"{cfg}_{sc}_s{s}")
        for p in path:
            if v is None:
                return None
            v = v.get(p)
        return v

    dec = {"seeds": args.seeds, "folds": args.folds}
    for cfg in ("hab2", "hab3"):
        if get(cfg, "effect", args.seeds[0], "locomotor") is None:
            continue
        d = {}
        for key, path in (("H3_all", ("locomotor", "E1_rel")), ("H3_no_generalist", ("locomotor", "subsets", "no_generalist"))):
            g1 = {s: bool(get(cfg, "effect", s, *path)["ci95"][0] > 0) for s in args.seeds}
            g2 = {s: g2_ok(get(cfg, "null", s, *path), get(cfg, "effect", s, *path)) for s in args.seeds}
            d[key] = {"G1": g1, "G2": g2, "pass": bool(all(g1.values()) and all(g2.values()))}
        gl1 = {s: bool(get(cfg, "effect", s, "L1_transition")["ci95"][0] > 0
                       and g2_ok(get(cfg, "null", s, "L1_transition"), get(cfg, "effect", s, "L1_transition")))
               for s in args.seeds}
        d["G_L1"] = {"per_seed": gl1, "pass": bool(all(gl1.values()))}
        e, n = get(cfg, "effect", "pooled", "locomotor", "E2_mean_abs"), get(cfg, "null", "pooled", "locomotor", "E2_mean_abs")
        d["E2_null_ratio"] = float(n / e) if e else float("nan")
        d["E2_rho"] = get(cfg, "effect", "pooled", "locomotor", "E2_rho")
        dec[cfg] = d

    if "hab2" in dec and "hab3" in dec:      # 第 3 层的去留（设计草案第 6 节）
        def ratio(path):
            a, b = get("hab3", "effect", "pooled", *path)["gain"], get("hab2", "effect", "pooled", *path)["gain"]
            return float(a / b) if b > 0 else float("nan")
        r_l1, r_l2 = ratio(("L1_transition",)), ratio(("locomotor", "E1_rel"))
        c1 = bool(r_l1 >= KEEP_RATIO and r_l2 >= KEEP_RATIO)
        c2 = bool(all(g2_ok(get("hab3", "null", s, *p), get("hab3", "effect", s, *p))
                      for s in args.seeds for p in (("L1_transition",), ("locomotor", "E1_rel"))))
        l3 = get("hab3", "effect", "pooled", "realm", "E1_rel")
        c3 = bool(l3["ci95"][0] > 0)
        dec["L3_rule"] = {"ratio_L1": r_l1, "ratio_L2": r_l2, "c1_no_harm": c1, "c2_null_clean": c2,
                          "c3_L3_gain": c3, "L3_gain": l3["gain"], "L3_ci95": l3["ci95"], "keep_L3": bool(c1 and c2 and c3)}

    (out / "decision.json").write_text(json.dumps(dec, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(strip_boot(summ), ensure_ascii=False, indent=2, default=float),
                                      encoding="utf-8")
    write_report(out, summ, dec, args, missing)
    print(json.dumps(dec, ensure_ascii=False))
    print(f"报告：{out / 'report.md'}")


def fmt(g, k="gain"):
    return f"{g[k]:+.4f} [{g['ci95'][0]:+.4f}, {g['ci95'][1]:+.4f}]"


def write_report(out, summ, dec, args, missing):
    yn = lambda b: "通过" if b else "未通过"  # noqa: E731
    rows = [k for k in summ if not k.endswith("_spooled")]
    L = ["# 开发检查：按习性尺度分层（探索性）", "",
         "设计草案 `docs/design/第二层设计草案_2026-10-15.md`。第 0 阶段的模拟表（种子 " + "、".join(map(str, args.seeds))
         + "，带早期爆发背景），第 21 棵树，效应与零模型各 " + str(len(args.folds)) + " 折。数值单位 log10；"
         f"增益 = 无习性 RMSE − 有习性 RMSE，区间按科自助重抽样 {N_BOOT} 次。只作开发，不作判定。", ""]
    if missing:
        L += [f"缺少 {len(missing)} 个模型，相应的比较跳过。", ""]
    L += ["## 开发判断", ""]
    for cfg in ("hab2", "hab3"):
        if cfg not in dec:
            continue
        d = dec[cfg]
        L.append(f"- **{CFG_ZH[cfg]}**：H3 读数（全部第 2 层步）G1 {yn(all(d['H3_all']['G1'].values()))}、G2 {yn(all(d['H3_all']['G2'].values()))}；"
                 f"不含泛化的子集 G1 {yn(all(d['H3_no_generalist']['G1'].values()))}、G2 {yn(all(d['H3_no_generalist']['G2'].values()))}；"
                 f"第 1 层（G-L1）{yn(d['G_L1']['pass'])}；E2 零模型/效应 = {d['E2_null_ratio']:.0%}，ρ = {d['E2_rho']:.3f}")
    if "L3_rule" in dec:
        r = dec["L3_rule"]
        L.append(f"- **第 3 层去留**：{'保留' if r['keep_L3'] else '不保留'}（第 1 层读数为两层模型的 {r['ratio_L1']:.0%}，"
                 f"第 2 层 {r['ratio_L2']:.0%}，界限 {KEEP_RATIO:.0%}：{yn(r['c1_no_harm'])}；零模型：{yn(r['c2_null_clean'])}；"
                 f"第 3 层自己的增益 {r['L3_gain']:+.4f} [{r['L3_ci95'][0]:+.4f}, {r['L3_ci95'][1]:+.4f}]：{yn(r['c3_L3_gain'])}）")
    L += ["", "## 第 1 层：运动方式变化对上的相对跗跖增益（G-L1、S1）与运动方式相同的相邻对（S2）", "",
          "| 模型 | 变化对 | 相同对（S2） | 全部相邻对 |", "|---|---|---|---|"]
    for k in rows:
        v = summ[k]
        L.append(f"| {k} | {fmt(v['L1_transition'])} | {fmt(v['L1_within'])} | {fmt(v['L1_all'])} |")
    L += ["", "## 第 2 层：H3 的读数（相对跗跖增益）与子集（S4）", "",
          "| 模型 | 全部第 2 层步 | 不含泛化 | 目标单种 | 目标多种 | 2 到 4 种 | 5 到 16 种 | 超过 16 种 | 预测数 | 涉及泛化 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for k in rows:
        v = summ[k].get("locomotor")
        if not v:
            continue
        sb = v["subsets"]
        L.append(f"| {k} | {fmt(v['E1_rel'])} | {fmt(sb['no_generalist'])} | {fmt(sb['single'])} | {fmt(sb['multi'])} | "
                 f"{fmt(sb['len_2_4'])} | {fmt(sb['len_5_16'])} | {fmt(sb['len_gt16'])} | {v['n_pred']} | {v['share_generalist']:.0%} |")
    L += ["", "## 第 2 层：反事实对比（H4 的读数，相对跗跖，相对栖树）", "",
          "| 模型 | " + " | ".join(ZH[m] for m in MODES) + " | 平均绝对值 | ρ（对真值） |", "|---" * (len(MODES) + 3) + "|"]
    for k in rows:
        v = summ[k].get("locomotor")
        if not v:
            continue
        rho = v.get("E2_rho")
        L.append(f"| {k} | " + " | ".join(f"{v['E2'][m]['mean']:+.4f}" for m in MODES)
                 + f" | {v['E2_mean_abs']:.4f} | {'' if rho is None else f'{rho:.3f}'} |")
    L += ["", "## 第 2 层：线性基线（S3）与第 2 层对第 1 层（S5）", "",
          "| 模型 | 线性增益（只用变化 对 无习性） | 世界模型减线性 | S5 RMSE：第 2 层 / 第 1 层直跳 / 第 1 层连推 | 直跳减第 2 层 | 连推减第 2 层 |",
          "|---|---|---|---|---|---|"]
    for k in rows:
        v = summ[k].get("locomotor")
        if not v:
            continue
        s5 = v["S5"]
        L.append(f"| {k} | {fmt(v['linear_gain'])} | {fmt(v['wm_minus_linear'], 'diff')} | "
                 f"{s5['rmse_l2']:.4f} / {s5['rmse_jump']:.4f} / {s5['rmse_roll']:.4f} | {fmt(s5['jump_minus_l2'])} | {fmt(s5['roll_minus_l2'])} |")
    if any("realm" in summ[k] for k in rows):
        L += ["", "## 第 3 层（栖境，只作描述）", "",
              "| 模型 | 相对跗跖增益 | 体重增益 | HWI 增益 | 反事实相对跗跖：水中 / 空中（相对陆地） |", "|---|---|---|---|---|"]
        for k in rows:
            v = summ[k].get("realm")
            if not v:
                continue
            L.append(f"| {k} | {fmt(v['E1_rel'])} | {fmt(v['E1_m_mass'])} | {fmt(v['E1_m_hwi'])} | "
                     + " / ".join(f"{v['E2'][m]['mean']:+.4f}" for m in ("water", "air") if m in v["E2"]) + " |")
    L += ["", "## 探针（留出单位上的 R²，各折平均）", "", "| 模型 | 层 | " + " | ".join(["m_tarsus", "m_mass", "m_hwi"]) + " |",
          "|---|---|---|---|---|"]
    for k in rows:
        for name in ("locomotor", "realm"):
            v = summ[k].get(name)
            if v:
                L.append(f"| {k} | {name} | " + " | ".join(f"{v['probe_r2'].get(c, float('nan')):.3f}"
                                                         for c in ("m_tarsus", "m_mass", "m_hwi")) + " |")
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
