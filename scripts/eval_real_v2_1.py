"""真实数据主分析 v2.1 的评估（修订 docs/prereg/real_main_v2_1.md；其余同 docs/prereg/real_main_v2.md 第 5 到 7 节）。

与 v2 的评估（scripts/eval_real_v2.py）只有两处不同：
  1. 进入条件：第 0 阶段（outputs/stage0_v2/decision.json）的 G1、G2 都通过即可评估；G3 是否通过只决定 H2 的地位。
  2. G3 未通过时，H2a 到 H2c 只作描述（报告对比、区间、各运行方向一致比例、是否超过零模型阈值），不下"支持"或"相反"的结论；
     H1 单独检验，α = 0.025 单侧。G3 也通过时与 v2 完全相同。
模型用"只含习性变化"的动作（检查点的 action_mode 必须是 change）。主分析：Hackett 第 21、41、61、81、100 棵树 × 训练种子 1、2
× 5 折；稳健性：Ericson 第 21、41、61 棵树 × 种子 1 × 5 折。所有读数在观测空间（log10），各运行的预测合并后按簇自助重抽样 10,000 次。
在 EvoWM 根目录运行：python scripts/eval_real_v2_1.py
输出只写到 outputs/real_v2_1/：results.json、report.md。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evowm.data.tree import read_trees  # noqa: E402
from evowm.eval.pgls import ou_stationary, pgls_lambda, tip_vcv  # noqa: E402
from evowm.eval.pooled import boot_gain, boot_gain_diff, boot_mean, holm, p_one_sided, strip_boot  # noqa: E402
from evowm.eval.recovery import fit_half_life  # noqa: E402
from evowm.eval.runs import N_FOLDS, ckpt_action_mode, relative_errors, run_all  # noqa: E402
from evowm.sim.regimes import regime_locomotor  # noqa: E402

PREREG = ROOT / "docs/prereg/real_main_v2.md"
AMEND = ROOT / "docs/prereg/real_main_v2_1.md"
TABLE = "data/processed/species_traits.csv"
N_BOOT = 10000
ALPHA = 0.025
REG6 = ["terrestrial", "aerial", "wading", "diving", "surface", "generalist"]
ZH = {"insessorial": "栖树", "terrestrial": "地栖", "aerial": "空中", "wading": "涉水", "diving": "潜水",
      "surface": "水面", "generalist": "泛化"}
H2 = {"H2a": ("wading", +1), "H2b": ("terrestrial", +1), "H2c": ("aerial", -1)}
SKELETAL = ["k_tarsometatarsus", "k_tibiotarsus", "k_femur"]
OU_SPECIES = 2000  # 平稳 OU 基线每棵树随机取的物种数（固定种子 0）


def tasks_for(backbone, trees, seeds, ckpt_root, required):
    tasks = []
    for t in trees:
        tree_file = f"data/processed/trees/{backbone}_t{t:03d}.nwk"
        for s in seeds:
            cks = [ckpt_root / f"{backbone}_t{t:03d}_s{s}_f{k}" / "evowm_l1_object.ckpt" for k in range(N_FOLDS)]
            if not (ROOT / tree_file).exists() or not all(c.exists() for c in cks):
                if required:
                    sys.exit(f"缺少 {tree_file} 或 {backbone}_t{t:03d}_s{s} 的模型")
                print(f"跳过 {backbone}_t{t:03d}_s{s}（树或模型缺失）")
                continue
            for k, ck in enumerate(cks):
                if ckpt_action_mode(ck) != "change":
                    sys.exit(f"{ck} 不是用只含习性变化的动作训练的（配置应为 recovery_v5_l1）")
                tasks.append({"root": str(ROOT), "table_rel": TABLE, "tree_file": tree_file, "fold": k, "seed": s,
                              "ckpt": str(ck), "truth": None, "real": True, "linear": backbone == "hackett",
                              "key": f"{backbone}_t{t:03d}_s{s}"})
    return tasks


def summarize(grp, table, boot_level):
    fam = np.concatenate([r["family"] for r in grp])
    trans = np.concatenate([r["transition"] for r in grp])
    rel = [relative_errors(r) for r in grp]
    re_, rn_ = np.concatenate([a for a, _ in rel]), np.concatenate([b for _, b in rel])
    cols = grp[0]["E1_state"]["state_cols"]
    jt, jm = cols.index("m_tarsus"), cols.index("m_mass")
    ee = np.concatenate([r["E1_state"]["err_eco"] for r in grp])
    en = np.concatenate([r["E1_state"]["err_no_eco"] for r in grp])
    nan = np.full(len(fam), np.nan)
    out = {"n_runs": len({r["key"] for r in grp}),
           "E1_rel": boot_gain(re_, rn_, fam, N_BOOT, 0),
           "E1_rel_transition": boot_gain(np.where(trans, re_, nan), np.where(trans, rn_, nan), fam, N_BOOT, 0),
           "E1_abs": boot_gain(ee[:, jt], en[:, jt], fam, N_BOOT, 0),
           "E1_mass": boot_gain(ee[:, jm], en[:, jm], fam, N_BOOT, 0),
           "share_transition": float(np.mean(trans))}
    if all("linear" in r for r in grp):
        lc = np.concatenate([r["linear"]["err_change"] for r in grp])
        le = np.concatenate([r["linear"]["err_eco"] for r in grp])
        ln = np.concatenate([r["linear"]["err_no"] for r in grp])
        tr = lambda x: np.where(trans, x, nan)  # noqa: E731
        out["linear_change_E1_transition"] = boot_gain(tr(lc), tr(ln), fam, N_BOOT, 0)
        out["linear_change_E1"] = boot_gain(lc, ln, fam, N_BOOT, 0)
        out["linear_level_E1"] = boot_gain(le, ln, fam, N_BOOT, 0)
        out["model_minus_linear_transition"] = boot_gain_diff(tr(re_), tr(rn_), tr(lc), tr(ln), fam, N_BOOT, 0)
        out["model_minus_linear"] = boot_gain_diff(re_, rn_, lc, ln, fam, N_BOOT, 0)
        out["linear_alpha"] = {"habit": float(np.median([r["linear"]["alpha_habit"] for r in grp])),
                               "no_habit": float(np.median([r["linear"]["alpha_no"] for r in grp]))}
        out["linear_kappa"] = float(np.median([r["linear"]["kappa"] for r in grp]))
    # E2：逐窗口相对跗跖对比合并，按第 0 阶段确定的层级重抽样；另报每次运行的方向
    tgt = np.concatenate([r["E2_window"]["target_rows"] for r in grp])
    cl = table[boot_level or "family"].to_numpy()[tgt]
    keys = sorted({r["key"] for r in grp})
    e2 = {}
    for x in REG6:
        vals = np.concatenate([r["E2_window"]["rel"][x]["m_tarsus"] for r in grp])
        e2[x] = boot_mean(vals, cl, N_BOOT, 0)
        e2[x]["per_run"] = {k: float(np.mean(np.concatenate([r["E2_window"]["rel"][x]["m_tarsus"] for r in grp if r["key"] == k])))
                            for k in keys}
    out["E2"] = e2
    out["E2_skeletal_mean"] = {x: {c: float(np.mean([r["res"]["E2_relative"]["contrasts"][x][c] for r in grp]))
                                   for c in SKELETAL if c in grp[0]["res"]["E2_relative"]["contrasts"][x]} for x in REG6}
    # E3：每次运行合并 5 折拟合
    th = {}
    for k in keys:
        g = [r for r in grp if r["key"] == k]
        th[k] = fit_half_life(np.concatenate([r["E3"]["distance"] for r in g]),
                              np.concatenate([r["E3"]["sensitivity"] for r in g]))["t_half_free_amplitude"]
    v = np.array(list(th.values()))
    out["E3"] = {"per_run": th, "median": float(np.nanmedian(v)), "min": float(np.nanmin(v)), "max": float(np.nanmax(v)),
                 "n_failed_fits": int(np.isnan(v).sum())}
    out["probe_r2_val_mean"] = {c: float(np.mean([r["res"]["probe_r2_val"][c] for r in grp])) for c in grp[0]["res"]["probe_r2_val"]}
    return out


def run_pgls(trees, table, max_species, seed=0):
    """每棵树：PGLS（Pagel λ，全部物种）与平稳 OU 回归（随机 OU_SPECIES 种）。模型都是 log 跗跖 ~ log 体重 + 运动方式。"""
    el = table[["m_tarsus", "m_mass"]].notna().all(axis=1).to_numpy()
    names = ["intercept", "log_mass"] + REG6
    out = {}
    for t in trees:
        tree = read_trees(ROOT / f"data/processed/trees/hackett_t{t:03d}.nwk", max_trees=1)[0]
        pos = {n: i for i, n in enumerate(tree.tip_names)}
        C_all = tip_vcv(tree)
        rows_all = np.array([i for i in np.flatnonzero(el) if table["tip"].iat[i] in pos])

        def design(rows):
            idx = np.array([pos[table["tip"].iat[i]] for i in rows])
            reg = table["sim_regime"].to_numpy()[rows]
            X = np.column_stack([np.ones(len(rows)), table["m_mass"].to_numpy()[rows]] + [(reg == x).astype(float) for x in REG6])
            return table["m_tarsus"].to_numpy()[rows].astype(float), X, C_all[np.ix_(idx, idx)]

        rows = rows_all
        if max_species and len(rows) > max_species:
            rows = np.sort(np.random.default_rng(seed).choice(rows, max_species, replace=False))
        out[t] = pgls_lambda(*design(rows), names)
        rows_ou = np.sort(np.random.default_rng(seed).choice(rows_all, min(OU_SPECIES, len(rows_all)), replace=False))
        out[t]["ou"] = ou_stationary(*design(rows_ou), names)
        C_all = None  # 释放内存（约 0.8 GB）
        print(f"  PGLS 树 {t}：λ = {out[t]['lambda']:.3f}，n = {out[t]['n']}；OU t½ = {out[t]['ou']['t_half']:.1f} Myr", flush=True)
    return out


def fmt(d, key="gain"):
    return f"{d[key]:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hackett-trees", type=int, nargs="+", default=[21, 41, 61, 81, 100])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--ericson-trees", type=int, nargs="*", default=[21, 41, 61])
    ap.add_argument("--ckpt-root", default=str(ROOT / "outputs/hjepa/ckpts/real_v2"))
    ap.add_argument("--stage0", default=str(ROOT / "outputs/stage0_v2/decision.json"))
    ap.add_argument("--out", default=str(ROOT / "outputs/real_v2_1"))
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--pgls-max-species", type=int, default=0, help="只用于测试；0 表示全部物种")
    ap.add_argument("--skip-pgls", action="store_true", help="只用于测试")
    args = ap.parse_args()
    for p in (PREREG, AMEND):
        if not p.exists():
            sys.exit(f"缺少预注册 {p.relative_to(ROOT)}")
    dpath = Path(args.stage0)
    if not dpath.exists():
        sys.exit("缺少第 0 阶段结果，真实数据不能评估")
    decision = json.loads(dpath.read_text(encoding="utf-8"))
    g1, g2 = decision.get("G1") or {}, decision.get("G2") or {}
    if not (g1 and g2 and all(g1.values()) and all(g2.values())):
        sys.exit("第 0 阶段 G1 或 G2 未通过，真实数据不能评估（修订 v2.1 第 2 节）")
    h2_confirm = bool(decision.get("pass"))  # G3 也通过时，H2 与 v2 相同；否则只作描述
    level = decision.get("e2_ci_level")
    boot_level = level if level in ("family", "order") else None
    thr = float(decision["e2_null_threshold"]) if level == "threshold" else None
    e3_ref = decision.get("e3_sim_t_half_median") or {}
    out, ckpt_root = Path(args.out), Path(args.ckpt_root)
    out.mkdir(parents=True, exist_ok=True)

    tasks = tasks_for("hackett", args.hackett_trees, args.seeds, ckpt_root, True) + \
        tasks_for("ericson", args.ericson_trees, [1], ckpt_root, False)
    results = run_all(tasks, args.workers)
    table = pd.read_csv(ROOT / TABLE)
    table["sim_regime"] = regime_locomotor(table).to_numpy()

    main_ = summarize([r for r in results if r["key"].startswith("hackett_")], table, boot_level)
    eric = [r for r in results if r["key"].startswith("ericson_")]
    robust = summarize(eric, table, boot_level) if eric else None

    # ---- 判定用检验（预注册第 5 节）----
    pvals = {"H1": p_one_sided(main_["E1_rel_transition"]["boot"], +1)}
    if boot_level and h2_confirm:
        for h, (x, d) in H2.items():
            pvals[h] = p_one_sided(main_["E2"][x]["boot"], d)
    rej = holm(pvals, ALPHA)
    verdict = {"H1": "支持" if rej["H1"] else "未检出"}
    consistency = {}
    for h, (x, d) in H2.items():
        runs = main_["E2"][x]["per_run"]
        consistency[h] = float(np.mean([np.sign(v) == d for v in runs.values()]))
        lo, hi = main_["E2"][x]["ci95"]
        est = main_["E2"][x]["mean"]
        if not h2_confirm:  # 修订 v2.1：G3 未通过，H2 只作描述
            if thr is not None:
                desc = ("超过零模型阈值，方向与预期一致" if d * est > thr else
                        ("超过零模型阈值，方向与预期相反" if -d * est > thr else "未超过零模型阈值"))
            else:
                desc = "方向与预期一致" if np.sign(est) == d else "方向与预期相反"
            verdict[h] = "描述（不判定）：" + desc
        elif thr is not None:  # 零模型阈值：预期方向上超过阈值为支持，相反方向上超过为相反
            verdict[h] = "支持" if d * est > thr else ("相反" if -d * est > thr else "不确定")
        elif not boot_level:  # 旧版 decision.json（没有阈值）
            verdict[h] = "仅方向：" + ("一致" if np.sign(est) == d else "相反")
        elif rej[h]:
            verdict[h] = "支持"
        elif (d > 0 and hi < 0) or (d < 0 and lo > 0):
            verdict[h] = "相反"
        else:
            verdict[h] = "不确定"
    pg = None if args.skip_pgls else run_pgls(args.hackett_trees, table, args.pgls_max_species)

    res = {"stage0": decision, "h2_confirmatory": h2_confirm,
           "tests": {"p_one_sided": pvals, "holm_reject": rej, "verdict": verdict,
                                         "direction_consistency": consistency},
           "main": main_, "robust_ericson": robust, "pgls": pg}
    (out / "results.json").write_text(json.dumps(strip_boot(res), ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    lvzh = {"family": "按科重抽样", "order": "按目重抽样", "threshold": f"零模型阈值 ±{thr or 0:.4f}（不进 Holm）",
            None: "只报方向"}[level]
    L = ["# 真实数据主分析 v2.1 结果（动作只含习性变化）", "",
         f"预注册 `docs/prereg/real_main_v2.md`，修订 `docs/prereg/real_main_v2_1.md`。Hackett 第 "
         f"{'、'.join(map(str, args.hackett_trees))} 棵树 × 训练种子 {'、'.join(map(str, args.seeds))} × 5 折，"
         f"共 {main_['n_runs']} 次运行合并。数值单位 log10。第 0 阶段：G1、G2 通过；G3 "
         + ("通过，H2 为确认性检验" if h2_confirm else "未通过，按修订 v2.1，H2 只作描述，H1 单独检验")
         + f"；E2 判定方式为{lvzh}。", "",
         ("## 判定用检验（Holm，总体 α = 0.025 单侧）" if h2_confirm else
          "## 判定用检验（H1 单独检验，α = 0.025 单侧；H2 只作描述）"), "",
         "| 假设 | 读数 | 估计 | 95% CI | 单侧 p | 结论 | 各运行方向一致比例 |", "|---|---|---|---|---|---|---|",
         f"| H1 习性变化的可迁移预测价值 | 运动方式变化对上的相对跗跖 E1 增益 | {main_['E1_rel_transition']['gain']:+.4f} | "
         f"[{main_['E1_rel_transition']['ci95'][0]:+.4f}, {main_['E1_rel_transition']['ci95'][1]:+.4f}] | {pvals['H1']:.4f} | "
         f"{verdict['H1']} | |"]
    for h, (x, d) in H2.items():
        e = main_["E2"][x]
        p = f"{pvals[h]:.4f}" if h in pvals else "无"
        L.append(f"| {h} {ZH[x]}{'＞' if d > 0 else '＜'}栖树 | E2 相对跗跖对比 | {e['mean']:+.4f} | "
                 f"[{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}] | {p} | {verdict[h]} | {consistency[h]:.0%} |")
    flag = [h for h, c in consistency.items() if c < 0.8]
    if flag:
        L.append(f"\n各运行方向一致比例低于 80%：{'、'.join(flag)}，结论注明随树或种子不稳定。")
    m = main_
    L += ["", "## E1 明细（描述）", "", "| 读数 | 增益 [95% CI] |", "|---|---|",
          f"| 相对跗跖，运动方式变化的物种对（H1，占 {m['share_transition']:.1%}） | {fmt(m['E1_rel_transition'])} |",
          f"| 相对跗跖，全部相邻对（次要） | {fmt(m['E1_rel'])} |",
          f"| 绝对跗跖 | {fmt(m['E1_abs'])} |", f"| 体重 | {fmt(m['E1_mass'])} |"]
    if "linear_change_E1" in m:
        L += [f"| 线性基线（习性只当变化用），变化对 | {fmt(m['linear_change_E1_transition'])} |",
              f"| 线性基线（习性只当变化用），全部相邻对 | {fmt(m['linear_change_E1'])} |",
              f"| 世界模型 − 线性基线，变化对 | {fmt(m['model_minus_linear_transition'], 'diff')} |",
              f"| 世界模型 − 线性基线，全部相邻对 | {fmt(m['model_minus_linear'], 'diff')} |",
              f"| 参照：线性基线（习性当水平用，v1 的版本） | {fmt(m['linear_level_E1'])} |"]
    L += ["", f"## E2 运动方式对比（相对跗跖，相对栖树；区间按{'目' if boot_level == 'order' else '科'}重抽样"
          + (f"，判定用零模型阈值 ±{thr:.4f}" if thr is not None else "") + "）", "",
          "| 运动方式 | 对比 | 95% CI | 各运行估计的范围 |", "|---|---|---|---|"]
    for x in REG6:
        e = m["E2"][x]
        pr = list(e["per_run"].values())
        L.append(f"| {ZH[x]} | {e['mean']:+.4f} | [{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}] | {min(pr):+.4f} 到 {max(pr):+.4f} |")
    ref = "，".join(f"{'效应' if k == 'effect' else '零模型'} {v:.1f} Myr" for k, v in e3_ref.items())
    L += ["", "## E3 近亲信息的衰减（描述）", "", f"每次运行合并 5 折拟合（带振幅）：t½ 中位数 {m['E3']['median']:.1f} Myr，范围 "
          f"{m['E3']['min']:.1f} 到 {m['E3']['max']:.1f}（拟合失败 {m['E3']['n_failed_fits']} 次）。"
          + (f"参照：第 0 阶段模拟（习性响应 t½ 12 Myr 加早期爆发背景）中为{ref}。" if ref else "")
          + "相对跗跖的平稳 OU 拟合见基线表；旧 OU 估计：u 约 12.3 Myr，HWI 约 14.6 Myr（`data/prior_results/S2_lability.csv`）。", ""]
    L += ["| 运行 | t½ |", "|---|---|"] + [f"| {k} | {v:.1f} |" for k, v in m["E3"]["per_run"].items()]
    if pg:
        L += ["", "## 基线：PGLS（log 跗跖 ~ log 体重 + 运动方式，Pagel λ 最大似然）", "",
              "| 树 | λ | n | log 体重 | " + " | ".join(ZH[x] for x in REG6) + " |", "|---" * (len(REG6) + 4) + "|"]
        for t, r in pg.items():
            c = r["coef"]
            L.append(f"| {t} | {r['lambda']:.3f} | {r['n']} | {c['log_mass']['beta']:+.3f} | "
                     + " | ".join(f"{c[x]['beta']:+.3f} ± {c[x]['se']:.3f}" for x in REG6) + " |")
        L += ["", f"## 基线：平稳 OU 回归（同一模型，每棵树随机 {OU_SPECIES} 种，最大似然）", "",
              "| 树 | t½（Myr） | 平稳 SD | 测量误差占比 | " + " | ".join(ZH[x] for x in REG6) + " |", "|---" * (len(REG6) + 4) + "|"]
        for t, r in pg.items():
            o = r["ou"]
            L.append(f"| {t} | {o['t_half']:.1f} | {o['sd_stat']:.3f} | {o['q_me']:.2f} | "
                     + " | ".join(f"{o['coef'][x]:+.3f}" for x in REG6) + " |")
    if robust:
        L += ["", f"## 稳健性：Ericson 树（{robust['n_runs']} 次运行，不判定）", "",
              f"- 相对跗跖 E1：变化对 {fmt(robust['E1_rel_transition'])}；全部相邻对 {fmt(robust['E1_rel'])}",
              "- E2：" + "，".join(f"{ZH[x]} {robust['E2'][x]['mean']:+.4f}" for x in REG6),
              f"- E3 中位数 {robust['E3']['median']:.1f} Myr"]
    L += ["", "## 探索：骨段性状的 E2 平均对比（相对，描述）", "", "| 运动方式 | " + " | ".join(SKELETAL) + " |",
          "|---" * (len(SKELETAL) + 1) + "|"]
    for x in REG6:
        L.append(f"| {ZH[x]} | " + " | ".join(f"{m['E2_skeletal_mean'][x].get(c, float('nan')):+.4f}" for c in SKELETAL) + " |")
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("结论：", verdict)
    print(f"报告：{out / 'report.md'}")


if __name__ == "__main__":
    main()
