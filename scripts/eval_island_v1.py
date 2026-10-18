"""岛屿对的外部检验（设计草案 docs/design/外部检验_岛屿_草案_2026-10-18.md）。

--mode stage0（第 0 步，预注册之前）：主分析第 0 阶段的三层模型（units_main_v1：效应、零模型 × 种子 13、14 ×
  第 21、41、61、81、100 棵树 × 5 折）与它们的模拟表。模拟表里习性是真实的，形态是模拟值，只用来看 I1、I2 的
  功效与假阳性，据此定预注册的读数。
--mode real（预注册之后）：主分析的 50 个三层模型（Hackett 第 21、41、61、81、100 棵树 × 种子 1、2 × 5 折）与物种表；
  另有 15 个 Ericson 模型作稳健性（检查点缺失时跳过）。需要预注册 docs/prereg/island_v1.md。

每一对只交给两种所在科都被留出的那一折的模型；全部运行的预测合并后，按 I 所在的科自助重抽样。
输出写到 outputs/island_v1/<mode>/：results.json、report.md、predictions.csv.gz（逐预测）。
在 EvoWM 根目录运行：python scripts/eval_island_v1.py --mode stage0
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from evowm.eval.island import PERM_READOUT, annotate, load_pairs, observed, run_island_task  # noqa: E402
from evowm.eval.pooled import boot_gain, boot_mean, p_one_sided, strip_boot  # noqa: E402
from evowm.eval.runs import N_FOLDS, ckpt_action_mode, run_all  # noqa: E402

DESIGN = "docs/design/外部检验_岛屿_草案_2026-10-18.md"
PREREG = ROOT / "docs/prereg/island_v1.md"
TABLE = "data/processed/species_traits.csv"
TABLE_MD5 = "872bcc14eecae132a83d5f3dec5ea299"
UNITS = ["locomotor", "realm"]
ZH = {"rtarsus": "相对跗跖", "mass": "体重", "hwi": "HWI", "rwing": "相对翅长",
      "rfemur": "相对股骨", "rtibio": "相对胫跗骨", "rtmt": "相对跗跖骨"}
ROBUST_ZH = {"I2_ponti_migration": "I2，G 组改用 Ponti 的迁徙状态", "I1_ponti_targets": "I1，目标值改用 Ponti 的标本测量",
             "I2_ponti_targets": "I2，目标值改用 Ponti 的标本测量", "I1_same_family": "I1，去掉 I 与 M 不同科的对",
             "I2_same_family": "I2，去掉 I 与 M 不同科的对", "I1_filtered": "I1，只用 Ponti 的 filtered 表",
             "I2_filtered": "I2，只用 Ponti 的 filtered 表"}


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def check_ckpt(ck: Path) -> None:
    if not ck.exists():
        sys.exit(f"缺少模型 {ck}")
    cfg = yaml.safe_load((ck.parent / "config.yaml").read_text(encoding="utf-8")) or {}
    lv = (((cfg.get("data") or {}).get("dataset") or {}).get("units") or {}).get("levels")
    if ckpt_action_mode(ck) != "change" or list(lv or []) != UNITS:
        sys.exit(f"{ck} 的动作方式或上层单位不是 change 与 {UNITS}")


def build_tasks(args) -> list[dict]:
    tasks = []
    base = dict(root=str(ROOT), n_perm=args.n_perm)
    if args.mode == "stage0":
        root = Path(args.ckpt_root or ROOT / "outputs/hjepa/ckpts/stage0_units_v1")
        for s in args.seeds:
            for sc in args.scenarios:
                for t in args.trees:
                    for k in args.folds:
                        ck = root / f"{sc}_s{s}_t{t:03d}_f{k}" / "evowm_hab3_object.ckpt"
                        check_ckpt(ck)
                        tasks.append(dict(base, table_rel=f"data/processed/sim/recovery_v5/{sc}_s{s}/species_traits.csv",
                                          tree_file=f"data/processed/trees/hackett_t{t:03d}.nwk", fold=k, seed=s,
                                          ckpt=str(ck), key=f"{sc}_s{s}_t{t:03d}", analysis=f"{sc}_s{s}", wright=False))
    else:
        root = Path(args.ckpt_root or ROOT / "outputs/hjepa/ckpts/real_units_v1")
        runs = [("hackett", t, s) for t in args.trees for s in args.seeds]
        if not args.no_ericson:
            runs += [("ericson", t, 1) for t in (21, 41, 61)]
        for tree, t, s in runs:
            for k in args.folds:
                ck = root / f"{tree}_t{t:03d}_s{s}_f{k}" / "evowm_hab3_object.ckpt"
                if tree == "ericson" and not ck.exists():
                    print(f"跳过（没有 Ericson 模型）：{ck}")
                    continue
                check_ckpt(ck)
                tasks.append(dict(base, table_rel=args.table, tree_file=f"data/processed/trees/{tree}_t{t:03d}.nwk", fold=k,
                                  seed=s, ckpt=str(ck), key=f"{tree}_t{t:03d}_s{s}", analysis=tree, wright=True))
    return tasks


# ---------------------------------------------------------------------------
def collect(results: list[dict], meta: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """逐预测的长表（每行 = 一次运行对一对的预测）与打乱对照的平方误差和。"""
    frames = []
    for r in results:
        df = pd.DataFrame({"run": r["key"], "fold": r["fold"], "pairID": r["pairID"], "distance": r["distance"]})
        for part in ("obs_I", "obs_M", "eco", "no", "B1"):
            for k, v in r[part].items():
                df[f"{part}_{k}"] = v
        for k, v in r["B2"].items():
            df[f"B2_{k}"] = v
        df["slope_tarsus"] = r["slopes"]["m_tarsus"]
        frames.append(df)
    long = pd.concat(frames, ignore_index=True).merge(meta, on="pairID", how="left", validate="many_to_one")
    long["ponti_rtarsus"] = long["p_tarsus_I"] - long["slope_tarsus"] * long["p_mass_I"]
    long["ponti_hwi"] = long["p_hwi_I"]

    perm = {}
    for grp, readout in PERM_READOUT.items():
        sse_p, sse_no, sse_eco, n = None, 0.0, 0.0, 0
        for r in results:
            pi, pp = r["perm"][grp]["pair_index"], r["perm"][grp][readout]
            if pp.shape[0] == 0:
                continue
            y = r["obs_I"][readout][pi]
            ok = np.isfinite(y) & np.isfinite(r["no"][readout][pi]) & np.isfinite(r["eco"][readout][pi])
            s = ((pp[:, ok] - y[ok]) ** 2).sum(axis=1)
            sse_p = s if sse_p is None else sse_p + s
            sse_no += float(((r["no"][readout][pi][ok] - y[ok]) ** 2).sum())
            sse_eco += float(((r["eco"][readout][pi][ok] - y[ok]) ** 2).sum())
            n += int(ok.sum())
        if sse_p is None or n == 0:
            perm[grp] = None
            continue
        g_obs = np.sqrt(sse_no / n) - np.sqrt(sse_eco / n)
        g_perm = np.sqrt(sse_no / n) - np.sqrt(sse_p / n)
        perm[grp] = {"readout": readout, "gain": float(g_obs), "n_pred": n, "n_perm": int(len(g_perm)),
                     "perm_q95": float(np.percentile(g_perm, 95)), "perm_mean": float(g_perm.mean()),
                     "p_perm": float((1 + np.sum(g_perm >= g_obs)) / (len(g_perm) + 1))}
    return long, perm


def gain_block(df: pd.DataFrame, mask, readout: str, n_boot: int, obs_col: str | None = None) -> dict:
    y = df[obs_col or f"obs_I_{readout}"]
    e_eco, e_no = df[f"eco_{readout}"] - y, df[f"no_{readout}"] - y
    sub = np.asarray(mask, bool) & np.isfinite(e_eco).to_numpy() & np.isfinite(e_no).to_numpy()
    if sub.sum() == 0:
        return {"gain": float("nan"), "ci95": [float("nan")] * 2, "n_pred": 0, "n_pairs": 0}
    g = boot_gain(e_eco[sub], e_no[sub], df.loc[sub, "family_I"].to_numpy(), n_boot, 0)
    g["p_one_sided"] = p_one_sided(g.pop("boot"), +1)
    g["n_pairs"] = int(df.loc[sub, "pairID"].nunique())
    per = {}
    for run, d in df[sub].groupby("run"):
        ye = d[f"eco_{readout}"] - d[obs_col or f"obs_I_{readout}"]
        yn = d[f"no_{readout}"] - d[obs_col or f"obs_I_{readout}"]
        per[run] = float(np.sqrt(np.mean(yn ** 2)) - np.sqrt(np.mean(ye ** 2)))
    g["per_run"] = per
    g["frac_runs_positive"] = float(np.mean([v > 0 for v in per.values()])) if per else float("nan")
    return g


def direction_block(df: pd.DataFrame, mask, readout: str, n_boot: int) -> dict:
    """习性效应 h = 带习性 − 不带习性；观测偏差 o = 观测 − 不带习性；观测变化 = I − M。"""
    h = (df[f"eco_{readout}"] - df[f"no_{readout}"]).to_numpy()
    o = (df[f"obs_I_{readout}"] - df[f"no_{readout}"]).to_numpy()
    dobs = (df[f"obs_I_{readout}"] - df[f"obs_M_{readout}"]).to_numpy()
    fam = df["family_I"].to_numpy()
    m = np.asarray(mask, bool)
    sub = m & np.isfinite(h) & np.isfinite(o)
    if sub.sum() == 0:
        return {"n_pairs": 0}
    out = {"n_pairs": int(df.loc[sub, "pairID"].nunique()),
           "h_mean": strip_boot(boot_mean(h[sub], fam[sub], n_boot, 0))}
    sd = m & np.isfinite(dobs)
    if sd.any():
        out["obs_change_mean"] = strip_boot(boot_mean(dobs[sd], fam[sd], n_boot, 0))
    nz = sub & (h != 0) & (o != 0)
    out["sign_agreement"] = float(np.mean(np.sign(h[nz]) == np.sign(o[nz]))) if nz.any() else float("nan")
    pp = pd.DataFrame({"pairID": df.pairID.to_numpy()[sub], "h": h[sub], "o": o[sub]}).groupby("pairID").mean()
    out["spearman_pairs"] = float(spearmanr(pp.h, pp.o).statistic) if len(pp) > 2 else float("nan")
    return out


def baseline_block(df: pd.DataFrame, mask, readout: str) -> dict:
    y = df[f"obs_I_{readout}"]
    preds = {"B0 不变": df[f"obs_M_{readout}"], "B1 常数岛屿效应": df[f"B1_{readout}"]}
    if f"B2_{readout}" in df:
        preds["B2 线性基线"] = df[f"B2_{readout}"]
    preds["模型不带习性"] = df[f"no_{readout}"]
    preds["模型带习性"] = df[f"eco_{readout}"]
    common = np.asarray(mask, bool) & np.isfinite(y).to_numpy()
    for p in preds.values():
        common &= np.isfinite(p).to_numpy()
    return {"n_pred": int(common.sum()), "n_pairs": int(df.loc[common, "pairID"].nunique()),
            "rmse": {k: float(np.sqrt(np.mean((p[common] - y[common]) ** 2))) for k, p in preds.items()}}


def summarize(df: pd.DataFrame, perm: dict, n_boot: int, real: bool) -> dict:
    L, G, N = df.L.to_numpy(bool), df.G.to_numpy(bool), df.N.to_numpy(bool)
    allp = np.ones(len(df), bool)
    res = {"n_pairs": int(df.pairID.nunique()), "n_families": int(df.family_I.nunique()),
           "n_runs": int(df.run.nunique()), "n_pred": int(len(df)),
           "group_pairs": {g: int(df.loc[df[g].to_numpy(bool), "pairID"].nunique())
                           for g in ("L", "G", "N", "toward_T", "away_T", "mig_down")}}
    res["I1"] = gain_block(df, L, "rtarsus", n_boot)
    res["I2"] = gain_block(df, G, "hwi", n_boot)
    res["I1"]["permutation"], res["I2"]["permutation"] = perm.get("L"), perm.get("G")
    res["direction"] = {
        "toward_T_rtarsus": direction_block(df, df.toward_T.to_numpy(bool), "rtarsus", n_boot),
        "away_T_rtarsus": direction_block(df, df.away_T.to_numpy(bool), "rtarsus", n_boot),
        "L_rtarsus": direction_block(df, L, "rtarsus", n_boot),
        "mig_down_hwi": direction_block(df, df.mig_down.to_numpy(bool), "hwi", n_boot),
        "G_hwi": direction_block(df, G, "hwi", n_boot),
        "all_rtarsus": direction_block(df, allp, "rtarsus", n_boot),
        "all_hwi": direction_block(df, allp, "hwi", n_boot)}
    res["N"] = {"rtarsus": gain_block(df, N, "rtarsus", n_boot), "hwi": gain_block(df, N, "hwi", n_boot)}
    res["baselines"] = {"L_rtarsus": baseline_block(df, L, "rtarsus"), "G_hwi": baseline_block(df, G, "hwi"),
                        "all_rtarsus": baseline_block(df, allp, "rtarsus"), "all_hwi": baseline_block(df, allp, "hwi")}
    res["other_readouts"] = {
        "rtarsus_L_or_G": gain_block(df, L | G, "rtarsus", n_boot),
        "rtarsus_all": gain_block(df, allp, "rtarsus", n_boot),
        "hwi_all": gain_block(df, allp, "hwi", n_boot),
        "mass_all": gain_block(df, allp, "mass", n_boot),
        "rwing_all": gain_block(df, allp, "rwing", n_boot)}
    res["bones"] = {b: {"all": gain_block(df, allp, b, n_boot), "L_or_G": gain_block(df, L | G, b, n_boot),
                        "direction_all": direction_block(df, allp, b, n_boot)}
                    for b in ("rfemur", "rtibio", "rtmt")}
    if real:
        res["robustness"] = {
            "I2_ponti_migration": gain_block(df, df.G_ponti.to_numpy(bool), "hwi", n_boot),
            "I1_ponti_targets": gain_block(df, L, "rtarsus", n_boot, obs_col="ponti_rtarsus"),
            "I2_ponti_targets": gain_block(df, G, "hwi", n_boot, obs_col="ponti_hwi"),
            "I1_same_family": gain_block(df, L & ~df.cross_family.to_numpy(bool), "rtarsus", n_boot),
            "I2_same_family": gain_block(df, G & ~df.cross_family.to_numpy(bool), "hwi", n_boot),
            "I1_filtered": gain_block(df, L & df.in_filtered.to_numpy(bool), "rtarsus", n_boot),
            "I2_filtered": gain_block(df, G & df.in_filtered.to_numpy(bool), "hwi", n_boot)}
    return res


# ---------------------------------------------------------------------------
def fmt_gain(g: dict) -> str:
    if not g or not g.get("n_pred"):
        return "（无数据）"
    return f"{g['gain']:+.4f} [{g['ci95'][0]:+.4f}, {g['ci95'][1]:+.4f}]，p = {g['p_one_sided']:.4f}"


def fmt_mean(m: dict | None) -> str:
    if not m:
        return "（无数据）"
    return f"{m['mean']:+.4f} [{m['ci95'][0]:+.4f}, {m['ci95'][1]:+.4f}]"


def report(mode: str, out: dict) -> str:
    title = "第 0 步（模拟校准，预注册之前）" if mode == "stage0" else "真实数据"
    lines = [f"# 岛屿外部检验：{title}", "",
             f"设计草案 `{DESIGN}`。脚本 `scripts/eval_island_v1.py --mode {mode}`。", "",
             "## 符号与术语", "",
             "- **M、I**：一对物种，I 为岛屿特有种，M 为 Ponti 等 2025 给它配的最近的大陆亲属。",
             "- **L 组**：运动方式（主分析的 7 类）不同的对；**G 组**：迁徙程度（AVONET 三级）不同的对；"
             "**N 组**：两者都相同的对。**变得更地栖**：I 为地栖、M 不是；**离开地栖**：M 为地栖、I 不是。",
             "- **相对跗跖**：log10 跗跖 − b × log10 体重，b 为该折训练物种上的斜率。**HWI**：翅尖指数（/100）。",
             "- **G（习性增益）**：RMSE(不带习性的预测) − RMSE(带习性的预测)，大于 0 表示告诉模型习性变化后预测更准；"
             "区间为按 I 所在的科自助重抽样的 95% 区间，p 为单侧。",
             "- **I1**：L 组上相对跗跖的 G；**I2**：G 组上 HWI 的 G。",
             "- **打乱**：组内把每对的习性变化换成另一对的（保留自己的距离），重复多次；**打乱 p** 为打乱后增益不低于真实增益的比例。",
             "- **h（习性效应）**：带习性 − 不带习性的预测，即模型认为习性变化带来的形态变化；**观测变化**：I − M 的观测值。",
             "- **B0**：I = M；**B1**：M 加上训练科里全部对的平均差；**B2**：线性基线（相对跗跖按运动方式，HWI 按迁徙程度）。", ""]
    lines += ["## I1、I2", "",
              "| 分析 | 对数（科） | I1：L 组相对跗跖 G | 打乱 95% 分位，打乱 p | I2：G 组 HWI G | 打乱 95% 分位，打乱 p |",
              "|---|---|---|---|---|---|"]
    for name, r in out["analyses"].items():
        p1, p2 = r["I1"].get("permutation") or {}, r["I2"].get("permutation") or {}
        pf = lambda p: f"{p['perm_q95']:+.4f}，{p['p_perm']:.3f}" if p else "（无）"  # noqa: E731
        lines.append(f"| {name} | {r['n_pairs']}（{r['n_families']}） | {fmt_gain(r['I1'])} | {pf(p1)} | "
                     f"{fmt_gain(r['I2'])} | {pf(p2)} |")
    lines += ["", "每次运行的 G 为正的比例："]
    for name, r in out["analyses"].items():
        lines.append(f"- {name}：I1 {r['I1'].get('frac_runs_positive', float('nan')):.2f}，"
                     f"I2 {r['I2'].get('frac_runs_positive', float('nan')):.2f}")
    lines += ["", "## 方向（描述）", "",
              "| 分析 | 变得更地栖：h（相对跗跖） | 观测变化 | 离开地栖：h | 观测变化 | 迁徙降低：h（HWI） | 观测变化 |",
              "|---|---|---|---|---|---|---|"]
    for name, r in out["analyses"].items():
        d = r["direction"]
        lines.append(f"| {name} | {fmt_mean(d['toward_T_rtarsus'].get('h_mean'))} | "
                     f"{fmt_mean(d['toward_T_rtarsus'].get('obs_change_mean'))} | {fmt_mean(d['away_T_rtarsus'].get('h_mean'))} | "
                     f"{fmt_mean(d['away_T_rtarsus'].get('obs_change_mean'))} | {fmt_mean(d['mig_down_hwi'].get('h_mean'))} | "
                     f"{fmt_mean(d['mig_down_hwi'].get('obs_change_mean'))} |")
    lines += ["", "h 与观测偏差（观测 − 不带习性的预测）的符号一致率与逐对 Spearman 相关："]
    for name, r in out["analyses"].items():
        d = r["direction"]
        lines.append(f"- {name}：L 组相对跗跖 {d['L_rtarsus'].get('sign_agreement', float('nan')):.3f}、"
                     f"ρ = {d['L_rtarsus'].get('spearman_pairs', float('nan')):+.3f}；G 组 HWI "
                     f"{d['G_hwi'].get('sign_agreement', float('nan')):.3f}、ρ = {d['G_hwi'].get('spearman_pairs', float('nan')):+.3f}")
    lines += ["", "全部对：观测变化与 h 的平均（Ponti 的\"岛屿种跗跖更长\"有多少能由习性解释）："]
    for name, r in out["analyses"].items():
        d = r["direction"]
        lines.append(f"- {name}：相对跗跖 观测 {fmt_mean(d['all_rtarsus'].get('obs_change_mean'))}，h "
                     f"{fmt_mean(d['all_rtarsus'].get('h_mean'))}；HWI 观测 {fmt_mean(d['all_hwi'].get('obs_change_mean'))}，"
                     f"h {fmt_mean(d['all_hwi'].get('h_mean'))}")
    lines += ["", "## 对照预测的 RMSE（同一批预测）", ""]
    for name, r in out["analyses"].items():
        for key, lab in (("L_rtarsus", "L 组相对跗跖"), ("G_hwi", "G 组 HWI")):
            b = r["baselines"][key]
            lines.append(f"- {name}，{lab}（{b['n_pairs']} 对）：" + "，".join(f"{k} {v:.4f}" for k, v in b["rmse"].items()))
    lines += ["", "## 其他读数（描述）", "",
              "| 分析 | N 组相对跗跖 | N 组 HWI | L 或 G 组相对跗跖 | 全部对相对跗跖 | 全部对 HWI | 全部对体重 |",
              "|---|---|---|---|---|---|---|"]
    for name, r in out["analyses"].items():
        o = r["other_readouts"]
        lines.append(f"| {name} | {fmt_gain(r['N']['rtarsus'])} | {fmt_gain(r['N']['hwi'])} | {fmt_gain(o['rtarsus_L_or_G'])} | "
                     f"{fmt_gain(o['rtarsus_all'])} | {fmt_gain(o['hwi_all'])} | {fmt_gain(o['mass_all'])} |")
    lines += ["", "三段腿骨（相对体重；描述）：", ""]
    for name, r in out["analyses"].items():
        lines.append(f"- {name}：" + "；".join(
            f"{ZH[b]} 全部 {fmt_gain(v['all'])}（{v['all'].get('n_pairs', 0)} 对），L 或 G {fmt_gain(v['L_or_G'])}"
            for b, v in r["bones"].items()))
    if mode == "real":
        lines += ["", "## 稳健性", ""]
        for name, r in out["analyses"].items():
            if "robustness" not in r:
                continue
            for k, v in r["robustness"].items():
                lines.append(f"- {name}，{ROBUST_ZH.get(k, k)}：{fmt_gain(v)}（{v.get('n_pairs', 0)} 对）")
    if mode == "stage0":
        s = out.get("stage0_summary", {})
        lines += ["", "## 第 0 步小结（据此定预注册的读数）", ""]
        for k, v in s.items():
            lines.append(f"- {k}：{v}")
    if out.get("wright"):
        w = out["wright"]
        lines += ["", f"Wright 等 2016：{w.get('n_species')} 种，与物种表共有 {w.get('n_overlap')} 种，"
                  f"平均差（已减去）{json.dumps(w.get('offsets'), ensure_ascii=False)}，补上 {w.get('n_filled_values')} 个值。"]
    return "\n".join(lines) + "\n"


def stage0_summary(analyses: dict) -> dict:
    sig = lambda g: bool(g.get("n_pred")) and g["ci95"][0] > 0  # noqa: E731
    eff = [a for a in analyses if a.startswith("effect")]
    nul = [a for a in analyses if a.startswith("null")]
    return {
        "I1 功效（效应情景各种子 95% 区间下限 > 0）": {a: sig(analyses[a]["I1"]) for a in eff},
        "I1 假阳性（零模型 95% 区间下限 > 0）": {a: sig(analyses[a]["I1"]) for a in nul},
        "I2 假阳性（全部情景；模拟里 HWI 不受习性影响）": {a: sig(analyses[a]["I2"]) for a in analyses},
        "备选：L 或 G 组相对跗跖": {a: sig(analyses[a]["other_readouts"]["rtarsus_L_or_G"]) for a in analyses},
        "备选：全部对相对跗跖": {a: sig(analyses[a]["other_readouts"]["rtarsus_all"]) for a in analyses},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["stage0", "real"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="stage0 为数据种子（默认 13 14），real 为训练种子（默认 1 2）")
    ap.add_argument("--trees", type=int, nargs="+", default=[21, 41, 61, 81, 100])
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(N_FOLDS)))
    ap.add_argument("--scenarios", nargs="+", default=["effect", "null"])
    ap.add_argument("--no-ericson", action="store_true")
    ap.add_argument("--ckpt-root", default=None)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--table", default=TABLE, help="只供测试：real 模式换用别的表（此时不核对 md5）")
    ap.add_argument("--skip-prereg", action="store_true", help="只供测试：real 模式不检查预注册")
    args = ap.parse_args()
    args.seeds = args.seeds or ([13, 14] if args.mode == "stage0" else [1, 2])
    real = args.mode == "real"
    if real:
        if args.skip_prereg:
            print("测试：不检查预注册")
        elif not PREREG.exists():
            sys.exit(f"缺少预注册 {PREREG.relative_to(ROOT)}；真实数据只在预注册提交之后运行")
        if args.table != TABLE:
            print(f"测试：用 {args.table}，不核对物种主表的 md5")
        elif md5(ROOT / TABLE) != TABLE_MD5:
            sys.exit("物种主表的 md5 与主分析不同，停止")
    out_dir = Path(args.out or ROOT / "outputs/island_v1" / args.mode)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(args)
    print(f"{len(tasks)} 个（模型，折）任务")
    results = run_all(tasks, args.workers, fn=run_island_task)

    analyses, frames, wright_info = {}, [], {}
    for name in dict.fromkeys(t["analysis"] for t in tasks):
        rs = [r for r in results if r["analysis"] == name]
        t0 = next(t for t in tasks if t["analysis"] == name)
        table = pd.read_csv(ROOT / t0["table_rel"])
        meta = annotate(load_pairs(ROOT, table), table)
        if real and not wright_info:
            wright_info = observed(ROOT, table, True)[1]
        keep = ["pairID", "tip_I", "tip_M", "family_I", "family_M", "order_I", "loco_I", "loco_M", "migr_I", "migr_M",
                "L", "G", "N", "toward_T", "away_T", "mig_down", "G_ponti", "cross_family", "in_filtered",
                "p_tarsus_I", "p_mass_I", "p_hwi_I"]
        df, perm = collect(rs, meta[keep])
        analyses[name] = summarize(df, perm, args.n_boot, real)
        analyses[name]["linear_fits"] = {f"{r['key']}_f{r['fold']}": r["linear"] for r in rs}
        frames.append(df.assign(analysis=name))
        r = analyses[name]
        print(f"{name}：{r['n_pairs']} 对，I1 {fmt_gain(r['I1'])}；I2 {fmt_gain(r['I2'])}", flush=True)

    out = {"mode": args.mode, "design": DESIGN, "n_tasks": len(tasks), "n_perm": args.n_perm, "n_boot": args.n_boot,
           "analyses": analyses}
    if real:
        out["wright"] = {k: v for k, v in wright_info.items()}
        out["table"] = args.table
        out["table_md5"] = md5(ROOT / args.table)
    else:
        out["stage0_summary"] = stage0_summary(analyses)
    (out_dir / "results.json").write_text(json.dumps(strip_boot(out), ensure_ascii=False, indent=1, default=float),
                                          encoding="utf-8")
    (out_dir / "report.md").write_text(report(args.mode, out), encoding="utf-8")
    pd.concat(frames, ignore_index=True).to_csv(out_dir / "predictions.csv.gz", index=False)
    print(f"写入 {out_dir}")


if __name__ == "__main__":
    main()
