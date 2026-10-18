"""探索：全身形态在各层的习性读数（设计 docs/design/探索_分层模块_2026-10-18.md）。探索性分析，不判定。

模型集（缺失的检查点跳过并记下）：
- real_hab3：主分析的三层模型（Hackett 第 21、41、61、81、100 棵树 × 种子 1、2 × 5 折）；real_hab3_ericson：Ericson 第 21、41、61 棵 × 种子 1。
- real_l1：v2.1 只训第一层的模型（同树、同种子、同折），作"不分层"的对照。
- sim_v5_effect、sim_v5_null：主分析第 0 阶段的三层模型与模拟表（只有腿随运动方式变；零模型）。
- sim_v6_hab3、sim_v6_l1：全身模块模拟表（scripts/sim_modular_tables.py，种子 15、16）上新训练的三层与只训第一层的模型。
每个模型集的合并结果存到 outputs/explore_modules_v1/sets/<模型集>.npz；--report-only 读取已有的全部模型集写报告。
在 EvoWM 根目录运行：python scripts/explore_modules_v1.py --sets real_hab3 real_l1 ...；python scripts/explore_modules_v1.py --report-only
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evowm.eval.runs import N_FOLDS, run_all  # noqa: E402
from evowm.eval.wholebody import TYPES, run_wholebody_task  # noqa: E402

OUT = ROOT / "outputs/explore_modules_v1"
TREES = [21, 41, 61, 81, 100]
SETS = {
    "real_hab3": dict(trees=TREES, seeds=[1, 2], root="outputs/hjepa/ckpts/real_units_v1", ck="evowm_hab3",
                      name="hackett_t{t:03d}_s{s}", tree="hackett", table="data/processed/species_traits.csv"),
    "real_hab3_ericson": dict(trees=[21, 41, 61], seeds=[1], root="outputs/hjepa/ckpts/real_units_v1", ck="evowm_hab3",
                              name="ericson_t{t:03d}_s{s}", tree="ericson", table="data/processed/species_traits.csv"),
    "real_l1": dict(trees=TREES, seeds=[1, 2], root="outputs/hjepa/ckpts/real_v2", ck="evowm_l1",
                    name="hackett_t{t:03d}_s{s}", tree="hackett", table="data/processed/species_traits.csv"),
    "sim_v5_effect": dict(trees=TREES, seeds=[13, 14], root="outputs/hjepa/ckpts/stage0_units_v1", ck="evowm_hab3",
                          name="effect_s{s}_t{t:03d}", tree="hackett",
                          table="data/processed/sim/recovery_v5/effect_s{s}/species_traits.csv"),
    "sim_v5_null": dict(trees=TREES, seeds=[13, 14], root="outputs/hjepa/ckpts/stage0_units_v1", ck="evowm_hab3",
                        name="null_s{s}_t{t:03d}", tree="hackett",
                        table="data/processed/sim/recovery_v5/null_s{s}/species_traits.csv"),
    "sim_v6_hab3": dict(trees=TREES, seeds=[15, 16], root="outputs/hjepa/ckpts/explore_v6_hab3", ck="evowm_hab3",
                        name="modular_s{s}_t{t:03d}", tree="hackett",
                        table="data/processed/sim/explore_v6/modular_s{s}/species_traits.csv"),
    "sim_v6_l1": dict(trees=TREES, seeds=[15, 16], root="outputs/hjepa/ckpts/explore_v6_l1", ck="evowm_l1",
                      name="modular_s{s}_t{t:03d}", tree="hackett",
                      table="data/processed/sim/explore_v6/modular_s{s}/species_traits.csv"),
}
SET_ZH = {"real_hab3": "真实·三层", "real_hab3_ericson": "真实·三层（Ericson）", "real_l1": "真实·只训第一层",
          "sim_v5_effect": "模拟 v5·只有腿", "sim_v5_null": "模拟 v5·零模型", "sim_v6_hab3": "模拟 v6·全身模块·三层",
          "sim_v6_l1": "模拟 v6·全身模块·只训第一层"}
GROUPS = {
    "后肢": ["m_tarsus", "k_femur", "k_tibiotarsus", "k_tarsometatarsus"],
    "飞行": ["m_wing", "m_kipps", "m_hwi", "k_humerus", "k_ulna", "k_radius", "k_carpometacarpus", "k_keel"],
    "喙": ["m_beak_culmen", "m_beak_nares", "m_beak_width", "m_beak_depth"],
    "其他": ["m_tail", "m_secondary1", "k_digit2", "k_furcula", "k_skull", "k_sclerotic"],
    "体重": ["m_mass"],
}
LEVEL_ZH = {"L1": "第一层（物种）", "L2": "第二层（运动方式单位）", "L3": "第三层（栖境单位）"}
TYPE_ZH = {"none": "不变", "within": "栖境内", "cross": "跨栖境"}
COLUMNS = [("L1", "within"), ("L1", "cross"), ("L2", "within"), ("L2", "cross"), ("L3", "cross")]


def build_tasks(names: list[str], n_orders: int) -> tuple[list[dict], dict]:
    tasks, missing = [], {}
    for sname in names:
        cfg = SETS[sname]
        miss = 0
        for s in cfg["seeds"]:
            table = cfg["table"].format(s=s)
            if not (ROOT / table).exists():
                miss += len(cfg["trees"]) * N_FOLDS
                continue
            for t in cfg["trees"]:
                run = cfg["name"].format(t=t, s=s)
                tree_file = f"data/processed/trees/{cfg['tree']}_t{t:03d}.nwk"
                for k in range(N_FOLDS):
                    ck = ROOT / cfg["root"] / f"{run}_f{k}" / f"{cfg['ck']}_object.ckpt"
                    if not ck.exists() or not (ROOT / tree_file).exists():
                        miss += 1
                        continue
                    tasks.append({"root": str(ROOT), "table_rel": table, "tree_file": tree_file, "fold": k, "seed": s,
                                  "ckpt": str(ck), "key": run, "set": sname, "n_orders": n_orders})
        missing[sname] = miss
    return tasks, missing


# ---------------------------------------------------------------------------
def pool(results: list[dict]) -> dict:
    """同一模型集的结果按科相加。"""
    ok = [r for r in results if "error" not in r]
    out = {"n_tasks": len(results), "n_ok": len(ok), "errors": [(r["key"], r["fold"], r["error"]) for r in results
                                                                if "error" in r], "levels": {}}
    if not ok:
        return out
    out["cols"], out["families"] = ok[0]["cols"], ok[0]["families"]
    for lv in ("L1", "L2", "L3"):
        rs = [r[lv] for r in ok if lv in r]
        if not rs:
            continue
        agg = {k: sum(r[k] for r in rs) for k in ("s_eco", "s_no", "n")}
        agg.update({f"lin_{k}": sum(r["lin"][k] for r in rs) for k in ("s_no", "s_ch", "w_eco", "w_no", "n")})
        agg["e2_s"] = sum(r["e2"]["s"] for r in rs)
        agg["e2_n"] = sum(r["e2"]["n"] for r in rs)
        agg["sd"] = np.nanmean(np.stack([r["sd"] for r in rs]), axis=0)
        out["levels"][lv] = {"arrays": agg, "e2_labels": list(rs[0]["e2"]["labels"]), "n_models": len(rs)}
    return out


def save_set(sname: str, pooled: dict) -> None:
    d = OUT / "sets"
    d.mkdir(parents=True, exist_ok=True)
    arrays = {f"{lv}__{k}": v for lv, x in pooled["levels"].items() for k, v in x["arrays"].items()}
    np.savez_compressed(d / f"{sname}.npz", **arrays)
    meta = {k: v for k, v in pooled.items() if k != "levels"}
    meta["levels"] = {lv: {"e2_labels": x["e2_labels"], "n_models": x["n_models"]} for lv, x in pooled["levels"].items()}
    (d / f"{sname}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


def load_set(sname: str) -> dict | None:
    d = OUT / "sets"
    if not (d / f"{sname}.npz").exists():
        return None
    meta = json.loads((d / f"{sname}.json").read_text(encoding="utf-8"))
    z = np.load(d / f"{sname}.npz")
    for lv in list(meta["levels"]):
        meta["levels"][lv]["arrays"] = {k.split("__", 1)[1]: z[k] for k in z.files if k.startswith(f"{lv}__")}
    return meta


# ---------------------------------------------------------------------------
def boot_matrix(F: int, nb: int, seed: int, keep: np.ndarray) -> np.ndarray:
    """按科自助重抽样的权重矩阵 (nb, F)：只在有数据的科（keep）里抽。"""
    idx = np.flatnonzero(keep)
    rng = np.random.default_rng(seed)
    M = np.zeros((nb, F))
    draws = rng.integers(0, len(idx), size=(nb, len(idx)))
    for i in range(nb):
        M[i] = np.bincount(idx[draws[i]], minlength=F)
    return M


def rel_gain(s_eco, s_no, M):
    """相对增益 = 1 − sqrt(SSE_有 / SSE_无)，即告诉模型习性后 RMSE 减少的比例。输入 (F, ...)。"""
    with np.errstate(divide="ignore", invalid="ignore"):
        point = 1 - np.sqrt(s_eco.sum(0) / s_no.sum(0))
        sh = s_eco.shape[1:]
        b = 1 - np.sqrt((M @ s_eco.reshape(len(s_eco), -1)) / (M @ s_no.reshape(len(s_no), -1)))
    return point, b.reshape((len(M),) + sh)


def summarize_set(sname: str, d: dict, nb: int) -> dict:
    cols = d["cols"]
    res = {"n_ok": d["n_ok"], "n_tasks": d["n_tasks"], "errors": d["errors"][:20], "levels": {}}
    for lv, x in d["levels"].items():
        A = x["arrays"]
        F = A["n"].shape[0]
        M = boot_matrix(F, nb, 0, A["n"].sum(axis=(1, 2)) > 0)
        g, gb = rel_gain(A["s_eco"], A["s_no"], M)                       # (T, C)
        n = A["n"].sum(0)
        lo, hi = np.nanpercentile(gb, [2.5, 97.5], axis=0)
        lg, lgb = rel_gain(A["lin_s_ch"], A["lin_s_no"], M)               # 线性：只用变化对无习性
        wg, wgb = rel_gain(A["lin_w_eco"], A["lin_w_no"], M)              # 世界模型，同一批预测
        dlo, dhi = np.nanpercentile(wgb - lgb, [2.5, 97.5], axis=0)
        groups = {}
        for gname, gcols in GROUPS.items():
            j = [cols.index(c) for c in gcols if c in cols]
            gm = np.nanmean(gb[:, :, j], axis=2)
            groups[gname] = {"gain": np.nanmean(g[:, j], axis=1).tolist(),
                             "lo": np.nanpercentile(gm, 2.5, axis=0).tolist(), "hi": np.nanpercentile(gm, 97.5, axis=0).tolist(),
                             "wm_same": np.nanmean(wg[:, j], axis=1).tolist(), "linear": np.nanmean(lg[:, j], axis=1).tolist(),
                             "wm_minus_linear_lo": np.nanpercentile(np.nanmean(wgb[:, :, j] - lgb[:, :, j], axis=2), 2.5, axis=0).tolist(),
                             "wm_minus_linear_hi": np.nanpercentile(np.nanmean(wgb[:, :, j] - lgb[:, :, j], axis=2), 97.5, axis=0).tolist()}
        # 反事实：按科重抽样的均值，并用训练物种的标准差标准化
        es, en = A["e2_s"], A["e2_n"]
        Me = boot_matrix(F, nb, 1, en.sum(axis=(1, 2)) > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            cm = es.sum(0) / en.sum(0)
            cb = (Me @ es.reshape(F, -1)) / (Me @ en.reshape(F, -1))
        cb = cb.reshape((nb,) + es.shape[1:])
        clo, chi = np.nanpercentile(cb, [2.5, 97.5], axis=0)
        sd = A["sd"]
        res["levels"][lv] = {"types": TYPES, "gain": g, "lo": lo, "hi": hi, "n": n, "groups": groups,
                             "linear_gain": lg, "wm_same": wg, "diff_lo": dlo, "diff_hi": dhi,
                             "e2_labels": x["e2_labels"], "contrast": cm, "c_lo": clo, "c_hi": chi,
                             "contrast_sd": cm / sd[None, :], "sd": sd, "n_models": x["n_models"]}
    return res


def truth_contrasts(sname: str, cols: list, level: str, labels: list) -> np.ndarray | None:
    """模拟的真值对比（相对基准类：运动方式为栖树，栖境为陆地）。真实数据返回 None。"""
    if not sname.startswith("sim_"):
        return None
    import sim_recovery_tables as v5
    from evowm.data.units import REALM_OF
    flight = {}
    if sname.startswith("sim_v6"):
        from sim_modular_tables import FLIGHT_OFFSETS
        flight = FLIGHT_OFFSETS
    effect = sname != "sim_v5_null"
    T = np.zeros((len(labels), len(cols)))
    for k, lab in enumerate(labels):
        for j, c in enumerate(cols):
            if not effect:
                continue
            if level in ("L1", "L2"):
                if c in v5.LEG_SOURCE:
                    off = v5.LEG_OFFSETS[v5.LEG_SOURCE[c]]
                    T[k, j] = off[lab] - off["insessorial"]
                elif c in flight:
                    T[k, j] = flight[c][REALM_OF[lab]] - flight[c]["land"]
            else:
                if c in flight:
                    T[k, j] = flight[c][lab] - flight[c]["land"]
                elif c in v5.LEG_SOURCE:
                    T[k, j] = np.nan          # 栖境一级的腿部真值取决于栖境内的运动方式组成，不比较
    return T


# ---------------------------------------------------------------------------
def pct(x, lo, hi) -> str:
    if not np.isfinite(x):
        return "·"
    mark = "*" if lo > 0 else ("↓" if hi < 0 else "")
    return f"{100 * x:+.1f}%{mark}"


def report(summ: dict, cols: list) -> str:
    L = ["# 探索：全身形态在各层的习性读数", "",
         "设计 `docs/design/探索_分层模块_2026-10-18.md`；脚本 `scripts/explore_modules_v1.py`。探索性分析，不作判定。", "",
         "## 符号与术语", "",
         "- **第一层**：物种一级；**第二层**：运动方式单位（叶序上连续、运动方式相同的一段物种）；"
         "**第三层**：栖境单位（陆、水、空）。",
         "- **栖境内**：前后两步的运动方式不同、栖境相同（如栖树到地栖）；**跨栖境**：栖境也不同（如陆到空、陆到水）。"
         "第三层的步都是跨栖境。涉水算水。",
         "- **相对习性增益**：1 − RMSE(带习性)/RMSE(不带习性)，即告诉模型习性变化后误差减少的比例；"
         "长度类形态扣除体重通路，体重与 HWI 不扣。*：按科自助重抽样 95% 区间下限 > 0；↓：上限 < 0。",
         "- **组**：后肢（跗跖、股骨、胫跗骨、跗跖骨）；飞行（翅长、Kipp 距离、HWI、肱骨、尺骨、桡骨、腕掌骨、龙骨）；"
         "喙（4 项）；其他（尾、次级飞羽、第二指、叉骨、头骨、巩膜环）；体重。组的数值为组内各形态的平均。",
         "- **反事实对比**：把下一步的习性换成某一类的典型值，读出与基准类（栖树；栖境为陆地）的差，"
         "以训练物种（单位）的标准差为单位。",
         "- **线性**：每项形态各自的只用变化的线性基线；与世界模型比较时用同一批预测。",
         "- **模拟 v6 的真值**：后肢随运动方式变（同 v5），飞行器官只随栖境变，喙与其他不随习性变。", ""]
    L += ["## 运行概况", ""]
    for s, r in summ.items():
        lv_txt = "、".join("{}（{} 个）".format(lv, x["n_models"]) for lv, x in r["levels"].items())
        L.append(f"- {SET_ZH.get(s, s)}：成功 {r['n_ok']}/{r['n_tasks']} 个（模型，折）；层：{lv_txt}")
        for e in r["errors"][:5]:
            L.append(f"  - 出错：{e[0]} 折 {e[1]}：{e[2]}")
    L += ["", "## 1. 各组的相对习性增益", "",
          "| 模型集 | 组 | " + " | ".join(f"{lv} {TYPE_ZH[t]}" for lv, t in COLUMNS) + " |",
          "|---|---|" + "---|" * len(COLUMNS)]
    for s, r in summ.items():
        for gname in GROUPS:
            cells = []
            for lv, t in COLUMNS:
                x = r["levels"].get(lv)
                if x is None:
                    cells.append("·")
                    continue
                ti = TYPES.index(t)
                gg = x["groups"][gname]
                cells.append(pct(gg["gain"][ti], gg["lo"][ti], gg["hi"][ti]))
            L.append(f"| {SET_ZH.get(s, s)} | {gname} | " + " | ".join(cells) + " |")
    for s in ("real_hab3", "sim_v6_hab3"):
        if s not in summ:
            continue
        r = summ[s]
        L += ["", f"## 2. 逐项形态的相对习性增益：{SET_ZH[s]}", "",
              "| 形态 | 组 | " + " | ".join(f"{lv} {TYPE_ZH[t]}" for lv, t in COLUMNS) + " |",
              "|---|---|" + "---|" * len(COLUMNS)]
        for gname, gcols in GROUPS.items():
            for c in gcols:
                if c not in cols:
                    continue
                j = cols.index(c)
                cells = []
                for lv, t in COLUMNS:
                    x = r["levels"].get(lv)
                    ti = TYPES.index(t)
                    cells.append("·" if x is None else pct(x["gain"][ti, j], x["lo"][ti, j], x["hi"][ti, j]))
                L.append(f"| {c} | {gname} | " + " | ".join(cells) + " |")
    L += ["", "## 3. 反事实对比（组平均，单位：标准差）", ""]
    for s, r in summ.items():
        for lv in ("L1", "L2", "L3"):
            x = r["levels"].get(lv)
            if x is None:
                continue
            labs = x["e2_labels"]
            L.append(f"- {SET_ZH.get(s, s)}，{LEVEL_ZH[lv]}：" + "；".join(
                f"{lab}：" + "，".join(f"{g} {np.nanmean(x['contrast_sd'][k, [cols.index(c) for c in gc if c in cols]]):+.2f}"
                                     for g, gc in GROUPS.items() if g != "体重")
                for k, lab in enumerate(labs)))
    L += ["", "## 4. 模拟：反事实与真值", ""]
    for s, r in summ.items():
        for lv, x in r["levels"].items():
            T = truth_contrasts(s, cols, lv, x["e2_labels"])
            if T is None:
                continue
            cm = x["contrast"]
            eff = np.isfinite(T) & (T != 0) & np.isfinite(cm)
            nul = np.isfinite(T) & (T == 0) & np.isfinite(cm)
            rho = spearmanr(cm[eff], T[eff]).statistic if eff.sum() > 3 else float("nan")
            sign = float(np.mean(np.sign(cm[eff]) == np.sign(T[eff]))) if eff.any() else float("nan")
            ratio = float(np.median(cm[eff] / T[eff])) if eff.any() else float("nan")
            L.append(f"- {SET_ZH[s]}，{LEVEL_ZH[lv]}：有效应的格子 {int(eff.sum())} 个，与真值 Spearman {rho:+.2f}，"
                     f"同号 {sign:.2f}，大小中位比 {ratio:.2f}；无效应的格子 {int(nul.sum())} 个，"
                     f"|对比| 中位 {np.median(np.abs(cm[nul])) if nul.any() else float('nan'):.4f}"
                     f"（有效应的为 {np.median(np.abs(cm[eff])) if eff.any() else float('nan'):.4f}）")
    L += ["", "## 5. 世界模型对线性基线（同一批预测，组平均的相对增益）", "",
          "| 模型集 | 层 | 组 | 世界模型 栖境内 | 线性 栖境内 | 世界模型 跨栖境 | 线性 跨栖境 | 差（跨栖境）95% 区间 |",
          "|---|---|---|---|---|---|---|---|"]
    for s, r in summ.items():
        for lv, x in r["levels"].items():
            for gname, gg in x["groups"].items():
                w, li = gg["wm_same"], gg["linear"]
                L.append(f"| {SET_ZH.get(s, s)} | {lv} | {gname} | {100 * w[1]:+.1f}% | {100 * li[1]:+.1f}% | "
                         f"{100 * w[2]:+.1f}% | {100 * li[2]:+.1f}% | [{100 * gg['wm_minus_linear_lo'][2]:+.1f}%, "
                         f"{100 * gg['wm_minus_linear_hi'][2]:+.1f}%] |")
    L += ["", "## 6. 分层对不分层（信噪比 = 相对增益 / 95% 区间半宽）", ""]
    for a, b in (("sim_v6_hab3", "sim_v6_l1"), ("real_hab3", "real_l1")):
        if a not in summ or b not in summ:
            continue
        for gname in ("后肢", "飞行"):
            parts = []
            for s, lv, t in ((b, "L1", "within"), (b, "L1", "cross"), (a, "L1", "within"), (a, "L1", "cross"),
                             (a, "L2", "within"), (a, "L2", "cross"), (a, "L3", "cross")):
                x = summ[s]["levels"].get(lv)
                if x is None:
                    continue
                ti = TYPES.index(t)
                gg = x["groups"][gname]
                hw = (gg["hi"][ti] - gg["lo"][ti]) / 2
                snr = gg["gain"][ti] / hw if hw > 0 else float("nan")
                parts.append(f"{'不分层' if s == b else '三层'} {lv} {TYPE_ZH[t]} {snr:+.1f}")
            L.append(f"- {SET_ZH[a]} 对 {SET_ZH[b]}，{gname}：" + "；".join(parts))
    return "\n".join(L) + "\n"


def write_csv(summ: dict, cols: list) -> None:
    rows, crow = [], []
    for s, r in summ.items():
        for lv, x in r["levels"].items():
            for ti, t in enumerate(TYPES):
                for j, c in enumerate(cols):
                    rows.append({"set": s, "level": lv, "type": t, "trait": c,
                                 "group": next((g for g, gc in GROUPS.items() if c in gc), ""),
                                 "rel_gain": x["gain"][ti, j], "lo": x["lo"][ti, j], "hi": x["hi"][ti, j],
                                 "n_pred": x["n"][ti, j], "linear_rel_gain": x["linear_gain"][ti, j],
                                 "wm_rel_gain_same": x["wm_same"][ti, j]})
            for k, lab in enumerate(x["e2_labels"]):
                for j, c in enumerate(cols):
                    crow.append({"set": s, "level": lv, "category": lab, "trait": c, "contrast": x["contrast"][k, j],
                                 "lo": x["c_lo"][k, j], "hi": x["c_hi"][k, j], "contrast_sd": x["contrast_sd"][k, j]})
    pd.DataFrame(rows).to_csv(OUT / "traits_gain.csv", index=False)
    pd.DataFrame(crow).to_csv(OUT / "contrasts.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="*", default=[])
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--n-orders", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.report_only:
        unknown = [s for s in args.sets if s not in SETS]
        if unknown:
            sys.exit(f"未知的模型集：{unknown}")
        tasks, missing = build_tasks(args.sets, args.n_orders)
        print(f"{len(tasks)} 个（模型，折）任务；缺失：{missing}", flush=True)
        results = run_all(tasks, args.workers, fn=run_wholebody_task) if tasks else []
        for sname in args.sets:
            rs = [r for r in results if r.get("set") == sname]
            if not rs:
                print(f"{sname}：没有可评估的模型（缺失 {missing.get(sname)}）")
                continue
            pooled = pool(rs)
            pooled["missing"] = missing.get(sname, 0)
            if pooled["n_ok"]:
                save_set(sname, pooled)
            print(f"{sname}：成功 {pooled['n_ok']}/{pooled['n_tasks']}，出错 {len(pooled['errors'])}", flush=True)
            for e in pooled["errors"][:3]:
                print("  ", e)
    summ, cols = {}, None
    for sname in SETS:
        d = load_set(sname)
        if d is None:
            continue
        cols = cols or d["cols"]
        with warnings.catch_warnings():          # 空的格子（如第三层的栖境内）求均值时的提示
            warnings.simplefilter("ignore", RuntimeWarning)
            summ[sname] = summarize_set(sname, d, args.n_boot)
    if not summ:
        print("还没有任何模型集的结果")
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        (OUT / "report.md").write_text(report(summ, cols), encoding="utf-8")
        write_csv(summ, cols)
    js = {s: {"n_ok": r["n_ok"], "n_tasks": r["n_tasks"], "errors": r["errors"],
              "groups": {lv: x["groups"] for lv, x in r["levels"].items()}} for s, r in summ.items()}
    (OUT / "summary.json").write_text(json.dumps(js, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"写入 {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
