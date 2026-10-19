"""预测变化量重训的评估（设计 docs/design/预测变化量_重训_2026-10-19.md）。探索性分析；只有岛屿检验用哪种模型与读数按事先定的
规则自动判定。

新模型（检查点在 outputs/hjepa/ckpts/residual_v1/）用与读数诊断 v1 完全相同的窗口、种子、读数与简单方法
（evowm.eval.readout）；旧模型的结果直接读 outputs/diag_readout_v1/sets，不重算。
结果写到 outputs/residual_v1/：report.md、residual_v1.csv、sets/。
在 EvoWM 根目录运行：python scripts/eval_residual_v1.py --sets real_hab3_res ...；python scripts/eval_residual_v1.py --report-only
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import diag_readout_v1 as dr  # noqa: E402
import explore_modules_v1 as em1  # noqa: E402
from evowm.eval.readout import KEYS, run_readout_task  # noqa: E402
from evowm.eval.runs import run_all  # noqa: E402
from evowm.eval.wholebody import TYPES  # noqa: E402
from explore_modules_v2 import GROUPS, MAIN_GROUPS, group_mean  # noqa: E402

OUT = ROOT / "outputs/residual_v1"
OLD = ROOT / "outputs/diag_readout_v1/sets"
CK = "outputs/hjepa/ckpts/residual_v1"
# 新模型集：与旧模型集相同的表、树、种子、折与检查点名，只是检查点目录不同
RES = {"real_hab3_res": ("real_hab3", f"{CK}/real_hab3"),
       "real_hab3_ericson_res": ("real_hab3_ericson", f"{CK}/real_hab3"),
       "real_l1_res": ("real_l1", f"{CK}/real_l1"),
       "sim_v5_effect_res": ("sim_v5_effect", f"{CK}/sim_v5"),
       "sim_v5_null_res": ("sim_v5_null", f"{CK}/sim_v5"),
       "sim_v6_hab3_res": ("sim_v6_hab3", f"{CK}/sim_v6_hab3"),
       "sim_v6_l1_res": ("sim_v6_l1", f"{CK}/sim_v6_l1")}
for _new, (_old, _root) in RES.items():
    em1.SETS[_new] = dict(em1.SETS[_old], root=_root)
dr.OUT = OUT                                   # pool/save_set/load_set 写到本目录
ZH = {"real_hab3": "真实鸟类", "real_hab3_ericson": "真实鸟类（换一套进化树）", "real_l1": "真实鸟类·不分层模型",
      "sim_v5_effect": "模拟：只有腿随生活方式变", "sim_v5_null": "模拟：生活方式毫无影响（空白）",
      "sim_v6_hab3": "模拟：腿随走动方式、翅膀只随环境变", "sim_v6_l1": "模拟（同上）·不分层模型"}
COLUMNS, COL_ZH = dr.COLUMNS, dr.COL_ZH
CRIT_TRAITS, TOL = dr.CRIT_TRAITS, dr.CRIT_TOL
READOUTS = (("d_eco", "新读数"), ("a_eco", "原读数"))


# ---------------------------------------------------------------------------
def check_same_windows(new: dict, old: dict | None) -> str:
    """新旧两次评估的照抄与简单方法只依赖数据与窗口，应完全相同（核对窗口、种子与同一批预测是否一致）。"""
    if old is None:
        return "没有旧模型的结果，跳过"
    parts = []
    for lv, x in new["levels"].items():
        y = old["levels"].get(lv)
        if y is None:
            continue
        A, B = x["arrays"], y["arrays"]
        worst = max(float(np.max(np.abs(A[k] - B[k]) / np.maximum(np.abs(B[k]), 1e-12))) for k in ("copy", "l_full", "n"))
        parts.append(f"{lv} 最大相对差 {worst:.1e}")
    return "；".join(parts) if parts else "没有可比的层"


def criteria(new: dict, old: dict) -> list:
    """设计第 4 节：第一层，相对跗跖与 HWI，三类步共 6 格；每种读数三条都成立才算通过。"""
    need_new = ("real_hab3", "sim_v5_null", "sim_v6_hab3")
    miss = [f"新模型 {s}" for s in need_new if s not in new or "L1" not in new[s]["levels"]]
    if "sim_v6_hab3" not in old or "L1" not in old["sim_v6_hab3"]["levels"]:
        miss.append("旧模型 sim_v6_hab3")
    L = ["## 判据（岛屿检验用哪种模型与读数，服务器运行前定下）", "",
         "第一层，相对跗跖与 HWI，三类步共 6 格。对新模型的每一种读数，三条在 6 格上都成立才算通过（设计第 4 节）。", ""]
    if miss:
        return L + [f"缺少 {'、'.join(miss)}，暂不能判定。", ""]
    cols = new["real_hab3"]["cols"]
    xr, xn, xv = (new[s]["levels"]["L1"] for s in need_new)
    ov = old["sim_v6_hab3"]["levels"]["L1"]
    passed = {}
    for key, zh in READOUTS:
        L += [f"**新模型·{zh}**", "",
              "| 读数 | 步 | 真实：÷ 全部 47 项（95% 区间下限 ≤ 1） | 空白模拟：÷ 照抄（≤ 1.05） | v6 模拟：÷ 旧模型原读数（≤ 1.05） |",
              "|---|---|---|---|---|"]
        ok_all = True
        for c, czh in CRIT_TRAITS:
            j = cols.index(c)
            for ti, t in enumerate(TYPES):
                with np.errstate(divide="ignore", invalid="ignore"):
                    pt = xr[f"vs_copy_{key}"][ti, j] / xr["vs_copy_l_full"][ti, j]
                    bs = xr[f"vs_copy_{key}_b"][:, ti, j] / xr["vs_copy_l_full_b"][:, ti, j]
                lo, hi = np.nanpercentile(bs, [2.5, 97.5])
                c1 = bool(np.isfinite(lo) and lo <= 1.0)
                r2 = xn[f"vs_copy_{key}"][ti, j]
                c2 = bool(np.isfinite(r2) and r2 <= TOL)
                r3 = xv[f"vs_copy_{key}"][ti, j] / ov["vs_copy_a_eco"][ti, j]
                c3 = bool(np.isfinite(r3) and r3 <= TOL)
                ok_all &= c1 and c2 and c3
                mk = lambda b: "成立" if b else "不成立"  # noqa: E731
                L.append(f"| {czh} | {COL_ZH[('L1', t)][4:]} | {pt:.3f} [{lo:.3f}, {hi:.3f}] {mk(c1)} | {r2:.3f} {mk(c2)} | "
                         f"{r3:.3f} {mk(c3)} |")
        passed[key] = ok_all
        L += ["", f"新模型·{zh}：{'通过' if ok_all else '不通过'}。", ""]
    if passed["d_eco"]:
        verdict = "岛屿检验用新模型，新读数为主读数，原读数作次要结果；先用新模型重跑岛屿第 0 步，再写预注册。"
    elif passed["a_eco"]:
        verdict = "岛屿检验用新模型、原读数；先用新模型重跑岛屿第 0 步，再写预注册。"
    else:
        verdict = "两种读数都不通过：岛屿检验用旧模型、原读数（读数诊断 v1 的结论）。"
    return L + [f"**结论：{verdict}**", ""]


def _ratio_cell(x: dict, key: str, ti: int, j: list) -> str:
    v = np.nanmean(x[f"vs_copy_{key}"][ti, j])
    return f"{v:.2f}" if np.isfinite(v) else "·"


def report(new: dict, old: dict, checks: dict) -> str:
    any_set = next(iter(new.values()))
    cols = any_set["cols"]
    gj = {g: [cols.index(c) for c in cs if c in cols] for g, cs in GROUPS.items()}
    L = ["# 预测变化量：新模型与旧模型（residual v1）", "",
         "设计 `docs/design/预测变化量_重训_2026-10-19.md`；脚本 `scripts/eval_residual_v1.py`。探索性分析，只有岛屿用哪种模型按事先定的规则判定。", "",
         "## 先说数字是什么意思", "",
         "- **旧模型**：原来的结构，预测器直接给出下一个的内部表示。**新模型**：预测器给出\"前一个的表示 + 变化量\"，训练开始时等于照抄表示；其余设置与旧模型完全相同。",
         "- **原读数**：把预测的表示读回身体尺寸。**新读数**：前一个的实测值，加上模型预测的变化量（同读数诊断 v1）。",
         "- 表 1、表 2 的数是\"误差是照抄的几倍\"：1.00 表示和照抄一样，小于 1 表示比照抄准。都是告诉了生活方式的预测。",
         "- 所有方法用同一批预测（前后两边都有实测值），与读数诊断 v1 完全相同的窗口与种子。长度类尺寸先扣掉体型大小的影响，体重与 HWI 不扣。",
         "- 区间：按科重抽样 95% 区间。增益：告诉生活方式以后误差小了百分之几。", "",
         "## 运行概况", ""]
    for s, r in new.items():
        lv_txt = "、".join("{}（{} 个模型）".format(lv, x["n_models"]) for lv, x in r["levels"].items())
        L.append(f"- 新模型·{ZH.get(s, s)}：成功 {r['n_ok']}/{r['n_tasks']}；{lv_txt}；与旧模型的照抄、简单方法核对：{checks.get(s, '·')}")
        for e in r["errors"][:5]:
            L.append(f"  - 出错：{e[0]} 折 {e[1]}：{e[2]}")
    L += [""] + criteria(new, old)

    rows = [("l_no", "简单方法·不告诉习性", "new"), ("l_full", "简单方法·全部 47 项", "new"),
            ("a_eco", "旧模型·原读数", "old"), ("d_eco", "旧模型·新读数", "old"),
            ("a_eco", "新模型·原读数", "new"), ("d_eco", "新模型·新读数", "new")]

    def ratio_table(s: str, groups: list) -> list:
        out = [f"**{ZH.get(s, s)}**", ""]
        for g in groups:
            j = gj[g]
            out += [g, "", "| 预测 | " + " | ".join(COL_ZH[c] for c in COLUMNS) + " |", "|---|" + "---|" * len(COLUMNS)]
            for key, zh, src in rows:
                d = (new if src == "new" else old).get(s)
                cells = []
                for lv, t in COLUMNS:
                    x = None if d is None else d["levels"].get(lv)
                    cells.append("·" if x is None else _ratio_cell(x, key, TYPES.index(t), j))
                out.append(f"| {zh} | " + " | ".join(cells) + " |")
            out.append("")
        return out

    L += ["## 表 1. 真实鸟类：误差是照抄的几倍（旧模型与新模型）", ""]
    if "real_hab3" in new:
        L += ratio_table("real_hab3", MAIN_GROUPS)
    L += ["## 表 2. 模拟检查：同样的比较", ""]
    for s in ("sim_v5_null", "sim_v6_hab3"):
        if s in new:
            L += ratio_table(s, ["翅形（翅长、Kipp、HWI）", "跗跖"])

    L += ["## 表 3. 谁更准：新模型的误差 ÷ 简单方法·全部 47 项的误差", "",
          "两边都告诉了生活方式；小于 1 表示新模型更准。每格：新读数的比值 [95% 区间]；原读数的比值；旧模型新读数的比值（读数诊断 v1 表 4）。", ""]
    for s in ("real_hab3", "real_hab3_ericson", "sim_v6_hab3", "sim_v5_null"):
        if s not in new:
            continue
        L += [f"**{ZH.get(s, s)}**", "", "| 部位 | " + " | ".join(COL_ZH[c] for c in COLUMNS) + " |", "|---|" + "---|" * len(COLUMNS)]
        for g in MAIN_GROUPS:
            j = gj[g]
            cells = []
            for lv, t in COLUMNS:
                x = new[s]["levels"].get(lv)
                if x is None:
                    cells.append("·")
                    continue
                ti = TYPES.index(t)
                pt = np.nanmean(x["dlt_vs_full"][ti, j])
                lo, hi = np.nanpercentile(group_mean(x["dlt_vs_full_b"][:, ti, :], j), [2.5, 97.5])
                ab = np.nanmean(x["abs_vs_full"][ti, j])
                y = old.get(s, {}).get("levels", {}).get(lv)
                od = np.nanmean(y["dlt_vs_full"][ti, j]) if y is not None else np.nan
                cells.append(f"{pt:.3f} [{lo:.3f}, {hi:.3f}]；原 {ab:.3f}；旧 {od:.3f}")
            L.append(f"| {g} | " + " | ".join(cells) + " |")
        L.append("")

    L += ["## 表 4. 告诉生活方式后误差小了百分之几", "",
          "每格五个数：新模型·新读数 / 新模型·原读数 / 旧模型·新读数 / 简单方法·只用类别 / 简单方法·全部 47 项；"
          "方括号为\"新模型·新读数 − 全部 47 项\"的 95% 区间。没有减空白水平。", ""]
    cols5 = [c for c in COLUMNS if c[1] != "none"]
    for s in ("real_hab3", "real_hab3_ericson", "sim_v6_hab3", "sim_v5_null"):
        if s not in new:
            continue
        L += [f"**{ZH.get(s, s)}**", "", "| 部位 | " + " | ".join(COL_ZH[c] for c in cols5) + " |", "|---|" + "---|" * len(cols5)]
        for g in MAIN_GROUPS:
            j = gj[g]
            cells = []
            for lv, t in cols5:
                x = new[s]["levels"].get(lv)
                if x is None:
                    cells.append("·")
                    continue
                ti = TYPES.index(t)
                y = old.get(s, {}).get("levels", {}).get(lv)
                v = [100 * np.nanmean(x["g_dlt"][ti, j]), 100 * np.nanmean(x["g_abs"][ti, j]),
                     100 * np.nanmean(y["g_dlt"][ti, j]) if y is not None else np.nan,
                     100 * np.nanmean(x["g_cat"][ti, j]), 100 * np.nanmean(x["g_full"][ti, j])]
                db = group_mean(x["g_dlt_b"][:, ti, :], j) - group_mean(x["g_full_b"][:, ti, :], j)
                lo, hi = np.nanpercentile(db, [2.5, 97.5]) * 100
                cells.append(" / ".join(f"{u:+.1f}" if np.isfinite(u) else "·" for u in v) + f" [{lo:+.1f}, {hi:+.1f}]")
            L.append(f"| {g} | " + " | ".join(cells) + " |")
        L.append("")

    L += ["## 表 5. 分层对不分层（新模型，第一层，新读数）", "",
          "每格：误差是照抄的几倍 / 告诉生活方式后误差小了百分之几。", ""]
    for a, b in (("real_hab3", "real_l1"), ("sim_v6_hab3", "sim_v6_l1")):
        if a not in new or b not in new:
            continue
        L += ["| 模型集 | 部位 | " + " | ".join(COL_ZH[("L1", t)] for t in TYPES) + " |", "|---|---|" + "---|" * 3]
        for s in (a, b):
            for g in ("翅形（翅长、Kipp、HWI）", "跗跖", "体重"):
                j = gj[g]
                x = new[s]["levels"]["L1"]
                cells = [f"{np.nanmean(x['vs_copy_d_eco'][ti, j]):.2f} / {100 * np.nanmean(x['g_dlt'][ti, j]):+.1f}"
                         for ti in range(3)]
                L.append(f"| 新模型·{ZH.get(s, s)} | {g} | " + " | ".join(cells) + " |")
        L.append("")

    L += ["## 怎么看（设计里先写下的）", "",
          "1. 岛屿检验用哪种模型与读数：见上面的判据，按规则定。",
          "2. 描述：新模型在第一层\"不变\"的步上离简单方法差多少（表 3）；第二、三层是否不再差于照抄（表 1）。",
          "3. 描述：翅形换环境上告诉生活方式的增益是否仍高于简单方法（表 4）；新结构是否让模型对生活方式的反应变小。",
          "4. 描述：分层对不分层（表 5）。"]
    return "\n".join(L) + "\n"


def write_csv(new: dict) -> None:
    rows = []
    for s, r in new.items():
        cols = r["cols"]
        for lv, x in r["levels"].items():
            for ti, t in enumerate(TYPES):
                for j, c in enumerate(cols):
                    row = {"set": f"{s}_res", "level": lv, "type": t, "trait": c,
                           "group": next((g for g, gc in GROUPS.items() if c in gc), ""), "n_pred": x["n"][ti, j]}
                    for k in KEYS:
                        row[f"rmse_ratio_{k}_to_copy"] = x[f"vs_copy_{k}"][ti, j]
                    for k in ("g_abs", "g_dlt", "g_cat", "g_full"):
                        row[k] = x[k][ti, j]
                    lo, hi = np.nanpercentile(x["dlt_vs_full_b"][:, ti, j], [2.5, 97.5])
                    row.update(dlt_vs_full=x["dlt_vs_full"][ti, j], dlt_vs_full_lo=lo, dlt_vs_full_hi=hi,
                               abs_vs_full=x["abs_vs_full"][ti, j])
                    rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "residual_v1.csv", index=False)


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
        unknown = [s for s in args.sets if s not in RES]
        if unknown:
            sys.exit(f"未知的模型集：{unknown}（可选 {list(RES)}）")
        tasks, missing = em1.build_tasks(args.sets, args.n_orders)
        print(f"{len(tasks)} 个（模型，折）任务；缺失：{missing}", flush=True)
        results = run_all(tasks, args.workers, fn=run_readout_task) if tasks else []
        for sname in args.sets:
            rs = [r for r in results if r.get("set") == sname]
            if not rs:
                print(f"{sname}：没有可评估的模型")
                continue
            pooled = dr.pool(rs)
            pooled["missing"] = missing.get(sname, 0)
            if pooled["n_ok"]:
                dr.save_set(sname, pooled)
            print(f"{sname}：成功 {pooled['n_ok']}/{pooled['n_tasks']}，出错 {len(pooled['errors'])}", flush=True)
            for e in pooled["errors"][:3]:
                print("  ", e)
    new, old, checks = {}, {}, {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for sname, (base, _) in RES.items():
            d = dr.load_set(sname, OUT / "sets")
            if d is None:
                continue
            o = dr.load_set(base, OLD)
            new[base] = dr.summarize(d, args.n_boot)
            checks[base] = check_same_windows(d, o)
            if o is not None:
                old[base] = dr.summarize(o, args.n_boot)
        for base in ("sim_v6_hab3",):           # 判据第 3 条要旧模型的 v6，即使新模型还没评估
            if base not in old:
                o = dr.load_set(base, OLD)
                if o is not None:
                    old[base] = dr.summarize(o, args.n_boot)
        if not new:
            print("还没有任何新模型集的结果")
            return
        (OUT / "report.md").write_text(report(new, old, checks), encoding="utf-8")
        write_csv(new)
    for s, c in checks.items():
        print(f"与旧模型核对 {s}：{c}")
    print(f"写入 {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
