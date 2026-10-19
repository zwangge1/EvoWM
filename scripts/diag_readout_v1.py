"""诊断：世界模型的点预测为什么不如照抄，以及变化量读数（设计 docs/design/诊断_读数_2026-10-19.md）。探索性分析，不判定。

与探索 v2（scripts/explore_modules_v2.py）用同样的 7 个模型集、同样的窗口与种子（evowm.eval.readout），每个预测另外算：
照抄、世界模型新读数（前一个的实测值 + 模型预测的变化量）、读数下限、表示照抄。所有读数与方法在同一批预测上比。
结果写到 outputs/diag_readout_v1/：report.md、readout_v1.csv、sets/。
在 EvoWM 根目录运行：python scripts/diag_readout_v1.py --sets real_hab3 ...；python scripts/diag_readout_v1.py --report-only
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evowm.eval.readout import KEYS, run_readout_task  # noqa: E402
from evowm.eval.runs import run_all  # noqa: E402
from evowm.eval.wholebody import TYPES  # noqa: E402
from explore_modules_v1 import SETS, build_tasks  # noqa: E402
from explore_modules_v2 import GROUPS, MAIN_GROUPS, SET_ZH, boot_matrix, group_mean  # noqa: E402

OUT = ROOT / "outputs/diag_readout_v1"
V2_SETS = ROOT / "outputs/explore_modules_v2/sets"
NULL = "sim_v5_null"
COLUMNS = [("L1", "none"), ("L1", "within"), ("L1", "cross"), ("L2", "within"), ("L2", "cross"), ("L3", "cross")]
COL_ZH = {("L1", "none"): "第一层·不变", ("L1", "within"): "第一层·同环境", ("L1", "cross"): "第一层·换环境",
          ("L2", "within"): "第二层·同环境", ("L2", "cross"): "第二层·换环境", ("L3", "cross"): "第三层·换环境"}
ROWS = [("l_no", "简单方法·不告诉习性"), ("l_full", "简单方法·全部 47 项"), ("a_eco", "世界模型·原读数"),
        ("d_eco", "世界模型·新读数"), ("floor", "读数下限"), ("zcopy", "表示照抄")]
ARRAYS = KEYS + ("n",)


# ---------------------------------------------------------------------------
def pool(results: list[dict]) -> dict:
    ok = [r for r in results if "error" not in r]
    out = {"n_tasks": len(results), "n_ok": len(ok), "levels": {},
           "errors": [(r["key"], r["fold"], r["error"]) for r in results if "error" in r]}
    if not ok:
        return out
    out["cols"], out["families"] = ok[0]["cols"], ok[0]["families"]
    for lv in ("L1", "L2", "L3"):
        rs = [r[lv] for r in ok if lv in r]
        if rs:
            out["levels"][lv] = {"arrays": {k: sum(r[k] for r in rs) for k in ARRAYS}, "n_models": len(rs)}
    return out


def save_set(sname: str, pooled: dict) -> None:
    d = OUT / "sets"
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / f"{sname}.npz", **{f"{lv}__{k}": v for lv, x in pooled["levels"].items()
                                              for k, v in x["arrays"].items()})
    meta = {k: v for k, v in pooled.items() if k != "levels"}
    meta["levels"] = {lv: {"n_models": x["n_models"]} for lv, x in pooled["levels"].items()}
    (d / f"{sname}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


def load_set(sname: str, d: Path = None) -> dict | None:
    d = d or OUT / "sets"
    if not (d / f"{sname}.npz").exists():
        return None
    meta = json.loads((d / f"{sname}.json").read_text(encoding="utf-8"))
    z = np.load(d / f"{sname}.npz")
    for lv in list(meta["levels"]):
        meta["levels"][lv]["arrays"] = {k.split("__", 1)[1]: z[k] for k in z.files if k.startswith(f"{lv}__")}
    return meta


# ---------------------------------------------------------------------------
def _ratio(num, den, M):
    """RMSE 之比 sqrt(num/den)：点估计 (T, C) 与重抽样 (nb, T, C)。"""
    F = num.shape[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        pt = np.sqrt(num.sum(0) / den.sum(0))
        bs = np.sqrt((M @ num.reshape(F, -1)) / (M @ den.reshape(F, -1)))
    return pt, bs.reshape((len(M),) + num.shape[1:])


def summarize(d: dict, nb: int) -> dict:
    res = {"n_ok": d["n_ok"], "n_tasks": d["n_tasks"], "errors": d["errors"][:20], "cols": d["cols"], "levels": {}}
    for lv, x in d["levels"].items():
        A = x["arrays"]
        F = A["n"].shape[0]
        M = boot_matrix(F, nb, 0, A["n"].sum(axis=(1, 2)) > 0)
        r = {"n": A["n"].sum(0), "n_models": x["n_models"]}
        for k in KEYS:                                  # 误差是照抄的几倍
            r[f"vs_copy_{k}"], r[f"vs_copy_{k}_b"] = _ratio(A[k], A["copy"], M)
        for name, num, den in (("g_abs", "a_eco", "a_no"), ("g_dlt", "d_eco", "d_no"),
                               ("g_cat", "l_ch", "l_no"), ("g_full", "l_full", "l_no")):
            pt, bs = _ratio(A[num], A[den], M)           # 增益 = 1 − 误差比
            r[name], r[f"{name}_b"] = 1 - pt, 1 - bs
        r["dlt_vs_full"], r["dlt_vs_full_b"] = _ratio(A["d_eco"], A["l_full"], M)
        r["abs_vs_full"], r["abs_vs_full_b"] = _ratio(A["a_eco"], A["l_full"], M)
        res["levels"][lv] = r
    return res


def check_v2(sname: str, d: dict) -> str:
    """原读数与 v2 同一批预测上的世界模型误差应完全一致（同样的模型、窗口与种子）。"""
    v2 = load_set(sname, V2_SETS)
    if v2 is None:
        return "没有 v2 的结果，跳过"
    parts = []
    for lv, x in d["levels"].items():
        y = v2["levels"].get(lv)
        if y is None:
            continue
        A, B = x["arrays"], y["arrays"]
        if "lin2_w_eco" not in B or A["a_eco"].shape != B["lin2_w_eco"].shape:
            parts.append(f"{lv} 形状不同")
            continue
        worst = 0.0
        for a, bk in (("a_eco", "lin2_w_eco"), ("a_no", "lin2_w_no"), ("l_full", "lin2_l_full"), ("n", "lin2_n")):
            den = np.maximum(np.abs(B[bk]), 1e-12)
            worst = max(worst, float(np.max(np.abs(A[a] - B[bk]) / den)))
        parts.append(f"{lv} 最大相对差 {worst:.1e}")
    return "；".join(parts) if parts else "没有可比的层"


# ---------------------------------------------------------------------------
def _cell_ratio(x: dict, key: str, ti: int, j: list, ci: bool = False) -> str:
    pt = np.nanmean(x[key][ti, j])
    if not np.isfinite(pt):
        return "·"
    if not ci:
        return f"{pt:.2f}"
    lo, hi = np.nanpercentile(group_mean(x[f"{key}_b"][:, ti, :], j), [2.5, 97.5])
    return f"{pt:.3f} [{lo:.3f}, {hi:.3f}]"


CRIT_TRAITS = [("m_tarsus", "相对跗跖"), ("m_hwi", "HWI")]
CRIT_TOL = 1.05


def criteria(summ: dict) -> list:
    """设计里先定下的判据（第一层，相对跗跖与 HWI，三类步共 6 格），决定岛屿预注册是否以新读数为主读数。"""
    need = ("real_hab3", NULL, "sim_v6_hab3")
    miss = [s for s in need if s not in summ or "L1" not in summ[s]["levels"]]
    L = ["## 判据（岛屿预注册用，服务器运行前定下）", "",
         "第一层，相对跗跖与 HWI，三类步共 6 格；三条在 6 格上都成立才算通过（见设计\"已确认\"一节）。", ""]
    if miss:
        return L + [f"缺少 {', '.join(miss)}，无法判定。", ""]
    cols = summ["real_hab3"]["cols"]
    xr, xn, xv = (summ[s]["levels"]["L1"] for s in need)
    L += ["| 读数 | 步 | 真实：新读数 ÷ 全部 47 项（95% 区间下限 ≤ 1） | 空白模拟：新读数 ÷ 照抄（≤ 1.05） | "
          "v6 模拟：新读数 ÷ 原读数（≤ 1.05） |", "|---|---|---|---|---|"]
    ok_all = True
    for c, zh in CRIT_TRAITS:
        j = cols.index(c)
        for ti, t in enumerate(TYPES):
            pt = xr["dlt_vs_full"][ti, j]
            lo, hi = np.nanpercentile(xr["dlt_vs_full_b"][:, ti, j], [2.5, 97.5])
            c1 = bool(np.isfinite(lo) and lo <= 1.0)
            r2 = xn["vs_copy_d_eco"][ti, j]
            c2 = bool(np.isfinite(r2) and r2 <= CRIT_TOL)
            r3 = xv["vs_copy_d_eco"][ti, j] / xv["vs_copy_a_eco"][ti, j]
            c3 = bool(np.isfinite(r3) and r3 <= CRIT_TOL)
            ok_all &= c1 and c2 and c3
            mk = lambda b: "成立" if b else "不成立"  # noqa: E731
            L.append(f"| {zh} | {COL_ZH[('L1', t)][4:]} | {pt:.3f} [{lo:.3f}, {hi:.3f}] {mk(c1)} | {r2:.3f} {mk(c2)} | "
                     f"{r3:.3f} {mk(c3)} |")
    L += ["", f"**结论：{'通过' if ok_all else '不通过'}。**" +
          ("岛屿预注册以新读数为主读数，原读数作次要结果；先用新读数重跑岛屿第 0 步。" if ok_all else
           "岛屿预注册保持原读数；新读数只作主分析的探索性结果。"), ""]
    return L


def report(summ: dict, checks: dict) -> str:
    cols = next(iter(summ.values()))["cols"]
    gj = {g: [cols.index(c) for c in cs if c in cols] for g, cs in GROUPS.items()}
    L = ["# 诊断：世界模型的点预测为什么不如照抄（读数 v1）", "",
         "设计 `docs/design/诊断_读数_2026-10-19.md`；脚本 `scripts/diag_readout_v1.py`。探索性分析，不判定。", "",
         "## 先说数字是什么意思", "",
         "- 用前一个物种（或前一组）预测下一个的身体尺寸，所有方法用同一批预测（前后两边都有实测值）。",
         "- **照抄**：直接把前一个的实测值当预测。",
         "- **简单方法**：\"不告诉习性\"是照抄后按亲缘距离往平均值拉一点；\"全部 47 项\"再把生活习性的变化线性地加上（同 v2）。",
         "- **世界模型·原读数**：模型给出下一个的内部表示，再读回尺寸（主分析与 v1、v2 都用它）。",
         "- **世界模型·新读数**：前一个的实测值，加上模型预测的变化量（读出的\"下一个 − 前一个\"）。模型认为不变时，它就等于照抄。",
         "- **读数下限**：假如模型把下一个的内部表示猜得完全对，原读数还剩的误差，即\"读回尺寸\"这一步本身的误差。",
         "- **表示照抄**：模型原样照抄前一个的内部表示，再读回尺寸。",
         "- 表 1、表 2 的数是\"误差是照抄的几倍\"：1.00 表示和照抄一样，小于 1 表示比照抄准。世界模型与\"全部 47 项\"都是告诉了生活方式的。",
         "- 长度类尺寸先扣掉体型大小的影响，体重与翅尖形状（HWI）不扣。",
         "- 区间：按科重抽样 95% 区间。增益：告诉生活方式以后误差小了百分之几（同 v2）。", ""]
    L += ["## 运行概况", ""]
    for s, r in summ.items():
        lv_txt = "、".join("{}（{} 个模型）".format(lv, x["n_models"]) for lv, x in r["levels"].items())
        L.append(f"- {SET_ZH.get(s, s)}：成功 {r['n_ok']}/{r['n_tasks']}；{lv_txt}；与 v2 核对：{checks.get(s, '·')}")
        for e in r["errors"][:5]:
            L.append(f"  - 出错：{e[0]} 折 {e[1]}：{e[2]}")
    L += [""] + criteria(summ)

    def ratio_table(s: str, groups: list) -> list:
        out = [f"**{SET_ZH.get(s, s)}**", ""]
        for g in groups:
            j = gj[g]
            out += [f"{g}", "", "| 预测 | " + " | ".join(COL_ZH[c] for c in COLUMNS) + " |", "|---|" + "---|" * len(COLUMNS)]
            for key, zh in ROWS:
                cells = []
                for lv, t in COLUMNS:
                    x = summ[s]["levels"].get(lv)
                    cells.append("·" if x is None else _cell_ratio(x, f"vs_copy_{key}", TYPES.index(t), j))
                out.append(f"| {zh} | " + " | ".join(cells) + " |")
            out.append("")
        return out

    L += ["", "## 表 1. 真实鸟类：各种预测的误差是照抄的几倍", ""]
    if "real_hab3" in summ:
        L += ratio_table("real_hab3", MAIN_GROUPS)
    L += ["## 表 2. 模拟检查：同样的比较", ""]
    for s in (NULL, "sim_v6_hab3"):
        if s in summ:
            L += ratio_table(s, ["翅形（翅长、Kipp、HWI）", "跗跖"])

    L += ["## 表 3. 新读数下：告诉生活方式后误差小了百分之几", "",
          "每格四个数：世界模型·新读数 / 世界模型·原读数 / 简单方法·只用类别 / 简单方法·全部 47 项；"
          "方括号为\"新读数 − 全部 47 项\"的 95% 区间。没有减空白水平。", ""]
    cols5 = [c for c in COLUMNS if c[1] != "none"]
    for s in ("real_hab3", "real_hab3_ericson", "sim_v6_hab3", NULL):
        if s not in summ:
            continue
        L += [f"**{SET_ZH.get(s, s)}**", "", "| 部位 | " + " | ".join(COL_ZH[c] for c in cols5) + " |",
              "|---|" + "---|" * len(cols5)]
        for g in MAIN_GROUPS:
            j = gj[g]
            cells = []
            for lv, t in cols5:
                x = summ[s]["levels"].get(lv)
                if x is None:
                    cells.append("·")
                    continue
                ti = TYPES.index(t)
                v = [100 * np.nanmean(x[k][ti, j]) for k in ("g_dlt", "g_abs", "g_cat", "g_full")]
                db = group_mean(x["g_dlt_b"][:, ti, :], j) - group_mean(x["g_full_b"][:, ti, :], j)
                lo, hi = np.nanpercentile(db, [2.5, 97.5]) * 100
                cells.append(" / ".join(f"{u:+.1f}" for u in v) + f" [{lo:+.1f}, {hi:+.1f}]")
            L.append(f"| {g} | " + " | ".join(cells) + " |")
        L.append("")

    L += ["## 表 4. 新读数下谁更准：世界模型·新读数的误差 ÷ 简单方法·全部 47 项的误差", "",
          "两边都告诉了生活方式；小于 1 表示世界模型更准。每格：新读数的比值 [95% 区间]；原读数的同一比值（即 v2 表 4）。", ""]
    for s in ("real_hab3", "real_hab3_ericson", "sim_v6_hab3", NULL):
        if s not in summ:
            continue
        L += [f"**{SET_ZH.get(s, s)}**", "", "| 部位 | " + " | ".join(COL_ZH[c] for c in COLUMNS) + " |",
              "|---|" + "---|" * len(COLUMNS)]
        for g in MAIN_GROUPS:
            j = gj[g]
            cells = []
            for lv, t in COLUMNS:
                x = summ[s]["levels"].get(lv)
                if x is None:
                    cells.append("·")
                    continue
                ti = TYPES.index(t)
                old = np.nanmean(x["abs_vs_full"][ti, j])
                cells.append(_cell_ratio(x, "dlt_vs_full", ti, j, ci=True) + (f"；原 {old:.3f}" if np.isfinite(old) else ""))
            L.append(f"| {g} | " + " | ".join(cells) + " |")
        L.append("")

    L += ["## 表 5. 分层对不分层（第一层，新读数）", "",
          "每格：误差是照抄的几倍 / 告诉生活方式后误差小了百分之几。", ""]
    for a, b in (("real_hab3", "real_l1"), ("sim_v6_hab3", "sim_v6_l1")):
        if a not in summ or b not in summ:
            continue
        L += ["| 模型集 | 部位 | " + " | ".join(COL_ZH[("L1", t)] for t in TYPES) + " |", "|---|---|" + "---|" * 3]
        for s in (a, b):
            for g in ("翅形（翅长、Kipp、HWI）", "跗跖", "体重"):
                j = gj[g]
                x = summ[s]["levels"]["L1"]
                cells = [f"{np.nanmean(x['vs_copy_d_eco'][ti, j]):.2f} / {100 * np.nanmean(x['g_dlt'][ti, j]):+.1f}"
                         for ti in range(3)]
                L.append(f"| {SET_ZH.get(s, s)} | {g} | " + " | ".join(cells) + " |")
        L.append("")

    L += ["## 怎么看（设计里先写下的）", "",
          "1. 读数下限接近或超过照抄（尤其\"不变\"的步）：主要是读数的问题，用新读数。",
          "2. 新读数下世界模型的误差不大于简单方法（表 4 小于 1，或区间含 1）：普通预测不落后，不用重训。",
          "3. 新读数仍明显差于简单方法：问题在预测本身，下一步改成\"预测变化量\"重训（另行确认）。",
          "4. 表 3 里 v2 的翅形、换环境优势还在不在：如果消失，说明那部分优势来自原读数的偏差，要如实写。",
          "5. 先看模拟：空白模拟的新读数应接近照抄；v6 里新读数不应比原读数差。模拟里不成立的读数，真实数据上不解读。"]
    return "\n".join(L) + "\n"


def write_csv(summ: dict) -> None:
    rows = []
    for s, r in summ.items():
        cols = r["cols"]
        for lv, x in r["levels"].items():
            for ti, t in enumerate(TYPES):
                for j, c in enumerate(cols):
                    row = {"set": s, "level": lv, "type": t, "trait": c,
                           "group": next((g for g, gc in GROUPS.items() if c in gc), ""), "n_pred": x["n"][ti, j]}
                    for k in KEYS:
                        row[f"rmse_ratio_{k}_to_copy"] = x[f"vs_copy_{k}"][ti, j]
                    for k in ("g_abs", "g_dlt", "g_cat", "g_full"):
                        row[k] = x[k][ti, j]
                    lo, hi = np.nanpercentile(x["dlt_vs_full_b"][:, ti, j], [2.5, 97.5])
                    row.update(dlt_vs_full=x["dlt_vs_full"][ti, j], dlt_vs_full_lo=lo, dlt_vs_full_hi=hi,
                               abs_vs_full=x["abs_vs_full"][ti, j])
                    rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "readout_v1.csv", index=False)


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
        results = run_all(tasks, args.workers, fn=run_readout_task) if tasks else []
        for sname in args.sets:
            rs = [r for r in results if r.get("set") == sname]
            if not rs:
                print(f"{sname}：没有可评估的模型")
                continue
            pooled = pool(rs)
            pooled["missing"] = missing.get(sname, 0)
            if pooled["n_ok"]:
                save_set(sname, pooled)
            print(f"{sname}：成功 {pooled['n_ok']}/{pooled['n_tasks']}，出错 {len(pooled['errors'])}", flush=True)
            for e in pooled["errors"][:3]:
                print("  ", e)
    summ, checks = {}, {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for sname in SETS:
            d = load_set(sname)
            if d is not None:
                summ[sname] = summarize(d, args.n_boot)
                checks[sname] = check_v2(sname, d)
        if not summ:
            print("还没有任何模型集的结果")
            return
        (OUT / "report.md").write_text(report(summ, checks), encoding="utf-8")
        write_csv(summ)
    for s, c in checks.items():
        print(f"与 v2 核对 {s}：{c}")
    print(f"写入 {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
