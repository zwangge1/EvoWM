"""探索 v2：身体各部位在哪一层随生活方式变，加"全部习性列"的线性基线（设计 docs/design/探索_分层模块_2026-10-18.md
的补充，见 docs/design/探索_分层模块_v2_2026-10-19.md）。探索性分析，不判定。

与 v1（scripts/explore_modules_v1.py）用同样的模型集与评估（evowm.eval.wholebody），另外：
- 线性基线多一种：在无习性的预测上，用岭回归把世界模型拿到的同一组习性变化（47 列习性差与 3 个习性组标记）
  线性地加进去。五种预测（世界模型有、无习性；线性无习性、只用类别、全部习性列）用同一批。
- 报告用"比空白水平好多少个百分点"：每个数都减去模拟零模型（sim_v5_null）在同一格的值。
- 分组更细：翅形（翅长、Kipp、HWI）、跗跖、腿骨、飞行骨骼、喙、尾与次级飞羽、头骨与巩膜环、体重。
结果写到 outputs/explore_modules_v2/：report.md、traits_gain_v2.csv、sets/。
在 EvoWM 根目录运行：python scripts/explore_modules_v2.py --sets real_hab3 ...；python scripts/explore_modules_v2.py --report-only
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

from evowm.eval.runs import run_all  # noqa: E402
from evowm.eval.wholebody import TYPES, run_wholebody_task  # noqa: E402
from explore_modules_v1 import SETS, build_tasks  # noqa: E402

OUT = ROOT / "outputs/explore_modules_v2"
NULL = "sim_v5_null"
SET_ZH = {"real_hab3": "真实鸟类", "real_hab3_ericson": "真实鸟类（换一套进化树）", "real_l1": "真实鸟类·不分层模型",
          "sim_v5_effect": "模拟：只有腿随生活方式变", "sim_v5_null": "模拟：生活方式毫无影响（空白）",
          "sim_v6_hab3": "模拟：腿随走动方式、翅膀只随环境变", "sim_v6_l1": "模拟（同上）·不分层模型"}
GROUPS = {
    "翅形（翅长、Kipp、HWI）": ["m_wing", "m_kipps", "m_hwi"],
    "跗跖": ["m_tarsus"],
    "腿骨（三段）": ["k_femur", "k_tibiotarsus", "k_tarsometatarsus"],
    "飞行骨骼": ["k_humerus", "k_ulna", "k_radius", "k_carpometacarpus", "k_keel", "k_furcula", "k_digit2"],
    "喙": ["m_beak_culmen", "m_beak_nares", "m_beak_width", "m_beak_depth"],
    "尾与次级飞羽": ["m_tail", "m_secondary1"],
    "头骨与巩膜环": ["k_skull", "k_sclerotic"],
    "体重": ["m_mass"],
}
MAIN_GROUPS = ["翅形（翅长、Kipp、HWI）", "跗跖", "喙", "体重"]
COLUMNS = [("L1", "within"), ("L1", "cross"), ("L2", "within"), ("L2", "cross"), ("L3", "cross")]
COL_ZH = {("L1", "within"): "第一层·同环境", ("L1", "cross"): "第一层·换环境", ("L2", "within"): "第二层·同环境",
          ("L2", "cross"): "第二层·换环境", ("L3", "cross"): "第三层·换环境"}
LIN2 = ("w_eco", "w_no", "l_no", "l_ch", "l_full", "n")


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
        if not rs:
            continue
        A = {k: sum(r[k] for r in rs) for k in ("s_eco", "s_no", "n")}
        A.update({f"lin2_{k}": sum(r["lin2"][k] for r in rs) for k in LIN2})
        out["levels"][lv] = {"arrays": A, "n_models": len(rs)}
    return out


def save_set(sname: str, pooled: dict) -> None:
    d = OUT / "sets"
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / f"{sname}.npz", **{f"{lv}__{k}": v for lv, x in pooled["levels"].items()
                                              for k, v in x["arrays"].items()})
    meta = {k: v for k, v in pooled.items() if k != "levels"}
    meta["levels"] = {lv: {"n_models": x["n_models"]} for lv, x in pooled["levels"].items()}
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
    idx = np.flatnonzero(keep)
    draws = np.random.default_rng(seed).integers(0, len(idx), size=(nb, len(idx)))
    M = np.zeros((nb, F))
    for i in range(nb):
        M[i] = np.bincount(idx[draws[i]], minlength=F)
    return M


def gain_and_boot(num: np.ndarray, den: np.ndarray, M: np.ndarray):
    """相对增益 1 − sqrt(num/den)：点估计 (T, C) 与重抽样 (nb, T, C)。"""
    F = num.shape[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        pt = 1 - np.sqrt(num.sum(0) / den.sum(0))
        bs = 1 - np.sqrt((M @ num.reshape(F, -1)) / (M @ den.reshape(F, -1)))
    return pt, bs.reshape((len(M),) + num.shape[1:])


def ratio_and_boot(num: np.ndarray, den: np.ndarray, M: np.ndarray):
    """RMSE 之比 sqrt(num/den)。"""
    F = num.shape[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        pt = np.sqrt(num.sum(0) / den.sum(0))
        bs = np.sqrt((M @ num.reshape(F, -1)) / (M @ den.reshape(F, -1)))
    return pt, bs.reshape((len(M),) + num.shape[1:])


def group_mean(x: np.ndarray, j: list) -> np.ndarray:
    """最后一维为性状：组内平均。"""
    return np.nanmean(x[..., j], axis=-1)


def summarize(d: dict, nb: int) -> dict:
    cols = d["cols"]
    res = {"n_ok": d["n_ok"], "n_tasks": d["n_tasks"], "errors": d["errors"][:20], "levels": {}}
    for lv, x in d["levels"].items():
        A = x["arrays"]
        F = A["n"].shape[0]
        M = boot_matrix(F, nb, 0, A["n"].sum(axis=(1, 2)) > 0)
        g, gb = gain_and_boot(A["s_eco"], A["s_no"], M)
        wm, wmb = gain_and_boot(A["lin2_w_eco"], A["lin2_w_no"], M)
        lc, lcb = gain_and_boot(A["lin2_l_ch"], A["lin2_l_no"], M)
        lf, lfb = gain_and_boot(A["lin2_l_full"], A["lin2_l_no"], M)
        rr, rrb = ratio_and_boot(A["lin2_w_eco"], A["lin2_l_full"], M)
        res["levels"][lv] = {"gain": g, "gain_b": gb, "n": A["n"].sum(0), "wm": wm, "wm_b": wmb, "lc": lc, "lc_b": lcb,
                             "lf": lf, "lf_b": lfb, "ratio": rr, "ratio_b": rrb, "n2": A["lin2_n"].sum(0),
                             "n_models": x["n_models"]}
    res["cols"] = cols
    return res


def fmt(pt: float, lo: float, hi: float, scale: float = 100, digits: int = 1) -> str:
    if not np.isfinite(pt):
        return "·"
    mark = "*" if lo > 0 else ("↓" if hi < 0 else "")
    return f"{scale * pt:+.{digits}f}{mark}"


def corrected(summ: dict, s: str, lv: str, ti: int, j: list):
    """组平均的增益，减去空白（零模型）同一格；重抽样相互独立。返回 (点估计, 下限, 上限)。"""
    x, z = summ[s]["levels"].get(lv), summ[NULL]["levels"].get(lv)
    if x is None or z is None:
        return np.nan, np.nan, np.nan
    pt = np.nanmean(x["gain"][ti, j]) - np.nanmean(z["gain"][ti, j])
    bs = group_mean(x["gain_b"][:, ti, :], j) - group_mean(z["gain_b"][:, ti, :], j)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return pt, lo, hi


def report(summ: dict) -> str:
    cols = next(iter(summ.values()))["cols"]
    gj = {g: [cols.index(c) for c in cs if c in cols] for g, cs in GROUPS.items()}
    L = ["# 探索 v2：身体各部位在哪一层随生活方式变", "",
         "脚本 `scripts/explore_modules_v2.py`。探索性分析，不判定。", "",
         "## 先说数字是什么意思", "",
         "- 把物种按亲缘关系排成一列（相邻的是近亲），用前面的物种（或前面的组）预测下一个的身体尺寸。每次预测做两遍："
         "一遍只告诉模型亲缘有多远，一遍再告诉它生活方式是怎么变的。",
         "- 每个数字回答：告诉模型生活方式以后，预测误差比不告诉时小了百分之几。例如 +12 表示误差小了 12%。"
         "长度类尺寸先扣掉体型大小的影响（只看相对体型的长短），体重和翅尖形状（HWI）不扣。",
         "- **同环境**：大环境没变，只是走动方式变了（如树上到地上）。**换环境**：陆地、水里、空中之间换了。第三层的变化都是换环境。",
         "- **三层**：第一层看单个物种；第二层把亲缘相邻、走动方式相同的一串物种合成一组；第三层再把同一环境的几组合成一组。",
         "- **空白水平**：在模拟的、生活方式对身体毫无影响的世界里，同一个格子算出来的数（第二、三层一般是负几个百分点：给模型没用的信息，它反而略差）。"
         "表 1、表 2 的数都减去了空白水平，单位是百分点。",
         "- **区间与标记**：按科重抽样的 95% 区间；* 表示整个区间在 0 以上，↓ 表示整个区间在 0 以下，没有标记表示分不清。",
         "- **简单方法**：线性模型。\"只用类别\"只知道属于哪类生活方式（每类一个平均值）；\"全部 47 项\"拿到和世界模型完全相同的生活习性变化，"
         "但只能线性地相加。", ""]
    L += ["## 运行概况", ""]
    for s, r in summ.items():
        lv_txt = "、".join("{}（{} 个模型）".format(lv, x["n_models"]) for lv, x in r["levels"].items())
        L.append(f"- {SET_ZH.get(s, s)}：成功 {r['n_ok']}/{r['n_tasks']}；{lv_txt}")
        for e in r["errors"][:5]:
            L.append(f"  - 出错：{e[0]} 折 {e[1]}：{e[2]}")
    for title, sets in (("表 1. 真实鸟类：比空白水平好多少个百分点（世界模型）", ("real_hab3", "real_hab3_ericson")),
                        ("表 2. 模拟检查：同样的读数", ("sim_v6_hab3", "sim_v5_effect"))):
        L += ["", f"## {title}", ""]
        for s in sets:
            if s not in summ or NULL not in summ:
                continue
            L += [f"**{SET_ZH[s]}**", "", "| 部位 | " + " | ".join(COL_ZH[c] for c in COLUMNS) + " |",
                  "|---|" + "---|" * len(COLUMNS)]
            for g, j in gj.items():
                cells = [fmt(*corrected(summ, s, lv, TYPES.index(t), j)) for lv, t in COLUMNS]
                L.append(f"| {g} | " + " | ".join(cells) + " |")
            L.append("")
    L += ["## 表 3. 世界模型对简单方法（同一批预测上，误差小了百分之几；没有减空白水平）", "",
          "每格三个数：世界模型 / 简单方法·只用类别 / 简单方法·全部 47 项；方括号为\"世界模型 − 全部 47 项\"的 95% 区间。", ""]
    for s in ("real_hab3", "real_hab3_ericson", "sim_v6_hab3", NULL):
        if s not in summ:
            continue
        L += [f"**{SET_ZH[s]}**", "", "| 部位 | " + " | ".join(COL_ZH[c] for c in COLUMNS) + " |",
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
                w, c, f = (100 * np.nanmean(x[k][ti, j]) for k in ("wm", "lc", "lf"))
                db = group_mean(x["wm_b"][:, ti, :], j) - group_mean(x["lf_b"][:, ti, :], j)
                lo, hi = np.nanpercentile(db, [2.5, 97.5]) * 100
                cells.append(f"{w:+.1f} / {c:+.1f} / {f:+.1f} [{lo:+.1f}, {hi:+.1f}]")
            L.append(f"| {g} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["## 表 4. 谁预测得更准：世界模型（告诉生活方式）的误差 ÷ 简单方法·全部 47 项的误差", "",
          "小于 1 表示世界模型更准；方括号为 95% 区间。", ""]
    for s in ("real_hab3", "real_hab3_ericson", "sim_v6_hab3", NULL):
        if s not in summ:
            continue
        L += [f"**{SET_ZH[s]}**", "", "| 部位 | " + " | ".join(COL_ZH[c] for c in COLUMNS) + " |",
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
                pt = np.nanmean(x["ratio"][ti, j])
                lo, hi = np.nanpercentile(group_mean(x["ratio_b"][:, ti, :], j), [2.5, 97.5])
                cells.append(f"{pt:.3f} [{lo:.3f}, {hi:.3f}]")
            L.append(f"| {g} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["## 表 5. 分层对不分层（第一层，比空白水平好多少个百分点）", ""]
    for a, b in (("real_hab3", "real_l1"), ("sim_v6_hab3", "sim_v6_l1")):
        if a not in summ or b not in summ or NULL not in summ:
            continue
        for g in ("翅形（翅长、Kipp、HWI）", "跗跖"):
            j = gj[g]
            parts = []
            for s in (b, a):
                for t in ("within", "cross"):
                    parts.append(f"{SET_ZH[s]} {COL_ZH[('L1', t)]} {fmt(*corrected(summ, s, 'L1', TYPES.index(t), j))}")
            for lv, t in (("L2", "cross"), ("L3", "cross")):
                parts.append(f"{SET_ZH[a]} {COL_ZH[(lv, t)]} {fmt(*corrected(summ, a, lv, TYPES.index(t), j))}")
            L.append(f"- {g}：" + "；".join(parts))
    return "\n".join(L) + "\n"


def write_csv(summ: dict) -> None:
    rows = []
    for s, r in summ.items():
        cols = r["cols"]
        for lv, x in r["levels"].items():
            for ti, t in enumerate(TYPES):
                for j, c in enumerate(cols):
                    lo, hi = np.nanpercentile(x["gain_b"][:, ti, j], [2.5, 97.5])
                    rows.append({"set": s, "level": lv, "type": t, "trait": c,
                                 "group": next((g for g, gc in GROUPS.items() if c in gc), ""),
                                 "gain": x["gain"][ti, j], "lo": lo, "hi": hi, "n_pred": x["n"][ti, j],
                                 "wm_same": x["wm"][ti, j], "linear_category": x["lc"][ti, j], "linear_full": x["lf"][ti, j],
                                 "rmse_ratio_wm_to_full": x["ratio"][ti, j], "n_same": x["n2"][ti, j]})
    pd.DataFrame(rows).to_csv(OUT / "traits_gain_v2.csv", index=False)


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
                print(f"{sname}：没有可评估的模型")
                continue
            pooled = pool(rs)
            pooled["missing"] = missing.get(sname, 0)
            if pooled["n_ok"]:
                save_set(sname, pooled)
            print(f"{sname}：成功 {pooled['n_ok']}/{pooled['n_tasks']}，出错 {len(pooled['errors'])}", flush=True)
            for e in pooled["errors"][:3]:
                print("  ", e)
    summ = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for sname in SETS:
            d = load_set(sname)
            if d is not None:
                summ[sname] = summarize(d, args.n_boot)
        if not summ:
            print("还没有任何模型集的结果")
            return
        (OUT / "report.md").write_text(report(summ), encoding="utf-8")
        write_csv(summ)
    print(f"写入 {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
