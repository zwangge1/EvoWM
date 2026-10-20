"""多近亲 OU 对照（设计 docs/design/多近亲OU对照_2026-10-19.md）。探索性分析；判据在运行前定下，报告里自动判定。只算第一层。

与 OU 式对照 v1 同样的窗口、种子、读数与对照（evowm.eval.ou_hist），另加用窗口里前面全部近亲的两种 OU 预测。
默认只算判定要的 4 个模型集；--family res 时为重训（预测变化量）后的同样 4 个模型集。
v1 已有的各种预测的误差和与 outputs/ou_v1/sets 逐格核对，应一致。
结果写到 outputs/ou_v2/：report.md、ou_v2.csv（重训后的模型为 report_res.md、ou_v2_res.csv）、sets/。
在 EvoWM 根目录运行：python scripts/eval_ou_v2.py --sets real_hab3 ...；python scripts/eval_ou_v2.py --report-only
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

import eval_ou_v1 as eo1  # noqa: E402,F401  （同时登记重训后的模型集 *_res）
import explore_modules_v1 as em1  # noqa: E402
from evowm.eval.ou import KEYS as V1_KEYS  # noqa: E402
from evowm.eval.ou_hist import HPARAMS, KEYS, NPOS, run_hist_task  # noqa: E402
from evowm.eval.runs import run_all  # noqa: E402
from explore_modules_v2 import SET_ZH, boot_matrix  # noqa: E402

OUT = ROOT / "outputs/ou_v2"
REF = ROOT / "outputs/ou_v1/sets"
BASE_SETS = ["real_hab3", "real_hab3_ericson", "sim_v5_null", "sim_v6_hab3"]
FAMILY_ZH = {"old": "原来的模型", "res": "重训后的模型（预测变化量）"}
ARR = KEYS + ("n",)
TYPE_ZH = ["不变", "同环境", "换环境"]
LABEL = {"copy": "照抄", "l_no": "简单方法·不告诉习性", "l_full": "简单方法·全部 47 项", "o_ch": "OU 式·只用类别",
         "o_full": "OU 式·全部 47 项", "o_same": "OU 式·同样输入", "m_no": "多近亲·不告诉习性",
         "m_full": "多近亲·全部 47 项", "d_eco": "世界模型·新读数", "a_eco": "世界模型·原读数"}
TGROUPS = {"HWI": ["m_hwi"], "翅形（翅长、Kipp、HWI）": ["m_wing", "m_kipps", "m_hwi"], "跗跖": ["m_tarsus"]}
READOUTS = (("d_eco", "新读数"), ("a_eco", "原读数"))
PRIMARY_TRAIT, PRIMARY_TYPE, PRIMARY_POS = "m_hwi", 2, (2, 3)   # 主格：第一层、换环境、HWI、位置 2 到 3
NULL_TOL = 1.01
NULL_TRAITS = ("m_hwi", "m_tarsus")
TOL_EXACT = 1e-9
POS_ALL = (1, 2, 3)


def nm(base: str, family: str) -> str:
    return base if family == "old" else f"{base}_res"


# ---------------------------------------------------------------------------
def pool(results: list[dict]) -> dict:
    ok = [r for r in results if "error" not in r]
    out = {"n_tasks": len(results), "n_ok": len(ok), "levels": {},
           "errors": [(r["key"], r["fold"], r["error"]) for r in results if "error" in r]}
    if not ok:
        return out
    out["cols"], out["families"] = ok[0]["cols"], ok[0]["families"]
    rs = [r["L1"] for r in ok]
    out["levels"]["L1"] = {"arrays": {k: sum(r[k] for r in rs) for k in ARR},
                           "params": np.stack([r["params"] for r in rs]),
                           "hparams": np.stack([r["hparams"] for r in rs]), "n_models": len(rs)}
    return out


def save_set(sname: str, pooled: dict) -> None:
    d = OUT / "sets"
    d.mkdir(parents=True, exist_ok=True)
    x = pooled["levels"]["L1"]
    z = {f"L1__{k}": v for k, v in x["arrays"].items()}
    z.update({"L1__params": x["params"], "L1__hparams": x["hparams"]})
    np.savez_compressed(d / f"{sname}.npz", **z)
    meta = {k: v for k, v in pooled.items() if k != "levels"}
    meta["levels"] = {"L1": {"n_models": x["n_models"]}}
    (d / f"{sname}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


def load_set(sname: str, d: Path) -> dict | None:
    if not (d / f"{sname}.npz").exists():
        return None
    meta = json.loads((d / f"{sname}.json").read_text(encoding="utf-8"))
    z = np.load(d / f"{sname}.npz")
    extra = ("params", "hparams")
    for lv in list(meta["levels"]):
        keys = [k for k in z.files if k.startswith(f"{lv}__")]
        meta["levels"][lv]["arrays"] = {k.split("__", 1)[1]: z[k] for k in keys if k.split("__", 1)[1] not in extra}
        for e in extra:
            if f"{lv}__{e}" in z.files:
                meta["levels"][lv][e] = z[f"{lv}__{e}"]
    return meta


def check_ref(d: dict, ref: dict | None) -> tuple[str, bool]:
    """v1 已有的各种预测（含三种 OU 式对照）与个数，应与 outputs/ou_v1/sets 的第一层一致。"""
    if ref is None or "L1" not in ref["levels"]:
        return "没有 OU 式对照 v1 的结果，跳过", True
    A, B = d["levels"]["L1"]["arrays"], ref["levels"]["L1"]["arrays"]
    worst = 0.0
    for k in V1_KEYS + ("n",):
        if k not in B or A[k].shape != B[k].shape:
            return f"第一层 {k} 形状不同", False
        worst = max(worst, float(np.max(np.abs(A[k] - B[k]) / np.maximum(np.abs(B[k]), 1e-12))))
    return f"第一层最大相对差 {worst:.1e}", worst <= TOL_EXACT


# ---------------------------------------------------------------------------
def summarize(d: dict, nb: int) -> dict:
    x = d["levels"]["L1"]
    A = {k: v.reshape(v.shape[0], 3, NPOS, v.shape[-1]) for k, v in x["arrays"].items()}   # (科, 类型, 位置, 性状)
    F = A["n"].shape[0]
    return {"n_ok": d["n_ok"], "n_tasks": d["n_tasks"], "errors": d["errors"][:20], "cols": d["cols"],
            "n_models": x["n_models"], "A": A, "M": boot_matrix(F, nb, 0, A["n"].sum(axis=(1, 2, 3)) > 0),
            "params": x.get("params"), "hparams": x.get("hparams"), "cache": {}}


def ratio(x: dict, num: str, den: str, pos: tuple = POS_ALL, boot: bool = True, cache: bool = True):
    """RMSE 之比 sqrt(Σnum / Σden)（同一批预测，pos 里的位置合在一起）：点估计 (3, C) 与按科重抽样 (nb, 3, C)。"""
    key = (num, den, tuple(pos), boot)
    if cache and key in x["cache"]:
        return x["cache"][key]
    A, M = x["A"], x["M"]
    sl = [p - 1 for p in pos]
    a, b = (A[k][:, :, sl].sum(axis=2) for k in (num, den))
    F = a.shape[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        pt = np.sqrt(a.sum(0) / b.sum(0))
        bs = None
        if boot:
            bs = np.sqrt((M @ a.reshape(F, -1)) / (M @ b.reshape(F, -1))).reshape((len(M),) + a.shape[1:])
    if cache:
        x["cache"][key] = (pt, bs)
    return pt, bs


def cell(x: dict, num: str, den: str, ti: int, js: list, pos: tuple = POS_ALL) -> tuple[float, float, float]:
    """一组性状的误差比（组内平均）与 95% 区间。"""
    pt, bs = ratio(x, num, den, pos)
    p = float(np.nanmean(pt[ti, js]))
    if not np.isfinite(p):
        return np.nan, np.nan, np.nan
    lo, hi = np.nanpercentile(np.nanmean(bs[:, ti, js], axis=-1), [2.5, 97.5])
    return p, float(lo), float(hi)


def fmt(c: tuple, ci: bool = True) -> str:
    p, lo, hi = c
    if not np.isfinite(p):
        return "·"
    return f"{p:.3f} [{lo:.3f}, {hi:.3f}]" if ci else f"{p:.2f}"


# ---------------------------------------------------------------------------
def null_check(summ: dict) -> tuple[bool, float] | None:
    """空白模拟：多近亲·全部 47 项 ÷ 多近亲·不告诉习性，第一层三类步、HWI 与相对跗跖、位置 2 到 3，最大值不超过 1.01。"""
    s = summ.get("sim_v5_null")
    if s is None:
        return None
    pt, _ = ratio(s, "m_full", "m_no", PRIMARY_POS)
    w = max(float(pt[ti, s["cols"].index(c)]) for ti in range(3) for c in NULL_TRAITS)
    return bool(np.isfinite(w) and w <= NULL_TOL), w


def judge(summ: dict, key: str) -> dict:
    need = ("real_hab3", "real_hab3_ericson", "sim_v6_hab3")
    miss = [s for s in need if s not in summ]
    if miss:
        return {"missing": miss}
    out = {}
    for name, s in (("c1", "real_hab3"), ("c2", "real_hab3_ericson"), ("c3", "sim_v6_hab3")):
        j = summ[s]["cols"].index(PRIMARY_TRAIT)
        out[name] = cell(summ[s], key, "m_full", PRIMARY_TYPE, [j], PRIMARY_POS)
    ok = [out["c1"][2] < 1, out["c2"][0] < 1, out["c3"][2] >= 1]
    out["ok"] = ok
    out["verdict"] = "甲" if not ok[0] else ("乙" if all(ok) else "丙")
    return out


VERDICT_ZH = {"甲": "结论甲：多看近亲能解释，世界模型在主格上不比用同样近亲的 OU 式预测显著更准。",
              "乙": "结论乙：还有多看近亲解释不了的优势，成为候选。",
              "丙": "结论丙：介于两者之间（见上表哪一条不成立），如实描述，另行讨论。"}


def criteria_section(summ: dict) -> list:
    L = ["## 判据（运行前定下）", "",
         "主格：第一层、换环境、HWI、位置 2 到 3。误差比 = 世界模型（告诉生活方式）的误差 ÷ 多近亲·全部 47 项的误差，"
         "小于 1 表示世界模型更准；方括号为按科重抽样的 95% 区间。", ""]
    nc = null_check(summ)
    if nc is None:
        return L + ["缺少空白模拟的结果，暂不能判定。", ""]
    L += [f"检查（空白模拟，多近亲·全部 47 项 ÷ 多近亲·不告诉习性，第一层三类步、HWI 与相对跗跖、位置 2 到 3，"
          f"最大值 ≤ 1.01）：{nc[1]:.3f}，{'满足' if nc[0] else '不满足，不判定，只描述'}。", ""]
    if not nc[0]:
        return L
    for key, zh in READOUTS:
        r = judge(summ, key)
        L += [f"**世界模型·{zh}**" + ("（主要）" if key == "d_eco" else "（次要）"), ""]
        if "missing" in r:
            L += [f"缺少 {'、'.join(SET_ZH.get(s, s) for s in r['missing'])}，暂不能判定。", ""]
            continue
        rows = (("c1", "真实鸟类（Hackett 树）", "区间上限 < 1"), ("c2", "真实鸟类（Ericson 树）", "误差比 < 1"),
                ("c3", "v6 模拟", "区间上限 ≥ 1"))
        L += ["| 条件 | 数据 | 误差比 [95% 区间] | 要求 | 成立 |", "|---|---|---|---|---|"]
        for i, (c, what, req) in enumerate(rows):
            L.append(f"| {i + 1} | {what} | {fmt(r[c])} | {req} | {'是' if r['ok'][i] else '否'} |")
        L += ["", "第 1 条不成立为结论甲；三条都成立为结论乙；其余为结论丙。", "", f"**{VERDICT_ZH[r['verdict']]}**", ""]
    return L


def report(summ: dict, checks: dict, family: str) -> str:
    cols = next(iter(summ.values()))["cols"]
    gj = {g: [cols.index(c) for c in cs if c in cols] for g, cs in TGROUPS.items()}
    L = [f"# 多近亲 OU 对照（{FAMILY_ZH[family]}）", "",
         "设计 `docs/design/多近亲OU对照_2026-10-19.md`；脚本 `scripts/eval_ou_v2.py`。探索性分析，只有下面的判据按事先定的规则判定。",
         "只算第一层。", "",
         "## 先说数字是什么意思", "",
         "- 与 OU 式对照 v1 同一批预测（前后两边都有实测值）。**位置**：窗口里第几个预测，位置 p 时世界模型看得到前面 p 个物种。",
         "- **多近亲·不告诉习性**：OU 模型下，已知窗口里前面这几个近亲的实测时，下一个物种的条件期望（近的权重大）。",
         "- **多近亲·全部 47 项**：再按同样的权重，把各近亲与下一个物种的习性差平均，线性地加上（形式同 OU 式·全部 47 项）。",
         "  位置 1 时它就是 OU 式·全部 47 项；位置 2、3 时它与世界模型看到的近亲相同。",
         "- **误差是照抄的几倍**：1.00 表示和照抄一样，小于 1 表示比照抄准。**误差比**：两种预测的均方根误差之比。",
         "- 长度类尺寸先扣掉体型大小的影响，体重与 HWI 不扣。区间为按科重抽样 95% 区间。", ""]
    L += ["## 运行概况", ""]
    for b, r in summ.items():
        L.append(f"- {SET_ZH.get(b, b)}：成功 {r['n_ok']}/{r['n_tasks']}（{r['n_models']} 个模型）；"
                 f"与 OU 式对照 v1 核对：{checks.get(b, ('·', True))[0]}")
        for e in r["errors"][:5]:
            L.append(f"  - 出错：{e[0]} 折 {e[1]}：{e[2]}")
    bad = [b for b, (_, ok) in checks.items() if not ok]
    if bad:
        L += ["", f"**注意：{'、'.join(SET_ZH.get(b, b) for b in bad)} 的核对不一致，这些模型集的数字先不解读。**"]
    L += [""] + criteria_section(summ)

    rows1 = ("l_no", "m_no", "o_ch", "o_full", "m_full", "d_eco", "a_eco")
    heads = [f"{TYPE_ZH[t]}·位置 {p}" for t in (2, 0) for p in POS_ALL]
    L += ["## 表 1. 第一层：各种预测的误差是照抄的几倍（按位置）", ""]
    for b in ("real_hab3", "sim_v6_hab3"):
        if b not in summ:
            continue
        x = summ[b]
        L += [f"**{SET_ZH.get(b, b)}**", ""]
        for g in TGROUPS:
            L += [f"{g}", "", "| 预测 | " + " | ".join(heads) + " |", "|---|" + "---|" * len(heads)]
            for k in rows1:
                cells = [fmt(cell(x, k, "copy", t, gj[g], (p,)), ci=False) for t in (2, 0) for p in POS_ALL]
                L.append(f"| {LABEL[k]} | " + " | ".join(cells) + " |")
            L.append("")

    L += ["## 表 2. 第一层：世界模型·新读数 ÷ 多近亲·全部 47 项", "",
          "小于 1 表示世界模型更准。每格：误差比 [95% 区间]；括号里为 ÷ OU 式·全部 47 项（只用前一个近亲）。", ""]
    for b in ("real_hab3", "real_hab3_ericson", "sim_v6_hab3"):
        if b not in summ:
            continue
        x = summ[b]
        L += [f"**{SET_ZH.get(b, b)}**", ""]
        for g in ("HWI", "翅形（翅长、Kipp、HWI）"):
            L += [f"{g}", "", "| 位置 | " + " | ".join(TYPE_ZH) + " |", "|---|---|---|---|"]
            for pos, pzh in ((POS_ALL, "全部"), ((1,), "1"), ((2,), "2"), ((3,), "3"), ((2, 3), "2 到 3")):
                cells = []
                for ti in range(3):
                    o = cell(x, "d_eco", "o_full", ti, gj[g], pos)[0]
                    extra = f"（{o:.3f}）" if np.isfinite(o) else ""
                    cells.append(fmt(cell(x, "d_eco", "m_full", ti, gj[g], pos)) + extra)
                L.append(f"| {pzh} | " + " | ".join(cells) + " |")
            L.append("")

    L += ["## 表 3. 多近亲预测的拟合参数（各模型的中位数）", "",
          "回拉半衰期 = ln2/α（百万年，按两物种的亲缘距离计）；η = 不随树走的变异占的比例（测量误差、种内差异）。", ""]
    for b in ("real_hab3", "sim_v6_hab3"):
        x = summ.get(b)
        if x is None or x.get("hparams") is None:
            continue
        P = np.nanmedian(x["hparams"], axis=0)
        L += [f"**{SET_ZH.get(b, b)}**", "", "| 尺寸 | 回拉半衰期 | η |", "|---|---|---|"]
        for c, zh in (("m_hwi", "HWI"), ("m_kipps", "Kipp"), ("m_wing", "翅长"), ("m_tarsus", "跗跖"), ("m_mass", "体重"),
                      ("m_beak_culmen", "喙长")):
            if c in cols:
                a, e = P[cols.index(c)]
                L.append(f"| {zh} | {np.log(2) / a:.1f} | {e:.3f} |")
        L.append("")

    L += ["## 怎么看（设计里先写下的）", "",
          "1. 结论甲：加上 v1 的结果，信息相同时世界模型的单步预测与线性的 OU 式预测一样准，目前没有读出 OU 之外的东西。",
          "2. 结论乙：剩下的优势成为候选，下一步先看集中在哪些科，再与用整棵树的标准方法比。",
          "3. 结论丙：如实描述，另行讨论。",
          "4. 空白模拟的检查不满足时不判定。原读数、重训后的模型各自判定，分别报告，不事后挑选。"]
    return "\n".join(L) + "\n"


def write_csv(summ: dict, path: Path) -> None:
    rows = []
    poss = ((POS_ALL, "all"), ((1,), "1"), ((2,), "2"), ((3,), "3"), ((2, 3), "2-3"))
    for b, x in summ.items():
        cols = x["cols"]
        for pos, pname in poss:
            n = x["A"]["n"][:, :, [p - 1 for p in pos]].sum(axis=(0, 2))
            vc = {k: ratio(x, k, "copy", pos, boot=False, cache=False)[0] for k in KEYS}
            vs = {(w, k): ratio(x, w, k, pos, cache=False) for w, _ in READOUTS for k in ("o_full", "m_no", "m_full")}
            for ti in range(3):
                for j, c in enumerate(cols):
                    row = {"set": b, "level": "L1", "type": ["none", "within", "cross"][ti], "position": pname,
                           "trait": c, "n_pred": n[ti, j]}
                    for k in KEYS:
                        row[f"rmse_ratio_{k}_to_copy"] = vc[k][ti, j]
                    for (w, k), (pt, bs) in vs.items():
                        lo, hi = np.nanpercentile(bs[:, ti, j], [2.5, 97.5])
                        row.update({f"{w}_vs_{k}": pt[ti, j], f"{w}_vs_{k}_lo": lo, f"{w}_vs_{k}_hi": hi})
                    rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=("old", "res"), default="old")
    ap.add_argument("--sets", nargs="*", default=[])
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--n-orders", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    fam = args.family
    names = {nm(b, fam): b for b in BASE_SETS}
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.report_only:
        unknown = [s for s in args.sets if s not in names]
        if unknown:
            sys.exit(f"未知的模型集：{unknown}（--family {fam} 可选 {list(names)}）")
        tasks, missing = em1.build_tasks(args.sets, args.n_orders)
        print(f"{len(tasks)} 个（模型，折）任务；缺失：{missing}", flush=True)
        results = run_all(tasks, args.workers, fn=run_hist_task) if tasks else []
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
        for sname, b in names.items():
            d = load_set(sname, OUT / "sets")
            if d is None:
                continue
            summ[b] = summarize(d, args.n_boot)
            checks[b] = check_ref(d, load_set(sname, REF))
        if not summ:
            print("还没有任何模型集的结果")
            return
        suffix = "" if fam == "old" else "_res"
        (OUT / f"report{suffix}.md").write_text(report(summ, checks, fam), encoding="utf-8")
        write_csv(summ, OUT / f"ou_v2{suffix}.csv")
    for b, (c, ok) in checks.items():
        print(f"核对 {b}：{c}{'' if ok else '（不一致）'}")
    print(f"写入 {OUT / f'report{suffix}.md'}")


if __name__ == "__main__":
    main()
