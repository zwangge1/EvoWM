"""OU 式对照（设计 docs/design/OU式对照_2026-10-19.md）。探索性分析；判据在运行前定下，报告里自动判定。

与读数诊断 v1 同样的 7 个模型集、窗口、种子与读数（evowm.eval.ou），每个预测另算三种 OU 式对照，并按窗口位置拆分。
--family res：重训（预测变化量）后的 7 个模型集（检查点 outputs/hjepa/ckpts/residual_v1/，与 scripts/eval_residual_v1.py 相同）。
照抄、已有简单方法与世界模型读数与读数诊断（重训后的模型则与 outputs/residual_v1）逐格核对，应一致。
结果写到 outputs/ou_v1/：report.md、ou_v1.csv（重训后的模型为 report_res.md、ou_v1_res.csv）、sets/。
在 EvoWM 根目录运行：python scripts/eval_ou_v1.py --sets real_hab3 ...；python scripts/eval_ou_v1.py --report-only
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

import explore_modules_v1 as em1  # noqa: E402
from evowm.eval.ou import KEYS, NPOS, OU_KEYS, PARAMS, run_ou_task  # noqa: E402
from evowm.eval.readout import KEYS as READOUT_KEYS  # noqa: E402
from evowm.eval.runs import run_all  # noqa: E402
from explore_modules_v2 import SET_ZH, boot_matrix  # noqa: E402

OUT = ROOT / "outputs/ou_v1"
BASE_SETS = ["real_hab3", "real_hab3_ericson", "real_l1", "sim_v5_effect", "sim_v5_null", "sim_v6_hab3", "sim_v6_l1"]
CK = "outputs/hjepa/ckpts/residual_v1"
RES_ROOT = {"real_hab3": f"{CK}/real_hab3", "real_hab3_ericson": f"{CK}/real_hab3", "real_l1": f"{CK}/real_l1",
            "sim_v5_effect": f"{CK}/sim_v5", "sim_v5_null": f"{CK}/sim_v5", "sim_v6_hab3": f"{CK}/sim_v6_hab3",
            "sim_v6_l1": f"{CK}/sim_v6_l1"}          # 与 scripts/eval_residual_v1.py 的 RES 相同
for _b, _r in RES_ROOT.items():
    em1.SETS[f"{_b}_res"] = dict(em1.SETS[_b], root=_r)
REF = {"old": ROOT / "outputs/diag_readout_v1/sets", "res": ROOT / "outputs/residual_v1/sets"}
FAMILY_ZH = {"old": "原来的模型", "res": "重训后的模型（预测变化量）"}
ARR = KEYS + ("n",)
TYPE_ZH = ["不变", "同环境", "换环境"]
LABEL = {"copy": "照抄", "l_no": "简单方法·不告诉习性", "l_ch": "简单方法·只用类别", "l_full": "简单方法·全部 47 项",
         "o_ch": "OU 式·只用类别", "o_full": "OU 式·全部 47 项", "o_same": "OU 式·同样输入",
         "d_eco": "世界模型·新读数", "a_eco": "世界模型·原读数"}
SHORT = {"o_ch": "只用类别", "o_full": "全部 47 项", "o_same": "同样输入"}
TGROUPS = {"HWI": ["m_hwi"], "翅形（翅长、Kipp、HWI）": ["m_wing", "m_kipps", "m_hwi"], "跗跖": ["m_tarsus"],
           "喙": ["m_beak_culmen", "m_beak_nares", "m_beak_width", "m_beak_depth"], "体重": ["m_mass"]}
READOUTS = (("d_eco", "新读数"), ("a_eco", "原读数"))
PRIMARY_TRAIT, PRIMARY_TYPE = "m_hwi", 2         # 主格：第一层、换环境、HWI
NULL_TOL = 1.01
NULL_TRAITS = (("m_hwi", "HWI"), ("m_tarsus", "相对跗跖"))
TOL_EXACT = 1e-9


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
    for lv in ("L1", "L2", "L3"):
        rs = [r[lv] for r in ok if lv in r]
        if rs:
            out["levels"][lv] = {"arrays": {k: sum(r[k] for r in rs) for k in ARR},
                                 "params": np.stack([r["params"] for r in rs]), "n_models": len(rs)}
    return out


def save_set(sname: str, pooled: dict) -> None:
    d = OUT / "sets"
    d.mkdir(parents=True, exist_ok=True)
    z = {f"{lv}__{k}": v for lv, x in pooled["levels"].items() for k, v in x["arrays"].items()}
    z.update({f"{lv}__params": x["params"] for lv, x in pooled["levels"].items()})
    np.savez_compressed(d / f"{sname}.npz", **z)
    meta = {k: v for k, v in pooled.items() if k != "levels"}
    meta["levels"] = {lv: {"n_models": x["n_models"]} for lv, x in pooled["levels"].items()}
    (d / f"{sname}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


def load_set(sname: str, d: Path) -> dict | None:
    if not (d / f"{sname}.npz").exists():
        return None
    meta = json.loads((d / f"{sname}.json").read_text(encoding="utf-8"))
    z = np.load(d / f"{sname}.npz")
    for lv in list(meta["levels"]):
        keys = [k for k in z.files if k.startswith(f"{lv}__")]
        meta["levels"][lv]["arrays"] = {k.split("__", 1)[1]: z[k] for k in keys if not k.endswith("__params")}
        if f"{lv}__params" in z.files:
            meta["levels"][lv]["params"] = z[f"{lv}__params"]
    return meta


def check_ref(d: dict, ref: dict | None) -> tuple[str, bool]:
    """照抄、已有简单方法、世界模型读数与个数，对位置求和后应与读数诊断（或重训评估）的结果一致。"""
    if ref is None:
        return "没有可核对的结果，跳过", True
    parts, good = [], True
    for lv, x in d["levels"].items():
        y = ref["levels"].get(lv)
        if y is None:
            continue
        worst = 0.0
        for k in READOUT_KEYS + ("n",):
            a = x["arrays"][k]
            a = a.reshape(a.shape[0], 3, NPOS, a.shape[-1]).sum(axis=2)
            b = y["arrays"][k]
            if a.shape != b.shape:
                worst = np.inf
                break
            worst = max(worst, float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12))))
        good &= worst <= TOL_EXACT
        parts.append(f"{lv} 最大相对差 {worst:.1e}")
    return ("；".join(parts) if parts else "没有可比的层"), good


# ---------------------------------------------------------------------------
def summarize(d: dict, nb: int) -> dict:
    res = {"n_ok": d["n_ok"], "n_tasks": d["n_tasks"], "errors": d["errors"][:20], "cols": d["cols"], "levels": {}}
    for lv, x in d["levels"].items():
        A = {k: v.reshape(v.shape[0], 3, NPOS, v.shape[-1]) for k, v in x["arrays"].items()}   # (科, 类型, 位置, 性状)
        F = A["n"].shape[0]
        res["levels"][lv] = {"A": A, "M": boot_matrix(F, nb, 0, A["n"].sum(axis=(1, 2, 3)) > 0),
                             "params": x.get("params"), "n_models": x["n_models"], "cache": {}}
    return res


def ratio(x: dict, num: str, den: str, pos: int | None = None, boot: bool = True, cache: bool = True):
    """RMSE 之比 sqrt(Σnum / Σden)（同一批预测）：点估计 (3, C) 与按科重抽样 (nb, 3, C)（boot=False 时为 None）；
    pos 为 None 时为全部位置。"""
    key = (num, den, pos, boot)
    if cache and key in x["cache"]:
        return x["cache"][key]
    A, M = x["A"], x["M"]
    a, b = (A[k].sum(axis=2) if pos is None else A[k][:, :, pos - 1] for k in (num, den))
    F = a.shape[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        pt = np.sqrt(a.sum(0) / b.sum(0))
        bs = None
        if boot:
            bs = np.sqrt((M @ a.reshape(F, -1)) / (M @ b.reshape(F, -1))).reshape((len(M),) + a.shape[1:])
    if cache:
        x["cache"][key] = (pt, bs)
    return pt, bs


def cell(x: dict, num: str, den: str, ti: int, js: list, pos: int | None = None) -> tuple[float, float, float]:
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


def best_ou(x: dict, ti: int, j: int, pos: int | None, valid: dict) -> str | None:
    """该格误差最小的 OU 式对照（只在通过空白检查的对照里选；同一批预测，比误差平方和即可）。"""
    best, val = None, np.inf
    for k in OU_KEYS:
        if not valid.get(k, False):
            continue
        a = x["A"][k].sum(axis=2) if pos is None else x["A"][k][:, :, pos - 1]
        v = float(a.sum(0)[ti, j])
        if np.isfinite(v) and v < val:
            best, val = k, v
    return best


# ---------------------------------------------------------------------------
def null_check(summ: dict) -> dict | None:
    """空白模拟：每种 OU 式对照的误差 ÷ 不告诉习性的误差，第一层三类步、HWI 与相对跗跖，最大值不超过 1.01。"""
    s = summ.get("sim_v5_null")
    if s is None or "L1" not in s["levels"]:
        return None
    x, cols = s["levels"]["L1"], s["cols"]
    out = {}
    for k in OU_KEYS:
        pt, _ = ratio(x, k, "l_no")
        w = max(float(pt[ti, cols.index(c)]) for ti in range(3) for c, _ in NULL_TRAITS)
        out[k] = (bool(np.isfinite(w) and w <= NULL_TOL), w)
    return out


def judge(summ: dict, valid: dict, key: str) -> dict:
    """判据：第一层、换环境、HWI；世界模型（读数 key）÷ 该格最准的 OU 式对照。"""
    need = ("real_hab3", "real_hab3_ericson", "sim_v6_hab3")
    miss = [s for s in need if s not in summ or "L1" not in summ[s]["levels"]]
    if miss:
        return {"missing": miss}
    out = {}
    for name, s, pos in (("c1", "real_hab3", None), ("c2", "real_hab3", 1), ("c3", "real_hab3_ericson", None),
                         ("c4", "sim_v6_hab3", None)):
        x = summ[s]["levels"]["L1"]
        j = summ[s]["cols"].index(PRIMARY_TRAIT)
        k = best_ou(x, PRIMARY_TYPE, j, pos, valid)
        if k is None:
            return {"missing": ["可用的 OU 式对照"]}
        out[name] = (k,) + cell(x, key, k, PRIMARY_TYPE, [j], pos)
    ok = [out["c1"][3] < 1, out["c2"][3] < 1, out["c3"][1] < 1, out["c4"][3] >= 1]
    out["ok"] = ok
    out["verdict"] = "甲" if all(ok) else ("乙" if out["c1"][3] >= 1 else "丙")
    return out


VERDICT_ZH = {"甲": "结论甲：翅形（HWI）在换环境时的优势，在最准的 OU 式对照面前仍在，成为\"OU 之外\"的候选。",
              "乙": "结论乙：OU 式对照能解释，世界模型在主格上不再显著更准。",
              "丙": "结论丙：介于两者之间（见上表哪一条不成立），如实描述，下一步另行讨论。"}


def criteria_section(summ: dict) -> list:
    L = ["## 判据（运行前定下）", "",
         "主格：第一层、换环境、HWI。误差比 = 世界模型（告诉生活方式）的误差 ÷ 该格最准的一种 OU 式对照的误差，"
         "小于 1 表示世界模型更准；方括号为按科重抽样的 95% 区间。", ""]
    nc = null_check(summ)
    if nc is None:
        return L + ["缺少空白模拟的结果，暂不能判定。", ""]
    L += ["检查（空白模拟，第一层三类步、HWI 与相对跗跖，误差 ÷ 不告诉习性，最大值 ≤ 1.01）：" +
          "；".join(f"{LABEL[k]} {w:.3f}{'' if ok else ' 不满足，判定时不用'}" for k, (ok, w) in nc.items()), ""]
    valid = {k: ok for k, (ok, _) in nc.items()}
    for key, zh in READOUTS:
        r = judge(summ, valid, key)
        L += [f"**世界模型·{zh}**" + ("（主要）" if key == "d_eco" else "（次要）"), ""]
        if "missing" in r:
            L += [f"缺少 {'、'.join(r['missing'])}，暂不能判定。", ""]
            continue
        rows = (("c1", "真实鸟类（Hackett 树），全部位置", "区间上限 < 1"),
                ("c2", "真实鸟类（Hackett 树），只看位置 1", "区间上限 < 1"),
                ("c3", "真实鸟类（Ericson 树），全部位置", "误差比 < 1"),
                ("c4", "v6 模拟，全部位置", "区间上限 ≥ 1"))
        L += ["| 条件 | 数据 | 最准的 OU 式对照 | 误差比 [95% 区间] | 要求 | 成立 |", "|---|---|---|---|---|---|"]
        for i, (c, what, req) in enumerate(rows):
            k, p, lo, hi = r[c]
            L.append(f"| {i + 1} | {what} | {SHORT[k]} | {p:.3f} [{lo:.3f}, {hi:.3f}] | {req} | "
                     f"{'是' if r['ok'][i] else '否'} |")
        L += ["", f"**{VERDICT_ZH[r['verdict']]}**", ""]
    return L


# ---------------------------------------------------------------------------
def report(summ: dict, checks: dict, family: str) -> str:
    any_set = next(iter(summ.values()))
    cols = any_set["cols"]
    gj = {g: [cols.index(c) for c in cs if c in cols] for g, cs in TGROUPS.items()}
    L = [f"# OU 式对照（{FAMILY_ZH[family]}）", "",
         "设计 `docs/design/OU式对照_2026-10-19.md`；脚本 `scripts/eval_ou_v1.py`。探索性分析，只有下面的判据按事先定的规则判定。", "",
         "## 先说数字是什么意思", "",
         "- 用前一个物种（或前一组）预测下一个的身体尺寸，所有方法用同一批预测（前后两边都有实测值），与读数诊断相同。",
         "- **简单方法**（原来就有）：生活方式的修正是一个不随分开时间变的数。",
         "- **OU 式·只用类别**：每类生活方式有自己的最优值，换了类别以后，向新最优值靠近的程度随分开时间变大。",
         "- **OU 式·全部 47 项**：与世界模型相同的动作（47 项习性差、3 个习性组标记、2 列距离），每一项习性的影响都可以随时间变。",
         "- **OU 式·同样输入**：再加前一个物种的全部实测。位置 1 的预测上，它与世界模型拿到的信息完全一样，只是线性的。",
         "- **位置**：窗口里第几个预测。位置 1 时世界模型只看到前一个物种；位置 2、3 时还看到更早的近亲。对照都只用前一个。",
         "- **误差是照抄的几倍**：1.00 表示和照抄一样，小于 1 表示比照抄准。**误差比**：两种预测的均方根误差之比。",
         "- 长度类尺寸先扣掉体型大小的影响，体重与 HWI 不扣。区间为按科重抽样 95% 区间。", ""]
    L += ["## 运行概况", ""]
    for b, r in summ.items():
        lv_txt = "、".join(f"{lv}（{x['n_models']} 个模型）" for lv, x in r["levels"].items())
        L.append(f"- {SET_ZH.get(b, b)}：成功 {r['n_ok']}/{r['n_tasks']}；{lv_txt}；核对：{checks.get(b, ('·', True))[0]}")
        for e in r["errors"][:5]:
            L.append(f"  - 出错：{e[0]} 折 {e[1]}：{e[2]}")
    bad = [b for b, (_, ok) in checks.items() if not ok]
    if bad:
        L += ["", f"**注意：{'、'.join(SET_ZH.get(b, b) for b in bad)} 的核对不一致，这些模型集的数字先不解读。**"]
    L += [""] + criteria_section(summ)

    rows1 = ("l_no", "l_ch", "l_full", "o_ch", "o_full", "o_same", "d_eco", "a_eco")
    L += ["## 表 1. 第一层：各种预测的误差是照抄的几倍（全部位置）", ""]
    for b, groups in (("real_hab3", ("HWI", "翅形（翅长、Kipp、HWI）", "跗跖", "体重")), ("sim_v6_hab3", ("HWI", "跗跖")),
                      ("sim_v5_null", ("HWI", "跗跖"))):
        if b not in summ or "L1" not in summ[b]["levels"]:
            continue
        x = summ[b]["levels"]["L1"]
        L += [f"**{SET_ZH.get(b, b)}**", ""]
        for g in groups:
            L += [f"{g}", "", "| 预测 | " + " | ".join(TYPE_ZH) + " |", "|---|---|---|---|"]
            for k in rows1:
                cells = [fmt(cell(x, k, "copy", ti, gj[g]), ci=False) for ti in range(3)]
                L.append(f"| {LABEL[k]} | " + " | ".join(cells) + " |")
            L.append("")

    L += ["## 表 2. 第一层：世界模型·新读数 ÷ 各种对照（全部位置）", "",
          "小于 1 表示世界模型更准。每格：新读数的误差比 [95% 区间]；分号后为原读数的误差比。", ""]
    for b in ("real_hab3", "real_hab3_ericson", "sim_v6_hab3"):
        if b not in summ or "L1" not in summ[b]["levels"]:
            continue
        x = summ[b]["levels"]["L1"]
        L += [f"**{SET_ZH.get(b, b)}**", ""]
        for g in ("HWI", "翅形（翅长、Kipp、HWI）"):
            L += [f"{g}", "", "| 对照 | " + " | ".join(TYPE_ZH) + " |", "|---|---|---|---|"]
            for k in ("l_full", "o_ch", "o_full", "o_same"):
                cells = []
                for ti in range(3):
                    old = cell(x, "a_eco", k, ti, gj[g])[0]
                    cells.append(fmt(cell(x, "d_eco", k, ti, gj[g])) + (f"；原 {old:.3f}" if np.isfinite(old) else ""))
                L.append(f"| {LABEL[k]} | " + " | ".join(cells) + " |")
            L.append("")

    nc = null_check(summ) or {}
    valid = {k: ok for k, (ok, _) in nc.items()} if nc else {k: True for k in OU_KEYS}
    L += ["## 表 3. 按位置：世界模型·新读数 ÷ 该格最准的 OU 式对照（第一层）", "",
          "位置 1：两边拿到的信息一样。括号里为该格最准的对照（翅形三项各自选，再平均）。", ""]
    for b in ("real_hab3", "sim_v6_hab3"):
        if b not in summ or "L1" not in summ[b]["levels"]:
            continue
        x = summ[b]["levels"]["L1"]
        L += [f"**{SET_ZH.get(b, b)}**", ""]
        for g in ("HWI", "翅形（翅长、Kipp、HWI）"):
            L += [f"{g}", "", "| 位置 | " + " | ".join(TYPE_ZH) + " |", "|---|---|---|---|"]
            for pos, pzh in ((None, "全部"), (1, "1"), (2, "2"), (3, "3")):
                cells = []
                for ti in range(3):
                    # 每项尺寸各自选该格最准的对照，组内再平均
                    pts, bss = [], []
                    names = set()
                    for j in gj[g]:
                        k = best_ou(x, ti, j, pos, valid)
                        if k is None:
                            continue
                        names.add(SHORT[k])
                        pt, bs = ratio(x, "d_eco", k, pos)
                        pts.append(pt[ti, j])
                        bss.append(bs[:, ti, j])
                    if not pts or not np.isfinite(np.nanmean(pts)):
                        cells.append("·")
                        continue
                    lo, hi = np.nanpercentile(np.nanmean(np.stack(bss, axis=1), axis=1), [2.5, 97.5])
                    cells.append(f"{np.nanmean(pts):.3f} [{lo:.3f}, {hi:.3f}]（{'、'.join(sorted(names))}）")
                L.append(f"| {pzh} | " + " | ".join(cells) + " |")
            L.append("")

    cols4 = (("L2", 1, "第二层·同环境"), ("L2", 2, "第二层·换环境"), ("L3", 2, "第三层·换环境"))
    L += ["## 表 4. 第二、三层：世界模型·新读数 ÷ OU 式·同样输入（全部位置，只描述）", "",
          "每格：误差比 [95% 区间]；括号里为 ÷ 简单方法·全部 47 项（读数诊断表 4 的数）。", ""]
    for b in ("real_hab3", "sim_v6_hab3"):
        if b not in summ:
            continue
        L += [f"**{SET_ZH.get(b, b)}**", "", "| 部位 | " + " | ".join(c[2] for c in cols4) + " |", "|---|---|---|---|"]
        for g in ("翅形（翅长、Kipp、HWI）", "跗跖", "喙", "体重"):
            cells = []
            for lv, ti, _ in cols4:
                x = summ[b]["levels"].get(lv)
                if x is None:
                    cells.append("·")
                    continue
                old = cell(x, "d_eco", "l_full", ti, gj[g])[0]
                cells.append(fmt(cell(x, "d_eco", "o_same", ti, gj[g])) + (f"（{old:.3f}）" if np.isfinite(old) else ""))
            L.append(f"| {g} | " + " | ".join(cells) + " |")
        L.append("")

    L += ["## 表 5. OU 式·只用类别的拟合参数（第一层，各模型的中位数）", "",
          "回拉半衰期 = ln2/α′（与本类最优值的偏离减半所需的分开距离，百万年）；适应半衰期 = ln2/β（换类别以后，"
          "向新最优值走完一半所需的分开距离，百万年）；κ 为最终走到类别平均之差的几倍。分开距离是两物种到共同祖先的距离之和。", ""]
    for b in ("real_hab3", "sim_v6_hab3"):
        x = summ.get(b, {}).get("levels", {}).get("L1")
        if x is None or x.get("params") is None:
            continue
        P = np.nanmedian(x["params"], axis=0)                     # (C, 3)
        L += [f"**{SET_ZH.get(b, b)}**", "", "| 尺寸 | 回拉半衰期 | κ | 适应半衰期 |", "|---|---|---|---|"]
        for c, zh in (("m_hwi", "HWI"), ("m_kipps", "Kipp"), ("m_wing", "翅长"), ("m_tarsus", "跗跖"), ("m_mass", "体重"),
                      ("m_beak_culmen", "喙长")):
            if c not in cols:
                continue
            a, k, be = P[cols.index(c)]
            L.append(f"| {zh} | {np.log(2) / a:.1f} | {k:.2f} | {np.log(2) / be:.1f} |")
        L.append("")

    L += ["## 怎么看（设计里先写下的）", "",
          "1. 结论甲：翅形在换环境时的优势成为\"OU 之外\"的候选，下一步用三关去筛（先看集中在哪些科）。",
          "2. 结论乙：v2 的优势来自按时间缩放，世界模型目前没有读出 OU 之外的东西，论文定位另行讨论。",
          "3. 结论丙：例如只有位置 2、3 显著（优势来自多看了几个近亲），或 v6 里世界模型也显著更好（比较偏向世界模型）；"
          "如实描述，另行讨论。",
          "4. 空白模拟里不满足检查的对照，判定时不用。原读数、重训后的模型各自判定，分别报告，不事后挑选。"]
    return "\n".join(L) + "\n"


def write_csv(summ: dict, path: Path) -> None:
    rows = []
    for b, r in summ.items():
        cols = r["cols"]
        for lv, x in r["levels"].items():
            for pos in (None, 1, 2, 3):
                n = x["A"]["n"].sum(axis=(0, 2)) if pos is None else x["A"]["n"][:, :, pos - 1].sum(0)
                vc = {k: ratio(x, k, "copy", pos, boot=False, cache=False)[0] for k in KEYS}
                vs = {(w, k): ratio(x, w, k, pos, cache=False) for w, _ in READOUTS for k in ("l_full",) + OU_KEYS}
                for ti in range(3):
                    for j, c in enumerate(cols):
                        row = {"set": b, "level": lv, "type": ["none", "within", "cross"][ti],
                               "position": "all" if pos is None else pos, "trait": c, "n_pred": n[ti, j]}
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
        results = run_all(tasks, args.workers, fn=run_ou_task) if tasks else []
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
            checks[b] = check_ref(d, load_set(sname, REF[fam]))
        if not summ:
            print("还没有任何模型集的结果")
            return
        suffix = "" if fam == "old" else "_res"
        (OUT / f"report{suffix}.md").write_text(report(summ, checks, fam), encoding="utf-8")
        write_csv(summ, OUT / f"ou_v1{suffix}.csv")
    for b, (c, ok) in checks.items():
        print(f"核对 {b}：{c}{'' if ok else '（不一致）'}")
    print(f"写入 {OUT / f'report{suffix}.md'}")


if __name__ == "__main__":
    main()
