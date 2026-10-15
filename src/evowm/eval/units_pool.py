"""按习性分层模型的多次运行合并（开发检查、第 0 阶段、真实数据共用）。输入是 evowm.eval.runs.run_unit_task 的输出。

读数与 evowm.eval.units 相同，这里只做合并与按簇自助重抽样（evowm.eval.pooled）。
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from .pooled import boot_gain, boot_gain_diff, boot_mean

GENERALIST = 2       # evowm.data.units.LOCOMOTOR 里"泛化"的下标
PRIMARY = {"all": "全部第 2 层步", "no_generalist": "不涉及泛化的第 2 层步"}


def g2_ok(null: dict, effect: dict) -> bool:
    """G2 的写法：零模型区间含 0，或小于效应的 10%。"""
    lo, hi = null["ci95"]
    return bool((lo <= 0 <= hi) or (null["gain"] < 0.1 * effect["gain"]))


def _cat(grp, get):
    return np.concatenate([get(r) for r in grp])


def common_modes(lv: list[dict]) -> list[str]:
    """所有运行都有原型的类别（某折缺某类的训练单位时，该类不进合并）。"""
    return [m for m in lv[0]["E2"]["rel"] if all(m in v["E2"]["rel"] for v in lv)]


def summarize_l1(grp, n_boot: int) -> dict:
    tr = _cat(grp, lambda r: r["L1"]["transition"])
    fam = _cat(grp, lambda r: r["L1"]["family"])
    re1, rn1 = _cat(grp, lambda r: r["L1"]["rel_eco"]), _cat(grp, lambda r: r["L1"]["rel_no"])
    nan = np.full(len(fam), np.nan)
    return {"L1_transition": boot_gain(np.where(tr, re1, nan), np.where(tr, rn1, nan), fam, n_boot, 0),
            "L1_within": boot_gain(np.where(~tr, re1, nan), np.where(~tr, rn1, nan), fam, n_boot, 0),
            "L1_all": boot_gain(re1, rn1, fam, n_boot, 0)}


def summarize_level(grp, name: str, n_boot: int, truth: dict | None = None, fam_to_order: dict | None = None) -> dict:
    """一组运行在某一层（locomotor 或 realm）上的合并读数。"""
    lv = [r["units"]["levels"][name] for r in grp]
    targets = grp[0]["units"]["targets"]
    keys = ("rel_eco", "rel_no", "family", "prev_label", "target_label", "target_len", "err_eco", "err_no")
    e1 = {k: np.concatenate([v["E1"][k] for v in lv]) for k in keys}
    f = e1["family"]
    s = {"E1_rel": boot_gain(e1["rel_eco"], e1["rel_no"], f, n_boot, 0), "n_pred": int(len(f)),
         "probe_r2": {c: float(np.nanmean([v["probe_r2_val_units"][c] for v in lv])) for c in targets}}
    for c in ("m_mass", "m_hwi"):
        if c in targets:
            j = targets.index(c)
            s[f"E1_{c}"] = boot_gain(e1["err_eco"][:, j], e1["err_no"][:, j], f, n_boot, 0)
    nan = np.full(len(f), np.nan)
    tl = e1["target_len"]
    masks = {"single": tl == 1, "multi": tl > 1, "len_2_4": (tl >= 2) & (tl <= 4),
             "len_5_16": (tl >= 5) & (tl <= 16), "len_gt16": tl > 16}
    if name == "locomotor":
        masks["no_generalist"] = (e1["prev_label"] != GENERALIST) & (e1["target_label"] != GENERALIST)
        s["share_generalist"] = float(np.mean(~masks["no_generalist"]))
    s["subsets"] = {k: boot_gain(np.where(m, e1["rel_eco"], nan), np.where(m, e1["rel_no"], nan), f, n_boot, 0)
                    for k, m in masks.items()}
    lin_no = np.concatenate([v["linear"]["err_no"] for v in lv])
    lin_ch = np.concatenate([v["linear"]["err_change"] for v in lv])
    s["linear_gain"] = boot_gain(lin_ch, lin_no, f, n_boot, 0)
    s["wm_minus_linear"] = boot_gain_diff(e1["rel_eco"], e1["rel_no"], lin_ch, lin_no, f, n_boot, 0)
    if name == "locomotor":
        ng = masks["no_generalist"]
        s["linear_gain_no_generalist"] = boot_gain(np.where(ng, lin_ch, nan), np.where(ng, lin_no, nan), f, n_boot, 0)
        s["wm_minus_linear_no_generalist"] = boot_gain_diff(
            np.where(ng, e1["rel_eco"], nan), np.where(ng, e1["rel_no"], nan),
            np.where(ng, lin_ch, nan), np.where(ng, lin_no, nan), f, n_boot, 0)

    f2 = np.concatenate([v["E2"]["family"] for v in lv])
    modes = common_modes(lv)
    rel2 = {m: np.concatenate([v["E2"]["rel"][m] for v in lv]) for m in modes}
    s["E2"] = {m: boot_mean(rel2[m], f2, n_boot, 0) for m in modes}
    if fam_to_order is not None:
        o2 = np.array([fam_to_order.get(x, x) for x in f2], dtype=object)
        s["E2_order"] = {m: boot_mean(rel2[m], o2, n_boot, 0) for m in modes}
    s["E2_mean_abs"] = float(np.mean([abs(s["E2"][m]["mean"]) for m in modes]))
    if name == "locomotor" and truth is not None:
        off = truth["leg_offsets"]["L_tmt"]
        tv = [off[m] - off["insessorial"] for m in modes]
        s["E2_rho"] = float(spearmanr([s["E2"][m]["mean"] for m in modes], tv).statistic) if np.ptp(tv) > 0 else float("nan")
    if name == "realm":
        for c in ("m_mass", "m_hwi"):
            if c in targets:
                j = targets.index(c)
                s[f"E2_{c}"] = {m: boot_mean(np.concatenate([v["E2"]["targets"][m][:, j] for v in lv]), f2, n_boot, 0)
                                for m in modes}
    if name == "locomotor" and all("S5" in v for v in lv):
        s5 = {k: np.concatenate([v["S5"][k] for v in lv]) for k in ("err_l2", "err_jump", "err_roll", "family")}
        s["S5"] = {"rmse_l2": float(np.sqrt(np.mean(s5["err_l2"] ** 2))),
                   "rmse_jump": float(np.sqrt(np.mean(s5["err_jump"] ** 2))),
                   "rmse_roll": float(np.sqrt(np.mean(s5["err_roll"] ** 2))),
                   "jump_minus_l2": boot_gain(s5["err_l2"], s5["err_jump"], s5["family"], n_boot, 0),
                   "roll_minus_l2": boot_gain(s5["err_l2"], s5["err_roll"], s5["family"], n_boot, 0),
                   "n_units": int(len(s5["family"]))}
    return s


def h3_readout(s_loc: dict, primary: str) -> dict:
    """H3 的读数：全部第 2 层步，或不涉及泛化的子集。"""
    return s_loc["E1_rel"] if primary == "all" else s_loc["subsets"]["no_generalist"]


def run_means(grp, name: str, modes: list[str]) -> dict:
    """每次运行（同一个 key 的 5 折合并）各类反事实对比的均值，用于零模型阈值。"""
    out = {}
    for key in sorted({r["key"] for r in grp}):
        g = [r["units"]["levels"][name] for r in grp if r["key"] == key]
        out[key] = {m: float(np.mean(np.concatenate([v["E2"]["rel"][m] for v in g]))) for m in modes}
    return out
