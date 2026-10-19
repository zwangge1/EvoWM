"""读数诊断与"变化量读数"（探索性；设计 docs/design/诊断_读数_2026-10-19.md）。

逐个（模型，折）计算，只返回按（科，类型，性状）汇总的平方误差和，合并后按科自助重抽样。窗口、探针、单位与简单方法
都与 evowm.eval.wholebody 相同（同样的种子），所有读数与方法在同一批预测上比（前后两边都有实测值）：
- copy：照抄前一个的实测值 y_A；
- l_no、l_ch、l_full：简单方法，不告诉习性、只用类别、全部 47 项（与 wholebody 相同）；
- a_eco、a_no：世界模型原读数 p(ẑ_B)，告诉与不告诉生活方式（与 wholebody 的 w_eco、w_no 相同）；
- d_eco、d_no：世界模型新读数 y_A + p(ẑ_B) − p(z_A)，即前一个的实测值加模型预测的变化量；
- floor：读数下限 p(z_B) − y_B（预测器把目标的表示猜得完全对时，原读数还剩的误差）；
- zcopy：表示照抄 p(z_A) − y_B（预测器原样照抄前一个的表示时，原读数的误差）。
p 为读数探针（第 1 层在训练物种上拟合，第 2、3 层在训练单位上拟合）；z 为编码器给出的表示。
长度类形态扣除体重通路（相对值 = 值 − b × 体重，b 为该折训练物种上的 OLS 斜率）；体重与 HWI 不扣。
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .wholebody import _linear, _linear_units_full, agg, datasets, rel, slopes_for, step_types

KEYS = ("copy", "l_no", "l_ch", "l_full", "a_eco", "a_no", "d_eco", "d_no", "floor", "zcopy")


def _sums(fam: np.ndarray, typ: np.ndarray, errs: dict, F: int) -> dict:
    """同一批预测（全部读数与方法都有限）上的平方误差和，形状 (F, 3, C)；另返回个数 n。"""
    ok = np.logical_and.reduce([np.isfinite(errs[k]) for k in KEYS])
    out = {}
    n = None
    for k in KEYS:
        out[k], n = agg(fam, typ, np.where(ok, errs[k] ** 2, np.nan), F, 3)
    out["n"] = n
    return out


def eval_l1_readout(ds_tr, ds_va, model, table: pd.DataFrame, fam_codes: np.ndarray, F: int, n_orders: int,
                    seed: int) -> dict:
    """第 1 层（物种）：留出科叶序上的窗口（与 wholebody.eval_l1 相同），逐预测的各种读数。"""
    from ..data.units import habit_codes
    from .recovery import Level1, apply_probe, fit_probe, sample_windows
    L1 = Level1(model)
    H = int(L1.jepa.predictor.predictor.pos_embedding.shape[1]) if hasattr(L1.jepa.predictor, "predictor") else 3
    cols = list(ds_tr.state_cols)
    S = ds_tr.S.astype(float)
    jm = cols.index("m_mass")
    C = len(cols)
    rows_tr = np.flatnonzero(ds_tr.rows_mask)
    b = slopes_for(S, cols, rows_tr)
    probes = fit_probe(L1.encode(ds_tr.pix), S, rows_tr)
    W = sample_windows(ds_va, n_orders, H + 1, seed)
    act_eco = np.stack([ds_va.build_action(r, d, eco=True) for r, d in zip(W.rows, W.dist)])
    act_no = np.stack([ds_va.build_action(r, d, eco=False) for r, d in zip(W.rows, W.dist)])
    emb = L1.encode(ds_va.pix[W.rows])                                  # (N, H+1, D)
    pe = apply_probe(probes, L1.predict(emb[:, :H], act_eco[:, :H]))    # (N, H, C)
    pn = apply_probe(probes, L1.predict(emb[:, :H], act_no[:, :H]))
    pz = apply_probe(probes, emb)                                       # 编码器表示的读数 (N, H+1, C)
    prv, tgt = W.rows[:, :H], W.rows[:, 1:H + 1]
    yA, yB, pA, pB = S[prv], S[tgt], pz[:, :H], pz[:, 1:]

    def R(E):
        return rel(E, b, jm).reshape(-1, C)

    errs = {"copy": R(yA - yB), "a_eco": R(pe - yB), "a_no": R(pn - yB),
            "d_eco": R(yA + pe - pA - yB), "d_no": R(yA + pn - pA - yB),
            "floor": R(pB - yB), "zcopy": R(pA - yB)}

    # 简单方法：训练科窗口上拟合，逐项形态（与 wholebody.eval_l1 相同）
    loco = habit_codes(table)
    Wt = sample_windows(ds_tr, n_orders, H + 1, seed)
    a_t, b_t, d_t = Wt.rows[:, :H].reshape(-1), Wt.rows[:, 1:H + 1].reshape(-1), Wt.dist[:, :H].reshape(-1)
    a, bb, d = prv.reshape(-1), tgt.reshape(-1), W.dist[:, :H].reshape(-1)
    Rs = rel(S, b, jm)
    A_dim = act_eco.shape[-1]
    Xt = np.stack([ds_tr.build_action(r, dd, eco=True) for r, dd in zip(Wt.rows, Wt.dist)])[:, :H].reshape(-1, A_dim)[:, :-2]
    Xv = act_eco[:, :H].reshape(-1, A_dim)[:, :-2]
    for k in ("l_no", "l_ch", "l_full"):
        errs[k] = np.full((len(a), C), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for j in range(C):
            errs["l_no"][:, j], errs["l_ch"][:, j], errs["l_full"][:, j] = _linear(
                Rs[:, j], loco, a_t, b_t, d_t, a, bb, d, Xt, Xv)
    typ = step_types(loco[prv].reshape(-1), loco[tgt].reshape(-1))
    fam = fam_codes[tgt].reshape(-1)
    return _sums(fam, typ, errs, F)


def eval_units_readout(ds_tr, ds_va, model, table: pd.DataFrame, fam_codes: np.ndarray, F: int, n_orders: int,
                       seed: int, b: np.ndarray) -> dict:
    """第 2、3 层（单位）：与 wholebody.eval_units_wb 的 E1 相同的窗口与探针，逐预测的各种读数。"""
    from .units import Level, _cat, apply_probes, e1_level, fill_units, fit_probes, linear_units, make_orders
    cols = list(ds_tr.state_cols)
    jm = cols.index("m_mass")
    V = ds_tr.S.astype(float)
    C = len(cols)
    z1_all = Level(model, 1).encode(ds_tr.pix)
    tr = make_orders(ds_tr, n_orders, seed)
    va = make_orders(ds_va, n_orders, seed + 1)
    fill_units(model, tr + va, z1_all, V, fam_codes)
    out = {}
    for name in model.evowm_units["levels"]:
        Ztr, _ = _cat(tr, lambda o: o.z[name])
        Ytr, _ = _cat(tr, lambda o: o.y[name])
        probes = fit_probes(Ztr, Ytr)
        Yva, off = _cat(va, lambda o: o.y[name])
        Zva, _ = _cat(va, lambda o: o.z[name])
        PZ = apply_probes(probes, Zva)                                  # 单位表示的读数
        lab, _ = _cat(va, lambda o: o.levels[name].label)
        famv, _ = _cat(va, lambda o: o.fam[name])
        e1 = e1_level(model, va, name, probes)
        gp, gt = off[e1["order"]] + e1["prev"], off[e1["order"]] + e1["target"]
        yA, yB, pA, pB = Yva[gp], Yva[gt], PZ[gp], PZ[gt]
        pe, pn = e1["pred_eco"], e1["pred_no"]

        def R(E):
            return rel(E, b, jm)

        errs = {"copy": R(yA - yB), "a_eco": R(pe - yB), "a_no": R(pn - yB),
                "d_eco": R(yA + pe - pA - yB), "d_no": R(yA + pn - pA - yB),
                "floor": R(pB - yB), "zcopy": R(pA - yB)}
        for k in ("l_no", "l_ch", "l_full"):
            errs[k] = np.full((len(gp), C), np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for j in range(C):
                r_tr = [rel(o.y[name], b, jm)[:, j] for o in tr]
                r_va = [rel(o.y[name], b, jm)[:, j] for o in va]
                if np.isfinite(np.concatenate(r_tr)).sum() < 10:
                    continue
                lu = linear_units(tr, va, name, r_tr, r_va, e1["order"], e1["prev"], e1["target"])
                errs["l_no"][:, j], errs["l_ch"][:, j] = lu["err_no"], lu["err_change"]
                _, errs["l_full"][:, j] = _linear_units_full(tr, va, name, r_tr, r_va, e1["order"], e1["prev"],
                                                             e1["target"])
        typ = step_types(lab[gp], lab[gt]) if name == "locomotor" else np.where(lab[gp] == lab[gt], 0, 2)
        out[name] = _sums(famv[gt], typ, errs, F)
    return out


def run_readout_task(task: dict) -> dict:
    """task：root, table_rel, tree_file, fold, ckpt, seed, key, set。出错时返回 {"error": ...}，不中断整批运行。"""
    try:
        import json

        import torch
        torch.set_num_threads(1)
        from .recovery import load_world_model
        from .runs import ckpt_action_mode
        root = Path(task["root"])
        table = pd.read_csv(root / task["table_rel"])
        meta = json.loads((root / task["table_rel"]).with_suffix(".columns.json").read_text(encoding="utf-8"))
        cols = list(meta["morph"]) + list(meta["skeleton"])
        ckpt = Path(task["ckpt"])
        mode = ckpt_action_mode(ckpt)
        if mode != "change":
            raise ValueError(f"动作方式为 {mode}，只支持 change")
        model = load_world_model(ckpt, root / "hjepa")
        units = getattr(model, "evowm_units", None)
        ds_tr, ds_va = datasets(root, task["table_rel"], task["tree_file"], task["fold"], mode, units, cols)
        fam_codes, fam_names = pd.factorize(table["family"])
        F = len(fam_names)
        n_orders = task.get("n_orders", 8)
        out = {"key": task["key"], "fold": task["fold"], "set": task["set"], "cols": cols,
               "families": [str(x) for x in fam_names]}
        out["L1"] = eval_l1_readout(ds_tr, ds_va, model, table, fam_codes, F, n_orders, task["seed"])
        if units:
            b = slopes_for(ds_tr.S.astype(float), cols, np.flatnonzero(ds_tr.rows_mask))
            lv = eval_units_readout(ds_tr, ds_va, model, table, fam_codes, F, n_orders, task["seed"], b)
            if "locomotor" in lv:
                out["L2"] = lv["locomotor"]
            if "realm" in lv:
                out["L3"] = lv["realm"]
        return out
    except Exception as e:  # noqa: BLE001
        import traceback
        return {"key": task.get("key"), "fold": task.get("fold"), "set": task.get("set"),
                "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]}
