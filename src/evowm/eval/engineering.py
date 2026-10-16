"""工程验证 v1 的模型读数（预注册 docs/prereg/engineering_drop_v1.md 第 2 节）：
第二层"栖树 → 地栖"反事实下，三段骨与体重的对数长度怎么变。

算法与主分析的 E2 相同（evowm.eval.units.e2_level：留出科叶序上每个窗口的第 1 个单位作上下文，下一个单位的平均习性
换成某一类的典型值，读出该单位成员平均的预测值），只取地栖与栖树两类之差。叶序、单位表示、探针与主分析逐一相同
（同样的种子与取法），所以全部窗口上相对跗跖的差与主分析的 E2 地栖对比一致，脚本据此自检。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..data.units import LOCOMOTOR
from .units import Level, _cat, e2_level, fill_units, fit_probes, make_orders, prototypes

SEGMENTS = ["k_tarsometatarsus", "k_tibiotarsus", "k_femur"]
COLS = SEGMENTS + ["m_mass", "m_tarsus"]


def delta_for_model(ds_tr, ds_va, model, table: pd.DataFrame, n_orders: int = 8, seed: int = 0) -> dict:
    """一个（模型，折）：逐窗口的"地栖典型习性 − 栖树典型习性"预测差（COLS 各列，log10），目标单位的主要科，
    以及与主分析相同口径的相对跗跖差（跗跖 − b × 体重，b 为该折训练物种上的异速斜率）。"""
    state_cols = list(ds_tr.state_cols)
    S = ds_tr.S
    it, im = state_cols.index("m_tarsus"), state_cols.index("m_mass")
    rows_tr = np.flatnonzero(ds_tr.rows_mask)
    ok = rows_tr[np.isfinite(S[rows_tr, it]) & np.isfinite(S[rows_tr, im])]
    b = float(np.polyfit(S[ok, im], S[ok, it], 1)[0])
    fam_codes, fam_names = pd.factorize(table["family"])
    fam_names = np.asarray(fam_names, dtype=object)

    z1_all = Level(model, 1).encode(ds_tr.pix)
    tr = make_orders(ds_tr, n_orders, seed)
    va = make_orders(ds_va, n_orders, seed + 1)
    fill_units(model, tr + va, z1_all, S.astype(float), fam_codes)
    name = "locomotor"
    Ztr, _ = _cat(tr, lambda o: o.z[name])
    Ytr, _ = _cat(tr, lambda o: o.y[name])
    probes = fit_probes(Ztr, Ytr)
    protos = prototypes(tr, name)
    ti, bi = LOCOMOTOR.index("terrestrial"), LOCOMOTOR.index("insessorial")
    if ti not in protos or bi not in protos:
        raise RuntimeError("训练单位里缺少地栖或栖树的典型习性")
    e2 = e2_level(model, va, name, probes, {bi: protos[bi], ti: protos[ti]})
    _, off = _cat(va, lambda o: o.y[name])
    famv, _ = _cat(va, lambda o: o.fam[name])
    g2 = off[e2["order"]] + e2["target"]
    diff = e2["pred"][ti] - e2["pred"][bi]
    j = [state_cols.index(c) for c in COLS]
    return {"family": fam_names[famv[g2]], "diff": diff[:, j].astype(np.float64), "cols": list(COLS),
            "rel_tarsus": (diff[:, it] - b * diff[:, im]).astype(np.float64), "slope": b}


def run_delta_task(task: dict) -> dict:
    """task 同 evowm.eval.runs.run_unit_task（root、table_rel、tree_file、fold、seed、ckpt、key）。"""
    import torch
    torch.set_num_threads(1)
    from .recovery import load_world_model
    from .runs import ckpt_action_mode, fold_datasets
    root = Path(task["root"])
    table = pd.read_csv(root / task["table_rel"])
    ckpt = Path(task["ckpt"])
    model = load_world_model(ckpt, root / "hjepa")
    units = getattr(model, "evowm_units", None)
    if not units:
        raise RuntimeError(f"{ckpt} 不是分层模型")
    ds_tr, ds_va = fold_datasets(root, task["table_rel"], task["tree_file"], task["fold"], ckpt_action_mode(ckpt),
                                 units=units)
    out = delta_for_model(ds_tr, ds_va, model, table, n_orders=8, seed=task["seed"])
    out.update(key=task["key"], fold=task["fold"])
    return out
