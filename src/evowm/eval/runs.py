"""第 0 阶段与真实数据主分析 v1 共用：逐个（模型，折）评估的任务，多进程并行；线性基线。

一个任务 = 一棵训练树上训练的一个折模型，在同一棵树的留出科窗口上评估（窗口与 v3、v4 的取法相同）。
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

N_FOLDS = 5
LOCOMOTOR = ["insessorial", "terrestrial", "aerial", "wading", "diving", "surface", "generalist"]


def fold_datasets(root: Path, table_rel: str, tree_file: str, fold: int, action_mode: str = "full"):
    from evowm.data.transect import TreeTransectDataset
    kw = dict(species_table=table_rel, tree_file=tree_file, level1={"frameskip": 1, "num_steps": 4}, evowm_root=str(root),
              n_trees=1, holdout={"level": "family", "frac": 0, "folds": N_FOLDS, "fold": fold, "seed": 0}, n_orders=4,
              action_mode=action_mode)
    return TreeTransectDataset(name="evowm_train", **kw), TreeTransectDataset(name="evowm_val", **kw)


def ckpt_action_mode(ckpt: Path) -> str:
    """训练时的动作方式，读检查点目录里的 config.yaml（没有这一项的是旧模型，即 full）。"""
    import yaml
    cfg = yaml.safe_load((Path(ckpt).parent / "config.yaml").read_text(encoding="utf-8"))
    return str(((cfg or {}).get("data") or {}).get("dataset", {}).get("action_mode", "full"))


def linear_baseline(ds_tr, ds_va, table: pd.DataFrame, raw: dict, slope: float, seed: int, n_orders: int = 8) -> dict:
    """线性基线（方案二的线性极限），预测目标种的相对跗跖 r = 跗跖 − b × 体重：
    无习性 r̂_B = μ + exp(−αd)(r_A − μ)，μ 为训练物种的总体均值；
    习性当水平用（err_eco，v1）：r̂_B = μ_c(B) + exp(−αd)(r_A − μ_c(A))，μ_c 为训练物种各运动方式的均值；
    习性只当变化用（err_change，v2 起）：无习性预测 + κ(μ_c(B) − μ_c(A))，只在近亲之间运动方式不同时起作用。
    α 各自在训练科的窗口上用最小二乘拟合，κ 也是（限制在 [0, 1]）。留出科上的窗口与 E1 相同，误差顺序与 E1 一致。"""
    from evowm.eval.recovery import sample_windows
    S, cols = ds_tr.S, ds_tr.state_cols
    r_all = S[:, cols.index("m_tarsus")].astype(float) - slope * S[:, cols.index("m_mass")].astype(float)
    reg = table["sim_regime"].to_numpy()
    tr_rows = np.flatnonzero(ds_tr.rows_mask)
    mu = float(np.nanmean(r_all[tr_rows]))
    mu_c = {c: float(np.nanmean(r_all[tr_rows][reg[tr_rows] == c])) for c in np.unique(reg[tr_rows])}
    m_c = np.array([mu_c.get(x, mu) for x in reg])
    H = int(raw["windows"]["H"])

    def flat(rows, dist):
        return rows[:, :H].reshape(-1), rows[:, 1:H + 1].reshape(-1), dist[:, :H].reshape(-1)

    def predict(a, b, d, alpha, habit):
        base_a = m_c[a] if habit else mu
        base_b = m_c[b] if habit else mu
        return base_b + np.exp(-alpha * d) * (r_all[a] - base_a)

    Wt = sample_windows(ds_tr, n_orders, H + 1, seed)
    a_t, b_t, d_t = flat(Wt.rows, Wt.dist)
    alphas = {}
    for habit in (True, False):
        f = lambda la: float(np.nanmean((predict(a_t, b_t, d_t, np.exp(la), habit) - r_all[b_t]) ** 2))  # noqa: E731
        alphas[habit] = float(np.exp(minimize_scalar(f, bounds=(np.log(1e-4), np.log(2.0)), method="bounded").x))
    res_t = r_all[b_t] - predict(a_t, b_t, d_t, alphas[False], False)
    dlt = m_c[b_t] - m_c[a_t]
    ok = np.isfinite(res_t) & (dlt != 0)
    kappa = float(np.clip(np.sum(res_t[ok] * dlt[ok]) / max(float(np.sum(dlt[ok] ** 2)), 1e-12), 0.0, 1.0))
    a, b, d = flat(raw["windows"]["rows"], raw["windows"]["dist"])
    base = predict(a, b, d, alphas[False], False)
    return {"alpha_habit": alphas[True], "alpha_no": alphas[False], "kappa": kappa,
            "err_eco": predict(a, b, d, alphas[True], True) - r_all[b],
            "err_change": base + kappa * (m_c[b] - m_c[a]) - r_all[b],
            "err_no": base - r_all[b]}


def run_task(task: dict) -> dict:
    """task：root, table_rel, tree_file, fold, ckpt, seed, truth（路径或 None）, real（真实数据时为 True）, linear, key。"""
    import torch
    torch.set_num_threads(1)
    from evowm.eval.recovery import evaluate_run, load_world_model
    from evowm.sim.regimes import regime_locomotor
    root = Path(task["root"])
    table = pd.read_csv(root / task["table_rel"])
    truth = None
    if task.get("truth"):
        truth = json.loads(Path(task["truth"]).read_text(encoding="utf-8"))
    if task.get("real"):
        table["sim_regime"] = regime_locomotor(table).to_numpy()
    ds_tr, ds_va = fold_datasets(root, task["table_rel"], task["tree_file"], task["fold"], ckpt_action_mode(Path(task["ckpt"])))
    model = load_world_model(Path(task["ckpt"]), root / "hjepa")
    raw: dict = {}
    res = evaluate_run(ds_tr, ds_va, model, table, truth, n_orders=8, seed=task["seed"], raw=raw,
                       regimes=None if truth is not None else LOCOMOTOR)
    out = {"key": task["key"], "fold": task["fold"], "res": res,
           "E1_state": raw["E1_state"], "family": raw["E1"]["m_tarsus"]["family"],
           "transition": raw["E1"]["m_tarsus"]["transition"],
           "E2_window": raw["E2_window"], "E3": raw["E3"],
           "slope": float(res["E2_relative"]["allometric_slopes"]["m_tarsus"])}
    if task.get("linear"):
        out["linear"] = linear_baseline(ds_tr, ds_va, table, raw, out["slope"], task["seed"])
    return out


def run_all(tasks: list[dict], workers: int | None = None) -> list[dict]:
    workers = int(workers or os.environ.get("EVAL_WORKERS", 8))
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    if workers <= 1:
        return [run_task(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as ex:
        out = []
        for i, r in enumerate(ex.map(run_task, tasks)):
            out.append(r)
            print(f"  评估 {i + 1}/{len(tasks)}：{r['key']} 折 {r['fold']}", flush=True)
        return out


def relative_errors(r: dict) -> tuple[np.ndarray, np.ndarray]:
    """相对跗跖的逐预测误差：跗跖误差 − b × 体重误差（b 为该折训练物种上的异速斜率）。"""
    cols = r["E1_state"]["state_cols"]
    jt, jm = cols.index("m_tarsus"), cols.index("m_mass")
    ee, en = r["E1_state"]["err_eco"], r["E1_state"]["err_no_eco"]
    return ee[:, jt] - r["slope"] * ee[:, jm], en[:, jt] - r["slope"] * en[:, jm]
