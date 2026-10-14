"""事后诊断（v3 判定之后写，不参与任何判定）：零模型里给出习性，为什么让新科上的预测变差。

只用 v3 已训练的模型重新评估，不重新训练。检验三个可能的原因：
1. 体重通路：模型由习性推测目标种的体重变化，再经异速关系影响腿长。
   对照：E1 改用相对腿长（m_tarsus − b·m_mass，b 为该折训练物种上的异速斜率），并单独看体重的 E1。
2. 过拟合：在训练科上也算 E1。训练科上习性有益、新科上有害，即为学到了不可迁移的关联。
3. 按目标种的习性拆开：代价集中在哪几类；比较模型由习性推出的体重偏移，与训练科、新科里的实际残差。

新科上的窗口、探针与自助重抽样和 eval_recovery_v3.py 相同，m_tarsus 的结果应与 v3 报告一致（作为核对）。
在 EvoWM 根目录运行：python scripts/exploratory/diag_e1_null_v3.py [--seeds 1 2 5 6]
输出只写到 outputs/recovery_v3/diag_e1_null/（report.md、diag.json）。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from evowm.data.transect import TreeTransectDataset  # noqa: E402
from evowm.eval.recovery import (Level1, apply_probe, family_bootstrap_rmse_diff, fit_probe,  # noqa: E402
                                 load_world_model, sample_windows)

N_FOLDS, N_ORDERS = 5, 8
TREE = "data/raw/birdtree/hackett_tree1.nwk"
REG_ORDER = ["insessorial", "terrestrial", "aerial", "wading", "diving", "surface", "generalist"]
REG_ZH = {"insessorial": "栖树", "terrestrial": "地栖", "aerial": "空中", "wading": "涉水", "diving": "潜水",
          "surface": "水面", "generalist": "泛化"}
METRICS = {"tarsus": "跗跖（绝对）", "rel_tarsus": "跗跖（相对）", "mass": "体重"}


def fold_datasets(rel: str, fold: int):
    kw = dict(species_table=rel, tree_file=TREE, level1={"frameskip": 1, "num_steps": 4}, evowm_root=str(ROOT),
              n_trees=1, holdout={"level": "family", "frac": 0, "folds": N_FOLDS, "fold": fold, "seed": 0}, n_orders=4)
    return TreeTransectDataset(name="evowm_train", **kw), TreeTransectDataset(name="evowm_val", **kw)


def fold_arrays(L1, probes, ds_tr, ds_eval, table: pd.DataFrame, seed: int, H: int, b: float) -> dict:
    """在 ds_eval 的样带窗口上，逐预测给出有、无习性两种输入的误差（预测 − 真值）。"""
    W = sample_windows(ds_eval, N_ORDERS, H + 1, seed)
    act_eco = np.stack([ds_eval.build_action(r, d, eco=True) for r, d in zip(W.rows, W.dist)])
    act_no = np.stack([ds_eval.build_action(r, d, eco=False) for r, d in zip(W.rows, W.dist)])
    emb = L1.encode(ds_eval.pix[W.rows])
    pe = apply_probe(probes, L1.predict(emb[:, :H], act_eco[:, :H]))
    pn = apply_probe(probes, L1.predict(emb[:, :H], act_no[:, :H]))
    prev, tgt = W.rows[:, :H], W.rows[:, 1:H + 1]
    true = ds_tr.S[tgt]
    jt, jm = ds_tr.state_cols.index("m_tarsus"), ds_tr.state_cols.index("m_mass")
    reg = table["sim_regime"].to_numpy()
    out = {"family": table["family"].to_numpy()[tgt].reshape(-1),
           "target_regime": reg[tgt].reshape(-1),
           "transition": (reg[prev] != reg[tgt]).reshape(-1)}
    for tag, p in (("eco", pe), ("no", pn)):
        e = p - true
        out[f"tarsus_{tag}"] = e[..., jt].reshape(-1)
        out[f"mass_{tag}"] = e[..., jm].reshape(-1)
        out[f"rel_tarsus_{tag}"] = (e[..., jt] - b * e[..., jm]).reshape(-1)
    return out


def collect(sc: str, s: int, ckpt_root: Path) -> dict:
    """5 折合并：新科（各折留出科）与训练科（各折训练科）两组逐预测误差。"""
    rel = f"data/processed/sim/recovery_v3/{sc}_s{s}/species_traits.csv"
    table = pd.read_csv(ROOT / rel)
    pooled = {"new": [], "train": []}
    slopes = []
    for k in range(N_FOLDS):
        ds_tr, ds_va = fold_datasets(rel, k)
        model = load_world_model(ckpt_root / f"{sc}_s{s}_f{k}" / "evowm_l1_object.ckpt", ROOT / "hjepa")
        L1 = Level1(model)
        H = int(L1.jepa.predictor.predictor.pos_embedding.shape[1]) if hasattr(L1.jepa.predictor, "predictor") else 3
        rows_tr = np.flatnonzero(ds_tr.rows_mask)
        probes = fit_probe(L1.encode(ds_tr.pix), ds_tr.S, rows_tr)
        jt, jm = ds_tr.state_cols.index("m_tarsus"), ds_tr.state_cols.index("m_mass")
        ok = rows_tr[np.isfinite(ds_tr.S[rows_tr, jt]) & np.isfinite(ds_tr.S[rows_tr, jm])]
        b = float(np.polyfit(ds_tr.S[ok, jm], ds_tr.S[ok, jt], 1)[0])
        slopes.append(b)
        pooled["new"].append(fold_arrays(L1, probes, ds_tr, ds_va, table, s, H, b))
        pooled["train"].append(fold_arrays(L1, probes, ds_tr, ds_tr, table, s, H, b))
    out = {key: {f: np.concatenate([d[f] for d in lst]) for f in lst[0]} for key, lst in pooled.items()}
    out["slopes"] = slopes
    return out


def gain(a: dict, metric: str, mask: np.ndarray | None, seed: int) -> dict:
    ea, en = a[f"{metric}_eco"], a[f"{metric}_no"]
    if mask is not None:
        ea, en = np.where(mask, ea, np.nan), np.where(mask, en, np.nan)
    g, lo, hi, n, nf = family_bootstrap_rmse_diff(ea, en, a["family"], seed=seed)
    return {"gain": g, "ci95": [lo, hi], "n_pred": n, "n_families": nf}


def summarize(arr: dict, s: int) -> dict:
    res = {"slopes": arr["slopes"]}
    for where in ("new", "train"):
        a = arr[where]
        res[where] = {m: {"all": gain(a, m, None, s), "transition": gain(a, m, a["transition"], s)} for m in METRICS}
        per = {}
        for r in REG_ORDER:
            sel = a["target_regime"] == r
            if sel.sum() == 0:
                continue
            per[r] = {"n_pred": int(sel.sum()), "n_families": int(len(np.unique(a["family"][sel]))),
                      # 模型由习性推出的偏移（有习性预测 − 无习性预测），与实际残差（真值 − 无习性预测）
                      "mass_shift_model": float(np.nanmean((a["mass_eco"] - a["mass_no"])[sel])),
                      "mass_residual": float(np.nanmean(-a["mass_no"][sel])),
                      "tarsus_shift_model": float(np.nanmean((a["tarsus_eco"] - a["tarsus_no"])[sel])),
                      "tarsus_residual": float(np.nanmean(-a["tarsus_no"][sel]))}
            for m in METRICS:
                per[r][f"gain_{m}"] = gain(a, m, sel, s)
        res[where]["per_regime"] = per
    return res


def ci(d: dict) -> str:
    return f"{d['gain']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"


def report(res: dict, path: Path):
    L = ["# 事后诊断：零模型里给出习性为什么让预测变差（v3 模型，不参与判定）", "",
         "增益 = 无习性 RMSE − 有习性 RMSE（log10），为负表示给出习性后预测变差；95% 置信区间按科自助重抽样 1,000 次。"
         "\"新科\"是各折留出的科（与 v3 评估相同，跗跖绝对值的一列应与 v3 报告一致）；\"训练科\"是各折训练用的科。"
         "相对跗跖 = 跗跖 − b × 体重，b 为各折训练物种上的异速斜率。", "",
         "## 1. 全部相邻对：新科与训练科", "",
         "| 模型 | 位置 | 跗跖（绝对） | 跗跖（相对） | 体重 |", "|---|---|---|---|---|"]
    for k, r in res.items():
        for where, zh in (("new", "新科"), ("train", "训练科")):
            L.append(f"| {k} | {zh} | " + " | ".join(ci(r[where][m]["all"]) for m in METRICS) + " |")
    L += ["", "## 2. 习性变化的物种对：新科与训练科", "",
          "| 模型 | 位置 | 跗跖（绝对） | 跗跖（相对） | 体重 |", "|---|---|---|---|---|"]
    for k, r in res.items():
        for where, zh in (("new", "新科"), ("train", "训练科")):
            L.append(f"| {k} | {zh} | " + " | ".join(ci(r[where][m]["transition"]) for m in METRICS) + " |")
    L += ["", "## 3. 零模型按目标种习性拆开（新科）", "",
          "体重偏移：模型因为知道习性而改变的体重预测（有习性 − 无习性）。实际残差：真值 − 无习性预测，"
          "即习性\"能\"解释的部分。零模型里习性没有作用，这些残差只是类群上的巧合：模型的偏移如果跟着训练科的残差走、"
          "而和新科的残差对不上，就说明它学到的是训练科里的巧合。", "",
          "| 模型 | 习性 | 科数 | 体重偏移（模型） | 体重残差（训练科） | 体重残差（新科） | 增益：跗跖绝对 | 增益：跗跖相对 | 增益：体重 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, r in res.items():
        if not k.startswith("null"):
            continue
        for reg, d in r["new"]["per_regime"].items():
            tr = r["train"]["per_regime"].get(reg, {})
            L.append(f"| {k} | {REG_ZH[reg]} | {d['n_families']} | {d['mass_shift_model']:+.4f} | "
                     f"{tr.get('mass_residual', float('nan')):+.4f} | {d['mass_residual']:+.4f} | "
                     f"{d['gain_tarsus']['gain']:+.4f} | {d['gain_rel_tarsus']['gain']:+.4f} | {d['gain_mass']['gain']:+.4f} |")
    L += ["", "## 4. 效应情景按目标种习性拆开（新科，只列增益）", "",
          "| 模型 | 习性 | 科数 | 跗跖绝对 | 跗跖相对 | 体重 |", "|---|---|---|---|---|---|"]
    for k, r in res.items():
        if not k.startswith("effect"):
            continue
        for reg, d in r["new"]["per_regime"].items():
            L.append(f"| {k} | {REG_ZH[reg]} | {d['n_families']} | {d['gain_tarsus']['gain']:+.4f} | "
                     f"{d['gain_rel_tarsus']['gain']:+.4f} | {d['gain_mass']['gain']:+.4f} |")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 5, 6])
    ap.add_argument("--ckpt-root", default=str(ROOT / "outputs/hjepa/ckpts/recovery_v3"))
    ap.add_argument("--out", default=str(ROOT / "outputs/recovery_v3/diag_e1_null"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for s in args.seeds:
        for sc in ("null", "effect"):
            key = f"{sc}_s{s}"
            res[key] = summarize(collect(sc, s, Path(args.ckpt_root)), s)
            n, t = res[key]["new"], res[key]["train"]
            print(f"{key}: 新科 跗跖绝对 {ci(n['tarsus']['all'])}，跗跖相对 {ci(n['rel_tarsus']['all'])}，"
                  f"体重 {ci(n['mass']['all'])}；训练科 跗跖绝对 {ci(t['tarsus']['all'])}", flush=True)
            (out / "diag.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    report(res, out / "report.md")
    print(f"报告：{out / 'report.md'}")


if __name__ == "__main__":
    main()
