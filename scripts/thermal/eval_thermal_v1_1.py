"""热阀反事实 v1.1：事后修订的世界模型读数（在 v1 预注册读数之外加"扣除体重"的版本，标为事后）。

v1 的校准把世界模型的原始温度斜率与 PGLS 的斜率对表，但 PGLS 的斜率是控制 log10 体重之后的（相对跗跖），
而世界模型的反事实连体重一起变（气候减 ΔT 时体重也变）。两者口径不同。本脚本对每个预测算
  相对斜率 = 该列的斜率 − b × 体重的斜率，
b 为该折训练物种上该列对 log10 体重的 OLS 异速斜率（与主分析的"相对跗跖"同一口径），再做校准、梯度与线性三项读数。
v1 的判定不改；本脚本的结果单独写 outputs/thermal_v1/report_v1_1.md、results_v1_1.json，论文里注明为事后修订。
在 EvoWM 根目录运行：python scripts/thermal/eval_thermal_v1_1.py [--ckpt-root ...] [--trees ...] [--seeds ...] [--folds ...]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/thermal"))

from eval_thermal_v1 import (COLS, COLS_ZH, DELTAS, DESIGN, DEVMODE, GROUPS_ZH, MAIN_DT, PGLS_REF, TABLE,  # noqa: E402
                             boot_ols, ci_str, tasks_for)
from evowm.eval.pooled import boot_mean, strip_boot  # noqa: E402
from evowm.eval.runs import N_FOLDS, run_all  # noqa: E402

LEG = ["m_tarsus", "k_tarsometatarsus", "k_tibiotarsus", "k_femur"]


def run_task_rel(task: dict) -> dict:
    import torch
    torch.set_num_threads(1)
    from evowm.eval.recovery import Level1, apply_probe, fit_probe, load_world_model, sample_windows
    from evowm.eval.runs import fold_datasets
    root = Path(task["root"])
    ds_tr, ds_va = fold_datasets(root, task["table_rel"], task["tree_file"], task["fold"], "change")
    model = load_world_model(Path(task["ckpt"]), root / "hjepa")
    L1 = Level1(model)
    S = ds_tr.S
    cols = list(ds_tr.state_cols)
    rows_tr = np.flatnonzero(ds_tr.rows_mask)
    im = cols.index("m_mass")
    b = {}
    for c in LEG:
        j = cols.index(c)
        ok = rows_tr[np.isfinite(S[rows_tr, j]) & np.isfinite(S[rows_tr, im])]
        b[c] = float(np.polyfit(S[ok, im], S[ok, j], 1)[0])
    z_all = L1.encode(ds_tr.pix)
    probes = fit_probe(z_all, S, rows_tr)
    W = sample_windows(ds_va, 8, 2, task["seed"])
    g_clim = ds_va.eco_group_names.index("climate")
    clim_idx = np.array([i for i, c in enumerate(ds_va.eco_cols) if c.startswith("e_clim_")])
    emb = L1.encode(ds_va.pix[W.rows])[:, :1]
    acts = {0: []}
    acts.update({dt: [] for dt in DELTAS})
    keep = []
    for rr, d in zip(W.rows, W.dist):
        E, Eg = ds_va.eco_rows(rr)
        keep.append(bool(Eg[0, g_clim] > 0 and Eg[1, g_clim] > 0))
        acts[0].append(ds_va.build_action(rr, d, eco=True, E=E, Eg=Eg))
        for dt in DELTAS:
            E2 = E.copy()
            E2[1, clim_idx] -= dt / 10.0
            acts[dt].append(ds_va.build_action(rr, d, eco=True, E=E2, Eg=Eg))
    j = [cols.index(c) for c in COLS]
    pred = {dt: apply_probe(probes, L1.predict(emb, np.stack(a)[:, :1]))[:, 0, j] for dt, a in acts.items()}
    keep = np.array(keep)
    jm = COLS.index("m_mass")
    slope, rel = {}, {}
    for dt in DELTAS:
        s = ((pred[0] - pred[dt]) / (dt / 10.0))[keep].astype(np.float64)
        slope[dt] = s
        rel[dt] = np.stack([s[:, COLS.index(c)] - b[c] * s[:, jm] for c in LEG], axis=1)
    return {"key": task["key"], "fold": task["fold"], "target": W.rows[keep, 1].astype(int), "slope": slope, "rel": rel, "b": b}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", default=str(ROOT / "outputs/hjepa/ckpts/thermal_v1"))
    ap.add_argument("--trees", type=int, nargs="+", default=[21, 41, 61, 81, 100])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(N_FOLDS)))
    ap.add_argument("--out", default=str(ROOT / "outputs/thermal_v1"))
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    if not DESIGN.exists():
        sys.exit(f"缺少设计文档 {DESIGN.relative_to(ROOT)}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(ROOT / TABLE)
    dev = pd.read_csv(DEVMODE)
    fam_exposed = dict(zip(dev["family"], dev["exposed"]))
    is_pass = (table["order"] == "Passeriformes").to_numpy()
    exposed = table["family"].map(fam_exposed).to_numpy()
    X = table[["e_fs_ground", "e_fs_wataroundsurf", "e_fs_watbelowsurf"]].sum(axis=1, min_count=3).to_numpy()
    has_k = table["k_tarsometatarsus"].notna().to_numpy()
    fam = table["family"].to_numpy()
    groups = {"all": np.ones(len(table), bool), "nonpass": ~is_pass, "passerine": is_pass,
              "nonpass_exposed": ~is_pass & (exposed == 1), "nonpass_nest": ~is_pass & (exposed == 0),
              "anatidae": (table["family"] == "Anatidae").to_numpy()}
    tasks = tasks_for(Path(args.ckpt_root), args.trees, args.seeds, args.folds)
    print(f"评估 {len(tasks)} 个（模型，折），相对斜率版")
    runs = run_all(tasks, workers=args.workers, fn=run_task_rel)
    tgt = np.concatenate([r["target"] for r in runs])
    R = {dt: np.concatenate([r["rel"][dt] for r in runs]) for dt in DELTAS}
    Sm = {dt: np.concatenate([r["slope"][dt][:, COLS.index("m_mass")] for r in runs]) for dt in DELTAS}
    b_mean = {c: float(np.mean([r["b"][c] for r in runs])) for c in LEG}
    res = {"n_runs": len(runs), "n_pred": int(len(tgt)), "allometric_slopes_mean": b_mean, "groups": {}, "calibration": {},
           "gradient": {}, "mass": {}}
    for g, mask in groups.items():
        sel = mask[tgt]
        res["groups"][g] = {}
        for j, c in enumerate(LEG):
            sel_c = sel & (has_k[tgt] if c.startswith("k_") else True)
            if sel_c.sum() == 0:
                continue
            res["groups"][g][c] = {dt: strip_boot(boot_mean(R[dt][sel_c, j], fam[tgt][sel_c])) for dt in DELTAS}
        if sel.sum():
            res["mass"][g] = strip_boot(boot_mean(Sm[MAIN_DT][sel], fam[tgt][sel]))
    for key, (pt, lo, hi, src) in PGLS_REF.items():
        g, c = key.split("/")
        m = res["groups"].get(g, {}).get(c, {}).get(MAIN_DT)
        if m is None:
            continue
        res["calibration"][key] = {"wm_rel": m["mean"], "wm_ci95": m["ci95"], "pgls": pt, "pgls_ci95": [lo, hi],
                                   "inside": bool(lo <= m["mean"] <= hi), "same_sign": bool(np.sign(m["mean"]) == np.sign(pt)), "source": src}
    main4 = ["nonpass_exposed/m_tarsus", "nonpass_nest/m_tarsus", "passerine/m_tarsus", "anatidae/k_tarsometatarsus"]
    inside = sum(res["calibration"][k]["inside"] for k in main4 if k in res["calibration"])
    signs = all(res["calibration"][k]["same_sign"] for k in main4 if k in res["calibration"])
    res["calibration"]["verdict"] = {"n_inside_of_4": int(inside), "all_same_sign": bool(signs), "pass": bool(inside >= 3 and signs)}
    df = pd.DataFrame({"tip": tgt, "s": R[MAIN_DT][:, 0]})
    sp = df.groupby("tip")["s"].mean()
    rows = sp.index.to_numpy()
    for g in ["nonpass", "passerine"]:
        sel = groups[g][rows] & np.isfinite(X[rows])
        r_ = boot_ols(X[rows][sel], sp.to_numpy()[sel], fam[rows][sel])
        r_["pass"] = bool(r_["ci95"][0] > 0)
        res["gradient"][g] = r_
    res["gradient"]["verdict"] = {"both": bool(all(res["gradient"][g]["pass"] for g in ["nonpass", "passerine"]))}
    (out / "results_v1_1.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    cv = res["calibration"]["verdict"]
    L = ["# 热阀反事实 v1.1：世界模型读数，扣除体重（事后修订）", "",
         f"{len(runs)} 个（模型，折），{len(tgt)} 个预测。相对斜率 = 该列斜率 − b × 体重斜率，b 为各折训练物种上的异速斜率"
         f"（均值：" + "，".join(f"{COLS_ZH[c]} {b_mean[c]:.3f}" for c in LEG) + "）。与 PGLS（控制体重）同口径。方括号为按科自助重抽样的 95% 区间。", "",
         "## 世界模型的体重响应（每升高 10 °C，log10）", ""]
    for g in ["nonpass_exposed", "nonpass_nest", "passerine", "anatidae"]:
        if g in res["mass"]:
            L.append(f"- {GROUPS_ZH[g]}：{ci_str(res['mass'][g])}")
    L += ["", f"## 校准（相对斜率，ΔT = {MAIN_DT}）：{'通过' if cv['pass'] else '不通过'}（4 项里 {cv['n_inside_of_4']} 项在 PGLS 区间内，方向{'全对' if cv['all_same_sign'] else '有错'}）", "",
          "| 读数 | 世界模型（扣体重） | PGLS 参照 | 在区间内 |", "|---|---|---|---|"]
    for key, v in res["calibration"].items():
        if key == "verdict":
            continue
        g, c = key.split("/")
        L.append(f"| {GROUPS_ZH[g]}，{COLS_ZH[c]} | {v['wm_rel']:+.4f} [{v['wm_ci95'][0]:+.4f}, {v['wm_ci95'][1]:+.4f}] | "
                 f"{v['pgls']:+.4f} [{v['pgls_ci95'][0]:+.4f}, {v['pgls_ci95'][1]:+.4f}] | {'是' if v['inside'] else '否'} |")
    L += ["", f"## 梯度（相对斜率对 X，ΔT = {MAIN_DT}）：" + ("两组都为正" if res["gradient"]["verdict"]["both"] else "未两组都为正"), ""]
    for g in ["nonpass", "passerine"]:
        r_ = res["gradient"][g]
        L.append(f"- {GROUPS_ZH[g]}：X 每增加 1，相对斜率增加 {r_['slope']:+.4f} [{r_['ci95'][0]:+.4f}, {r_['ci95'][1]:+.4f}]（{r_['n']} 种，{r_['n_clusters']} 科）")
    L += ["", "## 分组相对斜率（每升高 10 °C）", "", "| 组 | 列 | " + " | ".join(f"ΔT = {dt}" for dt in DELTAS) + " | n |", "|---|---|" + "---|" * len(DELTAS) + "---|"]
    for g in ["nonpass_exposed", "nonpass_nest", "nonpass", "passerine", "anatidae", "all"]:
        for c in LEG:
            d = res["groups"][g].get(c)
            if not d:
                continue
            L.append(f"| {GROUPS_ZH[g]} | {COLS_ZH[c]} | " + " | ".join(ci_str(d[dt]) for dt in DELTAS) + f" | {d[MAIN_DT]['n']} |")
    (out / "report_v1_1.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
