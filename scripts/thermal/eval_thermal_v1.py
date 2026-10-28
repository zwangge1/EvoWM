"""热阀反事实 v1 的世界模型读数（设计 docs/design/热阀与落震_反事实闭环_设计_2026-10-28.md 第 4、5 节的 Q3）。

每个（模型，折）：留出科的窗口（上下文 1 种 → 目标种），目标种的三项气候各减 ΔT（10、20、30 °C；主口径 20），
其余动作不变，预测经探针读出 m_tarsus、k_tarsometatarsus、k_tibiotarsus、k_femur、m_mass（log10），
(观测动作预测 − 反事实预测) / (ΔT / 10) 就是模型给出的"每升高 10 °C"的斜率。50 个模型全部窗口合并，按科自助重抽样。
读数：
  校准：非雀形目暴露组、巢内组、雀形目的 m_tarsus 斜率，鸭科的 k_tarsometatarsus 斜率，是否落在 PGLS 的 95% 区间内；
  梯度：非雀形目内、雀形目内，物种级斜率对成鸟暴露 X 的回归系数是否 > 0（按科自助重抽样）；
  线性：ΔT = 10、20、30 的斜率是否一致。
在 EvoWM 根目录运行：python scripts/thermal/eval_thermal_v1.py [--ckpt-root ...] [--trees ...] [--seeds ...] [--folds ...]
输出 outputs/thermal_v1/results.json、report.md。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from evowm.eval.pooled import boot_mean, strip_boot  # noqa: E402
from evowm.eval.runs import N_FOLDS, ckpt_action_mode, run_all  # noqa: E402

DESIGN = ROOT / "docs/design/热阀与落震_反事实闭环_设计_2026-10-28.md"
TABLE = "data/processed/species_traits_clim.csv"
DEVMODE = ROOT / "data/raw/devmode/devmode_families.csv"
COLS = ["m_tarsus", "k_tarsometatarsus", "k_tibiotarsus", "k_femur", "m_mass"]
DELTAS = [10, 20, 30]
MAIN_DT = 20
N_BOOT = 10000
# 校准参照：PGLS 的点估计与 95% 区间（每 10 °C，log10），来源见设计第 4 节
PGLS_REF = {
    "nonpass_exposed/m_tarsus": (0.0125, 0.0082, 0.0168, "发育方式_结果_2026-10-26 主模型，暴露组"),
    "nonpass_nest/m_tarsus": (0.0034, -0.0003, 0.0071, "发育方式_结果_2026-10-26 主模型，巢内组"),
    "passerine/m_tarsus": (-0.0057, -0.0076, -0.0039, "M1M2 结果 2026-10-21 事后，雀形目 5,584 种"),
    "anatidae/k_tarsometatarsus": (0.0137, 0.0010, 0.0265, "成鸟暴露与鸭科复核_结果_2026-10-27，Watanabe 90 种"),
    "anatidae/k_femur": (0.0029, -0.0058, 0.0117, "同上"),
    "passerine/k_tarsometatarsus": (-0.0008, -0.0041, 0.0025, "M1M2 结果，Skelevision 1,727 种（含 86 鸭科）"),
    "passerine/k_femur": (0.0026, 0.0005, 0.0048, "同上"),
}
GROUPS_ZH = {"all": "全部", "nonpass_exposed": "非雀形目暴露组（P + SP）", "nonpass_nest": "非雀形目巢内组（SA + A）",
             "nonpass": "非雀形目", "passerine": "雀形目", "anatidae": "鸭科"}
COLS_ZH = {"m_tarsus": "外测跗跖", "k_tarsometatarsus": "跗跖骨", "k_tibiotarsus": "胫跗骨", "k_femur": "股骨", "m_mass": "体重"}


def run_thermal_task(task: dict) -> dict:
    import torch
    torch.set_num_threads(1)
    from evowm.eval.recovery import Level1, apply_probe, fit_probe, load_world_model, sample_windows
    from evowm.eval.runs import fold_datasets
    root = Path(task["root"])
    ds_tr, ds_va = fold_datasets(root, task["table_rel"], task["tree_file"], task["fold"], "change")
    model = load_world_model(Path(task["ckpt"]), root / "hjepa")
    L1 = Level1(model)
    z_all = L1.encode(ds_tr.pix)
    rows_tr = np.flatnonzero(ds_tr.rows_mask)
    probes = fit_probe(z_all, ds_tr.S, rows_tr)
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
    j = [ds_tr.state_cols.index(c) for c in COLS]
    pred = {dt: apply_probe(probes, L1.predict(emb, np.stack(a)[:, :1]))[:, 0, j] for dt, a in acts.items()}
    keep = np.array(keep)
    out = {"key": task["key"], "fold": task["fold"], "target": W.rows[keep, 1].astype(int), "cols": COLS,
           "slope": {dt: ((pred[0] - pred[dt]) / (dt / 10.0))[keep].astype(np.float64) for dt in DELTAS}}
    return out


def tasks_for(ckpt_root: Path, trees, seeds, folds):
    tasks = []
    for t in trees:
        tree_file = f"data/processed/trees/hackett_t{t:03d}.nwk"
        for s in seeds:
            for k in folds:
                ck = ckpt_root / f"hackett_t{t:03d}_s{s}_f{k}" / "evowm_l1_object.ckpt"
                if not ck.exists():
                    sys.exit(f"缺少模型 {ck}")
                if ckpt_action_mode(ck) != "change":
                    sys.exit(f"{ck} 的动作方式不是 change")
                tasks.append({"root": str(ROOT), "table_rel": TABLE, "tree_file": tree_file, "fold": k, "seed": s,
                              "ckpt": str(ck), "key": f"hackett_t{t:03d}_s{s}"})
    return tasks


def ci_str(m):
    return f"{m['mean']:+.4f} [{m['ci95'][0]:+.4f}, {m['ci95'][1]:+.4f}]"


def boot_ols(x, y, clusters, n_boot=N_BOOT, seed=0):
    """y 对 x 的 OLS 斜率，按簇自助重抽样。"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, cl = x[ok], y[ok], np.asarray(clusters, dtype=object)[ok].astype(str)
    uniq, inv = np.unique(cl, return_inverse=True)
    def slope(w):
        ww = w[inv]
        xm, ym = np.average(x, weights=ww), np.average(y, weights=ww)
        return float(np.sum(ww * (x - xm) * (y - ym)) / np.sum(ww * (x - xm) ** 2))
    point = slope(np.ones(len(uniq)))
    rng = np.random.default_rng(seed)
    boot = np.array([slope(np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq)).astype(float))
                     for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"slope": point, "ci95": [float(lo), float(hi)], "n": int(ok.sum()), "n_clusters": int(len(uniq))}


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
    print(f"评估 {len(tasks)} 个（模型，折）")
    runs = run_all(tasks, workers=args.workers, fn=run_thermal_task)
    tgt = np.concatenate([r["target"] for r in runs])
    S = {dt: np.concatenate([r["slope"][dt] for r in runs]) for dt in DELTAS}
    res = {"n_runs": len(runs), "n_pred": int(len(tgt)), "deltas": DELTAS, "main_dt": MAIN_DT, "cols": COLS, "groups": {},
           "calibration": {}, "gradient": {}, "linearity": {}}

    # ---- 分组均值 ----
    for g, mask in groups.items():
        sel = mask[tgt]
        res["groups"][g] = {}
        for j, c in enumerate(COLS):
            sel_c = sel & (has_k[tgt] if c.startswith("k_") else True)
            for dt in DELTAS:
                if sel_c.sum() == 0:
                    continue
                m = strip_boot(boot_mean(S[dt][sel_c, j], fam[tgt][sel_c]))
                res["groups"][g].setdefault(c, {})[dt] = m

    # ---- 校准 ----
    for key, (pt, lo, hi, src) in PGLS_REF.items():
        g, c = key.split("/")
        m = res["groups"].get(g, {}).get(c, {}).get(MAIN_DT)
        if m is None:
            continue
        res["calibration"][key] = {"wm": m["mean"], "wm_ci95": m["ci95"], "pgls": pt, "pgls_ci95": [lo, hi],
                                   "inside": bool(lo <= m["mean"] <= hi), "same_sign": bool(np.sign(m["mean"]) == np.sign(pt)),
                                   "source": src}
    main4 = ["nonpass_exposed/m_tarsus", "nonpass_nest/m_tarsus", "passerine/m_tarsus", "anatidae/k_tarsometatarsus"]
    inside = sum(res["calibration"][k]["inside"] for k in main4 if k in res["calibration"])
    signs = all(res["calibration"][k]["same_sign"] for k in main4 if k in res["calibration"])
    res["calibration"]["verdict"] = {"n_inside_of_4": int(inside), "all_same_sign": bool(signs),
                                     "pass": bool(inside >= 3 and signs)}

    # ---- 梯度：物种级斜率对 X ----
    df = pd.DataFrame({"tip": tgt, "s": S[MAIN_DT][:, COLS.index("m_tarsus")]})
    sp = df.groupby("tip")["s"].mean()
    rows = sp.index.to_numpy()
    for g in ["nonpass", "passerine"]:
        sel = groups[g][rows] & np.isfinite(X[rows])
        r_ = boot_ols(X[rows][sel], sp.to_numpy()[sel], fam[rows][sel])
        r_["pass"] = bool(r_["ci95"][0] > 0)
        res["gradient"][g] = r_
    res["gradient"]["verdict"] = {"both": bool(all(res["gradient"][g]["pass"] for g in ["nonpass", "passerine"])),
                                  "any": bool(any(res["gradient"][g]["pass"] for g in ["nonpass", "passerine"]))}

    # ---- 线性 ----
    for g in ["nonpass_exposed", "passerine"]:
        res["linearity"][g] = {dt: res["groups"][g]["m_tarsus"][dt]["mean"] for dt in DELTAS}

    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- 报告 ----
    L = ["# 热阀反事实 v1：世界模型读数", "",
         f"设计 `{DESIGN.relative_to(ROOT)}`。{len(runs)} 个（模型，折），{len(tgt)} 个预测（留出科窗口，上下文 1 种）。",
         "斜率 = (观测动作预测 − 气候减 ΔT 的预测) / (ΔT / 10)，即模型给出的每升高 10 °C 的 log10 变化；0.0043 = 1%。",
         "方括号为按科自助重抽样 10,000 次的 95% 区间。k_ 开头的读数只统计有骨长观测的目标种。", "",
         f"## 校准（ΔT = {MAIN_DT}）：{'通过' if res['calibration']['verdict']['pass'] else '不通过'}"
         f"（4 项里 {inside} 项落在 PGLS 区间内，方向{'全对' if signs else '有错'}）", "",
         "| 读数 | 世界模型 | PGLS 参照 | 在区间内 | 来源 |", "|---|---|---|---|---|"]
    for key, v in res["calibration"].items():
        if key == "verdict":
            continue
        g, c = key.split("/")
        L.append(f"| {GROUPS_ZH[g]}，{COLS_ZH[c]} | {v['wm']:+.4f} [{v['wm_ci95'][0]:+.4f}, {v['wm_ci95'][1]:+.4f}] | "
                 f"{v['pgls']:+.4f} [{v['pgls_ci95'][0]:+.4f}, {v['pgls_ci95'][1]:+.4f}] | {'是' if v['inside'] else '否'} | {v['source']} |")
    gv = res["gradient"]["verdict"]
    L += ["", f"## 梯度（物种级斜率对成鸟暴露 X 的回归，ΔT = {MAIN_DT}）：" +
          ("两组都为正，模型自己找到了露天梯度" if gv["both"] else ("一组为正" if gv["any"] else "没有找到")), ""]
    for g in ["nonpass", "passerine"]:
        r_ = res["gradient"][g]
        L.append(f"- {GROUPS_ZH[g]}：X 每增加 1，斜率增加 {r_['slope']:+.4f} [{r_['ci95'][0]:+.4f}, {r_['ci95'][1]:+.4f}]"
                 f"（{r_['n']} 种，{r_['n_clusters']} 科）")
    L += ["", "## 分组读数（每升高 10 °C）", "", "| 组 | 列 | " + " | ".join(f"ΔT = {dt}" for dt in DELTAS) + " | n |",
          "|---|---|" + "---|" * len(DELTAS) + "---|"]
    for g in ["nonpass_exposed", "nonpass_nest", "nonpass", "passerine", "anatidae", "all"]:
        for c in COLS:
            d = res["groups"][g].get(c)
            if not d or MAIN_DT not in d:
                continue
            L.append(f"| {GROUPS_ZH[g]} | {COLS_ZH[c]} | " + " | ".join(ci_str(d[dt]) if dt in d else "" for dt in DELTAS) +
                     f" | {d[MAIN_DT]['n']} |")
    L += ["", "## 读法", "",
          "- 校准通过 = 世界模型在温度轴上与 PGLS 一致，可作第二版反事实腿（C_WM）的引擎；不通过则世界模型只进附录。",
          "- 梯度两组都为正 = 没有写交互项，模型推出的温度响应仍随露天程度变。",
          "- 三个 ΔT 的斜率接近 = 模型的温度响应是线性的，与 PGLS 的线性假设一致。"]
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
