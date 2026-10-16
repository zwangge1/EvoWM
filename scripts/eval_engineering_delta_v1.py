"""工程验证 v1 第一步：用主分析的 50 个三层模型补算"栖树 → 地栖"反事实的骨段变化量 Δ
（预注册 docs/prereg/engineering_drop_v1.md 第 2 节）。

算法与主分析的 E2 相同（evowm.eval.engineering.delta_for_model）：地栖典型习性减栖树典型习性，读出单位成员平均的
三段骨与体重的 log10 之差；只保留目标单位的主要科属于雀形目的窗口；全部窗口、全部模型合并取平均，按科自助重抽样给区间。
自检：全部窗口上相对跗跖之差的平均应与主分析 E2 的地栖对比（outputs/real_units_v1/results.json）相差 < 1e-6，否则停止。
在 EvoWM 根目录运行：python scripts/eval_engineering_delta_v1.py
输出 outputs/engineering_v1/delta.json。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from evowm.eval.engineering import COLS, run_delta_task  # noqa: E402
from evowm.eval.pooled import boot_mean  # noqa: E402
from evowm.eval.runs import N_FOLDS, ckpt_action_mode, run_all  # noqa: E402

PREREG = ROOT / "docs/prereg/engineering_drop_v1.md"
TABLE = "data/processed/species_traits.csv"
N_BOOT = 10000
LEVELS = ["locomotor", "realm"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=TABLE, help="只在测试时换成模拟表（测试时不做自检）")
    ap.add_argument("--trees", type=int, nargs="+", default=[21, 41, 61, 81, 100])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(N_FOLDS)), help="只在测试时缩减")
    ap.add_argument("--ckpt-root", default=str(ROOT / "outputs/hjepa/ckpts/real_units_v1"))
    ap.add_argument("--main-results", default=str(ROOT / "outputs/real_units_v1/results.json"))
    ap.add_argument("--out", default=str(ROOT / "outputs/engineering_v1"))
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    if not PREREG.exists():
        sys.exit(f"缺少预注册 {PREREG.relative_to(ROOT)}")
    test_mode = args.table != TABLE
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_root = Path(args.ckpt_root)

    tasks = []
    for t in args.trees:
        tree_file = f"data/processed/trees/hackett_t{t:03d}.nwk"
        if not (ROOT / tree_file).exists():
            sys.exit(f"缺少 {tree_file}（先运行 scripts/extract_trees.py）")
        for s in args.seeds:
            for k in args.folds:
                ck = ckpt_root / f"hackett_t{t:03d}_s{s}_f{k}" / "evowm_hab3_object.ckpt"
                if not ck.exists():
                    sys.exit(f"缺少模型 {ck}")
                cfg = yaml.safe_load((ck.parent / "config.yaml").read_text(encoding="utf-8")) or {}
                lv = (((cfg.get("data") or {}).get("dataset") or {}).get("units") or {}).get("levels")
                if ckpt_action_mode(ck) != "change" or list(lv or []) != LEVELS:
                    sys.exit(f"{ck} 的动作方式或上层单位与三层模型不符")
                tasks.append({"root": str(ROOT), "table_rel": args.table, "tree_file": tree_file, "fold": k,
                              "seed": s, "ckpt": str(ck), "key": f"hackett_t{t:03d}_s{s}"})
    results = run_all(tasks, args.workers, fn=run_delta_task)

    table = pd.read_csv(ROOT / args.table, usecols=["family", "order"])
    f2o = table.groupby("family")["order"].agg(lambda s: s.mode().iloc[0]).to_dict()
    fam = np.concatenate([r["family"] for r in results])
    diff = np.concatenate([r["diff"] for r in results])
    rel = np.concatenate([r["rel_tarsus"] for r in results])
    key = np.concatenate([np.full(len(r["family"]), r["key"], dtype=object) for r in results])
    pas = np.array([f2o.get(f) == "Passeriformes" for f in fam])

    # 自检：与主分析 E2 的地栖对比一致
    check = {"value": float(np.mean(rel)), "n_windows": int(len(rel))}
    mres = Path(args.main_results)
    if not test_mode:
        if not mres.exists():
            sys.exit(f"缺少主分析结果 {mres}，无法自检")
        ref = float(json.loads(mres.read_text(encoding="utf-8"))["main"]["locomotor"]["E2"]["terrestrial"]["mean"])
        check.update(main=ref, abs_diff=abs(check["value"] - ref), ok=bool(abs(check["value"] - ref) < 1e-6))
        if not check["ok"]:
            sys.exit(f"自检失败：全部窗口的相对跗跖差 {check['value']:.6f}，主分析 E2 地栖对比 {ref:.6f}")
    else:
        check["note"] = "测试模式（模拟表），不做自检"

    delta = {}
    for j, c in enumerate(COLS):
        b = boot_mean(diff[pas, j], fam[pas], N_BOOT, 0)
        delta[c] = {"mean": b["mean"], "ci95": b["ci95"], "n": b["n"], "n_clusters": b["n_clusters"]}
    per_run = {k: {c: float(np.nanmean(diff[pas & (key == k), j])) for j, c in enumerate(COLS)}
               for k in sorted(set(key))}
    res = {"cols": COLS, "delta": delta, "per_run": per_run, "self_check": check,
           "n_windows_passerine": int(pas.sum()), "n_models": len(tasks), "trees": args.trees, "seeds": args.seeds,
           "table": args.table, "test_mode": test_mode}
    (out / "delta.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({c: [round(delta[c]["mean"], 5), [round(x, 5) for x in delta[c]["ci95"]]] for c in COLS},
                     ensure_ascii=False))
    print(f"自检：{check}")
    print(f"写入 {out / 'delta.json'}")


if __name__ == "__main__":
    main()
