"""换系统发育树的稳健性（设计 docs/design/换树_稳健性_设计_2026-10-30.md）。
在 EvoWM 根目录运行：OMP_NUM_THREADS=2 python scripts/allen/tree_robustness_v1.py → outputs/allen_v1/tree_robustness.md、.json
"""
import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/allen"))
from devmode_v2 import base, fit as dm_fit, load_eliason  # noqa: E402
from evowm.data.tree import read_trees  # noqa: E402
from evowm.eval.pgls import tip_vcv  # noqa: E402
from mass_vs_exposure_v1 import fit as mx_fit  # noqa: E402
from migration_strata_v1 import fit as ms_fit  # noqa: E402

DESIGN = ROOT / "docs/design/换树_稳健性_设计_2026-10-30.md"
TREES = [f"hackett_t{t:03d}" for t in (21, 41, 61, 81, 100)] + [f"ericson_t{t:03d}" for t in (21, 41, 61)]


def ci(b, se):
    return f"{b:+.4f} [{b - 1.96 * se:+.4f}, {b + 1.96 * se:+.4f}]"


def main():
    if not DESIGN.exists():
        sys.exit("缺少设计文档")
    t0 = time.time()
    d = base()
    d["X"] = d[["e_fs_ground", "e_fs_wataroundsurf", "e_fs_watbelowsurf"]].sum(axis=1, min_count=3)
    meas = d[d["avonet_inferred"] == 0].dropna(subset=["m_tarsus"]).reset_index(drop=True)
    nonp = meas[meas["order"] != "Passeriformes"].reset_index(drop=True)
    sall = meas.dropna(subset=["X", "bio1_c", "bio5_c", "bio6_c"]).reset_index(drop=True)
    mc = float(sall["m_mass"].mean())
    pas = sall[sall["order"] == "Passeriformes"].reset_index(drop=True)
    el = d.merge(load_eliason(set(d["tip"])), on="tip", how="inner")
    elP = el[(el["order"] != "Passeriformes") & (el["mode"] == "P")].reset_index(drop=True)
    res = {}
    for tn in TREES:
        tree = read_trees(ROOT / f"data/processed/trees/{tn}.nwk", max_trees=1)[0]
        pos = {n: i for i, n in enumerate(tree.tip_names)}
        C_all = tip_vcv(tree)
        r = {}
        n, c = ms_fit(nonp[nonp["tip"].isin(pos)].reset_index(drop=True), C_all, pos, {"G"}, mc)
        r["devmode_diff"] = c["TxG"]
        sub = elP[elP["tip"].isin(pos)]
        for k in ("b_fem", "b_tmt", "b_r"):
            q = dm_fit(sub, C_all, pos, k)
            r[f"eliason_P_{k}"] = (q["T_b"], q["T_se"])
        s = sall[sall["tip"].isin(pos)].reset_index(drop=True)
        m = {k: mx_fit(s, C_all, pos, k, "bio1_c", mc) for k in ("M1", "M2", "M3")}
        r["aic_M2_minus_M1"] = m["M2"]["aic"] - m["M1"]["aic"]
        r["aic_M3_minus_M2"] = m["M3"]["aic"] - m["M2"]["aic"]
        r["M3_d"] = tuple(m["M3"]["coef"]["TxM"]); r["M3_f"] = tuple(m["M3"]["coef"]["TxX"])
        n, c = ms_fit(pas[pas["tip"].isin(pos)].reset_index(drop=True), C_all, pos, {"X"}, mc)
        r["pass_M2_f"] = c["TxX"]
        res[tn] = r
        print(tn, f"{time.time() - t0:.0f}s", {k: (round(v[0], 4) if isinstance(v, tuple) else round(v, 1)) for k, v in r.items()}, flush=True)
        del C_all, tree
        gc.collect()

    keys = [("devmode_diff", "非雀形目发育方式之差"), ("eliason_P_b_fem", "Eliason 早成鸟股骨"), ("eliason_P_b_tmt", "Eliason 早成鸟跗跖骨"),
            ("eliason_P_b_r", "Eliason 早成鸟比例 r"), ("M3_d", "两轴同放，体型轴 d"), ("M3_f", "两轴同放，暴露轴 f"), ("pass_M2_f", "雀形目暴露轴 f")]
    ref = res[TREES[0]]

    def excl(v):
        return v[0] - 1.96 * v[1] > 0 or v[0] + 1.96 * v[1] < 0

    changed = []
    for k, lab in keys:
        for tn in TREES[1:]:
            a, b = ref[k], res[tn][k]
            if (a[0] > 0) != (b[0] > 0) or excl(a) != excl(b):
                changed.append(f"{tn} 的{lab}")
    aic_ok = all(res[tn]["aic_M2_minus_M1"] <= -2 for tn in TREES)
    verdict = "稳健：8 棵树上所有读数的方向与显著性都与第 21 棵一致，暴露轴在每棵树上都优于体型轴" if not changed and aic_ok else \
        f"有变化：{'；'.join(changed) if changed else ''}{'' if aic_ok else '；暴露轴优于体型轴不是每棵树都成立'}"
    (ROOT / "outputs/allen_v1/tree_robustness.json").write_text(json.dumps({"verdict": verdict, "res": res}, ensure_ascii=False, indent=1), encoding="utf-8")
    L = ["# 换系统发育树的稳健性", "", f"设计 `{DESIGN.relative_to(ROOT)}`。系数为每升高 10 °C 的 log10 变化，方括号为 95% 区间。", "",
         f"## 判定：{verdict}", "", "| 读数 | " + " | ".join(TREES) + " |", "|---" * (len(TREES) + 1) + "|"]
    for k, lab in keys:
        L.append(f"| {lab} | " + " | ".join(ci(*res[tn][k]) for tn in TREES) + " |")
    L.append("| AIC M2 − M1 | " + " | ".join(f"{res[tn]['aic_M2_minus_M1']:+.1f}" for tn in TREES) + " |")
    L.append("| AIC M3 − M2 | " + " | ".join(f"{res[tn]['aic_M3_minus_M2']:+.1f}" for tn in TREES) + " |")
    (ROOT / "outputs/allen_v1/tree_robustness.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
