"""热端由体型、冷端由暴露：确认检验（设计 docs/design/热端冷端分工_确认_设计_2026-10-30.md）。
在 EvoWM 根目录运行：OMP_NUM_THREADS=2 python scripts/allen/heat_cold_split_v1.py → outputs/allen_v1/heat_cold_split.md
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/allen"))
from devmode_v2 import base, load_eliason  # noqa: E402
from evowm.data.tree import read_trees  # noqa: E402
from evowm.eval.pgls import tip_vcv  # noqa: E402
from two_axes_bones_v1 import ci, excl, gfit  # noqa: E402

DESIGN = ROOT / "docs/design/热端冷端分工_确认_设计_2026-10-30.md"
OUT = ROOT / "outputs/allen_v1/heat_cold_split.md"
H = [("bio5_c", "M"), ("bio6_c", "X")]
K = [("bio5_c", "X"), ("bio6_c", "M")]
F = H + K


def run(S, C_all, pos, resp, mc):
    out = {}
    for lab, inter in (("H", H), ("K", K), ("F", F)):
        out[lab] = gfit(S, C_all, pos, resp, inter, temp=None, mc=mc)
    return out


def main():
    if not DESIGN.exists():
        sys.exit("缺少设计文档")
    d = base()
    d["X"] = d[["e_fs_ground", "e_fs_wataroundsurf", "e_fs_watbelowsurf"]].sum(axis=1, min_count=3)
    tree = read_trees(ROOT / "data/processed/trees/hackett_t021.nwk", max_trees=1)[0]
    pos = {n: i for i, n in enumerate(tree.tip_names)}
    C_all = tip_vcv(tree)
    d = d[d["tip"].isin(pos)].dropna(subset=["X", "bio5_c", "bio6_c"]).reset_index(drop=True)
    sall = d[d["avonet_inferred"] == 0].dropna(subset=["m_tarsus"]).reset_index(drop=True)
    mc = float(sall["m_mass"].mean())
    rho = float(sall[["bio5_c", "bio6_c"]].corr().iloc[0, 1])
    el = d.merge(load_eliason(set(d["tip"])), on="tip", how="inner").dropna(subset=["b_tmt", "b_fem"]).reset_index(drop=True)
    jobs = [("S_all 外测跗跖（主）", sall, "m_tarsus"), ("非雀形目 外测跗跖", sall[sall["order"] != "Passeriformes"].reset_index(drop=True), "m_tarsus"),
            ("雀形目 外测跗跖", sall[sall["order"] == "Passeriformes"].reset_index(drop=True), "m_tarsus"),
            ("留鸟 外测跗跖", sall[sall["e_migr"] == 0].reset_index(drop=True), "m_tarsus"),
            ("Eliason 跗跖骨", el, "b_tmt"), ("Eliason 股骨", el, "b_fem"), ("Eliason 比例 r", el, "b_r")]
    L = ["# 热端由体型、冷端由暴露：确认检验", "", f"设计 `{DESIGN.relative_to(ROOT)}`。系数为每升高 10 °C 的 log10 变化，方括号为 95% 区间。"
         f"S_all 中最热月最高温与最冷月最低温的相关 r = {rho:.2f}。", ""]
    rows, verdict = [], None
    for lab, S, resp in jobs:
        r = run(S, C_all, pos, resp, mc)
        dAIC = r["H"]["aic"] - r["K"]["aic"]
        c = r["F"]["coef"]
        dh, fc, fh, dc = c["bio5_cxM"], c["bio6_cxX"], c["bio5_cxX"], c["bio6_cxM"]
        rows.append(f"| {lab} | {r['H']['n']} | {dAIC:+.1f} | {r['F']['aic'] - r['H']['aic']:+.1f} | {ci(dh)} | {ci(fc)} | {ci(fh)} | {ci(dc)} |")
        print(lab, f"H-K {dAIC:+.1f}", ci(dh), ci(fc), ci(fh), ci(dc), flush=True)
        if verdict is None:
            pattern = dh[0] > 0 and excl(dh) and fc[0] > 0 and excl(fc) and not excl(fh) and not excl(dc)
            if dAIC <= -4 and pattern:
                verdict = "分工成立"
            elif dAIC <= -4:
                verdict = "部分成立（H 优于 K，但全模型的显著性格局不完整）"
            else:
                verdict = "不成立（H 不比 K 好）"
    L += [f"## 主判定（S_all 外测跗跖）：{verdict}", "",
          "| 样本 | n | AIC H − K | AIC F − H | F：热端 × 体型 $d_h$ | F：冷端 × 暴露 $f_c$ | F：热端 × 暴露 $f_h$ | F：冷端 × 体型 $d_c$ |",
          "|---|---|---|---|---|---|---|---|"] + rows
    OUT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
