"""评估模拟恢复检验 v2，对照 docs/prereg/sim_recovery_v2.md 的判据出报告。

在 EvoWM 根目录运行：python scripts/eval_recovery_v2.py
读取 outputs/hjepa/ckpts/recovery_v2/<情景>_s<种子>_<tree1|post>/evowm_l1_object.ckpt；
逐个结果写到 outputs/recovery_v2/*.json，报告写到 outputs/recovery_v2/sim_recovery_v2.md（入库版本在 docs/results/，由本地复制）。
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_recovery import datasets  # noqa: E402
from evowm.eval.recovery import evaluate_run, load_world_model  # noqa: E402

GATE = [("effect", 3), ("effect", 4), ("null", 3), ("null", 4)]
ROBUST = [("effect", 3), ("null", 3)]
CKPT_ROOT = ROOT / "outputs/hjepa/ckpts/recovery_v2"
TREES = {"tree1": ("data/raw/birdtree/hackett_tree1.nwk", 1), "post": ("data/raw/birdtree/hackett_100.nwk", 100)}


def run_one(sc: str, s: int, tr: str) -> dict:
    name = f"{sc}_s{s}"
    rel = f"data/processed/sim/recovery_v2/{name}/species_traits.csv"
    table = pd.read_csv(ROOT / rel)
    truth = json.loads((ROOT / rel).with_name("truth.json").read_text(encoding="utf-8"))
    ds_tr, ds_va = datasets(rel, *TREES[tr])
    model = load_world_model(CKPT_ROOT / f"{name}_{tr}" / "evowm_l1_object.ckpt", ROOT / "hjepa")
    return evaluate_run(ds_tr, ds_va, model, table, truth, n_orders=8, seed=s)


def main():
    out_dir = ROOT / "outputs/recovery_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {}
    for sc, s in GATE:
        res[f"{sc}_s{s}_tree1"] = run_one(sc, s, "tree1")
    for sc, s in ROBUST:
        key = f"{sc}_s{s}_post"
        if (CKPT_ROOT / key / "evowm_l1_object.ckpt").exists():
            res[key] = run_one(sc, s, "post")
    for k, v in res.items():
        (out_dir / f"{k}.json").write_text(json.dumps(v, ensure_ascii=False, indent=2), encoding="utf-8")

    g = lambda sc, s: res[f"{sc}_s{s}_tree1"]  # noqa: E731
    c1 = {s: g("effect", s)["E1"]["m_tarsus"]["ci95"][0] > 0 for s in (3, 4)}
    c2 = {}
    for s in (3, 4):
        n, e = g("null", s)["E1"]["m_tarsus"], g("effect", s)["E1"]["m_tarsus"]
        c2[s] = (n["ci95"][0] <= 0 <= n["ci95"][1]) or (n["gain"] < 0.1 * e["gain"])
    c3 = {}
    for s in (3, 4):
        ee, nn = g("effect", s)["E2_relative"], g("null", s)["E2_relative"]
        c3[s] = (ee["spearman_rho_identifiable"] >= 0.7) and (nn["m_tarsus_mean_abs"] < 0.25 * ee["m_tarsus_mean_abs"])
    c4 = {k: 6 <= res[k]["E3"]["t_half_free_amplitude"] <= 24 for k in res if k.endswith("tree1")}
    crit = {"C1": all(c1.values()), "C2": all(c2.values()), "C3": all(c3.values()), "C4": all(c4.values())}

    yn = lambda b: "通过" if b else "未通过"  # noqa: E731
    L = ["# 模拟恢复检验 v2 结果", "",
         "判据见 `docs/prereg/sim_recovery_v2.md`。判定只用真树训练的 4 个模型（效应、零模型 × 种子 3、4）；"
         "后验树版本为稳健性结果。数值单位均为 log10；增益 = 无习性 RMSE − 有习性 RMSE，95% 置信区间按留出科自助重抽样 1,000 次。", "",
         f"**总体：{'通过' if all(crit.values()) else '未通过'}**（C1 {yn(crit['C1'])}，C2 {yn(crit['C2'])}，"
         f"C3 {yn(crit['C3'])}，C4 {yn(crit['C4'])}）", "",
         "## E1 m_tarsus 习性增益（C1、C2）", "",
         "| 模型 | 有习性 RMSE | 无习性 RMSE | 增益 | 95% CI | 体制转变对增益 | 理想预测器（转变对） |", "|---|---|---|---|---|---|---|"]
    for k, r in res.items():
        e, t = r["E1"]["m_tarsus"], r["E1_transition"]["m_tarsus"]
        L.append(f"| {k} | {e['rmse_eco']:.4f} | {e['rmse_no_eco']:.4f} | {e['gain']:+.4f} | "
                 f"[{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}] | {t['gain']:+.4f} [{t['ci95'][0]:+.4f}, {t['ci95'][1]:+.4f}] | "
                 f"{t['oracle_gain']:+.4f} |")
    L += ["", "## E2 相对腿长反事实对比（C3）", "",
          "| 模型 | 可识别组合数 | ρ（与真值） | m_tarsus 相对对比平均绝对值 | m_tarsus 相对对比（地栖 / 空中 / 涉水 / 潜水 / 水面 / 泛化） |",
          "|---|---|---|---|---|"]
    for k, r in res.items():
        e = r["E2_relative"]
        rho = e["spearman_rho_identifiable"]
        rho_s = "不适用" if rho != rho else f"{rho:.3f}"
        mt = " / ".join(f"{e['contrasts'][x]['m_tarsus']:+.3f}" for x in ["terrestrial", "aerial", "wading", "diving", "surface", "generalist"])
        L.append(f"| {k} | {e['n_identifiable']} | {rho_s} | {e['m_tarsus_mean_abs']:.4f} | {mt} |")
    L += ["", "真值偏移（L_tmt，相对栖树）：地栖 +0.08，空中 −0.15，涉水 +0.20，潜水 −0.08，水面 −0.10，泛化 +0.03；零模型全为 0。", "",
          "## E3 半衰期（C4，真值 12 Myr）", "", "| 模型 | t½（带振幅） | 振幅 | t½（不带振幅） |", "|---|---|---|---|"]
    for k, r in res.items():
        e3 = r["E3"]
        L.append(f"| {k} | {e3['t_half_free_amplitude']:.1f} | {e3['amplitude']:.2f} | {e3['t_half']:.1f} |")
    L += ["", "## 探针 R²（留出物种）", ""]
    cols = list(next(iter(res.values()))["probe_r2_val"])
    L += ["| 模型 | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for k, r in res.items():
        L.append(f"| {k} | " + " | ".join(f"{r['probe_r2_val'][c]:.3f}" for c in cols) + " |")
    L += ["", "## 判据明细", "",
          f"- C1（效应增益下限 > 0）：种子 3 {yn(c1[3])}，种子 4 {yn(c1[4])}",
          f"- C2（零模型无假阳性）：种子 3 {yn(c2[3])}，种子 4 {yn(c2[4])}",
          f"- C3（排序 ρ ≥ 0.7 且零模型对比 < 25%）：种子 3 {yn(c3[3])}，种子 4 {yn(c3[4])}",
          "- C4（t½ 在 6 到 24 Myr）：" + "，".join(f"{k} {yn(v)}" for k, v in c4.items())]
    rep = ROOT / "outputs/recovery_v2/sim_recovery_v2.md"
    rep.write_text("\n".join(L) + "\n", encoding="utf-8")
    (out_dir / "criteria.json").write_text(json.dumps({"criteria": crit, "C1": c1, "C2": c2, "C3": c3, "C4": c4},
                                                      ensure_ascii=False, indent=2), encoding="utf-8")
    print("判据：", crit)
    print(f"报告：{rep}")


if __name__ == "__main__":
    main()
