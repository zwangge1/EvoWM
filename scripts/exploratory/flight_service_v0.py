"""飞行服务腿长，里程碑 1：预测 1 的粗检验（设计 docs/design/飞行服务腿长_里程碑1_2026-10-21.md）。探索性分析。

AVONET 实测形态的物种（约 9,500 种），Hackett 第 21 棵树。模型都是 PGLS，Pagel λ 在模型 A 上估一次，之后沿用。
- 模型 A：T ~ M + HWI + 协变量（复现 Xu 2023 的 HWI 效应）。
- 模型 B：T − 2β·logWL ~ 同上，β ∈ {0.31, 0.5}（扣掉起跳要求随翼载荷变化的部分）。
- 模型 C：T ~ M + logWL + HWI + 协变量（logWL 系数自由拟合）；再按起飞方式分组各做一次。
logWL = M − W − S（翼载荷代理，常数省略）。协变量：栖境密度、迁徙等级、生活方式 5 类（栖树为基准）。
判据写在设计里，报告末尾自动判定。结果 outputs/flight_service_v0/report.md、coef.csv。
在 EvoWM 根目录运行：python scripts/exploratory/flight_service_v0.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cholesky, solve_triangular

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from evowm.data.tree import read_trees  # noqa: E402
from evowm.eval.pgls import pgls_lambda, tip_vcv  # noqa: E402
from evowm.sim.regimes import LIFESTYLE, regime_lifestyle  # noqa: E402

OUT = ROOT / "outputs/flight_service_v0"
BETAS = (0.31, 0.5)
BASE_LIFE = "insessorial"
JUMP = ("insessorial", "terrestrial", "generalist")        # 跳跃起飞组；非跳跃组 = aquatic, aerial
N_BOOT = 500


def load() -> pd.DataFrame:
    t = pd.read_csv(ROOT / "data/processed/species_traits.csv")
    cols = ["m_tarsus", "m_mass", "m_wing", "m_secondary1", "m_hwi"]
    d = t[["tip", "family", "order", "avonet_inferred", "e_habdens", "e_migr"] + cols].copy()
    for c in cols:
        d.loc[d[c] <= -0.5, c] = np.nan                        # 缺测记为 −1
    d["life"] = regime_lifestyle(t).astype(str)
    ok = d[cols + ["e_habdens", "e_migr"]].notna().all(axis=1) & (d["avonet_inferred"] == 0) & d["life"].isin(LIFESTYLE)
    d = d.loc[ok].reset_index(drop=True)
    d["logWL"] = d["m_mass"] - d["m_wing"] - d["m_secondary1"]
    d["jump"] = d["life"].isin(JUMP)
    return d


def design(d: pd.DataFrame, with_wl: bool) -> tuple[np.ndarray, list]:
    names = ["intercept", "M"] + (["logWL"] if with_wl else []) + ["HWI", "habdens", "migr"]
    cols = [np.ones(len(d)), d["m_mass"].to_numpy()] + ([d["logWL"].to_numpy()] if with_wl else []) + \
           [d["m_hwi"].to_numpy(), d["e_habdens"].to_numpy(float), d["e_migr"].to_numpy(float)]
    for c in LIFESTYLE:
        if c != BASE_LIFE:
            names.append(f"life_{c}")
            cols.append((d["life"] == c).to_numpy(float))
    return np.column_stack(cols), names


class GLS:
    """固定 λ 的 PGLS：一次 Cholesky，多次回归。"""

    def __init__(self, C: np.ndarray, lam: float):
        n = len(C)
        V = C * lam
        V[np.arange(n), np.arange(n)] = np.diag(C)
        self.L = cholesky(V, lower=True, overwrite_a=True, check_finite=False)
        self.n = n

    def fit(self, y: np.ndarray, X: np.ndarray, names: list) -> dict:
        z = solve_triangular(self.L, y, lower=True, check_finite=False)
        Z = solve_triangular(self.L, X, lower=True, check_finite=False)
        beta, *_ = np.linalg.lstsq(Z, z, rcond=None)
        r = z - Z @ beta
        s2 = float(r @ r) / (self.n - X.shape[1])
        se = np.sqrt(np.diag(np.linalg.inv(Z.T @ Z)) * s2)
        return {k: (float(b), float(e)) for k, b, e in zip(names, beta, se)}


def family_boot(d: pd.DataFrame, gls: GLS, ys: dict, Xs: dict, seed: int = 0) -> dict:
    """按科重抽样（科内整体抽）：对每个模型给 HWI 与 logWL 系数的 95% 区间。用 GLS 的白化矩阵近似（重抽样行）。"""
    rng = np.random.default_rng(seed)
    fams, inv = np.unique(d["family"].to_numpy(), return_inverse=True)
    out = {k: {"HWI": [], "logWL": []} for k in ys}
    W = {k: solve_triangular(gls.L, X, lower=True, check_finite=False) for k, (X, _) in Xs.items()}
    zs = {k: solve_triangular(gls.L, y, lower=True, check_finite=False) for k, y in ys.items()}
    names = {k: Xs[k][1] for k in Xs}
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(fams), len(fams))
        w = np.bincount(pick, minlength=len(fams))[inv].astype(float)
        sw = np.sqrt(w)
        for k in ys:
            Zw, zw = W[k] * sw[:, None], zs[k] * sw
            b, *_ = np.linalg.lstsq(Zw, zw, rcond=None)
            nm = names[k]
            out[k]["HWI"].append(b[nm.index("HWI")])
            if "logWL" in nm:
                out[k]["logWL"].append(b[nm.index("logWL")])
    return {k: {q: (np.percentile(v, [2.5, 97.5]).tolist() if v else None) for q, v in x.items()} for k, x in out.items()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    d = load()
    tree = read_trees(ROOT / "data/processed/trees/hackett_t021.nwk", max_trees=1)[0]
    pos = {n: i for i, n in enumerate(tree.tip_names)}
    rows = np.array([pos[x] for x in d["tip"]])
    C = tip_vcv(tree)[np.ix_(rows, rows)]
    print(f"物种 {len(d)}；C 用时 {time.time() - t0:.0f}s", flush=True)

    XA, nA = design(d, with_wl=False)
    XC, nC = design(d, with_wl=True)
    T, M = d["m_tarsus"].to_numpy(), d["m_mass"].to_numpy()
    base = pgls_lambda(T, XA, C, nA)
    lam = base["lambda"]
    print(f"模型 A λ = {lam:.3f}，用时 {time.time() - t0:.0f}s", flush=True)
    gls = GLS(C, lam)
    fits = {"A": gls.fit(T, XA, nA), "C": gls.fit(T, XC, nC)}
    ys = {"A": T, "C": T}
    Xs = {"A": (XA, nA), "C": (XC, nC)}
    for b in BETAS:
        k = f"B{b}"
        ys[k] = T - 2 * b * d["logWL"].to_numpy()
        Xs[k] = (XA, nA)
        fits[k] = gls.fit(ys[k], XA, nA)
    boot = family_boot(d, gls, ys, Xs)
    # 分组（固定同一个 λ，各组自己的协方差子块）
    grp = {}
    for name, mask in (("跳跃起飞组", d["jump"].to_numpy()), ("非跳跃组", ~d["jump"].to_numpy())):
        idx = np.flatnonzero(mask)
        g = GLS(C[np.ix_(idx, idx)], lam)
        dd = d.iloc[idx]
        X, nm = design(dd, with_wl=True)
        keep = [i for i, c in enumerate(nm) if not c.startswith("life_") or X[:, i].sum() >= 10]
        grp[name] = (len(idx), g.fit(T[idx], X[:, keep], [nm[i] for i in keep]))
    print(f"全部拟合完成，用时 {time.time() - t0:.0f}s", flush=True)

    # 报告
    b0 = fits["A"]["HWI"][0]
    L = ["# 飞行服务腿长，里程碑 1：预测 1 的粗检验", "",
         "设计 `docs/design/飞行服务腿长_里程碑1_2026-10-21.md`；脚本 `scripts/exploratory/flight_service_v0.py`。探索性分析。", "",
         "## 先说数字是什么意思", "",
         "- 响应是跗跖长（log10）。**模型 A** 复现 Xu 2023：HWI 越高跗跖越短应为负系数。",
         "- **模型 B** 先从跗跖长里扣掉起跳要求随翼载荷变化的部分（2β·logWL），再看 HWI 还剩多少效应。HWI 系数缩小，说明那部分效应是起跳力学。",
         "- **模型 C** 让数据自己给翼载荷的系数；物理预期是正的，大小在 2β 即 0.62 到 1.0 之间。",
         "- 系数后括号为 PGLS 标准误；方括号为按科重抽样 500 次的 95% 区间。生活方式以栖树为基准。", "",
         f"## 运行概况", "", f"- 物种 {len(d)}（AVONET 实测、形态与协变量齐全）；Hackett 第 21 棵树；模型 A 的 λ = {lam:.3f}，各模型沿用。",
         f"- 生活方式：" + "、".join(f"{c} {int((d.life == c).sum())}" for c in LIFESTYLE) + "。", ""]
    L += ["## 表 1. 各模型的系数", "", "| 系数 | 模型 A（原始） | " + " | ".join(f"模型 B β={b}" for b in BETAS) + " | 模型 C（翼载荷自由） |",
          "|---|---|" + "---|" * (len(BETAS) + 1)]
    for c in nC:
        cells = []
        for k in ["A"] + [f"B{b}" for b in BETAS] + ["C"]:
            f = fits[k]
            if c in f:
                s = f"{f[c][0]:+.4f} ({f[c][1]:.4f})"
                if c in ("HWI", "logWL") and boot[k].get(c):
                    lo, hi = boot[k][c]
                    s += f" [{lo:+.4f}, {hi:+.4f}]"
                cells.append(s)
            else:
                cells.append("·")
        L.append(f"| {c} | " + " | ".join(cells) + " |")
    L += ["", "## 表 2. 起飞方式分组（模型 C）", "", "| 组 | 物种 | logWL | HWI | M |", "|---|---|---|---|---|"]
    for name, (n, f) in grp.items():
        L.append(f"| {name} | {n} | {f['logWL'][0]:+.4f} ({f['logWL'][1]:.4f}) | {f['HWI'][0]:+.4f} ({f['HWI'][1]:.4f}) | "
                 f"{f['M'][0]:+.4f} ({f['M'][1]:.4f}) |")
    L += ["", "## 判据（运行前定下）", ""]
    ratios = {b: fits[f"B{b}"]["HWI"][0] / b0 if b0 != 0 else np.nan for b in BETAS}
    wl = fits["C"]["logWL"]
    wl_lo, wl_hi = boot["C"]["logWL"]
    L += [f"- 模型 A 的 HWI 系数 b0 = {b0:+.4f}。扣除后与 b0 之比：" +
          "；".join(f"β={b}：{fits[f'B{b}']['HWI'][0]:+.4f}，比值 {ratios[b]:.2f}" for b in BETAS) + "。",
          f"- 模型 C 的 logWL 系数 {wl[0]:+.4f}，按科重抽样区间 [{wl_lo:+.4f}, {wl_hi:+.4f}]（物理预期 0.62 到 1.0）。"]
    halved = all(abs(ratios[b]) <= 0.5 for b in BETAS) and wl[0] > 0 and wl_lo > 0
    failed = all(abs(ratios[b]) >= 0.75 for b in BETAS) or wl[0] <= 0
    verdict = "预测 1 成立：HWI 效应至少减半，翼载荷系数为正。" if halved else (
        "预测 1 不成立：HWI 效应基本不被起跳要求吸收，本线停止，另议。" if failed else
        "介于两者之间：部分吸收。继续，但故事改为\"部分吸收\"，并把翼载荷的校准提前。")
    L += ["", f"**{verdict}**", "",
          "## 描述（不判定）", "",
          f"- 异速指数（M 的系数）：模型 A {fits['A']['M'][0]:.3f}，" +
          "，".join(f"模型 B β={b} {fits[f'B{b}']['M'][0]:.3f}" for b in BETAS) + f"，模型 C {fits['C']['M'][0]:.3f}。",
          f"- 预测 4 粗版：跳跃起飞组 logWL 系数 {grp['跳跃起飞组'][1]['logWL'][0]:+.3f}，非跳跃组 {grp['非跳跃组'][1]['logWL'][0]:+.3f}；"
          "预期前者更大。"]
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    rows_out = []
    for k, f in fits.items():
        for c, (b, se) in f.items():
            rows_out.append({"model": k, "coef": c, "beta": b, "se": se})
    pd.DataFrame(rows_out).to_csv(OUT / "coef.csv", index=False)
    print(f"写入 {OUT / 'report.md'}，总用时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
