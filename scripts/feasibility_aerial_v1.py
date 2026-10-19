"""可行性检查：变成空中型的独立起源，第 1 关（标签）与第 0 关（时间）。
设计与判据：docs/design/可行性_空中型起源_2026-10-19.md。不用模型，只用物种主表与系统发育树。探索性分析。
结果写到 outputs/feasibility_aerial_v1/：report.md、origins.csv。
在 EvoWM 根目录运行：python scripts/feasibility_aerial_v1.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evowm.data.tree import read_trees  # noqa: E402
from evowm.data.units import LOCOMOTOR, habit_codes  # noqa: E402

OUT = ROOT / "outputs/feasibility_aerial_v1"
TREES = [21, 41, 61, 81, 100]
MAIN_TREE = 21
TARGET = ["Caprimulgidae", "Trochilidae"]
SECONDARY = ["Accipitridae", "Falconidae", "Laridae"]
FAM_ZH = {"Caprimulgidae": "夜鹰科", "Trochilidae": "蜂鸟科", "Accipitridae": "鹰科", "Falconidae": "隼科", "Laridae": "鸥科",
          "Apodidae": "雨燕科", "Hirundinidae": "燕科", "Hemiprocnidae": "凤头雨燕科", "Procellariidae": "鹱科",
          "Hydrobatidae": "海燕科", "Fregatidae": "军舰鸟科", "Phaethontidae": "鹲科", "Sulidae": "鲣鸟科",
          "Artamidae": "燕鵙科", "Strigidae": "鸱鸮科", "Coraciidae": "佛法僧科", "Pelecanidae": "鹈鹕科",
          "Meropidae": "蜂虎科", "Stercorariidae": "贼鸥科", "Glareolidae": "燕鸻科", "Tytonidae": "草鸮科"}
TRAITS = {"m_hwi": "HWI", "m_kipps": "Kipp 距离", "m_wing": "相对翅长", "m_tarsus": "相对跗跖"}
SECOND = ["m_kipps", "m_wing"]
CONFIRM = 0.2
NB = 2000


def relative_traits(T: pd.DataFrame) -> pd.DataFrame:
    """长度类减去体重通路（全部物种上的 OLS 斜率）；HWI 不扣。"""
    out = {}
    for c in TRAITS:
        if c == "m_hwi":
            out[c] = T[c]
            continue
        ok = T[c].notna() & T["m_mass"].notna()
        b = np.polyfit(T.loc[ok, "m_mass"], T.loc[ok, c], 1)[0]
        out[c] = T[c] - b * T["m_mass"]
    return pd.DataFrame(out)


def origins(tree, T: pd.DataFrame, R: pd.DataFrame, aerial: np.ndarray, known: np.ndarray) -> pd.DataFrame:
    """最大全空中型子树（起源）及其姐妹支的平均、变化量、分开时间。"""
    row = {n: i for i, n in enumerate(T["tip"])}
    N = len(tree.parent)
    H = tree.depth[tree.tip_nodes].max()
    tip_row = np.full(N, -1)
    for v, nm in zip(tree.tip_nodes, tree.tip_names):
        tip_row[v] = row.get(nm, -1)
    st = np.full(N, -1)
    for v in tree.tip_nodes:
        i = tip_row[v]
        if i >= 0 and known[i]:
            st[v] = int(aerial[i])
    n_lab, n_air = np.zeros(N), np.zeros(N)
    for v in range(N - 1, -1, -1):          # 子节点编号大于父节点
        if st[v] >= 0:
            n_lab[v] += 1
            n_air[v] += st[v]
        p = tree.parent[v]
        if p >= 0:
            n_lab[p] += n_lab[v]
            n_air[p] += n_air[v]
    allair = (n_lab > 0) & (n_air == n_lab)

    def tip_rows(v):
        out, stack = [], [v]
        while stack:
            u = stack.pop()
            ch = tree.children(u)
            if len(ch) == 0:
                if tip_row[u] >= 0:
                    out.append(tip_row[u])
            else:
                stack.extend(ch)
        return out

    fs = T["e_fs_aerial"].to_numpy(float)
    recs = []
    for v in range(N):
        if not allair[v] or (tree.parent[v] >= 0 and allair[tree.parent[v]]):
            continue
        p = tree.parent[v]
        if p < 0:
            continue
        clade = [i for i in tip_rows(v) if known[i] and aerial[i]]
        sis = [i for i in set(tip_rows(p)) - set(tip_rows(v)) if known[i] and not aerial[i]]
        if not clade or not sis:
            continue
        rec = {"family": T["family"].iloc[clade].mode().iloc[0], "order": T["order"].iloc[clade].mode().iloc[0],
               "n_clade": len(clade), "n_sister": len(sis), "t_myr": float(H - tree.depth[p]),
               "fs_aerial_clade": np.nanmean(fs[clade]) if np.isfinite(fs[clade]).any() else np.nan,
               "fs_aerial_sister": np.nanmean(fs[sis]) if np.isfinite(fs[sis]).any() else np.nan,
               "example": T["tip"].iloc[clade[0]]}
        for c in TRAITS:
            a, b = R[c].to_numpy(float)[clade], R[c].to_numpy(float)[sis]
            ok_a, ok_b = np.isfinite(a), np.isfinite(b)
            rec[f"s_{c}"] = b[ok_b].mean() if ok_b.any() else np.nan
            rec[f"d_{c}"] = a[ok_a].mean() - b[ok_b].mean() if ok_a.any() and ok_b.any() else np.nan
        recs.append(rec)
    d = pd.DataFrame(recs)
    d["d_fs_aerial"] = d["fs_aerial_clade"] - d["fs_aerial_sister"]
    d["confirmed"] = d["d_fs_aerial"] >= CONFIRM
    d["group"] = np.where(d["family"].isin(TARGET), "target", np.where(d["family"].isin(SECONDARY), "secondary", "other"))
    return d


def ou(X, theta, log_alpha):
    s, t = X
    return (theta - s) * (1 - np.exp(-np.exp(log_alpha) * t))


def gate0(d: pd.DataFrame, c: str, seed: int = 0) -> dict:
    """非目标科的起源拟合 OU 预期，预测目标科；返回目标科平均标准化残差及区间、各科平均残差。"""
    s, dd = f"s_{c}", f"d_{c}"
    ok = d[s].notna() & d[dd].notna() & d["t_myr"].gt(0)
    fit = d[ok & (d["group"] != "target")]
    tgt = d[ok & (d["group"] == "target")]
    if len(fit) < 10 or len(tgt) < 3:
        return {"ok": False, "n_fit": int(len(fit)), "n_target": int(len(tgt))}
    p0 = [float(np.nanmean(fit[s] + fit[dd])), np.log(0.05)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        (th, la), _ = curve_fit(ou, (fit[s].to_numpy(), fit["t_myr"].to_numpy()), fit[dd].to_numpy(), p0=p0,
                                bounds=([-np.inf, np.log(1e-4)], [np.inf, np.log(2.0)]), maxfev=20000)
    res_fit = fit[dd] - ou((fit[s].to_numpy(), fit["t_myr"].to_numpy()), th, la)
    sd = float(np.std(res_fit, ddof=2))
    z_all = (d.loc[ok, dd] - ou((d.loc[ok, s].to_numpy(), d.loc[ok, "t_myr"].to_numpy()), th, la)) / sd
    z_t = z_all[tgt.index].to_numpy()
    rng = np.random.default_rng(seed)
    bs = np.array([rng.choice(z_t, len(z_t)).mean() for _ in range(NB)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    fam = d.loc[ok, "family"]
    by_fam = pd.DataFrame({"family": fam, "z": z_all}).groupby("family")["z"].agg(["size", "mean"])
    return {"ok": True, "theta": float(th), "alpha": float(np.exp(la)), "half_life": float(np.log(2) / np.exp(la)),
            "sd": sd, "n_fit": int(len(fit)), "n_target": int(len(tgt)), "z_mean": float(z_t.mean()), "lo": float(lo),
            "hi": float(hi), "by_family": by_fam, "z_all": z_all}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    T = pd.read_csv(ROOT / "data/processed/species_traits.csv")
    R = relative_traits(T)
    codes = habit_codes(T)
    known = codes >= 0
    aerial = codes == LOCOMOTOR.index("aerial")
    per_tree, main = {}, None
    for t in TREES:
        f = ROOT / f"data/processed/trees/hackett_t{t:03d}.nwk"
        if not f.exists():
            continue
        d = origins(read_trees(f, max_trees=1)[0], T, R, aerial, known)
        d.insert(0, "tree", t)
        per_tree[t] = d
        if t == MAIN_TREE:
            main = d
    if main is None:
        sys.exit("缺少 Hackett 第 21 棵树")
    pd.concat(per_tree.values()).to_csv(OUT / "origins.csv", index=False)

    L = ["# 可行性检查：变成空中型的独立起源（第 0 关时间、第 1 关标签）", "",
         "设计与判据 `docs/design/可行性_空中型起源_2026-10-19.md`；脚本 `scripts/feasibility_aerial_v1.py`。不用模型。探索性分析。", "",
         "## 先说数字是什么意思", "",
         "- **起源**：树上一次\"变成空中型\"（最大的全空中型子树）。**姐妹支**：同一父节点下的非空中型物种，代表变化前。",
         "- **分开时间**：起源与姐妹支分开多少百万年。**变化量**：起源平均减姐妹支平均（HWI 的单位是 0 到 1，乘 100 即翅尖指数分数）。",
         "- **行为确认**：起源的空中觅食比例（EltonTraits）比姐妹支高至少 20 个百分点。",
         "- **OU 预期**：只用分开时间与起点算出的\"应该变多少\"；**标准化残差**：实际比预期多变（正）或少变（负）了几个标准差。", ""]

    # 概况
    g = main.groupby("family").agg(起源=("n_clade", "size"), 物种=("n_clade", "sum"), 分开时间中位数=("t_myr", "median"),
                                   HWI变化中位数=("d_m_hwi", "median"), 起点HWI中位数=("s_m_hwi", "median"),
                                   行为确认=("confirmed", "sum")).sort_values("起源", ascending=False)
    L += ["## 第 21 棵树上的起源", "",
          f"共 {len(main)} 个起源（有姐妹支数据的），分属 {main['family'].nunique()} 个科；行为确认 {int(main['confirmed'].sum())} 个。", "",
          "| 科 | 起源 | 物种 | 分开时间中位数（百万年） | 起点 HWI 中位数 | HWI 变化中位数 | 行为确认 |", "|---|---|---|---|---|---|---|"]
    for fam, r in g.iterrows():
        L.append(f"| {FAM_ZH.get(fam, fam)} | {int(r['起源'])} | {int(r['物种'])} | {r['分开时间中位数']:.1f} | "
                 f"{100 * r['起点HWI中位数']:.1f} | {100 * r['HWI变化中位数']:+.1f} | {int(r['行为确认'])} |")

    # 第 1 关
    tc = main[main["group"] == "target"]
    n_conf = {f: int(tc.loc[tc["family"] == f, "confirmed"].sum()) for f in TARGET}
    g1 = sum(n_conf.values()) >= 10 and all(v >= 3 for v in n_conf.values())
    L += ["", "## 第 1 关：是不是只是标签变了", "",
          "空中觅食比例之差（起源减姐妹支）的分布与行为确认数：", ""]
    for f in TARGET + SECONDARY:
        x = main.loc[main["family"] == f, "d_fs_aerial"]
        if len(x):
            L.append(f"- {FAM_ZH.get(f, f)}：{len(x)} 个起源，有觅食数据 {int(x.notna().sum())} 个，中位数 {np.nanmedian(x):+.2f}，"
                     f"行为确认 {int((x >= CONFIRM).sum())} 个")
    oth = main.loc[main["group"] == "other", "d_fs_aerial"]
    L.append(f"- 其他科：{len(oth)} 个起源，中位数 {np.nanmedian(oth):+.2f}，行为确认 {int((oth >= CONFIRM).sum())} 个")
    L += ["", f"**第 1 关：{'通过' if g1 else '不通过'}**（目标两科行为确认 {n_conf}；要求合计至少 10、每科至少 3）。", ""]

    # 第 0 关
    L += ["## 第 0 关：是不是只是时间不够", "",
          "用非目标科的起源拟合 OU 预期（共同的最优值与回拉速度），再看目标科比预期多变还是少变。", ""]
    verdict0 = None
    for label, sub in (("行为确认的起源（判定用）", main[main["confirmed"]]), ("全部起源（描述）", main)):
        L += [f"**{label}**", "", "| 读数 | 拟合用起源 | 目标科起源 | 半衰期（百万年） | 目标科平均标准化残差 [95% 区间] |", "|---|---|---|---|---|"]
        res = {}
        for c in ("m_hwi", "m_kipps", "m_wing", "m_tarsus"):
            r = gate0(sub.reset_index(drop=True), c)
            res[c] = r
            if not r["ok"]:
                L.append(f"| {TRAITS[c]} | {r['n_fit']} | {r['n_target']} | · | 样本不足 |")
                continue
            L.append(f"| {TRAITS[c]} | {r['n_fit']} | {r['n_target']} | {r['half_life']:.1f} | {r['z_mean']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] |")
        L.append("")
        if res["m_hwi"]["ok"]:
            bf = res["m_hwi"]["by_family"]
            show = [f for f in TARGET + SECONDARY + ["Apodidae", "Hirundinidae", "Hemiprocnidae", "Fregatidae", "Hydrobatidae"] if f in bf.index]
            L.append("HWI 各科平均标准化残差：" + "；".join(f"{FAM_ZH.get(f, f)} {bf.loc[f, 'mean']:+.2f}（{int(bf.loc[f, 'size'])}）" for f in show))
            L.append("")
        if label.startswith("行为确认") and g1:
            h = res["m_hwi"]
            sec = any(res[c]["ok"] and res[c]["z_mean"] < 0 for c in SECOND)
            verdict0 = bool(h["ok"] and h["z_mean"] < 0 and h["hi"] < 0 and sec)
    if verdict0 is None:
        L += ["**第 0 关：不判定**（第 1 关没有通过，按设计只作描述）。", ""]
    else:
        L += [f"**第 0 关：{'通过' if verdict0 else '不通过'}**（HWI 平均标准化残差小于 0 且区间上限小于 0，"
              "Kipp 距离或相对翅长至少一项也小于 0）。", ""]

    # 稳健性
    L += ["## 稳健性：5 棵树", "", "| 树 | 起源 | 目标科行为确认（夜鹰、蜂鸟） | HWI 目标科平均标准化残差（全部起源） |", "|---|---|---|---|"]
    for t, d in per_tree.items():
        nc = [int(d.loc[d["family"] == f, "confirmed"].sum()) for f in TARGET]
        r = gate0(d.reset_index(drop=True), "m_hwi")
        z = f"{r['z_mean']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]" if r["ok"] else "样本不足"
        L.append(f"| {t} | {len(d)} | {nc[0]}、{nc[1]} | {z} |")
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
