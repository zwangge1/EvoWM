# -*- coding: utf-8 -*-
"""稿件图 1 到图 5（matplotlib，静态出版图）。数值来自预注册结果文档；系数换算成每升高 10 °C 的百分比。"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch
import numpy as np

OUT = Path("/home/claude/paper_v1/figs")
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "Noto Sans CJK SC", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.titlesize": 9, "axes.titleweight": "bold",
                     "legend.frameon": False, "savefig.dpi": 300})
BLUE, VERM, GREEN, GOLD, GREY, LBLUE = "#0072B2", "#D55E00", "#009E73", "#B8860B", "#7A7A7A", "#79B5DD"


def pct(c):
    return (10 ** np.asarray(c, float) - 1) * 100


def forest(ax, rows, colors, bold=None, xlab="每升高 10 °C 的长度变化（%）", zero=True, label_n=True):
    """rows: (label, est, lo, hi, n)."""
    y = np.arange(len(rows))[::-1]
    for yi, (lab, est, lo, hi, n), col in zip(y, rows, colors):
        ax.plot([pct(lo), pct(hi)], [yi, yi], color=col, lw=1.6, solid_capstyle="round")
        ax.plot(pct(est), yi, "o", color=col, ms=5, mec="white", mew=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r[0]}（{r[4]:,} 种）" if label_n and r[4] else r[0] for r in rows])
    if zero:
        ax.axvline(0, color="#333", lw=0.8)
    ax.set_xlabel(xlab)
    ax.grid(axis="x", color="#e6e6e6", lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)


def leg_schematic(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.add_patch(Ellipse((2.2, 7.6), 4.0, 2.6, facecolor="#e9eef6", edgecolor="#8aa0c4", lw=0.8))
    ax.text(2.0, 7.6, "躯干与腹羽", ha="center", va="center", fontsize=7.5, color="#445")
    segs = [((3.0, 7.2), (4.0, 5.6), GREY, "股骨\n体壁内"),
            ((4.0, 5.6), (3.2, 3.4), LBLUE, "胫跗骨\n覆羽，逆流热交换"),
            ((3.2, 3.4), (3.8, 1.0), BLUE, "跗跖骨\n裸露，热交换面")]
    for (x0, y0), (x1, y1), col, lab in segs:
        ax.plot([x0, x1], [y0, y1], color=col, lw=7, solid_capstyle="round")
        ax.text(5.4, (y0 + y1) / 2, lab, va="center", fontsize=7.5, color="#222" if col != BLUE else BLUE,
                fontweight="bold" if col == BLUE else "normal")
    for dx, dy in ((-0.9, -0.6), (0.1, -0.8), (0.9, -0.5)):
        ax.plot([3.8, 3.8 + dx], [1.0, 1.0 + dy], color=BLUE, lw=2.5, solid_capstyle="round")


def fig1():
    fig = plt.figure(figsize=(7.2, 6.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.35], height_ratios=[1, 1.45], hspace=0.45, wspace=0.9)
    ax = fig.add_subplot(gs[0, 0]); leg_schematic(ax); ax.set_title("a", loc="left")
    ax = fig.add_subplot(gs[0, 1])
    rows = [("早成 P", 0.0152, 0.0098, 0.0207, 865), ("半早成 SP", 0.0112, 0.0021, 0.0203, 190), ("半晚成 SA", 0.0052, -0.0016, 0.0121, 653),
            ("晚成，非雀形目", -0.0019, -0.0064, 0.0025, 1866), ("晚成，雀形目", -0.0057, -0.0076, -0.0039, 5584)]
    forest(ax, rows, [BLUE, BLUE, GREY, GREY, VERM]); ax.set_title("b", loc="left"); ax.set_xlim(-2.5, 5.5)
    ax = fig.add_subplot(gs[1, :])
    ords = [("鹤形目", .937, .0203, .0045, .0361, 134, "P"), ("鸻形目", .967, .0146, .0074, .0218, 310, "P"), ("鸡形目", .853, .0138, .0039, .0237, 275, "P"),
            ("䳍形目", .900, .0125, -.0212, .0461, 43, "P"), ("雁形目", 1.0, .0115, .0023, .0207, 150, "P"), ("鹈形目", 1.0, .0127, -.0130, .0385, 90, "SA"),
            ("鹰形目", .584, .0068, -.0031, .0166, 225, "SA"), ("鹦形目", .119, .0043, -.0066, .0153, 334, "A"), ("雨燕目", .015, .0020, -.0101, .0141, 418, "A"),
            ("鸮形目", .566, .0018, -.0113, .0150, 165, "SA"), ("鹃形目", .350, .0012, -.0144, .0168, 129, "A"), ("夜鹰目", .164, -.0017, -.0384, .0350, 96, "SP"),
            ("䴕形目", .099, -.0054, -.0113, .0005, 402, "A"), ("佛法僧目", .589, -.0055, -.0236, .0126, 144, "A"), ("犀鸟目", .253, -.0094, -.0615, .0428, 64, "A"),
            ("鸽形目", .405, -.0110, -.0218, -.0003, 283, "A"), ("隼形目", .409, -.0174, -.0391, .0043, 60, "SA"), ("咬鹃目", .062, -.0413, -.0641, -.0186, 41, "A"),
            ("雀形目", .303, -.0057, -.0076, -.0039, 5584, "Pas")]
    colmap = {"P": BLUE, "SP": BLUE, "SA": GREY, "A": GREY, "Pas": VERM}
    offs = {"雁形目": (6, 7), "鹈形目": (6, -11), "鸻形目": (-44, 9), "䳍形目": (-40, -9), "鸡形目": (-22, -4), "鹤形目": (-20, 4),
            "鹰形目": (6, 3), "佛法僧目": (6, -8), "鸮形目": (6, 0), "鹦形目": (6, 3), "雨燕目": (-4, -12), "犀鸟目": (-14, -13),
            "鸽形目": (6, -6), "隼形目": (6, -6), "鹃形目": (6, 2), "夜鹰目": (6, 2), "䴕形目": (6, -8), "咬鹃目": (6, -2), "雀形目": (-12, -17)}
    for nm, x, b, lo, hi, n, md in ords:
        c = colmap[md]
        ax.plot([x, x], [pct(lo), pct(hi)], color=c, lw=0.8, alpha=0.35)
        ax.scatter(x, pct(b), s=10 + 2.5 * np.sqrt(n), color=c, edgecolor="white", lw=0.6, zorder=3)
        dx, dy = offs.get(nm, (6, 0))
        ax.annotate(nm, (x, pct(b)), xytext=(dx, dy), textcoords="offset points", fontsize=6.8, color="#333", va="center")
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xlim(-0.05, 1.08); ax.set_ylim(-11, 7)
    ax.set_xlabel("该目物种地面与水面觅食比例的均值"); ax.set_ylabel("跗跖对温度的斜率（% 每 10 °C）")
    ax.grid(color="#eeeeee", lw=0.5); ax.set_axisbelow(True)
    ax.text(0.01, 0.96, "19 个目，Spearman ρ = 0.65，P = 0.002；竖线为 95% 区间（部分超出坐标范围）", transform=ax.transAxes, fontsize=7, color="#444", va="top")
    ax.scatter([], [], s=40, color=BLUE, label="早成与半早成"); ax.scatter([], [], s=40, color=GREY, label="半晚成与晚成"); ax.scatter([], [], s=40, color=VERM, label="雀形目")
    ax.legend(loc="lower right", fontsize=7.5)
    ax.set_title("c", loc="left")
    fig.savefig(OUT / "fig1.png", bbox_inches="tight"); fig.savefig(OUT / "fig1.pdf", bbox_inches="tight"); plt.close(fig)


def fig2():
    data = {"Eliason\n早成鸟（259 种）": [("股骨", 0.0002, -0.0056, 0.0059), ("胫跗骨", 0.0084, 0.0000, 0.0167), ("跗跖骨", 0.0161, 0.0055, 0.0267), ("比例 r", 0.0157, 0.0081, 0.0233)],
            "Watanabe\n鸭科（86 种）": [("股骨", 0.0029, -0.0058, 0.0117), ("胫跗骨", 0.0073, -0.0022, 0.0169), ("跗跖骨", 0.0137, 0.0010, 0.0265), ("比例 r", 0.0107, 0.0033, 0.0181)],
            "Eliason\n雀形目（383 种）": [("股骨", 0.0021, -0.0027, 0.0068), ("胫跗骨", -0.0042, -0.0101, 0.0017), ("跗跖骨", -0.0051, -0.0130, 0.0028), ("比例 r", -0.0065, -0.0129, -0.0002)],
            "Skelevision\n雀形目（1,641 种）": [("股骨", 0.0026, 0.0005, 0.0048), ("胫跗骨", -0.0026, -0.0050, -0.0002), ("跗跖骨", -0.0008, -0.0041, 0.0025), ("比例 r", -0.0036, -0.0066, -0.0007)]}
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.5), sharex=True, sharey=True, gridspec_kw={"wspace": 0.25})
    for ax, (title, rows), letter in zip(axes, data.items(), "abcd"):
        cols = [GREY, LBLUE, VERM if "雀形目" in title else BLUE, VERM if "雀形目" in title else BLUE]
        forest(ax, [(r[0], r[1], r[2], r[3], 0) for r in rows], cols, label_n=False, xlab="")
        ax.set_title(f"{letter}  {title}", loc="left", fontsize=8)
    axes[0].set_xlim(-3.5, 6.5)
    fig.supxlabel("每升高 10 °C 的长度变化（%）", fontsize=8.5, y=-0.04)
    fig.savefig(OUT / "fig2.png", bbox_inches="tight"); fig.savefig(OUT / "fig2.pdf", bbox_inches="tight"); plt.close(fig)


def fig3():
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.6), gridspec_kw={"width_ratios": [1.05, 1, 1.0], "wspace": 1.25})
    ax = axes[0]
    rows = [("温度 × 发育方式，单独", 0.0083, 0.0025, 0.0140, 0), ("温度 × 发育方式，加 X 后", 0.0034, -0.0033, 0.0102, 0), ("温度 × 成鸟暴露 X", 0.0092, 0.0015, 0.0169, 0)]
    forest(ax, rows, [GREY, GREY, BLUE], label_n=False, xlab="斜率之差（% 每 10 °C）"); ax.set_title("a  非雀形目（3,230 种）", loc="left", fontsize=8); ax.set_xlim(-1.5, 4.5)
    ax = axes[1]
    cv = json.loads(Path("/home/claude/paper_v1/passerine_TX_cov.json").read_text())
    x = np.linspace(0, 1, 101)
    s = cv["T"] + cv["TxX"] * x
    se = np.sqrt(cv["var_T"] + x ** 2 * cv["var_TxX"] + 2 * x * cv["cov"])
    ax.fill_between(x, pct(s - 1.96 * se), pct(s + 1.96 * se), color=VERM, alpha=0.18, lw=0)
    ax.plot(x, pct(s), color=VERM, lw=1.8)
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xlabel("成鸟暴露 X（地面 + 水面觅食比例）"); ax.set_ylabel("跗跖对温度的斜率（% 每 10 °C）")
    ax.set_title("b  雀形目（4,882 种）", loc="left", fontsize=8); ax.set_ylim(-3, 1.2)
    ax.grid(color="#e6e6e6", lw=0.5); ax.set_axisbelow(True)
    ax = axes[2]
    rows = [("早成的地栖与水生鸟", 0.0161, 0.0055, 0.0267, 0), ("半早成、半晚成", 0.0052, -0.0016, 0.0121, 0), ("晚成非雀形目", -0.0019, -0.0064, 0.0025, 0),
            ("地面觅食的雀形目", -0.0005, -0.0036, 0.0026, 0), ("树栖与空中的雀形目", -0.0076, -0.0101, -0.0052, 0)]
    forest(ax, rows, [BLUE, GREY, GREY, VERM, VERM], label_n=False); ax.set_title("c  按成年后腿的露天程度排列", loc="left", fontsize=8); ax.set_xlim(-3, 7)
    fig.savefig(OUT / "fig3.png", bbox_inches="tight"); fig.savefig(OUT / "fig3.pdf", bbox_inches="tight"); plt.close(fig)


def fig4():
    r = json.loads(Path("/home/claude/paper_v1/mass_vs_exposure.json").read_text(encoding="utf-8"))
    mc = r["m_center_log10"]; A = r["models"]["S_all"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), gridspec_kw={"width_ratios": [1, 1.15], "wspace": 0.95})
    ax = axes[0]
    lm = np.linspace(0.8, 3.6, 100)
    c1 = {k: v[0] for k, v in A["M1"]["coef"].items()}; c3 = {k: v[0] for k, v in A["M3"]["coef"].items()}
    ax.plot(lm, pct(c1["T"] + c1["TxM"] * (lm - mc)), color=GREY, lw=1.6, ls="--", label="只含体型轴")
    ax.plot(lm, pct(c3["T"] + c3["TxM"] * (lm - mc)), color=VERM, lw=1.8, label="两轴，X = 0（树栖、空中）")
    ax.plot(lm, pct(c3["T"] + c3["TxM"] * (lm - mc) + c3["TxX"]), color=BLUE, lw=1.8, label="两轴，X = 1（地面、水面）")
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["10 g", "100 g", "1 kg"])
    ax.set_xlabel("体重"); ax.set_ylabel("跗跖对年均温的斜率（% 每 10 °C）")
    ax.grid(color="#eeeeee", lw=0.5); ax.set_axisbelow(True); ax.legend(fontsize=6.8, loc="lower right")
    ax.set_title("a  8,112 种", loc="left", fontsize=8.5)
    ax = axes[1]
    temps = [("年均温", A["M3"]), ("最热月最高温", r["robust"]["bio5_c"]["M3"]), ("最冷月最低温", r["robust"]["bio6_c"]["M3"]), ("盒均值年均温", r["robust"]["bio1_box"]["M3"])]
    rows, cols = [], []
    for lab, m in temps:
        for key, nm, col in (("TxM", "体型轴", GOLD), ("TxX", "暴露轴", BLUE)):
            b, se = m["coef"][key]
            rows.append((f"{lab}，{nm}", b, b - 1.96 * se, b + 1.96 * se, 0)); cols.append(col)
    forest(ax, rows, cols, label_n=False, xlab="斜率的变化（% 每 10 °C）")
    for yv in (1.5, 3.5, 5.5):
        ax.axhline(yv, color="#dddddd", lw=0.6)
    ax.set_xlim(-1, 4.5)
    ax.set_title("b  两轴同放，四种温度变量", loc="left", fontsize=8.5)
    fig.savefig(OUT / "fig4.png", bbox_inches="tight"); fig.savefig(OUT / "fig4.pdf", bbox_inches="tight"); plt.close(fig)


def fig5():
    dec = json.loads(Path("/home/claude/evowm_dev/outputs/thermal_v1/drop_decision.json").read_text(encoding="utf-8"))
    sw = json.loads(Path("/home/claude/evowm_dev/outputs/thermal_v1/segment_swap.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6), gridspec_kw={"width_ratios": [1.15, 1.1, 0.9], "wspace": 0.55})
    doses = [10, 20, 40]
    ax = axes[0]
    ax.axhspan(-5, 5, color="#eeeeee", lw=0)
    for g, col, lab in (("exposed", BLUE, "露天组（早成鸟，643 g）"), ("arboreal", VERM, "树栖雀形目（17.5 g）")):
        d = dec[g]["diff"]
        y = [100 * d[f"C{k}"]["diff"] for k in doses]
        lo = [100 * d[f"C{k}"]["ci95"][0] for k in doses]; hi = [100 * d[f"C{k}"]["ci95"][1] for k in doses]
        ax.errorbar(doses, y, yerr=[np.subtract(y, lo), np.subtract(hi, y)], fmt="o-", color=col, lw=1.4, ms=5, capsize=2.5, label=lab)
    ax.axhline(0, color="#333", lw=0.8); ax.set_xticks(doses); ax.set_xlabel("去散热剂量 ΔT（°C）"); ax.set_ylabel("可行率之差（去散热腿 − 真实腿，百分点）", fontsize=7.5)
    ax.set_ylim(-8, 8); ax.set_title("a", loc="left"); ax.legend(fontsize=7, loc="upper left")
    ax.text(40, -5.4, "有意义的差 = 5 个百分点", fontsize=6.5, color="#666", ha="right", va="top")
    ax = axes[1]
    for g, col in (("exposed", BLUE), ("arboreal", VERM)):
        sec = dec[g]["secondary"]["a_res"]
        y = [sec[f"C{k}"]["median"] for k in doses]; lo = [sec[f"C{k}"]["ci95"][0] for k in doses]; hi = [sec[f"C{k}"]["ci95"][1] for k in doses]
        ax.errorbar(doses, y, yerr=[np.subtract(y, lo), np.subtract(hi, y)], fmt="o-", color=col, lw=1.4, ms=5, capsize=2.5)
    ax.axhline(0, color="#333", lw=0.8); ax.set_xticks(doses); ax.set_xlabel("去散热剂量 ΔT（°C）"); ax.set_ylabel("峰值合加速度之差（g）")
    ax.set_title("b", loc="left")
    ax = axes[2]
    names = ["Ct20", "Cb20", "Cf20"]; labs = ["只缩\n跗跖骨", "只缩\n胫跗骨", "只缩\n股骨"]
    y = [100 * sw["diff"][n]["diff"] for n in names]; lo = [100 * sw["diff"][n]["ci95"][0] for n in names]; hi = [100 * sw["diff"][n]["ci95"][1] for n in names]
    ax.axhspan(-5, 5, color="#eeeeee", lw=0)
    ax.errorbar(range(3), y, yerr=[np.subtract(y, lo), np.subtract(hi, y)], fmt="o", color=GOLD, ms=5, capsize=2.5)
    ax.axhline(0, color="#333", lw=0.8); ax.set_xticks(range(3)); ax.set_xticklabels(labs); ax.set_ylim(-8, 8)
    ax.set_ylabel("可行率之差（百分点）"); ax.set_title("c  同样的 7.2% 缩短放在哪一段", loc="left", fontsize=8)
    fig.savefig(OUT / "fig5.png", bbox_inches="tight"); fig.savefig(OUT / "fig5.pdf", bbox_inches="tight"); plt.close(fig)


def fig6():
    fig = plt.figure(figsize=(7.6, 3.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.3)
    ax = fig.add_subplot(gs[0]); ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.set_aspect("equal"); ax.axis("off"); ax.set_title("a", loc="left")

    def box(x, y, w, h, text, fc="#f3f6fb", ec="#2b5fa8", fs=6.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#333", shrinkA=0, shrinkB=0))

    box(0.3, 4.6, 3.0, 1.9, "物种 A 的形态\n23 个性状 + 缺失掩码\n（46 维）")
    box(3.9, 4.6, 2.4, 1.9, "编码器\nMLP\n46 → 256 → 128")
    box(0.3, 1.6, 3.0, 2.2, "动作 A → B（56 维）\n习性差 47，气候差 3\n组掩码 4，距离 2", fc="#fdf5ec", ec=GOLD)
    box(3.9, 1.6, 2.4, 2.2, "动作嵌入\nMLP")
    box(7.0, 3.0, 2.6, 2.4, "预测器\n因果 Transformer\n窗口 4，深度 3，4 头")
    box(10.0, 3.0, 1.8, 2.4, "探针\n岭回归\n→ 性状", fc="#e8f5ef", ec=GREEN)
    box(7.0, 0.4, 4.8, 1.4, "训练目标\n预测嵌入 ≈ B 的编码 + SIGReg 正则", fc="#ffffff", ec="#999", fs=6.0)
    arrow(3.3, 5.55, 3.9, 5.55); arrow(3.3, 2.7, 3.9, 2.7); arrow(6.3, 5.55, 7.0, 4.7); arrow(6.3, 2.7, 7.0, 3.7); arrow(9.6, 4.2, 10.0, 4.2)
    ax.text(0.3, 0.75, "反事实：B 的气候减 ΔT，\n其余动作不变，读两次预测之差", fontsize=6.3, color=GOLD, va="center", linespacing=1.3)

    ax = fig.add_subplot(gs[1]); ax.set_title("b", loc="left")
    rows = [("体重，露天组", -0.0163, -0.0283, -0.0069, -0.0284, -0.0474, -0.0095), ("体重，巢内组", -0.0220, -0.0364, -0.0064, -0.0429, -0.0556, -0.0302),
            ("体重，雀形目", -0.0146, -0.0176, -0.0119, -0.0150, -0.0217, -0.0084),
            ("跗跖，露天组", 0.0029, -0.0005, 0.0077, 0.0125, 0.0082, 0.0168), ("跗跖，雀形目", -0.0072, -0.0081, -0.0062, -0.0057, -0.0076, -0.0039),
            ("跗跖骨，鸭科", 0.0023, 0.0023, 0.0023, 0.0137, 0.0010, 0.0265), ("股骨，雀形目", 0.0019, 0.0008, 0.0032, 0.0026, 0.0005, 0.0048)]
    y = np.arange(len(rows))[::-1]
    for yi, (lab, w, wl, wh, p, pl, ph) in zip(y, rows):
        ax.plot([pct(pl), pct(ph)], [yi + 0.18, yi + 0.18], color=BLUE, lw=1.6); ax.plot(pct(p), yi + 0.18, "o", color=BLUE, ms=4.5, mec="white", mew=0.5)
        ax.plot([pct(wl), pct(wh)], [yi - 0.18, yi - 0.18], color=GREEN, lw=1.6); ax.plot(pct(w), yi - 0.18, "s", color=GREEN, ms=4.5, mec="white", mew=0.5)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows]); ax.axvline(0, color="#333", lw=0.8)
    ax.axhline(3.5, color="#ccc", lw=0.6, ls="--")
    ax.set_xlabel("每升高 10 °C 的变化（%）"); ax.grid(axis="x", color="#e6e6e6", lw=0.5); ax.set_axisbelow(True); ax.tick_params(axis="y", length=0)
    ax.set_xlim(-13, 8); ax.set_ylim(-0.6, 7.6)
    ax.plot([], [], "o-", color=BLUE, label="系统发育回归"); ax.plot([], [], "s-", color=GREEN, label="演化模型（扣体重）"); ax.legend(fontsize=7, loc="upper right")
    fig.savefig(OUT / "fig6.png", bbox_inches="tight"); fig.savefig(OUT / "fig6.pdf", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
    print(sorted(p.name for p in OUT.iterdir()))
