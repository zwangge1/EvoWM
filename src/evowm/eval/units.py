"""按习性尺度分层的评估（设计草案 docs/design/第二层设计草案_2026-10-15.md 第 5 节）。逐个（模型，折）计算，
多折、多次运行的合并与自助重抽样在脚本里做（evowm.eval.pooled）。

全部读数在观测空间（log10）：
- 单位读数：每层用训练科叶序上训练单位的真实表示拟合岭回归探针，读回单位成员的平均值。相对跗跖 = 跗跖 − b × 体重，
  b 为训练物种上跗跖对体重的斜率（与第 1 层相同）。评估时单位表示用全部成员（训练时每个单位至多抽 16 种）。
- E1（第 2 层为 H3）：留出科叶序上的连续 4 个单位，用前面的单位与单位一级的动作预测后面的单位（教师强制，历史 1 到 3，
  与第 1 层 E1 的取法相同）。有习性与无习性（动作的习性部分置零，只留距离）两种输入下的误差。
- E2（第 2 层为 H4）：反事实。与 E1 相同的窗口，取每个窗口的第 1 个单位作上下文，把下一个单位的平均习性换成某一类的
  典型值（训练单位的类别平均，只用各习性组都有数据的单位），距离用真实值；读出与基准类（栖树；栖境层为陆地）的差。
- S3 线性基线、S4 子集标记、S5 第 2 层对第 1 层（逐种连推、一步直跳）。第 3 层只作描述。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from sklearn.linear_model import Ridge

from ..data.units import D_MAX, LOCOMOTOR, REALMS

BASE = {"locomotor": "insessorial", "realm": "land"}
LABELS = {"locomotor": LOCOMOTOR, "realm": REALMS}
LEVEL_OF = {"locomotor": 2, "realm": 3}
EXTRA_TARGETS = ["m_hwi"]            # 除状态列外，单位一级还读回 HWI（第 3 层的体型读数）
H = 3                                # 各层的历史长度（配置里 history_size = 3）


class Level:
    """某一层的编码器、动作嵌入与预测器（不求梯度）。"""

    def __init__(self, model, level: int):
        self.jepa = model.get_level(level)

    @torch.no_grad()
    def encode(self, x: np.ndarray, batch: int = 65536) -> np.ndarray:
        flat = np.ascontiguousarray(x.reshape(-1, x.shape[-1]), dtype=np.float32)
        out = [self.jepa.encoder(torch.from_numpy(flat[i:i + batch])).numpy() for i in range(0, len(flat), batch)]
        return np.concatenate(out).reshape(*x.shape[:-1], -1)

    @torch.no_grad()
    def predict(self, emb: np.ndarray, act: np.ndarray, batch: int = 4096) -> np.ndarray:
        """emb (N, h, D)，act (N, h, A) → (N, h, D)，第 j 个输出对应第 j+1 个位置。"""
        out = []
        for i in range(0, len(emb), batch):
            e = torch.from_numpy(np.ascontiguousarray(emb[i:i + batch], dtype=np.float32))
            a = torch.from_numpy(np.ascontiguousarray(act[i:i + batch], dtype=np.float32))
            out.append(self.jepa.predict(e, self.jepa.action_embed(self.jepa.action_encoder(a))).numpy())
        return np.concatenate(out) if out else np.zeros((0,) + emb.shape[1:], np.float32)


def fit_probes(Z: np.ndarray, Y: np.ndarray, alpha: float = 1.0) -> list:
    """每个目标列一个岭回归，只用该列有观测的样本。"""
    probes = []
    for j in range(Y.shape[1]):
        ok = np.isfinite(Y[:, j])
        probes.append(Ridge(alpha=alpha).fit(Z[ok], Y[ok, j]))
    return probes


def apply_probes(probes, Z: np.ndarray) -> np.ndarray:
    flat = Z.reshape(-1, Z.shape[-1])
    return np.stack([p.predict(flat) for p in probes], axis=1).reshape(*Z.shape[:-1], len(probes))


def unit_means(V: np.ndarray, rows: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """单位成员的平均值（每列只在有观测的成员上平均；全缺失为 NaN）。"""
    X = V[rows]
    ok = np.isfinite(X)
    s = np.add.reduceat(np.where(ok, X, 0.0), starts, axis=0)
    n = np.add.reduceat(ok.astype(float), starts, axis=0)
    return np.where(n > 0, s / np.maximum(n, 1.0), np.nan)


def majority(codes: np.ndarray, rows: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """每个单位成员里最多的类别（科）。"""
    out = np.empty(len(starts), dtype=codes.dtype)
    for k, (s, e) in enumerate(zip(starts, ends)):
        v, c = np.unique(codes[rows[s:e]], return_counts=True)
        out[k] = v[np.argmax(c)]
    return out


@dataclass
class Order:
    rows: np.ndarray
    dist: np.ndarray
    lca: np.ndarray
    dtip: np.ndarray
    levels: dict
    z: dict = field(default_factory=dict)       # 层名 → 单位表示 (n_units, D)
    y: dict = field(default_factory=dict)       # 层名 → 单位平均值 (n_units, n_targets)
    fam: dict = field(default_factory=dict)     # 层名 → 单位的主要科（编号）


def make_orders(ds, n_orders: int, seed: int) -> list[Order]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_orders):
        rows, dist, lca, dtip = ds._order(rng)
        out.append(Order(rows, dist, lca, dtip, ds.unit_levels(rows, lca, dtip)))
    return out


def fill_units(model, orders: list[Order], z1_all: np.ndarray, V: np.ndarray, fam_codes: np.ndarray) -> None:
    """每条叶序上各层单位的表示、成员平均值与主要科。"""
    L2 = Level(model, 2)
    L3 = Level(model, 3) if model.num_levels >= 3 else None
    for o in orders:
        lo = o.levels["locomotor"]
        z2 = L2.encode(np.add.reduceat(z1_all[o.rows], lo.starts, axis=0) / (lo.ends - lo.starts)[:, None])
        o.z["locomotor"] = z2
        o.y["locomotor"] = unit_means(V, o.rows, lo.starts)
        o.fam["locomotor"] = majority(fam_codes, o.rows, lo.starts, lo.ends)
        if L3 is not None and "realm" in o.levels:
            re = o.levels["realm"]
            k = (re.sub[:, 1] - re.sub[:, 0])[:, None]
            o.z["realm"] = L3.encode(np.add.reduceat(z2, re.sub[:, 0], axis=0) / k)
            o.y["realm"] = unit_means(V, o.rows, re.starts)
            o.fam["realm"] = majority(fam_codes, o.rows, re.starts, re.ends)


def _cat(orders: list[Order], get) -> tuple[np.ndarray, np.ndarray]:
    """把各叶序上的单位数组接起来，返回 (数组, 每条叶序的偏移)。"""
    parts = [get(o) for o in orders]
    off = np.concatenate([[0], np.cumsum([len(p) for p in parts])])
    return np.concatenate(parts), off


# ---------------------------------------------------------------------------
def e1_level(model, va: list[Order], name: str, probes) -> dict:
    """留出科叶序上全部连续 H+1 个单位的窗口；返回逐预测（窗口 × 位置）的有、无习性预测与索引。"""
    L = Level(model, LEVEL_OF[name])
    ctx, act, oi, idx_all = [], [], [], []
    for i, o in enumerate(va):
        lv = o.levels[name]
        n = len(lv)
        if n < H + 1:
            continue
        idx = np.arange(n - H)[:, None] + np.arange(H + 1)[None, :]
        ctx.append(o.z[name][idx[:, :H]])
        act.append(lv.action[idx[:, :H]])
        oi.append(np.full(len(idx), i))
        idx_all.append(idx)
    ctx, act_eco, oi, idx = map(np.concatenate, (ctx, act, oi, idx_all))
    act_no = act_eco.copy()
    act_no[..., :-2] = 0.0
    pe = apply_probes(probes, L.predict(ctx, act_eco))
    pn = apply_probes(probes, L.predict(ctx, act_no))
    C = pe.shape[-1]
    return {"pred_eco": pe.reshape(-1, C), "pred_no": pn.reshape(-1, C),
            "order": np.repeat(oi, H), "prev": idx[:, :H].reshape(-1), "target": idx[:, 1:].reshape(-1),
            "position": np.tile(np.arange(1, H + 1), len(idx)), "window_start": np.repeat(idx[:, 0], H)}


def prototypes(tr: list[Order], name: str) -> dict:
    """各类的典型习性：训练单位（各习性组都有数据）的平均习性。"""
    hab, _ = _cat(tr, lambda o: o.levels[name].habit)
    g, _ = _cat(tr, lambda o: o.levels[name].g_any)
    lab, _ = _cat(tr, lambda o: o.levels[name].label)
    full = g.min(axis=1) > 0
    return {c: hab[(lab == c) & full].mean(axis=0) for c in range(len(LABELS[name])) if np.any((lab == c) & full)}


def e2_level(model, va: list[Order], name: str, probes, protos: dict) -> dict:
    """反事实：每个窗口（同 E1）的第 1 个单位作上下文，下一个单位换成各类的典型习性。返回逐窗口、逐类的预测读数。"""
    L = Level(model, LEVEL_OF[name])
    ctx, base, oi, tgt = [], [], [], []
    for i, o in enumerate(va):
        lv = o.levels[name]
        n = len(lv)
        if n < H + 1:
            continue
        t = np.arange(n - H)
        ctx.append(o.z[name][t][:, None])
        base.append((lv.habit[t], lv.col_ok[t], lv.g_any[t], lv.dist[t]))
        oi.append(np.full(len(t), i))
        tgt.append(t + 1)
    ctx = np.concatenate(ctx)
    hab, col_ok, g_any, d = (np.concatenate([b[j] for b in base]) for j in range(4))
    out = {}
    for c, p in protos.items():
        act = np.concatenate([(p[None, :] - hab) * col_ok, g_any,
                              np.log1p(d)[:, None] / np.log1p(D_MAX), d[:, None] / D_MAX], axis=1)[:, None]
        out[c] = apply_probes(probes, L.predict(ctx, act.astype(np.float32)))[:, 0]
    return {"pred": out, "order": np.concatenate(oi), "target": np.concatenate(tgt)}


def linear_units(tr: list[Order], va: list[Order], name: str, rel_tr: list, rel_va: list,
                 oi: np.ndarray, prev: np.ndarray, target: np.ndarray) -> dict:
    """单位一级的线性基线，预测目标单位的平均相对跗跖（只用紧邻的前一个单位）：
    无习性 r̂_B = μ + exp(−αd)(r_A − μ)；只用变化：再加 κ(μ_c(B) − μ_c(A))，μ_c 为训练单位各类的均值。
    α、κ 在训练科叶序的相邻单位上用最小二乘拟合（κ 限制在 [0, 1]）。"""
    n_lab = len(LABELS[name])
    r_tr = np.concatenate(rel_tr)
    lab_tr = np.concatenate([o.levels[name].label for o in tr])
    mu = float(np.nanmean(r_tr))
    mu_c = np.array([np.nanmean(r_tr[lab_tr == c]) if np.any(lab_tr == c) else mu for c in range(n_lab)])
    a_t = np.concatenate([r[:-1] for r in rel_tr])
    b_t = np.concatenate([r[1:] for r in rel_tr])
    d_t = np.concatenate([o.levels[name].dist for o in tr])
    la = np.concatenate([o.levels[name].label[:-1] for o in tr])
    lb = np.concatenate([o.levels[name].label[1:] for o in tr])

    def f(log_alpha):
        return float(np.nanmean((mu + np.exp(-np.exp(log_alpha) * d_t) * (a_t - mu) - b_t) ** 2))
    alpha = float(np.exp(minimize_scalar(f, bounds=(np.log(1e-4), np.log(2.0)), method="bounded").x))
    res = b_t - (mu + np.exp(-alpha * d_t) * (a_t - mu))
    dlt = mu_c[lb] - mu_c[la]
    ok = np.isfinite(res) & (dlt != 0)
    kappa = float(np.clip(np.sum(res[ok] * dlt[ok]) / max(float(np.sum(dlt[ok] ** 2)), 1e-12), 0.0, 1.0))

    rv = np.concatenate(rel_va)
    off = np.concatenate([[0], np.cumsum([len(r) for r in rel_va])])
    labv, _ = _cat(va, lambda o: o.levels[name].label)
    dv = np.concatenate([np.append(o.levels[name].dist, np.nan) for o in va])   # 第 k 个单位到第 k+1 个的距离
    ga, gb = off[oi] + prev, off[oi] + target
    base = mu + np.exp(-alpha * dv[ga]) * (rv[ga] - mu)
    return {"alpha": alpha, "kappa": kappa, "err_no": base - rv[gb],
            "err_change": base + kappa * (mu_c[labv[gb]] - mu_c[labv[ga]]) - rv[gb]}


def s5_l2_vs_l1(model, ds, va: list[Order], probes2, probes1, z1_all: np.ndarray, b: float, it: int, im: int) -> dict:
    """S5：预测留出运动方式单位的平均相对跗跖。第 2 层：前 3 个单位、一步；第 1 层：从目标单位前面紧邻的 3 种出发，
    （a）一步直跳到目标单位的每一种（动作为出发种到该种的习性差与进化距离），（b）沿目标单位的物种逐种连推
    （动作为真实的相邻习性差与距离，状态用预测值）；两者都对成员取平均。只取第 2 层历史满 3 个单位的目标。"""
    L1, L2 = Level(model, 1), Level(model, 2)
    rel = lambda Y: Y[..., it] - b * Y[..., im]  # noqa: E731
    l2_ctx, l2_act, tgt_true, fam, length = [], [], [], [], []
    jump_rows, jump_d, jump_unit = [], [], []
    roll = []                                  # 每个目标单位：(叶序号, p, s, e)
    u = 0
    for i, o in enumerate(va):
        lv = o.levels["locomotor"]
        for k in range(H, len(lv)):
            s, e = lv.starts[k], lv.ends[k]
            p = s - 1
            l2_ctx.append(o.z["locomotor"][k - H:k])
            l2_act.append(lv.action[k - H:k])
            tgt_true.append(rel(o.y["locomotor"][k]))
            fam.append(o.fam["locomotor"][k])
            length.append(e - s)
            m = np.minimum.accumulate(o.lca[p:e - 1])            # 出发种 p 到 s..e-1 的 LCA 深度
            dq = o.dtip[p] + o.dtip[s:e] - 2.0 * m
            for j, q in enumerate(range(s, e)):
                jump_rows.append(o.rows[[p - 2, p - 1, p, q]])
                jump_d.append([o.dist[p - 2], o.dist[p - 1], dq[j]])
                jump_unit.append(u)
            roll.append((i, p, s, e))
            u += 1
    n_units = u
    pred2 = apply_probes(probes2, L2.predict(np.stack(l2_ctx), np.stack(l2_act))[:, -1])
    l2 = rel(pred2)

    jr, jd, ju = np.stack(jump_rows), np.asarray(jump_d, np.float32), np.asarray(jump_unit)
    act = np.stack([ds.build_action(r, d)[:H] for r, d in zip(jr, jd)])
    pj = rel(apply_probes(probes1, L1.predict(z1_all[jr[:, :H]], act)[:, -1]))
    jump = np.bincount(ju, pj, n_units) / np.bincount(ju, minlength=n_units)

    # 逐种连推：所有目标单位一起按步推进
    lens = np.array([e - s for _, _, s, e in roll])
    buf = np.stack([z1_all[va[i].rows[p - 2:p + 1]] for i, p, _, _ in roll])          # (U, 3, D)
    acts = [ds.build_action(va[i].rows[p - 2:e], va[i].dist[p - 2:e - 1]) for i, p, s, e in roll]
    sums = np.zeros(n_units)
    for step in range(int(lens.max())):
        act_ = np.stack([acts[a][step:step + H] for a in np.flatnonzero(lens > step)])
        on = np.flatnonzero(lens > step)
        nxt = L1.predict(buf[on], act_)[:, -1]
        sums[on] += rel(apply_probes(probes1, nxt))
        buf[on] = np.concatenate([buf[on][:, 1:], nxt[:, None]], axis=1)
    rollout = sums / lens
    y = np.asarray(tgt_true)
    return {"err_l2": l2 - y, "err_jump": jump - y, "err_roll": rollout - y, "family": np.asarray(fam),
            "length": np.asarray(length)}


# ---------------------------------------------------------------------------
def evaluate_units(ds_tr, ds_va, model, table: pd.DataFrame, n_orders: int = 8, seed: int = 0) -> dict:
    """一个（模型，折）的上层读数。返回逐预测的数组，供多折合并。"""
    from .recovery import fit_probe
    state_cols = list(ds_tr.state_cols)
    it, im = state_cols.index("m_tarsus"), state_cols.index("m_mass")
    extra = [c for c in EXTRA_TARGETS if c in table.columns]
    V = np.column_stack([ds_tr.S.astype(float)] + [table[c].to_numpy(float) for c in extra])
    targets = state_cols + extra
    S = ds_tr.S
    rows_tr = np.flatnonzero(ds_tr.rows_mask)
    ok = rows_tr[np.isfinite(S[rows_tr, it]) & np.isfinite(S[rows_tr, im])]
    b = float(np.polyfit(S[ok, im], S[ok, it], 1)[0])       # 与第 1 层 evaluate_run 相同的异速斜率
    fam_codes, fam_names = pd.factorize(table["family"])
    fam_names = np.asarray(fam_names, dtype=object)

    z1_all = Level(model, 1).encode(ds_tr.pix)
    tr = make_orders(ds_tr, n_orders, seed)
    va = make_orders(ds_va, n_orders, seed + 1)
    fill_units(model, tr + va, z1_all, V, fam_codes)
    rel = lambda Y: Y[..., it] - b * Y[..., im]  # noqa: E731
    out = {"slope": b, "targets": targets, "levels": {}}
    probes1 = fit_probe(z1_all, S, rows_tr)                   # 第 1 层的物种探针（S5 用），与 evaluate_run 相同

    for name in model.evowm_units["levels"]:
        Ztr, _ = _cat(tr, lambda o: o.z[name])
        Ytr, _ = _cat(tr, lambda o: o.y[name])
        probes = fit_probes(Ztr, Ytr)
        Yva, off = _cat(va, lambda o: o.y[name])
        Zva, _ = _cat(va, lambda o: o.z[name])
        Yhat = apply_probes(probes, Zva)
        r2 = {}
        for j, c in enumerate(targets):
            okj = np.isfinite(Yva[:, j])
            ss = float(np.sum((Yva[okj, j] - Yva[okj, j].mean()) ** 2)) if okj.sum() >= 3 else 0.0
            r2[c] = float(1 - np.sum((Yva[okj, j] - Yhat[okj, j]) ** 2) / ss) if ss > 0 else float("nan")
        lab, _ = _cat(va, lambda o: o.levels[name].label)
        famv, _ = _cat(va, lambda o: o.fam[name])
        lens, _ = _cat(va, lambda o: o.levels[name].ends - o.levels[name].starts)

        e1 = e1_level(model, va, name, probes)
        gp, gt = off[e1["order"]] + e1["prev"], off[e1["order"]] + e1["target"]
        true = Yva[gt]
        err_eco, err_no = e1["pred_eco"] - true, e1["pred_no"] - true
        lin = linear_units(tr, va, name, [rel(o.y[name]) for o in tr], [rel(o.y[name]) for o in va],
                           e1["order"], e1["prev"], e1["target"])
        protos = prototypes(tr, name)
        e2 = e2_level(model, va, name, probes, protos)
        g2 = off[e2["order"]] + e2["target"]
        base_c = LABELS[name].index(BASE[name])
        res = {
            "probe_r2_val_units": r2, "n_units_val": int(len(Yva)),
            "E1": {"err_eco": err_eco.astype(np.float32), "err_no": err_no.astype(np.float32),
                   "rel_eco": rel(err_eco), "rel_no": rel(err_no),
                   "family": fam_names[famv[gt]], "prev_label": lab[gp], "target_label": lab[gt],
                   "target_len": lens[gt], "position": e1["position"]},
            "linear": lin,
            "E2": {"family": fam_names[famv[g2]], "rel": {}, "targets": {}},
        }
        if base_c in e2["pred"]:
            for c, pr in e2["pred"].items():
                if c == base_c:
                    continue
                diff = pr - e2["pred"][base_c]
                res["E2"]["rel"][LABELS[name][c]] = rel(diff)
                res["E2"]["targets"][LABELS[name][c]] = diff.astype(np.float32)
        if name == "locomotor":
            s5 = s5_l2_vs_l1(model, ds_va, va, probes, probes1, z1_all, b, it, im)
            s5["family"] = fam_names[s5["family"]]
            res["S5"] = s5
        out["levels"][name] = res
    return out
