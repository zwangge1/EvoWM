"""系统发育样带数据集：把系统发育树"排成轨迹"，接到 H-JEPA 的训练入口。

一条样本 = 随机旋转的深度优先叶序里一段连续的物种（相邻的是近亲）。
- 状态 pixels：该物种的形态向量（z 分数，缺失填 0）拼上缺失掩码。键名沿用上游的 pixels，
  这样 H-JEPA 模型代码不用改。
- 动作 action[t]：从第 t 种到第 t+1 种的"习性转变"。action_mode="full"（v1 到 v4、真实数据主分析 v1）：
  习性差、下一种的习性、下一种各习性组是否有数据、两种之间的进化距离。action_mode="change"
  （真实数据主分析 v2 起）：只有习性差、两种的各习性组是否都有数据、进化距离，不含下一种习性的绝对值，
  模型只能把习性当作近亲之间的变化来用（原因见 docs/results/real_main_v1_stage0.md）。最后一步没有下一种，填 0。
- 状态读数 state：原始观测值（log10 mm 等，缺失为 NaN），只供探针评估，训练不用。

返回字典的键带 _level1 后缀，与上游 stable_worldmodel.data.HDF5Dataset 一致；
高层由模型内部按步长抽样构造。

units 不为空时（按习性尺度分层，设计草案 docs/design/第二层设计草案_2026-10-15.md），每个样本另带上层单位的窗口：
同一条叶序上按运动方式（第 2 层）与栖境（第 3 层）切段（evowm.data.units），各层在单位序列上各自均匀地取连续
history+1 个单位。键：u2_pixels (T, M, P)、u2_mask (T, M)、u2_action (T, A)；u3_pixels (T, K, M, P)、
u3_mask (T, K, M)、u3_submask (T, K)、u3_action (T, A)。T = history+1，M = 每个单位至多抽的物种数，
K = 每个栖境单位至多抽的运动方式单位数；不足的位置补 0，掩码为 False。习性置零对同一样本的所有层同时生效。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .tree import leaf_order, pair_distance, read_trees, restrict

D_MAX = 220.0  # 最大进化距离（Myr），约为 BirdTree 根高的两倍，用于把距离缩放到 [0, 1]


def required_level1_steps(level_cfgs: list[dict]) -> int:
    """与上游 HDF5Dataset._setup_levels 相同的算法：最高层需要多少个第 1 层步。"""
    def need(level_idx: int) -> int:
        req = int(level_cfgs[level_idx - 1]["num_steps"])
        for upper in range(level_idx - 1, 0, -1):
            up = level_cfgs[upper]
            req = (req - 1) * int(up["frameskip"]) + int(up.get("window_size", 1))
        return req
    return max(need(i) for i in range(1, len(level_cfgs) + 1))


def holdout_families(table: pd.DataFrame, eligible: np.ndarray, frac: float, seed: int, level: str = "family",
                     stratify: str | None = None) -> set[str]:
    """随机抽整科（或整目）留出，直到留出物种数达到 frac。

    stratify="passerine" 时在雀形目与非雀形目内各自按 frac 抽，保证有骨段数据的雀形目在留出集里有足够的科。
    """
    if stratify == "passerine":
        strata = [(table["order"] == "Passeriformes").to_numpy(), (table["order"] != "Passeriformes").to_numpy()]
    elif stratify is None:
        strata = [np.ones(len(table), bool)]
    else:
        raise ValueError(f"未知的分层方式：{stratify}")
    rng = np.random.default_rng(seed)
    chosen = []
    for st in strata:
        el = eligible & st
        groups = table.loc[el, level].value_counts()
        order = rng.permutation(groups.index.to_numpy())
        target, total = frac * el.sum(), 0
        for g in order:
            if total >= target:
                break
            chosen.append(g)
            total += groups[g]
    return set(chosen)


def family_folds(table: pd.DataFrame, eligible: np.ndarray, n_folds: int, seed: int,
                 level: str = "family") -> dict[str, int]:
    """把整科分成 n_folds 份做交叉验证，每科恰好属于一份，各份物种数尽量相等。

    科先随机打乱（seed），再按物种数从大到小（同样大小保持随机顺序）依次放进当前物种最少的一份。
    """
    groups = table.loc[eligible, level].value_counts()
    order = np.random.default_rng(seed).permutation(groups.index.to_numpy())
    order = sorted(order, key=lambda g: -groups[g])
    load = np.zeros(n_folds)
    fold_of = {}
    for g in order:
        k = int(np.argmin(load))
        fold_of[g] = k
        load[k] += groups[g]
    return fold_of


class TreeTransectDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        name: str,
        species_table: str,
        tree_file: str,
        level1: dict,
        columns_json: str | None = None,
        evowm_root: str | None = None,
        n_trees: int = 100,
        morph_cols: list[str] | None = None,
        state_cols: list[str] | None = None,
        require_cols: tuple[str, ...] = ("m_tarsus", "m_mass"),
        holdout: dict | None = None,
        n_orders: int = 64,
        refresh_every: int = 256,
        samples_per_epoch: int | None = None,
        eco_dropout: float = 0.0,
        action_mode: str = "full",
        units: dict | None = None,
        seed: int = 0,
        keys_to_load=None,
        keys_to_cache=None,
        cache_dir=None,
        **level_kwargs,
    ):
        root = Path(evowm_root or os.environ.get("EVOWM_ROOT", "."))
        table_path = root / species_table
        meta_path = root / columns_json if columns_json else table_path.with_suffix(".columns.json")
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        table = pd.read_csv(table_path)

        self.split = "val" if str(name).endswith("val") else "train"
        if level1.get("frameskip", 1) != 1:
            raise ValueError("EvoWM 的第 1 层 frameskip 必须为 1")
        levels = [dict(level1)] + [dict(level_kwargs[k]) for k in sorted(level_kwargs) if k.startswith("level")]
        self.span = required_level1_steps(levels)

        morph_cols = list(morph_cols or (meta["morph"] + meta["skeleton"]))
        state_cols = list(state_cols or meta["state_default"])
        eco_groups = meta["eco_groups"]
        eco_cols = [c for grp in eco_groups.values() for c in grp]

        # ---- 可用物种与留出 ----
        eligible = table[list(require_cols)].notna().all(axis=1).to_numpy()
        holdout = dict(holdout or {})
        if holdout.get("folds"):  # 按科 K 折交叉验证：留出第 fold 份
            fold_of = family_folds(table, eligible, int(holdout["folds"]), int(holdout.get("seed", 0)),
                                   holdout.get("level", "family"))
            held = {g for g, k in fold_of.items() if k == int(holdout.get("fold", 0))}
        elif holdout.get("frac", 0):
            held = holdout_families(table, eligible, float(holdout["frac"]), int(holdout.get("seed", 0)),
                                    holdout.get("level", "family"), holdout.get("stratify"))
        else:
            held = set()
        in_val = table[holdout.get("level", "family")].isin(held).to_numpy()
        train_rows = eligible & ~in_val
        self.rows_mask = (eligible & in_val) if self.split == "val" else train_rows
        self.heldout_groups = sorted(held)

        # ---- 形态：用训练集统计量做 z 分数 ----
        X = table[morph_cols].to_numpy(np.float64)
        mu = np.nanmean(X[train_rows], axis=0)
        sd = np.nanstd(X[train_rows], axis=0)
        sd[~np.isfinite(sd) | (sd == 0)] = 1.0
        Xn = (X - mu) / sd
        mask = np.isfinite(Xn)
        self.pix = np.concatenate([np.nan_to_num(Xn), mask], axis=1).astype(np.float32)
        self.morph_cols, self.morph_mu, self.morph_sd = morph_cols, mu, sd

        # ---- 习性 ----
        E = table[eco_cols].to_numpy(np.float64)
        group_names = list(eco_groups)
        self.col_group = np.concatenate([[gi] * len(eco_groups[g]) for gi, g in enumerate(group_names)]).astype(int)
        Eg = np.stack([table[eco_groups[g]].notna().all(axis=1).to_numpy() for g in group_names], axis=1)
        self.E0 = np.nan_to_num(E).astype(np.float32)
        self.Eg = Eg.astype(np.float32)
        self.eco_cols, self.eco_group_names = eco_cols, group_names

        # ---- 上层单位（按习性事件分段）：叶序建好之前就要确定 ----
        self.units = None
        if units:
            from .units import UNIT_LEVELS, habit_codes
            u = dict(units)
            self.units = {"levels": list(u.get("levels", ["locomotor"])), "history": int(u.get("history", 3)),
                          "max_members": int(u.get("max_members", 16)), "max_subunits": int(u.get("max_subunits", 8))}
            if any(lv not in UNIT_LEVELS for lv in self.units["levels"]):
                raise ValueError(f"units.levels 只能取 {UNIT_LEVELS}，收到 {self.units['levels']}")
            if action_mode != "change":
                raise ValueError("上层单位只支持 action_mode=change")
            self.loco_code = habit_codes(table)
            self.Ecmask = self.Eg[:, self.col_group]

        self.S = table[state_cols].to_numpy(np.float32)
        self.state_cols = state_cols
        self.tips = table["tip"].tolist()

        # ---- 树 ----
        self.trees = read_trees(root / tree_file, max_trees=n_trees)
        row_index = {t: i for i, t in enumerate(self.tips)}
        self.row_of_tip = [np.array([row_index[n] for n in tr.tip_names]) for tr in self.trees]

        self.seed = int(seed)
        self.eco_dropout = float(eco_dropout)   # 训练时整条样带的习性部分置零的概率（只留进化距离）
        self.n_orders = int(n_orders)
        self.refresh_every = int(refresh_every)
        init_rng = np.random.default_rng(self.seed + (0 if self.split == "train" else 10_000))
        self.pool = [self._pool_entry(init_rng) for _ in range(self.n_orders)]

        n_rows = int(self.rows_mask.sum())
        self._len = int(samples_per_epoch or n_rows)
        self.lengths = np.array([n_rows])  # 上游归一化统计接口会读
        self.transform = None
        self.pixel_dim = self.pix.shape[1]
        if action_mode not in ("full", "change"):
            raise ValueError(f"action_mode 只能是 full 或 change，收到 {action_mode}")
        self.action_mode = action_mode
        n_eco = (2 if action_mode == "full" else 1) * len(eco_cols)
        self.action_dim = n_eco + len(group_names) + 2

    # ------------------------------------------------------------------
    def _order(self, rng: np.random.Generator):
        """一条随机树、随机旋转的叶序：物种表行号、相邻进化距离、相邻 LCA 深度、各叶到根的距离。"""
        ti = int(rng.integers(len(self.trees)))
        tr, rot = self.trees[ti], self.row_of_tip[ti]
        tips, lca = leaf_order(tr, rng)
        keep = self.rows_mask[rot]               # 按树的叶编号索引
        tips, lca = restrict(tips, lca, keep)
        dist = pair_distance(tr, tips, lca)
        return rot[tips], dist.astype(np.float32), lca, tr.depth[tr.tip_nodes[tips]]

    def _new_order(self, rng: np.random.Generator):
        """(行号, 相邻距离)；随机数的用法与只有第 1 层时完全相同，评估取窗口的结果不变。"""
        rows, dist, _, _ = self._order(rng)
        return rows, dist

    def unit_levels(self, rows: np.ndarray, lca: np.ndarray, dtip: np.ndarray):
        """一条叶序上的上层单位（evowm.data.units.build_unit_levels）。"""
        from .units import build_unit_levels
        return build_unit_levels(rows, lca, dtip, self.loco_code, self.E0, self.Ecmask, self.Eg, self.units["levels"])

    def _pool_entry(self, rng: np.random.Generator):
        rows, dist, lca, dtip = self._order(rng)
        if self.units is None:
            return rows, dist
        return rows, dist, self.unit_levels(rows, lca, dtip)

    def __len__(self) -> int:
        return self._len

    def _rng(self) -> np.random.Generator:
        return np.random.default_rng(int(torch.randint(0, 2**62, ()).item()))

    def __getitem__(self, idx: int) -> dict:
        rng = self._rng()
        j = int(rng.integers(self.n_orders))
        if self.split == "train" and self.refresh_every > 0 and rng.random() < 1.0 / self.refresh_every:
            self.pool[j] = self._pool_entry(rng)
        rows_all, dist_all = self.pool[j][:2]
        L = self.span
        if len(rows_all) < L:
            raise ValueError(f"可用物种 {len(rows_all)} 少于样带长度 {L}")
        start = int(rng.integers(0, len(rows_all) - L + 1))
        rows = rows_all[start:start + L]
        d = dist_all[start:start + L - 1]
        eco = not (self.split == "train" and self.eco_dropout > 0 and rng.random() < self.eco_dropout)
        action = self.build_action(rows, d, eco=eco)

        steps = {
            "pixels": torch.from_numpy(self.pix[rows]),
            "action": torch.from_numpy(action),
            "state": torch.from_numpy(self.S[rows]),
        }
        if self.transform:
            steps = self.transform(steps)
        out = {f"{k}_level1": v for k, v in steps.items()}
        if self.units is not None:
            out.update(self._unit_sample(rng, rows_all, self.pool[j][2], eco))
        return out

    # ---- 上层单位的样本 ----
    def _members(self, rows_all: np.ndarray, s: int, e: int, M: int, rng: np.random.Generator) -> np.ndarray:
        """单位 [s, e) 的成员行号：不超过 M 种时全取，否则随机抽 M 种（不放回）。"""
        pos = np.arange(s, e) if e - s <= M else s + np.sort(rng.choice(e - s, M, replace=False))
        return rows_all[pos]

    def _unit_action(self, lv, idx: np.ndarray, eco: bool) -> np.ndarray:
        act = lv.action[idx].copy()
        act[-1] = 0.0                                   # 窗口最后一个单位之后没有转变，与第 1 层相同填 0
        if not eco:
            act[:, :-2] = 0.0                           # 习性置零：只留进化距离
        return act

    def _unit_sample(self, rng: np.random.Generator, rows_all: np.ndarray, levels: dict, eco: bool) -> dict:
        T = self.units["history"] + 1
        M, K, P = self.units["max_members"], self.units["max_subunits"], self.pix.shape[1]
        out = {}
        if "locomotor" in levels:
            lv = levels["locomotor"]
            if len(lv) < T:
                raise ValueError(f"运动方式单位只有 {len(lv)} 个，少于窗口长度 {T}")
            t0 = int(rng.integers(0, len(lv) - T + 1))
            pix = np.zeros((T, M, P), np.float32)
            mask = np.zeros((T, M), bool)
            for i, k in enumerate(range(t0, t0 + T)):
                r = self._members(rows_all, lv.starts[k], lv.ends[k], M, rng)
                pix[i, :len(r)] = self.pix[r]
                mask[i, :len(r)] = True
            out.update(u2_pixels=torch.from_numpy(pix), u2_mask=torch.from_numpy(mask),
                       u2_action=torch.from_numpy(self._unit_action(lv, np.arange(t0, t0 + T), eco)))
        if "realm" in levels:
            lv, sub = levels["realm"], levels["realm"].sub
            lo = levels["locomotor"] if "locomotor" in levels else None
            if lo is None:
                raise ValueError("栖境层需要运动方式层")
            if len(lv) < T:
                raise ValueError(f"栖境单位只有 {len(lv)} 个，少于窗口长度 {T}")
            t0 = int(rng.integers(0, len(lv) - T + 1))
            pix = np.zeros((T, K, M, P), np.float32)
            mask = np.zeros((T, K, M), bool)
            smask = np.zeros((T, K), bool)
            for i, k in enumerate(range(t0, t0 + T)):
                k0, k1 = sub[k]
                kk = np.arange(k0, k1) if k1 - k0 <= K else k0 + np.sort(rng.choice(k1 - k0, K, replace=False))
                for a, q in enumerate(kk):
                    r = self._members(rows_all, lo.starts[q], lo.ends[q], M, rng)
                    pix[i, a, :len(r)] = self.pix[r]
                    mask[i, a, :len(r)] = True
                smask[i, :len(kk)] = True
            out.update(u3_pixels=torch.from_numpy(pix), u3_mask=torch.from_numpy(mask),
                       u3_submask=torch.from_numpy(smask),
                       u3_action=torch.from_numpy(self._unit_action(lv, np.arange(t0, t0 + T), eco)))
        return out

    def eco_rows(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """物种的习性向量（缺失列置 0）与习性组掩码。"""
        cmask = self.Eg[rows][:, self.col_group]
        return self.E0[rows] * cmask, self.Eg[rows]

    def build_action(self, rows: np.ndarray, d: np.ndarray, eco: bool = True,
                     E: np.ndarray | None = None, Eg: np.ndarray | None = None) -> np.ndarray:
        """一条样带的动作序列 (L, action_dim)。E、Eg 可传入替换值（反事实评估用）。"""
        if E is None or Eg is None:
            E, Eg = self.eco_rows(rows)
        cmask = Eg[:, self.col_group]
        both = cmask[1:] * cmask[:-1]
        if self.action_mode == "change":
            parts = [
                (E[1:] - E[:-1]) * both,                     # 习性差
                Eg[1:] * Eg[:-1],                            # 两种的各习性组是否都有数据（区分"没变"与"没数据"）
            ]
        else:
            parts = [
                (E[1:] - E[:-1]) * both,                     # 习性差
                E[1:],                                       # 下一种的习性
                Eg[1:],                                      # 下一种各习性组是否有数据
            ]
        if not eco:
            parts = [np.zeros_like(p) for p in parts]
        action = np.concatenate(parts + [
            np.log1p(d)[:, None] / np.log1p(D_MAX),          # 进化距离（对数）
            d[:, None] / D_MAX,                              # 进化距离（线性）
        ], axis=1)
        action = np.concatenate([action, np.zeros((1, action.shape[1]))], axis=0)
        return action.astype(np.float32)

    # 上游 get_column_normalizer / build_normalizer_artifact 的接口；本数据集已自行归一化
    def get_col_stats(self, col: str):
        dim = {"action": self.action_dim, "state": self.S.shape[1], "pixels": self.pixel_dim}[col]
        return np.zeros((1, dim)), np.ones((1, dim))

    def describe(self) -> str:
        s = (f"TreeTransectDataset[{self.split}] 物种 {int(self.rows_mask.sum())}，样带长度 {self.span}，"
             f"pixels 维 {self.pixel_dim}，action 维 {self.action_dim}（{self.action_mode}），树 {len(self.trees)} 棵，"
             f"留出类群 {len(self.heldout_groups)} 个")
        if self.units is not None:
            n = {k: int(np.mean([len(p[2][k]) for p in self.pool])) for k in self.units["levels"]}
            s += f"；上层单位 {self.units}，每条叶序平均单位数 {n}"
        return s
