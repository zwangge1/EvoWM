"""Allen-specific conditions on the existing phylogenetic JEPA data interface.

Leaves are comparative samples, not observed ancestor-descendant time steps.
The optional second level uses only tree-defined clades, never ecological labels.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .transect import D_MAX, TreeTransectDataset
from .tree import leaf_order, pair_distance, restrict
from .units import adjacent_unit_distance, runs


MASK_MODES = ("full", "no_temperature", "no_motion", "neither")
FACTOR_MASK_MODES = MASK_MODES + ("no_exposure", "no_development", "no_temperature_exposure", "no_temperature_development")
HIDDEN_GROUPS = {
    "full": (), "no_temperature": ("temperature",), "no_motion": ("motion",),
    "neither": ("temperature", "motion"), "no_exposure": ("exposure",),
    "no_development": ("development",),
    "no_temperature_exposure": ("temperature", "exposure"),
    "no_temperature_development": ("temperature", "development"),
}


class AllenDataset(TreeTransectDataset):
    def __init__(self, *args, condition_mask_probs=None,
                 clade_age=20.0, units=None, **kwargs):
        if kwargs.get("eco_dropout", 0) != 0:
            raise ValueError("Allen uses explicit group masks; eco_dropout must be zero")
        if units and list(units.get("levels", [])) != ["phylo"]:
            raise ValueError("Allen upper level must be ['phylo']; ecological grouping leaks motion")
        if int(kwargs.get("n_orders", 64)) < 1:
            raise ValueError("n_orders must be positive")
        if units and int(units.get("history", 3)) < 1:
            raise ValueError("units.history must be positive")
        self.clade_age = float(clade_age)
        self.clade_codes = None
        super().__init__(*args, units=None, **kwargs)
        root = Path(kwargs.get("evowm_root") or __import__("os").environ.get("EVOWM_ROOT", "."))
        table_path = root / kwargs["species_table"]
        self.table = pd.read_csv(table_path)
        if self.table.tip.duplicated().any():
            raise ValueError("Species table must have unique tips")
        self.condition_groups = dict(zip(self.eco_group_names,
            [[c for c, g in zip(self.eco_cols, self.col_group) if g == i] for i in range(len(self.eco_group_names))]))
        if list(self.condition_groups) == ["temperature", "motion", "background"]:
            self.mask_modes = MASK_MODES
        elif list(self.condition_groups) == ["temperature", "exposure", "motion", "development", "background"]:
            self.mask_modes = FACTOR_MASK_MODES
        else:
            raise ValueError("Expected the documented Allen v1 or five-group v2 condition schema")
        probs = np.asarray(condition_mask_probs if condition_mask_probs is not None else
                           [1 / len(self.mask_modes)] * len(self.mask_modes), dtype=float)
        if probs.shape != (len(self.mask_modes),) or not np.all(np.isfinite(probs)) or np.any(probs < 0) or not np.isclose(probs.sum(), 1):
            raise ValueError(f"condition_mask_probs must contain {len(self.mask_modes)} probabilities summing to one")
        self.condition_mask_probs = probs
        forbidden = set(self.morph_cols) & set(self.eco_cols)
        if forbidden:
            raise ValueError(f"Target morphology cannot be a condition: {forbidden}")
        eligible = self.table[list(kwargs.get("require_cols", ("m_tarsus", "m_mass")))].notna().all(axis=1).to_numpy()
        holdout = dict(kwargs.get("holdout") or {})
        if "allen_frozen_family_fold" in self.table:
            if holdout.get("folds") != 5 or holdout.get("seed", 0) != 0 or holdout.get("level", "family") != "family":
                raise ValueError("v2 inputs require their frozen five family folds (seed=0)")
            frozen_val = eligible & (self.table.allen_frozen_family_fold.to_numpy() == int(holdout.get("fold", 0)))
            actual_val = eligible & self.table.family.isin(self.heldout_groups).to_numpy()
            if not np.array_equal(frozen_val, actual_val):
                raise ValueError("Recomputed family split differs from the frozen input fold assignment")
        train = eligible & ~self.table[dict(kwargs.get("holdout") or {}).get("level", "family")].isin(self.heldout_groups).to_numpy()
        self.training_rows = train
        raw = self.table[self.eco_cols].to_numpy(float)
        if not np.isfinite(raw[train]).any(axis=0).all():
            raise ValueError("A condition column has no training observations")
        self.condition_mu = np.nanmean(np.where(np.isfinite(raw[train]), raw[train], np.nan), axis=0)
        self.condition_sd = np.nanstd(np.where(np.isfinite(raw[train]), raw[train], np.nan), axis=0)
        self.condition_sd[self.condition_sd == 0] = 1
        self.condition_values, self.condition_valid = self.normalize_conditions(raw)
        self.condition_raw = raw
        if not np.isfinite(self.morph_mu).all():
            raise ValueError("A morphology column has no training observations; change split or data")
        self.action_layout = {}
        offset = 0
        for group, columns in self.condition_groups.items():
            n = len(columns)
            # source, target, difference, source observed, target observed, available-to-model
            self.action_layout[group] = (offset, offset + 5 * n + 1)
            offset += 5 * n + 1
        self.action_dim = offset + 2
        self.action_mode = "allen_conditions"
        if units:
            from .allen_hierarchy import phylogenetic_clade_codes
            self.units = {"levels": ["phylo"], "history": int(units.get("history", 3)),
                          "max_members": int(units.get("max_members", 16)), "max_subunits": 1}
            if self.units["max_members"] < 1:
                raise ValueError("max_members must be positive")
            self.clade_codes = [phylogenetic_clade_codes(t, self.clade_age) for t in self.trees]
            rng = np.random.default_rng(self.seed + (0 if self.split == "train" else 10000))
            self.pool = [self._pool_entry(rng) for _ in range(self.n_orders)]

    def normalize_conditions(self, raw):
        raw = np.asarray(raw, float)
        valid = np.isfinite(raw)
        return np.where(valid, (raw - self.condition_mu) / self.condition_sd, 0).astype(np.float32), valid

    def action_from_conditions(self, values, valid, distance, mode="full"):
        if mode not in self.mask_modes:
            raise ValueError(f"Unknown condition mask {mode}")
        values, valid, distance = np.asarray(values), np.asarray(valid, bool), np.asarray(distance)
        if values.ndim != 2 or values.shape != valid.shape or values.shape[1] != len(self.eco_cols) or distance.ndim != 1 or len(distance) != len(values) - 1:
            raise ValueError("Condition/action dimensions disagree")
        if not np.isfinite(values[valid]).all():
            raise ValueError("Observed conditions must be finite")
        if not np.isfinite(distance).all() or np.any(distance < 0):
            raise ValueError("Phylogenetic distances must be finite and nonnegative")
        blocks = []
        for gi, group in enumerate(self.condition_groups):
            idx = self.col_group == gi
            x = np.where(valid[:, idx], values[:, idx], 0)
            m = valid[:, idx]
            both = m[:-1] & m[1:]
            block = np.concatenate([x[:-1], x[1:], (x[1:] - x[:-1]) * both,
                                    m[:-1], m[1:], np.ones((len(distance), 1))], axis=1)
            hidden = group in HIDDEN_GROUPS[mode]
            blocks.append(np.zeros_like(block) if hidden else block)
        blocks += [np.log1p(distance)[:, None] / np.log1p(D_MAX), distance[:, None] / D_MAX]
        a = np.concatenate(blocks, axis=1)
        return np.concatenate([a, np.zeros((1, a.shape[1]))]).astype(np.float32)

    def build_action(self, rows, d, eco=True, *, mode=None, raw_conditions=None):
        mode = mode or ("full" if eco else "neither")
        values, valid = (self.normalize_conditions(raw_conditions) if raw_conditions is not None else
                         (self.condition_values[rows], self.condition_valid[rows]))
        return self.action_from_conditions(values, valid, d, mode)

    def _pool_entry(self, rng):
        if self.clade_codes is None:
            return super()._pool_entry(rng)
        ti = int(rng.integers(len(self.trees)))
        tree, row_map = self.trees[ti], self.row_of_tip[ti]
        tips, lca = leaf_order(tree, rng)
        tips, lca = restrict(tips, lca, self.rows_mask[row_map])
        rows = row_map[tips]
        starts, ends = runs(self.clade_codes[ti][tips])
        dtip = tree.depth[tree.tip_nodes[tips]]
        unit_dist = adjacent_unit_distance(starts, ends, lca, dtip)
        if len(starts) < self.units["history"] + 1:
            raise ValueError("Too few phylogenetic clades in this split for the upper-level window")
        return rows, pair_distance(tree, tips, lca).astype(np.float32), (starts, ends, unit_dist)

    def sample(self, rng, *, mode=None):
        j = int(rng.integers(self.n_orders))
        if self.split == "train" and self.refresh_every > 0 and rng.random() < 1 / self.refresh_every:
            self.pool[j] = self._pool_entry(rng)
        rows_all, dist_all = self.pool[j][:2]
        if len(rows_all) < self.span:
            raise ValueError("Split has fewer species than the prediction window")
        start = int(rng.integers(len(rows_all) - self.span + 1))
        rows = rows_all[start:start + self.span]
        d = dist_all[start:start + self.span - 1]
        mode = mode or (str(rng.choice(self.mask_modes, p=self.condition_mask_probs)) if self.split == "train" else "full")
        out = {"pixels_level1": torch.from_numpy(self.pix[rows]),
               "action_level1": torch.from_numpy(self.build_action(rows, d, mode=mode)),
               "state_level1": torch.from_numpy(self.S[rows]),
               "rows_level1": torch.from_numpy(rows.copy())}
        if self.transform:
            steps = self.transform({k.removesuffix("_level1"): v for k, v in out.items()})
            out = {f"{k}_level1": v for k, v in steps.items()}
        if self.units:
            starts, ends, ud = self.pool[j][2]
            T, M = self.units["history"] + 1, self.units["max_members"]
            first = int(rng.integers(len(starts) - T + 1))
            pixels = np.zeros((T, M, self.pixel_dim), np.float32)
            mask = np.zeros((T, M), bool)
            cv, cm = [], []
            for i, k in enumerate(range(first, first + T)):
                # Same sampled members determine state and conditions; no target-conditioned selection.
                members = self._members(rows_all, starts[k], ends[k], M, rng)
                pixels[i, :len(members)] = self.pix[members]
                mask[i, :len(members)] = True
                valid = self.condition_valid[members]
                count = valid.sum(axis=0)
                cv.append((self.condition_values[members] * valid).sum(axis=0) / np.maximum(count, 1))
                cm.append(count > 0)
            action = self.action_from_conditions(np.array(cv), np.array(cm), ud[first:first + T - 1], mode)
            out.update(u2_pixels=torch.from_numpy(pixels), u2_mask=torch.from_numpy(mask),
                       u2_action=torch.from_numpy(action))
        return out

    def __getitem__(self, idx):
        return self.sample(self._rng())

    def describe(self):
        return (f"AllenDataset[{self.split}] n={self.rows_mask.sum()} pixels={self.pixel_dim} "
                f"actions={self.action_dim}; masks={self.mask_modes}; hierarchy={self.units}")

    def audit(self):
        return {"split": self.split, "n_species": int(self.rows_mask.sum()),
                "heldout_families": self.heldout_groups, "state_cols": self.state_cols,
                "morph_mu": self.morph_mu.tolist(), "morph_sd": self.morph_sd.tolist(),
                "condition_columns": self.eco_cols, "condition_groups": self.condition_groups,
                "mask_modes": self.mask_modes, "condition_mask_probs": self.condition_mask_probs.tolist(),
                "condition_mu": self.condition_mu.tolist(), "condition_sd": self.condition_sd.tolist(),
                "action_layout": self.action_layout, "action_dim": self.action_dim,
                "pixel_dim": self.pixel_dim, "clade_age": self.clade_age if self.units else None}
