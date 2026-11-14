"""Training-only joint-input support for frozen Allen ecological probes.

This is an exploratory distance screen, not a guarantee of causal positivity
or reliable extrapolation. Only complete training bone labels determine the
eligible support set; their values never enter distances or thresholds.
"""
from __future__ import annotations

import copy
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


DEVELOPMENT = ('P', 'SP', 'SA', 'A')
DEVELOPMENT_COLUMNS = tuple('e_dev_' + x for x in DEVELOPMENT)


def _condition_columns(columns):
    if columns is None:
        from phylo_baseline_data import CONDITION_COLUMNS
        columns = CONDITION_COLUMNS
    columns = tuple(columns)
    if len(set(columns)) != len(columns) or not set(DEVELOPMENT_COLUMNS) <= set(columns):
        raise ValueError('Unique condition columns must include all development indicators')
    return columns


def _consistent_development(raw, labels, columns):
    values = raw[:, [columns.index(c) for c in DEVELOPMENT_COLUMNS]]
    finite = np.isfinite(values).all(axis=1)
    binary = (np.isclose(values, 0., atol=1e-8, rtol=0.) |
              np.isclose(values, 1., atol=1e-8, rtol=0.)).all(axis=1)
    one = np.isclose(values.sum(axis=1), 1., atol=1e-8, rtol=0.)
    known = np.isin(labels, DEVELOPMENT)
    declared = np.asarray(DEVELOPMENT)[np.argmax(np.nan_to_num(values, nan=-np.inf), axis=1)]
    return finite & binary & one & known & (declared == labels)


class JointSupport:
    """A same-development fifth-neighbour screen fitted without target outcomes.

    Use ``JointSupport.fit(table, train_rows)`` then ``evaluate(raw, groups)``.
    Every scenario (including the unmodified baseline) must pass separately.
    ``metadata()`` and ``training_records`` provide an auditable support record.
    """

    @classmethod
    def fit(cls, table, train_rows, *, condition_columns=None,
            bone_columns=('b_fem', 'b_tbt', 'b_tmt'), group_column='mode',
            k=5, quantile=.95, min_group_rows=12, min_group_families=3,
            tolerance=1e-8):
        columns = _condition_columns(condition_columns)
        rows = np.asarray(train_rows)
        if (rows.ndim != 1 or rows.dtype.kind not in 'iu' or not len(rows)
                or len(np.unique(rows)) != len(rows)
                or (rows < 0).any() or (rows >= len(table)).any()):
            raise ValueError('train_rows must contain distinct valid integer positions')
        if (not isinstance(k, (int, np.integer)) or k < 1
                or min_group_rows < k + 1 or min_group_families < 1
                or not np.isfinite(quantile) or not 0 < quantile <= 1
                or not np.isfinite(tolerance) or tolerance <= 0):
            raise ValueError('Invalid support calibration settings')
        required = [*columns, *bone_columns, group_column, 'tip', 'family']
        absent = set(required) - set(table)
        if absent:
            raise ValueError('Missing support columns: ' + ', '.join(sorted(absent)))
        train = table.iloc[rows].copy()
        raw = train[list(columns)].to_numpy(float)
        bones = train[list(bone_columns)].to_numpy(float)
        complete_bones = np.isfinite(bones).all(axis=1)
        complete_conditions = np.isfinite(raw).all(axis=1)
        labels = train[group_column].fillna('').astype(str).to_numpy()
        identity = train[['tip', 'family', group_column]].notna().all(axis=1).to_numpy()
        eligible = complete_bones & complete_conditions & identity
        consistent = _consistent_development(raw, labels, columns)
        if (eligible & ~consistent).any():
            bad = train.loc[eligible & ~consistent, 'tip'].tolist()
            raise ValueError('Development label/one-hot mismatch in training support: ' + str(bad[:8]))
        eligible &= consistent
        self = cls()
        self.columns = columns
        self.k = int(k)
        self.tolerance = float(tolerance)
        self._groups = {}
        self._metadata = {
            'condition_columns': list(columns), 'group_column': group_column,
            'k': int(k), 'quantile': float(quantile),
            'min_group_rows': int(min_group_rows), 'min_group_families': int(min_group_families),
            'n_training_rows': len(rows), 'n_complete_training_bones': int(complete_bones.sum()),
            'n_training_support_rows': int(eligible.sum()),
            'scaling': 'Within-development mean and population SD on eligible training bone rows only; active standardized coordinates divided by sqrt(active coordinate count).',
            'calibration': 'Quantile of kth nearest distance among same-development training rows after excluding self; other members of the same family may remain.',
            'query_rule': 'All finite conditions, matching development one-hot, within each training marginal range, and kth-neighbour distance <= calibrated threshold.',
            'interpretation': 'Exploratory input-support screen, not a causal positivity or prediction-accuracy guarantee. Same-family neighbours may make leave-one-out calibration optimistic for held-out-family transfer.',
            'groups': {},
        }
        records = []
        for label in DEVELOPMENT:
            pos = np.flatnonzero(eligible & (labels == label))
            part = train.iloc[pos]
            count = len(pos)
            families = part.family.astype(str).to_numpy()
            info = {'n_species': count, 'n_families': len(set(families)),
                    'training_rows': rows[pos].tolist(),
                    'training_tips': part.tip.astype(str).tolist(),
                    'training_families': families.tolist()}
            reason = ('too_few_training_species' if count < min_group_rows else
                      'too_few_training_families' if len(set(families)) < min_group_families else '')
            if reason:
                info.update(supported=False, reason=reason, threshold=None)
                self._metadata['groups'][label] = info
                for row, tip, family in zip(rows[pos], part.tip, families):
                    records.append({'row': int(row), 'tip': tip, 'family': family,
                                    'group': label, 'calibration_distance': np.nan,
                                    'group_supported': False})
                continue
            values = raw[pos]
            mean = values.mean(axis=0)
            sd = values.std(axis=0)
            active = sd > 1e-12
            if not active.any():
                # A fully discrete identical condition cloud can only support
                # itself. Keep one constant zero coordinate for cKDTree.
                standardized = np.zeros((count, 1))
            else:
                standardized = (values[:, active] - mean[active]) / sd[active] / np.sqrt(active.sum())
            tree = cKDTree(standardized)
            # Request k+1 and explicitly remove self. Coincident observations
            # are legitimate other species; their zero distances remain.
            distance, indices = tree.query(standardized, k=int(k) + 1)
            loo = np.empty(count)
            for i in range(count):
                other = distance[i][indices[i] != i]
                if len(other) < k:
                    raise RuntimeError('Insufficient leave-one-out neighbours')
                loo[i] = np.sort(other)[k - 1]
            threshold = float(np.quantile(loo, quantile))
            info.update(supported=True, reason='', mean=mean.tolist(), sd=sd.tolist(),
                        active_columns=[c for c, keep in zip(columns, active) if keep],
                        constant_columns=[c for c, keep in zip(columns, active) if not keep],
                        minimum=values.min(axis=0).tolist(), maximum=values.max(axis=0).tolist(),
                        threshold=threshold, calibration_distances=loo.tolist())
            self._metadata['groups'][label] = info
            self._groups[label] = {'tree': tree, 'mean': mean, 'sd': sd, 'active': active,
                                   'min': values.min(axis=0), 'max': values.max(axis=0),
                                   'threshold': threshold, 'families': families}
            for row, tip, family, radius in zip(rows[pos], part.tip, families, loo):
                records.append({'row': int(row), 'tip': tip, 'family': family,
                                'group': label, 'calibration_distance': float(radius),
                                'group_supported': True})
        self.training_records = pd.DataFrame(records, columns=[
            'row', 'tip', 'family', 'group', 'calibration_distance', 'group_supported'])
        return self

    def metadata(self):
        return copy.deepcopy(self._metadata)

    def evaluate(self, raw_conditions, groups):
        raw = np.asarray(raw_conditions, dtype=float)
        labels = np.asarray(groups).astype(str)
        if raw.ndim != 2 or raw.shape[1] != len(self.columns) or labels.shape != (len(raw),):
            raise ValueError('Expected N by condition-count values and N development labels')
        finite = np.isfinite(raw).all(axis=1)
        consistent = _consistent_development(raw, labels, self.columns)
        result = pd.DataFrame({
            'group': labels, 'supported': np.zeros(len(raw), dtype=bool),
            'standardized_neighbor_distance': np.full(len(raw), np.nan),
            'threshold': np.full(len(raw), np.nan),
            'neighbor_family_count': np.zeros(len(raw), dtype=int),
            'within_marginal_range': np.zeros(len(raw), dtype=bool),
            'reason': np.full(len(raw), 'unknown_development_group', dtype=object),
        })
        result.loc[~finite, 'reason'] = 'nonfinite_conditions'
        result.loc[finite & ~consistent, 'reason'] = 'development_label_mismatch'
        for label in DEVELOPMENT:
            selected = finite & consistent & (labels == label)
            positions = np.flatnonzero(selected)
            if not len(positions):
                continue
            if label not in self._groups:
                result.loc[positions, 'reason'] = self._metadata['groups'][label]['reason']
                continue
            g = self._groups[label]
            values = raw[positions]
            marginal = ((values >= g['min'] - self.tolerance) &
                        (values <= g['max'] + self.tolerance)).all(axis=1)
            if not g['active'].any():
                standardized = np.zeros((len(values), 1))
            else:
                active = g['active']
                standardized = (values[:, active] - g['mean'][active]) / g['sd'][active] / np.sqrt(active.sum())
            distance, indices = g['tree'].query(standardized, k=self.k)
            if self.k == 1:
                distance, indices = distance[:, None], indices[:, None]
            kth = distance[:, -1]
            family_count = np.asarray([len(set(g['families'][idx])) for idx in indices])
            supported = marginal & (kth <= g['threshold'] + self.tolerance)
            reason = np.where(~marginal, 'outside_training_marginal_range',
                              np.where(supported, 'supported', 'outside_training_joint_neighbourhood'))
            result.loc[positions, 'standardized_neighbor_distance'] = kth
            result.loc[positions, 'threshold'] = g['threshold']
            result.loc[positions, 'neighbor_family_count'] = family_count
            result.loc[positions, 'within_marginal_range'] = marginal
            result.loc[positions, 'supported'] = supported
            result.loc[positions, 'reason'] = reason
        return result
