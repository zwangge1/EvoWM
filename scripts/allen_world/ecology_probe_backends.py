"""Frozen, matched neural and BM/OU backends for target-only ecology probes.

No fitting occurs here. Each backend first reproduces every saved observed-
reference prediction, then changes only the target's climate/foraging inputs.
Classic predictions reuse the original fitted mean, covariance and reference
posterior correction. Importing/using the classic backend requires no PyTorch.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from asymmetry_v1_evaluate import (STATE_COLUMNS, load_cell, mask_reference_pixels,
                                  sha, validate_architecture, visible_references)
from phylo_baseline_data import CONDITION_COLUMNS, FORAGING_COLUMNS, TEMPERATURE_COLUMNS

ALLOWED_COLUMNS = tuple(TEMPERATURE_COLUMNS) + tuple(FORAGING_COLUMNS)


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _run_path(source_run, root):
    run = Path(source_run)
    return (Path(root) / run).resolve() if not run.is_absolute() else run.resolve()


def find_job(source_run, fold, seed, variant, root=ROOT):
    """Resolve actual manifest seed identities; do not infer replicate from seed."""
    root = Path(root).resolve()
    run = _run_path(source_run, root)
    variant = str(variant).lower()
    if variant not in ('full', 'single', 'supervised', 'bm', 'ou'):
        raise ValueError('Unknown frozen predictor variant: ' + variant)
    manifest = _json(run / 'manifest.json')
    if manifest.get('kind') != 'allen_asymmetric_data_v1':
        raise ValueError('Expected the original Allen asymmetry v1 manifest')
    wanted = 'classic' if variant in ('bm', 'ou') else variant
    jobs = [j for j in manifest['jobs'] if j['condition'] == 'complete100'
            and int(j['fold']) == int(fold) and int(j['seed']) == int(seed)
            and j['variant'] == wanted]
    if len(jobs) != 1:
        raise ValueError('Expected exactly one complete100 job for fold/seed/variant')
    return run, jobs[0], manifest


def inspect_checkpoint_paths(source_run, fold, seed, variant, root=ROOT):
    """No-torch preflight. Missing checkpoint is reported, never downloaded/refit."""
    run, job, _ = find_job(source_run, fold, seed, variant, root)
    evaluation = run / 'evaluation' / job['name']
    result = _json(evaluation / 'result.json')
    if result.get('status') != 'complete' or result.get('job') != job:
        raise ValueError('Saved evaluation does not match manifest job')
    paths = {'result': evaluation / 'result.json',
             'predictions': evaluation / 'predictions.csv.gz'}
    expected = {'predictions': result['output_sha256']['predictions.csv.gz']}
    variant = str(variant).lower()
    if variant in ('bm', 'ou'):
        paths.update(fit=evaluation / (variant + '_fit.json'), design=evaluation / 'design.json')
        expected.update(fit=result['output_sha256'][variant + '_fit.json'],
                        design=result['output_sha256']['design.json'])
    else:
        # Original checkpoint path is a Linux absolute path. Relocate only its
        # filename beneath this run's unchanged job directory.
        folder = run / 'training' / 'ckpts' / job['name']
        paths['checkpoint'] = folder / Path(result['checkpoint'].replace('\\', '/')).name
        paths['config'] = folder / Path(result['config'].replace('\\', '/')).name
        expected.update(checkpoint=result['checkpoint_sha256'], config=result['config_sha256'])
    actual = {key: sha(path) for key, path in paths.items() if path.is_file()}
    for key, digest in expected.items():
        if key in actual and actual[key] != digest:
            raise ValueError('Saved artifact hash mismatch: ' + str(paths[key]))
    return {'source_run': str(run), 'job': job, 'variant': variant,
            'paths': {key: str(path) for key, path in paths.items()},
            'expected_sha256': expected, 'actual_sha256': actual,
            'missing': [str(path) for path in paths.values() if not path.is_file()],
            'ready': all(path.is_file() for path in paths.values())}


def verify_inference_source(run, manifest, root, neural):
    """Require original executable inference sources; allow Git newline changes."""
    common = {'scripts/allen_world/asymmetry_v1_evaluate.py',
              'scripts/allen_world/asymmetry_v1_plan.py',
              'scripts/allen_world/phylo_baseline_data.py',
              'scripts/allen_world/phylo_baseline_core.py',
              'src/evowm/data/tree.py'}
    recorded = manifest.get('source_sha256', {})
    if not common <= set(recorded):
        raise ValueError('Source run lacks required inference-code hashes')
    critical = set(common)
    if neural:
        critical.update(rel for rel in recorded
                        if rel.startswith(('src/evowm/', 'hjepa/h_jepa/')))
        critical.add('scripts/allen_world/evaluate.py')
    verified = {}
    for rel in sorted(critical):
        snapshot, live = Path(run) / 'source' / rel, Path(root) / rel
        if rel not in recorded or not snapshot.is_file() or sha(snapshot) != recorded[rel]:
            raise ValueError('Original inference snapshot missing or modified: ' + rel)
        if not live.is_file() or (snapshot.read_bytes().replace(b'\r\n', b'\n') !=
                                  live.read_bytes().replace(b'\r\n', b'\n')):
            raise ValueError('Inference code changed since original fit: ' + rel)
        verified[rel] = sha(live)
    return verified


class FrozenConditionDesign:
    """Evaluate saved standardization/basis, without re-estimating any choice."""
    def __init__(self, design):
        self.info = design
        self.condition_columns = tuple(design['condition_columns'])
        if self.condition_columns != tuple(CONDITION_COLUMNS):
            raise ValueError('Classic condition columns differ from the frozen task')
        self.mu = np.array([design['scaling'][c]['mean'] for c in self.condition_columns], float)
        self.sd = np.array([design['scaling'][c]['sd'] for c in self.condition_columns], float)
        if not np.isfinite(self.mu).all() or not np.isfinite(self.sd).all() or np.any(self.sd <= 0):
            raise ValueError('Invalid saved condition normalization')
        self.pairs = [tuple(pair) for pair in design['interaction_pairs']]
        candidate = ['intercept', *self.condition_columns,
                     *['missing_' + c for c in self.condition_columns],
                     *[a + ':' + b for a, b in self.pairs]]
        if candidate != design['candidate_columns']:
            raise ValueError('Saved classic candidate basis is inconsistent')
        self.keep = np.asarray(design['kept_indices'], dtype=int)
        if (len(np.unique(self.keep)) != len(self.keep) or np.any(self.keep < 0)
                or np.any(self.keep >= len(candidate))
                or [candidate[i] for i in self.keep] != design['columns']):
            raise ValueError('Invalid saved identifiable basis')

    def transform(self, conditions):
        raw = np.asarray(conditions, float)
        if raw.ndim != 2 or raw.shape[1] != len(self.condition_columns) or np.isinf(raw).any():
            raise ValueError('Expected N by original condition columns, NaN allowed')
        finite = np.isfinite(raw)
        z = np.where(finite, (raw - self.mu) / self.sd, 0.)
        parts = [np.ones((len(raw), 1)), z, (~finite).astype(float)]
        for first, second in self.pairs:
            parts.append((z[:, self.condition_columns.index(first)] *
                          z[:, self.condition_columns.index(second)])[:, None])
        return np.column_stack(parts)[:, self.keep]


def validate_target_changes(original, changed, condition_columns=CONDITION_COLUMNS):
    """Enforce target-only ecology/climate changes and unchanged observation mask."""
    before, after = np.asarray(original, float), np.asarray(changed, float)
    columns = tuple(condition_columns)
    if before.ndim != 2 or after.shape != before.shape or before.shape[1] != len(columns):
        raise ValueError('Target condition shapes disagree')
    if np.isinf(before).any() or np.isinf(after).any():
        raise ValueError('Infinite target conditions are not permitted')
    if not np.array_equal(np.isfinite(before), np.isfinite(after)):
        raise ValueError('Condition probe must preserve the target observation mask')
    fixed = [j for j, name in enumerate(columns) if name not in ALLOWED_COLUMNS]
    if not np.array_equal(before[:, fixed], after[:, fixed], equal_nan=True):
        raise ValueError('Probe changed a non-ecology/non-temperature condition')
    fs = [columns.index(c) for c in FORAGING_COLUMNS]
    a, b = before[:, fs], after[:, fs]
    moved = ~np.all((a == b) | (np.isnan(a) & np.isnan(b)), axis=1)
    if moved.any():
        if not np.isfinite(a[moved]).all() or not np.isfinite(b[moved]).all():
            raise ValueError('Foraging replacement requires all seven measured fractions')
        if np.any(b[moved] < 0) or np.any(b[moved] > 1):
            raise ValueError('Foraging replacement left the allowed fraction range')
        if not np.allclose(a[moved].sum(axis=1), b[moved].sum(axis=1), atol=1e-10, rtol=0):
            raise ValueError('Foraging replacement must preserve the original composition sum')
    return after.copy()


def _window_indices(indices, n):
    if indices is None:
        return np.arange(n, dtype=int)
    values = np.asarray(indices)
    if (values.ndim != 1 or not np.issubdtype(values.dtype, np.integer)
            or np.any(values < 0) or np.any(values >= n) or len(np.unique(values)) != len(values)):
        raise ValueError('Invalid or repeated saved window indices')
    return values.astype(int, copy=True)


def _saved_predictions(cell, variant, path):
    frame = pd.read_csv(path)
    model = variant.upper() if variant in ('bm', 'ou') else variant
    frame = frame.loc[frame.model.eq(model) & frame.reference_case.eq('observed')].copy()
    frame = frame.sort_values('window').reset_index(drop=True)
    if len(frame) != len(cell.target_rows) or not np.array_equal(frame.window, np.arange(len(frame))):
        raise ValueError('Saved observed predictions lack exact frozen window support')
    for key in ('fold', 'replicate', 'seed', 'mask_seed', 'condition'):
        if not frame[key].eq(cell.job[key]).all():
            raise ValueError('Saved prediction job identity differs: ' + key)
    expected = {'target_row': cell.target_rows,
                'target_tip': cell.table.tip.to_numpy()[cell.target_rows],
                'family': cell.table.family.to_numpy()[cell.target_rows]}
    expected.update({f'context_{j}_tip': cell.table.tip.to_numpy()[cell.reference_rows[:, j]] for j in range(3)})
    for key, values in expected.items():
        if not np.array_equal(frame[key].to_numpy(), values):
            raise ValueError('Saved prediction/reference identity differs: ' + key)
    observed = frame[['observed_' + c for c in STATE_COLUMNS]].to_numpy(float)
    if not np.allclose(observed, cell.states[cell.target_rows], equal_nan=True, rtol=1e-7, atol=1e-7):
        raise ValueError('Saved scoring observations differ from frozen table')
    pred = frame[['predicted_' + c for c in STATE_COLUMNS]].to_numpy(float)
    if not np.isfinite(pred).all():
        raise ValueError('Nonfinite saved predictions')
    return pred


class _Backend:
    def __init__(self, cell, variant, window_indices, audit):
        self.cell, self.variant, self.audit = cell, variant, audit
        self.condition_columns = tuple(CONDITION_COLUMNS)
        self.window_indices = _window_indices(window_indices, len(cell.target_rows))
        self._raw = cell.table[list(self.condition_columns)].to_numpy(float)
        self.target_conditions = self._raw[cell.target_rows[self.window_indices]].copy()

    def _targets(self, conditions, window_indices):
        idx = self.window_indices if window_indices is None else _window_indices(window_indices, len(self.cell.target_rows))
        original = self._raw[self.cell.target_rows[idx]]
        return idx, validate_target_changes(original, conditions, self.condition_columns)

    def _gate(self, computed, saved, tolerance):
        if computed.shape != saved.shape or not np.isfinite(computed).all():
            raise ValueError('Baseline replay returned invalid four-trait predictions')
        error = float(np.max(np.abs(computed - saved)))
        if error > tolerance:
            raise ValueError(f'Frozen baseline does not reproduce saved predictions: max error {error:.9g}')
        self._baseline_all = computed
        self.baseline_predictions = computed[self.window_indices].copy()
        self.audit.update(baseline_gate_max_abs_error=error, baseline_gate_tolerance=tolerance,
                          baseline_gate_windows=len(computed),
                          selected_windows=len(self.window_indices),
                          target_measurements='Identity/scoring checks only, never predictor input.')


class ClassicBackend(_Backend):
    def __init__(self, cell, variant, window_indices, audit, fit, design, saved):
        super().__init__(cell, variant, window_indices, audit)
        from evowm.data.tree import read_trees
        from phylo_baseline_core import PhyloKernel, predict_joint
        self.design = FrozenConditionDesign(design)
        self.fit = fit
        self.beta = np.asarray(fit['beta'], float)
        if self.beta.shape != (len(design['columns']), 4) or not np.isfinite(self.beta).all():
            raise ValueError('Saved classic mean coefficients have incorrect dimensions')
        if fit['model'].lower() != variant or fit['n_morph'] != len(cell.train_rows):
            raise ValueError('Saved classic fit identity/training support differs')
        bones = np.isfinite(cell.states[cell.train_rows, 1:]).all(axis=1).sum()
        if fit['n_bone'] != bones:
            raise ValueError('Saved classic fit bone support differs')
        trees = read_trees(cell.tree_path)
        if len(trees) != 1:
            raise ValueError('Expected one frozen tree')
        required = np.unique(np.r_[cell.train_rows, cell.reference_rows.ravel(), cell.target_rows])
        if not np.array_equal(np.asarray(trees[0].tip_names)[cell.tip_ids_of_rows[required]],
                              cell.table.tip.to_numpy()[required]):
            raise ValueError('Frozen tree tip identities disagree with the table')
        kernel = PhyloKernel(trees[0])
        self.X = self.design.transform(self._raw)
        visible = visible_references(cell, 'observed')
        computed = np.empty_like(saved)
        for i, (refs, target) in enumerate(zip(cell.reference_rows, cell.target_rows)):
            computed[i] = predict_joint(kernel, fit, cell.tip_ids_of_rows[refs], self.X[refs], visible[i],
                cell.tip_ids_of_rows[target:target + 1], self.X[target:target + 1],
                ref_mask=np.isfinite(visible[i]))[0]
        self._gate(computed, saved, 1e-7)
        self.audit.update(alpha=float(fit['alpha']), eta=float(fit['eta']),
                          prediction_update='Fixed original reference posterior; target mean difference only.',
                          fit_reused=True, refitted=False)

    def predict(self, target_conditions, window_indices=None):
        idx, changed = self._targets(target_conditions, window_indices)
        new_x = self.design.transform(changed)
        old_x = self.X[self.cell.target_rows[idx]]
        result = self._baseline_all[idx] + (new_x - old_x) @ self.beta
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite classic probe prediction')
        return result


class NeuralBackend(_Backend):
    def __init__(self, cell, variant, window_indices, audit, paths, saved, batch, device):
        super().__init__(cell, variant, window_indices, audit)
        from evaluate import load_config, datasets, action_windows, AnatomicalPredictor
        from evowm.eval.recovery import Windows, load_world_model
        checkpoint, config = Path(paths['checkpoint']), Path(paths['config'])
        cfg, _ = load_config(checkpoint, config, cell.root)
        params = cfg['data']['dataset']
        for field, expected in (('species_table', cell.table_path), ('tree_file', cell.tree_path)):
            if (cell.root / params[field]).resolve() != expected:
                raise ValueError('Checkpoint configuration uses another ' + field)
        if int(params['holdout']['fold']) != int(cell.job['fold']) or int(cfg['seed']) != int(cell.job['seed']):
            raise ValueError('Checkpoint configuration has another fold or seed')
        train, validation = datasets(cfg, cell.root)
        if (not np.array_equal(np.flatnonzero(train.rows_mask), cell.train_rows)
                or not np.array_equal(np.flatnonzero(validation.rows_mask), cell.validation_rows)
                or not np.array_equal(train.S.astype(float), cell.states, equal_nan=True)):
            raise ValueError('Checkpoint dataset differs from frozen input/split')
        if tuple(train.morph_cols) != tuple(STATE_COLUMNS) or tuple(validation.eco_cols) != self.condition_columns:
            raise ValueError('Checkpoint state/condition column order differs')
        if not np.array_equal(self._raw, validation.condition_raw, equal_nan=True):
            raise ValueError('Checkpoint raw conditions differ from the frozen table')
        self.model = load_world_model(checkpoint, cell.root / 'hjepa').to(device).eval()
        validate_architecture(self.model.allen_anatomical_settings, variant, self.model.num_levels)
        if int(self.model.allen_anatomical_history) != 3:
            raise ValueError('Expected three reference species')
        self.dataset, self.batch = validation, int(batch)
        self.predictor = AnatomicalPredictor(self.model)
        self.windows = Windows(np.column_stack([cell.reference_rows, cell.target_rows]), cell.distances)
        self.actions = action_windows(validation, self.windows, 3, mode='full')
        available = np.isfinite(visible_references(cell, 'observed'))
        self.context = mask_reference_pixels(validation.pix[cell.reference_rows], available)
        computed = self.predictor.predict(self.context, self.actions, batch=self.batch)[:, -1]
        self._gate(computed, saved, 1e-5)
        self.audit.update(device=str(device), architecture=self.model.allen_anatomical_settings,
                          prediction_update='Original reference pixels and distances; rebuild actions after target-only change.',
                          refitted=False)

    def predict(self, target_conditions, window_indices=None):
        idx, changed = self._targets(target_conditions, window_indices)
        if not len(idx):
            return np.empty((0, 4), dtype=float)
        actions = []
        for position, target in zip(idx, changed):
            rows = self.windows.rows[position]
            raw = self.dataset.condition_raw[rows].copy()
            raw[-1] = target
            actions.append(self.dataset.build_action(rows, self.windows.dist[position], mode='full', raw_conditions=raw)[:3])
        actions = np.asarray(actions, dtype=np.float32)
        return self.predictor.predict(self.context[idx], actions, batch=self.batch)[:, -1]


def load_backend(source_run, fold, seed, variant, *, root=ROOT, window_indices=None,
                 batch=512, device='cpu'):
    """Load one frozen model and gate its complete original observed prediction set.

    ``window_indices`` are original saved window numbers, optionally selecting
    one/two predetermined windows per species. ``predict`` accepts N by 22 raw
    target conditions in ``condition_columns`` order; explicit indices always
    refer to the original window table, never to positions in a subset.
    """
    if int(batch) < 1:
        raise ValueError('Batch size must be positive')
    root = Path(root).resolve()
    info = inspect_checkpoint_paths(source_run, fold, seed, variant, root)
    if not info['ready']:
        raise FileNotFoundError('Frozen inference artifacts missing; retain original server weights: ' + '; '.join(info['missing']))
    run, job, manifest = find_job(source_run, fold, seed, variant, root)
    variant = info['variant']
    code = verify_inference_source(run, manifest, root, neural=variant not in ('bm', 'ou'))
    cell = load_cell(run, job['name'], root)
    result = _json(info['paths']['result'])
    if cell.hashes != result['input_sha256']:
        raise ValueError('Frozen evaluation input hashes differ from current loaded cell')
    saved = _saved_predictions(cell, variant, info['paths']['predictions'])
    audit = {'source_run': str(run), 'job': job, 'variant': variant,
             'input_sha256': dict(cell.hashes), 'artifact_sha256': info['actual_sha256'],
             'inference_source_sha256': code, 'reference_case': 'observed',
             'condition_columns': list(CONDITION_COLUMNS), 'allowed_changed_columns': list(ALLOWED_COLUMNS)}
    if variant in ('bm', 'ou'):
        return ClassicBackend(cell, variant, window_indices, audit,
                              _json(info['paths']['fit']), _json(info['paths']['design']), saved)
    return NeuralBackend(cell, variant, window_indices, audit, info['paths'], saved, batch, device)
