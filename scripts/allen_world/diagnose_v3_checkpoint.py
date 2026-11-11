"""Frozen v3 checkpoint diagnostics; never train or overwrite the source run.

Interventions measure model dependence, not ecological causality or an attainable
predictive gain. Prediction always receives reference morphology only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as ev

OUTCOMES = ['m_tarsus', 'b_fem', 'b_tbt', 'b_tmt', 'log10_tmt_over_fem', *ev.ANATOMICAL_OUTCOMES]


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def matched_permutation(log_total, seed, bins=10):
    """Within size-rank bins, cyclically permute all windows, independently of batches.

    No observed target or prediction error enters binning. A singleton cannot
    move. Matching predicted size mitigates but cannot eliminate distribution
    shift; donors may still represent the same biological target.
    """
    values = np.asarray(log_total, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or bins < 1:
        raise ValueError('Expected nonempty finite predicted totals and positive bins')
    rng = np.random.default_rng(seed)
    order = np.argsort(values, kind='stable')
    donor, labels = np.arange(len(values)), np.empty(len(values), dtype=int)
    for label, group in enumerate(np.array_split(order, min(bins, max(1, len(values)//2)))):
        labels[group] = label
        shuffled = rng.permutation(group)
        if len(group) > 1:
            donor[shuffled] = np.roll(shuffled, int(rng.integers(1, len(group))))
    return donor, labels


def trace_predictions(model, context, actions, batch):
    """Capture the unconditioned fine state at the canonical adapter boundary."""
    import torch
    from evowm.models.allen_anatomical import predict_from_context
    if len(context) != len(actions) or not len(context) or batch < 1:
        raise ValueError('Matching nonempty inputs and positive batch required')
    model.eval()
    head = model.get_level(1).anatomical_readout
    chunks = {}
    for start in range(0, len(context), batch):
        captured = []
        handle = head.adapter.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0].detach()))
        try:
            with torch.no_grad():
                output = predict_from_context(model, torch.as_tensor(context[start:start+batch], dtype=torch.float32),
                                              torch.as_tensor(actions[start:start+batch], dtype=torch.float32))
        finally:
            handle.remove()
        if len(captured) != 1:
            raise AssertionError('Expected exactly one canonical adapter call')
        item = {k: v[:, -1].detach().cpu().numpy() for k, v in output.items() if k.startswith('predicted_')}
        item['fine_base'] = captured[0][:, -1, :head.fine_dim].cpu().numpy()
        item['conditioning_scalar'] = captured[0][:, -1, -1].cpu().numpy()
        for key, value in item.items():
            if not np.isfinite(value).all():
                raise ValueError('Nonfinite canonical output: ' + key)
            chunks.setdefault(key, []).append(value)
    return {k: np.concatenate(v) for k, v in chunks.items()}


def intervention_predictions(model, trace, train_rich_mean, permutations, batch, permutation_seed=20261111):
    """Change only upper conditioning; retain the original reconstructed total."""
    import torch
    head = model.get_level(1).anatomical_readout
    settings = model.allen_anatomical_settings
    predictions = {'baseline': trace['predicted_log_lengths']}
    if settings['mode'] == 'single':
        return predictions, []
    rich = trace['predicted_embedding_2']
    total = trace['predicted_log_total']
    # Capture the exact scalar used in the canonical call, avoiding floating
    # point reconstruction and retaining exact no-link negative controls.
    scalar = trace['conditioning_scalar']
    modifications = [('zero_rich', np.zeros_like(rich), settings['connection']),
                     ('mean_rich', np.broadcast_to(train_rich_mean, rich.shape).copy(), settings['connection']),
                     ('zero_upper', rich, 'none')]
    donor_records = []
    for repeat in range(permutations):
        donor, bins = matched_permutation(total, permutation_seed + repeat)
        name = f'shuffle_rich_{repeat:02d}'
        modifications.append((name, rich[donor], settings['connection']))
        donor_records.append({'method': name, 'donor': donor, 'bin': bins})
    for name, changed_rich, connection in modifications:
        values = []
        with torch.no_grad():
            for start in range(0, len(total), batch):
                idx = slice(start, start + batch)
                refined = head.condition(torch.as_tensor(trace['fine_base'][idx]), torch.as_tensor(changed_rich[idx]),
                                         torch.as_tensor(scalar[idx]), connection)
                values.append(head.decode(refined, torch.as_tensor(total[idx]))['predicted_log_lengths'].numpy())
        predictions[name] = np.concatenate(values)
        if not np.isfinite(predictions[name]).all():
            raise ValueError('Nonfinite intervention: ' + name)
    return predictions, donor_records


def latent_statistics(values):
    values = np.asarray(values, dtype=float)
    variance = np.var(values, axis=0)
    eigen = np.linalg.eigvalsh(np.atleast_2d(np.cov(values, rowvar=False, bias=True))).clip(min=0)
    p = eigen / eigen.sum() if eigen.sum() > 0 else eigen
    positive = p[p > 0]
    return {'n': len(values), 'dimension': values.shape[1], 'mean': values.mean(axis=0).tolist(),
            'std': np.sqrt(variance).tolist(), 'effective_rank': float(np.exp(-np.sum(positive*np.log(positive)))) if len(positive) else 0.,
            'covariance_trace': float(eigen.sum()), 'note': 'Representation variation; not feature importance.'}


def gradient_diagnostics(model, cfg, ds, windows, actions, max_per_group=32):
    """Training-only, final-target, eval-mode directional gradient diagnostics.

    Excludes SIGReg, dropout, earlier positions and masked conditions: not the
    complete stochastic training objective. Nothing is optimized or updated.
    """
    import torch
    from evowm.models.allen_anatomical import predict_from_context, anatomical_coordinates, _encode, _masked_mse
    from evowm.models.allen_readout import masked_standardized_mse
    rows = windows.rows[:, -1]
    complete = np.isfinite(ds.S[rows, 1:]).all(axis=1)
    measured = np.isfinite(ds.S[rows]).any(axis=1)
    rng = np.random.default_rng(20261111)
    indices = np.concatenate([rng.permutation(np.flatnonzero(selected))[:max_per_group]
                              for selected in (complete, measured & ~complete)])
    if not len(indices):
        return {'status': 'no_training_targets'}
    if not ds.rows_mask[windows.rows[indices]].all():
        raise AssertionError('Gradient diagnostic may only use training species')
    pixels = torch.as_tensor(ds.pix[windows.rows[indices, :-1]], dtype=torch.float32)
    target_pixels = torch.as_tensor(ds.pix[rows[indices]], dtype=torch.float32)
    targets = torch.as_tensor(ds.S[rows[indices]], dtype=torch.float32)
    model.eval()
    out = predict_from_context(model, pixels, torch.as_tensor(actions[indices], dtype=torch.float32))
    head = model.get_level(1).anatomical_readout
    total, ratios, valid = anatomical_coordinates(targets)
    settings = model.allen_anatomical_settings
    losses = {'observation': settings['observation_weight'] * masked_standardized_mse(out['predicted_log_lengths'][:, -1], targets, head.morph_sd),
              'total_auxiliary': settings['total_weight'] * _masked_mse(out['predicted_log_total'][:, -1]/head.total_sd, total/head.total_sd, valid),
              'ratio_auxiliary': settings['ratio_weight'] * _masked_mse(out['predicted_log_ratios'][:, -1]/head.ratio_sd, ratios/head.ratio_sd, valid)}
    losses['supervised_sum'] = sum(losses.values())
    if settings['objective'] == 'jepa':
        target_embedding = _encode(model.get_level(1).encoder, target_pixels)
        for level in range(1, int(model.num_levels)+1):
            if level == 2:
                target_embedding = _encode(model.get_level(2).encoder, target_embedding)
            setting = cfg[f'level{level}']
            component = setting['loss'].get('embed', setting['loss']).get('pred', {})
            if component.get('enabled', True):
                target = target_embedding.detach() if setting['wm'].get('detach_pred_target', False) else target_embedding
                mask = torch.isfinite(targets).any(dim=-1) if level == 1 else valid
                losses[f'latent_prediction_level{level}'] = float(component.get('weight', 1.)) * _masked_mse(
                    out[f'predicted_embedding_{level}'][:, -1], target, mask)
    groups = {'encoder_level1': list(model.get_level(1).encoder.parameters())}
    if int(model.num_levels) == 2:
        groups['encoder_level2'] = list(model.get_level(2).encoder.parameters())
    result = {'status': 'ok', 'n_windows': len(indices), 'n_complete': int(valid.sum()),
              'target_tips': ds.table.tip.to_numpy()[rows[indices]].tolist(),
              'losses': {k: float(v.detach()) for k,v in losses.items()}, 'parameter_groups': {},
              'scope': 'Training-only final target, full conditions, eval mode. Weighted observation/coordinate and latent prediction losses; excludes SIGReg, dropout, earlier positions and other mask modes. Local diagnostic, no parameter update.'}
    for name, parameters in groups.items():
        parameters = [p for p in parameters if p.requires_grad]
        vectors = {}
        for key, loss in losses.items():
            gradient = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
            vector = torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(parameters,gradient)]).detach()
            if not torch.isfinite(vector).all():
                raise ValueError('Nonfinite diagnostic gradient')
            vectors[key] = vector
        cosines = {}
        reference = vectors['supervised_sum']
        for key, value in vectors.items():
            denom = torch.linalg.vector_norm(reference) * torch.linalg.vector_norm(value)
            cosines[key] = float(torch.clamp(torch.dot(reference,value)/denom, -1., 1.)) if float(denom) > 0 else None
        result['parameter_groups'][name] = {'gradient_norms': {k:float(torch.linalg.vector_norm(v)) for k,v in vectors.items()},
                                           'cosine_with_supervised_sum': cosines}
    return result


def assert_identity(ds, windows, original):
    target, source = windows.rows[:, -1], windows.rows[:, -2]
    data = {'window': np.arange(len(target)), 'target_tip': ds.table.tip.to_numpy()[target],
            'source_tip': ds.table.tip.to_numpy()[source], 'family': ds.table.family.to_numpy()[target]}
    data.update({f'context_{i}_tip': ds.table.tip.to_numpy()[windows.rows[:, i]] for i in range(windows.rows.shape[1]-1)})
    identity = pd.DataFrame(data)
    pd.testing.assert_frame_equal(identity, original[list(identity)], check_dtype=False)
    observed = ds.S[target].astype(float)
    np.testing.assert_allclose(observed, original[['observed_'+c for c in ['m_tarsus','b_fem','b_tbt','b_tmt']]], atol=1e-12, rtol=0, equal_nan=True)
    complete = np.isfinite(observed[:,1:]).all(axis=1)
    if not np.array_equal(complete, original.complete_three_bones.to_numpy(bool)):
        raise AssertionError('Complete-bone support changed')
    return identity, observed, complete


def verify_provenance(root, run, checkpoint, config_path, cfg, old, manifest):
    source = old['provenance']
    table = root / cfg['data']['dataset']['species_table']
    tree = root / cfg['data']['dataset']['tree_file']
    paths = {'checkpoint_sha256': checkpoint, 'config_sha256': config_path, 'table_sha256': table,
             'table_metadata_sha256': table.with_suffix('.columns.json'), 'tree_sha256': tree}
    hashes = {key:file_sha(path) for key,path in paths.items()}
    if any(hashes[key] != source[key] for key in paths):
        raise ValueError('Checkpoint/config/input differs from original evaluation: '+str(hashes))
    # Inference/transect code must be the original code; only newline conversion
    # is tolerated between Git platforms. New diagnostic modules are additional.
    critical = [p for p in manifest['code_sha256'] if p.startswith(('src/evowm/', 'hjepa/h_jepa/'))]
    critical.append('scripts/allen_world/evaluate.py')
    for rel in critical:
        snapshot = run / 'source' / rel
        if file_sha(snapshot) != manifest['code_sha256'][rel]:
            raise ValueError('Original source snapshot modified: '+rel)
        if snapshot.read_bytes().replace(b'\r\n', b'\n') != (root/rel).read_bytes().replace(b'\r\n', b'\n'):
            raise ValueError('Inference source changed since training: '+rel)
    return hashes | {'inference_source_files_verified': len(critical)}


def diagnose(args):
    import torch
    from threadpoolctl import threadpool_limits
    from evowm.eval.recovery import load_world_model, sample_windows
    root, run, out_dir = args.root.resolve(), args.run.resolve(), args.out_dir.resolve()
    if out_dir == run or run in out_dir.parents:
        raise ValueError('Diagnostics must be written outside the original run')
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError('Diagnostic output must be new or empty')
    if not re.fullmatch(r'[A-Za-z0-9_]+', args.job):
        raise ValueError('Invalid job name')
    manifest = json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    jobs = [j for j in manifest['jobs'] if j['name'] == args.job]
    if len(jobs) != 1:
        raise ValueError('Job must occur exactly once in original manifest')
    job = jobs[0]
    saved = run/'evaluation'/args.job
    old = json.loads((saved/'pilot_results.json').read_text(encoding='utf-8'))
    checkpoint = run/'training/ckpts'/args.job/'allen_v3_object.ckpt'
    cfg, config_path = ev.load_config(checkpoint, checkpoint.parent/'config.yaml', root)
    hashes = verify_provenance(root, run, checkpoint, config_path, cfg, old, manifest)
    model = load_world_model(checkpoint, root/'hjepa').cpu().eval()
    settings = model.allen_anatomical_settings
    for key in ('mode','connection','objective'):
        if settings[key] != old['prediction_readout'][key]:
            raise ValueError('Saved model identity changed: '+key)
    if int(model.num_levels) != old['prediction_readout']['levels']:
        raise ValueError('Saved model level count changed')
    torch.set_num_threads(args.threads)
    ds_tr, ds_va = ev.datasets(cfg, root)
    history, seed = int(old['history']), int(old['seed'])
    windows = sample_windows(ds_va, int(old['n_orders']), history+1, seed+1)
    train_windows = sample_windows(ds_tr, int(old['n_orders']), history+1, seed)
    original = pd.read_csv(saved/'prediction_windows.csv.gz')
    identity, observed, complete = assert_identity(ds_va, windows, original)
    if not ds_va.rows_mask[windows.rows].all() or not ds_tr.rows_mask[train_windows.rows].all():
        raise AssertionError('Window split mismatch')
    actions = ev.action_windows(ds_va, windows, history)
    train_actions = ev.action_windows(ds_tr, train_windows, history)
    with threadpool_limits(limits=args.threads):
        trace = trace_predictions(model, ds_va.pix[windows.rows[:,:history]], actions, args.batch)
        baseline_saved = original[['jepa_full_'+c for c in ['m_tarsus','b_fem','b_tbt','b_tmt']]].to_numpy()
        error = float(np.max(np.abs(trace['predicted_log_lengths'] - baseline_saved)))
        if error > 1e-5:
            raise ValueError(f'Canonical baseline differs from original evaluation (max {error:g})')
        train_trace = trace_predictions(model, ds_tr.pix[train_windows.rows[:,:history]], train_actions, args.batch)
        mean = train_trace['predicted_embedding_2'].mean(axis=0) if settings['mode']=='hierarchical' else None
        predictions, donors = intervention_predictions(model, trace, mean, args.permutations, args.batch)
        gradient = gradient_diagnostics(model, cfg, ds_tr, train_windows, train_actions)
    data = identity.copy()
    for col in ('target_order','target_mode','target_X','target_m_mass'):
        if col in original: data[col] = original[col]
    data['complete_three_bones'] = complete
    observed_coords = ev.outcome_values(observed, list(ds_tr.state_cols), OUTCOMES)
    fields = {f'observed_{name}': observed_coords[:,i] for i,name in enumerate(OUTCOMES)}
    changes = {}
    for method, lengths in predictions.items():
        coords = ev.outcome_values(lengths, list(ds_tr.state_cols), OUTCOMES)
        fields.update({f'{method}_{name}':coords[:,i] for i,name in enumerate(OUTCOMES)})
        changes[method] = float(np.max(np.abs(lengths-predictions['baseline'])))
        if method != 'baseline':
            total_change = np.max(np.abs(coords[:,OUTCOMES.index('log10_total_leg')] - trace['predicted_log_total']))
            if total_change > 2e-6:
                raise AssertionError('Intervention changed reconstructed total')
    data = pd.concat([data,pd.DataFrame(fields)],axis=1)
    for donor in donors:
        data[donor['method']+'_donor_window'] = donor['donor']
        data[donor['method']+'_size_bin'] = donor['bin']
    donor_summary = []
    for donor in donors:
        idx = donor['donor']
        gap = np.abs(trace['predicted_log_total'][idx] - trace['predicted_log_total'])
        donor_summary.append({'method':donor['method'],
                              'different_target_fraction':float(np.mean(identity.target_tip.to_numpy()[idx] != identity.target_tip.to_numpy())),
                              'different_family_fraction':float(np.mean(identity.family.to_numpy()[idx] != identity.family.to_numpy())),
                              'mean_absolute_predicted_log_total_gap':float(gap.mean()),
                              'max_absolute_predicted_log_total_gap':float(gap.max())})
    negative_controls = [name for name in predictions if name != 'baseline' and
                         (settings['connection'] == 'none' or (settings['connection'] == 'scalar' and name != 'zero_upper'))]
    if any(changes[name] > 2e-6 for name in negative_controls):
        raise AssertionError('An originally masked upper channel changed predictions')
    stats = {}
    if settings['mode']=='hierarchical':
        stats = {split:latent_statistics(t['predicted_embedding_2']) for split,t in [('training',train_trace),('held_out',trace)]}
    head = model.get_level(1).anatomical_readout
    weights = head.adapter[0].weight.detach().numpy()
    weight_stats = {'fine_block_frobenius':float(np.linalg.norm(weights[:,:head.fine_dim])),
                    'rich_block_frobenius':float(np.linalg.norm(weights[:,head.fine_dim:head.fine_dim+head.global_dim])),
                    'scalar_block_frobenius':float(np.linalg.norm(weights[:,-1:])),
                    'note':'Raw weight norms are not feature importance; input scales and downstream effects differ.'}
    result = {'schema_version':1, 'job':job, 'prediction_methods':list(predictions), 'settings':dict(settings),
              'n_windows':len(data), 'baseline_max_absolute_difference':error, 'baseline_tolerance':1e-5,
              'maximum_absolute_prediction_change':changes, 'provenance':hashes,
              'negative_controls':negative_controls, 'negative_controls_verified':True,
              'upper_latent_statistics':stats, 'adapter_first_layer_weight_norms':weight_stats,
              'conditioning_input_std':{'fine':trace['fine_base'].std(axis=0).tolist(),
                                        'scalar':float(trace['conditioning_scalar'].std())},
              'permutation_donor_summary':donor_summary,
              'gradient_diagnostics':gradient,
              'intervention_scope':'Frozen final-target readout only. Preserve original predicted total in bone reconstruction. Original connection masks respected. Rich mean uses training reference windows only. Size-matched shuffles use predicted totals, not observations, and span all held-out windows. These may be out-of-distribution and measure model dependence, not causal biological effects or retrained accuracy.',
              'donor_seed':20261111, 'permutations':args.permutations,
              'worker_sha256':file_sha(__file__)}
    out_dir.mkdir(parents=True,exist_ok=True)
    data.to_csv(out_dir/'diagnostic_windows.csv.gz',index=False,compression='gzip')
    (out_dir/'diagnosis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'job':args.job,'status':'complete','windows':len(data),'baseline_max_error':error}),flush=True)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--job',required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--batch',type=int,default=512)
    p.add_argument('--permutations',type=int,default=5)
    return p


def main():
    args = parser().parse_args()
    if min(args.threads,args.batch,args.permutations)<1:
        raise ValueError('threads, batch and permutations must be positive')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    diagnose(args)


if __name__ == '__main__':
    main()
