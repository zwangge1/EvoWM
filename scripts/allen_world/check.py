"""Validate all folds; optionally exercise original JEPA forward/backward on CPU.

This is an implementation check, NOT a simulation-recovery or biological pass.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "hjepa/h_jepa")]


def config_for(name, table, tree, fold):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    os.environ["EVOWM_ROOT"] = str(ROOT)
    os.environ["HJEPA_HOME"] = str(ROOT / "outputs/hjepa")
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs/hjepa")):
        cfg = compose(config_name=name, overrides=[f"data.dataset.species_table={table}",
                      f"data.dataset.tree_file={tree}", f"data.dataset.holdout.fold={fold}"])
    return cfg


def check_one(task):
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from evowm.data.allen import AllenDataset, MASK_MODES
    torch.set_num_threads(1)
    name, table, tree, fold = task
    cfg = config_for(name, table, tree, fold)
    kw = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    kw.pop("type")
    val_name = kw.pop("val_name")
    kw.update(n_orders=2, refresh_every=0)
    tr = AllenDataset(**kw)
    kw["name"] = val_name
    va = AllenDataset(**kw)
    assert not np.any(tr.rows_mask & va.rows_mask)
    np.testing.assert_array_equal(tr.morph_mu, va.morph_mu)
    np.testing.assert_array_equal(tr.condition_mu, va.condition_mu)
    assert tr.pixel_dim == cfg.level1.encoder.input_dim
    assert tr.action_dim == cfg.level1.wm.action_dim
    samples = {}
    for mode in tr.mask_modes:
        sample = tr.sample(np.random.default_rng(71), mode=mode)
        assert all(torch.isfinite(v).all() for k, v in sample.items() if "pixels" in k or "action" in k)
        assert all(tr.rows_mask[sample["rows_level1"].numpy()])
        samples[mode] = sample
    a, b = tr.action_layout["background"]
    for mode in tr.mask_modes:
        torch.testing.assert_close(samples[mode]["action_level1"][:, a:], samples["full"]["action_level1"][:, a:])
    table_frame = tr.table
    counts = {}
    for label, ds in (("train", tr), ("val", va)):
        sub = table_frame.loc[ds.rows_mask]
        counts[label] = {"species": len(sub), "families": int(sub.family.nunique()),
                         "measured": {c: int(sub[c].notna().sum()) for c in ds.state_cols}}
    return {"config": name, "fold": fold, "counts": counts, "audit": tr.audit()}


def smoke_model(name, table, tree):
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from evowm.data.allen import AllenDataset
    from evowm.models.residual import enable_residual
    torch.set_num_threads(2)
    torch.manual_seed(19)
    cfg = config_for(name, table, tree, 0)
    kw = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    kw.pop("type"); kw.pop("val_name")
    kw.update(n_orders=1, refresh_every=0)
    ds = AllenDataset(**kw)
    if cfg.get("evowm_allen_readout", False):
        from evowm.models.allen_readout import create_world_model, allen_forward as forward, configure_readout
    elif cfg.get("evowm_units", False):
        from evowm.models.units import create_world_model, units_forward as forward
    else:
        from hjepa_utils import create_world_model, hjepa_forward as forward
    model, losses = create_world_model(cfg)
    if cfg.get("evowm_allen_readout", False):
        configure_readout(model, ds)
    if cfg.get("evowm_residual", False):
        enable_residual(model)

    class Harness(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model
            for k, v in losses.items():
                setattr(self, k, v)
        def log_dict(self, *args, **kwargs):
            pass

    harness = Harness()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    values = []
    readout_checks = []
    for step in range(2):
        samples = [ds.sample(np.random.default_rng(100 + step * 4 + j)) for j in range(4)]
        if cfg.get("evowm_allen_readout", False):
            history = int(cfg.level1.wm.history_size)
            def has_bone_triplet(sample):
                target_bones = sample["state_level1"][1:history + 1, 1:4]
                return bool(torch.isfinite(target_bones).all(dim=-1).any())
            # The B=4 smoke can otherwise contain no observed skeletons and
            # silently exercise only the external-tarsus head. This deterministic
            # sampling is ONLY for implementation checks, never real training.
            if not any(has_bone_triplet(sample) for sample in samples):
                for attempt in range(10000):
                    candidate = ds.sample(np.random.default_rng(100000 + step * 100000 + attempt))
                    if has_bone_triplet(candidate):
                        samples[0] = candidate
                        break
                else:
                    raise ValueError("Could not find measured bone targets for readout smoke check")
        batch = {k: torch.stack([s[k] for s in samples]) for k in samples[0]}
        if cfg.get("evowm_allen_readout", False):
            heads = model.get_level(1).allen_readout.heads
            before = [[p.detach().clone() for p in head.parameters()] for head in heads]
            bone_labels = int(torch.isfinite(batch["state_level1"][:, 1:history + 1, 1:4]).sum())
            assert bone_labels >= 3
        opt.zero_grad()
        out = forward(harness, batch, "train", cfg, normalize_batch=lambda b: None)
        loss = out["loss"]
        assert torch.isfinite(loss)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        if cfg.get("evowm_allen_readout", False):
            gradient_norms = []
            for head in heads:
                gradients = [p.grad for p in head.parameters() if p.grad is not None]
                assert gradients and all(torch.isfinite(g).all() for g in gradients)
                norm = float(torch.sqrt(sum(g.detach().square().sum() for g in gradients)))
                assert norm > 0, "Readout head has no gradient despite measured targets"
                gradient_norms.append(norm)
        opt.step()
        if cfg.get("evowm_allen_readout", False):
            changes = [float(torch.sqrt(sum((p.detach() - old).square().sum()
                       for p, old in zip(head.parameters(), previous))))
                       for head, previous in zip(heads, before)]
            assert all(change > 0 for change in changes)
            readout_checks.append({"measured_bone_labels": bone_labels,
                                   "head_gradient_l2": gradient_norms,
                                   "head_parameter_change_l2": changes})
        values.append(float(loss.detach()))
    return {"config": name, "optimizer_steps": 2, "loss": values,
            "parameters": sum(p.numel() for p in model.parameters()),
            "readout_checks": readout_checks,
            "note": "CPU implementation smoke only; no biological inference"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--table", default="outputs/allen_world_v1/input/species_traits.csv")
    p.add_argument("--tree", default="data/processed/trees/hackett_t021.nwk")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--model-smoke", action="store_true")
    p.add_argument("--configs", nargs="+", default=["allen_world_v1_l1", "allen_world_v1_l2"])
    p.add_argument("--out-dir", type=Path, default=ROOT / "outputs/allen_world_v1/check")
    args = p.parse_args()
    if args.workers < 1:
        p.error("workers must be positive")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    names = args.configs
    tasks = [(n, args.table, args.tree, f) for n in names for f in range(5)]
    if args.workers == 1:
        results = list(map(check_one, tasks))
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
            results = list(pool.map(check_one, tasks))
    smoke = [smoke_model(n, args.table, args.tree) for n in names] if args.model_smoke else []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {"implementation_pass": True, "scientific_recovery_pass": None,
              "workers_requested": args.workers, "checks": results, "model_smoke": smoke}
    (args.out_dir / "check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"implementation_pass": True, "fold_configs": len(results), "smoke": smoke}))


if __name__ == "__main__":
    main()
