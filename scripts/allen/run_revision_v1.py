"""Run Allen corrections and paired bone-gradient tests; output-only on server.

python scripts/allen/run_revision_v1.py
python scripts/allen/run_revision_v1.py --check-only
"""
import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

# Set before numpy/scipy import. Two CPU threads; no GPU allocations.
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"

import numpy as np
import pandas as pd
from revision_core import (BONES, CONTRASTS, ROOT, TREES, covariance, design,
                           estimate, fit, get_tree, gradient_cases, holm, load_samples)
from test_revision_v1 import run_checks

DESIGN = ROOT / "docs/design/Allen骨段梯度与口径修订_2026-11-03.md"


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def metadata(out, args, paths):
    def git(*command):
        p = subprocess.run(["git", "-c", f"safe.directory={ROOT.as_posix()}", *command],
                           cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        return {"returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    source = [DESIGN, *Path(__file__).parent.glob("*revision*.py"),
              ROOT / "scripts/allen/devmode_v2.py", ROOT / "scripts/allen/pgls_species_v1.py",
              ROOT / "src/evowm/data/species_table.py", ROOT / "src/evowm/data/tree.py",
              ROOT / "src/evowm/eval/pgls.py"]
    inputs = [ROOT / p for p in ["data/processed/species_traits.csv", "data/processed/species_climate.csv",
              "data/raw/devmode/devmode_families.csv", "data/raw/eliason2023/S1_limb_data.csv",
              "data/raw/skelevision/Skelevision_Only_Dataset_v1.csv",
              "data/raw/avonet/AVONET_Supplementary_dataset1.xlsx"]]
    files = {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(set(source + inputs + paths))}
    info = {"created": datetime.now().astimezone().isoformat(), "arguments": vars(args),
            "python": sys.version, "git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--short"),
            "versions": {p: importlib.metadata.version(p) for p in ["numpy", "scipy", "pandas", "openpyxl", "matplotlib"]},
            "sha256": files}
    write_json(out / "manifest.json", info)
    # Keep exact changed source, not just a commit hash of an uncommitted tree.
    (out / "code_changes.patch").write_text(git("diff", "--", "scripts/allen", "docs/design")["stdout"], encoding="utf-8")
    snapshot = out / "source_snapshot"
    snapshot.mkdir()
    for p in source:
        target = snapshot / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())


def save_sample(out, name, d):
    path = out / "samples" / f"{name}.csv"
    path.parent.mkdir(exist_ok=True)
    d.to_csv(path, index=False)
    return path.relative_to(out).as_posix()


def gradient(out, trees, datasets, skeletons):
    records, diagnostics = [], []
    for tree_name, tree in trees.items():
        full = tree_name == TREES[0]
        for version, label, model, term, sub, primary in gradient_cases(datasets, skeletons, full):
            sub = sub.sort_values("tip").reset_index(drop=True)
            print(f"GRADIENT {tree_name} {version} {label} {model} n={len(sub)}", flush=True)
            X, names, dropped = design(sub, model)
            C = covariance(tree, sub.tip)
            Y = sub[BONES].to_numpy(float)
            result = fit(Y, X, C)
            sample = save_sample(out, f"{version}_{label}_{model}", sub)
            diag = {"tree": tree_name, "version": version, "sample": label, "model": model,
                    "columns": names, "dropped_nuisance": dropped, "lambda": result['lambda'],
                    "residual_covariance": result['sigma'].tolist(), "m_center": float(sub.m_mass.mean()),
                    "x_center": float(sub.X.mean()) if "X" in sub else None}
            diagnostics.append(diag)
            common = {"tree": tree_name, "version": version, "sample": label, "model": model,
                      "term": term, "n": len(sub), "sample_file": sample,
                      "n_families": sub.family.nunique(), "primary_question": primary}
            for bone, c in zip(BONES, np.eye(3)):
                records.append(dict(common, method="joint", response=bone, is_contrast=False,
                                    lambda_hat=result['lambda'], **estimate(result, names.index(term), c)))
            for label_c, c in CONTRASTS.items():
                records.append(dict(common, method="joint", response=label_c, is_contrast=True,
                                    lambda_hat=result['lambda'], **estimate(result, names.index(term), c)))
                # Only primary tree: sensitivity to a separate lambda for each paired ratio.
                if full:
                    r = fit(Y @ c, X, C)
                    records.append(dict(common, method="ratio_separate_lambda", response=label_c,
                                        is_contrast=True, lambda_hat=r['lambda'],
                                        **estimate(r, names.index(term), [1.])))
            pd.DataFrame(records).to_csv(out / "gradient_partial.csv", index=False)
    df = pd.DataFrame(records)
    df['p_holm'] = np.nan
    df['multiplicity_family'] = "descriptive_segment"
    for i, row in df[df.is_contrast].iterrows():
        if row.primary_question and row.version == "original" and row.tree == TREES[0] and row.method == "joint":
            family = "primary_six"
        else:
            family = "|".join(str(row[x]) for x in ["tree", "version", "sample", "model", "method"])
        df.loc[i, 'multiplicity_family'] = family
    for _, group in df[df.is_contrast].groupby('multiplicity_family'):
        df.loc[group.index, 'p_holm'] = holm(group.p_raw)
    df.to_csv(out / "gradient_results.csv", index=False)
    write_json(out / "gradient_diagnostics.json", diagnostics)
    report_gradient(out, df)
    return df


def verdict(group):
    positive = {r.response: r.beta > 0 and r.p_holm < .05 for r in group.itertuples()}
    a, b, c = (positive.get(k, False) for k in CONTRASTS)
    if a and b:
        return "两个相邻差值均为正并通过校正：支持近端至远端增强"
    if b and c:
        return "胫跗骨、跗跖骨均强于股骨；未检出跗跖骨强于胫跗骨"
    if c:
        return "跗跖骨强于股骨；未达到完整梯度判据"
    if a or b:
        return "仅部分相邻比较成立；未达到完整梯度判据"
    return "未检出预期正向骨段差异；负向结果和区间见表"


def report_gradient(out, df):
    lines = ["# 骨段响应直接比较", "", "T 是温度斜率；TxX 是温度与觅食环境指标的交互。两者分别解释。",
             "区间为未校正的近似 95% t 区间，判定用 Holm 校正 p；不把股骨未显著写成股骨为零。", ""]
    contrasts = df[df.is_contrast]
    for keys, g in contrasts.groupby(['tree', 'version', 'sample', 'model', 'method'], sort=False):
        lines += ["## " + " / ".join(keys), "", f"n={int(g.n.iloc[0])}；{verdict(g)}。", "",
                  "| 比较 | 估计 [95% 区间] | 校正 p |", "|---|---|---|"]
        for r in g.itertuples():
            lines.append(f"| {r.response} | {r.beta:+.5f} [{r.lo95:+.5f}, {r.hi95:+.5f}] | {r.p_holm:.5g} |")
        lines.append("")
    (out / "gradient_report.md").write_text("\n".join(lines), encoding="utf-8")


def figure2(out, tree, datasets, skeletons):
    # Descriptive univariate refits, matching the historical four panels.
    A = datasets['original']
    groups = {"Eliason_precocial": A[(A.order != "Passeriformes") & (A['mode'] == "P")],
              "Watanabe": skeletons['Watanabe'],
              "Eliason_passerines": A[A.order == "Passeriformes"],
              "Skelevision_passerines": skeletons['Skelevision_passerines']}
    records = []
    from evowm.eval.pgls import pgls_lambda
    for label, sub in groups.items():
        X, names, _ = design(sub, "slope")
        C = covariance(tree, sub.tip)
        for bone in [*BONES, "tmt_minus_fem"]:
            y = (sub.tmt - sub.fem).to_numpy() if bone == "tmt_minus_fem" else sub[bone].to_numpy()
            r = pgls_lambda(y, X, C, names)
            b, se = r['coef']['T']['beta'], r['coef']['T']['se']
            records.append({"sample": label, "response": bone, "n": len(sub), "beta": b, "se": se,
                            "lo95": b - 1.96 * se, "hi95": b + 1.96 * se,
                            "lambda": r['lambda'], "method": "historical_univariate_PGLS_normal_CI"})
    table = pd.DataFrame(records)
    table.to_csv(out / "fig2_source.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.8), sharex=True)
    labels = ['Femur', 'Tibiotarsus', 'Tarsometatarsus', 'log10(Tmt/Fem)']
    for ax, (label, g) in zip(axes.flat, table.groupby('sample', sort=False)):
        ax.axvline(0, color='0.7', linewidth=1)
        ax.errorbar(g.beta, range(4), xerr=[g.beta - g.lo95, g.hi95 - g.beta], fmt='o', capsize=3)
        ax.set_yticks(range(4), labels)
        ax.invert_yaxis()
        ax.set_title(f"{label.replace('_', ' ')}\nn = {int(g.n.iloc[0])}", fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
    fig.supxlabel('Temperature slope: log10 change per 10 C (95% CI)')
    fig.tight_layout()
    fig.savefig(out / "fig2_corrected.png", dpi=220)
    fig.savefig(out / "fig2_corrected.pdf")
    plt.close(fig)
    return table


def fair_aic(out, trees, external):
    records = []
    for tree_name, tree in trees.items():
        samples = {'all': external}
        if tree_name == TREES[0]:
            samples.update({'nonpass': external[external.order != 'Passeriformes'],
                            'passerines': external[external.order == 'Passeriformes']})
        for label, sub in samples.items():
            C = covariance(tree, sub.tip)
            sample = save_sample(out, 'external_' + label, sub)
            full_X, full_names, dropped = design(sub, 'M3', external.m_mass.mean(), external.X.mean())
            for model in ['M0', 'M1', 'M2', 'M3']:
                print(f"FAIR AIC {tree_name} {label} {model} n={len(sub)}", flush=True)
                excluded = {'M0': ['TxM', 'TxX'], 'M1': ['TxX'], 'M2': ['TxM'], 'M3': []}[model]
                names = [name for name in full_names if name not in excluded]
                X = full_X[:, [full_names.index(name) for name in names]]
                r = fit(sub.m_tarsus.to_numpy(), X, C)
                rec = {'tree': tree_name, 'sample': label, 'model': model, 'n': len(sub),
                       'sample_file': sample, 'lambda': r['lambda'], 'p': len(names),
                       'loglik': r['loglik'], 'aic': 2 * (len(names) + 2) - 2 * r['loglik'],
                       'columns': '|'.join(names), 'dropped_nuisance': '|'.join(dropped)}
                for term in ['T', 'TxM', 'TxX']:
                    if term in names:
                        rec.update({f'{term}_{k}': v for k, v in estimate(r, names.index(term), [1.]).items()})
                records.append(rec)
                pd.DataFrame(records).to_csv(out / 'fair_aic_results.csv', index=False)
    lines = ['# 相同主效应下的两轴模型比较', '', '所有模型包含 T、M、X 和相同协变量。差值为前者 AIC 减后者。', '',
             '| 树 | 样本 | n | M2−M1 | M3−M1 | M3−M2 |', '|---|---|---|---|---|---|']
    for (tree_name, label), g in pd.DataFrame(records).groupby(['tree', 'sample'], sort=False):
        a = g.set_index('model').aic
        lines.append(f'| {tree_name} | {label} | {int(g.n.iloc[0])} | {a.M2-a.M1:+.3f} | {a.M3-a.M1:+.3f} | {a.M3-a.M2:+.3f} |')
    (out / 'fair_aic_report.md').write_text('\n'.join(lines), encoding='utf-8')


class Tee:
    def __init__(self, original, logfile):
        self.original, self.logfile = original, logfile
    def write(self, value):
        self.original.write(value)
        self.logfile.write(value)
        self.logfile.flush()
    def flush(self):
        self.original.flush()
        self.logfile.flush()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['all', 'gradient', 'aic'], default='all')
    p.add_argument('--trees', choices=['all', 'primary'], default='all')
    p.add_argument('--check-only', action='store_true')
    args = p.parse_args()
    out = ROOT / 'outputs' / ('allen_revision_v1_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True)
    logfile = (out / 'console.log').open('w', encoding='utf-8', buffering=1)
    oldout, olderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = Tee(oldout, logfile), Tee(olderr, logfile)
    try:
        print(f'OUTPUT DIRECTORY: {out}', flush=True)
        if not DESIGN.exists():
            raise FileNotFoundError(DESIGN)
        checks = run_checks()
        write_json(out / 'implementation_checks.json', checks)
        print('PASS: numerical and real-tree synthetic checks (two seeds)', flush=True)
        names = TREES if args.trees == 'all' else TREES[:1]
        loaded = {name: get_tree(name) for name in names}
        trees = {name: obj[0] for name, obj in loaded.items()}
        common = set.intersection(*(set(tree.tip_names) for tree in trees.values()))
        metadata(out, args, [obj[1] for obj in loaded.values()])
        datasets, skeletons, external, overlap = load_samples(common)
        overlap.to_csv(out / 'catalog_overlap_records.csv', index=False)
        counts = []
        for version, label, model, term, sub, primary in gradient_cases(datasets, skeletons, True):
            X, cols, dropped = design(sub, model)
            if sub.tip.duplicated().any():
                raise ValueError(f'Duplicate tips: {label}')
            counts.append({'version': version, 'sample': label, 'model': model, 'term': term,
                           'n': len(sub), 'n_families': sub.family.nunique(), 'columns': '|'.join(cols),
                           'dropped_nuisance': '|'.join(dropped)})
        pd.DataFrame(counts).to_csv(out / 'sample_counts.csv', index=False)
        print(pd.DataFrame(counts)[['version', 'sample', 'model', 'n']].to_string(index=False), flush=True)
        print(f'Catalog matches: {overlap.catalog_key.nunique()}; external sample: {len(external)}', flush=True)
        if args.check_only:
            write_json(out / 'CHECK_ONLY_SUCCESS.json', {'status': 'input_and_implementation_checks_only'})
            print('CHECK ONLY PASSED; no real phenotype models fitted.', flush=True)
            return
        if args.stage in ['all', 'gradient']:
            gradient(out, trees, datasets, skeletons)
            figure2(out, trees[TREES[0]], datasets, skeletons)
        if args.stage in ['all', 'aic']:
            fair_aic(out, trees, external)
        write_json(out / 'SUCCESS.json', {'status': 'complete', 'stage': args.stage, 'trees': names,
                                          'finished': datetime.now().astimezone().isoformat()})
        print(f'SUCCESS. Download this entire directory: {out}', flush=True)
    except BaseException:
        traceback.print_exc()
        print(f'FAILED. Retained logs: {out}', flush=True)
        raise SystemExit(1)
    finally:
        sys.stdout, sys.stderr = oldout, olderr
        logfile.close()


if __name__ == '__main__':
    main()
