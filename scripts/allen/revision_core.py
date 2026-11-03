"""Matched samples and matrix-normal PGLS for the 2026-11-03 follow-up.

Historical Allen scripts remain unchanged. See the dated design document.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize_scalar
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from devmode_v2 import LIFE, base, crosswalk, load_eliason
from evowm.data.species_table import _tip
from evowm.data.tree import read_trees
from pgls_species_v1 import load as skeleton_table

BONES = ["fem", "tbt", "tmt"]
CONTRASTS = {"tmt_minus_tbt": np.array([0., -1., 1.]),
             "tbt_minus_fem": np.array([-1., 1., 0.]),
             "tmt_minus_fem": np.array([-1., 0., 1.])}
TREES = [f"hackett_t{i:03d}" for i in (21, 41, 61, 81, 100)] + [
    f"ericson_t{i:03d}" for i in (21, 41, 61)]


def get_tree(name):
    """Read processed tree, or extract in memory; never change server data/."""
    path = ROOT / f"data/processed/trees/{name}.nwk"
    if path.exists():
        return read_trees(path, max_trees=1)[0], path
    backbone, index = name.split("_t")
    path = ROOT / f"data/raw/birdtree/{backbone}_100.nwk"
    from evowm.data.tree import parse_newick
    parts = [s.strip() for s in path.read_text().split(";") if s.strip()]
    if len(parts) != 100:
        raise ValueError(f"Expected 100 trees: {path}; found {len(parts)}")
    return parse_newick(parts[int(index) - 1] + ";"), path


def covariance(tree, tips):
    """Same root/LCA covariance as tip_vcv, allocated only for requested tips."""
    tips = list(tips)
    if len(tips) != len(set(tips)):
        raise ValueError("Duplicate species in covariance request")
    wanted = set(tips)
    ordered = [x for x in tree.tip_names if x in wanted]
    if len(ordered) != len(tips):
        raise ValueError("Species absent from tree")
    pos = {x: i for i, x in enumerate(ordered)}
    n = len(tips)
    lo = np.full(len(tree.parent), n, dtype=int)
    hi = np.full(len(tree.parent), -1, dtype=int)
    diag = np.empty(n)
    for node, tip in zip(tree.tip_nodes, tree.tip_names):
        if tip in pos:
            j = pos[tip]
            lo[node], hi[node], diag[j] = j, j + 1, tree.depth[node]
    for v in range(len(tree.parent) - 1, 0, -1):
        p = tree.parent[v]
        lo[p], hi[p] = min(lo[p], lo[v]), max(hi[p], hi[v])
    C = np.zeros((n, n))
    for v in range(len(tree.parent)):
        for child in tree.children(v):
            if hi[child] < 0:
                continue
            C[lo[child]:hi[child], lo[v]:lo[child]] = tree.depth[v]
            C[lo[child]:hi[child], hi[child]:hi[v]] = tree.depth[v]
    np.fill_diagonal(C, diag)
    ix = [pos[x] for x in tips]
    return C[np.ix_(ix, ix)]


def design(d, model="slope", m_center=None, x_center=None):
    T = d.bio1_c.to_numpy(float) / 10
    M = d.m_mass.to_numpy(float) - (d.m_mass.mean() if m_center is None else m_center)
    cols = {"intercept": np.ones(len(d)), "M": M, "T": T}
    if model != "slope":
        X = d.X.to_numpy(float) - (d.X.mean() if x_center is None else x_center)
        cols["X"] = X
        if model in ("M1", "M3"):
            cols["TxM"] = T * M
        if model in ("M2", "M3"):
            cols["TxX"] = T * X
        if model not in ("M0", "M1", "M2", "M3"):
            raise ValueError(model)
    mandatory = list(cols)
    if np.linalg.matrix_rank(np.column_stack(list(cols.values()))) < len(cols):
        raise ValueError(f"Unidentifiable target terms: {mandatory}")
    nuisance = {"habdens": d.e_habdens.to_numpy(float), "migr": d.e_migr.to_numpy(float),
                "lat_span": d.lat_span.to_numpy(float) / 10}
    nuisance.update({f"life_{k}": d[f"e_life_{k}"].to_numpy(float) for k in LIFE})
    dropped = []
    for name, values in nuisance.items():
        candidate = np.column_stack([*cols.values(), values])
        if np.linalg.matrix_rank(candidate) == len(cols) + 1:
            cols[name] = values
        else:
            dropped.append(name)
    X = np.column_stack(list(cols.values()))
    if not np.isfinite(X).all() or len(d) <= X.shape[1] + 3:
        raise ValueError("Insufficient observations or invalid design")
    return X, list(cols), dropped


def gls_at(Y, X, C, lam):
    Y = np.asarray(Y, float)
    if Y.ndim == 1:
        Y = Y[:, None]
    n, q = Y.shape
    V = C * lam
    np.fill_diagonal(V, np.diag(C))
    L = cholesky(V, lower=True, check_finite=False)
    Z = solve_triangular(L, X, lower=True, check_finite=False)
    z = solve_triangular(L, Y, lower=True, check_finite=False)
    B = np.linalg.lstsq(Z, z, rcond=None)[0]
    E = z - Z @ B
    sigma_ml = E.T @ E / n
    sign, ld = np.linalg.slogdet(sigma_ml)
    if sign <= 0:
        raise ValueError("Singular residual covariance")
    ll = -.5 * (n * q * (np.log(2 * np.pi) + 1) +
                 q * 2 * np.log(np.diag(L)).sum() + n * ld)
    return {"beta": B, "sigma": E.T @ E / (n - X.shape[1]),
            "sigma_ml": sigma_ml, "xtx_inv": np.linalg.inv(Z.T @ Z),
            "loglik": float(ll), "lambda": float(lam), "n": n, "p": X.shape[1]}


def fit(Y, X, C):
    if not all(np.isfinite(a).all() for a in (Y, X, C)):
        raise ValueError("Nonfinite input")
    if np.linalg.matrix_rank(X) != X.shape[1]:
        raise ValueError("Rank deficient design")
    # Check endpoints and a coarse grid as well as the bounded profile optimum.
    grid = np.linspace(0, 1, 5)
    candidates = [(gls_at(Y, X, C, x)["loglik"], x) for x in grid]
    bounds = [(0., 1.)]
    for a, b in bounds:
        opt = minimize_scalar(lambda x: -gls_at(Y, X, C, x)["loglik"],
                              bounds=(a, b), method="bounded", options={"xatol": 1e-4})
        if not opt.success:
            raise RuntimeError(opt.message)
        candidates.append((-opt.fun, opt.x))
    best_grid = max(range(len(grid)), key=lambda i: candidates[i][0])
    if candidates[best_grid][0] > candidates[-1][0] and 0 < best_grid < len(grid) - 1:
        opt = minimize_scalar(lambda x: -gls_at(Y, X, C, x)["loglik"],
                              bounds=(grid[best_grid - 1], grid[best_grid + 1]),
                              method="bounded", options={"xatol": 1e-4})
        if not opt.success:
            raise RuntimeError(opt.message)
        candidates.append((-opt.fun, opt.x))
    return gls_at(Y, X, C, max(candidates)[1])


def estimate(result, term_index, contrast):
    c = np.asarray(contrast)
    b = float(result["beta"][term_index] @ c)
    se = float(np.sqrt(max(0, result["xtx_inv"][term_index, term_index] *
                           (c @ result["sigma"] @ c))))
    df = result["n"] - result["p"]
    crit = student_t.ppf(.975, df)
    if se <= 0:
        raise ValueError("Zero uncertainty; degenerate response")
    return {"beta": b, "se": se, "lo95": b - crit * se, "hi95": b + crit * se,
            "p_raw": float(2 * student_t.sf(abs(b / se), df)), "df": df}


def holm(pvalues):
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adjusted = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    out = np.empty_like(p)
    out[order] = adjusted
    return out


def catalog_key(series):
    return series.fillna("").astype(str).str.strip().str.upper().str.replace(r"[\s_]+", "", regex=True)


def load_samples(common_tips):
    d = base()
    d = d[d.tip.isin(common_tips)].copy()
    d["X"] = d[["e_fs_ground", "e_fs_wataroundsurf", "e_fs_watbelowsurf"]].sum(axis=1, min_count=3)
    e = pd.read_csv(ROOT / "data/raw/eliason2023/S1_limb_data.csv")
    sk = pd.read_csv(ROOT / "data/raw/skelevision/Skelevision_Only_Dataset_v1.csv")
    e["catalog_key"], sk["catalog_key"] = catalog_key(e.Catnum_or_ref), catalog_key(sk.ID)
    matched = set(e.catalog_key) & set(sk.catalog_key) - {""}
    overlap = e[e.catalog_key.isin(matched)].merge(
        sk[sk.catalog_key.isin(matched)], on="catalog_key", suffixes=("_eliason", "_skelevision"))
    overlap["taxon_name_differs"] = overlap.Taxon.map(_tip) != overlap.species.map(_tip)
    clean = e[~e.catalog_key.isin(matched)].copy()
    clean["tip"] = clean.Taxon.str.strip().map(_tip)
    cw = crosswalk()
    miss = ~clean.tip.isin(set(d.tip))
    clean.loc[miss, "tip"] = clean.loc[miss, "tip"].map(cw)
    raw_cols = ["fem_leng_mm", "tbt_leng_mm", "tmt_leng_mm"]
    clean[raw_cols] = clean[raw_cols].apply(pd.to_numeric, errors="coerce")
    clean = clean.dropna(subset=["tip"]).groupby("tip")[raw_cols].mean().reset_index()
    for bone, col in zip(BONES, raw_cols):
        clean[f"b_{bone}"] = np.log10(clean[col].where(clean[col] > 0))
    datasets = {}
    for version, el in [("original", load_eliason(set(d.tip))), ("no_catalog_overlap", clean)]:
        A = d.merge(el[["tip", "b_fem", "b_tbt", "b_tmt"]], on="tip", validate="one_to_one")
        A = A.rename(columns={f"b_{b}": b for b in BONES}).dropna(subset=BONES)
        datasets[version] = A.sort_values("tip").reset_index(drop=True)
    skel = skeleton_table()
    skel = skel[skel.tip.isin(common_tips)].rename(columns={"f": "fem", "t": "tbt", "m": "tmt"})
    skel = skel.sort_values("tip").reset_index(drop=True)
    skeletons = {"Skelevision_passerines": skel[(skel.src_skeleton == "skelevision") &
                    (skel.order == "Passeriformes")].copy(),
                 "Watanabe": skel[skel.src_skeleton == "watanabe2017"].copy()}
    external = d[d.avonet_inferred == 0].dropna(subset=["m_tarsus", "X", "bio1_c", "bio5_c", "bio6_c"])
    return datasets, skeletons, external.sort_values("tip").reset_index(drop=True), overlap


def gradient_cases(datasets, skeletons, full):
    cases = []
    for version, A in datasets.items():
        if version != "original" and not full:
            continue
        cases.extend([(version, "Eliason_P", "slope", "T", A[(A.order != "Passeriformes") & (A['mode'] == "P")], True),
                      (version, "Eliason_all", "M3", "TxX", A.dropna(subset=["X"]) , True)])
        if full and version == "original":
            cases.extend([(version, "Eliason_P_SP", "slope", "T", A[(A.order != "Passeriformes") & A['mode'].isin(["P", "SP"])], False),
                          (version, "Eliason_nest", "slope", "T", A[(A.order != "Passeriformes") & A['mode'].isin(["SA", "A"])], False),
                          (version, "Eliason_passerines", "slope", "T", A[A.order == "Passeriformes"], False),
                          (version, "Eliason_nonpass", "M3", "TxX", A[A.order != "Passeriformes"].dropna(subset=["X"]), False)])
    if full:
        cases.extend(("original", name, "slope", "T", sub, False) for name, sub in skeletons.items())
    return cases
