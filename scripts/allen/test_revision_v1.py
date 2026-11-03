"""Numerical cross-checks; no real phenotype data are fitted here."""
import numpy as np
from scipy.linalg import block_diag
from scipy.stats import matrix_normal

from revision_core import (CONTRASTS, covariance, estimate, fit, get_tree,
                           gls_at, holm)
from evowm.data.tree import parse_newick
from evowm.eval.pgls import tip_vcv


def run_checks(real_tree=True):
    checks = []
    tree = parse_newick("((a:1,b:1):2,(c:2,(d:1,e:1):1):1);")
    requested = ["e", "a", "c"]
    ix = [tree.tip_names.index(t) for t in requested]
    np.testing.assert_allclose(covariance(tree, requested), tip_vcv(tree)[np.ix_(ix, ix)])
    checks.append("subset covariance equals full-tree covariance in arbitrary tip order")
    np.testing.assert_allclose(holm([.01, .04, .03]), [.03, .06, .06])
    checks.append("Holm adjusted p values")
    for seed in (173, 947):
        rng = np.random.default_rng(seed)
        n = 30
        C = .6 * np.eye(n) + .4 * np.ones((n, n))
        X = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])
        B = rng.normal(size=(3, 3))
        sigma = np.array([[1., .8, .6], [.8, 1.2, .7], [.6, .7, 1.]]) * .02
        lam = .65
        V = lam * C + (1 - lam) * np.diag(np.diag(C))
        Y = X @ B + np.linalg.cholesky(V) @ rng.normal(size=(n, 3)) @ np.linalg.cholesky(sigma).T
        r = gls_at(Y, X, C, lam)
        ll = matrix_normal.logpdf(Y, mean=X @ r['beta'], rowcov=V, colcov=r['sigma_ml'])
        np.testing.assert_allclose(r['loglik'], ll, atol=1e-9)
        XX = block_diag(X, X, X)
        W = np.linalg.inv(np.kron(r['sigma'], V))
        exact_cov = np.linalg.inv(XX.T @ W @ XX)
        exact_beta = exact_cov @ XX.T @ W @ Y.ravel(order="F")
        np.testing.assert_allclose(exact_beta, r['beta'].ravel(order="F"), atol=1e-9)
        for c in CONTRASTS.values():
            h = np.kron(c, [0., 1., 0.])
            e = estimate(r, 1, c)
            np.testing.assert_allclose(e['se'] ** 2, h @ exact_cov @ h, atol=1e-10)
            ratio = gls_at(Y @ c, X, C, lam)
            np.testing.assert_allclose(e['beta'], ratio['beta'][1, 0], atol=1e-10)
            np.testing.assert_allclose(e['se'], estimate(ratio, 1, [1.])['se'], atol=1e-10)
        perm = rng.permutation(n)
        permfit = gls_at(Y[perm], X[perm], C[np.ix_(perm, perm)], lam)
        np.testing.assert_allclose(r['beta'], permfit['beta'], atol=1e-10)
        checks.append(f"seed {seed}: SciPy likelihood, explicit Kronecker GLS, paired ratio, row permutation")
    if real_tree:
        tree, _ = get_tree("hackett_t021")
        tips = tree.tip_names[::max(1, len(tree.tip_names) // 160)][:160]
        C = covariance(tree, tips)
        C /= np.max(np.diag(C))
        V = .65 * C + .35 * np.diag(np.diag(C))
        sigma = np.array([[1., .8, .6], [.8, 1.2, .7], [.6, .7, 1.]]) * .0001
        for seed in (173, 947):
            rng = np.random.default_rng(seed)
            X = np.column_stack([np.ones(len(tips)), rng.normal(size=len(tips))])
            E = np.linalg.cholesky(V) @ rng.normal(size=(len(tips), 3)) @ np.linalg.cholesky(sigma).T
            for label, slopes in [("null", [0., 0., 0.]), ("gradient", [0., .02, .04])]:
                Y = X @ np.array([[1., 1.2, 1.1], slopes]) + E
                r = fit(Y, X, C)
                results = [estimate(r, 1, c) for c in CONTRASTS.values()]
                for c, est in zip(CONTRASTS.values(), results):
                    truth = np.asarray(slopes) @ c
                    if abs(est['beta'] - truth) > 4 * est['se']:
                        raise AssertionError(f"Recovery failed: {seed}, {label}, {est}, truth={truth}")
                adjusted = holm([v['p_raw'] for v in results])
                if label == "gradient" and not np.all(adjusted < .05):
                    raise AssertionError("Strong synthetic gradient was not recovered")
                if label == "null" and np.any(adjusted < .05):
                    raise AssertionError("Fixed-seed null produced a false detection; inspect")
                checks.append(f"real-tree synthetic {seed} {label}: recovered; lambda={r['lambda']:.4f}")
    return checks


if __name__ == "__main__":
    print("\n".join(run_checks()))
