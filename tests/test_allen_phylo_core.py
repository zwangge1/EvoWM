"""Exact dense references for nested multivariate BM/OU and conditioning."""
from pathlib import Path
import inspect
import sys
import unittest
from unittest.mock import patch

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize as scipy_minimize, minimize_scalar as scipy_minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "allen_world"))
sys.path.insert(0, str(ROOT / "src"))

from phylo_baseline_core import PhyloKernel, PhyloOptimizationError, fit_nested, fit_nested_fixed, predict_joint, _NestedData
from evowm.data.tree import parse_newick
from evowm.eval.pgls import tip_vcv


def fixture(seed=41):
    rng = np.random.default_rng(seed)
    def branch(ids):
        label = f"s{ids[0]}" if len(ids) == 1 else "(" + branch(ids[:len(ids)//2]) + "," + branch(ids[len(ids)//2:]) + ")"
        return label + ":" + str(rng.uniform(.2, 1.2))
    tree = parse_newick(branch(list(range(32))) + ";")
    kernel = PhyloKernel(tree)
    ids = np.arange(32)
    X = np.column_stack([np.ones(32), rng.normal(size=(32, 2))])
    beta = rng.normal(size=(3, 4))
    factor = np.array([[1., 0, 0, 0], [.7, 1., 0, 0], [-.4, .3, .6, 0], [.3, -.4, .1, .5]])
    Y = X @ beta + np.linalg.cholesky(kernel.covariance(ids, alpha=.7, eta=.04)) @ rng.normal(size=(32, 4)) @ factor.T
    Y[20:, 1:] = np.nan
    return tree, kernel, ids, X, Y


def dense_covariance(tree, ids, alpha, eta):
    C = tip_vcv(tree) / max(tree.depth[tree.tip_nodes])
    if alpha:
        D = np.diag(C)[:, None] + np.diag(C)[None, :] - 2 * C
        C = np.exp(-alpha * D) * (-np.expm1(-2 * alpha * C)) / (2 * alpha)
    return C[np.ix_(ids, ids)] + eta * np.eye(len(ids))


def dense_nested(tree, ids, X, Y, alpha, eta):
    m = np.isfinite(Y[:, 0])
    b = np.isfinite(Y[:, 1])
    def regress(ids, X, Y):
        K = dense_covariance(tree, ids, alpha, eta)
        L = np.linalg.cholesky(K)
        XX = solve_triangular(L, X, lower=True)
        YY = solve_triangular(L, Y, lower=True)
        beta = np.linalg.lstsq(XX, YY, rcond=None)[0]
        residual = YY - XX @ beta
        sigma = residual.T @ residual / len(X)
        return beta, sigma
    bm, sm = regress(ids[m], X[m], Y[m, :1])
    bc, psi = regress(ids[b], np.column_stack([X[b], Y[b, 0]]), Y[b, 1:])
    gamma = bc[-1]
    beta = np.column_stack([bm[:, 0], bc[:-1] + bm * gamma])
    sigma = np.empty((4, 4))
    sigma[0, 0] = sm[0, 0]
    sigma[0, 1:] = sigma[1:, 0] = sm[0, 0] * gamma
    sigma[1:, 1:] = psi + sm[0, 0] * np.outer(gamma, gamma)
    mask = np.isfinite(Y).ravel()
    # Independently evaluate the original full observed-data likelihood.  This
    # verifies that the nested factorization is not a composite likelihood.
    covariance = np.kron(dense_covariance(tree, ids, alpha, eta), sigma)[np.ix_(mask, mask)]
    L = np.linalg.cholesky(covariance)
    residual = (Y - X @ beta).ravel()[mask]
    z = solve_triangular(L, residual, lower=True)
    loglik = -.5 * (len(z) * np.log(2 * np.pi) + 2 * np.log(np.diag(L)).sum() + z @ z)
    return beta, sigma, loglik


class PhyloCoreTests(unittest.TestCase):
    def test_covariance_and_pruning_match_dense_on_nonultrametric_tree(self):
        tree, kernel, ids, X, Y = fixture()
        order = np.array([28, 7, 12, 1, 31, 0, 14, 19, 3])
        values = np.column_stack([X[order], Y[order, 0]])
        for alpha in (0., 1e-9, .6, 50.):
            for eta in (0., .03):
                with self.subTest(alpha=alpha, eta=eta):
                    K = dense_covariance(tree, order, alpha, eta)
                    np.testing.assert_allclose(kernel.covariance(order, alpha=alpha, eta=eta), K, rtol=1e-12, atol=1e-14)
                    white, logdet = kernel.whiten(order, values, alpha=alpha, eta=eta)
                    np.testing.assert_allclose(white.T @ white, values.T @ np.linalg.solve(K, values), rtol=3e-11, atol=3e-10)
                    np.testing.assert_allclose(logdet, np.linalg.slogdet(K)[1], rtol=1e-12, atol=1e-11)
                    block = kernel.covariance(order[:4], order[4:], alpha=alpha, eta=eta)
                    np.testing.assert_allclose(block, K[:4, 4:], rtol=1e-12, atol=1e-14)

    def test_fixed_nested_gls_and_full_joint_likelihood_match_dense(self):
        tree, kernel, ids, X, Y = fixture()
        Y[30:, :] = np.nan
        for alpha, eta in ((0., 0.), (0., .04), (.8, 0.), (2., .07)):
            with self.subTest(alpha=alpha, eta=eta):
                actual = fit_nested_fixed(kernel, ids, X, Y, alpha=alpha, eta=eta)
                beta, sigma, loglik = dense_nested(tree, ids, X, Y, alpha, eta)
                np.testing.assert_allclose(actual["beta"], beta, atol=2e-11, rtol=2e-11)
                np.testing.assert_allclose(actual["sigma"], sigma, atol=2e-11, rtol=2e-11)
                self.assertAlmostEqual(actual["loglik"], loglik, places=9)
                self.assertEqual(actual["n_observed_coordinates"], 30 + 3 * 20)

    def test_alpha_zero_is_exact_bm_and_limit_is_continuous(self):
        _, kernel, ids, X, Y = fixture()
        bm = fit_nested(kernel, ids, X, Y, model="BM", eta=.03)
        zero = fit_nested(kernel, ids, X, Y, model="OU", alpha=0., eta=.03)
        near = fit_nested_fixed(kernel, ids, X, Y, alpha=1e-9, eta=.03)
        for key in ("beta", "sigma", "loglik"):
            np.testing.assert_array_equal(bm[key], zero[key])
            np.testing.assert_allclose(bm[key], near[key], atol=3e-8, rtol=3e-8)

    def test_partial_reference_conditioning_matches_dense_and_uses_cross_traits(self):
        tree, kernel, ids, X, Y = fixture()
        fit = fit_nested_fixed(kernel, ids[:26], X[:26], Y[:26], alpha=.8, eta=.04)
        ref, target = np.array([3, 8, 16]), np.array([5, 31])
        mask = np.array([[True, False, True, False], [True, True, False, True], [False, True, True, True]])
        mean, cov = predict_joint(kernel, fit, ref, X[ref], Y[ref], target, X[target], ref_mask=mask, return_cov=True)
        all_ids = np.concatenate([ref, target])
        K = dense_covariance(tree, all_ids, .8, .04)
        V = np.kron(K, fit["sigma"])
        observed = np.flatnonzero(mask.ravel())
        q = np.arange(4 * len(ref), 4 * len(all_ids))
        R, C = V[np.ix_(observed, observed)], V[np.ix_(q, observed)]
        residual = (Y[ref] - X[ref] @ fit["beta"]).ravel()[observed]
        expected_mean = (X[target] @ fit["beta"]).ravel() + C @ np.linalg.solve(R, residual)
        expected_cov = V[np.ix_(q, q)] - C @ np.linalg.solve(R, C.T)
        np.testing.assert_allclose(mean.ravel(), expected_mean, atol=1e-11)
        np.testing.assert_allclose(cov, expected_cov, atol=1e-11)
        self.assertGreater(np.linalg.eigvalsh(cov).min(), 0)
        changed = Y[ref].copy()
        changed[0, 0] += 3
        changed_mean = predict_joint(kernel, fit, ref, X[ref], changed, target, X[target], ref_mask=mask)
        self.assertGreater(np.max(np.abs(changed_mean[:, 1] - mean[:, 1])), 1e-5)

    def test_masked_values_and_all_missing_refs_do_not_leak(self):
        _, kernel, ids, X, Y = fixture()
        fit = fit_nested_fixed(kernel, ids[:26], X[:26], Y[:26])
        ref, target = ids[:3], ids[-2:]
        mask = np.array([[True, False, False, True], [False, True, True, False], [True, True, False, False]])
        base = predict_joint(kernel, fit, ref, X[ref], Y[ref], target, X[target], ref_mask=mask)
        for hidden in (np.nan, 1e100, -1e100):
            altered = Y[ref].copy()
            altered[~mask] = hidden
            actual = predict_joint(kernel, fit, ref, X[ref], altered, target, X[target], ref_mask=mask)
            np.testing.assert_array_equal(actual, base)
        unconditional = predict_joint(kernel, fit, ref, X[ref], np.full((3, 4), np.nan), target, X[target])
        np.testing.assert_array_equal(unconditional, X[target] @ fit["beta"])
        self.assertNotIn("target_Y", inspect.signature(predict_joint).parameters)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            predict_joint(kernel, fit, ref, X[ref], Y[ref], ref, X[ref])

    def test_complete_reference_traits_have_separable_conditional_mean(self):
        _, kernel, ids, X, Y = fixture()
        fit = fit_nested_fixed(kernel, ids[:26], X[:26], Y[:26], alpha=.8, eta=.04)
        ref, target = np.array([3, 8, 16]), np.array([5, 31])
        actual = predict_joint(kernel, fit, ref, X[ref], Y[ref], target, X[target])
        rr = kernel.covariance(ref, alpha=.8, eta=.04)
        qr = kernel.covariance(target, ref, alpha=.8, eta=.04)
        expected = X[target] @ fit["beta"] + qr @ np.linalg.solve(rr, Y[ref] - X[ref] @ fit["beta"])
        # Sigma cancels only when the four reference masks are identical.  This
        # is a property of the classical separable model, not an implementation
        # claim that cross-trait correlations always improve prediction.
        np.testing.assert_allclose(actual, expected, atol=1e-11)

    def test_nested_mask_rank_and_singularity_fail_explicitly(self):
        _, kernel, ids, X, Y = fixture()
        for broken in (np.column_stack([X, X[:, 1]]), X * np.nan):
            with self.assertRaises(ValueError):
                fit_nested_fixed(kernel, ids, broken, Y)
        missing = Y.copy()
        missing[0, 2] = np.nan
        with self.assertRaisesRegex(ValueError, "nested"):
            fit_nested_fixed(kernel, ids, X, missing)
        missing = Y.copy()
        missing[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "nested"):
            fit_nested_fixed(kernel, ids, X, missing)
        singular = Y.copy()
        singular[:, 0] = X @ np.ones(3)
        with self.assertRaisesRegex(ValueError, "rank-deficient"):
            fit_nested_fixed(kernel, ids, X, singular)
        singular = Y.copy()
        singular[:, 2] = singular[:, 1]
        with self.assertRaisesRegex(ValueError, "residual covariance"):
            fit_nested_fixed(kernel, ids, X, singular)
        tiny = PhyloKernel(parse_newick("((a:0,b:0):1,c:1);"))
        with self.assertRaisesRegex(ValueError, "singular"):
            tiny.whiten(np.arange(3), np.ones((3, 1)), eta=0.)
        white, _ = tiny.whiten(np.arange(3), np.ones((3, 1)), eta=.01)
        self.assertTrue(np.isfinite(white).all())

    def test_optimization_contains_boundary_and_never_worsens_bm(self):
        _, kernel, ids, X, Y = fixture()
        bm = fit_nested(kernel, ids, X, Y, model="BM")
        ou = fit_nested(kernel, ids, X, Y, model="OU")
        self.assertGreaterEqual(ou["loglik"] + 1e-8, bm["loglik"])
        self.assertTrue(any(p["alpha"] == 0. and p["eta"] == 0. for p in ou["profiles"]))
        self.assertTrue(any(p["alpha"] > 0 and p["eta"] == 0. for p in ou["profiles"]))
        self.assertEqual(ou["likelihood"], "exact_nested_multivariate_ML")
        self.assertAlmostEqual(ou["alpha_original_units"] * ou["tree_height"], ou["alpha"])
        self.assertTrue(ou["convergence"]["accepted"])
        self.assertTrue(all({"alpha", "eta", "loglik", "success"} <= set(row) for row in ou["optimizers"]))

    def test_all_optimizer_failures_raise_with_audit_and_one_refinement(self):
        _, kernel, ids, X, Y = fixture()
        def fail_scalar(*args, **kwargs):
            result = scipy_minimize_scalar(*args, **kwargs)
            result.success, result.message = False, "injected scalar failure"
            return result
        def fail_refinement(*args, **kwargs):
            result = scipy_minimize(*args, **kwargs)
            result.success, result.message = False, "injected refinement failure"
            return result
        with patch("phylo_baseline_core.minimize_scalar", side_effect=fail_scalar), \
                patch("phylo_baseline_core.minimize", side_effect=fail_refinement), \
                self.assertRaises(PhyloOptimizationError) as caught:
            fit_nested(kernel, ids, X, Y, model="BM")
        audit = caught.exception.diagnostics
        self.assertTrue(audit["refinement_attempted"])
        self.assertEqual(sum(row["kind"] == "selected_candidate_refinement" for row in audit["optimizers"]), 1)
        self.assertFalse(any(row["success"] for row in audit["optimizers"]))
        self.assertTrue(audit["profiles"])
        self.assertTrue(all(row["loglik"] is not None for row in audit["optimizers"]))

    def test_successful_refinement_can_certify_failed_initial_search(self):
        _, kernel, ids, X, Y = fixture()
        def fail_scalar(*args, **kwargs):
            result = scipy_minimize_scalar(*args, **kwargs)
            result.success, result.message = False, "injected scalar failure"
            return result
        with patch("phylo_baseline_core.minimize_scalar", side_effect=fail_scalar):
            fit = fit_nested(kernel, ids, X, Y, model="BM")
        self.assertTrue(fit["convergence"]["accepted"])
        self.assertTrue(fit["convergence"]["refinement_attempted"])
        self.assertEqual(fit["convergence"]["failed_optimizer_count"], 2)
        self.assertTrue(fit["optimizers"][-1]["success"])

    def test_exact_search_endpoint_requires_successful_adjacent_profile(self):
        _, kernel, ids, X, Y = fixture()
        original = _NestedData.at
        def increasing_eta(data, alpha, eta):
            result = original(data, alpha, eta)
            result["loglik"] = eta  # A known upper-bound optimum.
            return result
        with patch.object(_NestedData, "at", increasing_eta):
            fit = fit_nested(kernel, ids, X, Y, model="BM")
        self.assertEqual(fit["eta"], 10.)
        self.assertEqual(fit["convergence"]["kind"], "accepted_boundary")
        self.assertEqual(fit["convergence"]["parameter_boundaries"]["eta"], "search_upper")
        self.assertFalse(fit["convergence"]["refinement_attempted"])

    def test_fixed_parameters_require_no_optimizer_certification(self):
        _, kernel, ids, X, Y = fixture()
        with patch("phylo_baseline_core.minimize", side_effect=AssertionError("unexpected optimizer")), \
                patch("phylo_baseline_core.minimize_scalar", side_effect=AssertionError("unexpected optimizer")):
            fit = fit_nested(kernel, ids, X, Y, model="OU", alpha=.7, eta=.03)
        self.assertEqual(fit["convergence"]["kind"], "fixed_parameters")
        self.assertEqual(fit["optimizers"], [])


if __name__ == "__main__":
    unittest.main()
