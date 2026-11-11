"""Exact classical multivariate BM / fixed-root OU conditional prediction.

Residuals have covariance ``K(alpha, eta) (x) Sigma`` in species-major order.
Branch lengths are divided by the original tree height.  BM uses shared path
length; OU starts with residual zero at the original root and has diffusion
variance one.  ``eta I`` is optional *proportional tip error*: all four traits
share the same phylogenetic/error ratio and cross-trait covariance Sigma.

The observed-data likelihood is exact for the Allen nested observation pattern:
trait zero is observed everywhere that the other three traits are observed,
and those three traits have identical observation masks.  Its factorization is
p(y0_all) p(y123_complete | y0_complete), not a composite likelihood.  No target
responses are accepted by the prediction API.  All fitted parameters, including
the optional nugget and common OU alpha, must be fitted on training rows only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize, minimize_scalar


@dataclass
class _PruningPlan:
    children: list[np.ndarray]
    branch: np.ndarray
    tip_row: np.ndarray

    def whiten(self, values, alpha, eta):
        """Independent contrasts including the fixed-root innovation.

        Returns W @ values and log|K| without forming W or K.  Unary paths have
        already been collapsed.  Applying contrasts to all design/response
        columns together costs O(n * columns); Gram products subsequently use
        BLAS, avoiding one large outer product per tree node.
        """
        z = np.asarray(values, dtype=float)
        if z.ndim != 2 or not np.isfinite(z).all():
            raise ValueError("whitening values must be a finite matrix")
        n = int(np.sum(self.tip_row >= 0))
        if len(z) != n:
            raise ValueError("whitening values and selected tips disagree")
        means = np.zeros((len(self.branch), z.shape[1]))
        variance = np.zeros(len(self.branch))
        observed = self.tip_row >= 0
        means[observed] = z[self.tip_row[observed]]
        variance[observed] = eta
        a = np.exp(-alpha * self.branch)
        q = (self.branch.copy() if alpha == 0 else
             -np.expm1(-2 * alpha * self.branch) / (2 * alpha))
        white = np.empty_like(z)
        row, logdet = 0, 0.0
        for node in range(len(self.branch) - 1, -1, -1):
            children = self.children[node]
            if not len(children):
                continue
            child = children[0]
            mean = means[child] / a[child]
            var = (variance[child] + q[child]) / a[child] ** 2
            logdet -= 2 * alpha * self.branch[child]
            for child in children[1:]:
                other_mean = means[child] / a[child]
                other_var = (variance[child] + q[child]) / a[child] ** 2
                logdet -= 2 * alpha * self.branch[child]
                total = var + other_var
                if not np.isfinite(total) or total <= 0:
                    raise ValueError("singular phylogenetic covariance (zero-length duplicate observations)")
                white[row] = (mean - other_mean) / np.sqrt(total)
                row += 1
                logdet += np.log(total)
                # This convex combination also handles one exact observation.
                fraction = var / total
                mean = (1 - fraction) * mean + fraction * other_mean
                var = var * (other_var / total)
            means[node], variance[node] = mean, var
        if variance[0] <= 0 or not np.isfinite(variance[0]):
            raise ValueError("singular covariance at the fixed root")
        white[row] = means[0] / np.sqrt(variance[0])
        logdet += np.log(variance[0])
        if row != n - 1 or not np.isfinite(white).all():
            raise ValueError("invalid pruning contrast result")
        return white, float(logdet)


class PhyloKernel:
    """Tree-only covariance and pruning plans; no trait data are stored here."""

    def __init__(self, tree):
        self.tree = tree
        self.parent = np.asarray(tree.parent, dtype=np.int64)
        self.depth = np.asarray(tree.depth, dtype=float)
        self.tip_nodes = np.asarray(tree.tip_nodes, dtype=np.int64)
        n = len(self.parent)
        if (tree.root != 0 or self.parent.shape != (n,) or self.depth.shape != (n,)
                or not n or self.parent[0] != -1
                or np.any(self.parent[1:] < 0)
                or np.any(self.parent[1:] >= np.arange(1, n))
                or not np.isfinite(self.depth).all()):
            raise ValueError("tree must have root zero, finite depths, and parent-before-child numbering")
        if (self.tip_nodes.ndim != 1 or not len(self.tip_nodes)
                or np.any(self.tip_nodes < 0) or np.any(self.tip_nodes >= n)
                or len(np.unique(self.tip_nodes)) != len(self.tip_nodes)):
            raise ValueError("invalid tip nodes")
        self.depth = self.depth - self.depth[0]
        branch = self.depth[1:] - self.depth[self.parent[1:]]
        if np.any(branch < 0):
            raise ValueError("negative tree branch lengths")
        counts = np.bincount(self.parent[1:], minlength=n)
        if not np.array_equal(np.sort(self.tip_nodes), np.flatnonzero(counts == 0)):
            raise ValueError("tip nodes must enumerate the tree leaves")
        self.tree_height = float(np.max(self.depth[self.tip_nodes]))
        if self.tree_height <= 0:
            raise ValueError("tree must have positive height")
        self.depth = self.depth / self.tree_height
        self._plans = {}
        # Binary lifting supports small cross-covariance queries without an
        # all-species O(n^2) shared-path matrix.
        self._levels = np.zeros(n, dtype=np.int64)
        for node in range(1, n):
            self._levels[node] = self._levels[self.parent[node]] + 1
        self._up = np.empty((max(1, int(self._levels.max()).bit_length()), n), dtype=np.int64)
        self._up[0] = np.maximum(self.parent, 0)
        for k in range(1, len(self._up)):
            self._up[k] = self._up[k - 1, self._up[k - 1]]

    def _tip_ids(self, ids, *, unique=True):
        arr = np.asarray(ids)
        if arr.ndim != 1 or arr.dtype.kind not in "iu":
            raise ValueError("tip identifiers must be a one-dimensional integer array")
        arr = arr.astype(np.int64, copy=False)
        if (np.any(arr < 0) or np.any(arr >= len(self.tip_nodes))
                or (unique and len(np.unique(arr)) != len(arr))):
            raise ValueError("tip identifiers must be in range and unique")
        return arr

    @staticmethod
    def _parameters(alpha, eta):
        if not np.isfinite(alpha) or alpha < 0 or not np.isfinite(eta) or eta < 0:
            raise ValueError("alpha and eta must be finite and nonnegative")

    def plan(self, tip_ids):
        ids = self._tip_ids(tip_ids)
        if not len(ids):
            raise ValueError("at least one observed tip is required")
        key = ids.tobytes()
        if key in self._plans:
            return self._plans[key]
        active = np.zeros(len(self.parent), dtype=bool)
        active[self.tip_nodes[ids]] = True
        for node in range(len(active) - 1, 0, -1):
            if active[node]:
                active[self.parent[node]] = True
        count = np.bincount(self.parent[1:][active[1:]], minlength=len(active))
        retain = active & (count != 1)
        retain[0] = True
        nodes = np.flatnonzero(retain)
        compact = np.full(len(active), -1, dtype=np.int64)
        compact[nodes] = np.arange(len(nodes))
        ancestor = np.zeros(len(active), dtype=np.int64)
        for node in range(1, len(active)):
            p = self.parent[node]
            ancestor[node] = p if retain[p] else ancestor[p]
        parent = compact[ancestor[nodes]]
        children = [[] for _ in nodes]
        for node in range(1, len(nodes)):
            children[parent[node]].append(node)
        branch = self.depth[nodes] - self.depth[ancestor[nodes]]
        branch[0] = 0
        row = np.full(len(nodes), -1, dtype=np.int64)
        row[compact[self.tip_nodes[ids]]] = np.arange(len(ids))
        plan = _PruningPlan([np.asarray(c, dtype=np.int64) for c in children], branch, row)
        self._plans[key] = plan
        return plan

    def whiten(self, tip_ids, values, *, alpha=0.0, eta=0.0):
        self._parameters(alpha, eta)
        return self.plan(tip_ids).whiten(values, alpha, eta)

    def covariance(self, tip_ids, other_tip_ids=None, *, alpha=0.0, eta=0.0):
        """Small dense K blocks, including eta only for matching tip identities."""
        self._parameters(alpha, eta)
        left = self._tip_ids(tip_ids)
        right = left if other_tip_ids is None else self._tip_ids(other_tip_ids)
        shape = (len(left), len(right))
        u = np.broadcast_to(self.tip_nodes[left, None], shape).copy().ravel()
        v = np.broadcast_to(self.tip_nodes[right][None, :], shape).copy().ravel()
        swap = self._levels[u] < self._levels[v]
        u[swap], v[swap] = v[swap].copy(), u[swap].copy()
        difference = self._levels[u] - self._levels[v]
        for bit in range(len(self._up)):
            move = (difference & (1 << bit)) != 0
            u[move] = self._up[bit, u[move]]
        for bit in range(len(self._up) - 1, -1, -1):
            move = self._up[bit, u] != self._up[bit, v]
            u[move] = self._up[bit, u[move]]
            v[move] = self._up[bit, v[move]]
        unequal = u != v
        u[unequal] = self.parent[u[unequal]]
        shared = self.depth[u].reshape(shape)
        if alpha == 0:
            result = shared
        else:
            distance = self.depth[self.tip_nodes[left], None] + self.depth[self.tip_nodes[right]][None, :] - 2 * shared
            result = np.exp(-alpha * np.maximum(distance, 0)) * (-np.expm1(-2 * alpha * shared)) / (2 * alpha)
        return result + eta * (left[:, None] == right[None, :])


class _NestedData:
    def __init__(self, kernel, tip_ids, X, Y):
        ids = kernel._tip_ids(tip_ids)
        X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
        if X.ndim != 2 or len(X) != len(ids) or X.shape[1] < 1 or not np.isfinite(X).all():
            raise ValueError("X must be a finite n by p design matrix with p >= 1")
        if Y.shape != (len(ids), 4) or np.isinf(Y).any():
            raise ValueError("Y must have four traits, using NaN for missing observations")
        observed = np.isfinite(Y)
        if (not np.array_equal(observed[:, 1], observed[:, 2])
                or not np.array_equal(observed[:, 1], observed[:, 3])
                or np.any(observed[:, 1] & ~observed[:, 0])):
            raise ValueError("exact nested likelihood requires identical bone masks nested in trait zero")
        m, b = observed[:, 0], observed[:, 1]
        self.p, self.n_m, self.n_b = X.shape[1], int(m.sum()), int(b.sum())
        if self.n_m <= self.p or self.n_b < self.p + 4:
            raise ValueError("insufficient observations for full-rank mean and residual covariance")
        design_m, design_b = X[m], np.column_stack([X[b], Y[b, 0]])
        if (np.linalg.matrix_rank(design_m) != self.p
                or np.linalg.matrix_rank(design_b) != self.p + 1):
            raise ValueError("rank-deficient design (including the conditional trait-zero covariate)")
        self.z_m = np.column_stack([design_m, Y[m, 0]])
        self.z_b = np.column_stack([design_b, Y[b, 1:]])
        self.plan_m, self.plan_b = kernel.plan(ids[m]), kernel.plan(ids[b])
        self.kernel = kernel

    def at(self, alpha, eta):
        self.kernel._parameters(alpha, eta)
        wm, logdet_m = self.plan_m.whiten(self.z_m, alpha, eta)
        wb, logdet_b = self.plan_b.whiten(self.z_b, alpha, eta)
        beta_m, sigma_m = _gls_whitened(wm[:, :self.p], wm[:, self.p:])
        conditional, psi = _gls_whitened(wb[:, :self.p + 1], wb[:, self.p + 1:])
        gamma = conditional[-1]
        beta = np.column_stack([beta_m[:, 0], conditional[:-1] + beta_m * gamma])
        sigma = np.empty((4, 4))
        sigma[0, 0] = sigma_m[0, 0]
        sigma[0, 1:] = sigma[1:, 0] = sigma_m[0, 0] * gamma
        sigma[1:, 1:] = psi + sigma_m[0, 0] * np.outer(gamma, gamma)
        loglik_m = -.5 * (self.n_m * (np.log(2 * np.pi) + 1 + np.log(sigma_m[0, 0])) + logdet_m)
        loglik_b = -.5 * (3 * self.n_b * (np.log(2 * np.pi) + 1)
                          + self.n_b * np.linalg.slogdet(psi)[1] + 3 * logdet_b)
        loglik = float(loglik_m + loglik_b)
        k = 4 * self.p + 10
        return {
            "beta": beta, "sigma": sigma, "alpha": float(alpha),
            "alpha_original_units": float(alpha / self.kernel.tree_height),
            "half_life_original_units": (float(np.log(2) * self.kernel.tree_height / alpha) if alpha > 0 else None),
            "eta": float(eta), "loglik": loglik, "k": k, "aic": 2 * k - 2 * loglik,
            "loglik_morph": float(loglik_m), "loglik_bone_conditional": float(loglik_b),
            "n_morph": self.n_m, "n_bone": self.n_b,
            "n_observed_coordinates": self.n_m + 3 * self.n_b,
            "p": self.p, "tree_height": self.kernel.tree_height,
            "sigma_units": "diffusion covariance per normalized tree height; divide by tree_height for original time units",
            "root": "fixed_zero_residual", "likelihood": "exact_nested_multivariate_ML",
            "tip_error": "eta times Sigma (proportional multivariate tip error)",
            "covariance": "(K_fixed_root_OU_on_height_normalized_tree + eta I) x Sigma",
        }


def _gls_whitened(X, Y):
    try:
        beta = cho_solve(cho_factor(X.T @ X, lower=True, check_finite=False), X.T @ Y, check_finite=False)
    except np.linalg.LinAlgError as exc:
        raise ValueError("numerically rank-deficient GLS design") from exc
    residual = Y - X @ beta
    sigma = residual.T @ residual / len(X)  # ML, not an unbiased/REML estimate.
    try:
        np.linalg.cholesky(sigma)
    except np.linalg.LinAlgError as exc:
        raise ValueError("singular residual covariance") from exc
    if np.min(np.linalg.eigvalsh(sigma)) <= np.finfo(float).eps * max(float(np.mean(Y * Y)), np.finfo(float).tiny):
        raise ValueError("singular or numerically degenerate residual covariance")
    return beta, sigma


def fit_nested_fixed(kernel, tip_ids, X, Y, *, alpha=0.0, eta=0.0):
    """Exact ML beta and Sigma at specified kernel parameters (train rows only)."""
    return _NestedData(kernel, tip_ids, X, Y).at(alpha, eta)


class PhyloOptimizationError(ValueError):
    """Uncertified likelihood optimum, with logs available for failed-run audit."""

    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def fit_nested(kernel, tip_ids, X, Y, *, model="BM", estimate_eta=True,
               eta=None, alpha=None, alpha_bounds=(1e-7, 50.0), eta_bounds=(1e-8, 10.0)):
    """Profile exact ML with explicit BM/zero-nugget boundary candidates.

    Supplying alpha or eta fixes that parameter.  Without explicit eta,
    estimate_eta=False fixes it at zero.  OU alpha is measured per normalized
    tree height; alpha_original_units divides it by the original tree height.
    Grid candidates, two local scalar searches, and three joint log-parameter
    starts are retained in profiles; this is a finite numerical search, not a
    guarantee of the global maximum.  A selected solution must match a finite
    successful optimizer endpoint in likelihood, or be an explicit boundary
    supported by a successful adjacent profile.  Otherwise one bounded local
    refinement is attempted, then PhyloOptimizationError retains the full audit.
    """
    model = str(model).upper()
    if model not in ("BM", "OU") or (model == "BM" and alpha not in (None, 0)):
        raise ValueError("model must be BM (alpha zero) or OU")
    for bounds in (alpha_bounds, eta_bounds):
        if len(bounds) != 2 or not 0 < bounds[0] < bounds[1] or not np.isfinite(bounds).all():
            raise ValueError("optimization bounds must be finite, positive, increasing pairs")
    fixed_alpha = 0.0 if model == "BM" else alpha
    fixed_eta = eta if eta is not None else (None if estimate_eta else 0.0)
    if fixed_alpha is not None:
        kernel._parameters(fixed_alpha, 0)
    if fixed_eta is not None:
        kernel._parameters(0, fixed_eta)
    data = _NestedData(kernel, tip_ids, X, Y)
    profiles, cache, optimizers = [], {}, []

    def evaluate(a, e):
        key = (float(a), float(e))
        if key not in cache:
            try:
                value = data.at(*key)
                profiles.append({"alpha": key[0], "eta": key[1], "loglik": value["loglik"]})
            except ValueError as exc:
                value = None
                profiles.append({"alpha": key[0], "eta": key[1], "loglik": None, "error": str(exc)})
            cache[key] = value
        return cache[key]

    def objective(a, e):
        result = evaluate(a, e)
        return 1e100 if result is None else -result["loglik"]

    def record_optimizer(opt, kind, a, e, **details):
        result = evaluate(a, e)
        optimizers.append({"kind": kind, **details, "alpha": float(a), "eta": float(e),
                           "loglik": None if result is None else result["loglik"],
                           "success": bool(opt.success) and result is not None,
                           "message": str(opt.message), "nfev": int(opt.nfev)})

    def axis(which, other, bounds, landmarks):
        grid = sorted(set([bounds[0], bounds[1]] + [v for v in landmarks if bounds[0] < v < bounds[1]]))
        def fun(v):
            return objective(v, other) if which == "alpha" else objective(other, v)
        fun(0.0)
        scores = [fun(v) for v in grid]
        for i in np.argsort(scores)[:2]:
            lo, hi = np.log(grid[max(0, i - 1)]), np.log(grid[min(len(grid) - 1, i + 1)])
            if hi > lo:
                opt = minimize_scalar(lambda logv: fun(float(np.exp(logv))), bounds=(lo, hi), method="bounded",
                                      options={"xatol": 2e-4, "maxiter": 55})
                v = float(np.exp(opt.x))
                a, e = (v, other) if which == "alpha" else (other, v)
                record_optimizer(opt, which + "_profile", a, e, other=float(other),
                                 bracket=[float(np.exp(lo)), float(np.exp(hi))])

    if fixed_alpha is not None and fixed_eta is not None:
        evaluate(fixed_alpha, fixed_eta)
    elif fixed_alpha is not None:
        axis("eta", fixed_alpha, eta_bounds, (1e-4, .01, .3, 1.0))
    elif fixed_eta is not None:
        axis("alpha", fixed_eta, alpha_bounds, (.003, .1, 1.0, 10.0))
    else:
        # These are the entire BM boundary and the entire eta=0 OU boundary.
        axis("eta", 0.0, eta_bounds, (1e-4, .01, .3, 1.0))
        axis("alpha", 0.0, alpha_bounds, (.003, .1, 1.0, 10.0))
        aa = sorted(set([alpha_bounds[0], alpha_bounds[1]] + [v for v in (.01, .3, 3.) if alpha_bounds[0] < v < alpha_bounds[1]]))
        ee = sorted(set([eta_bounds[0], eta_bounds[1]] + [v for v in (.001, .03, .3) if eta_bounds[0] < v < eta_bounds[1]]))
        grid = [(objective(a, e), a, e) for a in aa for e in ee]
        for _, a, e in sorted(grid)[:3]:
            opt = minimize(lambda v: objective(*np.exp(v)), np.log([a, e]), method="L-BFGS-B",
                           bounds=[np.log(alpha_bounds), np.log(eta_bounds)],
                           options={"maxiter": 60, "ftol": 1e-9, "gtol": 2e-5})
            record_optimizer(opt, "joint_log_alpha_eta", *np.exp(opt.x), start=[float(a), float(e)])
    available = [v for v in cache.values() if v is not None]
    if not available:
        raise ValueError("no nonsingular likelihood candidate: " + profiles[0].get("error", "unknown failure"))
    best = dict(max(available, key=lambda value: value["loglik"]))
    free = (["alpha"] if fixed_alpha is None else []) + (["eta"] if fixed_eta is None else [])
    parameter_bounds = {"alpha": alpha_bounds, "eta": eta_bounds}

    def certify(candidate):
        tolerance = max(1e-6, 1e-8 * abs(candidate["loglik"]))
        base = {"accepted": True, "likelihood_tolerance": tolerance,
                "failed_optimizer_count": sum(not record["success"] for record in optimizers)}
        if not free:
            return dict(base, kind="fixed_parameters", optimizer_indices=[])
        endpoints = [i for i, record in enumerate(optimizers) if record["success"]
                     and record["loglik"] is not None
                     and candidate["loglik"] - record["loglik"] <= tolerance]
        if endpoints:
            return dict(base, kind="optimizer", optimizer_indices=endpoints)
        # Exact zero is outside the positive log grid; the positive lower/upper
        # endpoints are also evaluated exactly.  A grid point alone is not a
        # certificate: its adjacent scalar profile must have converged.  For
        # the all-zero OU corner, require both one-sided profile directions.
        boundary = []
        for i, record in enumerate(optimizers):
            if not record["success"] or not record["kind"].endswith("_profile"):
                continue
            which = record["kind"].removesuffix("_profile")
            other = "eta" if which == "alpha" else "alpha"
            if candidate[other] != record["other"]:
                continue
            lo, hi = parameter_bounds[which]
            value = candidate[which]
            at_lower = value == 0 or np.isclose(value, lo, rtol=1e-12, atol=0)
            at_upper = np.isclose(value, hi, rtol=1e-12, atol=0)
            if ((at_lower and np.isclose(record["bracket"][0], lo, rtol=1e-12, atol=0))
                    or (at_upper and np.isclose(record["bracket"][1], hi, rtol=1e-12, atol=0))):
                boundary.append((i, which))
        needed = set(free) if all(candidate[name] == 0 for name in free) else set()
        if boundary and (not needed or needed <= {which for _, which in boundary}):
            return dict(base, kind="accepted_boundary", optimizer_indices=[i for i, _ in boundary])
        return None

    convergence = certify(best)
    refined = False
    if convergence is None:
        refined = True
        # One derivative-free local refinement also handles exact-zero corners.
        # log(parameter + offset) admits zero, unlike the original positive
        # log search; nonzero initial simplex edges avoid a degenerate corner.
        offsets = np.array([parameter_bounds[name][0] for name in free])
        upper = np.array([parameter_bounds[name][1] for name in free])
        lower_log, upper_log = np.log(offsets), np.log(upper + offsets)
        start = np.log(np.array([best[name] for name in free]) + offsets)
        simplex = np.tile(start, (len(free) + 1, 1))
        for j in range(len(free)):
            step = .05 * (upper_log[j] - lower_log[j])
            simplex[j + 1, j] += step if start[j] + step <= upper_log[j] else -step

        def unpack(logvalues):
            values = dict(alpha=fixed_alpha, eta=fixed_eta)
            values.update(zip(free, np.clip(np.exp(logvalues) - offsets, 0, upper)))
            return float(values["alpha"]), float(values["eta"])

        opt = minimize(lambda v: objective(*unpack(v)), start, method="Nelder-Mead",
                       bounds=list(zip(lower_log, upper_log)),
                       options={"initial_simplex": simplex, "maxiter": 300, "maxfev": 600,
                                "xatol": 2e-5, "fatol": max(1e-7, 1e-9 * abs(best["loglik"]))})
        record_optimizer(opt, "selected_candidate_refinement", *unpack(opt.x),
                         start=[best["alpha"], best["eta"]])
        best = dict(max((v for v in cache.values() if v is not None), key=lambda value: value["loglik"]))
        convergence = certify(best)
    if convergence is None:
        diagnostics = {"model": model, "profiles": profiles, "optimizers": optimizers,
                       "best_candidate": {k: best[k] for k in ("alpha", "eta", "loglik")},
                       "refinement_attempted": refined,
                       "likelihood_tolerance": max(1e-6, 1e-8 * abs(best["loglik"]))}
        raise PhyloOptimizationError(
            "selected phylogenetic likelihood candidate is not certified after refinement "
            f"(alpha={best['alpha']:.9g}, eta={best['eta']:.9g}, loglik={best['loglik']:.9g}); "
            "inspect exception.diagnostics for all optimizer failures", diagnostics)
    convergence["refinement_attempted"] = refined
    convergence["parameter_boundaries"] = {}
    for name in free:
        value, (lo, hi) = best[name], parameter_bounds[name]
        if value == 0:
            label = "BM_boundary" if name == "alpha" else "zero_tip_error"
        elif np.isclose(value, lo, rtol=1e-3, atol=0):
            label = "positive_search_lower"
        elif np.isclose(value, hi, rtol=1e-3, atol=0):
            label = "search_upper"
        else:
            label = "interior"
        convergence["parameter_boundaries"][name] = label
    best["k"] += int(fixed_alpha is None) + int(fixed_eta is None)
    best["aic"] = 2 * best["k"] - 2 * best["loglik"]
    best.update(model=model, estimate_eta=fixed_eta is None, profiles=profiles, optimizers=optimizers,
                convergence=convergence,
                alpha_bounds=list(alpha_bounds), eta_bounds=list(eta_bounds),
                alpha_at_bound=bool(fixed_alpha is None and (best["alpha"] == 0 or
                    np.isclose(best["alpha"], alpha_bounds[0], rtol=1e-3, atol=0) or
                    np.isclose(best["alpha"], alpha_bounds[1], rtol=1e-3, atol=0))),
                eta_at_bound=bool(fixed_eta is None and (best["eta"] == 0 or
                    np.isclose(best["eta"], eta_bounds[0], rtol=1e-3, atol=0) or
                    np.isclose(best["eta"], eta_bounds[1], rtol=1e-3, atol=0))),
                numerical_search="boundary profiles plus finite grid and three joint multistarts")
    return best


def predict_joint(kernel, fit, ref_tip_ids, ref_X, ref_Y, target_tip_ids, target_X,
                  *, ref_mask=None, return_cov=False):
    """Condition only on observed reference coordinates; never accept target Y.

    Returned covariance is plug-in predictive covariance including proportional
    tip error, with species-major coordinate order.  It does not integrate
    uncertainty in fitted beta, Sigma, alpha, or eta.
    """
    ref = kernel._tip_ids(ref_tip_ids)
    target = kernel._tip_ids(target_tip_ids)
    if np.intersect1d(ref, target).size:
        raise ValueError("target and reference tips must be disjoint")
    beta, sigma = np.asarray(fit["beta"], dtype=float), np.asarray(fit["sigma"], dtype=float)
    ref_X, target_X, ref_Y = map(lambda a: np.asarray(a, dtype=float), (ref_X, target_X, ref_Y))
    if (beta.ndim != 2 or beta.shape[1] != 4 or sigma.shape != (4, 4)
            or ref_X.shape != (len(ref), len(beta)) or target_X.shape != (len(target), len(beta))
            or ref_Y.shape != (len(ref), 4) or not np.isfinite(ref_X).all()
            or not np.isfinite(target_X).all() or not np.isfinite(beta).all() or not np.isfinite(sigma).all()):
        raise ValueError("incompatible/nonfinite prediction inputs")
    if ref_mask is None:
        mask = np.isfinite(ref_Y)
    else:
        mask = np.asarray(ref_mask)
        if mask.dtype != np.bool_ or mask.shape != ref_Y.shape:
            raise ValueError("ref_mask must be boolean and match reference trait shape")
        if not np.isfinite(ref_Y[mask]).all():
            raise ValueError("observed reference traits must be finite")
    a, e = fit["alpha"], fit["eta"]
    mean = (target_X @ beta).reshape(-1)
    observed = np.flatnonzero(mask.reshape(-1))
    covariance = np.kron(kernel.covariance(target, alpha=a, eta=e), sigma) if return_cov else None
    if len(observed):
        rr = np.kron(kernel.covariance(ref, alpha=a, eta=e), sigma)[np.ix_(observed, observed)]
        qr = np.kron(kernel.covariance(target, ref, alpha=a, eta=e), sigma)[:, observed]
        residual = (ref_Y - ref_X @ beta).reshape(-1)[observed]
        try:
            factor = cho_factor(rr, lower=True, check_finite=False)
            mean += qr @ cho_solve(factor, residual, check_finite=False)
            if return_cov:
                covariance -= qr @ cho_solve(factor, qr.T, check_finite=False)
                covariance = (covariance + covariance.T) / 2
        except np.linalg.LinAlgError as exc:
            raise ValueError("singular observed-reference covariance") from exc
    result = mean.reshape(len(target), 4)
    return (result, covariance) if return_cov else result
