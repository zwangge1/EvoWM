"""Optional topology-only units for an Allen-focused JEPA hierarchy.

A unit is a maximal monophyletic clade whose crown age does not exceed a
predeclared threshold (20 Myr by default for BirdTree trees).  The result depends
only on topology and branch lengths: no morphology, climate, developmental mode,
locomotion label, or outcome is used to create the units.  These are phylogenetic
aggregation units, NOT an observed evolutionary time series or ancestor states.

Compute codes on the original tree, then filter members to the current data split
with ``clade_members``.  Do not pool held-out members into training units.  A
pruned unit can have fewer members, but retains the original node identifier.

The existing Tree parser represents omitted Newick branch lengths as zero.
Consequently this module cannot distinguish every omitted length from a genuine
zero-length branch.  It rejects invalid, non-ultrametric, and all-zero-height
trees; callers must also use a dated source with explicit branch lengths.
"""
from __future__ import annotations

import numpy as np

from .tree import Tree


def _integer_vector(values, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1 or arr.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a one-dimensional integer array")
    return arr


def _validated_tree(tree: Tree, rtol: float, atol: float):
    """Validate the array tree and return a root-first traversal and crown ages."""
    parent = _integer_vector(tree.parent, "parent")
    ptr = _integer_vector(tree.child_ptr, "child_ptr")
    children = _integer_vector(tree.child_idx, "child_idx")
    tips = _integer_vector(tree.tip_nodes, "tip_nodes")
    depth = np.asarray(tree.depth, dtype=np.float64)
    n = len(parent)
    root = tree.root
    if not isinstance(root, (int, np.integer)) or not 0 <= root < n:
        raise ValueError("tree must have a valid root node")
    if depth.shape != (n,) or not np.isfinite(depth).all():
        raise ValueError("node depths must be finite and match the node count")
    if len(ptr) != n + 1 or ptr[0] != 0 or ptr[-1] != len(children) or np.any(np.diff(ptr) < 0):
        raise ValueError("invalid child CSR pointers")
    if len(children) != n - 1 or np.any(children < 0) or np.any(children >= n):
        raise ValueError("invalid child node identifiers or edge count")
    expected_children = np.delete(np.arange(n), root)
    if not np.array_equal(np.sort(children), expected_children):
        raise ValueError("each non-root node must have exactly one incoming edge")
    if parent[root] != -1 or np.any(parent[expected_children] < 0) or np.any(parent[expected_children] >= n):
        raise ValueError("invalid parent identifiers")
    traversal = []
    seen = np.zeros(n, dtype=bool)
    stack = [int(root)]
    while stack:
        node = stack.pop()
        if seen[node]:
            raise ValueError("tree contains a cycle or duplicate child")
        seen[node] = True
        traversal.append(node)
        ch = children[ptr[node]:ptr[node + 1]]
        if np.any(parent[ch] != node):
            raise ValueError("parent and child arrays disagree")
        if np.any(depth[ch] < depth[node]):
            raise ValueError("branch lengths must be nonnegative")
        stack.extend(int(v) for v in ch)
    if not seen.all():
        raise ValueError("tree contains disconnected nodes")
    leaves = np.flatnonzero(np.diff(ptr) == 0)
    if not np.array_equal(np.sort(tips), leaves):
        raise ValueError("tip_nodes must enumerate every leaf exactly once")
    if len(tree.tip_names) != len(tips) or len(set(tree.tip_names)) != len(tips):
        raise ValueError("tip names must be unique and match tip_nodes")
    heights = depth[tips] - depth[root]
    if not len(heights) or not np.max(heights) > 0:
        raise ValueError("a positive dated tree height is required")
    if not np.allclose(heights, np.max(heights), rtol=rtol, atol=atol):
        raise ValueError("phylogenetic crown-age units require an ultrametric tree")

    # Use the deepest descendant tip (equivalent to tip height on an exact
    # ultrametric tree), so tiny accepted rounding differences are conservative.
    max_tip_depth = depth.copy()
    for node in reversed(traversal):
        ch = children[ptr[node]:ptr[node + 1]]
        if len(ch):
            max_tip_depth[node] = np.max(max_tip_depth[ch])
    return traversal, max_tip_depth - depth


def phylogenetic_clade_codes(
    tree: Tree,
    crown_age: float = 20.0,
    *,
    ultrametric_rtol: float = 1e-5,
    ultrametric_atol: float = 1e-6,
) -> np.ndarray:
    """Return original clade-root node IDs, in ``tree.tip_nodes`` order.

    A root-first cut selects the largest complete clade at or below the age
    threshold.  Clades above the threshold are subdivided until every leaf is
    assigned exactly once; isolated leaves are legitimate singleton units.
    Age uses the tree's branch-length units (Myr in the dated BirdTree inputs).
    The threshold must be fixed before examining prediction results.
    """
    if not np.isfinite(crown_age) or crown_age < 0:
        raise ValueError("crown_age must be finite and nonnegative")
    if (not np.isfinite(ultrametric_rtol) or not np.isfinite(ultrametric_atol)
            or ultrametric_rtol < 0 or ultrametric_atol < 0):
        raise ValueError("ultrametric tolerances must be finite and nonnegative")
    traversal, ages = _validated_tree(tree, ultrametric_rtol, ultrametric_atol)
    node_code = np.full(len(tree.parent), -1, dtype=np.int64)
    for node in traversal:
        parent = int(tree.parent[node])
        if parent >= 0 and node_code[parent] >= 0:
            node_code[node] = node_code[parent]
        elif ages[node] <= crown_age:
            node_code[node] = node
    codes = node_code[tree.tip_nodes]
    if np.any(codes < 0):  # An internal invariant, never silently drop a leaf.
        raise RuntimeError("unassigned leaf after crown-age cut")
    return codes


def clade_members(codes: np.ndarray, keep: np.ndarray) -> dict[int, np.ndarray]:
    """Map clade root to selected tip positions, without mixing data splits.

    ``keep`` must be a boolean array explicitly defining the current split and
    eligibility.  Returned values index the *original tree tip order*, not the
    selected subset.  Empty units are omitted; codes are never recomputed.
    """
    codes = _integer_vector(codes, "codes")
    keep = np.asarray(keep)
    if np.any(codes < 0):
        raise ValueError("clade codes must be nonnegative node identifiers")
    if keep.dtype != np.bool_ or keep.shape != codes.shape:
        raise ValueError("keep must be a boolean mask with one entry per tip")
    indices = np.flatnonzero(keep)
    return {int(code): indices[codes[indices] == code] for code in np.unique(codes[indices])}
