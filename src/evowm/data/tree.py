"""系统发育树工具：读 Newick、随机旋转的深度优先叶序、相邻叶之间的进化距离。

只依赖 numpy。树存成数组：parent、children（CSR 形式）、到根的距离 depth、叶名。

核心事实：在任一深度优先叶序里，两片叶的最近共同祖先（LCA）深度，等于它们之间
所有相邻叶对 LCA 深度的最小值。所以只要记下相邻叶对的 LCA 深度，就能在剔除部分物种
（例如留出类群）后直接算出剩余相邻叶对的进化距离。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_TOKEN = re.compile(r"\(|\)|,|;|:[^,();]+|[^,();:]+")


@dataclass
class Tree:
    parent: np.ndarray      # (n_nodes,) 根为 -1；节点按先序编号，父节点编号小于子节点
    child_ptr: np.ndarray   # (n_nodes + 1,) children 的 CSR 偏移
    child_idx: np.ndarray   # (n_nodes - 1,)
    depth: np.ndarray       # (n_nodes,) 到根的距离（与枝长同单位，BirdTree 为百万年）
    tip_nodes: np.ndarray   # (n_tips,) 叶节点编号，按 Newick 出现顺序
    tip_names: list[str]    # 与 tip_nodes 同序
    root: int = 0

    @property
    def n_tips(self) -> int:
        return len(self.tip_nodes)

    def children(self, v: int) -> np.ndarray:
        return self.child_idx[self.child_ptr[v]:self.child_ptr[v + 1]]


def parse_newick(s: str) -> Tree:
    parent: list[int] = []
    blen: list[float] = []
    names: dict[int, str] = {}
    stack: list[int] = []
    last = -1            # 最近创建或刚闭合的节点，后续的 ":长度" 或标签作用于它
    after_close = False  # 上一个记号是否为 ")"

    def new_node(p: int) -> int:
        parent.append(p)
        blen.append(0.0)
        return len(parent) - 1

    for tok in _TOKEN.findall(s):
        if tok == "(":
            stack.append(new_node(stack[-1] if stack else -1))
            last, after_close = -1, False
        elif tok == ",":
            last, after_close = -1, False
        elif tok == ")":
            last, after_close = stack.pop(), True
        elif tok == ";":
            break
        elif tok[0] == ":":
            blen[last] = float(tok[1:])
            after_close = False
        else:
            label = tok.strip()
            if not label:
                continue
            if after_close:            # 内部节点标签，忽略
                after_close = False
                continue
            v = new_node(stack[-1])
            names[v] = label
            last = v
    if stack:
        raise ValueError("Newick 括号不配对")

    parent_a = np.asarray(parent, dtype=np.int64)
    blen_a = np.asarray(blen, dtype=np.float64)
    n = len(parent_a)
    depth = np.zeros(n)
    for v in range(1, n):  # 先序编号保证父节点先算
        depth[v] = depth[parent_a[v]] + blen_a[v]

    nonroot = np.flatnonzero(parent_a >= 0)
    order = nonroot[np.argsort(parent_a[nonroot], kind="stable")]
    counts = np.bincount(parent_a[nonroot], minlength=n)
    child_ptr = np.concatenate([[0], np.cumsum(counts)])
    is_tip = counts == 0
    tip_nodes = np.flatnonzero(is_tip)
    missing = [v for v in tip_nodes if v not in names]
    if missing:
        raise ValueError(f"{len(missing)} 片叶没有名字")
    return Tree(
        parent=parent_a, child_ptr=child_ptr, child_idx=order, depth=depth,
        tip_nodes=tip_nodes, tip_names=[names[v] for v in tip_nodes],
    )


def read_trees(path: str | Path, max_trees: int | None = None) -> list[Tree]:
    """读一个含多棵树的 Newick 文件（每棵以 ; 结尾）。"""
    text = Path(path).read_text()
    chunks = [c for c in text.split(";") if c.strip()]
    if max_trees is not None:
        chunks = chunks[:max_trees]
    return [parse_newick(c + ";") for c in chunks]


def leaf_order(tree: Tree, rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """深度优先叶序。

    rng 为 None 时按 Newick 原顺序；否则每个内部节点的子节点随机排列（随机旋转）。
    返回 (tips, lca_depth)：tips 是叶在 tree.tip_names 里的下标，长度 n_tips；
    lca_depth[k] 是 tips[k] 与 tips[k+1] 的 LCA 到根的距离，长度 n_tips - 1。
    """
    tip_pos = np.full(len(tree.parent), -1, dtype=np.int64)
    tip_pos[tree.tip_nodes] = np.arange(tree.n_tips)
    tips: list[int] = []
    lca: list[float] = []
    stack = [tree.root]
    after_leaf = False
    ptr, idx, depth, parent = tree.child_ptr, tree.child_idx, tree.depth, tree.parent
    while stack:
        v = stack.pop()
        if after_leaf:  # 叶之后弹出的第一个节点，其父节点就是相邻两叶的 LCA
            lca.append(depth[parent[v]])
            after_leaf = False
        a, b = ptr[v], ptr[v + 1]
        if a == b:
            tips.append(tip_pos[v])
            after_leaf = True
        else:
            ch = idx[a:b]
            stack.extend(ch[::-1] if rng is None else rng.permutation(ch))
    return np.asarray(tips, dtype=np.int64), np.asarray(lca)


def restrict(tips: np.ndarray, lca_depth: np.ndarray, keep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """只保留 keep[tips] 为真的叶，返回新的叶序与相邻叶 LCA 深度。"""
    pos = np.flatnonzero(keep[tips])
    if len(pos) < 2:
        return tips[pos], np.empty(0)
    # 截到 pos[-1] 后 reduceat 在每个 [pos[i], pos[i+1]) 上取最小，正好覆盖两片保留叶之间的所有相邻叶对
    new_lca = np.minimum.reduceat(lca_depth[:pos[-1]], pos[:-1])
    return tips[pos], new_lca


def pair_distance(tree: Tree, tips: np.ndarray, lca_depth: np.ndarray) -> np.ndarray:
    """相邻叶之间的进化距离（沿树的路径长度）。"""
    d_tip = tree.depth[tree.tip_nodes[tips]]
    return d_tip[:-1] + d_tip[1:] - 2.0 * lca_depth
