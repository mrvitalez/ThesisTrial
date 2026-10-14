"""Classroom interaction network topologies.

Each builder returns an undirected ``networkx.Graph`` whose nodes 0..n-1 are
students and whose edges are the peer-discussion links that exist during the
Peer Instruction (PI) "convince your neighbour" phase.

Topologies
----------
complete      Whole-class discussion: everyone can talk to everyone.
small_groups  Fixed discussion groups (cliques) of ``group_size`` students,
              optionally with a few random links between groups (``p_between``).
pairs         Think-pair-share: groups of two (small_groups with group_size=2).
grid          Seating chart: students in ``rows x cols`` seats talk to the
              students next to them (von Neumann or Moore neighbourhood).
ring          Each student talks to the ``k`` nearest classmates in a circle.
small_world   Watts-Strogatz: seating-like ring with random rewiring ``p``
              (friends sitting far apart, students walking around).
scale_free    Barabasi-Albert: a few highly connected "hub" students.
random        Erdos-Renyi with the requested mean degree.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np

TOPOLOGIES = (
    "complete",
    "small_groups",
    "pairs",
    "grid",
    "ring",
    "small_world",
    "scale_free",
    "random",
)


def _grid_shape(n: int) -> tuple[int, int]:
    """Closest-to-square rows x cols with rows * cols == n (falls back to n x 1)."""
    rows = int(math.isqrt(n))
    while rows > 1 and n % rows:
        rows -= 1
    return rows, n // rows


def small_groups(n: int, group_size: int = 4, p_between: float = 0.0,
                 seed: int | None = None) -> nx.Graph:
    rng = np.random.default_rng(seed)
    g = nx.Graph()
    g.add_nodes_from(range(n))
    groups = [list(range(i, min(i + group_size, n))) for i in range(0, n, group_size)]
    for members in groups:
        for a_idx, a in enumerate(members):
            for b in members[a_idx + 1:]:
                g.add_edge(a, b)
        for a in members:
            g.nodes[a]["group"] = groups.index(members)
    if p_between > 0:
        for a in range(n):
            for b in range(a + 1, n):
                if g.nodes[a]["group"] != g.nodes[b]["group"] and rng.random() < p_between:
                    g.add_edge(a, b)
    return g


def grid(n: int, moore: bool = False) -> nx.Graph:
    rows, cols = _grid_shape(n)
    g = nx.grid_2d_graph(rows, cols)
    if moore:  # include diagonal seatmates
        for r in range(rows - 1):
            for c in range(cols):
                if c + 1 < cols:
                    g.add_edge((r, c), (r + 1, c + 1))
                if c - 1 >= 0:
                    g.add_edge((r, c), (r + 1, c - 1))
    mapping = {(r, c): r * cols + c for r, c in g.nodes}
    g = nx.relabel_nodes(g, mapping)
    for (r, c), i in mapping.items():
        g.nodes[i]["seat"] = (r, c)
    return g


def build_network(name: str, n: int = 40, seed: int | None = None, **kw) -> nx.Graph:
    """Build a classroom network by name. Extra keyword args go to the builder."""
    if name == "complete":
        g = nx.complete_graph(n)
    elif name == "small_groups":
        g = small_groups(n, kw.get("group_size", 4), kw.get("p_between", 0.0), seed)
    elif name == "pairs":
        g = small_groups(n, 2, kw.get("p_between", 0.0), seed)
    elif name == "grid":
        g = grid(n, kw.get("moore", False))
    elif name == "ring":
        g = nx.watts_strogatz_graph(n, kw.get("k", 4), 0.0, seed=seed)
    elif name == "small_world":
        g = nx.connected_watts_strogatz_graph(n, kw.get("k", 4), kw.get("p", 0.1), seed=seed)
    elif name == "scale_free":
        g = nx.barabasi_albert_graph(n, kw.get("m", 2), seed=seed)
    elif name == "random":
        mean_k = kw.get("mean_degree", 4)
        g = nx.gnp_random_graph(n, mean_k / (n - 1), seed=seed)
    else:
        raise ValueError(f"unknown topology {name!r}; choose from {TOPOLOGIES}")
    g.graph["topology"] = name
    return g


def network_metrics(g: nx.Graph) -> dict:
    """Structural descriptors to relate topology to learning outcomes."""
    degrees = np.array([d for _, d in g.degree()])
    components = list(nx.connected_components(g))
    largest = g.subgraph(max(components, key=len))
    return {
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
        "mean_degree": degrees.mean(),
        "degree_std": degrees.std(),
        "density": nx.density(g),
        "clustering": nx.average_clustering(g),
        "components": len(components),
        # path length / diameter are measured on the largest component
        "avg_path_length": nx.average_shortest_path_length(largest) if len(largest) > 1 else 0.0,
        "diameter": nx.diameter(largest) if len(largest) > 1 else 0,
    }
