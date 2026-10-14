"""Stochastic SEI model of concept spread on a classroom network.

Compartments (per student)
--------------------------
S  Susceptible  - does not yet understand the concept (wrong answer).
E  Exposed      - has heard a correct explanation from a peer but has not yet
                  internalised it (cannot explain it to others yet).
I  Infected     - understands the concept and can convince peers ("knower").

There is no recovery compartment: within one Peer Instruction session we
assume learning is not lost. (Set ``forget`` > 0 to let E students drop back
to S if you want to test that assumption.)

Discrete-time dynamics, one step = one discussion round (e.g. ~30 s):

    S -> E   with probability 1 - (1 - beta)^m      m = number of I neighbours
    S -> E   with probability alpha                 (self study / instructor cue)
    E -> I   with probability sigma                 mean incubation 1/sigma rounds
    E -> S   with probability forget                (optional, default 0)

Peer Instruction protocol (Mazur):
  1. ConcepTest posed, students vote individually  -> initial I = correct voters
  2. Peer discussion on the interaction network    -> SEI dynamics above
  3. Re-vote                                       -> final fraction in I
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np

S, E, I = 0, 1, 2


@dataclass
class SEIParams:
    beta: float = 0.15          # per-contact, per-round transmission probability
    sigma: float = 0.3          # E -> I internalisation probability per round
    alpha: float = 0.0          # spontaneous S -> E (non-peer learning)
    forget: float = 0.0         # E -> S (exposure forgotten before internalised)
    initial_correct: float = 0.3  # fraction correct in the first (individual) vote
    seeding: str = "random"     # "random", "high_degree" or "low_degree"
    steps: int = 30             # discussion rounds


@dataclass
class SEIResult:
    S: np.ndarray
    E: np.ndarray
    I: np.ndarray
    final_state: np.ndarray
    params: SEIParams = field(repr=False)

    @property
    def n(self) -> int:
        return int(self.S[0] + self.E[0] + self.I[0])


def initial_state(g: nx.Graph, p: SEIParams, rng: np.random.Generator) -> np.ndarray:
    n = g.number_of_nodes()
    k = int(round(p.initial_correct * n))
    state = np.full(n, S, dtype=np.int8)
    if k == 0:
        return state
    if p.seeding == "random":
        seeds = rng.choice(n, size=k, replace=False)
    elif p.seeding in ("high_degree", "low_degree"):
        deg = np.array([g.degree(v) for v in range(n)], dtype=float)
        deg += rng.random(n) * 1e-6  # random tie-breaking
        order = np.argsort(deg)
        seeds = order[-k:] if p.seeding == "high_degree" else order[:k]
    else:
        raise ValueError(f"unknown seeding {p.seeding!r}")
    state[seeds] = I
    return state


def simulate(g: nx.Graph, p: SEIParams, rng: np.random.Generator | int | None = None,
             state0: np.ndarray | None = None) -> SEIResult:
    """Run one stochastic realisation. Nodes must be labelled 0..n-1."""
    rng = np.random.default_rng(rng)
    A = nx.to_scipy_sparse_array(g, nodelist=range(g.number_of_nodes()), format="csr")
    state = initial_state(g, p, rng) if state0 is None else state0.copy()

    hist = np.zeros((p.steps + 1, 3), dtype=int)
    hist[0] = np.bincount(state, minlength=3)
    for t in range(1, p.steps + 1):
        m = A @ (state == I).astype(float)                 # infected neighbours
        p_expose = 1.0 - (1.0 - p.alpha) * (1.0 - p.beta) ** m
        r = rng.random(state.size)
        # all transitions use the state at the start of the round (synchronous update)
        new_E = (state == S) & (r < p_expose)
        new_I = (state == E) & (r < p.sigma)
        lost = (state == E) & (r >= p.sigma) & (r < p.sigma + p.forget)
        state[new_E] = E
        state[new_I] = I
        state[lost] = S
        hist[t] = np.bincount(state, minlength=3)

    return SEIResult(hist[:, S], hist[:, E], hist[:, I], state, p)


def monte_carlo(g_factory, p: SEIParams, runs: int = 200, seed: int = 0):
    """Repeat ``simulate`` over ``runs`` realisations.

    ``g_factory(rng_seed)`` returns a graph, so stochastic topologies
    (random, small-world, scale-free) are re-sampled every run.
    Returns an array of shape (runs, steps + 1, 3) with S, E, I counts.
    """
    ss = np.random.SeedSequence(seed)
    out = np.zeros((runs, p.steps + 1, 3), dtype=int)
    for k, child in enumerate(ss.spawn(runs)):
        rng = np.random.default_rng(child)
        g = g_factory(int(rng.integers(2**31)))
        res = simulate(g, p, rng)
        out[k] = np.stack([res.S, res.E, res.I], axis=1)
    return out


def time_to_fraction(I_curve: np.ndarray, n: int, target: float) -> float:
    """First round at which I/n >= target (NaN if never reached)."""
    hit = np.nonzero(I_curve / n >= target)[0]
    return float(hit[0]) if hit.size else float("nan")


def mean_field(n: int, mean_degree: float, p: SEIParams) -> np.ndarray:
    """Homogeneous-mixing discrete-time SEI (deterministic baseline).

    Each S student meets ``mean_degree`` peers, a fraction I/n of whom know the
    concept. Useful to show how far each network departs from "well mixed".
    """
    s, e, i = 1.0 - p.initial_correct, 0.0, p.initial_correct
    out = [(s, e, i)]
    for _ in range(p.steps):
        p_expose = 1.0 - (1.0 - p.alpha) * (1.0 - p.beta) ** (mean_degree * i)
        new_e, new_i, lost = s * p_expose, e * p.sigma, e * p.forget
        s, e, i = s - new_e + lost, e + new_e - new_i - lost, i + new_i
        out.append((s, e, i))
    return np.array(out) * n
