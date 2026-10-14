"""Stochastic SEI / SEIR model of concept spread on a classroom network.

Compartments (per student)
--------------------------
S  Susceptible  - does not yet understand the concept (wrong answer).
E  Exposed      - has heard a correct explanation from a peer but has not yet
                  internalised it (cannot explain it to others yet).
I  Infected     - understands the concept and can convince peers ("knower").
R  Recovered    - (SEIR only) understood it but lost it again; no longer
                  convinces peers.

With ``gamma`` = 0 (default) there is no recovery and the model is SEI: within
one Peer Instruction session learning is not lost. ``gamma`` > 0 turns it into
SEIR, and ``relearn`` > 0 into SEIRS (lost students can be re-convinced).
(Set ``forget`` > 0 to let E students drop back to S.)

Discrete-time dynamics, one step = one discussion round (e.g. ~30 s):

    S -> E   with probability 1 - (1 - beta)^m      m = number of I neighbours
    S -> E   with probability alpha                 (self study / instructor cue)
    E -> I   with probability sigma                 mean incubation 1/sigma rounds
    E -> S   with probability forget                (optional, default 0)
    I -> R   with probability gamma                 SEIR: learning lost after ~1/gamma rounds
    R -> S   with probability relearn               SEIRS: lost students can learn again

Peer Instruction protocol (Mazur):
  1. ConcepTest posed, students vote individually  -> initial I = correct voters
  2. Peer discussion on the interaction network    -> SEI dynamics above
  3. Re-vote                                       -> final fraction in I
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np

S, E, I, R = 0, 1, 2, 3


@dataclass
class SEIParams:
    beta: float = 0.15          # per-contact, per-round transmission probability
    sigma: float = 0.3          # E -> I internalisation probability per round
    alpha: float = 0.0          # spontaneous S -> E (non-peer learning)
    forget: float = 0.0         # E -> S (exposure forgotten before internalised)
    gamma: float = 0.0          # I -> R (learning lost); 0 = SEI, > 0 = SEIR
    relearn: float = 0.0        # R -> S (lost students susceptible again; SEIRS)
    initial_correct: float = 0.3  # fraction correct in the first (individual) vote
    seeding: str = "random"     # "random", "high_degree" or "low_degree"
    steps: int = 30             # discussion rounds


@dataclass
class SEIResult:
    S: np.ndarray
    E: np.ndarray
    I: np.ndarray
    R: np.ndarray
    final_state: np.ndarray
    params: SEIParams = field(repr=False)

    @property
    def n(self) -> int:
        return int(self.S[0] + self.E[0] + self.I[0] + self.R[0])


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

    hist = np.zeros((p.steps + 1, 4), dtype=int)
    hist[0] = np.bincount(state, minlength=4)
    for t in range(1, p.steps + 1):
        m = A @ (state == I).astype(float)                 # infected neighbours
        p_expose = 1.0 - (1.0 - p.alpha) * (1.0 - p.beta) ** m
        r = rng.random(state.size)
        # all transitions use the state at the start of the round (synchronous update)
        new_E = (state == S) & (r < p_expose)
        new_I = (state == E) & (r < p.sigma)
        lost = (state == E) & (r >= p.sigma) & (r < p.sigma + p.forget)
        new_R = (state == I) & (r < p.gamma)
        relearn = (state == R) & (r < p.relearn)
        state[new_E] = E
        state[new_I] = I
        state[lost] = S
        state[new_R] = R
        state[relearn] = S
        hist[t] = np.bincount(state, minlength=4)

    return SEIResult(hist[:, S], hist[:, E], hist[:, I], hist[:, R], state, p)


def monte_carlo(g_factory, p: SEIParams, runs: int = 200, seed: int = 0):
    """Repeat ``simulate`` over ``runs`` realisations.

    ``g_factory(rng_seed)`` returns a graph, so stochastic topologies
    (random, small-world, scale-free) are re-sampled every run.
    Returns an array of shape (runs, steps + 1, 4) with S, E, I, R counts.
    """
    ss = np.random.SeedSequence(seed)
    out = np.zeros((runs, p.steps + 1, 4), dtype=int)
    for k, child in enumerate(ss.spawn(runs)):
        rng = np.random.default_rng(child)
        g = g_factory(int(rng.integers(2**31)))
        res = simulate(g, p, rng)
        out[k] = np.stack([res.S, res.E, res.I, res.R], axis=1)
    return out


def time_to_fraction(I_curve: np.ndarray, n: int, target: float) -> float:
    """First round at which I/n >= target (NaN if never reached)."""
    hit = np.nonzero(I_curve / n >= target)[0]
    return float(hit[0]) if hit.size else float("nan")


def mean_field(n: int, mean_degree: float, p: SEIParams) -> np.ndarray:
    """Homogeneous-mixing discrete-time SEI/SEIR (deterministic baseline).

    Each S student meets ``mean_degree`` peers, a fraction I/n of whom know the
    concept. Useful to show how far each network departs from "well mixed".
    """
    s, e, i, r = 1.0 - p.initial_correct, 0.0, p.initial_correct, 0.0
    out = [(s, e, i, r)]
    for _ in range(p.steps):
        p_expose = 1.0 - (1.0 - p.alpha) * (1.0 - p.beta) ** (mean_degree * i)
        new_e, new_i, lost = s * p_expose, e * p.sigma, e * p.forget
        new_r, back = i * p.gamma, r * p.relearn
        s, e, i, r = (s - new_e + lost + back, e + new_e - new_i - lost,
                      i + new_i - new_r, r + new_r - back)
        out.append((s, e, i, r))
    return np.array(out) * n
