import numpy as np
import pytest

from sei_classroom import TOPOLOGIES, SEIParams, build_network, simulate


@pytest.mark.parametrize("name", TOPOLOGIES)
def test_population_conserved_and_monotone(name):
    g = build_network(name, 40, seed=3)
    r = simulate(g, SEIParams(steps=20), rng=1)
    assert np.all(r.S + r.E + r.I == 40)
    assert np.all(np.diff(r.I) >= 0)  # no recovery: knowers never decrease
    assert np.all(np.diff(r.S) <= 0)  # forget = 0: nobody returns to S


def test_isolated_students_do_not_learn_from_peers():
    g = build_network("pairs", 40)
    # one knower in pair (0, 1); everyone else is S
    state0 = np.zeros(40, dtype=np.int8)
    state0[0] = 2
    r = simulate(g, SEIParams(beta=1.0, sigma=1.0, steps=10), rng=0, state0=state0)
    assert r.I[-1] == 2


def test_complete_graph_full_learning():
    g = build_network("complete", 30)
    r = simulate(g, SEIParams(beta=0.5, sigma=1.0, initial_correct=0.1, steps=10), rng=0)
    assert r.I[-1] == 30
