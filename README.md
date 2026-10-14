# Learning Dynamics Over Different Interaction Network Topologies during Peer Instruction

Simulation code that models how understanding of a concept spreads through a
classroom during Peer Instruction (PI), using a **Susceptible–Exposed–Infected
(SEI)** model on different student interaction networks.

## The model

| Compartment | Meaning in the classroom |
|---|---|
| **S** (Susceptible) | Student does not yet understand the concept (wrong answer in the vote) |
| **E** (Exposed) | Student has heard a correct explanation from a peer or a teacher but has not internalised it yet |
| **I** (Infected) | Student understands the concept and can convince others |

Each time step is one peer-discussion round. Updates are synchronous:

- **S → E** with probability `1 − (1 − α)(1 − β)^m`, where `m` is the number of
  neighbours in **I**, `β` is how convincing a single peer explanation is, and
  `α` is optional learning that doesn't come from peers (default 0).
- **E → I** with probability `σ`, so internalising the idea takes `1/σ` rounds on average.
- **E → S** with probability `forget` (optional, default 0).

The model has no recovery: within a single PI session, nobody unlearns.

How the model maps onto the PI cycle (Mazur):
1. A ConcepTest is posed and students vote individually. The fraction who are
   correct is `initial_correct` and is the initial **I**.
2. Peer discussion runs on the interaction network. This is the SEI dynamics.
3. Students re-vote. The fraction in **I** after `steps` rounds is the outcome.

## Topologies (`sei_classroom/topologies.py`)

`complete` (whole class), `small_groups` (groups of 4), `pairs` (think-pair-share),
`grid` (seating chart, neighbours next to you), `ring`, `small_world`
(Watts–Strogatz), `scale_free` (Barabási–Albert, "hub" students) and `random`
(Erdős–Rényi). You can add your own observed classroom network by building an
`nx.Graph` with nodes `0..n-1` and passing it to `simulate`.

## Running

```bash
pip install -r requirements.txt
python run_experiments.py                      # n=40, 300 runs per topology
python run_experiments.py --help               # all parameters
python run_experiments.py --initial-correct 0.5 --seeding high_degree --beta 0.1
python -m pytest tests
```

Outputs go to `results/`:

- `summary.csv`: network metrics (mean degree, clustering, path length,
  components) next to learning outcomes (final fraction in I, normalised gain,
  median rounds to reach 80% understanding).
- `timeseries.csv`: mean and 95% interval of S/E/I for every round.
- `beta_sweep.csv`: final learning as a function of β.
- `sei_counts_vs_model.csv`: number of students in S, E and I at every round
  for each network (mean and 95% interval), next to the classic well-mixed SEI
  model given the same mean number of contacts.
- Figures: `fig_networks.png`, `fig_learning_curves.png`, `fig_beta_sweep.png`,
  `fig_sei_<topology>.png` (one per network: S → E → I counts from the
  simulation as solid lines, the SEI model as dashed lines) and
  `fig_sei_all_networks.png` (all eight comparisons in one grid).

## Using it in your own scripts

```python
from sei_classroom import SEIParams, build_network, simulate, monte_carlo

g = build_network("grid", n=40, moore=True)
res = simulate(g, SEIParams(beta=0.2, sigma=0.4, initial_correct=0.35), rng=1)
print(res.I)        # number of students in I at each round

sims = monte_carlo(lambda s: build_network("small_world", 40, seed=s, k=4, p=0.2),
                   SEIParams(), runs=500)
```

## Notes for the thesis methodology

- **Mean degree confound.** The topologies don't all have the same number of
  contacts. `complete` has k = 39, while `pairs` has k = 1. The `--mean-degree`
  option matches the ring, small-world, random and scale-free networks so they
  can be compared fairly. Report `mean_degree` together with the outcomes.
- **Fragmented networks.** Groups and pairs are disconnected. A group with no
  initial knower can never learn from peers, which is why those networks
  plateau below 100%. You can test inter-group mixing with
  `small_groups` + `p_between`, or with instructor input via `alpha`.
- **Calibration.** Estimate `initial_correct` from first-vote data. Fit `β`
  and `σ` so the simulated second-vote fraction matches your observed
  re-vote results, for example with a grid search that minimises squared error.
- **Seeding.** `--seeding high_degree` puts the students who answered
  correctly at well-connected positions, for example the instructor seating
  strong students centrally. Compare it with random seeding.
