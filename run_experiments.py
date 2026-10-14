"""Compare SEI learning dynamics across classroom network topologies.

Usage
-----
    python run_experiments.py                       # default experiment
    python run_experiments.py --n 40 --runs 500 --beta 0.15 --sigma 0.3
    python run_experiments.py --initial-correct 0.5 --seeding high_degree

Outputs (in --out, default ./results):
    summary.csv               network metrics + learning outcomes per topology
    timeseries.csv            mean/CI of S, E, I per round per topology
    beta_sweep.csv            final learning gain vs beta per topology
    fig_networks.png          drawing of every topology
    fig_learning_curves.png   fraction in I over discussion rounds
    sei_counts_vs_model.csv   S/E/I counts per round: network vs. SEI model
    fig_sei_<topology>.png    S, E, I counts on one network vs. the SEI model
    fig_sei_all_networks.png  the same comparison for every topology in one grid
    fig_beta_sweep.png        final fraction in I vs transmission probability
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from sei_classroom import (TOPOLOGIES, SEIParams, build_network, mean_field, monte_carlo,
                           network_metrics, time_to_fraction)

# Validated categorical palette (fixed order) and recessive text/grid inks.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
TOPO_COLOR = dict(zip(TOPOLOGIES, PALETTE))
LABELS = {
    "complete": "Whole class", "small_groups": "Small groups (4)", "pairs": "Pairs",
    "grid": "Seating grid", "ring": "Ring", "small_world": "Small world",
    "scale_free": "Scale free", "random": "Random",
}

plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "lines.linewidth": 2, "font.size": 10, "legend.frameon": False,
})


def summarise(sims: np.ndarray, n: int, p: SEIParams) -> dict:
    I_frac = sims[:, :, 2] / n
    gain = (I_frac[:, -1] - I_frac[:, 0]) / np.maximum(1 - I_frac[:, 0], 1e-9)
    t80 = np.array([time_to_fraction(run[:, 2], n, 0.8) for run in sims])
    return {
        "initial_I": I_frac[:, 0].mean(),
        "final_I_mean": I_frac[:, -1].mean(),
        "final_I_sd": I_frac[:, -1].std(),
        # Hake-style normalised gain: share of initially-wrong students who learned
        "normalised_gain": gain.mean(),
        "t80_median": np.nanmedian(t80) if np.isfinite(t80).any() else np.nan,
        "p_reach_80": np.isfinite(t80).mean(),
    }


COMPARTMENTS = [("S", "Susceptible", PALETTE[0]),
                ("E", "Exposed", PALETTE[1]),
                ("I", "Infected", PALETTE[2])]


def average_degree(name: str, n: int, kw: dict, samples: int = 50) -> float:
    """Mean degree averaged over samples, since random topologies vary per draw."""
    return float(np.mean([network_metrics(build_network(name, n, seed=s, **kw))["mean_degree"]
                          for s in range(samples)]))


def _draw_comparison(ax, sims: np.ndarray, model: np.ndarray, rounds: np.ndarray):
    for c, (_, label, color) in enumerate(COMPARTMENTS):
        v = sims[:, :, c]
        ax.fill_between(rounds, np.percentile(v, 2.5, 0), np.percentile(v, 97.5, 0),
                        color=color, alpha=0.12, linewidth=0)
        ax.plot(rounds, v.mean(0), color=color, label=f"{label} (network)")
        ax.plot(rounds, model[:, c], color=color, linestyle="--", linewidth=1.5,
                label=f"{label} (SEI model)")


def compare_with_sei_model(all_sims: dict, n: int, p: SEIParams, kw: dict,
                           out: Path) -> pd.DataFrame:
    """Number of students in S, E and I per round on each network, next to the
    classic well-mixed SEI model with the same mean number of contacts.

    Writes one figure per topology plus a combined grid; returns the counts table.
    """
    rounds = np.arange(p.steps + 1)
    rows = []
    grid_fig, grid_axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
    for ax_grid, name in zip(grid_axes.flat, TOPOLOGIES):
        sims = all_sims[name]
        k = average_degree(name, n, kw)
        model = mean_field(n, k, p)
        for t in rounds:
            row = {"topology": name, "mean_degree": k, "round": t}
            for c, (comp, _, _) in enumerate(COMPARTMENTS):
                row[f"{comp}_network"] = sims[:, t, c].mean()
                row[f"{comp}_network_lo95"] = np.percentile(sims[:, t, c], 2.5)
                row[f"{comp}_network_hi95"] = np.percentile(sims[:, t, c], 97.5)
                row[f"{comp}_model"] = model[t, c]
            rows.append(row)

        fig, ax = plt.subplots(figsize=(8, 5.6))
        _draw_comparison(ax, sims, model, rounds)
        ax.set_xlabel("Peer discussion round")
        ax.set_ylabel("Number of students")
        ax.set_ylim(0, n * 1.02)
        ax.set_title(f"{LABELS[name]}: S → E → I on the network vs. SEI model "
                     f"(mean contacts = {k:.1f})", color=INK)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=8, ncol=3)
        fig.tight_layout()
        fig.savefig(out / f"fig_sei_{name}.png", dpi=150)
        plt.close(fig)

        _draw_comparison(ax_grid, sims, model, rounds)
        ax_grid.set_title(f"{LABELS[name]} (k = {k:.1f})", color=INK)
        ax_grid.set_ylim(0, n * 1.02)

    for ax in grid_axes[1]:
        ax.set_xlabel("Peer discussion round")
    for ax in grid_axes[:, 0]:
        ax.set_ylabel("Number of students")
    handles, labels = grid_axes.flat[0].get_legend_handles_labels()
    grid_fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=9)
    grid_fig.suptitle("Susceptible → Exposed → Infected counts: network simulation (solid, 95% band) "
                      "vs. well-mixed SEI model (dashed)", color=INK)
    grid_fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    grid_fig.savefig(out / "fig_sei_all_networks.png", dpi=150)
    plt.close(grid_fig)
    return pd.DataFrame(rows)


def plot_networks(n: int, kw: dict, out: Path):
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for ax, name in zip(axes.flat, TOPOLOGIES):
        g = build_network(name, n, seed=1, **kw)
        if name == "grid":
            pos = {v: (d["seat"][1], -d["seat"][0]) for v, d in g.nodes(data=True)}
        elif name in ("ring", "small_world"):
            pos = nx.circular_layout(g)
        else:
            pos = nx.spring_layout(g, seed=1)
        nx.draw_networkx_edges(g, pos, ax=ax, edge_color="#b7b6b0", width=0.6)
        nx.draw_networkx_nodes(g, pos, ax=ax, node_size=40, node_color=TOPO_COLOR[name],
                               edgecolors="white", linewidths=1)
        ax.set_title(LABELS[name], color=INK)
        ax.axis("off")
    fig.suptitle(f"Classroom interaction networks (n = {n})", color=INK)
    fig.tight_layout()
    fig.savefig(out / "fig_networks.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=40, help="class size")
    ap.add_argument("--runs", type=int, default=300, help="Monte Carlo runs per topology")
    ap.add_argument("--steps", type=int, default=30, help="discussion rounds")
    ap.add_argument("--beta", type=float, default=0.15)
    ap.add_argument("--sigma", type=float, default=0.3)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--forget", type=float, default=0.0)
    ap.add_argument("--initial-correct", type=float, default=0.3)
    ap.add_argument("--seeding", default="random", choices=["random", "high_degree", "low_degree"])
    ap.add_argument("--mean-degree", type=float, default=4,
                    help="target mean degree for ring / small-world / random / scale-free")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    p = SEIParams(beta=a.beta, sigma=a.sigma, alpha=a.alpha, forget=a.forget,
                  initial_correct=a.initial_correct, seeding=a.seeding, steps=a.steps)
    k = int(a.mean_degree)
    kw = {"k": k, "m": max(1, k // 2), "mean_degree": a.mean_degree, "p": 0.1, "group_size": 4}
    rounds = np.arange(p.steps + 1)

    plot_networks(a.n, kw, out)

    rows, ts_rows, all_sims = [], [], {}
    for name in TOPOLOGIES:
        factory = lambda s, name=name: build_network(name, a.n, seed=s, **kw)
        sims = monte_carlo(factory, p, runs=a.runs, seed=a.seed)
        all_sims[name] = sims
        metrics = network_metrics(build_network(name, a.n, seed=a.seed, **kw))
        rows.append({"topology": name, **metrics, **summarise(sims, a.n, p)})
        frac = sims / a.n
        for t in rounds:
            for c, comp in enumerate("SEI"):
                v = frac[:, t, c]
                ts_rows.append({"topology": name, "round": t, "compartment": comp,
                                "mean": v.mean(), "lo95": np.percentile(v, 2.5),
                                "hi95": np.percentile(v, 97.5)})
        print(f"{LABELS[name]:<18} final I = {rows[-1]['final_I_mean']:.2f}  "
              f"gain = {rows[-1]['normalised_gain']:.2f}  t80 = {rows[-1]['t80_median']}")

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False, float_format="%.4f")
    pd.DataFrame(ts_rows).to_csv(out / "timeseries.csv", index=False, float_format="%.4f")

    # --- learning curves: mean fraction in I over time (95% bands are in timeseries.csv)
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in TOPOLOGIES:
        I = all_sims[name][:, :, 2] / a.n
        ax.plot(rounds, I.mean(0), color=TOPO_COLOR[name], label=LABELS[name])
    mf = mean_field(a.n, a.mean_degree, p)[:, 2] / a.n
    ax.plot(rounds, mf, color=MUTED, linestyle="--", linewidth=1.5,
            label=f"Mean field (k = {a.mean_degree:g})")
    ax.set_xlabel("Peer discussion round")
    ax.set_ylabel("Fraction who understand (I)")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"SEI concept spread by network topology  "
                 f"(β = {p.beta}, σ = {p.sigma}, I₀ = {p.initial_correct})", color=INK)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_learning_curves.png", dpi=150)
    plt.close(fig)

    # --- S -> E -> I counts per topology: network simulation vs. SEI model ----
    compare = compare_with_sei_model(all_sims, a.n, p, kw, out)
    compare.to_csv(out / "sei_counts_vs_model.csv", index=False, float_format="%.2f")

    # --- sweep transmission probability beta ----------------------------------
    betas = np.round(np.linspace(0.02, 0.5, 13), 3)
    sweep = []
    for name in TOPOLOGIES:
        factory = lambda s, name=name: build_network(name, a.n, seed=s, **kw)
        for b in betas:
            sims = monte_carlo(factory, replace(p, beta=float(b)), runs=max(50, a.runs // 4), seed=a.seed)
            sweep.append({"topology": name, "beta": b, **summarise(sims, a.n, p)})
    sweep = pd.DataFrame(sweep)
    sweep.to_csv(out / "beta_sweep.csv", index=False, float_format="%.4f")

    fig, ax = plt.subplots(figsize=(8, 5))
    for name in TOPOLOGIES:
        d = sweep[sweep.topology == name]
        ax.plot(d.beta, d.final_I_mean, marker="o", markersize=4,
                color=TOPO_COLOR[name], label=LABELS[name])
    ax.set_xlabel("Per-contact transmission probability β")
    ax.set_ylabel(f"Fraction who understand after {p.steps} rounds")
    ax.set_ylim(0, 1.02)
    ax.set_title("Final learning vs. persuasiveness of peer explanations", color=INK)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_beta_sweep.png", dpi=150)
    plt.close(fig)

    print(f"\nWrote results to {out.resolve()}")


if __name__ == "__main__":
    main()
