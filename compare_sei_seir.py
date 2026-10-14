"""SEI vs. SEIR: what changes when learning can be lost (I -> R)?

SEI   S -> E -> I            understanding, once gained, is kept.
SEIR  S -> E -> I -> R       a student who understands loses it again after
                             ~1/gamma rounds and stops convincing peers.

Usage
-----
    python compare_sei_seir.py                     # gamma = 0.05 (lost after ~20 rounds)
    python compare_sei_seir.py --gamma 0.1 --runs 500
    python compare_sei_seir.py --relearn 0.1       # SEIRS: lost students can relearn

Outputs (in --out, default ./results/seir):
    sei_vs_seir_summary.csv        per topology and gamma: final I, peak I and when,
                                   students who ever understood, final R
    sei_vs_seir_counts.csv         S/E/I/R counts per round, SEI and SEIR
    fig_seir_<topology>.png        SEIR S/E/I/R counts with SEI's I for reference
    fig_seir_all_networks.png      the same for all topologies in one grid
    fig_seir_gamma_sweep.png       final fraction in I vs. gamma per topology
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_experiments import INK, LABELS, MUTED, PALETTE, TOPO_COLOR
from sei_classroom import TOPOLOGIES, SEIParams, build_network, monte_carlo

STATES = [("S", "Susceptible", PALETTE[0]), ("E", "Exposed", PALETTE[1]),
          ("I", "Infected", PALETTE[2]), ("R", "Recovered (learning lost)", PALETTE[3])]


def outcomes(sims: np.ndarray, n: int) -> dict:
    I = sims[:, :, 2]
    peak_round = I.mean(0).argmax()
    return {
        "final_I": I[:, -1].mean() / n,
        "peak_I": I.mean(0).max() / n,
        "peak_round": int(peak_round),
        "final_R": sims[:, -1, 3].mean() / n,
        # everyone who reached I at some point is now in I or R
        "ever_understood": (sims[:, -1, 2] + sims[:, -1, 3]).mean() / n,
    }


def draw(ax, seir: np.ndarray, sei: np.ndarray, rounds: np.ndarray):
    for c, (_, label, color) in enumerate(STATES):
        v = seir[:, :, c]
        ax.fill_between(rounds, np.percentile(v, 2.5, 0), np.percentile(v, 97.5, 0),
                        color=color, alpha=0.12, linewidth=0)
        ax.plot(rounds, v.mean(0), color=color, label=f"{label} (SEIR)")
    ax.plot(rounds, sei[:, :, 2].mean(0), color=PALETTE[2], linestyle="--",
            linewidth=1.5, label="Infected (SEI, no loss)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--runs", type=int, default=300)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--beta", type=float, default=0.15)
    ap.add_argument("--sigma", type=float, default=0.3)
    ap.add_argument("--gamma", type=float, default=0.05, help="I -> R probability per round")
    ap.add_argument("--relearn", type=float, default=0.0, help="R -> S probability (SEIRS)")
    ap.add_argument("--initial-correct", type=float, default=0.3)
    ap.add_argument("--mean-degree", type=float, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="results/seir")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    k = int(a.mean_degree)
    kw = {"k": k, "m": max(1, k // 2), "mean_degree": a.mean_degree, "p": 0.1, "group_size": 4}
    sei_p = SEIParams(beta=a.beta, sigma=a.sigma, initial_correct=a.initial_correct, steps=a.steps)
    seir_p = replace(sei_p, gamma=a.gamma, relearn=a.relearn)
    rounds = np.arange(a.steps + 1)
    model = "SEIRS" if a.relearn > 0 else "SEIR"

    summary, counts = [], []
    grid_fig, grid_axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
    print(f"{'Network':<18}{'SEI final I':>12}{model + ' final I':>14}{'peak I (round)':>16}{'final R':>9}")
    for ax_grid, name in zip(grid_axes.flat, TOPOLOGIES):
        factory = lambda s, name=name: build_network(name, a.n, seed=s, **kw)
        sei = monte_carlo(factory, sei_p, runs=a.runs, seed=a.seed)
        seir = monte_carlo(factory, seir_p, runs=a.runs, seed=a.seed)
        o_sei, o_seir = outcomes(sei, a.n), outcomes(seir, a.n)
        summary.append({"topology": name, "model": "SEI", "gamma": 0.0, **o_sei})
        summary.append({"topology": name, "model": model, "gamma": a.gamma, **o_seir})
        print(f"{LABELS[name]:<18}{o_sei['final_I']:>12.2f}{o_seir['final_I']:>14.2f}"
              f"{o_seir['peak_I']:>10.2f} ({o_seir['peak_round']:>2}){o_seir['final_R']:>9.2f}")
        for label, sims in (("SEI", sei), (model, seir)):
            for t in rounds:
                counts.append({"topology": name, "model": label, "round": t,
                               **{st: sims[:, t, c].mean() for c, (st, _, _) in enumerate(STATES)}})

        fig, ax = plt.subplots(figsize=(8, 5.6))
        draw(ax, seir, sei, rounds)
        ax.set_xlabel("Peer discussion round")
        ax.set_ylabel("Number of students")
        ax.set_ylim(0, a.n * 1.02)
        ax.set_title(f"{LABELS[name]}: {model} (γ = {a.gamma}) vs. SEI", color=INK)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=8, ncol=3)
        fig.tight_layout()
        fig.savefig(out / f"fig_seir_{name}.png", dpi=150)
        plt.close(fig)

        draw(ax_grid, seir, sei, rounds)
        ax_grid.set_title(LABELS[name], color=INK)
        ax_grid.set_ylim(0, a.n * 1.02)

    for ax in grid_axes[1]:
        ax.set_xlabel("Peer discussion round")
    for ax in grid_axes[:, 0]:
        ax.set_ylabel("Number of students")
    handles, labels = grid_axes.flat[0].get_legend_handles_labels()
    grid_fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=9)
    grid_fig.suptitle(f"{model} with learning lost after ~{1 / a.gamma:.0f} rounds (γ = {a.gamma}); "
                      "dashed = Infected under SEI (no loss)", color=INK)
    grid_fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    grid_fig.savefig(out / "fig_seir_all_networks.png", dpi=150)
    plt.close(grid_fig)

    # --- how quickly does forgetting have to happen to matter? ----------------
    gammas = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
    sweep = []
    for name in TOPOLOGIES:
        factory = lambda s, name=name: build_network(name, a.n, seed=s, **kw)
        for g in gammas:
            sims = monte_carlo(factory, replace(seir_p, gamma=g), runs=max(50, a.runs // 3), seed=a.seed)
            sweep.append({"topology": name, "model": model if g else "SEI", "gamma": g, **outcomes(sims, a.n)})
    sweep_df = pd.DataFrame(sweep)
    pd.concat([pd.DataFrame(summary), sweep_df]).drop_duplicates(["topology", "gamma"]).sort_values(
        ["topology", "gamma"]).to_csv(out / "sei_vs_seir_summary.csv", index=False, float_format="%.4f")
    pd.DataFrame(counts).to_csv(out / "sei_vs_seir_counts.csv", index=False, float_format="%.2f")

    fig, ax = plt.subplots(figsize=(8, 5))
    for name in TOPOLOGIES:
        d = sweep_df[sweep_df.topology == name]
        ax.plot(d.gamma, d.final_I, marker="o", markersize=4, color=TOPO_COLOR[name], label=LABELS[name])
    ax.set_xlabel("Probability of losing understanding per round γ  (γ = 0 is SEI)")
    ax.set_ylabel(f"Fraction who understand at round {a.steps}")
    ax.set_ylim(0, 1.02)
    ax.set_title("Effect of forgetting on the re-vote outcome", color=INK)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_seir_gamma_sweep.png", dpi=150)
    plt.close(fig)
    print(f"\nWrote results to {out.resolve()}")


if __name__ == "__main__":
    main()
