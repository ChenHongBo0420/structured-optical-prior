#!/usr/bin/env python3
"""
Generate JOSA-B-style Fig. 2-4 PDFs from summary CSV files.

Example:
    python make_josa_figures.py --results_dir results --out_dir figures_josa

Required inputs:
    single_vs_dual_summary_all.csv
    coherent_vs_incoherent_summary_all.csv
    basis_ablation_summary_all.csv
    repr_ablation_summary_all.csv
    robustness_boundary/robustness_boundary_summary.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.8,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
})


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required CSV not found: {path}")
    return pd.read_csv(path)


def pick_row(df: pd.DataFrame, **conditions) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col, val in conditions.items():
        mask &= df[col].astype(str).str.lower().eq(str(val).lower())
    rows = df[mask]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one row for {conditions}, found {len(rows)}")
    return rows.iloc[0]


def setup_axis(ax: plt.Axes, ylabel: str | None = "Best test accuracy") -> None:
    ax.set_ylim(0.0, 0.95)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontsize=10,
            fontweight="bold", ha="left", va="top")


def point_panel(ax: plt.Axes, labels: Sequence[str], means: Sequence[float],
                stds: Sequence[float], title: str, ylabel: str | None) -> None:
    x = np.arange(len(labels), dtype=float)
    ax.errorbar(x, means, yerr=stds, fmt="o", markersize=4.8,
                capsize=2.5, linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(title, pad=5)
    ax.set_xlim(-0.45, len(labels)-0.55)
    setup_axis(ax, ylabel)


def save(fig: plt.Figure, out: Path) -> None:
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=350, bbox_inches="tight")
    plt.close(fig)


def make_fig2(results_dir: Path, out_dir: Path) -> Path:
    sd = read_csv(results_dir / "single_vs_dual_summary_all.csv")
    ci = read_csv(results_dir / "coherent_vs_incoherent_summary_all.csv")
    r_single = pick_row(sd, architecture="single")
    r_dual = pick_row(sd, architecture="dual")
    r_coh = pick_row(ci, combine="coherent")
    r_inc = pick_row(ci, combine="incoherent")

    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.65), constrained_layout=True, sharey=True)
    point_panel(ax[0], ["Single", "Dual"],
                [r_single.best_mean, r_dual.best_mean],
                [r_single.best_std, r_dual.best_std],
                "Branch parallelism", "Best test accuracy")
    add_panel_label(ax[0], "a")
    gap = float(r_dual.best_mean - r_single.best_mean)
    ax[0].text(0.5, 0.91, f"+{gap:.3f}", ha="center", va="center", fontsize=8)
    ax[0].plot([0, 1], [0.875, 0.875], linewidth=0.8, color="black")

    point_panel(ax[1], ["Coherent", "Incoherent"],
                [r_coh.best_mean, r_inc.best_mean],
                [r_coh.best_std, r_inc.best_std],
                "Pre-detection interaction", None)
    add_panel_label(ax[1], "b")
    ax[1].text(0.5, 0.36, "cross term\nremoved", ha="center", va="center", fontsize=8)
    ax[1].annotate("", xy=(1, float(r_inc.best_mean)+0.02), xytext=(0.55, 0.30),
                   arrowprops=dict(arrowstyle="->", linewidth=0.8, color="black"))
    out = out_dir / "fig2_mechanism_ablation.pdf"
    save(fig, out)
    return out


def make_fig3(results_dir: Path, out_dir: Path) -> Path:
    basis = read_csv(results_dir / "basis_ablation_summary_all.csv")
    rep = read_csv(results_dir / "repr_ablation_summary_all.csv")
    basis_order = ["identity", "hadamard", "random"]
    rep_order = ["raw_pool", "pca", "fft"]
    b_rows = [pick_row(basis, basis=b) for b in basis_order]
    r_rows = [pick_row(rep, representation=r) for r in rep_order]

    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.65), constrained_layout=True, sharey=True)
    point_panel(ax[0], ["Identity", "Hadamard", "Random"],
                [float(r.best_mean) for r in b_rows],
                [float(r.best_std) for r in b_rows],
                "Global basis mixing", "Best test accuracy")
    add_panel_label(ax[0], "a")
    # Bracket under Hadamard and Random to avoid crowding with the panel title.
    yb = 0.855
    ax[0].plot([0.9, 2.1], [yb, yb], linewidth=0.8, color="black")
    ax[0].plot([0.9, 0.9], [yb, yb+0.018], linewidth=0.8, color="black")
    ax[0].plot([2.1, 2.1], [yb, yb+0.018], linewidth=0.8, color="black")
    ax[0].text(1.5, yb-0.012, "nontrivial bases", ha="center", va="top", fontsize=7.8)

    point_panel(ax[1], ["Raw-pool", "PCA", "FFT"],
                [float(r.best_mean) for r in r_rows],
                [float(r.best_std) for r in r_rows],
                "Input representation", None)
    add_panel_label(ax[1], "b")
    # Arrow and text placed below the data to avoid overlap.
    ax[1].annotate("", xy=(2.0, 0.61), xytext=(0.0, 0.61),
                   arrowprops=dict(arrowstyle="->", linewidth=0.8, color="black"))
    ax[1].text(1.0, 0.635, "increasing spectral/modal alignment",
               ha="center", va="bottom", fontsize=7.8)

    out = out_dir / "fig3_prior_ablation.pdf"
    save(fig, out)
    return out


def sigma_to_gamma(sigma):
    return np.exp(-0.5*np.asarray(sigma)**2)


def gamma_to_sigma(gamma):
    gamma = np.clip(np.asarray(gamma), 1e-9, 1.0)
    return np.sqrt(-2*np.log(gamma))


def make_fig4(results_dir: Path, out_dir: Path) -> Path:
    rb = read_csv(results_dir / "robustness_boundary" / "robustness_boundary_summary.csv")
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True)

    model_order = ["hadamard_coherent", "identity_coherent", "hadamard_incoherent"]
    labels = {
        "hadamard_coherent": "Hadamard, coherent",
        "identity_coherent": "Identity, coherent",
        "hadamard_incoherent": "Hadamard, incoherent",
    }
    markers = {"hadamard_coherent":"o", "identity_coherent":"s", "hadamard_incoherent":"^"}
    for name in model_order:
        sub = rb[rb.model_name == name].sort_values("sigma_phi")
        ax[0].errorbar(sub.sigma_phi, sub.acc_mean, yerr=sub.acc_std,
                       marker=markers[name], markersize=4.0, capsize=2.3,
                       linewidth=1.2, label=labels[name])
    ax[0].set_title("Operating boundary", pad=5)
    ax[0].set_xlabel(r"Phase noise $\sigma_\phi$ [rad]")
    ax[0].set_ylabel("Test accuracy")
    ax[0].set_ylim(0.05, 0.94)
    ax[0].grid(alpha=0.25, linewidth=0.7)
    ax[0].spines["right"].set_visible(False)
    ax[0].legend(loc="upper right", frameon=True)
    sec0 = ax[0].secondary_xaxis("top", functions=(sigma_to_gamma, gamma_to_sigma))
    sec0.set_xlabel(r"Visibility proxy $\gamma=\exp(-\sigma_\phi^2/2)$")
    sec0.set_xticks([1.00, 0.98, 0.96, 0.94, 0.92])
    add_panel_label(ax[0], "a")

    pivot = rb.pivot(index="sigma_phi", columns="model_name", values="acc_mean").sort_index()
    gap_id = pivot["hadamard_coherent"] - pivot["identity_coherent"]
    gap_inc = pivot["hadamard_coherent"] - pivot["hadamard_incoherent"]
    x = pivot.index.values
    ax[1].plot(x, gap_id.values, marker="s", markersize=4.0, linewidth=1.2, label="over Identity")
    ax[1].plot(x, gap_inc.values, marker="^", markersize=4.0, linewidth=1.2, label="over Incoherent")
    ax[1].set_title("Erosion of coherent advantage", pad=5)
    ax[1].set_xlabel(r"Phase noise $\sigma_\phi$ [rad]")
    ax[1].set_ylabel("Accuracy advantage")
    ax[1].set_ylim(0.20, 0.82)
    ax[1].grid(alpha=0.25, linewidth=0.7)
    ax[1].spines["right"].set_visible(False)
    ax[1].legend(loc="upper right", frameon=True)
    sec1 = ax[1].secondary_xaxis("top", functions=(sigma_to_gamma, gamma_to_sigma))
    sec1.set_xlabel(r"Visibility proxy $\gamma=\exp(-\sigma_\phi^2/2)$")
    sec1.set_xticks([1.00, 0.98, 0.96, 0.94, 0.92])
    add_panel_label(ax[1], "b")

    out = out_dir / "fig4_coherence_boundary.pdf"
    save(fig, out)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=Path, default=Path("results"))
    p.add_argument("--out_dir", type=Path, default=Path("figures_josa"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    outs = [make_fig2(args.results_dir, args.out_dir),
            make_fig3(args.results_dir, args.out_dir),
            make_fig4(args.results_dir, args.out_dir)]
    for out in outs:
        print(out)
        print(out.with_suffix('.png'))

if __name__ == "__main__":
    main()
