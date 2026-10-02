"""Plotting helpers used by run_paper.py and robustness.py."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np


def visibility_from_sigma(sigma_phi: float) -> float:
    """Visibility proxy for Gaussian phase uncertainty."""
    return float(np.exp(-0.5 * sigma_phi ** 2))


def add_visibility_top_axis(ax):
    def sigma_to_vis(x):
        x = np.asarray(x)
        return np.exp(-0.5 * x ** 2)

    def vis_to_sigma(g):
        g = np.asarray(g)
        g = np.clip(g, 1e-12, 1.0)
        return np.sqrt(-2.0 * np.log(g))

    secax = ax.secondary_xaxis("top", functions=(sigma_to_vis, vis_to_sigma))
    secax.set_xlabel(r"visibility proxy $\gamma=e^{-\sigma_\phi^2/2}$")
    return secax


def plot_summary_bar(rows: List[Dict], out_path: Path, title: str = "summary") -> None:
    if not rows:
        return
    labels = []
    means, stds = [], []
    for r in rows:
        label = f"{r['architecture']}\n{r['combine']}\n{r['basis']}\n{r['representation']}"
        labels.append(label)
        means.append(float(r["best_mean"]))
        stds.append(float(r["best_std"]))
    fig, ax = plt.subplots(figsize=(max(7, len(rows) * 1.6), 4.5))
    x = np.arange(len(rows))
    ax.bar(x, means, yerr=stds, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("best accuracy")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def read_csv_rows(path: Path) -> List[Dict]:
    with path.open() as f:
        return list(csv.DictReader(f))
