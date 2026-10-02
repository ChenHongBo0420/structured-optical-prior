"""Training utilities for structured optical prior experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
import neuroptica as neu
from neuroptica.losses import CategoricalCrossEntropy

from .preprocess import (
    hadamard,
    identity_basis,
    random_unitary,
    preprocess_images_fft_center,
    preprocess_images_fft_center_power,
    preprocess_images_pca,
    preprocess_images_raw_center,
    preprocess_images_raw_first16,
    preprocess_images_raw_pool,
)
from .models import build_single_model, build_dual_model, count_params


REPRESENTATIONS = ("fft", "fft_power", "pca", "raw_pool", "raw_center", "raw_first16")
BASES = ("hadamard", "identity", "random")


def prepare_data(representation: str, N: int):
    (xtr_raw, ytr_idx), (xte_raw, yte_idx) = mnist.load_data()
    if representation == "fft":
        # Default: legacy max-normalized FFT preprocessing matching the original 0.88 runs.
        xtr = preprocess_images_fft_center(xtr_raw, N)
        xte = preprocess_images_fft_center(xte_raw, N)
    elif representation == "fft_power":
        # Optional control: fixed-total-power FFT input.
        xtr = preprocess_images_fft_center_power(xtr_raw, N)
        xte = preprocess_images_fft_center_power(xte_raw, N)
    elif representation == "pca":
        xtr, xte = preprocess_images_pca(xtr_raw, xte_raw, N)
    elif representation == "raw_pool":
        xtr = preprocess_images_raw_pool(xtr_raw, N)
        xte = preprocess_images_raw_pool(xte_raw, N)
    elif representation == "raw_center":
        xtr = preprocess_images_raw_center(xtr_raw, N)
        xte = preprocess_images_raw_center(xte_raw, N)
    elif representation == "raw_first16":
        xtr = preprocess_images_raw_first16(xtr_raw, N)
        xte = preprocess_images_raw_first16(xte_raw, N)
    elif representation == "raw":
        # Backward-compatible alias: stable raw baseline rather than legacy first-16 pixels.
        xtr = preprocess_images_raw_pool(xtr_raw, N)
        xte = preprocess_images_raw_pool(xte_raw, N)
    else:
        raise ValueError(f"Unknown representation: {representation}")
    ytr = np.eye(10)[ytr_idx]
    return xtr, ytr, xte, yte_idx


def get_basis(basis: str, N: int, seed: int = 0):
    if basis == "hadamard":
        return hadamard(N)
    if basis == "identity":
        return identity_basis(N)
    if basis == "random":
        return random_unitary(N, seed=seed)
    raise ValueError(f"Unknown basis: {basis}")


def build_model(*, architecture: str, combine: str, basis: str, N: int, out_ports: int, gsize: int, basis_seed: int):
    S = get_basis(basis, N, seed=basis_seed)
    if architecture == "single":
        return build_single_model(N=N, out_ports=out_ports, gsize=gsize, S=S)
    if architecture == "dual":
        return build_dual_model(N=N, out_ports=out_ports, gsize=gsize, S=S, combine=combine)
    raise ValueError(f"Unknown architecture: {architecture}")


def _safe_float(x):
    v = float(np.real(x))
    return v


def train_one_run(*, architecture: str, combine: str, basis: str, representation: str, N: int, gsize: int,
                  out_ports: int, lr: float, epochs: int, batch_size: int, seed: int, basis_seed: int = 0,
                  out_dir: str = "results") -> Dict:
    """Train one run and save per-seed JSON plus a training curve PNG."""
    np.random.seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xtr, ytr, xte, yte = prepare_data(representation, N)
    model = build_model(architecture=architecture, combine=combine, basis=basis, N=N,
                        out_ports=out_ports, gsize=gsize, basis_seed=basis_seed)

    opt = neu.InSituAdam(model, CategoricalCrossEntropy(), step_size=lr)
    train_losses, test_accs = [], []
    best_acc, final_acc = 0.0, 0.0
    stopped_early = False

    for epoch in range(epochs):
        loss_list = opt.fit(xtr.T, ytr.T, epochs=1, batch_size=batch_size,
                            cache_fields=False, use_partial_vectors=False)
        loss_val = _safe_float(loss_list[0])
        yp = model.forward_pass(xte.T)
        acc = float(np.mean(np.argmax(yp, axis=0) == yte))
        train_losses.append(loss_val)
        test_accs.append(acc)
        if np.isfinite(acc):
            best_acc = max(best_acc, acc)
            final_acc = acc
        if not np.isfinite(loss_val):
            stopped_early = True
            # Fill remaining entries for consistent plotting and JSON length.
            for _ in range(epoch + 1, epochs):
                train_losses.append(float("nan"))
                test_accs.append(final_acc if np.isfinite(final_acc) else 0.0)
            break

    stem = f"mnist_{representation}_{architecture}_{combine}_{basis}_seed{seed}"
    fig_path = out_dir / f"{stem}.png"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses)
    ax1.set_title("train loss")
    ax1.set_xlabel("epoch")
    ax1.grid(True, alpha=0.3)
    ax2.plot(test_accs)
    ax2.set_title("test acc")
    ax2.set_xlabel("epoch")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    payload = {
        "dataset": "mnist",
        "representation": representation,
        "architecture": architecture,
        "combine": combine,
        "basis": basis,
        "seed": seed,
        "basis_seed": basis_seed,
        "N": N,
        "gsize": gsize,
        "out_ports": out_ports,
        "lr": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "best_acc": best_acc,
        "final_acc": final_acc,
        "stopped_early": stopped_early,
        "param_stats": count_params(architecture, N, gsize),
        "train_losses": train_losses,
        "test_accs": test_accs,
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2))
    return payload


def summarize_runs(results: List[Dict], architecture: str, combine: str, basis: str, representation: str,
                   N: int, out_ports: int, gsize: int, basis_seed: int, out_dir: str = "results",
                   tag: str = "summary") -> Dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best = np.array([r["best_acc"] for r in results], dtype=float)
    final = np.array([r["final_acc"] for r in results], dtype=float)
    summary = {
        "n_runs": len(results),
        "best_acc": {
            "mean": float(np.nanmean(best)), "std": float(np.nanstd(best)),
            "min": float(np.nanmin(best)), "max": float(np.nanmax(best)),
        },
        "final_acc": {
            "mean": float(np.nanmean(final)), "std": float(np.nanstd(final)),
            "min": float(np.nanmin(final)), "max": float(np.nanmax(final)),
        },
        "param_stats_example": count_params(architecture, N, gsize),
        "model_spec": {
            "architecture": architecture,
            "combine": combine,
            "basis": basis,
            "N": N,
            "out_ports": out_ports,
            "gsize": gsize,
            "basis_seed": basis_seed,
        },
        "dataset": "mnist",
        "representation": representation,
    }
    (out_dir / f"{tag}.json").write_text(json.dumps(summary, indent=2))
    return summary


def save_summary_csv(rows: Iterable[Dict], path: Path) -> None:
    rows = list(rows)
    fieldnames = [
        "exp", "architecture", "combine", "basis", "representation",
        "best_mean", "best_std", "final_mean", "final_std",
        "mzi", "phase_shifters", "tunable_phases",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
