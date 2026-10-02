#!/usr/bin/env python3
"""
Figure 5: device-nonideality sensitivity for the structured optical prior.

Drop this file into the root directory of `structured_optical_prior_fixed/`, next to
`run_paper.py`, and run for example:

    python run_figure5_device_nonidealities.py \
        --epochs 300 \
        --train_seeds 0,1,2 \
        --noise_seeds 0,1,2 \
        --out_dir results/figure5_device_nonidealities

Fast smoke test:

    python run_figure5_device_nonidealities.py --epochs 2 --train_seeds 0 --noise_seeds 0

Outputs:
    figure5_device_nonidealities_summary.csv
    figure5_device_nonidealities_raw.csv
    figure5_device_nonidealities_summary.json
    fig5_device_nonidealities.pdf
    fig5_device_nonidealities.png

The experiment is a post-training perturbation study of the main model:
    dual + coherent + Hadamard + FFT.

It scans four hardware-oriented nonidealities:
    (a) phase-array quantization
    (b) splitter/combiner branch imbalance
    (c) differential branch loss
    (d) additive detector noise

This script intentionally does not replace the existing core experiments. It adds a compact
Figure-5-style tolerance map compatible with the existing optical_prior package.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import neuroptica as neu
from neuroptica.components import MZI, PhaseShifter
from neuroptica.layers import OpticalMeshNetworkLayer
from neuroptica.losses import CategoricalCrossEntropy

from optical_prior.train import prepare_data, build_model


# -------------------------
# Parsing utilities
# -------------------------


def parse_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


# -------------------------
# Phase component utilities
# -------------------------


def wrap_phase(x: float) -> float:
    return float((x + np.pi) % (2.0 * np.pi) - np.pi)


def collect_tunable_components(model, target: str = "spg"):
    """Collect tunable optical components.

    target='spg' selects the basis-domain phase arrays only, i.e. layer names containing
    SPG / PhaseDiag / diag. target='all' selects all tunable optical components.
    """
    comps = []
    for layer in model.layers:
        if not isinstance(layer, OpticalMeshNetworkLayer):
            continue
        name = getattr(layer, "__name__", layer.__class__.__name__).lower()
        if target == "spg":
            if ("spg" not in name) and ("phasediag" not in name) and ("diag" not in name):
                continue
        elif target == "all":
            pass
        else:
            raise ValueError(f"Unknown target: {target}")
        comps.extend(list(layer.mesh.all_tunable_components()))
    return comps


def snapshot_components(components) -> List[Tuple[object, Tuple[float, ...]]]:
    state = []
    for comp in components:
        if isinstance(comp, PhaseShifter):
            state.append((comp, (float(comp.phi),)))
        elif isinstance(comp, MZI):
            state.append((comp, (float(comp.theta), float(comp.phi))))
    return state


def restore_components(state: List[Tuple[object, Tuple[float, ...]]]) -> None:
    for comp, vals in state:
        if isinstance(comp, PhaseShifter):
            comp.phi = float(vals[0])
        elif isinstance(comp, MZI):
            comp.theta = float(vals[0])
            comp.phi = float(vals[1])


def quantize_components(components, bits: int) -> None:
    """Uniformly quantize phases to 2^bits levels over [0, 2pi)."""
    if bits <= 0:
        raise ValueError("bits must be positive")
    step = 2.0 * np.pi / (2 ** bits)

    def quantize_angle(phi: float) -> float:
        phi_mod = phi % (2.0 * np.pi)
        return wrap_phase(round(phi_mod / step) * step)

    for comp in components:
        if isinstance(comp, PhaseShifter):
            comp.phi = quantize_angle(float(comp.phi))
        elif isinstance(comp, MZI):
            comp.theta = quantize_angle(float(comp.theta))
            comp.phi = quantize_angle(float(comp.phi))


# -------------------------
# Training and evaluation
# -------------------------


def train_main_model(*, N: int, gsize: int, out_ports: int, epochs: int, batch_size: int,
                     lr: float, train_seed: int, basis_seed: int):
    """Train the main dual-coherent-Hadamard-FFT model and return model plus test set."""
    np.random.seed(train_seed)
    xtr, ytr, xte, yte = prepare_data("fft", N)
    model = build_model(
        architecture="dual",
        combine="coherent",
        basis="hadamard",
        N=N,
        out_ports=out_ports,
        gsize=gsize,
        basis_seed=basis_seed,
    )
    opt = neu.InSituAdam(model, CategoricalCrossEntropy(), step_size=lr)
    train_losses = []
    clean_test_accs = []
    for epoch in range(epochs):
        loss_list = opt.fit(xtr.T, ytr.T, epochs=1, batch_size=batch_size,
                            cache_fields=False, use_partial_vectors=False)
        loss_val = float(np.real(loss_list[0]))
        yp = model.forward_pass(xte.T)
        acc = float(np.mean(np.argmax(yp, axis=0) == yte))
        train_losses.append(loss_val)
        clean_test_accs.append(acc)
        if not np.isfinite(loss_val):
            break
    return model, xte, yte, train_losses, clean_test_accs


def branch_fields(model, X: np.ndarray):
    """Return branch fields y1,y2 for input X with shape (N, n_samples)."""
    y1 = model._forward_branch(X, model.branch1_layers, cache_fields=False, use_partial_vectors=False)
    y2 = model._forward_branch(X, model.branch2_layers, cache_fields=False, use_partial_vectors=False)
    return y1, y2


def accuracy_from_fields(y1: np.ndarray, y2: np.ndarray, y_true: np.ndarray, *, out_ports: int,
                         alpha1: float = 1.0, alpha2: float = 1.0,
                         detector_noise_rel: float = 0.0,
                         rng: np.random.RandomState | None = None,
                         clip_negative: bool = True) -> float:
    """Evaluate coherent recombination with optional amplitude/loss imbalance and detector noise.

    alpha1 and alpha2 are field-amplitude multipliers. Detector noise is additive in
    intensity and normalized to the mean detected intensity before DropMask.
    """
    y = alpha1 * y1 + alpha2 * y2
    intensity = np.abs(y) ** 2
    if detector_noise_rel > 0:
        if rng is None:
            rng = np.random.RandomState(0)
        scale = detector_noise_rel * float(np.mean(intensity))
        intensity = intensity + rng.normal(0.0, scale, size=intensity.shape)
        if clip_negative:
            intensity = np.maximum(intensity, 0.0)
    logits = intensity[:out_ports, :]
    return float(np.mean(np.argmax(logits, axis=0) == y_true))


def evaluate_clean(model, xte: np.ndarray, yte: np.ndarray) -> float:
    yp = model.forward_pass(xte.T)
    return float(np.mean(np.argmax(yp, axis=0) == yte))


# -------------------------
# Experiment scans
# -------------------------


def scan_model(model, xte: np.ndarray, yte: np.ndarray, *, out_ports: int, train_seed: int,
               phase_bits: Sequence[int], imbalance_etas: Sequence[float], loss_dbs: Sequence[float],
               detector_noises: Sequence[float], noise_seeds: Sequence[int], phase_target: str) -> List[Dict]:
    rows: List[Dict] = []
    X = xte.T

    # Baseline branch fields for scans that do not change internal phases.
    y1, y2 = branch_fields(model, X)

    # (a) phase quantization, applied to phase arrays and then evaluated.
    comps = collect_tunable_components(model, target=phase_target)
    base_state = snapshot_components(comps)
    for bits in phase_bits:
        restore_components(base_state)
        quantize_components(comps, int(bits))
        qy1, qy2 = branch_fields(model, X)
        acc = accuracy_from_fields(qy1, qy2, yte, out_ports=out_ports)
        rows.append({
            "panel": "phase_quantization",
            "perturbation": "phase_bits",
            "value": int(bits),
            "train_seed": int(train_seed),
            "noise_seed": "",
            "accuracy": acc,
        })
    restore_components(base_state)

    # (b) splitter/combiner imbalance. eta=0.5 gives alpha1=alpha2=1.
    for eta in imbalance_etas:
        eta = float(eta)
        alpha1 = np.sqrt(2.0 * eta)
        alpha2 = np.sqrt(2.0 * (1.0 - eta))
        acc = accuracy_from_fields(y1, y2, yte, out_ports=out_ports, alpha1=alpha1, alpha2=alpha2)
        rows.append({
            "panel": "branch_imbalance",
            "perturbation": "eta",
            "value": eta,
            "train_seed": int(train_seed),
            "noise_seed": "",
            "accuracy": acc,
        })

    # (c) differential loss on branch 2, in dB, relative to branch 1.
    for loss_db in loss_dbs:
        loss_db = float(loss_db)
        alpha2 = 10.0 ** (-loss_db / 20.0)  # field amplitude transmission
        acc = accuracy_from_fields(y1, y2, yte, out_ports=out_ports, alpha1=1.0, alpha2=alpha2)
        rows.append({
            "panel": "differential_loss",
            "perturbation": "loss_db",
            "value": loss_db,
            "train_seed": int(train_seed),
            "noise_seed": "",
            "accuracy": acc,
        })

    # (d) additive detector noise, normalized to mean intensity.
    for noise_rel in detector_noises:
        noise_rel = float(noise_rel)
        if noise_rel == 0:
            acc = accuracy_from_fields(y1, y2, yte, out_ports=out_ports)
            rows.append({
                "panel": "detector_noise",
                "perturbation": "sigma_I_over_meanI",
                "value": noise_rel,
                "train_seed": int(train_seed),
                "noise_seed": "",
                "accuracy": acc,
            })
        else:
            for ns in noise_seeds:
                rng = np.random.RandomState(100000 * int(train_seed) + int(ns))
                acc = accuracy_from_fields(y1, y2, yte, out_ports=out_ports,
                                           detector_noise_rel=noise_rel, rng=rng)
                rows.append({
                    "panel": "detector_noise",
                    "perturbation": "sigma_I_over_meanI",
                    "value": noise_rel,
                    "train_seed": int(train_seed),
                    "noise_seed": int(ns),
                    "accuracy": acc,
                })

    return rows


def summarize_rows(raw_rows: List[Dict]) -> List[Dict]:
    grouped: Dict[Tuple[str, str, float], List[float]] = {}
    for r in raw_rows:
        key = (r["panel"], r["perturbation"], float(r["value"]))
        grouped.setdefault(key, []).append(float(r["accuracy"]))
    out = []
    for (panel, perturbation, value), vals in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][2])):
        arr = np.asarray(vals, dtype=float)
        out.append({
            "panel": panel,
            "perturbation": perturbation,
            "value": value,
            "acc_mean": float(np.mean(arr)),
            "acc_std": float(np.std(arr)),
            "n_values": int(len(arr)),
        })
    return out


def write_csv(rows: Iterable[Dict], path: Path) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# -------------------------
# Plotting
# -------------------------


def _panel_data(summary_rows: List[Dict], panel: str):
    rows = [r for r in summary_rows if r["panel"] == panel]
    rows = sorted(rows, key=lambda r: float(r["value"]))
    x = np.asarray([float(r["value"]) for r in rows], dtype=float)
    y = np.asarray([float(r["acc_mean"]) for r in rows], dtype=float)
    e = np.asarray([float(r["acc_std"]) for r in rows], dtype=float)
    return x, y, e


def plot_figure5(summary_rows: List[Dict], out_dir: Path) -> None:
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
    })
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    axes = ax.ravel()

    panels = [
        ("phase_quantization", "Phase quantization", "Bits", None),
        ("branch_imbalance", "Branch imbalance", r"Power split $\eta$", 0.5),
        ("differential_loss", "Differential branch loss", r"Loss on branch 2 [dB]", 0.0),
        ("detector_noise", "Detector noise", r"$\sigma_I/\langle I\rangle$", 0.0),
    ]
    labels = ["a", "b", "c", "d"]
    for a, (panel, title, xlabel, refline), lab in zip(axes, panels, labels):
        x, y, e = _panel_data(summary_rows, panel)
        a.errorbar(x, y, yerr=e, marker="o", linewidth=1.2, capsize=2.5)
        if refline is not None:
            a.axvline(refline, linestyle="--", linewidth=0.8)
        a.set_title(title, pad=5)
        a.set_xlabel(xlabel)
        a.set_ylabel("Test accuracy")
        a.set_ylim(0.0, 0.95)
        a.grid(axis="y", alpha=0.25, linewidth=0.7)
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.text(-0.14, 1.10, lab, transform=a.transAxes, fontsize=10,
               fontweight="bold", ha="left", va="top")

    pdf_path = out_dir / "fig5_device_nonidealities.pdf"
    png_path = out_dir / "fig5_device_nonidealities.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=350, bbox_inches="tight")
    plt.close(fig)


# -------------------------
# Main entry
# -------------------------


def main():
    ap = argparse.ArgumentParser(description="Figure 5 device nonideality sensitivity for structured optical prior")
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--gsize", type=int, default=4)
    ap.add_argument("--out_ports", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--train_seeds", type=str, default="0,1,2")
    ap.add_argument("--noise_seeds", type=str, default="0,1,2")
    ap.add_argument("--basis_seed", type=int, default=0)
    ap.add_argument("--phase_bits", type=str, default="2,3,4,5,6,8,12")
    ap.add_argument("--imbalance_etas", type=str, default="0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70")
    ap.add_argument("--loss_dbs", type=str, default="0,1,2,3,5,8")
    ap.add_argument("--detector_noises", type=str, default="0,0.005,0.01,0.02,0.05,0.10,0.20")
    ap.add_argument("--phase_target", type=str, default="spg", choices=["spg", "all"],
                    help="Quantize SPG phase arrays only, or all tunable optical phases.")
    ap.add_argument("--out_dir", type=str, default="results/figure5_device_nonidealities")
    ap.add_argument("--plot_only", action="store_true",
                    help="Only regenerate the figure from an existing summary CSV in out_dir.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = out_dir / "figure5_device_nonidealities_summary.csv"
    raw_csv = out_dir / "figure5_device_nonidealities_raw.csv"
    json_path = out_dir / "figure5_device_nonidealities_summary.json"

    if args.plot_only:
        if not summary_csv.exists():
            raise FileNotFoundError(f"Cannot plot only: missing {summary_csv}")
        import pandas as pd
        summary_rows = pd.read_csv(summary_csv).to_dict("records")
        plot_figure5(summary_rows, out_dir)
        print(f"Saved {out_dir / 'fig5_device_nonidealities.pdf'}")
        print(f"Saved {out_dir / 'fig5_device_nonidealities.png'}")
        return

    train_seeds = parse_ints(args.train_seeds)
    noise_seeds = parse_ints(args.noise_seeds)
    phase_bits = parse_ints(args.phase_bits)
    imbalance_etas = parse_floats(args.imbalance_etas)
    loss_dbs = parse_floats(args.loss_dbs)
    detector_noises = parse_floats(args.detector_noises)

    raw_rows: List[Dict] = []
    training_curves: Dict[str, Dict] = {}

    for seed in train_seeds:
        print(f"\n=== Training main model, seed={seed} ===")
        model, xte, yte, train_losses, clean_test_accs = train_main_model(
            N=args.N,
            gsize=args.gsize,
            out_ports=args.out_ports,
            epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
            train_seed=seed,
            basis_seed=args.basis_seed,
        )
        clean_acc = evaluate_clean(model, xte, yte)
        print(f"Clean accuracy seed={seed}: {clean_acc:.4f}")
        training_curves[str(seed)] = {
            "train_losses": train_losses,
            "clean_test_accs": clean_test_accs,
            "clean_acc": clean_acc,
        }
        raw_rows.extend(scan_model(
            model, xte, yte,
            out_ports=args.out_ports,
            train_seed=seed,
            phase_bits=phase_bits,
            imbalance_etas=imbalance_etas,
            loss_dbs=loss_dbs,
            detector_noises=detector_noises,
            noise_seeds=noise_seeds,
            phase_target=args.phase_target,
        ))

    summary_rows = summarize_rows(raw_rows)
    write_csv(raw_rows, raw_csv)
    write_csv(summary_rows, summary_csv)

    payload = {
        "experiment": "figure5_device_nonidealities",
        "model": "dual_coherent_hadamard_fft",
        "N": args.N,
        "gsize": args.gsize,
        "out_ports": args.out_ports,
        "epochs": args.epochs,
        "batch_size": args.batch,
        "lr": args.lr,
        "train_seeds": train_seeds,
        "noise_seeds": noise_seeds,
        "phase_bits": phase_bits,
        "imbalance_etas": imbalance_etas,
        "loss_dbs": loss_dbs,
        "detector_noises": detector_noises,
        "phase_target": args.phase_target,
        "notes": {
            "phase_quantization": "Uniform quantization of phase-array phases over [0, 2pi).",
            "branch_imbalance": "eta is power split; alpha1=sqrt(2 eta), alpha2=sqrt(2(1-eta)), so eta=0.5 recovers y1+y2.",
            "differential_loss": "Branch 2 field amplitude is multiplied by 10^(-loss_db/20).",
            "detector_noise": "Gaussian additive intensity noise with sigma = value * mean(I), clipped to nonnegative intensity.",
        },
        "summary_rows": summary_rows,
        "training_curves": training_curves,
    }
    json_path.write_text(json.dumps(payload, indent=2))

    plot_figure5(summary_rows, out_dir)
    print(f"Saved {raw_csv}")
    print(f"Saved {summary_csv}")
    print(f"Saved {json_path}")
    print(f"Saved {out_dir / 'fig5_device_nonidealities.pdf'}")
    print(f"Saved {out_dir / 'fig5_device_nonidealities.png'}")


if __name__ == "__main__":
    main()
