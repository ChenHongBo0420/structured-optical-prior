"""Coherence/phase-noise boundary experiment."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import neuroptica as neu
from neuroptica.components import MZI, PhaseShifter
from neuroptica.layers import OpticalMeshNetworkLayer
from neuroptica.losses import CategoricalCrossEntropy

from .train import prepare_data, get_basis
from .models import build_dual_model
from .plotting import add_visibility_top_axis, visibility_from_sigma


def parse_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _wrap_phase(x: float) -> float:
    return float((x + np.pi) % (2.0 * np.pi) - np.pi)


def _collect_noisy_components(model, target: str = "spg"):
    comps = []
    for layer in model.layers:
        if not isinstance(layer, OpticalMeshNetworkLayer):
            continue
        nm = getattr(layer, "__name__", layer.__class__.__name__).lower()
        if target == "spg":
            if ("spg" not in nm) and ("mdcphase" not in nm) and ("diag" not in nm):
                continue
        elif target == "all":
            pass
        else:
            raise ValueError(f"Unknown noise target: {target}")
        comps.extend(list(layer.mesh.all_tunable_components()))
    return comps


def _apply_phase_noise_once(components, sigma_phi: float, rng: np.random.RandomState):
    eps_map = {}
    if sigma_phi <= 0:
        return eps_map
    for comp in components:
        if isinstance(comp, PhaseShifter):
            e = np.array([rng.normal(0.0, sigma_phi)], dtype=np.float64)
            comp.phi = _wrap_phase(comp.phi + float(e[0]))
            eps_map[comp] = e
        elif isinstance(comp, MZI):
            e = np.array([rng.normal(0.0, sigma_phi), rng.normal(0.0, sigma_phi)], dtype=np.float64)
            comp.theta = _wrap_phase(comp.theta + float(e[0]))
            comp.phi = _wrap_phase(comp.phi + float(e[1]))
            eps_map[comp] = e
    return eps_map


def _remove_phase_noise(eps_map):
    for comp, e in eps_map.items():
        if isinstance(comp, PhaseShifter):
            comp.phi = _wrap_phase(comp.phi - float(e[0]))
        elif isinstance(comp, MZI):
            comp.theta = _wrap_phase(comp.theta - float(e[0]))
            comp.phi = _wrap_phase(comp.phi - float(e[1]))


def _train_model(*, representation: str, basis: str, combine: str, N: int, gsize: int, out_ports: int,
                 lr: float, epochs: int, batch_size: int, seed: int, basis_seed: int):
    np.random.seed(seed)
    xtr, ytr, xte, yte = prepare_data(representation, N)
    S = get_basis(basis, N, seed=basis_seed)
    model = build_dual_model(N=N, out_ports=out_ports, gsize=gsize, S=S, combine=combine)
    opt = neu.InSituAdam(model, CategoricalCrossEntropy(), step_size=lr)
    train_losses, clean_test_accs = [], []
    for _ in range(epochs):
        loss_list = opt.fit(xtr.T, ytr.T, epochs=1, batch_size=batch_size,
                            cache_fields=False, use_partial_vectors=False)
        loss_val = float(np.real(loss_list[0]))
        yp = model.forward_pass(xte.T)
        acc = float(np.mean(np.argmax(yp, axis=0) == yte))
        train_losses.append(loss_val)
        clean_test_accs.append(acc)
    return model, xte, yte, train_losses, clean_test_accs


def _mc_accuracy(model, xte, yte, sigma_phi: float, mc_samples: int, noise_seed: int, noise_target: str):
    if sigma_phi <= 0:
        yp = model.forward_pass(xte.T)
        return float(np.mean(np.argmax(yp, axis=0) == yte))
    comps = _collect_noisy_components(model, target=noise_target)
    rng = np.random.RandomState(noise_seed)
    accs = []
    for _ in range(mc_samples):
        eps_map = _apply_phase_noise_once(comps, sigma_phi=sigma_phi, rng=rng)
        yp = model.forward_pass(xte.T)
        accs.append(float(np.mean(np.argmax(yp, axis=0) == yte)))
        _remove_phase_noise(eps_map)
    return float(np.mean(accs))


def _plot_training_curves(curves: Dict[str, Dict[str, Sequence[float]]], out_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for name, payload in curves.items():
        ax1.plot(payload["train_losses"], label=name)
        ax2.plot(payload["clean_test_accs"], label=name)
    ax1.set_title("train loss")
    ax2.set_title("clean test acc")
    ax1.set_xlabel("epoch")
    ax2.set_xlabel("epoch")
    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_boundary(rows: List[dict], out_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.2))
    grouped = {}
    for r in rows:
        grouped.setdefault(r["model_name"], []).append(r)

    markers = {"hadamard_coherent": "o", "identity_coherent": "s", "hadamard_incoherent": "^"}
    for model_name, rs in grouped.items():
        rs = sorted(rs, key=lambda z: z["sigma_phi"])
        xs = [z["sigma_phi"] for z in rs]
        ys = [z["acc_mean"] for z in rs]
        ye = [z["acc_std"] for z in rs]
        ax1.errorbar(xs, ys, yerr=ye, marker=markers.get(model_name, "o"), linewidth=1.5,
                     capsize=3, label=model_name)
    ax1.set_title("accuracy vs phase noise")
    ax1.set_xlabel(r"$\sigma_\phi$ [rad]")
    ax1.set_ylabel("test accuracy")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    add_visibility_top_axis(ax1)

    ref_name = "hadamard_coherent"
    if ref_name in grouped:
        ref = {r["sigma_phi"]: r["acc_mean"] for r in grouped[ref_name]}
        for model_name, rs in grouped.items():
            if model_name == ref_name:
                continue
            rs = sorted(rs, key=lambda z: z["sigma_phi"])
            xs = [z["sigma_phi"] for z in rs]
            ys = [ref[z["sigma_phi"]] - z["acc_mean"] for z in rs]
            ax2.plot(xs, ys, marker=markers.get(model_name, "o"), linewidth=1.5,
                     label=f"gain over {model_name}")
    ax2.set_title("relative gain of hadamard_coherent")
    ax2.set_xlabel(r"$\sigma_\phi$ [rad]")
    ax2.set_ylabel("accuracy gap")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    add_visibility_top_axis(ax2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_robustness_boundary(*, N: int, gsize: int, out_ports: int, epochs: int, batch_size: int, lr: float,
                            train_seeds: Sequence[int], noise_seeds: Sequence[int], phase_sigmas: Sequence[float],
                            basis_seed: int, out_dir: str, noise_target: str = "spg"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    configs = [
        {"model_name": "hadamard_coherent", "basis": "hadamard", "combine": "coherent"},
        {"model_name": "identity_coherent", "basis": "identity", "combine": "coherent"},
        {"model_name": "hadamard_incoherent", "basis": "hadamard", "combine": "incoherent"},
    ]
    all_rows = []
    training_curves = {}
    for cfg in configs:
        per_train_seed_records = []
        for train_seed in train_seeds:
            model, xte, yte, train_losses, clean_test_accs = _train_model(
                representation="fft", basis=cfg["basis"], combine=cfg["combine"], N=N, gsize=gsize,
                out_ports=out_ports, lr=lr, epochs=epochs, batch_size=batch_size, seed=train_seed,
                basis_seed=basis_seed,
            )
            if train_seed == train_seeds[0]:
                training_curves[cfg["model_name"]] = {"train_losses": train_losses, "clean_test_accs": clean_test_accs}
            sigma_to_accs = {}
            for sigma in phase_sigmas:
                accs = []
                for noise_seed in noise_seeds:
                    acc = _mc_accuracy(model, xte, yte, sigma_phi=float(sigma), mc_samples=1,
                                       noise_seed=(1000 * train_seed + noise_seed), noise_target=noise_target)
                    accs.append(acc)
                sigma_to_accs[float(sigma)] = accs
            per_train_seed_records.append({"train_seed": train_seed, "sigma_to_accs": sigma_to_accs})

        for sigma in phase_sigmas:
            values = []
            for rec in per_train_seed_records:
                values.extend(rec["sigma_to_accs"][float(sigma)])
            all_rows.append({
                "model_name": cfg["model_name"],
                "basis": cfg["basis"],
                "combine": cfg["combine"],
                "representation": "fft",
                "sigma_phi": float(sigma),
                "visibility_proxy": visibility_from_sigma(float(sigma)),
                "acc_mean": float(np.mean(values)),
                "acc_std": float(np.std(values)),
                "n_values": len(values),
            })

    csv_path = out_dir / "robustness_boundary_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model_name", "basis", "combine", "representation", "sigma_phi",
            "visibility_proxy", "acc_mean", "acc_std", "n_values",
        ])
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    payload = {
        "experiment": "robustness_boundary",
        "N": N,
        "gsize": gsize,
        "out_ports": out_ports,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "train_seeds": list(train_seeds),
        "noise_seeds": list(noise_seeds),
        "phase_sigmas": [float(x) for x in phase_sigmas],
        "noise_target": noise_target,
        "visibility_model": "gamma = exp(-sigma_phi^2/2)",
        "rows": all_rows,
    }
    (out_dir / "robustness_boundary_summary.json").write_text(json.dumps(payload, indent=2))
    _plot_training_curves(training_curves, out_dir / "robustness_training_curves.png")
    _plot_boundary(all_rows, out_dir / "robustness_boundary.png")
    return payload
