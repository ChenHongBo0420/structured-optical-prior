#!/usr/bin/env python3
"""Master runner for the structured optical prior paper experiments.

Examples
--------
Fast smoke test:
    python run_paper.py --exp smoke --epochs 2 --seeds 0

Core paper experiments:
    python run_paper.py --exp all_core --epochs 300 --seeds 0,1,2

Boundary experiment:
    python run_paper.py --exp robustness_boundary --epochs 300 --train_seeds 0,1,2 --noise_seeds 0,1,2
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

from optical_prior.models import count_params
from optical_prior.plotting import plot_summary_bar
from optical_prior.robustness import run_robustness_boundary, parse_floats, parse_ints
from optical_prior.train import save_summary_csv, summarize_runs, train_one_run


def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_seeds(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def run_block(*, architecture: str, combine: str, basis: str, representation: str, args, name: str) -> Dict:
    out_dir = Path(args.out_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in parse_seeds(args.seeds):
        r = train_one_run(
            architecture=architecture,
            combine=combine,
            basis=basis,
            representation=representation,
            N=args.N,
            gsize=args.gsize,
            out_ports=args.out_ports,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch,
            seed=seed,
            basis_seed=args.basis_seed,
            out_dir=str(out_dir),
        )
        results.append(r)
    return summarize_runs(
        results,
        architecture=architecture,
        combine=combine,
        basis=basis,
        representation=representation,
        N=args.N,
        out_ports=args.out_ports,
        gsize=args.gsize,
        basis_seed=args.basis_seed,
        out_dir=str(out_dir),
        tag="summary",
    )


def summary_to_row(exp: str, architecture: str, combine: str, basis: str, representation: str, summary: Dict) -> Dict:
    ps = summary.get("param_stats_example", count_params(architecture, summary["model_spec"]["N"], summary["model_spec"]["gsize"]))
    return {
        "exp": exp,
        "architecture": architecture,
        "combine": combine,
        "basis": basis,
        "representation": representation,
        "best_mean": summary["best_acc"]["mean"],
        "best_std": summary["best_acc"]["std"],
        "final_mean": summary["final_acc"]["mean"],
        "final_std": summary["final_acc"]["std"],
        "mzi": ps["mzi"],
        "phase_shifters": ps["phase_shifters"],
        "tunable_phases": ps["tunable_phases"],
    }


def save_and_plot_rows(rows: List[Dict], args, exp_name: str) -> None:
    out_dir = Path(args.out_dir)
    save_summary_csv(rows, out_dir / f"{exp_name}_summary_all.csv")
    plot_summary_bar(rows, out_dir / f"{exp_name}_summary_bar.png", title=exp_name)
    for row in rows:
        print(row)


def exp_main_anchor(args) -> List[Dict]:
    s = run_block(architecture="dual", combine="coherent", basis="hadamard", representation="fft", args=args,
                  name="main_anchor")
    return [summary_to_row("main_anchor", "dual", "coherent", "hadamard", "fft", s)]


def exp_single_vs_dual(args) -> List[Dict]:
    rows = []
    for arch in ["single", "dual"]:
        s = run_block(architecture=arch, combine="coherent", basis="hadamard", representation="fft", args=args,
                      name=f"single_vs_dual_{arch}")
        rows.append(summary_to_row("single_vs_dual", arch, "coherent", "hadamard", "fft", s))
    return rows


def exp_coherent_vs_incoherent(args) -> List[Dict]:
    rows = []
    for combine in ["coherent", "incoherent"]:
        s = run_block(architecture="dual", combine=combine, basis="hadamard", representation="fft", args=args,
                      name=f"coherent_vs_incoherent_{combine}")
        rows.append(summary_to_row("coherent_vs_incoherent", "dual", combine, "hadamard", "fft", s))
    return rows


def exp_basis_ablation(args) -> List[Dict]:
    rows = []
    for basis in parse_csv_list(args.bases):
        s = run_block(architecture="dual", combine="coherent", basis=basis, representation="fft", args=args,
                      name=f"basis_ablation_{basis}")
        rows.append(summary_to_row("basis_ablation", "dual", "coherent", basis, "fft", s))
    return rows


def exp_repr_ablation(args) -> List[Dict]:
    rows = []
    for representation in parse_csv_list(args.reprs):
        s = run_block(architecture="dual", combine="coherent", basis="hadamard", representation=representation,
                      args=args, name=f"repr_ablation_{representation}")
        rows.append(summary_to_row("repr_ablation", "dual", "coherent", "hadamard", representation, s))
    return rows


def exp_param_table(args) -> List[Dict]:
    rows = []
    configs = [
        ("single", "coherent", "hadamard", "fft"),
        ("dual", "coherent", "hadamard", "fft"),
        ("dual", "incoherent", "hadamard", "fft"),
        ("dual", "coherent", "identity", "fft"),
        ("dual", "coherent", "random", "fft"),
        ("dual", "coherent", "hadamard", "pca"),
        ("dual", "coherent", "hadamard", "raw_pool"),
    ]
    for architecture, combine, basis, representation in configs:
        ps = count_params(architecture, args.N, args.gsize)
        rows.append({
            "condition": f"{architecture}_{combine}_{basis}_{representation}",
            "architecture": architecture,
            "combine": combine,
            "basis": basis,
            "representation": representation,
            "mzi": ps["mzi"],
            "phase_shifters": ps["phase_shifters"],
            "tunable_phases": ps["tunable_phases"],
        })
    out_path = Path(args.out_dir) / "param_table.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {out_path}")
    for row in rows:
        print(row)
    return []


def exp_robustness_boundary(args) -> None:
    payload = run_robustness_boundary(
        N=args.N,
        gsize=args.gsize,
        out_ports=args.out_ports,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        train_seeds=parse_ints(args.train_seeds),
        noise_seeds=parse_ints(args.noise_seeds),
        phase_sigmas=parse_floats(args.phase_sigmas),
        basis_seed=args.basis_seed,
        out_dir=str(Path(args.out_dir) / "robustness_boundary"),
        noise_target=args.noise_target,
    )
    print(json.dumps({k: payload[k] for k in payload if k != "rows"}, indent=2))


def run_core(args) -> List[Dict]:
    all_rows: List[Dict] = []
    for name, fn in [
        ("main_anchor", exp_main_anchor),
        ("single_vs_dual", exp_single_vs_dual),
        ("coherent_vs_incoherent", exp_coherent_vs_incoherent),
        ("basis_ablation", exp_basis_ablation),
        ("repr_ablation", exp_repr_ablation),
    ]:
        print(f"\n=== Running {name} ===")
        rows = fn(args)
        save_and_plot_rows(rows, args, name)
        all_rows.extend(rows)
    save_and_plot_rows(all_rows, args, "all_core")
    exp_param_table(args)
    return all_rows


def main():
    ap = argparse.ArgumentParser(description="Structured Optical Prior paper experiment runner")
    ap.add_argument("--exp", type=str, required=True, choices=[
        "smoke", "main_anchor", "single_vs_dual", "coherent_vs_incoherent", "basis_ablation",
        "repr_ablation", "param_table", "robustness_boundary", "all_core", "all_full",
    ])
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--gsize", type=int, default=4)
    ap.add_argument("--out_ports", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--basis_seed", type=int, default=0)
    ap.add_argument("--bases", type=str, default="hadamard,identity,random")
    ap.add_argument("--reprs", type=str, default="fft,pca,raw_pool")
    ap.add_argument("--phase_sigmas", type=str, default="0.00,0.02,0.05,0.10,0.15,0.20,0.30,0.40")
    ap.add_argument("--train_seeds", type=str, default="0,1,2")
    ap.add_argument("--noise_seeds", type=str, default="0,1,2")
    ap.add_argument("--noise_target", type=str, default="spg", choices=["spg", "all"])
    ap.add_argument("--out_dir", type=str, default="results")
    args = ap.parse_args()

    if args.exp == "smoke":
        args.epochs = min(args.epochs, 2)
        args.seeds = args.seeds or "0"
        rows = exp_main_anchor(args)
        save_and_plot_rows(rows, args, "smoke")
    elif args.exp == "main_anchor":
        save_and_plot_rows(exp_main_anchor(args), args, "main_anchor")
    elif args.exp == "single_vs_dual":
        save_and_plot_rows(exp_single_vs_dual(args), args, "single_vs_dual")
    elif args.exp == "coherent_vs_incoherent":
        save_and_plot_rows(exp_coherent_vs_incoherent(args), args, "coherent_vs_incoherent")
    elif args.exp == "basis_ablation":
        save_and_plot_rows(exp_basis_ablation(args), args, "basis_ablation")
    elif args.exp == "repr_ablation":
        save_and_plot_rows(exp_repr_ablation(args), args, "repr_ablation")
    elif args.exp == "param_table":
        exp_param_table(args)
    elif args.exp == "robustness_boundary":
        exp_robustness_boundary(args)
    elif args.exp == "all_core":
        run_core(args)
    elif args.exp == "all_full":
        run_core(args)
        print("\n=== Running robustness_boundary ===")
        exp_robustness_boundary(args)


if __name__ == "__main__":
    main()
