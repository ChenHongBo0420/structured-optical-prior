# Structured Optical Prior

Research code accompanying **Interference-enabled structured optical priors in dual-branch photonic networks**, by Hongbo Chen.

[Paper: JOSA B (2026)](https://doi.org/10.1364/JOSAB.607960) | [MIT License](LICENSE)

This repository packages the author's supplied Neuroptica implementation of structured optical classifiers. It includes single- and dual-branch models, coherent and incoherent readout, fixed-basis and input-representation ablations, phase-noise sweeps, device-nonideality studies, and plotting scripts.

The source archive's model definitions, numerical methods, training settings, and experiment logic are retained. This is a source-code release: trained models, numerical result tables, and the publisher's proof PDF are not bundled.

## Repository Layout

```text
.
|-- optical_prior/                       # Models, layers, training, preprocessing
|-- run_paper.py                         # Experiment entry point
|-- run_figure5_device_nonidealities.py   # Device sensitivity sweeps
|-- plotting/make_josa_figures.py         # Plot summary tables
|-- scripts/                            # Bash and PowerShell shortcuts
|-- results/                            # Generated outputs (ignored by Git)
|-- requirements.txt                    # Runtime dependencies
|-- CITATION.cff                        # Machine-readable citation
`-- LICENSE                             # MIT license
```

## Installation

Use a separate **Python 3.9-3.11** environment. Keras/TensorFlow provide the original MNIST loader; model training uses Neuroptica and NumPy. A GPU is not required.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_paper.py --help
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run_paper.py --help
```

Run commands from the repository root. On Windows, use `.\.venv\Scripts\python.exe` instead of `python` unless the environment is activated.

## Dataset

The code loads MNIST using `keras.datasets.mnist.load_data()`. The first training run downloads `mnist.npz`; subsequent runs use the Keras cache, normally `~/.keras/datasets/mnist.npz`. Internet access is required if the dataset is not cached. MNIST is not redistributed here.

The default input has 16 optical modes, with the first 10 output ports used for classification. Input representations include centered complex FFT, PCA, and raw average pooling; see [preprocess.py](optical_prior/preprocess.py).

## Run Experiments

Start with a two-epoch smoke run. This still uses the full MNIST dataset and checks the workflow, not converged accuracy:

```bash
python run_paper.py --exp smoke --epochs 2 --seeds 0
```

Run the supplied core suite:

```bash
python run_paper.py --exp all_core --epochs 300 --seeds 0,1,2
```

Individual experiments:

```bash
python run_paper.py --exp main_anchor --epochs 300 --seeds 0,1,2
python run_paper.py --exp single_vs_dual --epochs 300 --seeds 0,1,2
python run_paper.py --exp coherent_vs_incoherent --epochs 300 --seeds 0,1,2
python run_paper.py --exp basis_ablation --epochs 300 --seeds 0,1,2
python run_paper.py --exp repr_ablation --epochs 300 --seeds 0,1,2
python run_paper.py --exp param_table
```

Phase-noise sweep:

```bash
python run_paper.py --exp robustness_boundary --epochs 300 --train_seeds 0,1,2 --noise_seeds 0,1,2 --phase_sigmas 0.00,0.02,0.05,0.10,0.15,0.20,0.30,0.40 --noise_target spg
```

`--exp all_full` runs the core suite followed by the phase-noise sweep. Run the device study separately:

```bash
python run_figure5_device_nonidealities.py --epochs 300 --train_seeds 0,1,2 --noise_seeds 0,1,2 --out_dir results/figure5_device_nonidealities
```

The device script scans phase quantization, branch power imbalance, differential branch loss, and additive detector noise. Use `--help` for all options. Use a separate `--out_dir` for each configuration to avoid overwriting earlier runs.

## Outputs and Plotting

Training writes per-run JSON files and PNG curves, plus aggregate CSV/JSON summaries. In this source version, `best_acc` means maximum test accuracy over epochs, and `final_acc` means last test accuracy.

After running the core suite and phase-noise sweep:

```bash
python plotting/make_josa_figures.py --results_dir results --out_dir results/figures_josa
```

Expected inputs:

```text
results/single_vs_dual_summary_all.csv
results/coherent_vs_incoherent_summary_all.csv
results/basis_ablation_summary_all.csv
results/repr_ablation_summary_all.csv
results/robustness_boundary/robustness_boundary_summary.csv
```

To redraw an existing device-study summary without retraining:

```bash
python run_figure5_device_nonidealities.py --plot_only --out_dir results/figure5_device_nonidealities
```

Figure filenames retain the numbering of the supplied code archive; use the experiment and panel labels when matching outputs to the article. Generated outputs and downloaded datasets are excluded from Git by default.

## Citation

Please cite the associated article when using this research code:

```bibtex
@article{chen2026structuredopticalpriors,
  author  = {Chen, Hongbo},
  title   = {Interference-enabled structured optical priors in dual-branch photonic networks},
  journal = {Journal of the Optical Society of America B},
  year    = {2026},
  volume  = {43},
  number  = {10},
  doi     = {10.1364/JOSAB.607960}
}
```

## License

The code is released under the [MIT License](LICENSE). Third-party libraries and datasets retain their own licenses. The software license does not cover the associated article.
