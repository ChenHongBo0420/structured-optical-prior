#!/usr/bin/env bash
set -e
python run_paper.py --exp robustness_boundary \
  --epochs 300 \
  --train_seeds 0,1,2 \
  --noise_seeds 0,1,2 \
  --phase_sigmas 0.00,0.02,0.05,0.10,0.15,0.20,0.30,0.40 \
  --noise_target spg
