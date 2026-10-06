#!/usr/bin/env bash
# Full synthetic experiments (no real data needed). n = 5000, runtime sweep up to n = 5000.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m predictability.cli exp1 --n 5000 --out results/exp1
python -m predictability.cli exp2 --n 5000 --out results/exp2
python -m predictability.cli exp3 --sizes 500,1000,2000,5000 --bf-max-n 1000 --repeats 3 --out results/exp3
python -m predictability.cli exp5 --config "" --out results/exp5_synthetic
