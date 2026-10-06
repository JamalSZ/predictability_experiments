#!/usr/bin/env bash
# Experiments that need the local real datasets configured in configs/datasets.yaml.
set -euo pipefail
cd "$(dirname "$0")/.."
CFG=${1:-configs/datasets.yaml}
python -m predictability.cli exp3 --config "$CFG" --datasets Temperature --synthetic "" --out results/exp3_real
python -m predictability.cli exp4 --config "$CFG" --out results/exp4
python -m predictability.cli exp5 --config "$CFG" --out results/exp5
