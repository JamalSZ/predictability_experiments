"""Command line interface.

    python -m predictability.cli exp1 [--n 5000 ...]
    python -m predictability.cli exp2
    python -m predictability.cli exp3 --sizes 500,1000,2000,5000
    python -m predictability.cli exp4 --config configs/datasets.yaml
    python -m predictability.cli exp5 --config configs/datasets.yaml
    python -m predictability.cli benchmark --config configs/datasets.yaml --eps 0.5,1,2
    python -m predictability.cli estimate --config configs/datasets.yaml --eps 0.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .experiments import exp1_upper_bound, exp2_entropy_accuracy, exp3_runtime, exp4_eps_sensitivity, exp5_model_independence
from .experiments.common import setup_logging

EXPERIMENTS = {
    "exp1": exp1_upper_bound,
    "exp2": exp2_entropy_accuracy,
    "exp3": exp3_runtime,
    "exp4": exp4_eps_sensitivity,
    "exp5": exp5_model_independence,
}


def _add_benchmark(sub) -> None:
    p = sub.add_parser("benchmark", help="run forecasting benchmarks on local datasets, write predictions + eps-accuracy CSVs")
    p.add_argument("--config", default="configs/datasets.yaml")
    p.add_argument("--datasets", default="")
    p.add_argument("--out", default="results/benchmark")
    p.add_argument("--models", default=",".join(exp4_eps_sensitivity.DEFAULT_MODELS))
    p.add_argument("--refit", default="False")
    p.add_argument("--eps", default="", help="comma-separated eps values for the accuracy table")
    p.add_argument("--strict-eps", action="store_true")


def _add_estimate(sub) -> None:
    p = sub.add_parser("estimate", help="entropy rate, alphabet size and Pi^max for local datasets at given eps")
    p.add_argument("--config", default="configs/datasets.yaml")
    p.add_argument("--datasets", default="")
    p.add_argument("--eps", default="0", help="comma-separated eps values")
    p.add_argument("--algorithm", default="ifi")
    p.add_argument("--boundary", default="lmp+", choices=["lmp+", "zero-fill"])
    p.add_argument("--variant", default="paper", choices=["paper", "fano"])
    p.add_argument("--out", default="results/estimate")


def run_benchmark(args) -> None:
    import pandas as pd

    from .benchmarks import eps_accuracy_sweep, run_statsforecast
    from .datasets import load_all
    from .experiments.common import ensure_dir, parse_float_list, save_csv

    out = ensure_dir(args.out)
    names = [d for d in args.datasets.split(",") if d] or None
    models = [m for m in args.models.split(",") if m]
    frames = []
    for ds in load_all(args.config, names):
        preds = run_statsforecast(ds.values, ds.name, models=models, season_length=ds.season_length, refit=exp4_eps_sensitivity.parse_refit(args.refit))
        save_csv(preds, out / f"predictions_{ds.name}.csv")
        frames.append(preds)
    all_preds = pd.concat(frames, ignore_index=True)
    save_csv(all_preds, out / "predictions_all.csv")
    if args.eps:
        acc = eps_accuracy_sweep(all_preds, parse_float_list(args.eps), strict=args.strict_eps)
        save_csv(acc, out / "accuracy.csv")
        print(acc.to_string(index=False))


def run_estimate(args) -> None:
    import pandas as pd

    from .datasets import load_all
    from .experiments.common import analyze_series, ensure_dir, parse_float_list, save_csv

    out = ensure_dir(args.out)
    names = [d for d in args.datasets.split(",") if d] or None
    rows = []
    for ds in load_all(args.config, names):
        for eps in parse_float_list(args.eps):
            row = analyze_series(ds.values, eps, args.algorithm, args.boundary, args.variant)
            row["dataset"] = ds.name
            rows.append(row)
    df = pd.DataFrame(rows)
    save_csv(df, out / "estimates.csv")
    print(df.to_string(index=False))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="predictability", description="Entropy-rate based predictability experiments")
    parser.add_argument("-q", "--quiet", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, mod in EXPERIMENTS.items():
        p = sub.add_parser(name, help=(mod.__doc__ or "").strip().splitlines()[0])
        mod.add_args(p)
    _add_benchmark(sub)
    _add_estimate(sub)
    args = parser.parse_args(argv)
    setup_logging(not args.quiet)
    if args.command in EXPERIMENTS:
        EXPERIMENTS[args.command].run(args)
    elif args.command == "benchmark":
        run_benchmark(args)
    elif args.command == "estimate":
        run_estimate(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
