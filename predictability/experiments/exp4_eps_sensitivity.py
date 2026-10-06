"""Experiment 4: sensitivity of Pi^max to eps on real-world datasets.

For every dataset in the YAML config the tolerance eps is swept (default:
pair-match quantiles 1 %, 2.5 %, 5 %, 10 %, 25 %, 50 %, 75 %, plus eps_min
and eps_max, or an explicit --eps list).  For each eps we report H_est,
effective alphabet sizes (range and recurrence definitions), Pi^max and the
realized accuracy of the forecasting benchmarks (predictions are computed
once per dataset; accuracy is re-evaluated per eps), the tightness gap and
an upper-bound validity flag.

Real datasets must be available locally; nothing is downloaded.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..benchmarks import DEFAULT_MODELS, eps_accuracy, run_statsforecast
from ..datasets import Dataset, load_all
from .common import analyze_series, ensure_dir, eps_grid, finish_plot, log, parse_float_list, save_csv, write_metadata


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default="configs/datasets.yaml")
    p.add_argument("--datasets", default="", help="comma-separated subset of dataset names")
    p.add_argument("--out", default="results/exp4")
    p.add_argument("--eps", default="", help="explicit comma-separated eps values (overrides quantile levels)")
    p.add_argument("--eps-levels", default="0,0.01,0.025,0.05,0.1,0.25,0.5,0.75,1")
    p.add_argument("--models", default=",".join(DEFAULT_MODELS))
    p.add_argument("--refit", default="False", help="statsforecast refit: True, False or an integer")
    p.add_argument("--algorithm", default="ifi")
    p.add_argument("--boundary", default="lmp+", choices=["lmp+", "zero-fill"])
    p.add_argument("--variant", default="paper", choices=["paper", "fano"])
    p.add_argument("--alphabet", default="range", choices=["range", "range_padded", "recurrence", "recurrence_permuted"], help="primary alphabet-size definition")
    p.add_argument("--strict-eps", action="store_true", help="use |true-pred| < eps instead of <=")


def parse_refit(text: str):
    t = str(text).strip().lower()
    if t in ("true", "yes", "1"):
        return True
    if t in ("false", "no", "0"):
        return False
    return int(t)


def benchmark_predictions(ds: Dataset, models: Iterable[str], refit, cache_dir: Path) -> pd.DataFrame:
    """Run (or load cached) rolling one-step-ahead predictions for a dataset."""
    cache = cache_dir / f"predictions_{ds.name}.csv"
    if cache.exists():
        log.info("loading cached predictions %s", cache)
        return pd.read_csv(cache)
    preds = run_statsforecast(ds.values, ds.name, models=list(models), season_length=ds.season_length, refit=refit)
    save_csv(preds, cache)
    return preds


def sweep_dataset(ds: Dataset, eps_values: List[float], preds: pd.DataFrame, args) -> pd.DataFrame:
    rows = []
    methods = sorted({"range", "recurrence", args.alphabet})
    for eps in eps_values:
        row = analyze_series(ds.values, eps, args.algorithm, args.boundary, args.variant, methods, primary_alphabet=args.alphabet)
        row["dataset"] = ds.name
        acc = eps_accuracy(preds, eps, strict=args.strict_eps)
        best = -np.inf
        for _, r in acc.iterrows():
            row[f"pi_{r['model']}"] = r["accuracy"]
            best = max(best, r["accuracy"])
        row["max_model_pi"] = best
        row["gap"] = row["pi_max"] - best
        row["bound_holds"] = bool(row["pi_max"] + 1e-12 >= best)
        rows.append(row)
        log.info("exp4 %s eps=%.5g: H=%.4f N=%.3g pi_max=%.4f best_model=%.4f", ds.name, eps, row["H_est"], row["N"], row["pi_max"], best)
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> pd.DataFrame:
    out = ensure_dir(args.out)
    names = [d for d in args.datasets.split(",") if d] or None
    datasets = load_all(args.config, names)
    models = [m for m in args.models.split(",") if m]
    refit = parse_refit(args.refit)
    frames = []
    for ds in datasets:
        preds = benchmark_predictions(ds, models, refit, out)
        if args.eps:
            eps_values = parse_float_list(args.eps)
        else:
            eps_values = sorted(set(eps_grid(ds.values, parse_float_list(args.eps_levels)).values()))
        frames.append(sweep_dataset(ds, eps_values, preds, args))
    df = pd.concat(frames, ignore_index=True)
    save_csv(df, out / "exp4_results.csv")
    write_metadata(out, vars(args))
    plot(df, out)
    return df


def plot(df: pd.DataFrame, out: Path) -> None:
    for name, sub in df.groupby("dataset"):
        sub = sub.sort_values("eps")
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        ax = axes[0]
        ax.plot(sub.eps, sub.pi_max, "k-o", label=r"$\Pi^{max}$")
        for c in [c for c in sub.columns if c.startswith("pi_") and not c.startswith("pi_max")]:
            ax.plot(sub.eps, sub[c], "-^", label=c[3:], alpha=0.8)
        ax.set_xlabel(r"$\epsilon$")
        ax.set_ylabel("predictability")
        ax.set_ylim(0, 1.05)
        ax.set_xscale("symlog", linthresh=max(sub.eps[sub.eps > 0].min(), 1e-9) if (sub.eps > 0).any() else 1)
        ax.legend(fontsize=7)
        ax = axes[1]
        ax.plot(sub.eps, sub.H_est, "k-o")
        ax.set_xlabel(r"$\epsilon$")
        ax.set_ylabel("entropy rate [bits]")
        ax.set_xscale(axes[0].get_xscale())
        fig.suptitle(f"Experiment 4 – {name}")
        finish_plot(fig, out / f"exp4_{name}.png")
