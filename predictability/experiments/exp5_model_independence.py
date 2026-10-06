"""Experiment 5: Pi^max as a model-independent characterisation + discretisation ablation.

eps is fixed per dataset at a relative level: eps = --eps-rel * std(x)
(default 0.25, Morse & Patel's quarter standard deviation of the normalised
series).  For real datasets (YAML config) and a set of synthetic chains we
compute Pi^max (tolerance based) and the realized accuracy of the benchmark
models, draw the scatter Pi^max vs Pi_model, and report Spearman rank
correlations and the gaps.

Ablation: the discretisation-based bound of Song et al. -- bin the series
with bin width eps, compute the symbolic Kontoyiannis estimate on the bin
indices (eps = 0), take N = number of occupied bins and invert the classical
Fano inequality (variant 'fano').  Accuracy of models is still measured with
tolerance eps on the raw values.  Datasets where the discretised bound lies
below a realized model accuracy are counted as failures of that bound.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..benchmarks import DEFAULT_MODELS, eps_accuracy, run_fomm, run_statsforecast
from ..datasets import Dataset, load_all
from ..estimators import estimate_entropy_rate
from ..predictability import pi_max_detailed
from ..synthetic import generate
from .common import analyze_series, ensure_dir, finish_plot, log, save_csv, write_metadata
from .exp4_eps_sensitivity import parse_refit

DEFAULT_SYNTHETIC = "N5Du,N10Du,N25Du,N50Du,N10Ds,N25Ds,N10D1,N10D3,N10D5"


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=None, help="datasets.yaml with real datasets (optional)")
    p.add_argument("--datasets", default="", help="comma-separated subset of real dataset names")
    p.add_argument("--synthetic", default=DEFAULT_SYNTHETIC, help="synthetic chains to include ('' to skip)")
    p.add_argument("--n", type=int, default=5000, help="length of synthetic series")
    p.add_argument("--out", default="results/exp5")
    p.add_argument("--eps-rel", type=float, default=0.25, help="eps as a fraction of the series' standard deviation")
    p.add_argument("--models", default=",".join(DEFAULT_MODELS))
    p.add_argument("--refit", default="False")
    p.add_argument("--algorithm", default="ifi")
    p.add_argument("--boundary", default="lmp+", choices=["lmp+", "zero-fill"])
    p.add_argument("--variant", default="paper", choices=["paper", "fano"])
    p.add_argument("--alphabet", default="range", choices=["range", "range_padded", "recurrence", "recurrence_permuted"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--strict-eps", action="store_true")


def discretized_pi_max(x: np.ndarray, eps: float, boundary: str) -> dict:
    """Song et al.-style bound after binning with width eps (classical Fano)."""
    x = np.asarray(x, dtype=float)
    if eps <= 0:
        bins = np.unique(x, return_inverse=True)[1]
    else:
        bins = np.floor((x - x.min()) / eps).astype(np.int64)
    N_bins = int(np.unique(bins).size)
    H = estimate_entropy_rate(bins.astype(float), 0.0, "ifi", boundary)
    pm = pi_max_detailed(H, N_bins, "fano")
    return {"H_discretized": H, "N_bins": N_bins, "pi_max_discretized": pm.pi_max, "pi_max_discretized_status": pm.status}


def _collect(args) -> List[Dataset]:
    data: List[Dataset] = []
    if args.config:
        names = [d for d in args.datasets.split(",") if d] or None
        data += load_all(args.config, names)
    for name in [s for s in args.synthetic.split(",") if s]:
        chain, x, _ = generate(name, min(args.n, 5000), seed=args.seed)
        ds = Dataset(name, x, None, 1)
        ds.chain = chain  # type: ignore[attr-defined]
        data.append(ds)
    return data


def run(args: argparse.Namespace) -> pd.DataFrame:
    out = ensure_dir(args.out)
    models = [m for m in args.models.split(",") if m]
    refit = parse_refit(args.refit)
    methods = sorted({"range", "recurrence", args.alphabet})
    rows, long_rows = [], []
    for ds in _collect(args):
        eps = args.eps_rel * float(np.std(ds.values))
        row = analyze_series(ds.values, eps, args.algorithm, args.boundary, args.variant, methods, primary_alphabet=args.alphabet)
        row["dataset"] = ds.name
        row["synthetic"] = hasattr(ds, "chain")
        row.update(discretized_pi_max(ds.values, eps, args.boundary))
        preds = run_statsforecast(ds.values, ds.name, models=models, season_length=ds.season_length, refit=refit)
        if hasattr(ds, "chain"):
            preds = pd.concat([preds, run_fomm(ds.values, ds.name, eps=eps)], ignore_index=True)
            row["pi_exact"] = ds.chain.optimal_predictability(eps)  # type: ignore[attr-defined]
        save_csv(preds, out / f"predictions_{ds.name}.csv")
        acc = eps_accuracy(preds, eps, strict=args.strict_eps)
        best = -np.inf
        for _, r in acc.iterrows():
            row[f"pi_{r['model']}"] = r["accuracy"]
            best = max(best, r["accuracy"])
            long_rows.append({"dataset": ds.name, "model": r["model"], "eps": eps, "pi_model": r["accuracy"], "pi_max": row["pi_max"], "pi_max_discretized": row["pi_max_discretized"], "gap": row["pi_max"] - r["accuracy"], "gap_discretized": row["pi_max_discretized"] - r["accuracy"]})
        row["max_model_pi"] = best
        row["bound_holds"] = bool(row["pi_max"] + 1e-12 >= best)
        row["discretized_bound_holds"] = bool(row["pi_max_discretized"] + 1e-12 >= best)
        if "pi_exact" in row:  # synthetic: also check against the exact optimal predictor
            row["bound_holds_exact"] = bool(row["pi_max"] + 1e-12 >= row["pi_exact"])
            row["discretized_bound_holds_exact"] = bool(row["pi_max_discretized"] + 1e-12 >= row["pi_exact"])
        rows.append(row)
        log.info("exp5 %s eps=%.4g: pi_max=%.4f discretized=%.4f best_model=%.4f", ds.name, eps, row["pi_max"], row["pi_max_discretized"], best)
    df = pd.DataFrame(rows)
    long_df = pd.DataFrame(long_rows)
    save_csv(df, out / "exp5_results.csv")
    save_csv(long_df, out / "exp5_pairs.csv")
    summary = correlations(long_df)
    save_csv(summary, out / "exp5_spearman.csv")
    log.info("exp5: tolerance bound holds for %d/%d datasets; discretised bound holds for %d/%d", df.bound_holds.sum(), len(df), df.discretized_bound_holds.sum(), len(df))
    write_metadata(out, vars(args))
    plot(df, long_df, out)
    return df


def correlations(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, s in long_df.groupby("model"):
        if len(s) >= 3:
            rho, p = spearmanr(s.pi_max, s.pi_model)
        else:
            rho, p = np.nan, np.nan
        rows.append({"model": model, "n_datasets": len(s), "spearman_rho": rho, "p_value": p, "min_gap": s.gap.min(), "violations": int((s.gap < -1e-12).sum())})
    if len(long_df) >= 3:
        rho, p = spearmanr(long_df.pi_max, long_df.pi_model)
        rows.append({"model": "ALL", "n_datasets": len(long_df), "spearman_rho": rho, "p_value": p, "min_gap": long_df.gap.min(), "violations": int((long_df.gap < -1e-12).sum())})
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, long_df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for model, s in long_df.groupby("model"):
        ax.scatter(s.pi_max, s.pi_model, label=model, alpha=0.8)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel(r"$\Pi^{max}$")
    ax.set_ylabel(r"$\Pi^{model}$")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7)
    ax.set_title("Experiment 5a – all points should lie below the diagonal")
    finish_plot(fig, out / "exp5a_scatter.png")

    order = df.sort_values("pi_max")
    cols = ["pi_max", "pi_max_discretized", "max_model_pi"]
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(order)), 4))
    order.set_index("dataset")[cols].plot.bar(ax=ax)
    ax.set_ylabel("predictability")
    ax.set_ylim(0, 1.05)
    ax.set_title("Experiment 5b – tolerance vs discretised bound vs best model")
    finish_plot(fig, out / "exp5b_bars.png")
