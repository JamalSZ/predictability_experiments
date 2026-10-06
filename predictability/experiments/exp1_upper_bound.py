"""Experiment 1: validity of Pi^max as an upper bound (synthetic Markov chains).

For N_x D_y chains (x in a list of state counts, y in {u, s}) with eps = 0:
  * Pi^max from the IFI and RNLJ entropy estimates (must coincide),
  * Pi_exact: exact Bayes-optimal predictability of the generating chain,
  * Pi_FOMM: realized accuracy of a first-order Markov predictor (80/20),
  * optional statsforecast baselines with predictions snapped to the nearest
    state value (continuous predictions never match exactly at eps = 0).
A violation Pi_model > Pi^max falsifies the bound.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..alphabet import alphabet_range
from ..benchmarks import eps_accuracy, run_fomm, run_statsforecast, snap_to_values
from ..estimators import estimate_entropy_rate
from ..predictability import pi_max_detailed
from ..synthetic import generate
from .common import ensure_dir, finish_plot, log, parse_int_list, save_csv, write_metadata


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--out", default="results/exp1")
    p.add_argument("--n", type=int, default=5000, help="series length (<= 5000)")
    p.add_argument("--states", default="5,10,15,20,25,30,35,40,45,50")
    p.add_argument("--dists", default="u,s")
    p.add_argument("--eps", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--boundary", default="lmp+", choices=["lmp+", "zero-fill"])
    p.add_argument("--variant", default="paper", choices=["paper", "fano"])
    p.add_argument("--benchmarks", default="Naive,AutoARIMA", help="statsforecast models ('' to skip)")
    p.add_argument("--no-snap", action="store_true", help="do not snap benchmark predictions to state values")


def run(args: argparse.Namespace) -> pd.DataFrame:
    out = ensure_dir(args.out)
    n = min(args.n, 5000)
    rows = []
    bench_models = [m for m in args.benchmarks.split(",") if m.strip()]
    for dist in args.dists.split(","):
        for ns in parse_int_list(args.states):
            name = f"N{ns}D{dist}"
            chain, x, _ = generate(name, n, seed=args.seed)
            log.info("exp1 %s: n=%d", name, n)
            H_exact = chain.entropy_rate()
            pi_exact = chain.optimal_predictability(args.eps)
            N = alphabet_range(x, args.eps)  # = number of distinct values for eps = 0
            row = {"dataset": name, "n_states": ns, "dist": dist, "n": n, "eps": args.eps, "H_exact": H_exact, "N": N, "pi_exact": pi_exact}
            for alg in ("ifi", "rnlj"):
                H = estimate_entropy_rate(x, args.eps, alg, args.boundary)
                pm = pi_max_detailed(H, N, args.variant)
                row[f"H_{alg}"] = H
                row[f"pi_max_{alg}"] = pm.pi_max
                row[f"pi_max_{alg}_status"] = pm.status
            fomm = run_fomm(x, name, eps=args.eps)
            row["pi_FOMM"] = float(eps_accuracy(fomm, args.eps)["accuracy"].iloc[0])
            if bench_models:
                preds = run_statsforecast(x, name, models=bench_models)
                if not args.no_snap:
                    preds = snap_to_values(preds, chain.values)
                acc = eps_accuracy(preds, args.eps)
                for _, r in acc.iterrows():
                    row[f"pi_{r['model']}"] = r["accuracy"]
            model_cols = [c for c in row if c.startswith("pi_") and not c.startswith("pi_max") and c != "pi_exact"]
            row["max_model_pi"] = max(row[c] for c in model_cols)
            row["bound_holds_models"] = bool(row["pi_max_ifi"] + 1e-12 >= row["max_model_pi"])
            row["bound_holds_exact"] = bool(row["pi_max_ifi"] + 1e-12 >= pi_exact)
            row["gap_to_FOMM"] = row["pi_max_ifi"] - row["pi_FOMM"]
            rows.append(row)
    df = pd.DataFrame(rows)
    save_csv(df, out / "exp1_results.csv")
    write_metadata(out, vars(args))
    plot(df, out)
    log.info("exp1 summary: bound holds vs models in %d/%d datasets, vs exact optimum in %d/%d", df.bound_holds_models.sum(), len(df), df.bound_holds_exact.sum(), len(df))
    return df


def plot(df: pd.DataFrame, out: Path) -> None:
    for dist, sub in df.groupby("dist"):
        fig, ax = plt.subplots(figsize=(6, 4))
        sub = sub.sort_values("n_states")
        ax.plot(sub.n_states, sub.pi_max_ifi, "k-o", label=r"$\Pi^{max}$ (IFI)")
        ax.plot(sub.n_states, sub.pi_max_rnlj, "k--", label=r"$\Pi^{max}$ (RNLJ)")
        ax.plot(sub.n_states, sub.pi_exact, "g-s", label=r"$\Pi^{exact}$")
        for c in [c for c in sub.columns if c.startswith("pi_") and not c.startswith("pi_max") and c != "pi_exact"]:
            ax.plot(sub.n_states, sub[c], "-^", label=r"$\Pi^{%s}$" % c[3:])
        ax.set_xlabel("number of states $N_s$")
        ax.set_ylabel("predictability")
        ax.set_title(f"Experiment 1 – family D{dist}")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        finish_plot(fig, out / f"exp1_D{dist}.png")
