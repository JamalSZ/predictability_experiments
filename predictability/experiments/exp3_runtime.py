"""Experiment 3: runtime and peak memory of BF / RNLJ / IESJ / IFI.

Series lengths n in {500, 1000, 2000, 5000} (cap 5000) and eps levels
eps_min (exact), eps_25%, eps_50%, eps_75% (pair-match quantiles) and
eps_max (all pairs match).  Data: synthetic N50Du and, optionally, real
datasets from the YAML config.

Time and memory are measured in separate runs: wall-clock time without any
tracing (median of --repeats), peak memory with ``tracemalloc`` (Python and
NumPy allocations above the baseline).  BF is only run for n <= --bf-max-n.
All pure-Python implementations are compared in the same language; the
NumPy-vectorised RNLJ (``rnlj_numpy``) can be added via --algorithms.
"""
from __future__ import annotations

import argparse
import gc
import time
import tracemalloc
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..estimators import compute_lambda
from ..synthetic import generate
from .common import ensure_dir, eps_grid, finish_plot, log, match_fraction, parse_float_list, parse_int_list, save_csv, write_metadata


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--out", default="results/exp3")
    p.add_argument("--sizes", default="500,1000,2000,5000")
    p.add_argument("--algorithms", default="bf,rnlj,iesj,ifi")
    p.add_argument("--bf-max-n", type=int, default=1000)
    p.add_argument("--eps-levels", default="0,0.25,0.5,0.75,1", help="pair-match quantiles (0=exact, 1=all)")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--synthetic", default="N50Du", help="synthetic dataset name ('' to skip)")
    p.add_argument("--config", default=None, help="datasets.yaml to include real datasets")
    p.add_argument("--datasets", default="", help="comma-separated subset of real dataset names")
    p.add_argument("--boundary", default="lmp+", choices=["lmp+", "zero-fill"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-memory", action="store_true", help="skip tracemalloc memory measurement")


def _time_once(alg: str, x: np.ndarray, eps: float, boundary: str) -> float:
    gc.collect()
    t0 = time.perf_counter()
    compute_lambda(x, eps, alg, boundary)
    return time.perf_counter() - t0


def _peak_memory_mb(alg: str, x: np.ndarray, eps: float, boundary: str) -> float:
    gc.collect()
    tracemalloc.start()
    base, _ = tracemalloc.get_traced_memory()
    compute_lambda(x, eps, alg, boundary)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return (peak - base) / 2**20


def _series_sources(args) -> Dict[str, np.ndarray]:
    sources: Dict[str, np.ndarray] = {}
    max_n = max(parse_int_list(args.sizes))
    if args.synthetic:
        _, x, _ = generate(args.synthetic, min(max_n, 5000), seed=args.seed)
        sources[args.synthetic] = x
    if args.config:
        from ..datasets import load_all

        names = [d for d in args.datasets.split(",") if d] or None
        for ds in load_all(args.config, names):
            sources[ds.name] = ds.values
    return sources


def run(args: argparse.Namespace) -> pd.DataFrame:
    out = ensure_dir(args.out)
    sizes = [min(s, 5000) for s in parse_int_list(args.sizes)]
    algorithms = [a for a in args.algorithms.split(",") if a]
    levels = parse_float_list(args.eps_levels)
    rows: List[dict] = []
    for src_name, full in _series_sources(args).items():
        for n in sizes:
            if n > len(full):
                log.warning("exp3 %s: only %d points available, skipping n=%d", src_name, len(full), n)
                continue
            x = full[:n]
            grid = eps_grid(x, levels)
            for eps_name, eps in grid.items():
                frac = match_fraction(x, eps)
                for alg in algorithms:
                    if alg == "bf" and n > args.bf_max_n:
                        continue
                    times = [_time_once(alg, x, eps, args.boundary) for _ in range(args.repeats)]
                    mem = np.nan if args.no_memory else _peak_memory_mb(alg, x, eps, args.boundary)
                    row = {
                        "source": src_name,
                        "n": n,
                        "eps_level": eps_name,
                        "eps": eps,
                        "match_fraction": frac,
                        "algorithm": alg,
                        "time_s_median": float(np.median(times)),
                        "time_s_min": float(np.min(times)),
                        "time_s_all": ";".join(f"{t:.6f}" for t in times),
                        "peak_mem_mb": mem,
                        "repeats": args.repeats,
                    }
                    rows.append(row)
                    log.info("exp3 %s n=%d %s(%.4g) %s: %.3fs, %.1f MB", src_name, n, eps_name, eps, alg, row["time_s_median"], mem)
    df = pd.DataFrame(rows)
    if df.empty:
        log.warning("exp3: nothing measured")
        return df
    ifi = df[df.algorithm == "ifi"].set_index(["source", "n", "eps_level"]).time_s_median
    rnlj = df[df.algorithm == "rnlj"].set_index(["source", "n", "eps_level"]).time_s_median
    speed = (rnlj / ifi).rename("speedup_rnlj_over_ifi").reset_index()
    df = df.merge(speed, on=["source", "n", "eps_level"], how="left")
    save_csv(df, out / "exp3_results.csv")
    write_metadata(out, vars(args))
    plot(df, out)
    return df


def plot(df: pd.DataFrame, out: Path) -> None:
    for src, sub in df.groupby("source"):
        # (a) log-log runtime vs n at eps_25% (or the first available level)
        level = "eps_25%" if "eps_25%" in set(sub.eps_level) else sub.eps_level.iloc[0]
        fig, ax = plt.subplots(figsize=(6, 4))
        for alg, s in sub[sub.eps_level == level].groupby("algorithm"):
            s = s.sort_values("n")
            ax.loglog(s.n, s.time_s_median, "-o", label=alg)
            if len(s) >= 2:
                slope = np.polyfit(np.log(s.n), np.log(s.time_s_median), 1)[0]
                ax.annotate(f"slope {slope:.2f}", (s.n.iloc[-1], s.time_s_median.iloc[-1]), fontsize=7)
        ax.set_xlabel("n")
        ax.set_ylabel("runtime [s]")
        ax.set_title(f"Experiment 3a – {src} @ {level}")
        ax.legend()
        finish_plot(fig, out / f"exp3a_runtime_vs_n_{src}.png")
        # (b) runtime vs match fraction at the largest n
        n_max = sub.n.max()
        fig, ax = plt.subplots(figsize=(6, 4))
        for alg, s in sub[sub.n == n_max].groupby("algorithm"):
            s = s.sort_values("match_fraction")
            ax.plot(s.match_fraction, s.time_s_median, "-o", label=alg)
        ax.set_xlabel("fraction of eps-matching pairs")
        ax.set_ylabel("runtime [s]")
        ax.set_title(f"Experiment 3b – {src}, n={n_max}")
        ax.legend()
        finish_plot(fig, out / f"exp3b_runtime_vs_eps_{src}.png")
        # (c) peak memory at the largest n
        s = sub[(sub.n == n_max) & sub.peak_mem_mb.notna()]
        if not s.empty:
            piv = s.pivot_table(index="algorithm", columns="eps_level", values="peak_mem_mb")
            fig, ax = plt.subplots(figsize=(6, 4))
            piv.plot.bar(ax=ax)
            ax.set_ylabel("peak memory [MB]")
            ax.set_title(f"Experiment 3c – {src}, n={n_max}")
            finish_plot(fig, out / f"exp3c_memory_{src}.png")
