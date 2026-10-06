"""Experiment 2: accuracy of the entropy-rate estimator vs the exact rate.

Families: (a) N10 D_i, i = 0..7 (increasing skew, decreasing entropy);
          (b) N_x D_u, x = 2..50 (increasing entropy).
Reports H_IFI, H_RNLJ (must be identical), H_exact, absolute/relative error
and sign, for both boundary conventions.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..estimators import estimate_entropy_rate
from ..synthetic import generate
from .common import ensure_dir, finish_plot, log, parse_int_list, save_csv, write_metadata


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--out", default="results/exp2")
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--skew-states", type=int, default=10)
    p.add_argument("--skew-indices", default="0,1,2,3,4,5,6,7")
    p.add_argument("--uniform-states", default=",".join(str(v) for v in range(2, 51)))
    p.add_argument("--eps", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--boundaries", default="lmp+,zero-fill")


def _evaluate(name: str, n: int, eps: float, seed: int, boundaries, family: str, index_value) -> list:
    chain, x, _ = generate(name, n, seed=seed)
    H_exact = chain.entropy_rate()
    rows = []
    for b in boundaries:
        H_ifi = estimate_entropy_rate(x, eps, "ifi", b)
        H_rnlj = estimate_entropy_rate(x, eps, "rnlj_numpy", b)
        rows.append(
            {
                "family": family,
                "dataset": name,
                "index": index_value,
                "n": n,
                "eps": eps,
                "boundary": b,
                "H_exact": H_exact,
                "H_ifi": H_ifi,
                "H_rnlj": H_rnlj,
                "ifi_equals_rnlj": bool(abs(H_ifi - H_rnlj) < 1e-12),
                "abs_error": abs(H_ifi - H_exact),
                "rel_error": (abs(H_ifi - H_exact) / H_exact) if H_exact > 0 else np.nan,
                "sign": "over" if H_ifi > H_exact else ("under" if H_ifi < H_exact else "exact"),
            }
        )
    log.info("exp2 %s: H_exact=%.4f H_ifi=%s", name, H_exact, [round(r["H_ifi"], 4) for r in rows])
    return rows


def run(args: argparse.Namespace) -> pd.DataFrame:
    out = ensure_dir(args.out)
    n = min(args.n, 5000)
    boundaries = [b for b in args.boundaries.split(",") if b]
    rows = []
    for i in parse_int_list(args.skew_indices):
        rows += _evaluate(f"N{args.skew_states}D{i}", n, args.eps, args.seed, boundaries, "skew", i)
    for ns in parse_int_list(args.uniform_states):
        rows += _evaluate(f"N{ns}Du", n, args.eps, args.seed, boundaries, "uniform", ns)
    df = pd.DataFrame(rows)
    save_csv(df, out / "exp2_results.csv")
    write_metadata(out, vars(args))
    plot(df, out)
    if not df.ifi_equals_rnlj.all():
        log.error("exp2: IFI and RNLJ estimates differ for %d rows!", (~df.ifi_equals_rnlj).sum())
    return df


def plot(df: pd.DataFrame, out: Path) -> None:
    for family, xlabel, fname in (("skew", "skew index $i$", "exp2_skew.png"), ("uniform", "number of states $N_s$", "exp2_uniform.png")):
        sub = df[df.family == family]
        if sub.empty:
            continue
        scales = ["linear", "log"] if family == "skew" else ["linear"]
        fig, axes = plt.subplots(1, len(scales), figsize=(6 * len(scales), 4), squeeze=False)
        for ax, scale in zip(axes[0], scales):
            for b, s in sub.groupby("boundary"):
                s = s.sort_values("index")
                ax.plot(s["index"], s.H_ifi, "-o", label=f"$H_{{est}}$ ({b})")
            s = sub[sub.boundary == sub.boundary.iloc[0]].sort_values("index")
            ax.plot(s["index"], s.H_exact, "k--", label="$H_{exact}$")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("entropy rate [bits]")
            ax.set_yscale(scale)
            ax.legend(fontsize=8)
        fig.suptitle(f"Experiment 2 – {family} family")
        finish_plot(fig, out / fname)
