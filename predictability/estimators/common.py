"""Shared definitions for the Lambda estimators.

All algorithms compute the vector Lambda = (lambda_1, ..., lambda_n) of
Definition def:lambda in main.tex under tolerance matching

    x_a ~eps x_b   <=>   |x_a - x_b| <= eps

and then the Kontoyiannis entropy-rate estimate

    H_est = log2(n) / ((1/n) * sum_t lambda_t).

Boundary conventions (main.tex, Sec. "Lempel-Ziv Encoding"):

* ``lmp+``      : lambda_t = max_{t'<t} M[t,t'] + 1 (longest matching prefix
                  plus one, even if that prefix runs into the end of the series).
* ``zero-fill`` : as lmp+, but lambda_t = 0 if t + max_{t'<t} M[t,t'] > n
                  (1-based), i.e. if the longest match reaches the end of the
                  series so that no novel prefix exists.

Internally all arrays are 0-based.  With t0 = t - 1 the 0-fill condition
becomes ``t0 + maxM >= n``.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Sequence

import numpy as np

LMP_PLUS = "lmp+"
ZERO_FILL = "zero-fill"
BOUNDARIES = (LMP_PLUS, ZERO_FILL)

# Default boundary convention.  main.tex presents lmp+ as the Kontoyiannis
# et al. rule and 0-fill as the Smart et al. implementation variant; the
# audit draft recommends lmp+ (it yields a lower entropy estimate and hence a
# safer upper bound Pi^max).
DEFAULT_BOUNDARY = LMP_PLUS


def check_boundary(boundary: str) -> str:
    if boundary not in BOUNDARIES:
        raise ValueError(f"boundary must be one of {BOUNDARIES}, got {boundary!r}")
    return boundary


def as_float_array(x: Sequence[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError("time series must be one-dimensional")
    if np.isnan(arr).any():
        raise ValueError("time series contains NaN values; clean the data first")
    return arr


def eps_match(a: float, b: float, eps: float) -> bool:
    """Tolerance matching |a - b| <= eps (Definition def:error in main.tex)."""
    return abs(a - b) <= eps


def apply_zero_fill(lam: np.ndarray, max_match: np.ndarray) -> np.ndarray:
    """Set lambda_t = 0 where the longest match reaches the end (0-fill)."""
    n = len(lam)
    t0 = np.arange(n)
    lam = lam.copy()
    lam[t0 + max_match >= n] = 0
    return lam


def entropy_rate_from_lambda(lam: Sequence[int]) -> float:
    """Kontoyiannis estimator H_est = log2(n) / mean(lambda) (Definition def:kontoyiannis)."""
    lam = np.asarray(lam, dtype=float)
    n = len(lam)
    if n == 0:
        raise ValueError("empty Lambda vector")
    mean_lam = lam.mean()
    if mean_lam <= 0:
        return math.inf
    return math.log2(n) / mean_lam


# Registry filled by the individual algorithm modules (see __init__.py).
ALGORITHMS: Dict[str, Callable[..., np.ndarray]] = {}


def register(name: str):
    def deco(fn):
        ALGORITHMS[name] = fn
        return fn

    return deco
