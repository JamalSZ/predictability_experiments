"""Effective alphabet size N of a numeric time series under tolerance eps.

Three definitions from main.tex are implemented:

``range``
    N = ceil((x_max - x_min) / eps)  -- the eps-cell count used in the
    derivation of eq:final (Sec. "Upper Bound for Predictability").  The
    source text writes (x_max - x_min)/eps without ceiling; the reviewer
    annotation (\\jm) and audit item G2 ask for the ceiling, which we apply.
``range_padded``
    N = ceil((x_max + eps - (x_min - eps)) / eps) = ceil((x_max - x_min)/eps) + 2,
    the form attributed to the published KDD'24 paper (audit item G2).
``recurrence``
    Expected-recurrence-time formulation, eq:distN / eq:numericdist:
    N = (1/n) * sum_t d_t with
        d_t = t - max{ i < t : x_i ~eps x_t }                 if such i exists
        d_t = t + (n - max{ i >= t : x_i ~eps x_t })          otherwise (circular)
    (1-based positions; the "otherwise" case includes i = t itself).
``recurrence_permuted``
    Mean of ``recurrence`` over R random permutations of the series (the
    order-independent estimate of main.tex), with the 95 % confidence
    half-width 1.96 * s / sqrt(R) reported.  R is either fixed or grown
    until the half-width is below an absolute error target.

For eps = 0 all variants based on recurrence reduce to the number of
distinct values; ``range`` with eps = 0 is defined as the number of distinct
values as well.

NOTE on the worked example in main.tex (Example ex:numericalphasize):
for the series (1.2, 3.5, 5.3, 4.1, 2.6, 6.2, 1.9, 5.0) with eps = 1 the
table lists d_8 = 5 but |5.0 - 4.1| = 0.9 <= 1 gives d_8 = 8 - 4 = 4 and a
total of 24/8 = 3.0 instead of 25/8 = 3.125.  The implementation follows the
formula, not the table.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

METHODS = ("range", "range_padded", "recurrence", "recurrence_permuted")
DEFAULT_METHOD = "range"


def _distinct_count(x: np.ndarray) -> int:
    return int(np.unique(x).size)


def alphabet_range(x, eps: float, padded: bool = False) -> float:
    x = np.asarray(x, dtype=float)
    if eps <= 0:
        return float(_distinct_count(x))
    span = float(x.max() - x.min())
    if padded:
        span += 2.0 * eps
    # guard against values like 3.0000000000000004 from floating point
    return float(math.ceil(span / eps - 1e-9))


def recurrence_distances(x, eps: float) -> np.ndarray:
    """d_t of eq:numericdist for every position (vectorised row by row, O(n^2))."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    d = np.empty(n, dtype=np.int64)
    for t in range(n):  # 0-based t; 1-based position is t + 1
        prev = np.flatnonzero(np.abs(x[:t] - x[t]) <= eps)
        if prev.size:
            d[t] = t - prev[-1]
        else:
            later = np.flatnonzero(np.abs(x[t:] - x[t]) <= eps)  # includes t itself
            last = t + later[-1]
            d[t] = (t + 1) + (n - (last + 1))
    return d


def alphabet_recurrence(x, eps: float) -> float:
    d = recurrence_distances(x, eps)
    return float(d.mean())


@dataclass
class PermutedAlphabet:
    mean: float
    std: float
    half_width_95: float
    R: int

    @property
    def value(self) -> float:
        return self.mean


def alphabet_recurrence_permuted(
    x,
    eps: float,
    R: Optional[int] = None,
    abs_error: float = 0.05,
    min_R: int = 10,
    max_R: int = 200,
    seed: int = 0,
) -> PermutedAlphabet:
    """Mean recurrence-based alphabet size over random permutations.

    If ``R`` is given exactly R permutations are used; otherwise permutations
    are added (starting from ``min_R``) until 1.96 * s / sqrt(R) <= abs_error
    or ``max_R`` is reached.
    """
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    samples = []

    def half_width() -> float:
        if len(samples) < 2:
            return math.inf
        return 1.96 * float(np.std(samples, ddof=1)) / math.sqrt(len(samples))

    target_R = R if R is not None else min_R
    while True:
        while len(samples) < target_R:
            perm = rng.permutation(x)
            samples.append(alphabet_recurrence(perm, eps))
        if R is not None or half_width() <= abs_error or len(samples) >= max_R:
            break
        target_R = min(max_R, max(len(samples) + min_R, int(math.ceil((1.96 * np.std(samples, ddof=1) / abs_error) ** 2))))
    std = float(np.std(samples, ddof=1)) if len(samples) > 1 else 0.0
    return PermutedAlphabet(float(np.mean(samples)), std, half_width(), len(samples))


def effective_alphabet_size(x, eps: float, method: str = DEFAULT_METHOD, **kwargs) -> float:
    """Dispatch to one of the alphabet-size definitions (see module docstring)."""
    if method == "range":
        return alphabet_range(x, eps, padded=False)
    if method == "range_padded":
        return alphabet_range(x, eps, padded=True)
    if method == "recurrence":
        return alphabet_recurrence(x, eps)
    if method == "recurrence_permuted":
        return alphabet_recurrence_permuted(x, eps, **kwargs).mean
    raise ValueError(f"unknown alphabet method {method!r}; choose from {METHODS}")
