"""Reversed Nested-Loop Join baseline (Algorithm alg:baseline in main.tex).

Two-row dynamic programme over the matrix M (main.tex eq:dpM):

    M[t,t'] = min(M[t+1,t'+1] + 1, t - t')   if x_t ~eps x_t'
            = 0                              otherwise

built bottom-up (t = n .. 2) with a reversed nested loop over t' < t.
Theta(n^2) time, O(n) space.

NOTE on the pseudocode in main.tex: the 0-fill condition there reads
``curr[t'] <= n`` which is always true.  Following the text above the
algorithm ("lambda_t = 0 if t + max M[t,t'] > n") the implemented condition
is ``t + curr[t'] <= n`` (1-based).

Two implementations are provided:

* ``lambda_rnlj``       -- pure Python, faithful to the pseudocode (used in
                           the runtime experiment so that all algorithms are
                           compared in the same language).
* ``lambda_rnlj_numpy`` -- row-vectorised NumPy variant (fast, same output).
"""
from __future__ import annotations

import numpy as np

from .common import DEFAULT_BOUNDARY, ZERO_FILL, as_float_array, check_boundary, register


@register("rnlj")
def lambda_rnlj(x, eps: float, boundary: str = DEFAULT_BOUNDARY) -> np.ndarray:
    check_boundary(boundary)
    x = as_float_array(x)
    n = len(x)
    lam = np.ones(n, dtype=np.int64)
    if n <= 1:
        return lam
    xl = x.tolist()
    zero_fill = boundary == ZERO_FILL
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    last = n - 1
    for t in range(last, 0, -1):
        xt = xl[t]
        best = 0
        if t == last:
            for tp in range(t - 1, -1, -1):
                if abs(xt - xl[tp]) <= eps:
                    curr[tp] = 1
                    best = 1
                else:
                    curr[tp] = 0
        else:
            for tp in range(t - 1, -1, -1):
                if abs(xt - xl[tp]) <= eps:
                    v = prev[tp + 1] + 1
                    cap = t - tp
                    if v > cap:
                        v = cap
                    curr[tp] = v
                    if v > best:
                        best = v
                else:
                    curr[tp] = 0
        if zero_fill and t + best >= n:
            lam[t] = 0
        else:
            lam[t] = best + 1
        prev, curr = curr, prev
    return lam


@register("rnlj_numpy")
def lambda_rnlj_numpy(x, eps: float, boundary: str = DEFAULT_BOUNDARY) -> np.ndarray:
    """Row-vectorised version of the baseline; identical output."""
    check_boundary(boundary)
    x = as_float_array(x)
    n = len(x)
    lam = np.ones(n, dtype=np.int64)
    if n <= 1:
        return lam
    zero_fill = boundary == ZERO_FILL
    prev = np.zeros(n, dtype=np.int64)  # row t+1, indexed by t' in [0, t]
    for t in range(n - 1, 0, -1):
        match = np.abs(x[:t] - x[t]) <= eps
        cap = t - np.arange(t)
        curr = np.where(match, np.minimum(prev[1 : t + 1] + 1, cap), 0)
        best = int(curr.max()) if t > 0 else 0
        if zero_fill and t + best >= n:
            lam[t] = 0
        else:
            lam[t] = best + 1
        prev = curr
    return lam
