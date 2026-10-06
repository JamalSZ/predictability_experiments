"""Inequality (epsilon) Self-Join algorithm (IESJ).

Phase 1 -- materialise the join Q_0 (audit draft eq:join): all pairs (t, t')
with t' < t and |x_t - x_t'| <= eps.  This is done with a sort of the values
followed by two vectorised binary searches (numpy.searchsorted), producing K
pairs in O(n log n + K) time and O(K) memory.
Phase 2 -- sort the pairs by (t descending, t' ascending), O(K log K).
Phase 3 -- dynamic programme eq:dpM over the sorted pairs, carrying the
previous row in a dictionary.

Space is O(n + K) with K up to n(n-1)/2, which is the main disadvantage
compared to IFI (O(n)).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .common import DEFAULT_BOUNDARY, ZERO_FILL, as_float_array, check_boundary, register


def materialize_pairs(x: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return arrays (T, TP) of all matching pairs with TP < T, sorted by
    T descending then TP ascending."""
    n = len(x)
    order = np.argsort(x, kind="stable")
    xs = x[order]
    scale = float(np.max(np.abs(x))) if n else 0.0
    slack = 1e-9 * (abs(eps) + scale) + 1e-300
    lo = np.searchsorted(xs, x - eps - slack, side="left")
    hi = np.searchsorted(xs, x + eps + slack, side="right")
    counts = hi - lo
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    t_rep = np.repeat(np.arange(n, dtype=np.int64), counts)
    starts = np.repeat(np.cumsum(counts) - counts, counts)
    offsets = np.arange(total, dtype=np.int64) - starts + np.repeat(lo, counts)
    tp = order[offsets]
    mask = (tp < t_rep) & (np.abs(x[t_rep] - x[tp]) <= eps)
    t_rep = t_rep[mask]
    tp = tp[mask]
    # sort by t descending, then t' ascending
    idx = np.lexsort((tp, -t_rep))
    return t_rep[idx], tp[idx]


@register("iesj")
def lambda_iesj(x, eps: float, boundary: str = DEFAULT_BOUNDARY) -> np.ndarray:
    check_boundary(boundary)
    x = as_float_array(x)
    n = len(x)
    lam = np.ones(n, dtype=np.int64)
    if n <= 1:
        return lam
    zero_fill = boundary == ZERO_FILL
    T, TP = materialize_pairs(x, eps)
    K = len(T)
    Tl = T.tolist()
    TPl = TP.tolist()
    prev_row = {}
    prev_t = None
    i = 0
    while i < K:
        t = Tl[i]
        # rows for t+1 that produced no pairs mean the previous row is empty
        if prev_t is not None and prev_t != t + 1:
            prev_row = {}
        curr_row = {}
        best = 0
        get = prev_row.get
        while i < K and Tl[i] == t:
            tp = TPl[i]
            v = get(tp + 1, 0) + 1
            cap = t - tp
            if v > cap:
                v = cap
            curr_row[tp] = v
            if v > best:
                best = v
            i += 1
        if zero_fill and t + best >= n:
            lam[t] = 0
        else:
            lam[t] = best + 1
        prev_row = curr_row
        prev_t = t
    # positions without any pair keep lambda = 1 (no match -> novel prefix of length 1)
    return lam
