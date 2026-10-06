"""Brute-force reference algorithm (O(n^3) worst case, O(1) extra space).

For every position t and every earlier position t' < t the longest
eps-matching prefix is extended element by element.  The match length is
capped at t - t' (the earlier occurrence must end before t, main.tex eq:dpM)
and at n - t + 1 (the prefix cannot run past the end of the series).
"""
from __future__ import annotations

import numpy as np

from .common import DEFAULT_BOUNDARY, ZERO_FILL, as_float_array, check_boundary, register


@register("bf")
def lambda_brute_force(x, eps: float, boundary: str = DEFAULT_BOUNDARY) -> np.ndarray:
    check_boundary(boundary)
    x = as_float_array(x)
    n = len(x)
    lam = np.ones(n, dtype=np.int64)
    xl = x.tolist()  # plain Python floats are much faster in tight loops
    for t in range(n):
        best = 0
        for tp in range(t):
            max_len = min(t - tp, n - t)
            if max_len <= best:
                # cannot beat the current best; still a correct pruning
                continue
            length = 0
            while length < max_len and abs(xl[t + length] - xl[tp + length]) <= eps:
                length += 1
            if length > best:
                best = length
        if boundary == ZERO_FILL and t + best >= n:
            lam[t] = 0
        else:
            lam[t] = best + 1
    return lam
