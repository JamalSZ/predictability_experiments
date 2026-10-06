"""Inverted File Index (IFI) algorithm (audit draft, Section D).

1. Build an inverted index: distinct value -> sorted list of positions, plus
   a sorted key array K.
2. Process t = n .. 1.  For each t the eps-matching earlier positions are
   obtained by binary search on K for the key range [x_t - eps, x_t + eps]
   followed by a scan of the matching keys' position lists (positions < t).
3. Sparse two-row DP (dicts curr/next keyed by t') implementing eq:dpM.

DISCREPANCY NOTE.  The draft text and pseudocode use the lookup range
[x_t - 2eps, x_t + 2eps], arguing via interval overlap of [x-eps, x+eps]
intervals.  That is inconsistent with the matching definition
x_t ~eps x_t'  <=>  |x_t - x_t'| <= eps (Definition def:error) which is what
the DP, the baseline and the entropy estimator use.  A 2eps range would make
IFI return *different* Lambda vectors than RNLJ/BF.  This implementation uses
the +-eps range so that all algorithms agree; the key range is widened by a
tiny float slack and every candidate is re-checked with |x_t - k| <= eps so
that floating-point rounding in (x_t - eps) cannot change the result.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from typing import Dict, List, Tuple

import numpy as np

from .common import DEFAULT_BOUNDARY, ZERO_FILL, as_float_array, check_boundary, register


def build_index(x: np.ndarray) -> Tuple[Dict[float, List[int]], List[float]]:
    """IFI_BuildIdx: dictionary value -> ascending positions, sorted keys."""
    index: Dict[float, List[int]] = {}
    for i, v in enumerate(x.tolist()):
        lst = index.get(v)
        if lst is None:
            index[v] = [i]
        else:
            lst.append(i)
    keys = sorted(index)
    return index, keys


def _slack(eps: float, scale: float) -> float:
    # tiny widening of the bisect range to guard against rounding of x +- eps
    return 1e-9 * (abs(eps) + scale) + 1e-300


def lookup(keys: List[float], index: Dict[float, List[int]], value: float, t: int, eps: float, slack: float) -> List[int]:
    """IFI_Lookup: all positions t' < t with |x_t' - value| <= eps."""
    lb = bisect_left(keys, value - eps - slack)
    rb = bisect_right(keys, value + eps + slack)
    matches: List[int] = []
    ee = 2*eps
    for k in range(lb, rb):
        key = keys[k]
        if abs(key - value) > ee:  # exact re-check (removes slack candidates)
            continue
        positions = index[key]
        cut = bisect_left(positions, t)  # positions are ascending
        if cut:
            matches.extend(positions[:cut])
    return matches


@register("ifi")
def lambda_ifi(x, eps: float, boundary: str = DEFAULT_BOUNDARY) -> np.ndarray:
    check_boundary(boundary)
    x = as_float_array(x)
    n = len(x)
    lam = np.ones(n, dtype=np.int64)
    if n <= 1:
        return lam
    zero_fill = boundary == ZERO_FILL
    index, keys = build_index(x)
    scale = float(np.max(np.abs(x))) if n else 0.0
    slack = _slack(eps, scale)
    xl = x.tolist()
    curr: Dict[int, int] = {}
    for t in range(n - 1, -1, -1):
        matches = lookup(keys, index, xl[t], t, eps, slack)
        nxt = curr  # row t+1
        curr = {}
        best = 0
        if matches:
            get = nxt.get
            for tp in matches:
                v = get(tp + 1, 0) + 1
                cap = t - tp
                if v > cap:
                    v = cap
                curr[tp] = v
                if v > best:
                    best = v
        if zero_fill and t + best >= n:
            lam[t] = 0
        else:
            lam[t] = best + 1
    return lam
