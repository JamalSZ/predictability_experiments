"""Algorithms computing the Lempel-Ziv coefficient vector Lambda under
tolerance matching, and the Kontoyiannis entropy-rate estimator.

Available algorithms (``ALGORITHMS`` registry):

    bf          brute force, O(n^3)                 (reference, small n only)
    rnlj        reversed nested-loop join, Theta(n^2) pure Python (paper baseline)
    rnlj_numpy  same DP, row-vectorised with NumPy
    ifi         inverted file index, O(n^2) worst case, O(n) space (proposed)
    iesj        inequality self-join, O(n^2 + K log K), O(n + K) space

FIT (forest of interval trees, O(n^3 log n) time / O(n^3) space) from the
complexity table of the draft is NOT implemented: it is impractical for any
n of interest and plays no role in the reported experiments.
"""
from __future__ import annotations

import numpy as np

from .common import (  # noqa: F401
    ALGORITHMS,
    BOUNDARIES,
    DEFAULT_BOUNDARY,
    LMP_PLUS,
    ZERO_FILL,
    entropy_rate_from_lambda,
)
from . import brute_force, rnlj, ifi, iesj  # noqa: F401  (register algorithms)

DEFAULT_ALGORITHM = "ifi"


def compute_lambda(x, eps: float, algorithm: str = DEFAULT_ALGORITHM, boundary: str = DEFAULT_BOUNDARY) -> np.ndarray:
    """Compute Lambda with the chosen algorithm."""
    try:
        fn = ALGORITHMS[algorithm]
    except KeyError as exc:
        raise ValueError(f"unknown algorithm {algorithm!r}; choose from {sorted(ALGORITHMS)}") from exc
    return fn(x, eps, boundary)


def estimate_entropy_rate(x, eps: float, algorithm: str = DEFAULT_ALGORITHM, boundary: str = DEFAULT_BOUNDARY) -> float:
    """Kontoyiannis entropy-rate estimate (bits per time step)."""
    return entropy_rate_from_lambda(compute_lambda(x, eps, algorithm, boundary))
