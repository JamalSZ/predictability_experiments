"""Upper bound Pi^max on predictability from an entropy rate (main.tex eq:final).

main.tex eq:final (tolerance-matching variant of Fano's inequality):

    H(X) <= f_paper(P) = -P (log2 P - 1) - (1 - P) (log2(1 - P) - log2(N - 2))
                       = H_b(P) + P + (1 - P) log2(N - 2)

The derivation assumes N eps-cells in the value domain, two of which form
the ball B_eps(x_hat) in which a prediction counts as correct, leaving N - 2
cells for an incorrect prediction.  f_paper is decreasing on (2/N, 1] and
f_paper(1) = 1 bit, so Pi^max = 1 whenever H <= 1 bit.

The classical (Song et al. / Fano) variant used by the discretisation
ablation is

    H(X) <= f_fano(P) = H_b(P) + (1 - P) log2(N - 1),

decreasing on (1/N, 1] with f_fano(1) = 0.

``pi_max`` inverts the chosen f numerically with scipy's brentq.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from scipy.optimize import brentq

Variant = Literal["paper", "fano"]
VARIANTS = ("paper", "fano")
DEFAULT_VARIANT: Variant = "paper"


def binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def fano_bound(p: float, N: float, variant: Variant = DEFAULT_VARIANT) -> float:
    """Right-hand side f(P) of the inequality H <= f(P) for the given variant."""
    if variant == "paper":
        outside = N - 2.0
        extra = p + (1.0 - p) * (math.log2(outside) if outside > 1.0 else 0.0)
        return binary_entropy(p) + extra
    if variant == "fano":
        outside = N - 1.0
        return binary_entropy(p) + (1.0 - p) * (math.log2(outside) if outside > 1.0 else 0.0)
    raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")


@dataclass
class PiMaxResult:
    pi_max: float
    status: str  # 'solved', 'clipped_high' (H below f(1)), 'clipped_low' (H above f(P_lo)), 'degenerate_N'
    p_lo: float
    variant: str


def pi_max_detailed(H: float, N: float, variant: Variant = DEFAULT_VARIANT) -> PiMaxResult:
    """Solve f(P) = H for P on the decreasing branch [P_lo, 1].

    P_lo = 2/N for the paper variant (f'(2/N) = 0) and 1/N for classical
    Fano (the uniform-guess level).  If H is below f(1) the bound is 1; if H
    exceeds f(P_lo) the equation has no solution on the branch and the result
    is clipped to P_lo (random guessing), which is flagged in ``status``.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
    if not math.isfinite(H) or H < 0:
        raise ValueError(f"entropy rate must be finite and non-negative, got {H}")
    min_N = 2.0 if variant == "paper" else 1.0
    if N <= min_N + 1e-12:
        # paper: no cell outside the eps-ball; fano: a single symbol -> trivially predictable
        return PiMaxResult(1.0, "degenerate_N", 1.0, variant)
    p_lo = (2.0 / N) if variant == "paper" else (1.0 / N)
    p_lo = min(p_lo, 1.0)
    f_top = fano_bound(1.0, N, variant)
    if H <= f_top + 1e-12:
        return PiMaxResult(1.0, "clipped_high", p_lo, variant)
    f_lo = fano_bound(p_lo, N, variant)
    if H >= f_lo - 1e-12:
        return PiMaxResult(p_lo, "clipped_low", p_lo, variant)
    g = lambda p: fano_bound(p, N, variant) - H  # noqa: E731
    root = brentq(g, p_lo, 1.0 - 1e-15, xtol=1e-12, maxiter=500)
    return PiMaxResult(float(root), "solved", p_lo, variant)


def pi_max(H: float, N: float, variant: Variant = DEFAULT_VARIANT) -> float:
    """Upper bound Pi^max for entropy rate H (bits) and effective alphabet size N."""
    return pi_max_detailed(H, N, variant).pi_max
