import math

import numpy as np
import pytest

from predictability.alphabet import (
    alphabet_range,
    alphabet_recurrence,
    alphabet_recurrence_permuted,
    effective_alphabet_size,
    recurrence_distances,
)
from predictability.predictability import fano_bound, pi_max, pi_max_detailed
from predictability.synthetic import best_window_mass, generate, make_chain


# ---------------------------------------------------------------- Pi^max
def test_fano_paper_formula_matches_eq_final():
    P, N = 0.7, 10
    expected = -P * (math.log2(P) - 1) - (1 - P) * (math.log2(1 - P) - math.log2(N - 2))
    assert math.isclose(fano_bound(P, N, "paper"), expected)
    expected_f = -P * math.log2(P) - (1 - P) * math.log2(1 - P) + (1 - P) * math.log2(N - 1)
    assert math.isclose(fano_bound(P, N, "fano"), expected_f)


@pytest.mark.parametrize("variant", ["paper", "fano"])
def test_pi_max_inverts_bound(variant):
    N = 12
    for P in [0.3, 0.5, 0.8, 0.95]:
        H = fano_bound(P, N, variant)
        assert math.isclose(pi_max(H, N, variant), P, abs_tol=1e-8)


def test_pi_max_monotone_in_H_and_clipping():
    N = 20
    hs = np.linspace(0.0, math.log2(N), 30)
    ps = [pi_max(h, N, "paper") for h in hs]
    assert all(a >= b - 1e-12 for a, b in zip(ps, ps[1:]))
    assert pi_max_detailed(0.5, N, "paper").status == "clipped_high" and ps[0] == 1.0
    assert pi_max_detailed(10.0, N, "paper").status == "clipped_low"
    assert math.isclose(pi_max(10.0, N, "paper"), 2.0 / N)
    assert math.isclose(pi_max(10.0, N, "fano"), 1.0 / N)
    assert pi_max(1.0, 2, "paper") == 1.0  # degenerate N
    with pytest.raises(ValueError):
        pi_max(1.0, 10, "other")


# ------------------------------------------------------------- alphabet N
def test_symbolic_recurrence_example_from_paper():
    x = [ord(c) for c in "bcbadbcd"]
    assert recurrence_distances(x, 0).tolist() == [3, 3, 2, 8, 5, 3, 5, 3]
    assert alphabet_recurrence(x, 0) == 4.0


def test_numeric_recurrence_example_follows_formula():
    # main.tex Example ex:numericalphasize lists d_8 = 5 (sum 25); the formula
    # gives d_8 = 4 because |5.0 - 4.1| <= 1.  We follow the formula.
    x = [1.2, 3.5, 5.3, 4.1, 2.6, 6.2, 1.9, 5.0]
    assert recurrence_distances(x, 1).tolist() == [2, 5, 3, 2, 3, 3, 2, 4]
    assert alphabet_recurrence(sorted(x), 1) == 1.75  # matches the paper


def test_recurrence_equals_distinct_count_for_eps_zero():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 7, 200).astype(float)
    assert alphabet_recurrence(x, 0) == len(np.unique(x))
    assert alphabet_range(x, 0) == len(np.unique(x))


def test_range_alphabet():
    x = np.array([0.0, 10.0])
    assert alphabet_range(x, 1.0) == 10
    assert alphabet_range(x, 3.0) == 4
    assert alphabet_range(x, 1.0, padded=True) == 12
    assert effective_alphabet_size(x, 1.0, "range_padded") == 12


def test_permuted_alphabet_statistics():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 80)
    res = alphabet_recurrence_permuted(x, 0.1, R=12)
    assert res.R == 12 and res.std >= 0 and res.half_width_95 >= 0
    adaptive = alphabet_recurrence_permuted(x, 0.1, abs_error=0.5, min_R=5, max_R=20)
    assert 5 <= adaptive.R <= 20


# -------------------------------------------------------------- synthetic
def test_uniform_chain_exact_quantities():
    chain = make_chain("N8Du")
    assert math.isclose(chain.entropy_rate(), 3.0)
    assert math.isclose(chain.optimal_predictability(0.0), 1 / 8)
    pi = chain.stationary()
    assert np.allclose(pi, 1 / 8)


def test_dominant_family_monotone_entropy():
    hs = [make_chain(f"N10D{i}").entropy_rate() for i in range(8)]
    assert math.isclose(hs[0], math.log2(10))
    assert all(a > b for a, b in zip(hs, hs[1:]))
    assert np.allclose(make_chain("N10D3").P.sum(axis=1), 1.0)


def test_zipf_chain_rows_and_predictability_bounds():
    chain = make_chain("N10Ds", seed=3)
    assert np.allclose(chain.P.sum(axis=1), 1.0)
    p0 = chain.optimal_predictability(0.0)
    assert 0 < p0 <= 1
    assert chain.optimal_predictability(0.2) >= p0  # larger tolerance can only help
    assert chain.optimal_predictability(2.0) == pytest.approx(1.0)


def test_best_window_mass():
    values = np.array([0.0, 0.1, 0.5])
    probs = np.array([0.3, 0.3, 0.4])
    mass, centre = best_window_mass(values, probs, 0.05)
    assert math.isclose(mass, 0.6) and math.isclose(centre, 0.05)
    assert best_window_mass(values, probs, 0.0) == (0.4, 0.5)


def test_generate_reproducible_and_states_consistent():
    chain, x, s = generate("N5Ds", 200, seed=7)
    chain2, x2, s2 = generate("N5Ds", 200, seed=7)
    assert np.array_equal(x, x2) and np.array_equal(s, s2)
    assert np.allclose(chain.values[s], x)
    with pytest.raises(ValueError):
        make_chain("foo")
