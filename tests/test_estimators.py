import math

import numpy as np
import pytest

from predictability.estimators import ALGORITHMS, compute_lambda, entropy_rate_from_lambda, estimate_entropy_rate

FAST_ALGORITHMS = [a for a in ALGORITHMS if a != "bf"]


def test_paper_example_lambda_lmp_plus_and_zero_fill():
    # main.tex Example ex:lambda, sequence abbaabaaabba
    x = [ord(c) for c in "abbaabaaabba"]
    expected_lmp = [1, 1, 2, 2, 3, 4, 3, 4, 5, 4, 3, 2]
    expected_zero = [1, 1, 2, 2, 3, 4, 3, 4, 0, 0, 0, 0]
    for alg in ALGORITHMS:
        assert compute_lambda(x, 0, alg, "lmp+").tolist() == expected_lmp, alg
        assert compute_lambda(x, 0, alg, "zero-fill").tolist() == expected_zero, alg


def test_paper_matrix_example():
    # main.tex example for M with T=(1.2,3.5,4.1,2.1,1.8,3.8), eps=1: M[5,1]=2 -> lambda_5 = 3
    x = [1.2, 3.5, 4.1, 2.1, 1.8, 3.8]
    for alg in ALGORITHMS:
        lam = compute_lambda(x, 1.0, alg)
        assert lam.tolist() == [1, 1, 2, 2, 3, 2], alg


@pytest.mark.parametrize("boundary", ["lmp+", "zero-fill"])
@pytest.mark.parametrize("kind", ["int", "float", "const", "lowres"])
@pytest.mark.parametrize("eps", [0.0, 0.05, 0.3, 1.0, 2.5])
def test_all_algorithms_agree_with_brute_force(boundary, kind, eps):
    rng = np.random.default_rng(hash((boundary, kind, eps)) % 2**32)
    for _ in range(15):
        n = int(rng.integers(1, 45))
        if kind == "int":
            x = rng.integers(0, 6, n).astype(float)
        elif kind == "float":
            x = rng.normal(size=n)
        elif kind == "const":
            x = np.full(n, 3.0)
        else:
            x = np.round(rng.uniform(0, 3, n), 1)
        ref = compute_lambda(x, eps, "bf", boundary)
        for alg in FAST_ALGORITHMS:
            out = compute_lambda(x, eps, alg, boundary)
            assert out.tolist() == ref.tolist(), (alg, boundary, eps, x.tolist())


def test_float_boundary_exactness():
    # values exactly eps apart must match (<= eps), slightly more must not
    x = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.0, 0.1, 0.2, 0.3])
    for alg in ALGORITHMS:
        a = compute_lambda(x, 0.1, alg)
        b = compute_lambda(x, 0.0999, alg)
        assert a.tolist() == compute_lambda(x, 0.1, "bf").tolist()
        assert b.tolist() == compute_lambda(x, 0.0999, "bf").tolist()
        assert a.sum() > b.sum()


def test_edge_cases():
    for alg in ALGORITHMS:
        assert compute_lambda([], 0.0, alg).tolist() == []
        assert compute_lambda([1.0], 0.0, alg).tolist() == [1]
        assert compute_lambda([1.0, 2.0], 5.0, alg, "lmp+").tolist() == [1, 2]
        assert compute_lambda([1.0, 2.0], 5.0, alg, "zero-fill").tolist() == [1, 0]
    with pytest.raises(ValueError):
        compute_lambda([1.0, 2.0], 0.0, "ifi", "bogus")
    with pytest.raises(ValueError):
        compute_lambda([1.0, 2.0], 0.0, "nope")
    with pytest.raises(ValueError):
        compute_lambda([1.0, float("nan")], 0.0, "ifi")


def test_entropy_rate_formula():
    lam = np.array([1, 1, 2, 2])
    assert math.isclose(entropy_rate_from_lambda(lam), math.log2(4) / 1.5)
    assert entropy_rate_from_lambda([0, 0]) == math.inf
    with pytest.raises(ValueError):
        entropy_rate_from_lambda([])


def test_entropy_of_iid_uniform_is_underestimated_but_in_range():
    rng = np.random.default_rng(1)
    x = rng.integers(0, 8, 3000).astype(float)
    H = estimate_entropy_rate(x, 0.0, "ifi")
    assert 2.0 < H <= 3.0 + 1e-9  # exact rate is 3 bits; LZ77 tends to underestimate


def test_periodic_sequence_has_low_entropy():
    x = np.tile([0.0, 1.0, 2.0, 3.0], 250)
    H = estimate_entropy_rate(x, 0.0, "ifi")
    assert H < 0.2
