"""Synthetic first-order Markov chain generators N_x D_y (audit draft, Sec. E).

* ``u``  uniform transitions: P[i, j] = 1/N_s.
* ``s``  skewed Zipf transitions with alpha = 2: in row i the N_s target
         states are ranked by a row-specific random permutation and
         P[i, j] proportional to 1/rank^alpha.
* ``i``  (i = 0..7) dominant-transition family: N_s - 1 weak transitions
         with probability p_w = 1/(10 * i * N_s) and one dominant transition
         (to state (i_state + 1) mod N_s) carrying the remaining mass.  For
         i = 0 the formula is undefined (1/0); we use the natural limit
         p_w = 1/N_s, i.e. the uniform chain (maximum entropy), so that
         entropy decreases monotonically in i.

States are mapped to numeric values drawn once uniformly from [0, 1] (seeded)
so that tolerance-based comparisons are possible; with eps = 0 the numeric
and symbolic settings coincide.

The exact entropy rate is H = sum_i pi_i H(P[i, :]) with pi the stationary
distribution.  The exact optimal (Bayes) predictability under eps-matching
is also available: in state i the best prediction is the centre of the
2eps-window covering the largest transition mass.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

_NAME_RE = re.compile(r"^N(?P<states>\d+)D(?P<dist>u|s|\d+)$")


def _entropy_bits(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def stationary_distribution(P: np.ndarray) -> np.ndarray:
    """Stationary distribution via the eigenvector for eigenvalue 1."""
    vals, vecs = np.linalg.eig(P.T)
    k = int(np.argmin(np.abs(vals - 1.0)))
    pi = np.real(vecs[:, k])
    pi = np.abs(pi) / np.abs(pi).sum()
    # polish with a few power iterations (robust for near-periodic chains)
    for _ in range(50):
        pi = pi @ P
    return pi / pi.sum()


@dataclass
class MarkovChain:
    name: str
    P: np.ndarray  # transition matrix (rows sum to 1)
    values: np.ndarray  # numeric value of each state

    @property
    def n_states(self) -> int:
        return self.P.shape[0]

    def stationary(self) -> np.ndarray:
        return stationary_distribution(self.P)

    def entropy_rate(self) -> float:
        pi = self.stationary()
        return float(sum(pi[i] * _entropy_bits(self.P[i]) for i in range(self.n_states)))

    def optimal_predictability(self, eps: float = 0.0) -> float:
        """Exact predictability of the Bayes-optimal one-step predictor.

        For eps = 0 this is sum_i pi_i max_j P[i, j].  For eps > 0 the best
        prediction in state i is a value v maximising the transition mass of
        states whose value lies in [v - eps, v + eps]; it suffices to check
        windows whose left edge sits on a state value.
        """
        pi = self.stationary()
        total = 0.0
        for i in range(self.n_states):
            total += pi[i] * best_window_mass(self.values, self.P[i], eps)[0]
        return float(total)

    def sample_states(self, n: int, seed: Optional[int] = 0, burn_in: int = 1000) -> np.ndarray:
        rng = np.random.default_rng(seed)
        cdf = np.cumsum(self.P, axis=1)
        cdf[:, -1] = 1.0
        s = int(rng.integers(self.n_states))
        out = np.empty(n + burn_in, dtype=np.int64)
        u = rng.random(n + burn_in)
        for k in range(n + burn_in):
            s = int(np.searchsorted(cdf[s], u[k], side="right"))
            if s >= self.n_states:
                s = self.n_states - 1
            out[k] = s
        return out[burn_in:]

    def sample(self, n: int, seed: Optional[int] = 0, burn_in: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Return (numeric series, state sequence)."""
        states = self.sample_states(n, seed, burn_in)
        return self.values[states], states


def best_window_mass(values: np.ndarray, probs: np.ndarray, eps: float) -> Tuple[float, float]:
    """Max probability mass of a window [v - eps, v + eps] and the centre v."""
    if eps <= 0:
        j = int(np.argmax(probs))
        return float(probs[j]), float(values[j])
    order = np.argsort(values)
    vs = values[order]
    ps = probs[order]
    best, best_v = -1.0, float(vs[0])
    for a in range(len(vs)):
        hi = np.searchsorted(vs, vs[a] + 2 * eps, side="right")
        mass = float(ps[a:hi].sum())
        if mass > best:
            best, best_v = mass, float(vs[a] + eps)
    return best, best_v


def make_transition_matrix(n_states: int, dist: str, seed: int = 0, zipf_alpha: float = 2.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if dist == "u":
        return np.full((n_states, n_states), 1.0 / n_states)
    if dist == "s":
        ranks = np.arange(1, n_states + 1, dtype=float)
        base = ranks ** (-zipf_alpha)
        base /= base.sum()
        P = np.empty((n_states, n_states))
        for i in range(n_states):
            perm = rng.permutation(n_states)
            P[i, perm] = base
        return P
    if dist.isdigit():
        i_idx = int(dist)
        if i_idx == 0:
            p_w = 1.0 / n_states
        else:
            p_w = 1.0 / (10.0 * i_idx * n_states)
        P = np.full((n_states, n_states), p_w)
        for s in range(n_states):
            P[s, (s + 1) % n_states] = 1.0 - (n_states - 1) * p_w
        return P
    raise ValueError(f"unknown distribution code {dist!r}")


def make_chain(name: str, seed: int = 0) -> MarkovChain:
    """Build the chain for a name like 'N10Du', 'N25Ds', 'N10D3'."""
    m = _NAME_RE.match(name)
    if not m:
        raise ValueError(f"invalid synthetic dataset name {name!r} (expected e.g. N10Du, N10Ds, N10D3)")
    n_states = int(m.group("states"))
    if n_states < 2:
        raise ValueError("need at least 2 states")
    P = make_transition_matrix(n_states, m.group("dist"), seed=seed)
    values = np.random.default_rng(seed + 10_000).uniform(0.0, 1.0, size=n_states)
    return MarkovChain(name, P, values)


def generate(name: str, n: int, seed: int = 0) -> Tuple[MarkovChain, np.ndarray, np.ndarray]:
    """Convenience: chain, numeric series, state sequence."""
    chain = make_chain(name, seed)
    x, states = chain.sample(n, seed)
    return chain, x, states
