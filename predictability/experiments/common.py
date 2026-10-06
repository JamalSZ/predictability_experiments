"""Helpers shared by the experiment runners."""
from __future__ import annotations

import json
import logging
import platform
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..alphabet import alphabet_range, alphabet_recurrence, alphabet_recurrence_permuted  # noqa: E402
from ..estimators import compute_lambda, entropy_rate_from_lambda  # noqa: E402
from ..predictability import pi_max_detailed  # noqa: E402

log = logging.getLogger("predictability")


def setup_logging(verbose: bool = True) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("statsforecast").setLevel(logging.WARNING)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_csv(df: pd.DataFrame, path: Path) -> Path:
    df.to_csv(path, index=False)
    log.info("wrote %s (%d rows)", path, len(df))
    return path


def write_metadata(out: Path, args: Dict) -> None:
    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "args": {k: (v if isinstance(v, (int, float, str, bool, list, type(None))) else str(v)) for k, v in args.items()},
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))


def pairwise_distance_quantiles(x: np.ndarray, quantiles: Sequence[float], max_points: int = 1500, seed: int = 0) -> np.ndarray:
    """Quantiles of |x_i - x_j| over (a sample of) all pairs i < j.

    eps = q-quantile makes roughly a fraction q of all pairs eps-match, which
    is how eps_{25%}, eps_{50%}, ... of Experiment 3 are defined.
    """
    x = np.asarray(x, dtype=float)
    if len(x) > max_points:
        rng = np.random.default_rng(seed)
        x = rng.choice(x, size=max_points, replace=False)
    iu = np.triu_indices(len(x), k=1)
    d = np.abs(x[:, None] - x[None, :])[iu]
    return np.quantile(d, quantiles)


def eps_grid(x: np.ndarray, levels: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0), **kw) -> Dict[str, float]:
    """Named eps levels: eps_min (exact matching), eps_q for quantiles, eps_max (all pairs match)."""
    x = np.asarray(x, dtype=float)
    out: Dict[str, float] = {}
    inner = [q for q in levels if 0.0 < q < 1.0]
    qs = pairwise_distance_quantiles(x, inner, **kw) if inner else np.array([])
    for q in levels:
        if q <= 0.0:
            out["eps_min"] = 0.0
        elif q >= 1.0:
            out["eps_max"] = float(x.max() - x.min())
        else:
            out[f"eps_{int(round(q * 100))}%"] = float(qs[inner.index(q)])
    return out


def match_fraction(x: np.ndarray, eps: float, max_points: int = 1500, seed: int = 0) -> float:
    """Fraction of pairs i<j with |x_i - x_j| <= eps (sampled for large n)."""
    x = np.asarray(x, dtype=float)
    if len(x) > max_points:
        x = np.random.default_rng(seed).choice(x, size=max_points, replace=False)
    iu = np.triu_indices(len(x), k=1)
    d = np.abs(x[:, None] - x[None, :])[iu]
    return float((d <= eps).mean())


def analyze_series(
    x: np.ndarray,
    eps: float,
    algorithm: str = "ifi",
    boundary: str = "lmp+",
    variant: str = "paper",
    alphabet_methods: Iterable[str] = ("range", "recurrence"),
    permuted_R: Optional[int] = 20,
    primary_alphabet: str = "range",
) -> Dict[str, float]:
    """Entropy-rate estimate, effective alphabet sizes and Pi^max for one eps."""
    lam = compute_lambda(x, eps, algorithm, boundary)
    H = entropy_rate_from_lambda(lam)
    row: Dict[str, float] = {"eps": eps, "n": len(x), "algorithm": algorithm, "boundary": boundary, "H_est": H, "mean_lambda": float(lam.mean())}
    for m in alphabet_methods:
        if m == "range":
            N = alphabet_range(x, eps)
        elif m == "range_padded":
            N = alphabet_range(x, eps, padded=True)
        elif m == "recurrence":
            N = alphabet_recurrence(x, eps)
        elif m == "recurrence_permuted":
            res = alphabet_recurrence_permuted(x, eps, R=permuted_R)
            N = res.mean
            row["N_recurrence_permuted_hw95"] = res.half_width_95
        else:
            raise ValueError(m)
        row[f"N_{m}"] = N
        pm = pi_max_detailed(H, N, variant)
        row[f"pi_max_{m}"] = pm.pi_max
        row[f"pi_max_{m}_status"] = pm.status
    row["N"] = row[f"N_{primary_alphabet}"]
    row["pi_max"] = row[f"pi_max_{primary_alphabet}"]
    row["pi_max_variant"] = variant
    return row


def parse_int_list(text: str) -> List[int]:
    return [int(v) for v in text.split(",") if v.strip()]


def parse_float_list(text: str) -> List[float]:
    return [float(v) for v in text.split(",") if v.strip()]


def finish_plot(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("wrote %s", path)
