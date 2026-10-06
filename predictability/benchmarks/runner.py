"""Forecasting benchmark runner.

Established univariate forecasting baselines are taken from Nixtla's
``statsforecast`` library (https://github.com/Nixtla/statsforecast), which
is the reference implementation behind several published univariate
benchmarks (M3/M4/M5, Tourism).  Each model produces one-step-ahead
forecasts over the test part of the series (last 20 % by default).

Protocol
--------
``StatsForecast.cross_validation(h=1, step_size=1, n_windows=n_test, refit=...)``
implements a rolling-origin evaluation: for every test position t the model
sees x_1..x_{t-1} and predicts x_t.  With ``refit=False`` (default here) the
model parameters are estimated once on the training part (first 80 %) and
then rolled forward over the test part with the state updated but the
parameters fixed; this is the efficient equivalent of a full re-fit at every
step (``refit=True``) or every k steps (``refit=k``).

Output
------
A long-format ``pandas.DataFrame`` with the columns
``dataset, model, t, true, predicted`` (t is the 0-based position in the
original series) that can be written to CSV, and ``eps_accuracy`` computing

    accuracy(eps) = count(|true - predicted| <= eps) / number of predictions

(``strict=True`` switches to ``<``, as in the user's wording; the default
``<=`` follows Definition def:error of the paper and is required for eps = 0
exact matching on synthetic data).

Additional model providers can be registered via ``MODEL_PROVIDERS`` (e.g.
neuralforecast / darts adapters); only statsforecast and the first-order
Markov predictor (``fomm``) are shipped and tested.
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

OUTPUT_COLUMNS = ["dataset", "model", "t", "true", "predicted"]
DEFAULT_MODELS = ("Naive", "SeasonalNaive", "WindowAverage", "AutoARIMA", "AutoETS", "AutoTheta")


def train_test_split_index(n: int, test_frac: float = 0.2) -> int:
    """Index of the first test position for an 80/20 split."""
    n_test = max(1, int(round(n * test_frac)))
    return n - n_test


def _build_statsforecast_models(names: Iterable[str], season_length: int):
    from statsforecast import models as sfm

    built = []
    for name in names:
        if name == "SeasonalNaive":
            if season_length <= 1:
                log.info("skipping SeasonalNaive (season_length <= 1 would equal Naive)")
                continue
            built.append(sfm.SeasonalNaive(season_length=season_length))
        elif name == "WindowAverage":
            built.append(sfm.WindowAverage(window_size=max(2, season_length if season_length > 1 else 5)))
        elif name in ("AutoARIMA", "AutoETS", "AutoTheta", "AutoCES"):
            cls = getattr(sfm, name)
            built.append(cls(season_length=season_length) if season_length > 1 else cls())
        elif hasattr(sfm, name):
            built.append(getattr(sfm, name)())
        else:
            raise ValueError(f"unknown statsforecast model {name!r}")
    return built


def run_statsforecast(
    y: Sequence[float],
    dataset: str,
    models: Iterable[str] = DEFAULT_MODELS,
    test_frac: float = 0.2,
    season_length: int = 1,
    refit: Union[bool, int] = False,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Rolling one-step-ahead forecasts with statsforecast models."""
    from statsforecast import StatsForecast

    y = np.asarray(y, dtype=float)
    n = len(y)
    split = train_test_split_index(n, test_frac)
    n_test = n - split
    sf_models = _build_statsforecast_models(models, season_length)
    if not sf_models:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    df = pd.DataFrame({"unique_id": dataset, "ds": np.arange(n), "y": y})
    # Models without a `forward` method (e.g. WindowAverage) cannot be rolled
    # forward without refitting; they are cheap, so they are re-fit each step.
    groups = []
    with_forward = [m for m in sf_models if hasattr(m, "forward")]
    without_forward = [m for m in sf_models if not hasattr(m, "forward")]
    if with_forward:
        groups.append((with_forward, refit))
    if without_forward:
        groups.append((without_forward, True))
    cvs = []
    for group, group_refit in groups:
        sf = StatsForecast(models=group, freq=1, n_jobs=n_jobs)
        cv = sf.cross_validation(df=df, h=1, step_size=1, n_windows=n_test, refit=group_refit)
        cv = cv.reset_index(drop=True) if "ds" in cv.columns else cv.reset_index()
        cvs.append(cv)
    cv = cvs[0]
    for extra in cvs[1:]:
        cv = cv.merge(extra.drop(columns=["y", "cutoff"], errors="ignore"), on=["unique_id", "ds"], how="inner")
    rows = []
    for m in sf_models:
        col = m.alias if hasattr(m, "alias") else repr(m)
        rows.append(
            pd.DataFrame(
                {
                    "dataset": dataset,
                    "model": col,
                    "t": cv["ds"].to_numpy(dtype=int),
                    "true": cv["y"].to_numpy(dtype=float),
                    "predicted": cv[col].to_numpy(dtype=float),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)[OUTPUT_COLUMNS]


def run_fomm(
    y: Sequence[float],
    dataset: str,
    test_frac: float = 0.2,
    eps: float = 0.0,
    decimals: Optional[int] = None,
) -> pd.DataFrame:
    """First-order Markov predictor (FOMM) for synthetic / discrete-valued series.

    Transition counts are estimated on the training part over the distinct
    values and updated online over the test part (every observation is known
    before the next prediction).  The prediction is the centre of the
    2eps-window with the largest estimated transition mass (for eps = 0 the
    most frequent successor).  Unseen states fall back to the most frequent
    value overall.
    """
    from ..synthetic import best_window_mass

    y = np.asarray(y, dtype=float)
    if decimals is not None:
        y = np.round(y, decimals)
    n = len(y)
    split = train_test_split_index(n, test_frac)
    values, codes = np.unique(y, return_inverse=True)
    k = len(values)
    counts = np.zeros((k, k))
    for a, b in zip(codes[: split - 1], codes[1:split]):
        counts[a, b] += 1
    marginal = np.bincount(codes[:split], minlength=k).astype(float)
    preds = np.empty(n - split)
    for i, t in enumerate(range(split, n)):
        prev = codes[t - 1]
        row = counts[prev]
        probs = row / row.sum() if row.sum() > 0 else marginal / marginal.sum()
        preds[i] = best_window_mass(values, probs, eps)[1]
        counts[prev, codes[t]] += 1
        marginal[codes[t]] += 1
    return pd.DataFrame(
        {"dataset": dataset, "model": "FOMM", "t": np.arange(split, n), "true": y[split:], "predicted": preds}
    )[OUTPUT_COLUMNS]


MODEL_PROVIDERS: Dict[str, Callable[..., pd.DataFrame]] = {
    "statsforecast": run_statsforecast,
    "fomm": run_fomm,
}


def eps_accuracy(predictions: pd.DataFrame, eps: float, strict: bool = False) -> pd.DataFrame:
    """Per (dataset, model) accuracy = count(|true - predicted| (<= or <) eps) / n_pred."""
    err = (predictions["true"] - predictions["predicted"]).abs()
    hit = err < eps if strict else err <= eps
    out = (
        predictions.assign(hit=hit.astype(float))
        .groupby(["dataset", "model"], sort=False)["hit"]
        .agg(accuracy="mean", n_pred="size")
        .reset_index()
    )
    out.insert(2, "eps", eps)
    return out


def eps_accuracy_sweep(predictions: pd.DataFrame, eps_values: Iterable[float], strict: bool = False) -> pd.DataFrame:
    return pd.concat([eps_accuracy(predictions, e, strict) for e in eps_values], ignore_index=True)


def snap_to_values(predictions: pd.DataFrame, values: Sequence[float]) -> pd.DataFrame:
    """Round continuous predictions to the nearest admissible value (used for
    exact matching, eps = 0, on synthetic state-valued series)."""
    vals = np.sort(np.asarray(values, dtype=float))
    p = predictions["predicted"].to_numpy(dtype=float)
    idx = np.clip(np.searchsorted(vals, p), 1, len(vals) - 1)
    left, right = vals[idx - 1], vals[idx]
    snapped = np.where(np.abs(p - left) <= np.abs(p - right), left, right)
    out = predictions.copy()
    out["predicted"] = snapped
    return out
