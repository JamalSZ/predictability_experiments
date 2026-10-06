"""Loading of local real-world datasets described in a YAML config.

Nothing is downloaded.  Example ``configs/datasets.yaml``::

    defaults:
      max_n: 5000          # truncate every series to at most this length
      slice: head          # 'head' (first max_n values) or 'tail' (last max_n)
    datasets:
      - name: ETTh1
        path: /ABSOLUTE/PATH/TO/ETTh1.csv
        column: OT
        season_length: 24  # optional, used by SeasonalNaive
      - name: SP500
        path: /ABSOLUTE/PATH/TO/sp500.csv
        column: Open

Supported file types: .csv/.txt (pandas.read_csv, optional ``sep``,
``header``), .parquet, .npy.  ``column`` may be a name or a 0-based index.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

MAX_N_HARD_LIMIT = 5000  # per user instruction: at most 5000 data points for now


@dataclass
class DatasetSpec:
    name: str
    path: str
    column: Any = None
    max_n: Optional[int] = MAX_N_HARD_LIMIT
    slice: str = "head"
    sep: Optional[str] = None
    header: Any = "infer"
    season_length: int = 1
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Dataset:
    name: str
    values: np.ndarray
    spec: Optional[DatasetSpec] = None
    season_length: int = 1

    @property
    def n(self) -> int:
        return len(self.values)


def load_config(path: str) -> List[DatasetSpec]:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    defaults = cfg.get("defaults", {}) or {}
    specs: List[DatasetSpec] = []
    for entry in cfg.get("datasets", []) or []:
        merged = {**defaults, **entry}
        known = {k: merged.pop(k) for k in list(merged) if k in DatasetSpec.__dataclass_fields__ and k != "extra"}
        specs.append(DatasetSpec(**known, extra=merged))
    return specs


def _read_values(spec: DatasetSpec) -> np.ndarray:
    p = Path(spec.path).expanduser()
    if not p.exists():
        raise FileNotFoundError(
            f"dataset {spec.name!r}: file {p} not found. Edit configs/datasets.yaml to point to your local copy."
        )
    suffix = p.suffix.lower()
    if suffix == ".npy":
        arr = np.load(p)
        return np.asarray(arr, dtype=float).ravel()
    if suffix == ".parquet":
        df = pd.read_parquet(p)
    else:
        kwargs = {}
        if spec.sep is not None:
            kwargs["sep"] = spec.sep
        df = pd.read_csv(p, header=spec.header, **kwargs)
    if spec.column is None:
        numeric = df.select_dtypes(include=[np.number])
        if numeric.shape[1] == 0:
            raise ValueError(f"dataset {spec.name!r}: no numeric column found")
        col = numeric.iloc[:, 0]
    elif isinstance(spec.column, int) and spec.column not in df.columns:
        col = df.iloc[:, spec.column]
    else:
        if spec.column not in df.columns:
            raise KeyError(f"dataset {spec.name!r}: column {spec.column!r} not in {list(df.columns)}")
        col = df[spec.column]
    return pd.to_numeric(col, errors="coerce").to_numpy(dtype=float)


def load_dataset(spec: DatasetSpec) -> Dataset:
    values = _read_values(spec)
    values = values[~np.isnan(values)]
    max_n = spec.max_n
    if max_n is None or max_n > MAX_N_HARD_LIMIT:
        max_n = MAX_N_HARD_LIMIT
    if len(values) > max_n:
        values = values[:max_n] if spec.slice == "head" else values[-max_n:]
    if len(values) < 10:
        raise ValueError(f"dataset {spec.name!r}: only {len(values)} numeric values after cleaning")
    return Dataset(spec.name, values, spec, season_length=int(spec.season_length or 1))


def load_all(config_path: str, names: Optional[List[str]] = None) -> List[Dataset]:
    specs = load_config(config_path)
    if names:
        specs = [s for s in specs if s.name in names]
        missing = set(names) - {s.name for s in specs}
        if missing:
            raise KeyError(f"datasets not in config: {sorted(missing)}")
    return [load_dataset(s) for s in specs]
