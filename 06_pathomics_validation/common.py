from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"
SUBDIRS = {
    "manifests": OUTPUT_ROOT / "manifests",
    "features": OUTPUT_ROOT / "features",
    "statistics": OUTPUT_ROOT / "statistics",
    "logs": OUTPUT_ROOT / "logs",
    "cellvit_outputs": OUTPUT_ROOT / "cellvit_outputs",
}


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def write_table(frame: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        try:
            frame.to_parquet(path, index=index)
            return path
        except (ImportError, ValueError):
            path = path.with_suffix(".csv")
    frame.to_csv(path, index=index)
    return path


def finite_numeric_columns(frame: pd.DataFrame, exclude: set[str] | None = None) -> list[str]:
    excluded = exclude or set()
    return [
        column for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
        and np.isfinite(pd.to_numeric(frame[column], errors="coerce").dropna()).any()
    ]


def bh_fdr(p_values) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    output = np.full(p.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if len(valid) == 0:
        return output
    order = valid[np.argsort(p[valid])]
    adjusted = p[order] * len(order) / np.arange(1, len(order) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output[order] = np.clip(adjusted, 0, 1)
    return output


def mad(values) -> float:
    x = np.asarray(values, dtype=float); x = x[np.isfinite(x)]
    return float(np.median(np.abs(x - np.median(x)))) if len(x) else np.nan


def coefficient_of_variation(values) -> float:
    x = np.asarray(values, dtype=float); x = x[np.isfinite(x)]
    if not len(x) or math.isclose(float(x.mean()), 0.0):
        return np.nan
    return float(x.std(ddof=1) / abs(x.mean())) if len(x) > 1 else 0.0


def gini(values) -> float:
    x = np.asarray(values, dtype=float); x = x[np.isfinite(x)]
    if not len(x): return np.nan
    if x.min() < 0: x = x - x.min()
    if x.sum() <= 0: return 0.0
    x = np.sort(x); n = len(x)
    return float(2 * np.sum(np.arange(1, n + 1) * x) / (n * x.sum()) - (n + 1) / n)


def shannon_from_counts(counts) -> float:
    values = np.asarray(counts, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.sum() <= 0: return np.nan
    probabilities = values / values.sum()
    return float(-np.sum(probabilities * np.log(probabilities)))


def simpson_from_counts(counts) -> float:
    values = np.asarray(counts, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.sum() <= 0: return np.nan
    probabilities = values / values.sum()
    return float(1 - np.sum(probabilities * probabilities))
