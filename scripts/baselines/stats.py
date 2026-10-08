"""Percentile bootstrap confidence intervals and paired differences.

All methods of one judgment are scored on the same resampled index sets, so a
difference between two methods is a paired comparison: in every resample, the
score of method A minus the score of method B on exactly the same issues.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

# A metric maps an index array (positions in the test set, possibly repeated) to a score.
Metric = Callable[[np.ndarray], float]


def resample_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    """n_boot x n matrix of positions drawn with replacement."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n), dtype=np.int32)


def percentile_ci(values: np.ndarray, level: float = 0.95) -> list[float]:
    tail = (1 - level) / 2 * 100
    return [float(np.percentile(values, tail)), float(np.percentile(values, 100 - tail))]


def boot_values(metric: Metric, idx: np.ndarray) -> np.ndarray:
    return np.array([metric(row) for row in idx])


def mean_boot_values(per_item: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Fast path for metrics that are a mean of per-item values (accuracy, hit rates, recall@k)."""
    return per_item[idx].mean(axis=1)


def summary(point: float, boots: np.ndarray) -> dict:
    return {"point": float(point), "ci95": percentile_ci(boots)}


def paired_difference(point_a: float, boots_a: np.ndarray,
                      point_b: float, boots_b: np.ndarray) -> dict:
    """A minus B. `excludes_zero` is True when the 95% interval lies entirely on one side of 0."""
    d = boots_a - boots_b
    lo, hi = percentile_ci(d)
    return {"point": float(point_a - point_b), "ci95": [lo, hi], "excludes_zero": bool(lo > 0 or hi < 0)}
