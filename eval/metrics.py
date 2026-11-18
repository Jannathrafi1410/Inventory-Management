"""Forecast evaluation metrics.

WAPE, MASE, pinball loss per quantile, PICP coverage, PIT histogram data.
All operate on arrays/series provided by caller (censored actuals in baseline use).
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np


def wape(y_true: Sequence[float] | np.ndarray, y_pred: Sequence[float] | np.ndarray) -> float:
    """Weighted Absolute Percentage Error = sum(|y - f|) / sum(|y|)."""
    y = np.asarray(y_true, dtype=float)
    f = np.asarray(y_pred, dtype=float)
    denom = np.sum(np.abs(y))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(np.abs(y - f)) / denom)


def mase(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
    y_train: Sequence[float] | np.ndarray,
    seasonality: int = 1,
) -> float:
    """Mean Absolute Scaled Error vs seasonal naive on in-sample y_train."""
    y = np.asarray(y_true, dtype=float)
    f = np.asarray(y_pred, dtype=float)
    yt = np.asarray(y_train, dtype=float)
    m = int(seasonality)
    if len(yt) <= m or m < 1:
        return float("nan")
    scale = np.mean(np.abs(yt[m:] - yt[:-m]))
    if scale == 0.0:
        return float("nan")
    return float(np.mean(np.abs(y - f)) / scale)


def pinball_loss(
    y_true: Sequence[float] | np.ndarray, y_pred_q: Sequence[float] | np.ndarray, q: float
) -> float:
    """Pinball (quantile) loss for quantile q in (0,1)."""
    y = np.asarray(y_true, dtype=float)
    f = np.asarray(y_pred_q, dtype=float)
    e = y - f
    qq = float(q)
    return float(np.mean(np.maximum(qq * e, (qq - 1.0) * e)))


def pinball_losses(
    y_true: Sequence[float] | np.ndarray, quantile_preds: Dict[float, Sequence[float] | np.ndarray]
) -> Dict[float, float]:
    """Pinball loss per quantile."""
    return {float(q): pinball_loss(y_true, yp, float(q)) for q, yp in quantile_preds.items()}


def picp(
    y_true: Sequence[float] | np.ndarray,
    lower: Sequence[float] | np.ndarray,
    upper: Sequence[float] | np.ndarray,
) -> float:
    """Prediction Interval Coverage Probability for [lower, upper]."""
    y = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    return float(np.mean((y >= lo) & (y <= hi)))


def pit_histogram_data(
    y_true: Sequence[float] | np.ndarray,
    cdf_values: Sequence[float] | np.ndarray,
    bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """PIT histogram data: counts and bin edges for CDF(y_true) in [0,1]."""
    u = np.clip(np.asarray(cdf_values, dtype=float), 0.0, 1.0)
    counts, edges = np.histogram(u, bins=int(bins), range=(0.0, 1.0))
    return counts.astype(int), edges.astype(float)


__all__ = ["wape", "mase", "pinball_loss", "pinball_losses", "picp", "pit_histogram_data"]
