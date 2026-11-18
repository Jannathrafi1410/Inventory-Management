"""Conformal prediction calibration (split-conformal / CQR) on a held-out time block.

Assumption: calibration block is a time-ordered holdout disjoint from training (and respects embargo
when used with splits). CQR adjusts quantile predictions using nonconformity scores. Score
quantile uses the finite-sample conformal correction ceil((n+1)*(1-alpha))/n (clipped to [0,1]).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class CQRCalibration:
    """CQR calibration offsets for a lower/upper quantile pair.

    Offsets are applied as [lo - q_lower_adjust, hi + q_upper_adjust]; negative adjustments are
    allowed (tightening) when intervals are over-conservative.
    """

    alpha: float
    q_lower_adjust: float
    q_upper_adjust: float
    n_calib: int


def _conformal_q_level(n: int, alpha: float) -> float:
    if n <= 0:
        return 0.0
    return float(np.clip(np.ceil((n + 1) * (1 - alpha)) / n, 0.0, 1.0))


def split_conformal_residuals(y_calib: np.ndarray, yhat_calib: np.ndarray, alpha: float = 0.1) -> float:
    """Split-conformal symmetric residual radius for nominal miscoverage alpha (e.g. 0.1 for 90%)."""
    resid = np.abs(np.asarray(y_calib, dtype=float) - np.asarray(yhat_calib, dtype=float))
    n = len(resid)
    if n == 0:
        return 0.0
    q_level = _conformal_q_level(n, alpha)
    return float(np.quantile(resid, q_level, method="higher"))


def cqr_calibrate(
    y_calib: np.ndarray,
    q_lo_calib: np.ndarray,
    q_hi_calib: np.ndarray,
    alpha: float = 0.1,
    symmetric: bool = False,
) -> CQRCalibration:
    """Conformalized Quantile Regression (CQR) calibration on holdout block.

    Default (asymmetric, recommended): per-side scores
      s_lo_i = q_lo_i - y_i,  s_hi_i = y_i - q_hi_i
    with conformal quantiles Q_lo, Q_hi giving [q_lo - Q_lo, q_hi + Q_hi].
    Symmetric mode uses E_i = max(s_lo_i, s_hi_i) for a single shared adjustment.
    Negative Q_* values are permitted to tighten over-wide intervals.
    """
    y = np.asarray(y_calib, dtype=float)
    lo = np.asarray(q_lo_calib, dtype=float)
    hi = np.asarray(q_hi_calib, dtype=float)
    n = len(y)
    if n == 0:
        return CQRCalibration(alpha=alpha, q_lower_adjust=0.0, q_upper_adjust=0.0, n_calib=0)
    q_level = _conformal_q_level(n, alpha)
    if symmetric:
        scores = np.maximum(lo - y, y - hi)
        qhat = float(np.quantile(scores, q_level, method="higher"))
        return CQRCalibration(alpha=alpha, q_lower_adjust=qhat, q_upper_adjust=qhat, n_calib=n)
    s_lo = lo - y
    s_hi = y - hi
    q_lo_adj = float(np.quantile(s_lo, q_level, method="higher"))
    q_hi_adj = float(np.quantile(s_hi, q_level, method="higher"))
    return CQRCalibration(alpha=alpha, q_lower_adjust=q_lo_adj, q_upper_adjust=q_hi_adj, n_calib=n)


def apply_cqr(q_lo: np.ndarray, q_hi: np.ndarray, calib: CQRCalibration) -> Tuple[np.ndarray, np.ndarray]:
    """Apply CQR offsets to prediction intervals and enforce non-negativity + monotonic order."""
    lo = np.asarray(q_lo, dtype=float) - float(calib.q_lower_adjust)
    hi = np.asarray(q_hi, dtype=float) + float(calib.q_upper_adjust)
    lo = np.maximum(lo, 0.0)
    hi = np.maximum(hi, lo)
    return lo, hi


def apply_split_conformal(yhat: np.ndarray, radius: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return symmetric interval [yhat-radius, yhat+radius]."""
    r = float(max(radius, 0.0))
    center = np.asarray(yhat, dtype=float)
    return np.maximum(center - r, 0.0), center + r


__all__ = ["CQRCalibration", "split_conformal_residuals", "cqr_calibrate", "apply_cqr", "apply_split_conformal"]

