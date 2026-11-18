"""Demand unconstraining for censored (sold-out) observations.

Implements EM unconstraining (normal and lognormal) and a projection-detruncation fallback.
Trains/operates only on censored observed values and sell-out indicators (no latent ground truth).

Assumption: censoring occurs at observed capacity/sales threshold s for sold-out rows (D >= s);
non-sold-out rows treated as uncensored observations of demand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
from scipy import stats


@dataclass
class EMResult:
    """Result of EM unconstraining."""

    mu: float
    sigma: float
    unconstrained: np.ndarray
    n_iter: int
    converged: bool


def _safe_sigma(sigma: float, floor: float = 1e-6) -> float:
    return float(max(sigma, floor))


def em_unconstrain_normal(
    y: Iterable[float],
    censored: Iterable[bool],
    s: Optional[Iterable[float]] = None,
    max_iter: int = 200,
    tol: float = 1e-4,
) -> EMResult:
    """EM unconstraining assuming Normal demand.

    E-step: for censored rows (D >= s), E[D|D>=s] = mu + sigma * phi(z)/(1-Phi(z)), z=(s-mu)/sigma.
    M-step: refit mu,sigma from imputed demands.
    `s` defaults to y for censored rows (capacity/observed-at-cap threshold).
    """
    y_arr = np.asarray(list(y), dtype=float)
    c_arr = np.asarray(list(censored), dtype=bool)
    s_arr = np.asarray(list(s), dtype=float) if s is not None else y_arr.copy()
    n = len(y_arr)
    if n == 0:
        return EMResult(0.0, 1.0, y_arr.copy(), 0, True)

    # initialize from uncensored if possible
    unc = y_arr[~c_arr]
    mu = float(np.mean(unc)) if len(unc) else float(np.mean(y_arr))
    sigma = float(np.std(unc)) if len(unc) > 1 else float(max(np.std(y_arr), 1.0))
    sigma = _safe_sigma(sigma)

    d_imp = y_arr.copy().astype(float)
    converged = False
    for it in range(1, max_iter + 1):
        mu_old, sigma_old = mu, sigma
        # E-step
        if c_arr.any():
            z = (s_arr[c_arr] - mu) / _safe_sigma(sigma)
            # survival = 1-Phi(z); mills ratio phi(z)/(1-Phi(z))
            survival = np.clip(stats.norm.sf(z), 1e-12, None)
            mills = stats.norm.pdf(z) / survival
            cond_mean = mu + sigma * mills
            d_imp[c_arr] = np.maximum(cond_mean, s_arr[c_arr])
        d_imp[~c_arr] = y_arr[~c_arr]
        # M-step
        mu = float(np.mean(d_imp))
        sigma = _safe_sigma(float(np.std(d_imp)))
        if abs(mu - mu_old) < tol * max(1.0, abs(mu_old)) and abs(sigma - sigma_old) < tol * max(1.0, abs(sigma_old)):
            converged = True
            break
    return EMResult(mu=mu, sigma=sigma, unconstrained=d_imp, n_iter=it, converged=converged)


def em_unconstrain_lognormal(
    y: Iterable[float],
    censored: Iterable[bool],
    s: Optional[Iterable[float]] = None,
    max_iter: int = 200,
    tol: float = 1e-4,
) -> EMResult:
    """EM unconstraining assuming LogNormal demand (Normal in log-space).

    Applies normal EM to log(y) with log(s) threshold; returns unconstrained means in original scale
    via imputed log-demands. mu/sigma reported in log-space.
    """
    y_arr = np.asarray(list(y), dtype=float)
    c_arr = np.asarray(list(censored), dtype=bool)
    s_arr = np.asarray(list(s), dtype=float) if s is not None else y_arr.copy()
    eps = 1e-6
    y_log = np.log(np.maximum(y_arr, eps))
    s_log = np.log(np.maximum(s_arr, eps))
    res_log = em_unconstrain_normal(y_log, c_arr, s_log, max_iter=max_iter, tol=tol)
    d_uncon = np.exp(res_log.unconstrained)
    # ensure censored not below threshold
    d_uncon[c_arr] = np.maximum(d_uncon[c_arr], s_arr[c_arr])
    d_uncon[~c_arr] = y_arr[~c_arr]
    return EMResult(mu=res_log.mu, sigma=res_log.sigma, unconstrained=d_uncon, n_iter=res_log.n_iter, converged=res_log.converged)


def projection_detruncation(
    y: Iterable[float],
    censored: Iterable[bool],
    s: Optional[Iterable[float]] = None,
    uplift_cap: float = 3.0,
) -> np.ndarray:
    """Simpler projection-detruncation fallback.

    Estimates uplift factor from ratio of mean observed to mean at-cap on censored rows using
    uncensored distribution moments; caps uplift for stability.
    Assumption: when too few uncensored points exist, applies a conservative 10% uplift to censored rows.
    """
    y_arr = np.asarray(list(y), dtype=float)
    c_arr = np.asarray(list(censored), dtype=bool)
    s_arr = np.asarray(list(s), dtype=float) if s is not None else y_arr.copy()
    out = y_arr.copy().astype(float)
    if not c_arr.any():
        return out
    unc = y_arr[~c_arr]
    if len(unc) >= 2:
        mu = float(np.mean(unc))
        sigma = _safe_sigma(float(np.std(unc)))
        # project censored above threshold by expected exceedance under fitted normal tail
        z = (s_arr[c_arr] - mu) / sigma
        survival = np.clip(stats.norm.sf(z), 1e-6, None)
        mills = stats.norm.pdf(z) / survival
        cond = mu + sigma * mills
        uplift = np.clip(cond / np.maximum(s_arr[c_arr], 1e-6), 1.0, uplift_cap)
        out[c_arr] = np.maximum(s_arr[c_arr] * uplift, s_arr[c_arr])
    else:
        out[c_arr] = s_arr[c_arr] * 1.10
    out[~c_arr] = y_arr[~c_arr]
    return out


__all__ = ["EMResult", "em_unconstrain_normal", "em_unconstrain_lognormal", "projection_detruncation"]
