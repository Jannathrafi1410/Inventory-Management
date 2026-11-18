"""Beta-binomial net-demand (cancellation) layer fitted per segment x lead-time bucket.

Models cancellations as Binomial(B, p) with p ~ Beta(alpha, beta) per (segment, lead_bucket),
yielding net demand = booked - cancelled. Provides mean/variance and sampling for scenario use.

Assumption: booking counts are censored observed bookings; cancellation probabilities are fit from
historical cancellation rates when available, else fall back to configured lead-time bucket priors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import load_assumptions


def lead_bucket(lead_time_days: float | int) -> str:
    lt = float(lead_time_days)
    if lt >= 60:
        return "gte_60"
    if lt >= 30:
        return "d30_59"
    if lt >= 7:
        return "d7_29"
    return "lt_7"


@dataclass
class BetaBinomialParams:
    """Beta prior parameters for cancellation probability."""

    alpha: float
    beta: float
    n_obs: int


class CancellationModel:
    """Beta-binomial cancellation model keyed by (segment, lead_bucket)."""

    def __init__(self, prior_strength: float = 20.0) -> None:
        self.prior_strength = float(prior_strength)
        self.params: Dict[Tuple[str, str], BetaBinomialParams] = {}
        self._load_priors()

    def _load_priors(self) -> None:
        cfg = load_assumptions()
        buckets = cfg.cancellation.lead_time_dependent_probabilities
        for bucket_name, node in buckets.items():
            if bucket_name == "description":
                continue
            try:
                p = float(getattr(node, "value", node["value"] if isinstance(node, dict) else 0.1))
            except Exception:
                continue
            a0 = max(p * self.prior_strength, 0.5)
            b0 = max((1.0 - p) * self.prior_strength, 0.5)
            # store as generic bucket prior (segment '*')
            self.params[("*", bucket_name)] = BetaBinomialParams(a0, b0, 0)

    def fit(self, history_dir: str = "data/history") -> "CancellationModel":
        """Fit/update Beta parameters from censored booking/cancellation history."""
        h = Path(history_dir)
        bpath = h / "booking.parquet"
        cpath = h / "cancellation.parquet"
        if not bpath.exists() or not cpath.exists():
            return self
        b = pd.read_parquet(bpath)
        c = pd.read_parquet(cpath)
        b = b.copy()
        b["lead_bucket"] = b["lead_time_days"].apply(lead_bucket)
        # segment proxy consistent with baselines
        b["segment"] = np.where(b["rate_plan"].astype(str).str.contains("CORP", na=False), "business", "leisure")
        cancelled = set(c["booking_id"].astype(str).tolist()) if "booking_id" in c.columns else set()
        b["is_cancelled"] = b["booking_id"].astype(str).isin(cancelled).astype(int)
        grp = b.groupby(["segment", "lead_bucket"])["is_cancelled"].agg(["sum", "count"]).reset_index()
        for _, row in grp.iterrows():
            seg = str(row["segment"])
            lb = str(row["lead_bucket"])
            canc = float(row["sum"])
            n = float(row["count"])
            prior = self.params.get(("*", lb), BetaBinomialParams(2.0, 18.0, 0))
            a = prior.alpha + canc
            bpar = prior.beta + (n - canc)
            self.params[(seg, lb)] = BetaBinomialParams(float(a), float(bpar), int(n))
        return self

    def _get(self, segment: str, lb: str) -> BetaBinomialParams:
        if (segment, lb) in self.params:
            return self.params[(segment, lb)]
        if ("*", lb) in self.params:
            return self.params[("*", lb)]
        # global fallback
        return BetaBinomialParams(2.0, 18.0, 0)

    def mean_cancel_prob(self, segment: str, lead_time_days: float | int) -> float:
        lb = lead_bucket(lead_time_days)
        p = self._get(segment, lb)
        return float(p.alpha / (p.alpha + p.beta))

    def net_demand_mean_var(self, booked: float, segment: str, lead_time_days: float | int) -> Tuple[float, float]:
        """Return (mean, var) of net demand after Beta-binomial cancellations."""
        n = max(float(booked), 0.0)
        lb = lead_bucket(lead_time_days)
        par = self._get(segment, lb)
        a, b = par.alpha, par.beta
        p_mean = a / (a + b)
        # Var[cancelled] for Beta-Binomial
        var_c = n * p_mean * (1 - p_mean) * (a + b + n) / (a + b + 1.0) if (a + b + 1.0) > 0 else n * p_mean * (1 - p_mean)
        mean_net = n * (1.0 - p_mean)
        return float(mean_net), float(var_c)

    def sample_net_demand(
        self, booked: np.ndarray | float, segment: str, lead_time_days: float | int, rng: np.random.Generator
    ) -> np.ndarray:
        """Sample net demand (booked - cancelled) via Beta-binomial draws."""
        b_arr = np.asarray(booked, dtype=float)
        lb = lead_bucket(lead_time_days)
        par = self._get(segment, lb)
        p = rng.beta(par.alpha, par.beta, size=b_arr.shape)
        canc = rng.binomial(np.maximum(np.round(b_arr).astype(int), 0), np.clip(p, 0.0, 1.0))
        net = np.maximum(b_arr - canc, 0.0)
        return net.astype(float)


__all__ = ["CancellationModel", "BetaBinomialParams", "lead_bucket"]
