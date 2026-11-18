"""Look-to-book (LTB) helpers: bot filtering and decomposition.

Assumption: bot behavior manifests as very high request cadence and low reputation; thresholds are
simulator defaults and recorded in code comments, not TBO production values.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd


def filter_bots(
    searches: pd.DataFrame,
    *,
    max_requests_per_minute: int = 30,
    min_reputation: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split searches into (human, bot) based on agent reputation and request cadence.

    Expects columns: `agent_id`, `ts` (datetime-like), `agent_reputation` (0-1) when available.
    If `agent_reputation` missing, defaults to 0.8 (assumed human) for simulator rows.
    Returns (filtered_human_searches, bot_searches) preserving original columns.
    """
    if searches.empty:
        return searches.copy(), searches.copy()

    df = searches.copy()
    df["ts"] = pd.to_datetime(df["ts"])
    if "agent_reputation" not in df.columns:
        df["agent_reputation"] = 0.8

    # cadence: requests per minute per agent in rolling 1-min windows approximated by minute bucket
    df["_minute"] = df["ts"].dt.floor("min")
    cadence = df.groupby(["agent_id", "_minute"]).size().rename("_rpm").reset_index()
    df = df.merge(cadence, on=["agent_id", "_minute"], how="left")

    is_bot = (df["agent_reputation"] < float(min_reputation)) | (
        df["_rpm"] > float(max_requests_per_minute)
    )
    bots = df[is_bot].drop(columns=["_minute", "_rpm"]).copy()
    humans = df[~is_bot].drop(columns=["_minute", "_rpm"]).copy()
    return humans, bots


def bookings_from_ltb(
    searches: float | int | np.ndarray | pd.Series,
    availability_rate: float | np.ndarray | pd.Series,
    win_rate: float | np.ndarray | pd.Series,
) -> np.ndarray:
    """Decompose bookings = searches * availability_rate * win_rate (elementwise)."""
    s = np.asarray(searches, dtype=float)
    a = np.asarray(availability_rate, dtype=float)
    w = np.asarray(win_rate, dtype=float)
    return s * a * w


__all__ = ["filter_bots", "bookings_from_ltb"]
