"""Rolling-origin time-based splits by stay date with embargo gap.

Assumption: forecast_horizon_days defines embargo so train stay dates are strictly before
(test_start - embargo), preventing leakage from near-future outcomes. Splits are by stay date only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import pandas as pd


@dataclass(frozen=True)
class SplitFold:
    """A single rolling-origin fold."""

    fold: int
    train_end_stay: pd.Timestamp
    test_start_stay: pd.Timestamp
    test_end_stay: pd.Timestamp
    embargo_days: int


def rolling_origin_splits(
    stay_dates: Sequence[pd.Timestamp] | pd.Series,
    forecast_horizon_days: int,
    test_window_days: int = 28,
    step_days: int = 28,
    min_train_days: int = 60,
) -> List[SplitFold]:
    """Generate rolling-origin splits by unique stay date.

    Train: stay_date <= train_end_stay.
    Test:  stay_date in [test_start_stay, test_end_stay].
    Embargo gap = forecast_horizon_days => train_end_stay < test_start_stay - embargo.
    Raises ValueError if any stay date appears in both train and test.
    """
    if forecast_horizon_days < 0:
        raise ValueError("forecast_horizon_days must be non-negative")
    dates = pd.to_datetime(pd.Series(list(stay_dates))).dt.normalize().drop_duplicates().sort_values()
    if dates.empty:
        return []
    dmin, dmax = dates.iloc[0], dates.iloc[-1]
    folds: List[SplitFold] = []
    origin = dmin + pd.Timedelta(days=int(min_train_days))
    embargo = pd.Timedelta(days=int(forecast_horizon_days))
    fold = 0
    while True:
        train_end = origin
        test_start = train_end + embargo + pd.Timedelta(days=1)
        test_end = test_start + pd.Timedelta(days=int(test_window_days) - 1)
        if test_start > dmax:
            break
        train_set = set(dates[dates <= train_end].tolist())
        test_set = set(dates[(dates >= test_start) & (dates <= test_end)].tolist())
        overlap = train_set & test_set
        if overlap:
            raise ValueError(f"Stay date appears in both train and test: {sorted(overlap)[:5]}")
        if test_set:
            folds.append(
                SplitFold(
                    fold=fold,
                    train_end_stay=pd.Timestamp(train_end),
                    test_start_stay=pd.Timestamp(test_start),
                    test_end_stay=pd.Timestamp(test_end),
                    embargo_days=int(forecast_horizon_days),
                )
            )
            fold += 1
        origin = origin + pd.Timedelta(days=int(step_days))
    return folds


def apply_split(
    df: pd.DataFrame, fold: SplitFold, stay_col: str = "stay_date"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, test_df) for a fold and assert no stay-date overlap."""
    s = pd.to_datetime(df[stay_col]).dt.normalize()
    train = df[s <= fold.train_end_stay].copy()
    test = df[(s >= fold.test_start_stay) & (s <= fold.test_end_stay)].copy()
    train_dates = set(pd.to_datetime(train[stay_col]).dt.normalize().unique().tolist())
    test_dates = set(pd.to_datetime(test[stay_col]).dt.normalize().unique().tolist())
    overlap = train_dates & test_dates
    if overlap:
        raise ValueError(f"Stay date appears in both train and test: {sorted(overlap)[:5]}")
    return train, test


__all__ = ["SplitFold", "rolling_origin_splits", "apply_split"]
