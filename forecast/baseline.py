"""Baseline forecasters trained on censored observed demand only (never latent evaluation table).

(a) Seasonal naive: same week last year scaled by YoY growth.
(b) Pace model: D_hat = B(t)/h(t) with isotonic h(t) per (segment, source_market, season),
    blended with additive pickup D_hat = B(t)+delta(t) at long lead times.

Assumption: censored final observed demand (confirmed booking count per cell/stay_date) is the target;
pace observations use booking lead times as-of stay date; season=quarter, segment inferred from rate_plan,
source_market defaults to 'IN' when unavailable. Simulator outputs under stated assumptions, not TBO real P&L.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


def _season_q(dt: pd.Series | pd.DatetimeIndex) -> pd.Series:
    return pd.to_datetime(dt).dt.quarter.astype(int)


def _week_key(dt: pd.Series | pd.DatetimeIndex) -> pd.Series:
    return pd.to_datetime(dt).dt.isocalendar().week.astype(int)


@dataclass
class SeasonalNaiveModel:
    """Seasonal naive: same ISO week last year scaled by YoY growth."""

    history: pd.DataFrame
    yoy_growth: float
    fallback_level: float
    y_col: str = "y_censored"

    @classmethod
    def fit(cls, train: pd.DataFrame, y_col: str = "y_censored") -> "SeasonalNaiveModel":
        df = train.copy()
        df["stay_date"] = pd.to_datetime(df["stay_date"]).dt.normalize()
        df["year"] = df["stay_date"].dt.year
        df["week"] = _week_key(df["stay_date"])
        cur = df[["cell_id", "week", "year", y_col]].rename(columns={y_col: "y"})
        prev = cur.copy()
        prev["year"] = prev["year"] + 1
        prev = prev.rename(columns={"y": "y_prev"})
        merged = cur.merge(prev, on=["cell_id", "week", "year"], how="inner")
        mask = merged["y_prev"] > 0
        if mask.any():
            ratios = (merged.loc[mask, "y"] / merged.loc[mask, "y_prev"]).replace([np.inf, -np.inf], np.nan).dropna()
            yoy = float(np.clip(ratios.median() if len(ratios) else 1.0, 0.5, 2.0))
        else:
            yoy = 1.0
        fallback = float(max(df[y_col].median() if len(df) else 1.0, 1e-6))
        hist = df[["cell_id", "week", "year", y_col]].copy()
        return cls(history=hist, yoy_growth=yoy, fallback_level=fallback, y_col=y_col)

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        df = test.copy()
        df["stay_date"] = pd.to_datetime(df["stay_date"]).dt.normalize()
        df["week"] = _week_key(df["stay_date"])
        df["year"] = df["stay_date"].dt.year
        lookup = (
            self.history.rename(columns={self.y_col: "y_hist"})
            .groupby(["cell_id", "week", "year"], as_index=False)["y_hist"]
            .mean()
        )
        lookup["year_match"] = lookup["year"] + 1
        # preserve test row order and cardinality
        df = df.reset_index(drop=True)
        out = df.merge(
            lookup[["cell_id", "week", "year_match", "y_hist"]],
            left_on=["cell_id", "week", "year"],
            right_on=["cell_id", "week", "year_match"],
            how="left",
        )
        # if duplicates somehow remain, keep first per original row by reindexing via group on original index proxy
        if len(out) != len(df):
            out = out.groupby(level=0).first() if out.index.name is not None else out.drop_duplicates(subset=df.index.names or None)
        cell_med = self.history.groupby("cell_id")[self.y_col].median()
        preds = []
        for cid, val in zip(out["cell_id"], out["y_hist"]):
            if pd.notna(val):
                preds.append(float(val) * self.yoy_growth)
            else:
                preds.append(float(cell_med.get(cid, self.fallback_level)))
        return np.asarray(preds, dtype=float)


@dataclass
class PaceCurve:
    """Monotone booking curve h(t) and additive pickup delta."""

    t_days: np.ndarray
    h: np.ndarray
    delta: float


class PacePickupModel:
    """Pace model D_hat = w*B(t)/h(t) + (1-w)*(B(t)+delta(t))."""

    def __init__(self, long_lead_days: int = 45, blend_mid_days: int = 21) -> None:
        self.long_lead_days = int(long_lead_days)
        self.blend_mid_days = int(blend_mid_days)
        self.curves: Dict[Tuple, PaceCurve] = {}
        self.global_curve: Optional[PaceCurve] = None

    @staticmethod
    def _fit_curve(group: pd.DataFrame) -> PaceCurve:
        g = group.copy()
        g = g[g["y_final"] > 0]
        if g.empty:
            t = np.array([60.0, 30.0, 14.0, 7.0, 0.0])
            h = np.array([0.2, 0.5, 0.75, 0.9, 1.0])
            return PaceCurve(t, h, 0.0)
        g["frac"] = np.clip(g["booked_by_t"].astype(float) / g["y_final"].astype(float), 0.0, 1.0)
        agg = g.groupby("lead_time_days", as_index=False)["frac"].mean()
        t_obs = agg["lead_time_days"].to_numpy(dtype=float)
        f_obs = agg["frac"].to_numpy(dtype=float)
        order = np.argsort(-t_obs)  # long lead -> short lead
        t_s = t_obs[order]
        f_s = f_obs[order]
        x = (t_s.max() - t_s)  # increasing as we approach arrival
        ir = IsotonicRegression(increasing=True, y_min=0.001, y_max=1.0)
        h_iso = ir.fit_transform(x, f_s)
        # ensure exact 1.0 at t==0 if present in noise-free data
        for i, tv in enumerate(t_s):
            if np.isclose(tv, 0.0):
                h_iso[i] = 1.0
        long_mask = g["lead_time_days"].astype(float) >= 30.0
        if long_mask.any():
            delta = float(np.mean(np.maximum(g.loc[long_mask, "y_final"] - g.loc[long_mask, "booked_by_t"], 0.0)))
        else:
            delta = float(np.mean(np.maximum(g["y_final"] - g["booked_by_t"], 0.0)))
        return PaceCurve(t_s.astype(float), h_iso.astype(float), float(max(delta, 0.0)))

    def fit(self, pace_obs: pd.DataFrame) -> "PacePickupModel":
        required = {"segment", "source_market", "season", "lead_time_days", "booked_by_t", "y_final"}
        missing = required - set(pace_obs.columns)
        if missing:
            raise ValueError(f"pace_obs missing columns: {missing}")
        df = pace_obs.copy()
        self.global_curve = self._fit_curve(df)
        for key, grp in df.groupby(["segment", "source_market", "season"]):
            self.curves[(str(key[0]), str(key[1]), int(key[2]))] = self._fit_curve(grp)
        for key, grp in df.groupby(["segment", "season"]):
            k2 = (str(key[0]), "*", int(key[1]))
            if k2 not in self.curves:
                self.curves[k2] = self._fit_curve(grp)
        return self

    def _get_curve(self, segment: str, source_market: str, season: int) -> PaceCurve:
        for key in [(str(segment), str(source_market), int(season)), (str(segment), "*", int(season))]:
            if key in self.curves:
                return self.curves[key]
        assert self.global_curve is not None
        return self.global_curve

    def _h_at(self, curve: PaceCurve, t: float) -> float:
        t_s = curve.t_days
        h_s = curve.h
        if len(t_s) == 0:
            return 1.0
        # interpolate over ascending t
        order = np.argsort(t_s)
        ts = t_s[order]
        hs = h_s[order]
        h = float(np.interp(float(t), ts, hs))
        return float(np.clip(h, 1e-6, 1.0))

    def predict_row(
        self, booked_by_t: float, lead_time_days: float, segment: str, source_market: str, season: int
    ) -> float:
        curve = self._get_curve(segment, source_market, int(season))
        t = float(max(lead_time_days, 0.0))
        h = self._h_at(curve, t)
        b = float(booked_by_t)
        pace = b / max(h, 1e-9)
        additive = b + float(curve.delta)
        if t >= self.long_lead_days:
            w = 0.2
        elif t <= self.blend_mid_days:
            w = 1.0
        else:
            w = 1.0 - 0.8 * (t - self.blend_mid_days) / max(self.long_lead_days - self.blend_mid_days, 1)
        yhat = w * pace + (1.0 - w) * additive
        return float(max(yhat, b, 0.0))

    def predict(self, obs: pd.DataFrame) -> np.ndarray:
        preds = []
        for _, row in obs.iterrows():
            preds.append(
                self.predict_row(
                    booked_by_t=float(row["booked_by_t"]),
                    lead_time_days=float(row["lead_time_days"]),
                    segment=str(row.get("segment", "leisure")),
                    source_market=str(row.get("source_market", "IN")),
                    season=int(row.get("season", 1)),
                )
            )
        return np.asarray(preds, dtype=float)


def build_censored_demand_panel(history_dir: str = "data/history") -> pd.DataFrame:
    """Build censored observed-demand panel per (cell_id, stay_date) from booking history only."""
    b = pd.read_parquet(f"{history_dir}/booking.parquet")
    b = b.copy()
    if "status" in b.columns:
        b = b[b["status"] == "confirmed"]
    b["stay_date"] = pd.to_datetime(b["checkin"]).dt.normalize()
    # cell proxy from available censored fields
    b["cell_id"] = b["hotel_id"].astype(str) + "|" + b["rate_plan"].astype(str)
    panel = b.groupby(["cell_id", "stay_date"], as_index=False).agg(y_censored=("booking_id", "count"))
    # grouping features for baselines (best-effort from censored fields)
    meta = b.sort_values("ts_booked").groupby(["cell_id", "stay_date"]).tail(1)[
        ["cell_id", "stay_date", "rate_plan"]
    ]
    panel = panel.merge(meta, on=["cell_id", "stay_date"], how="left")
    panel["segment"] = np.where(panel["rate_plan"].astype(str).str.contains("CORP", na=False), "business", "leisure")
    panel["source_market"] = "IN"
    panel["season"] = _season_q(panel["stay_date"])
    return panel[["cell_id", "stay_date", "y_censored", "segment", "source_market", "season", "rate_plan"]]


def build_pace_observations(history_dir: str = "data/history") -> pd.DataFrame:
    """Build pace observations (booked_by_t vs final censored y) from booking history only."""
    b = pd.read_parquet(f"{history_dir}/booking.parquet")
    b = b.copy()
    if "status" in b.columns:
        b = b[b["status"] == "confirmed"]
    b["stay_date"] = pd.to_datetime(b["checkin"]).dt.normalize()
    b["cell_id"] = b["hotel_id"].astype(str) + "|" + b["rate_plan"].astype(str)
    final = b.groupby(["cell_id", "stay_date"], as_index=False).agg(y_final=("booking_id", "count"))
    b = b.merge(final, on=["cell_id", "stay_date"], how="left")
    b["segment"] = np.where(b["rate_plan"].astype(str).str.contains("CORP", na=False), "business", "leisure")
    b["source_market"] = "IN"
    b["season"] = _season_q(b["stay_date"])
    # booked_by_t: bookings with lead_time_days >= t are on books t days before arrival
    # create observations at each booking's lead time (cumulative known at that as_of)
    b["lead_time_days"] = b["lead_time_days"].astype(float).clip(lower=0)
    # cumulative booked by t for that row: count of bookings for same cell/stay with lead >= t
    # efficient approximation: for each (cell,stay,t) compute via sort and reverse cumsum
    obs_rows = []
    for (cid, sdate), grp in b.groupby(["cell_id", "stay_date"]):
        y_final = float(len(grp))
        leads = np.sort(grp["lead_time_days"].to_numpy(dtype=float))
        # unique t points
        for t in np.unique(leads):
            booked_by_t = float(np.sum(leads >= t))
            seg = str(grp["segment"].iloc[0])
            sm = str(grp["source_market"].iloc[0])
            season = int(grp["season"].iloc[0])
            obs_rows.append(
                {
                    "cell_id": cid,
                    "stay_date": sdate,
                    "segment": seg,
                    "source_market": sm,
                    "season": season,
                    "lead_time_days": float(t),
                    "booked_by_t": booked_by_t,
                    "y_final": y_final,
                }
            )
    return pd.DataFrame(obs_rows)


__all__ = [
    "SeasonalNaiveModel",
    "PacePickupModel",
    "PaceCurve",
    "build_censored_demand_panel",
    "build_pace_observations",
]
