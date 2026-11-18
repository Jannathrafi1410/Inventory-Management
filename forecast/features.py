"""Point-in-time feature builder for demand distribution modeling.

All features are reconstructible as of the provided decision timestamp (`as_of_ts`); inputs are
filtered to `as_of_ts <= decision_ts` and only censored history tables are used.

Assumption: observations are keyed by (cell_id, as_of_ts) from `data/history/quote.parquet` joined
to `search_event.parquet` and mapped to simulator candidate cells via
(destination_city, source_market, rate_plan, contract_type). Lagged aggregates use completed weeks
as of decision day; holiday/event/FX signals are deterministic approximations from stay/source/geo.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sim.cells import CandidateCell, get_candidate_cells
from sim.look_to_book import filter_bots


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for point-in-time feature construction."""

    history_dir: str = "data/history"
    lag_weeks: Tuple[int, ...] = (4, 6, 8, 10, 12)
    pace_plan_days: Tuple[int, ...] = (60, 30, 14, 7)


def _cell_lookup() -> Dict[Tuple[str, str, str, str], CandidateCell]:
    cells = get_candidate_cells()
    return {(c.destination_city, c.source_market, c.rate_plan, c.contract_type): c for c in cells}


def _holiday_flags(stay_date: pd.Timestamp, source_market: str) -> Dict[str, int]:
    """Deterministic calendar/holiday approximations (no external leakage).

    Assumption: flags are rule-based proxies (fixed/approx windows) sufficient for simulator features.
    """
    m = int(stay_date.month)
    d = int(stay_date.day)
    dow = int(stay_date.weekday())  # Mon=0
    # long weekend: Fri/Mon adjacent in effect captured via weekend-like stays
    is_long_weekend = int(dow in (4, 0) or (dow == 6 and d <= 7))
    # school breaks (approx): summer/winter windows for IN/EU source markets
    is_school_break = int((m in (5, 6, 7) and source_market.startswith("IN")) or (m in (12, 1) and True))
    # Ramadan/Eid approximate windows by month (sim proxy; varies yearly in reality)
    is_ramadan = int(m in (3, 4))
    is_eid = int((m == 4 and d <= 15) or (m == 6 and d <= 10))
    # source-market holidays (generic proxies)
    sm_holiday = int(
        (source_market.startswith("IN") and ((m == 1 and d in (26,)) or (m == 8 and d == 15) or (m == 10 and d <= 10)))
        or (source_market.startswith("EU") and m == 12 and d >= 20)
        or (source_market.startswith("GCC") and m in (12, 1))
    )
    return {
        "is_long_weekend": is_long_weekend,
        "is_school_break": is_school_break,
        "is_ramadan": is_ramadan,
        "is_eid": is_eid,
        "is_source_market_holiday": sm_holiday,
    }


def _fx_factor(as_of_ts: pd.Timestamp, source_market: str) -> float:
    """Deterministic FX proxy as-of timestamp (no future info). Assumption: mild seasonal drift."""
    t = as_of_ts.value // 10**9
    base = {"IN": 1.0, "EU": 1.02, "GCC": 0.99, "APAC": 1.01}.get(source_market.split("-")[0], 1.0)
    wiggle = 0.01 * np.sin(2 * np.pi * (as_of_ts.timetuple().tm_yday / 365.25))
    return float(base * (1.0 + wiggle))


def load_base_observations(history_dir: str = "data/history") -> pd.DataFrame:
    """Load censored per-(cell_id, as_of_ts, stay_date) observations from history tables."""
    h = Path(history_dir)
    se = pd.read_parquet(h / "search_event.parquet")
    q = pd.read_parquet(h / "quote.parquet")
    ic = pd.read_parquet(h / "inventory_commit.parquet")

    se["ts"] = pd.to_datetime(se["ts"])
    se["as_of_ts"] = pd.to_datetime(se["as_of_ts"])
    se["checkin_date"] = pd.to_datetime(se["checkin_date"]).dt.normalize()
    q["as_of_ts"] = pd.to_datetime(q["as_of_ts"])
    ic["as_of_ts"] = pd.to_datetime(ic["as_of_ts"])

    m = se.merge(q, on=["search_id", "as_of_ts"], how="inner", suffixes=("_se", "_q"))
    # resolve hotel_id collision: prefer quote
    if "hotel_id_q" in m.columns:
        m["hotel_id"] = m["hotel_id_q"]
    cm = _cell_lookup()
    def map_cell(r: pd.Series) -> Optional[str]:
        c = cm.get((r["destination_city"], r["source_market"], r["rate_plan"], r["contract_type"]))
        return c.cell_id if c else None
    m["cell_id"] = m.apply(map_cell, axis=1)
    m = m[m["cell_id"].notna()].copy()

    # inventory commit per (hotel_id, stay_date_range context approximated by as_of day?) use as_of join
    # ground-truth aligned daily commits exist; join by hotel/supplier/contract/as_of
    ic = ic.rename(columns={"as_of_ts": "as_of_ts_ic"})
    m = m.merge(
        ic[["supplier_id", "hotel_id", "contract_type", "units_sold", "units", "as_of_ts_ic", "stay_date_range"]],
        left_on=["supplier_id", "hotel_id", "contract_type", "as_of_ts"],
        right_on=["supplier_id", "hotel_id", "contract_type", "as_of_ts_ic"],
        how="left",
    )
    m["observed_sales"] = m["units_sold"].fillna(0).astype(float)
    m["capacity_obs"] = m["units"].fillna(m["rooms_left"].fillna(0) + m["observed_sales"]).astype(float)
    m["sell_out_obs"] = m["is_sold_out"].astype(bool)
    m["stay_date"] = pd.to_datetime(m["checkin_date"]).dt.normalize()
    m["decision_ts"] = pd.to_datetime(m["as_of_ts"])
    m["lead_time_days"] = (m["stay_date"] - m["decision_ts"].dt.normalize()).dt.days.clip(lower=0)
    # attach cell static attrs
    cells = {c.cell_id: c for c in get_candidate_cells()}
    m["group"] = m["cell_id"].map(lambda x: cells[x].region_group)
    m["segment"] = m["cell_id"].map(lambda x: cells[x].segment)
    m["destination_city"] = m["cell_id"].map(lambda x: cells[x].destination_city)
    m["source_market_cell"] = m["cell_id"].map(lambda x: cells[x].source_market)
    m["p"] = m["cell_id"].map(lambda x: cells[x].p)
    m["c_d"] = m["cell_id"].map(lambda x: cells[x].c_d)
    m["c_b"] = m["cell_id"].map(lambda x: cells[x].c_b)
    m["k"] = m["cell_id"].map(lambda x: cells[x].k)
    m["r"] = m["cell_id"].map(lambda x: cells[x].r)
    m["a"] = m["cell_id"].map(lambda x: cells[x].a)
    m["g"] = m["cell_id"].map(lambda x: cells[x].g)
    m["cap"] = m["cell_id"].map(lambda x: (cells[x].supplier_cap if cells[x].supplier_cap is not None else cells[x].capacity))
    m["product_type"] = m["cell_id"].map(lambda x: cells[x].product_type)
    return m


def build_point_in_time_features(
    decision_ts: pd.Timestamp | str | None = None,
    history_dir: str = "data/history",
    config: Optional[FeatureConfig] = None,
) -> pd.DataFrame:
    """Build feature matrix for all observations with as_of_ts <= decision_ts (or all if None).

    Returns one row per (cell_id, as_of_ts/stay context) with censored, leakage-safe features.
    """
    cfg = config or FeatureConfig(history_dir=history_dir)
    base = load_base_observations(history_dir=cfg.history_dir)
    if decision_ts is not None:
        dts = pd.to_datetime(decision_ts)
        base = base[base["decision_ts"] <= dts].copy()
    if base.empty:
        return base

    # calendar features
    base["stay_month"] = base["stay_date"].dt.month.astype(int)
    base["stay_week"] = base["stay_date"].dt.isocalendar().week.astype(int)
    base["stay_dow"] = base["stay_date"].dt.weekday.astype(int)
    hol = base.apply(lambda r: _holiday_flags(r["stay_date"], str(r["source_market_cell"])), axis=1, result_type="expand")
    base = pd.concat([base.reset_index(drop=True), hol.reset_index(drop=True)], axis=1)

    # lead time / pace
    base["pace_index"] = base["observed_sales"] / (base["capacity_obs"].replace(0, np.nan))
    base["pace_index"] = base["pace_index"].fillna(0.0).clip(0, 2)
    # pace vs plan: compare to expected fraction by lead bucket (simple heuristic plan curve)
    def plan_frac(lt: float) -> float:
        # assumption: linear-ish build from 0 at 120d to 1 at 0d
        return float(np.clip(1.0 - (lt / 120.0), 0.05, 1.0))
    base["pace_vs_plan"] = base["pace_index"] / base["lead_time_days"].apply(plan_frac).replace(0, np.nan)
    base["pace_vs_plan"] = base["pace_vs_plan"].fillna(1.0).clip(0, 5)

    # price and rank
    base["price"] = base["our_sell_price"].astype(float)
    base["price_rank"] = base.groupby(["destination_city", "decision_ts"])["price"].rank(method="average", pct=True).fillna(0.5)
    base["price_rel_cd"] = (base["price"] / base["c_d"].replace(0, np.nan)).fillna(1.0)

    # sell-out rate at lead time (historical, by cell/lead bucket as-of prior observations)
    base["lead_bucket"] = pd.cut(base["lead_time_days"], bins=[-1, 7, 14, 30, 60, 10**6], labels=["0-7", "8-14", "15-30", "31-60", "60+"])
    base = base.sort_values(["cell_id", "decision_ts"])
    # rolling historical sell-out rate per (cell, lead_bucket) using past only
    base["sellout_rate_lead"] = (
        base.groupby(["cell_id", "lead_bucket"], observed=True)["sell_out_obs"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=[0, 1], drop=True)
    )
    base["sellout_rate_lead"] = base.groupby(["cell_id"])["sellout_rate_lead"].transform(lambda s: s.fillna(s.expanding().mean()))
    base["sellout_rate_lead"] = base["sellout_rate_lead"].fillna(base["sell_out_obs"].mean()).clip(0, 1)

    # own lags and YoY (per cell, by stay_date order observed over time; use past decisions only)
    base = base.sort_values(["cell_id", "stay_date", "decision_ts"])
    base["own_lag_1"] = base.groupby("cell_id")["observed_sales"].shift(1)
    base["own_lag_7"] = base.groupby("cell_id")["observed_sales"].shift(7)
    base["own_lag_28"] = base.groupby("cell_id")["observed_sales"].shift(28)
    for col in ["own_lag_1", "own_lag_7", "own_lag_28"]:
        base[col] = base[col].fillna(base.groupby("cell_id")["observed_sales"].transform("median")).fillna(0.0)
    # YoY: same stay month/day approx last year within cell
    base["stay_md"] = base["stay_date"].dt.strftime("%m-%d")
    yoy = base[["cell_id", "stay_md", "stay_date", "observed_sales"]].copy()
    yoy["year"] = yoy["stay_date"].dt.year
    yoy_prev = yoy.rename(columns={"observed_sales": "yoy_sales", "year": "year_prev"})
    yoy_prev["year_match"] = yoy_prev["year_prev"] + 1
    base["year"] = base["stay_date"].dt.year
    base = base.merge(
        yoy_prev[["cell_id", "stay_md", "year_match", "yoy_sales"]],
        left_on=["cell_id", "stay_md", "year"],
        right_on=["cell_id", "stay_md", "year_match"],
        how="left",
    )
    base["yoy_lag"] = base["yoy_sales"].fillna(base["own_lag_28"])

    # lagged search aggregates at 4/6/8/10/12 weeks (bot-filtered), point-in-time
    se = pd.read_parquet(Path(cfg.history_dir) / "search_event.parquet")
    se["ts"] = pd.to_datetime(se["ts"])
    se["as_of_ts"] = pd.to_datetime(se["as_of_ts"])
    se["checkin_date"] = pd.to_datetime(se["checkin_date"]).dt.normalize()
    # approximate agent_reputation absent -> default human
    if "agent_reputation" not in se.columns:
        se["agent_reputation"] = 0.8
    se_h, _ = filter_bots(se, max_requests_per_minute=30, min_reputation=0.2)
    se_h["week"] = se_h["as_of_ts"].dt.to_period("W").dt.start_time
    # map searches to cell
    cm = _cell_lookup()
    q = pd.read_parquet(Path(cfg.history_dir) / "quote.parquet")
    q["as_of_ts"] = pd.to_datetime(q["as_of_ts"])
    sem = se_h.merge(q[["search_id", "as_of_ts", "rate_plan", "contract_type"]], on=["search_id", "as_of_ts"], how="left")
    sem["cell_id"] = sem.apply(lambda r: (cm.get((r["destination_city"], r["source_market"], r.get("rate_plan"), r.get("contract_type"))).cell_id if cm.get((r["destination_city"], r["source_market"], r.get("rate_plan"), r.get("contract_type"))) else None), axis=1)
    sem = sem[sem["cell_id"].notna()]
    weekly = sem.groupby(["cell_id", "week"]).size().rename("searches").reset_index().sort_values(["cell_id", "week"])
    # pivot-like merge for each observation's decision week
    base["decision_week"] = base["decision_ts"].dt.to_period("W").dt.start_time
    for w in cfg.lag_weeks:
        lag = weekly.copy()
        lag["decision_week"] = lag["week"] + pd.to_timedelta(w * 7, unit="D")
        lag = lag.rename(columns={"searches": f"search_lag_{w}w"})
        base = base.merge(lag[["cell_id", "decision_week", f"search_lag_{w}w"]], on=["cell_id", "decision_week"], how="left")
        base[f"search_lag_{w}w"] = base[f"search_lag_{w}w"].fillna(0.0)

    # event decay features (distance from stay to representative event peak proxied by cell window mid)
    cells_map = {c.cell_id: c for c in get_candidate_cells()}
    def event_decay(row: pd.Series) -> float:
        c = cells_map[row["cell_id"]]
        mid = pd.Timestamp(c.checkin_start + (c.checkin_end - c.checkin_start) / 2)
        dist = abs((pd.Timestamp(row["stay_date"]) - mid).days)
        mag = 0.45 if c.segment == "event" else (0.15 if c.segment == "leisure" else 0.08)
        return float(mag * np.exp(-dist / 20.0))
    base["event_decay"] = base.apply(event_decay, axis=1)

    # FX
    base["fx_factor"] = base.apply(lambda r: _fx_factor(pd.Timestamp(r["decision_ts"]), str(r["source_market_cell"])), axis=1)

    # cleanup helper cols
    drop_cols = [c for c in ["year_prev", "year_match", "stay_md", "decision_week", "as_of_ts_ic", "stay_date_range", "lead_bucket", "year"] if c in base.columns]
    base = base.drop(columns=drop_cols, errors="ignore")
    # ensure no future data: all from rows with as_of_ts <= decision_ts by construction
    return base.sort_values(["cell_id", "decision_ts"]).reset_index(drop=True)


FEATURE_COLUMNS: List[str] = [
    "lead_time_days", "stay_month", "stay_week", "stay_dow",
    "is_long_weekend", "is_school_break", "is_ramadan", "is_eid", "is_source_market_holiday",
    "pace_index", "pace_vs_plan", "price", "price_rank", "price_rel_cd",
    "sellout_rate_lead", "own_lag_1", "own_lag_7", "own_lag_28", "yoy_lag",
    "search_lag_4w", "search_lag_6w", "search_lag_8w", "search_lag_10w", "search_lag_12w",
    "event_decay", "fx_factor",
]

__all__ = ["FeatureConfig", "build_point_in_time_features", "load_base_observations", "FEATURE_COLUMNS"]
