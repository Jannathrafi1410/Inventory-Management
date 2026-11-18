"""Synthetic data generator for TBO Inventory Optimization.

Assumption: simulator outputs are under stated assumptions, not TBO real P&L. History contains only
censored observations (observed_sales = min(D, capacity)); latent demand lives only in ground_truth.
Per-cell daily realizations over a fixed horizon produce required event tables with as_of_ts.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from config.settings import get_rng, load_assumptions
from sim.cells import CandidateCell, get_candidate_cells
from sim.schema import SCHEMAS


def _month_seasonality(month: int, segment: str) -> float:
    # Assumption: mild sinusoidal seasonality by month; event/leisure peak in winter-summer windows.
    base = 1.0 + 0.15 * np.sin(2 * np.pi * (month - 1) / 12.0)
    if segment in {"leisure", "event"} and month in {12, 1, 6, 7}:
        base *= 1.10
    if segment == "business" and month in {2, 5, 9, 11}:
        base *= 1.08
    return float(base)


def _source_factor(source_market: str) -> float:
    # Assumption: corridor/source-market demand multipliers for simulator diversity.
    factors = {"IN": 1.0, "IN-BOM": 1.05, "IN-DEL": 1.02, "EU": 0.95, "GCC": 1.03, "APAC": 0.97}
    return float(factors.get(source_market, 1.0))


def _event_lift(cell: CandidateCell, stay_date: date) -> float:
    # Assumption: event lift decays with distance from stay window midpoint; magnitude from segment.
    mid = cell.checkin_start + (cell.checkin_end - cell.checkin_start) / 2
    dist = abs((stay_date - mid).days)
    if cell.segment == "event":
        mag = 0.45
    elif cell.segment == "leisure":
        mag = 0.15
    else:
        mag = 0.08
    decay = np.exp(-dist / 20.0)
    return float(1.0 + mag * decay)


def _beta_booking_curve(rng: np.random.Generator, n: int, lead_min: int, lead_max: int, segment: str) -> np.ndarray:
    # Assumption: leisure books 60-120d out (skew early), business 0-14d (skew late), event/mixed between.
    if segment == "business":
        a, b = 2.0, 5.0
        lo, hi = 0, max(14, lead_max)
    elif segment == "leisure":
        a, b = 5.0, 2.0
        lo, hi = max(30, lead_min), max(120, lead_max)
    else:
        a, b = 3.0, 3.0
        lo, hi = lead_min, lead_max
    lo = max(0, int(lo))
    hi = max(lo + 1, int(hi))
    u = rng.beta(a, b, size=n)
    return (lo + u * (hi - lo)).astype(int)


def _cancellation_prob(days_before: int, buckets: Dict[str, Any]) -> float:
    # Use configured lead-time buckets when available; fallback to monotone curve.
    if days_before >= 60:
        key = "gte_60"
    elif days_before >= 30:
        key = "d30_59"
    elif days_before >= 7:
        key = "d7_29"
    else:
        key = "lt_7"
    node = buckets.get(key)
    if isinstance(node, dict) and "value" in node:
        # defensive if raw dict leaks; normally AssumptionValue
        try:
            return float(node["value"])
        except Exception:
            pass
    # node may be AssumptionValue-like via loader in config; here accept numeric fallback
    try:
        return float(getattr(node, "value"))
    except Exception:
        pass
    return float(np.clip(0.20 - 0.0025 * max(days_before, 0), 0.02, 0.20))


def _stable_id(prefix: str, *parts: Any) -> str:
    raw = "|".join([prefix] + [str(p) for p in parts]).encode("utf-8")
    return f"{prefix}_{hashlib.sha1(raw).hexdigest()[:12]}"


def generate_synthetic_data(master_seed: int = 42, out_dir: str | Path = "data") -> Dict[str, pd.DataFrame]:
    """Generate censored history and latent ground_truth tables and write them as parquet.

    Returns dict of table_name -> DataFrame for convenience (tests). Writes:
      - data/history/*.parquet (censored event tables)
      - data/ground_truth.parquet (latent demand; evaluation only)
    """
    out_path = Path(out_dir)
    hist_dir = out_path / "history"
    hist_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_assumptions()
    group_sigma = float(cfg.simulator.group_sigma.value)
    target_margin = float(cfg.market.dynamic_margin.value)  # 0.055
    elasticity = cfg.market.elasticities
    eps_leisure = float(elasticity["leisure"].value) if "leisure" in elasticity else -2.5
    eps_compress = float(elasticity["compression"].value) if "compression" in elasticity else -0.8
    cancel_buckets = cfg.cancellation.lead_time_dependent_probabilities

    cells = get_candidate_cells()
    # common horizon so group shocks align across cells (enables cross-cell correlation measurement)
    horizon_days = 90
    horizon_end = date(2026, 12, 31)

    rng = get_rng(master_seed, "sim:global")
    tables: Dict[str, List[Dict[str, Any]]] = {k: [] for k in SCHEMAS.keys()}
    gt_rows: List[Dict[str, Any]] = []

    # precompute group shock series aligned by day index (shared per region group)
    groups = sorted({c.region_group for c in cells})
    group_shock_by_group: Dict[str, np.ndarray] = {}
    for g in groups:
        grp_rng = get_rng(master_seed, f"group:{g}")
        group_shock_by_group[g] = grp_rng.normal(0.0, group_sigma, size=horizon_days)

    for c in cells:
        cell_rng = get_rng(master_seed, f"cell:{c.cell_id}")
        group_shock = group_shock_by_group[c.region_group]
        # add small cell-specific component but keep group signal dominant for within-group correlation
        idio = cell_rng.normal(0.0, group_sigma * 0.20, size=horizon_days)

        # anchor calibration: choose p0 and drift so mean realized dynamic margin ~5.5% and observed sell-out near (1-a)
        p0 = c.p
        eps = eps_compress if c.segment in {"event", "business"} else eps_leisure

        stay_rep = c.checkin_start + (c.checkin_end - c.checkin_start) / 2
        for d_idx in range(horizon_days):
            as_of = datetime.combine(horizon_end - timedelta(days=(horizon_days - 1 - d_idx)), datetime.min.time())
            stay_date = stay_rep  # representative stay date for the cell window
            # demand components
            base = cell_rng.lognormal(mean=np.log(max(c.base_demand_mean, 1e-6)), sigma=c.demand_sigma)
            season = _month_seasonality(stay_date.month, c.segment) * _source_factor(c.source_market)
            event = _event_lift(c, stay_date)
            corridor_trend = 1.0 + 0.0008 * d_idx  # mild trend
            noise_mult = float(np.exp(group_shock[d_idx] + idio[d_idx]))
            latent_d = base * season * event * corridor_trend * noise_mult
            # price response
            latent_d = latent_d * ((c.p / max(p0, 1e-6)) ** eps)
            latent_d = float(max(latent_d, 0.0))

            capacity = int(c.capacity)
            observed_sales = int(min(np.floor(latent_d + 1e-9), capacity))
            sell_out = bool(latent_d > capacity)

            # dynamic cost/margin calibrated around target 5.5% (+/- small noise)
            occ_pace = float(np.clip(observed_sales / max(capacity, 1), 0.0, 1.2))
            _ = occ_pace  # retained for realism; margin calibrated below
            fx_factor = float(np.exp(cell_rng.normal(0.0, 0.005)))
            _ = fx_factor
            margin_jitter = float(cell_rng.normal(0.0, 0.0015))
            desired_margin = float(np.clip(target_margin + margin_jitter, 0.01, 0.12))
            dyn_cost = float(np.clip(c.p * (1.0 - desired_margin), 0.5 * c.p, 0.995 * c.p))
            sell_price = float(c.p)
            realised_margin = (sell_price - dyn_cost) / max(sell_price, 1e-9)

            # fallback availability correlated within group
            a_i = float(np.clip(c.a + 0.03 * group_shock[d_idx] / max(group_sigma, 1e-6), 0.05, 0.99))
            gt_rows.append(
                {
                    "cell_id": c.cell_id,
                    "region_group": c.region_group,
                    "date": as_of.date().isoformat(),
                    "latent_demand": latent_d,
                    "capacity": capacity,
                    "observed_sales": observed_sales,
                    "sell_out": sell_out,
                    "fallback_availability": a_i,
                    "dynamic_net_cost": dyn_cost,
                    "dynamic_margin": realised_margin,
                    "p": sell_price,
                    "as_of_ts": as_of.isoformat(),
                }
            )

            # emit censored event rows
            search_id = _stable_id("SRCH", c.cell_id, d_idx)
            agent_reputation = float(np.clip(cell_rng.normal(0.75, 0.1), 0.0, 1.0))
            if cell_rng.random() < 0.03:
                agent_reputation = float(cell_rng.uniform(0.0, 0.15))
            agent_id = _stable_id("AGT", c.cell_id, int(agent_reputation * 1000), d_idx % 20)
            agency_tier = str(cell_rng.choice(["A", "B", "C"], p=[0.2, 0.5, 0.3]))
            channel = str(cell_rng.choice(["web", "api", "app"], p=[0.4, 0.4, 0.2]))
            checkout = stay_date + timedelta(days=c.stay_nights if c.stay_nights > 0 else 1)
            availability_rate = float(np.clip(observed_sales / max(capacity, 1) * (1.1 - 0.2 * sell_out), 0.05, 1.0))
            win_rate = float(np.clip(0.25 + 0.15 * (1 - sell_out), 0.05, 0.8))
            results_returned = int(np.clip(cell_rng.integers(5, 30), 1, 50))
            min_price_shown = float(max(sell_price * cell_rng.uniform(0.9, 1.05), dyn_cost * 1.01))
            our_rank = int(np.clip(cell_rng.integers(1, 8), 1, 20))

            tables["search_event"].append(
                {
                    "search_id": search_id,
                    "ts": (as_of - timedelta(minutes=int(cell_rng.integers(1, 120)))).isoformat(),
                    "agent_id": agent_id,
                    "agency_tier": agency_tier,
                    "source_market": c.source_market,
                    "destination_city": c.destination_city,
                    "checkin_date": stay_date.isoformat(),
                    "checkout_date": checkout.isoformat(),
                    "channel": channel,
                    "results_returned": results_returned,
                    "min_price_shown": round(min_price_shown, 2),
                    "our_rank": our_rank,
                    "as_of_ts": as_of.isoformat(),
                }
            )

            rooms_left = int(max(capacity - observed_sales, 0))
            tables["quote"].append(
                {
                    "search_id": search_id,
                    "hotel_id": f"H_{c.destination_city}",
                    "room_type": "STD",
                    "rate_plan": c.rate_plan,
                    "supplier_id": f"S_{c.region_group}",
                    "contract_type": c.contract_type,
                    "net_rate": round(dyn_cost, 2),
                    "our_sell_price": round(sell_price, 2),
                    "availability_flag": bool(rooms_left > 0),
                    "is_sold_out": bool(sell_out),
                    "rooms_left": rooms_left,  # censored view
                    "as_of_ts": as_of.isoformat(),
                }
            )

            if observed_sales > 0:
                lead_times = _beta_booking_curve(cell_rng, observed_sales, c.lead_time_min, c.lead_time_max, c.segment)
                for b_i in range(observed_sales):
                    lt = int(lead_times[b_i])
                    ts_booked = datetime.combine(stay_date - timedelta(days=lt), datetime.min.time()) + timedelta(
                        hours=int(cell_rng.integers(0, 23)), minutes=int(cell_rng.integers(0, 59))
                    )
                    b_as_of = ts_booked
                    booking_id = _stable_id("BK", c.cell_id, d_idx, b_i)
                    is_owned = c.contract_type in {"hard_buyout", "soft_release", "free_allotment", "air_bulk_fare"} and c.k > 0
                    margin_val = sell_price - dyn_cost
                    tables["booking"].append(
                        {
                            "booking_id": booking_id,
                            "search_id": search_id,
                            "ts_booked": ts_booked.isoformat(),
                            "agent_id": agent_id,
                            "hotel_id": f"H_{c.destination_city}",
                            "rate_plan": c.rate_plan,
                            "supplier_id": f"S_{c.region_group}",
                            "contract_type": c.contract_type,
                            "checkin": stay_date.isoformat(),
                            "room_nights": int(max(c.stay_nights, 1)),
                            "net_cost": round(dyn_cost, 2),
                            "sell_price": round(sell_price, 2),
                            "margin": round(margin_val, 2),
                            "lead_time_days": lt,
                            "is_owned_inventory": bool(is_owned),
                            "status": "confirmed",
                            "as_of_ts": b_as_of.isoformat(),
                        }
                    )
                    p_cancel = _cancellation_prob(lt, cancel_buckets)
                    p_cancel = float(np.clip(p_cancel * (1.0 + 0.15 * (lt / 60.0)), 0.01, 0.35))
                    if cell_rng.random() < p_cancel:
                        days_before = int(max(0, cell_rng.integers(0, max(lt, 1) + 1)))
                        ts_cancel = datetime.combine(stay_date - timedelta(days=days_before), datetime.min.time())
                        refund_amt = round(sell_price * float(cell_rng.uniform(0.5, 1.0)), 2)
                        tables["cancellation"].append(
                            {
                                "booking_id": booking_id,
                                "ts_cancelled": ts_cancel.isoformat(),
                                "days_before_checkin": days_before,
                                "refund_amount": refund_amt,
                                "as_of_ts": ts_cancel.isoformat(),
                            }
                        )

            # inventory_commit: units capped by supplier cap and capacity; sales censored
            units_committed = int(c.supplier_cap) if c.supplier_cap is not None else int(min(capacity, 50))
            units_committed = int(min(units_committed, capacity))
            units_sold_c = int(min(observed_sales, units_committed))
            unsold = int(max(units_committed - units_sold_c, 0))
            units_spoiled = int(unsold if c.r < 1.0 else 0)
            units_released = int(unsold if c.r >= 1.0 else 0)
            cash_prepaid = float(c.k * units_committed)
            realised_pnl = float(units_sold_c * (sell_price - c.c_b) - units_spoiled * c.k * (1.0 - c.r))
            release_deadline = datetime.combine(stay_date - timedelta(days=2), datetime.min.time())
            tables["inventory_commit"].append(
                {
                    "commit_id": _stable_id("CMT", c.cell_id, d_idx),
                    "ts": (as_of - timedelta(days=30)).isoformat(),
                    "supplier_id": f"S_{c.region_group}",
                    "hotel_id": f"H_{c.destination_city}",
                    "stay_date_range": f"{c.checkin_start.isoformat()}:{c.checkin_end.isoformat()}",
                    "units": units_committed,
                    "unit_net_cost": round(float(c.c_b), 2),
                    "cash_prepaid": round(cash_prepaid, 2),
                    "refund_frac": float(c.r),
                    "release_deadline": release_deadline.isoformat(),
                    "contract_type": c.contract_type,
                    "units_sold": units_sold_c,  # censored
                    "units_released": units_released,
                    "units_spoiled": units_spoiled,
                    "realised_pnl": round(realised_pnl, 2),
                    "as_of_ts": as_of.isoformat(),
                }
            )

            sig_val = float(100 * (1 + 0.2 * group_shock[d_idx]))
            tables["external_signal"].append(
                {
                    "date": as_of.date().isoformat(),
                    "geo_id": c.destination_city,
                    "signal_type": "demand_index",
                    "value": round(sig_val, 3),
                    "magnitude": round(float(abs(group_shock[d_idx])), 4),
                    "confidence": round(float(np.clip(0.7 + 0.2 * cell_rng.random(), 0, 1)), 3),
                    "as_of_ts": as_of.isoformat(),
                }
            )

    # build DataFrames with required columns (allow extra helper cols in search_event removed for strictness except kept harmlessly)
    history_tables: Dict[str, pd.DataFrame] = {}
    for name, rows in tables.items():
        df = pd.DataFrame(rows)
        # ensure required columns exist; keep extras out of strict schema tables except search_event agent_reputation dropped
        req = SCHEMAS[name]
        for col in req:
            if col not in df.columns:
                df[col] = np.nan
        # drop helper columns not in schema
        extra = [c for c in df.columns if c not in req]
        if extra:
            df = df.drop(columns=extra)
        df = df[req]
        history_tables[name] = df
        df.to_parquet(hist_dir / f"{name}.parquet", index=False)

    gt_df = pd.DataFrame(gt_rows)
    gt_df.to_parquet(out_path / "ground_truth.parquet", index=False)

    # convenience combined marker
    return {"ground_truth": gt_df, **history_tables}


def generate_and_write(master_seed: int = 42, out_dir: str | Path = "data") -> Path:
    """Generate and write all tables; return output directory path."""
    generate_synthetic_data(master_seed=master_seed, out_dir=out_dir)
    return Path(out_dir)


__all__ = ["generate_synthetic_data", "generate_and_write"]
