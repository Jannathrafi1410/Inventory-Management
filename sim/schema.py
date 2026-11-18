"""Schema definitions and helpers for simulator output tables.

All rows carry `as_of_ts` where applicable to support leakage-safe feature reconstruction.
Assumption: parquet is the canonical storage for generated tables (DuckDB can read them directly).
"""

from __future__ import annotations

from typing import Dict, List

# Required columns per table as specified by task.
SCHEMAS: Dict[str, List[str]] = {
    "search_event": [
        "search_id",
        "ts",
        "agent_id",
        "agency_tier",
        "source_market",
        "destination_city",
        "checkin_date",
        "checkout_date",
        "channel",
        "results_returned",
        "min_price_shown",
        "our_rank",
        "as_of_ts",
    ],
    "quote": [
        "search_id",
        "hotel_id",
        "room_type",
        "rate_plan",
        "supplier_id",
        "contract_type",
        "net_rate",
        "our_sell_price",
        "availability_flag",
        "is_sold_out",
        "rooms_left",
        "as_of_ts",
    ],
    "booking": [
        "booking_id",
        "search_id",
        "ts_booked",
        "agent_id",
        "hotel_id",
        "rate_plan",
        "supplier_id",
        "contract_type",
        "checkin",
        "room_nights",
        "net_cost",
        "sell_price",
        "margin",
        "lead_time_days",
        "is_owned_inventory",
        "status",
        "as_of_ts",
    ],
    "cancellation": [
        "booking_id",
        "ts_cancelled",
        "days_before_checkin",
        "refund_amount",
        "as_of_ts",
    ],
    "inventory_commit": [
        "commit_id",
        "ts",
        "supplier_id",
        "hotel_id",
        "stay_date_range",
        "units",
        "unit_net_cost",
        "cash_prepaid",
        "refund_frac",
        "release_deadline",
        "contract_type",
        "units_sold",
        "units_released",
        "units_spoiled",
        "realised_pnl",
        "as_of_ts",
    ],
    "external_signal": [
        "date",
        "geo_id",
        "signal_type",
        "value",
        "magnitude",
        "confidence",
        "as_of_ts",
    ],
}


def empty_columns(table: str) -> List[str]:
    """Return required columns for a table name."""
    if table not in SCHEMAS:
        raise KeyError(f"Unknown table: {table}")
    return list(SCHEMAS[table])


__all__ = ["SCHEMAS", "empty_columns"]
