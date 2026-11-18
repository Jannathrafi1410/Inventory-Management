"""Controls: exposure caps, reserve tranche, and kill-switch auto-release logic.

All controls operate on simulator quantities/cash metrics under stated assumptions (not TBO real P&L).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from optimizer.cell import Cell


@dataclass
class ControlConfig:
    """Control parameters.

    - per_destination_caps / per_supplier_caps: mapping of group key -> cap (fraction <=1 of deployable or absolute cash >1)
    - reserve_tranche_fraction: fraction of budget held as reserve (expected 0.15-0.20)
    - kill_switch_pace_threshold: trigger if pace < threshold * plan at (release_deadline - lead_days)
    """

    per_destination_caps: Optional[Mapping[str, float]] = None
    per_supplier_caps: Optional[Mapping[str, float]] = None
    reserve_tranche_fraction: float = 0.18
    kill_switch_pace_threshold: float = 0.80
    kill_switch_lead_days: int = 14


def effective_budget_with_reserve(budget: float, reserve_fraction: float) -> Tuple[float, float]:
    """Return (deployable_budget, reserve_amount) enforcing reserve tranche."""
    rf = float(np.clip(reserve_fraction, 0.0, 0.95))
    reserve = float(budget) * rf
    return float(budget) - reserve, reserve


def normalize_caps(caps: Optional[Mapping[str, float]], effective_budget: float) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not caps:
        return out
    for k, v in caps.items():
        vf = float(v)
        out[str(k)] = vf * effective_budget if 0 < vf <= 1.0 else vf
    return out


def apply_exposure_caps(
    cells: Sequence[Cell],
    quantities: np.ndarray,
    effective_budget: float,
    per_destination_caps: Optional[Mapping[str, float]] = None,
    per_supplier_caps: Optional[Mapping[str, float]] = None,
    destination_key: str = "group",
) -> np.ndarray:
    """Trim quantities to satisfy per-destination and per-supplier cash exposure caps.

    Trims lowest (delta_m/k) cash units within each over-cap group first. Zero-cash cells unaffected.
    Assumption: destination key defaults to cell.group; supplier cap applied per cell group key as well
    when provided (caller can map supplier groups into caps).
    """
    q = np.array(quantities, dtype=float, copy=True)
    cap_sets = [normalize_caps(per_destination_caps, effective_budget), normalize_caps(per_supplier_caps, effective_budget)]
    for caps in cap_sets:
        if not caps:
            continue
        # group by destination key (group) for destination caps; for supplier caps same mechanism using provided keys
        groups: Dict[str, List[int]] = {}
        for i, c in enumerate(cells):
            key = getattr(c, destination_key, c.group)
            groups.setdefault(str(key), []).append(i)
        for g, limit in caps.items():
            idxs = [i for i in groups.get(str(g), []) if cells[i].k > 0]
            if not idxs:
                continue
            spend_g = float(sum(cells[i].k * q[i] for i in idxs))
            if spend_g <= limit + 1e-9:
                continue
            idxs_sorted = sorted(idxs, key=lambda i: (cells[i].delta_m() / max(cells[i].k, 1e-12)))
            for i in idxs_sorted:
                if spend_g <= limit + 1e-9:
                    break
                k_i = float(cells[i].k)
                reducible = k_i * q[i]
                need = spend_g - limit
                red = min(reducible, need)
                q[i] = max(0.0, q[i] - red / k_i)
                spend_g -= red
    # ensure total within effective budget
    total = float(sum(cells[i].k * q[i] for i in range(len(cells))))
    if total > effective_budget + 1e-6 and total > 0:
        scale = effective_budget / total
        for i, c in enumerate(cells):
            if c.k > 0:
                q[i] *= scale
    return q


def kill_switch_should_release(
    pace_to_date: float,
    plan_to_date: float,
    days_to_release_deadline: int,
    threshold: float = 0.80,
    lead_days: int = 14,
) -> bool:
    """Return True if kill-switch should auto-release.

    Trigger: at (release_deadline - lead_days), if pace < threshold * plan.
    If days_to_release_deadline is greater than lead_days, no trigger yet (returns False).
    """
    if plan_to_date <= 0:
        return False
    if int(days_to_release_deadline) > int(lead_days):
        return False
    return float(pace_to_date) < float(threshold) * float(plan_to_date)


def apply_kill_switch(
    quantities: np.ndarray,
    pace_to_date: Sequence[float] | float,
    plan_to_date: Sequence[float] | float,
    days_to_release_deadline: int,
    threshold: float = 0.80,
    lead_days: int = 14,
) -> np.ndarray:
    """Auto-release (set to zero) cells that breach kill-switch at decision point.

    If scalar pace/plan provided, applies uniformly when triggered. If per-cell arrays provided,
    releases only breaching cells at the trigger horizon.
    """
    q = np.array(quantities, dtype=float, copy=True)
    if np.isscalar(pace_to_date) and np.isscalar(plan_to_date):
        if kill_switch_should_release(float(pace_to_date), float(plan_to_date), days_to_release_deadline, threshold, lead_days):
            q[:] = 0.0
        return q
    pace = np.asarray(pace_to_date, dtype=float)
    plan = np.asarray(plan_to_date, dtype=float)
    if int(days_to_release_deadline) > int(lead_days):
        return q
    breach = (plan > 0) & (pace < float(threshold) * plan)
    q[breach] = 0.0
    return q


__all__ = [
    "ControlConfig",
    "effective_budget_with_reserve",
    "normalize_caps",
    "apply_exposure_caps",
    "kill_switch_should_release",
    "apply_kill_switch",
]
