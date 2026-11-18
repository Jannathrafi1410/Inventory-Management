"""Budgeted water-filling allocator with bisection on shadow price lambda."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from optimizer.cell import Cell
from optimizer.fractile import optimal_fractile


@dataclass
class CellDiagnostic:
    """Per-cell allocation diagnostics."""

    cell_id: str
    fractile_used: float
    binding_cap: bool
    delta_m_over_k: float
    quantity: float
    spend: float
    group: str
    delta_m: float
    carry: float
    c_o: float
    k: float


@dataclass
class WaterfillResult:
    """Result of budgeted water-filling."""

    quantities: np.ndarray
    lambda_star: float
    spend: float
    diagnostics: List[CellDiagnostic]


def _quantile_from_sorted(sorted_col: np.ndarray, fractile: float) -> float:
    """Return demand quantile for fractile using pre-sorted scenarios (0<=fractile<=1)."""
    s = len(sorted_col)
    if s == 0:
        return 0.0
    f = float(np.clip(fractile, 0.0, 1.0))
    if f <= 0.0:
        return float(sorted_col[0])
    idx = int(np.ceil(f * s) - 1)
    idx = max(0, min(idx, s - 1))
    return float(sorted_col[idx])


def _normalize_group_caps(
    group_caps: Mapping[str, float] | None, effective_budget: float
) -> Dict[str, float]:
    """Normalize group caps to absolute cash limits.

    Assumption: values in (0,1] are treated as fraction of effective budget; values >1 as absolute Rs.
    """
    if not group_caps:
        return {}
    out: Dict[str, float] = {}
    for g, v in group_caps.items():
        vf = float(v)
        if vf <= 1.0:
            out[g] = vf * effective_budget
        else:
            out[g] = vf
    return out


def _apply_group_caps(
    cells: Sequence[Cell],
    q: np.ndarray,
    effective_budget: float,
    group_caps: Mapping[str, float] | None,
) -> np.ndarray:
    """Trim quantities to satisfy per-group cash concentration caps.

    Trims lowest delta_m/k cash cells within an over-cap group first. Zero-cash cells unaffected.
    """
    caps_abs = _normalize_group_caps(group_caps, effective_budget)
    if not caps_abs:
        return q
    q_adj = q.copy()
    groups: Dict[str, List[int]] = {}
    for i, c in enumerate(cells):
        groups.setdefault(c.group, []).append(i)
    for g, limit in caps_abs.items():
        idxs = [i for i in groups.get(g, []) if cells[i].k > 0]
        if not idxs:
            continue
        spend_g = float(sum(cells[i].k * q_adj[i] for i in idxs))
        if spend_g <= limit + 1e-9:
            continue
        # trim from lowest delta_m/k upward impact
        idxs_sorted = sorted(idxs, key=lambda i: (cells[i].delta_m() / cells[i].k))
        for i in idxs_sorted:
            if spend_g <= limit + 1e-9:
                break
            k_i = float(cells[i].k)
            max_reducible_cash = k_i * q_adj[i]
            need = spend_g - limit
            reduce_cash = min(max_reducible_cash, need)
            if k_i > 0:
                q_adj[i] = max(0.0, q_adj[i] - reduce_cash / k_i)
                spend_g -= reduce_cash
    return q_adj


def _allocate_for_lambda(
    cells: Sequence[Cell],
    sorted_demand: np.ndarray,
    lam: float,
    effective_budget: float,
    lock_days: int,
    r_annual: float,
    group_caps: Mapping[str, float] | None,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Compute quantities and spend for a given lambda, respecting caps and zero-cash rule."""
    n = len(cells)
    q = np.zeros(n, dtype=float)
    fractiles = np.zeros(n, dtype=float)
    for i, c in enumerate(cells):
        dm = float(c.delta_m())
        co = float(c.c_o())
        car = float(c.carry(lock_days=lock_days, r_annual=r_annual))
        if c.k <= 0.0:
            # zero-cash cells always taken to cap (free option / no cash at risk)
            f = 1.0 if (dm - car) > 0.0 else 0.0
        else:
            num = dm - car - lam * c.k
            if co == 0.0:
                f = 1.0 if num > 0.0 else 0.0
            elif dm <= 0.0 or num <= 0.0:
                f = 0.0
            else:
                f = float(np.clip(num / (dm + co), 0.0, 1.0))
        fractiles[i] = f
        q_star = _quantile_from_sorted(sorted_demand[:, i], f)
        q[i] = float(min(float(c.cap), max(q_star, 0.0)))

    q = _apply_group_caps(cells, q, effective_budget, group_caps)
    spend = float(sum(cells[i].k * q[i] for i in range(n)))
    return q, spend, fractiles


def waterfill(
    cells: Sequence[Cell],
    demand: np.ndarray,
    budget: float,
    lock_days: int = 120,
    r_annual: float = 0.12,
    reserve_fraction: float = 0.0,
    group_caps: Mapping[str, float] | None = None,
    tol: float = 1e-6,
    max_iter: int = 80,
) -> WaterfillResult:
    """Allocate quantities across cells under a cash budget via bisection on lambda.

    - Pre-sorts each demand column once.
    - Effective budget = budget * (1 - reserve_fraction).
    - If spend at lambda=0 <= effective budget, returns lambda=0 (budget non-binding).
    - Zero-cash (k==0) cells always taken to cap.
    - Bisection finds lambda so that spend ~= effective budget (or binding side) with doubling upper bound.
    """
    if demand.ndim != 2:
        raise ValueError("demand must be a 2D array of shape [S, n]")
    if demand.shape[1] != len(cells):
        raise ValueError("demand columns must match number of cells")
    n = len(cells)
    effective_budget = float(budget) * (1.0 - float(reserve_fraction))
    effective_budget = max(effective_budget, 0.0)
    sorted_demand = np.sort(np.asarray(demand, dtype=float), axis=0)

    q0, spend0, f0 = _allocate_for_lambda(
        cells, sorted_demand, 0.0, effective_budget, lock_days, r_annual, group_caps
    )
    if spend0 <= effective_budget + 1e-9:
        lam = 0.0
        q, spend, fractiles = q0, spend0, f0
    else:
        # Doubling to find an upper bound with feasible spend (<= effective budget).
        lam_lo = 0.0
        lam_hi = 1e-8
        q_hi, spend_hi, f_hi = q0, spend0, f0
        for _ in range(200):
            q_hi, spend_hi, f_hi = _allocate_for_lambda(
                cells, sorted_demand, lam_hi, effective_budget, lock_days, r_annual, group_caps
            )
            if spend_hi <= effective_budget + 1e-9:
                break
            lam_lo = lam_hi
            lam_hi *= 2.0

        # Bisection: keep lam_hi as feasible (spend <= budget), lam_lo as overspending/infeasible side.
        q, spend, fractiles = q_hi, spend_hi, f_hi
        lam = lam_hi
        for _ in range(max_iter):
            lam_mid = 0.5 * (lam_lo + lam_hi)
            q_mid, spend_mid, f_mid = _allocate_for_lambda(
                cells, sorted_demand, lam_mid, effective_budget, lock_days, r_annual, group_caps
            )
            if spend_mid > effective_budget + 1e-9:
                lam_lo = lam_mid
            else:
                lam_hi = lam_mid
                q, spend, fractiles = q_mid, spend_mid, f_mid
            lam = lam_hi
            if abs(lam_hi - lam_lo) <= tol * max(1.0, lam_hi):
                break

        # Ensure feasible result (spend <= effective budget).
        if spend > effective_budget + 1e-6:
            for _ in range(200):
                lam_hi = max(lam_hi * 2.0, lam_hi + 1e-12)
                q, spend, fractiles = _allocate_for_lambda(
                    cells, sorted_demand, lam_hi, effective_budget, lock_days, r_annual, group_caps
                )
                lam = lam_hi
                if spend <= effective_budget + 1e-6:
                    break
        # Final safety: if still over (numerical edge), scale down cash-cell quantities proportionally.
        if spend > effective_budget + 1e-6 and spend > 0:
            scale = effective_budget / spend
            q = q.copy()
            for i, c in enumerate(cells):
                if c.k > 0:
                    q[i] *= scale
            spend = float(sum(cells[i].k * q[i] for i in range(n)))

    diagnostics: List[CellDiagnostic] = []
    for i, c in enumerate(cells):
        dm = float(c.delta_m())
        car = float(c.carry(lock_days=lock_days, r_annual=r_annual))
        co = float(c.c_o())
        dm_over_k = float("inf") if c.k == 0 else float(dm / c.k)
        # recompute fractile used consistent with lambda for reporting
        if c.k <= 0.0:
            f_used = 1.0 if (dm - car) > 0.0 else 0.0
        else:
            num = dm - car - lam * c.k
            if co == 0.0:
                f_used = 1.0 if num > 0.0 else 0.0
            elif dm <= 0.0 or num <= 0.0:
                f_used = 0.0
            else:
                f_used = float(np.clip(num / (dm + co), 0.0, 1.0))
        binding = bool(q[i] >= float(c.cap) - 1e-9 and f_used > 0.0)
        diagnostics.append(
            CellDiagnostic(
                cell_id=c.id,
                fractile_used=f_used,
                binding_cap=binding,
                delta_m_over_k=dm_over_k,
                quantity=float(q[i]),
                spend=float(c.k * q[i]),
                group=c.group,
                delta_m=dm,
                carry=car,
                c_o=co,
                k=float(c.k),
            )
        )

    return WaterfillResult(quantities=q, lambda_star=float(lam), spend=float(spend), diagnostics=diagnostics)


__all__ = ["waterfill", "WaterfillResult", "CellDiagnostic"]
