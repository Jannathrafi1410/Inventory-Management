"""Baseline allocation strategies B0-B4 for capital-allocation comparison.

All strategies return per-cell quantities aligned to the provided cell order and a spend total.
B0: no pre-buy. B1: equal split of cash budget. B2: volume-proportional (to mean demand).
B3: mean-demand newsvendor ranked by margin (delta_m). B4: correct fractile without budget awareness.

Assumption: "mean-demand newsvendor" (B3) buys q=min(cap, round(E[D])) for highest-margin (delta_m)
cash cells first until budget exhausted; zero-cash positive cells are taken to cap as free options.
"Volume-proportional" (B2) allocates cash budget proportional to E[D]*k over cash-positive cells.
Simulator outputs under stated assumptions, not TBO real P&L.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np

from optimizer.cell import Cell
from optimizer.fractile import optimal_fractile
from optimizer.profit import per_scenario_profit
from optimizer.waterfill import _quantile_from_sorted


@dataclass
class BaselineResult:
    """Result of a baseline allocation."""

    name: str
    quantities: np.ndarray
    spend: float
    lambda_star: float = 0.0


def _cash_spend(cells: Sequence[Cell], q: np.ndarray) -> float:
    return float(sum(float(c.k) * float(qi) for c, qi in zip(cells, q)))


def _scale_cash_to_budget(cells: Sequence[Cell], q: np.ndarray, budget: float) -> np.ndarray:
    q2 = np.array(q, dtype=float, copy=True)
    spend = _cash_spend(cells, q2)
    if spend <= budget + 1e-9 or spend <= 0:
        return q2
    scale = budget / spend
    for i, c in enumerate(cells):
        if c.k > 0:
            q2[i] *= scale
    for _ in range(1000):
        spend = _cash_spend(cells, q2)
        if spend <= budget + 1e-6:
            break
        idxs = [i for i, c in enumerate(cells) if c.k > 0 and q2[i] > 1e-9]
        if not idxs:
            break
        i = min(idxs, key=lambda j: q2[j])
        q2[i] = max(0.0, q2[i] - min(q2[i], 1e-3))
    spend = _cash_spend(cells, q2)
    if spend > budget + 1e-6 and spend > 0:
        scale = budget / spend
        for i, c in enumerate(cells):
            if c.k > 0:
                q2[i] *= scale
    return q2


def b0_no_prebuy(cells: Sequence[Cell], demand: np.ndarray, budget: float) -> BaselineResult:
    """B0: no pre-buy (all quantities zero)."""
    q = np.zeros(len(cells), dtype=float)
    return BaselineResult(name="B0_no_prebuy", quantities=q, spend=0.0, lambda_star=0.0)


def b1_equal_split(cells: Sequence[Cell], demand: np.ndarray, budget: float) -> BaselineResult:
    """B1: equal split of cash budget across cash-consuming cells with delta_m>0; free cells to cap."""
    n = len(cells)
    q = np.zeros(n, dtype=float)
    cash_idxs = [i for i, c in enumerate(cells) if c.k > 0 and c.delta_m() > 0]
    free_idxs = [i for i, c in enumerate(cells) if c.k <= 0 and c.delta_m() - c.carry() > 0]
    for i in free_idxs:
        q[i] = float(cells[i].cap)
    if cash_idxs and budget > 0:
        per = float(budget) / len(cash_idxs)
        for i in cash_idxs:
            q[i] = min(float(cells[i].cap), per / float(cells[i].k))
    q = _scale_cash_to_budget(cells, q, budget)
    return BaselineResult(name="B1_equal_split", quantities=q, spend=_cash_spend(cells, q), lambda_star=0.0)


def b2_volume_proportional(cells: Sequence[Cell], demand: np.ndarray, budget: float) -> BaselineResult:
    """B2: volume-proportional allocation to E[D]*k over cash-positive delta_m>0 cells."""
    n = len(cells)
    q = np.zeros(n, dtype=float)
    means = np.mean(np.asarray(demand, dtype=float), axis=0)
    free_idxs = [i for i, c in enumerate(cells) if c.k <= 0 and c.delta_m() - c.carry() > 0]
    for i in free_idxs:
        q[i] = float(cells[i].cap)
    cash_idxs = [i for i, c in enumerate(cells) if c.k > 0 and c.delta_m() > 0]
    weights = np.array([max(float(means[i]), 0.0) * float(cells[i].k) for i in cash_idxs], dtype=float)
    if cash_idxs and budget > 0 and weights.sum() > 0:
        alloc_cash = budget * (weights / weights.sum())
        for j, i in enumerate(cash_idxs):
            q[i] = min(float(cells[i].cap), float(alloc_cash[j]) / float(cells[i].k))
    elif cash_idxs and budget > 0:
        per = float(budget) / len(cash_idxs)
        for i in cash_idxs:
            q[i] = min(float(cells[i].cap), per / float(cells[i].k))
    q = _scale_cash_to_budget(cells, q, budget)
    return BaselineResult(name="B2_volume_proportional", quantities=q, spend=_cash_spend(cells, q), lambda_star=0.0)


def b3_mean_demand_newsvendor_ranked_by_margin(
    cells: Sequence[Cell], demand: np.ndarray, budget: float
) -> BaselineResult:
    """B3: mean-demand newsvendor ranked by margin (delta_m).

    Buys q=min(cap, round(E[D])) for positive-delta_m cash cells in descending delta_m order until
    budget exhausted. Free positive cells taken to cap.
    """
    n = len(cells)
    q = np.zeros(n, dtype=float)
    means = np.mean(np.asarray(demand, dtype=float), axis=0)
    for i, c in enumerate(cells):
        if c.k <= 0 and c.delta_m() - c.carry() > 0:
            q[i] = float(c.cap)
    cash_order = sorted(
        [i for i, c in enumerate(cells) if c.k > 0 and c.delta_m() > 0],
        key=lambda i: float(cells[i].delta_m()),
        reverse=True,
    )
    remaining = float(budget)
    for i in cash_order:
        if remaining <= 1e-9:
            break
        desired = float(min(float(cells[i].cap), max(round(float(means[i])), 0.0)))
        affordable = remaining / float(cells[i].k)
        qi = min(desired, affordable)
        q[i] = qi
        remaining -= float(cells[i].k) * qi
    q = _scale_cash_to_budget(cells, q, budget)
    return BaselineResult(name="B3_mean_demand_newsvendor", quantities=q, spend=_cash_spend(cells, q), lambda_star=0.0)


def b4_correct_fractile_without_budget_awareness(
    cells: Sequence[Cell], demand: np.ndarray, budget: float, lock_days: int = 120, r_annual: float = 0.12
) -> BaselineResult:
    """B4: correct fractile per cell without budget awareness, then cap to budget.

    Assumption: applies unconstrained optimal fractile quantities independently; if total cash exceeds
    budget, scales cash-cell quantities proportionally to respect budget (never spends above budget).
    """
    n = len(cells)
    q = np.zeros(n, dtype=float)
    sorted_d = np.sort(np.asarray(demand, dtype=float), axis=0)
    for i, c in enumerate(cells):
        f = float(optimal_fractile(c, lock_days=lock_days, r_annual=r_annual))
        q_star = _quantile_from_sorted(sorted_d[:, i], f)
        q[i] = float(min(float(c.cap), max(q_star, 0.0)))
    q = _scale_cash_to_budget(cells, q, budget)
    return BaselineResult(name="B4_correct_fractile_no_budget", quantities=q, spend=_cash_spend(cells, q), lambda_star=0.0)


def evaluate_strategy_profit(
    cells: Sequence[Cell], quantities: np.ndarray, demand: np.ndarray, lock_days: int = 120, r_annual: float = 0.12
) -> Dict[str, object]:
    """Evaluate aggregate incremental profit metrics for a strategy on scenario matrix.

    Returns mean per-scenario incremental profit vs no-pre-buy baseline (by per_scenario_profit),
    expected cash spoilage (k*(1-r)*E[(q-D)+]), and fill-rate uplift proxy.
    """
    d = np.asarray(demand, dtype=float)
    q = np.asarray(quantities, dtype=float)
    total_profit_scen = np.zeros(d.shape[0], dtype=float)
    total_spoilage = 0.0
    baseline_fill = 0.0
    with_buy_fill = 0.0
    for i, c in enumerate(cells):
        di = d[:, i]
        qi = float(q[i])
        total_profit_scen += per_scenario_profit(c, qi, di, lock_days=lock_days, r_annual=r_annual)
        leftover = np.maximum(qi - di, 0.0)
        total_spoilage += float(c.k) * (1.0 - float(c.r)) * float(np.mean(leftover))
        served_no = float(c.a) * di
        served_yes = np.minimum(di, qi) + float(c.a) * np.maximum(di - qi, 0.0)
        denom = np.maximum(di, 1e-9)
        baseline_fill += float(np.mean(np.clip(served_no / denom, 0.0, 1.0)))
        with_buy_fill += float(np.mean(np.clip(served_yes / denom, 0.0, 1.0)))
    n = max(len(cells), 1)
    return {
        "incremental_profit_mean": float(np.mean(total_profit_scen)),
        "incremental_profit_per_scenario": total_profit_scen,
        "expected_cash_spoilage": float(total_spoilage),
        "fill_rate_uplift": float((with_buy_fill - baseline_fill) / n),
    }


__all__ = [
    "BaselineResult",
    "b0_no_prebuy",
    "b1_equal_split",
    "b2_volume_proportional",
    "b3_mean_demand_newsvendor_ranked_by_margin",
    "b4_correct_fractile_without_budget_awareness",
    "evaluate_strategy_profit",
]
