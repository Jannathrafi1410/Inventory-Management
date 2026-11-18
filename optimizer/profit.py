"""Profit calculations on Monte Carlo demand scenario matrices."""

from __future__ import annotations

import numpy as np

from optimizer.cell import Cell


def per_scenario_profit(
    cell: Cell, q: float, demand: np.ndarray, lock_days: int = 120, r_annual: float = 0.12
) -> np.ndarray:
    """Return per-scenario profit vector relative to the no-pre-buy baseline (dProfit) at quantity q.

    pi(q) = (p - c_b + k)*E[min(D,q)] + [a*(p - c_d) - (1-a)*g]*E[(D-q)+]
            - k*q + r*k*E[(q-D)+] - carry*q
    """
    d = np.asarray(demand, dtype=float).reshape(-1)
    qf = float(max(q, 0.0))
    sold = np.minimum(d, qf)
    short = np.maximum(d - qf, 0.0)
    leftover = np.maximum(qf - d, 0.0)
    carry = float(cell.carry(lock_days=lock_days, r_annual=r_annual))
    coeff_sold = (cell.p - cell.c_b + cell.k)
    coeff_short = (cell.a * (cell.p - cell.c_d) - (1.0 - cell.a) * cell.g)
    return coeff_sold * sold + coeff_short * short - cell.k * qf + cell.r * cell.k * leftover - carry * qf


def expected_profit(
    cell: Cell, q: float, demand: np.ndarray, lock_days: int = 120, r_annual: float = 0.12
) -> float:
    """Exact expected profit of a cell at quantity q on a scenario matrix/vector."""
    return float(np.mean(per_scenario_profit(cell, q, demand, lock_days=lock_days, r_annual=r_annual)))


__all__ = ["expected_profit", "per_scenario_profit"]
