"""Optimal critical-fractile calculation for a decision cell."""

from __future__ import annotations

import numpy as np

from optimizer.cell import Cell


def optimal_fractile(cell: Cell, lock_days: int = 120, r_annual: float = 0.12) -> float:
    """Return the unconstrained optimal service level F(q*), clipped to [0, 1].

    F(q*) = (delta_m - carry) / (delta_m + c_o).
    Special handling:
      - c_o == 0: return 1.0 if (delta_m - carry) > 0 else 0.0 (free/fully-refunded option).
      - delta_m <= 0 or (delta_m - carry) <= 0: return 0.0 when not a positive free option.
    """
    dm = float(cell.delta_m())
    co = float(cell.c_o())
    car = float(cell.carry(lock_days=lock_days, r_annual=r_annual))
    num = dm - car

    if co == 0.0:
        return 1.0 if num > 0.0 else 0.0
    if dm <= 0.0 or num <= 0.0:
        return 0.0

    f = num / (dm + co)
    return float(np.clip(f, 0.0, 1.0))


__all__ = ["optimal_fractile"]
