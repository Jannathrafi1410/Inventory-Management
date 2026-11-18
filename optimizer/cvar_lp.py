"""Rockafellar-Uryasev mean-CVaR portfolio optimization for pre-buy quantities.

Maximizes (1-beta)*E[pi(q)] + beta*CVaR_alpha(pi(q)) over all Monte Carlo scenarios
subject to cash budget, per-cell caps, concentration caps, and reserve tranche.

Linearization (standard, exact sample formulation):
- sold_{i,s} <= D_{i,s}, sold_{i,s} <= q_i
- short_{i,s} = D_{i,s} - sold_{i,s}
- leftover_{i,s} = q_i - sold_{i,s}
- z_s >= t - pi_s(q), z_s >= 0  =>  CVaR_alpha = t - (1/((1-alpha)*S)) * sum z_s

This is convex (LP) in (q, sold, short, leftover, t, z). Solved via scipy.optimize.linprog (HiGHS).
All numbers are simulator outputs under stated assumptions, not TBO real P&L.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog

from optimizer.cell import Cell


@dataclass
class CVaRLPResult:
    """Result of mean-CVaR LP optimization."""

    quantities: np.ndarray
    t: float
    expected_profit: float
    cvar: float
    objective: float
    spend: float
    reserve: float
    status: int
    success: bool
    message: str
    per_scenario_profit: np.ndarray


def _portfolio_profit_vector(
    cells: Sequence[Cell],
    demand: np.ndarray,
    quantities: np.ndarray,
    lock_days: int = 120,
    r_annual: float = 0.12,
    availability: Optional[np.ndarray] = None,
    cost_drift: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute per-scenario portfolio profit for fixed quantities with optional correlated shocks.

    - availability: scenario-specific effective fallback probability a_{i,s} in [0,1].
    - cost_drift: multiplicative drift on dynamic net cost c_d (affects fallback value/shortfall coeff).
    """
    d = np.asarray(demand, dtype=float)
    q = np.asarray(quantities, dtype=float).reshape(1, -1)
    s = d.shape[0]
    total = np.zeros(s, dtype=float)
    for i, c in enumerate(cells):
        di = d[:, i]
        qi = float(q[0, i])
        sold = np.minimum(di, qi)
        short = np.maximum(di - qi, 0.0)
        leftover = np.maximum(qi - di, 0.0)
        carry = float(c.carry(lock_days=lock_days, r_annual=r_annual))
        a_is = c.a
        if availability is not None:
            a_is = np.clip(np.asarray(availability, dtype=float)[:, i], 0.0, 1.0)
        c_d_eff = c.c_d
        if cost_drift is not None:
            c_d_eff = c.c_d * np.asarray(cost_drift, dtype=float)[:, i]
        coeff_sold = c.p - c.c_b + c.k
        coeff_short = a_is * (c.p - c_d_eff) - (1.0 - a_is) * c.g
        total += coeff_sold * sold + coeff_short * short - c.k * qi + c.r * c.k * leftover - carry * qi
    return total


def bruteforce_cvar(profits: np.ndarray, alpha: float = 0.95) -> float:
    """Brute-force sample CVaR_alpha (expected shortfall of lower tail) for validation.

    Uses the Rockafellar-Uryasev sample estimator: average of profits in the worst (1-alpha) tail.
    For continuous distributions equals inf_t {t - 1/((1-alpha)S)*sum (t-pi_s)_+}.
    """
    p = np.sort(np.asarray(profits, dtype=float).reshape(-1))
    s = len(p)
    if s == 0:
        return 0.0
    a = float(np.clip(alpha, 1e-6, 1 - 1e-6))
    k = int(np.ceil((1.0 - a) * s))
    k = max(1, min(k, s))
    # worst k outcomes (smallest profits)
    return float(np.mean(p[:k]))


def _build_and_solve(
    cells: Sequence[Cell],
    demand: np.ndarray,
    budget: float,
    alpha: float = 0.95,
    beta: float = 0.5,
    lock_days: int = 120,
    r_annual: float = 0.12,
    reserve_fraction: float = 0.0,
    group_caps: Optional[Mapping[str, float]] = None,
    availability: Optional[np.ndarray] = None,
    cost_drift: Optional[np.ndarray] = None,
    solver_options: Optional[Dict[str, Any]] = None,
) -> CVaRLPResult:
    d = np.asarray(demand, dtype=float)
    s, n = d.shape
    if n != len(cells):
        raise ValueError("demand columns must match number of cells")
    alpha = float(np.clip(alpha, 1e-6, 1 - 1e-6))
    beta = float(np.clip(beta, 0.0, 1.0))
    reserve_fraction = float(max(0.0, min(reserve_fraction, 0.95)))
    effective_budget = float(budget) * (1.0 - reserve_fraction)
    reserve = float(budget) - effective_budget

    # variables: q(n), sold(s*n), t(1), z(s)
    nq = n
    nsold = s * n
    nt = 1
    nz = s
    nv = nq + nsold + nt + nz
    idx_q = 0
    idx_sold = idx_q + nq
    idx_t = idx_sold + nsold
    idx_z = idx_t + nt

    def sold_idx(i_s: int, i_c: int) -> int:
        return idx_sold + i_s * n + i_c

    # coefficients per cell/scenario for profit
    # pi_s = sum_i ( A_i*sold_{i,s} + B_{i,s}*short_{i,s} - k_i*q_i + r_i*k_i*leftover_{i,s} - carry_i*q_i )
    # short = D - sold, leftover = q - sold
    # => pi_s = sum_i sold_{i,s}*(A_i - B_{i,s} - r_i*k_i) + q_i*(-k_i + r_i*k_i - carry_i) + B_{i,s}*D_{i,s}
    A = np.array([c.p - c.c_b + c.k for c in cells], dtype=float)
    K = np.array([c.k for c in cells], dtype=float)
    R = np.array([c.r for c in cells], dtype=float)
    CARRY = np.array([c.carry(lock_days=lock_days, r_annual=r_annual) for c in cells], dtype=float)
    # B_{i,s}
    B = np.zeros((s, n), dtype=float)
    Dconst = np.zeros(s, dtype=float)
    for i, c in enumerate(cells):
        a_is = c.a
        if availability is not None:
            a_is = np.clip(np.asarray(availability, dtype=float)[:, i], 0.0, 1.0)
        c_d_eff = c.c_d
        if cost_drift is not None:
            c_d_eff = c.c_d * np.asarray(cost_drift, dtype=float)[:, i]
        b_is = a_is * (c.p - c_d_eff) - (1.0 - a_is) * c.g
        B[:, i] = b_is
        Dconst += b_is * d[:, i]

    q_coeff_profit = -K + R * K - CARRY  # per i
    sold_coeff_profit = A.reshape(1, -1) - B - (R * K).reshape(1, -1)  # [s,n]

    # Objective: maximize (1-beta)*E[pi] + beta*(t - 1/((1-alpha)*S)*sum z_s)
    # linprog minimizes c^T x => c = -objective_coeffs
    c_obj = np.zeros(nv, dtype=float)
    inv_s = 1.0 / s
    inv_tail = 1.0 / ((1.0 - alpha) * s)
    # expected profit contribution
    c_obj[idx_q:idx_q + nq] += (1.0 - beta) * inv_s * q_coeff_profit
    for i_s in range(s):
        for i_c in range(n):
            c_obj[sold_idx(i_s, i_c)] += (1.0 - beta) * inv_s * sold_coeff_profit[i_s, i_c]
    # CVaR term: beta * t - beta*inv_tail*sum z_s
    c_obj[idx_t] += beta
    c_obj[idx_z:idx_z + nz] += -beta * inv_tail
    c_lp = -c_obj

    A_ub_rows: List[np.ndarray] = []
    b_ub: List[float] = []

    # budget: sum k_i*q_i <= effective_budget
    row = np.zeros(nv)
    row[idx_q:idx_q + nq] = K
    A_ub_rows.append(row)
    b_ub.append(effective_budget)

    # group caps
    if group_caps:
        groups: Dict[str, List[int]] = {}
        for i, cell in enumerate(cells):
            groups.setdefault(cell.group, []).append(i)
        for g, cap in group_caps.items():
            cap_v = float(cap)
            if 0 < cap_v <= 1.0:
                cap_v = cap_v * effective_budget
            idxs = groups.get(g, [])
            if not idxs:
                continue
            row = np.zeros(nv)
            for i in idxs:
                row[idx_q + i] = K[i]
            A_ub_rows.append(row)
            b_ub.append(cap_v)

    # sold constraints per (s,i): sold <= D and sold <= q
    for i_s in range(s):
        for i_c in range(n):
            # sold - q <= 0
            row = np.zeros(nv)
            row[sold_idx(i_s, i_c)] = 1.0
            row[idx_q + i_c] = -1.0
            A_ub_rows.append(row)
            b_ub.append(0.0)
            # sold <= D
            row = np.zeros(nv)
            row[sold_idx(i_s, i_c)] = 1.0
            A_ub_rows.append(row)
            b_ub.append(float(d[i_s, i_c]))

    # CVaR shortfall: t - pi_s - z_s <= 0  => t - z_s - sum sold_coeff*sold - sum q_coeff*q <= Dconst_s
    for i_s in range(s):
        row = np.zeros(nv)
        row[idx_t] = 1.0
        row[idx_z + i_s] = -1.0
        row[idx_q:idx_q + nq] += -q_coeff_profit
        for i_c in range(n):
            row[sold_idx(i_s, i_c)] += -sold_coeff_profit[i_s, i_c]
        A_ub_rows.append(row)
        b_ub.append(float(Dconst[i_s]))

    A_ub = np.vstack(A_ub_rows) if A_ub_rows else None
    b_ub_arr = np.array(b_ub, dtype=float) if b_ub else None

    bounds: List[Tuple[Optional[float], Optional[float]]] = []
    for i in range(n):
        bounds.append((0.0, float(max(cells[i].cap, 0.0))))
    for _ in range(nsold):
        bounds.append((0.0, None))
    bounds.append((None, None))  # t
    for _ in range(nz):
        bounds.append((0.0, None))  # z_s

    opts = {"presolve": True}
    if solver_options:
        opts.update(solver_options)
    res = linprog(c_lp, A_ub=A_ub, b_ub=b_ub_arr, bounds=bounds, method="highs", options=opts)

    q_opt = np.zeros(n, dtype=float)
    t_opt = 0.0
    if res.success and res.x is not None:
        q_opt = np.clip(res.x[idx_q:idx_q + nq], 0.0, [float(c.cap) for c in cells])
        t_opt = float(res.x[idx_t])
    else:
        # fallback: all zero (feasible)
        q_opt = np.zeros(n, dtype=float)
        t_opt = 0.0

    # compute realized metrics on the same scenarios (with shocks if provided)
    scen_profit = _portfolio_profit_vector(
        cells, d, q_opt, lock_days=lock_days, r_annual=r_annual, availability=availability, cost_drift=cost_drift
    )
    exp_profit = float(np.mean(scen_profit))
    cvar = bruteforce_cvar(scen_profit, alpha=alpha)
    obj = (1.0 - beta) * exp_profit + beta * cvar
    spend = float(np.sum(K * q_opt))

    return CVaRLPResult(
        quantities=q_opt,
        t=t_opt,
        expected_profit=exp_profit,
        cvar=cvar,
        objective=obj,
        spend=spend,
        reserve=reserve,
        status=int(res.status),
        success=bool(res.success),
        message=str(res.message),
        per_scenario_profit=scen_profit,
    )


def solve_mean_cvar_lp(
    cells: Sequence[Cell],
    demand: np.ndarray,
    budget: float,
    alpha: float = 0.95,
    beta: float = 0.5,
    lock_days: int = 120,
    r_annual: float = 0.12,
    reserve_fraction: float = 0.0,
    group_caps: Optional[Mapping[str, float]] = None,
    availability: Optional[np.ndarray] = None,
    cost_drift: Optional[np.ndarray] = None,
) -> CVaRLPResult:
    """Solve mean-CVaR LP (Rockafellar-Uryasev) via HiGHS.

    max (1-beta)*E[dProfit] + beta*CVaR_alpha(dProfit)
    s.t. budget (with reserve tranche), per-cell caps, concentration/group caps.
    """
    return _build_and_solve(
        cells=cells,
        demand=demand,
        budget=budget,
        alpha=alpha,
        beta=beta,
        lock_days=lock_days,
        r_annual=r_annual,
        reserve_fraction=reserve_fraction,
        group_caps=group_caps,
        availability=availability,
        cost_drift=cost_drift,
    )


__all__ = ["CVaRLPResult", "solve_mean_cvar_lp", "bruteforce_cvar"]
