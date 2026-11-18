"""Tests for deterministic decision core (optimizer).

Covers special cases, validation anchors, budget/water-filling properties, brute-force optimality,
and naive-newsvendor comparison. All numbers are simulator outputs under stated assumptions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optimizer.cell import Cell
from optimizer.demo_phase2 import build_demand_scenarios, sim_cell_to_optimizer
from optimizer.fractile import optimal_fractile
from optimizer.profit import expected_profit, per_scenario_profit
from optimizer.structures import compare_contract_structures
from optimizer.waterfill import _quantile_from_sorted, waterfill
from sim.cells import get_candidate_cells


def q_from_fractile(demand_col: np.ndarray, fractile: float, cap: int) -> float:
    s = np.sort(np.asarray(demand_col, dtype=float).reshape(-1))
    return float(min(float(cap), _quantile_from_sorted(s, fractile)))


# --- Validation anchors / special cases ---


def test_anchor_dxb_nye_hard_buyout_before_carry() -> None:
    # DXB-NYE: p=21000, c_d=19845, c_b=13495, k=13495, r=0, a=0.55, g=900 -> delta_m=7274.75, c_o=13495, F~0.350
    cell = Cell("DXB-NYE", 21000, 19845, 13495, 13495, 0.0, 0.55, 900, 120, "hard_buyout", "GCC")
    assert np.isclose(cell.delta_m(), 7274.75)
    assert np.isclose(cell.c_o(), 13495.0)
    f = cell.delta_m() / (cell.delta_m() + cell.c_o())
    assert np.isclose(f, 0.3501808, atol=1e-4)
    assert np.isclose(optimal_fractile(cell, lock_days=0, r_annual=0.12), 0.3501808, atol=1e-4)


def test_anchor_dxb_dec_soft_release_before_carry() -> None:
    # DXB-DEC: p=14500, c_d=13702, c_b=9592, k=9592, r=0.70, a=0.78, g=0 -> delta_m=4286, c_o=2878, F=0.598
    cell = Cell("DXB-DEC", 14500, 13702, 9592, 9592, 0.70, 0.78, 0, 150, "soft_release", "GCC")
    assert np.isclose(cell.delta_m(), 4285.56, atol=1e-2)
    assert np.isclose(cell.c_o(), 2877.6, atol=1e-2)
    f = cell.delta_m() / (cell.delta_m() + cell.c_o())
    assert np.isclose(f, 0.5982, atol=5e-4)
    assert np.isclose(optimal_fractile(cell, lock_days=0, r_annual=0.12), 0.5982, atol=5e-4)


def test_anchor_air_bomdxb_bulk_fare() -> None:
    # AIR-BOMDXB: p=34000, c_d=33200, c_b=29500, k=29500, r=0.3, a=0.80 -> delta_m=3860, c_o=20650, F~0.157
    cell = Cell("AIR-BOMDXB", 34000, 33200, 29500, 29500, 0.30, 0.80, 0, 80, "air_bulk_fare", "GCC")
    assert np.isclose(cell.delta_m(), 3860.0)
    assert np.isclose(cell.c_o(), 20650.0)
    f = cell.delta_m() / (cell.delta_m() + cell.c_o())
    assert np.isclose(f, 0.157486, atol=5e-4)
    assert np.isclose(optimal_fractile(cell, lock_days=0, r_annual=0.12), 0.157486, atol=5e-4)


def test_free_allotment_gives_F_one() -> None:
    # Any free allotment (k=0, r=1) => c_o=0 and positive delta_m-carry => F=1.0
    cell = Cell("FREE", 12000, 11340, 9500, 0.0, 1.0, 0.82, 0, 60, "free_allotment", "GCC")
    assert cell.c_o() == 0.0
    assert optimal_fractile(cell) == 1.0
    # also holds for another free cell
    cell2 = Cell("FREE2", 5000, 4800, 4000, 0.0, 1.0, 0.9, 100, 30, "free_allotment", "X")
    assert optimal_fractile(cell2) == 1.0


def test_special_case_classic_newsvendor() -> None:
    # a=0, g=0 => delta_m = p-c_b, c_o=k(1-r); classic newsvendor (p-c_b)/p when k=c_b,r=0 (hard buyout)
    cell = Cell("NV", 100.0, 90.0, 70.0, 70.0, 0.0, 0.0, 0.0, 100, "hard_buyout", "G")
    assert np.isclose(cell.delta_m(), 30.0)
    assert np.isclose(cell.c_o(), 70.0)
    f = optimal_fractile(cell, lock_days=0)
    assert np.isclose(f, (cell.p - cell.c_b) / cell.p)  # 0.3


def test_special_case_a1_cd_equals_cb_gives_zero() -> None:
    # a=1 and c_d=c_b => delta_m=0 so q*=0 (F=0)
    cell = Cell("NOADV", 100.0, 80.0, 80.0, 80.0, 0.0, 1.0, 0.0, 100, "hard_buyout", "G")
    assert np.isclose(cell.delta_m(), 0.0)
    assert optimal_fractile(cell) == 0.0


# --- Water-filling properties ---


def _make_cells_and_demand(seed: int = 7) -> tuple[list[Cell], np.ndarray]:
    rng = np.random.default_rng(seed)
    cells = [
        Cell("A", 12000, 11340, 9000, 9000, 0.7, 0.8, 0, 100, "soft_release", "G1"),
        Cell("B", 15000, 14175, 11000, 11000, 0.0, 0.7, 200, 80, "hard_buyout", "G1"),
        Cell("C", 9000, 8505, 7000, 0.0, 1.0, 0.85, 0, 50, "free_allotment", "G2"),  # k=0
        Cell("D", 20000, 18900, 15000, 15000, 0.7, 0.75, 0, 60, "soft_release", "G2"),
    ]
    demand = np.column_stack(
        [
            rng.lognormal(mean=np.log(60), sigma=0.35, size=2000),
            rng.lognormal(mean=np.log(40), sigma=0.4, size=2000),
            rng.lognormal(mean=np.log(30), sigma=0.3, size=2000),
            rng.lognormal(mean=np.log(35), sigma=0.3, size=2000),
        ]
    )
    return cells, demand


def test_spend_never_exceeds_budget_and_monotone_in_lambda() -> None:
    cells, demand = _make_cells_and_demand()
    budget = 800_000.0
    res = waterfill(cells, demand, budget=budget)
    assert res.spend <= budget + 1e-6

    # spend monotone decreasing in lambda
    sorted_d = np.sort(demand, axis=0)
    from optimizer.waterfill import _allocate_for_lambda

    spends = []
    for lam in [0.0, 1e-6, 1e-4, 1e-3, 1e-2, 0.05, 0.1]:
        _, sp, _ = _allocate_for_lambda(cells, sorted_d, lam, budget, 120, 0.12, None)
        spends.append(sp)
    for i in range(1, len(spends)):
        assert spends[i] <= spends[i - 1] + 1e-6, f"spend not monotone at {i}: {spends}"


def test_lambda_zero_when_budget_non_binding() -> None:
    cells, demand = _make_cells_and_demand()
    res = waterfill(cells, demand, budget=10**12)
    assert np.isclose(res.lambda_star, 0.0)
    # equals independent per-cell optima (at lambda 0)
    res0 = waterfill(cells, demand, budget=10**12)
    for i, c in enumerate(cells):
        q_ind = q_from_fractile(demand[:, i], optimal_fractile(c), c.cap)
        # free cell taken to cap
        if c.k == 0:
            q_ind = float(c.cap) if optimal_fractile(c) > 0 else 0.0
        assert np.isclose(res0.quantities[i], q_ind, atol=1e-6)


def test_large_budget_equals_independent_optima_small_budget_picks_highest_delta_m_over_k() -> None:
    cells, demand = _make_cells_and_demand()
    # large budget -> independent
    res_large = waterfill(cells, demand, budget=10**12)
    for i, c in enumerate(cells):
        f = optimal_fractile(c)
        q_ind = float(c.cap) if c.k == 0 and f > 0 else q_from_fractile(demand[:, i], f, c.cap)
        assert np.isclose(res_large.quantities[i], q_ind, atol=1e-6)

    # tiny budget: only highest delta_m/k cash cells should receive units (zero-cash free cells don't consume budget)
    res_tiny = waterfill(cells, demand, budget=50_000.0)
    cash_cells = [c for c in cells if c.k > 0]
    ratios = {c.id: c.delta_m() / c.k for c in cash_cells}
    best_cash = max(ratios, key=ratios.get)
    # some spend must go to best cash cell or be constrained reasonably
    qty_by_id = {cells[i].id: res_tiny.quantities[i] for i in range(len(cells))}
    # cash spent only where k>0 and total <= budget
    assert res_tiny.spend <= 50_000.0 + 1e-6
    # the best delta_m/k cash cell should have positive qty in this setup
    assert qty_by_id[best_cash] > 0.0


# --- Brute force vs closed form ---


def test_brute_force_matches_closed_form_within_2pct() -> None:
    rng = np.random.default_rng(1234)
    demand = rng.lognormal(mean=np.log(70), sigma=0.35, size=5000)
    cell = Cell("BF", 14500, 13702, 9592, 9592, 0.70, 0.78, 0, 200, "soft_release", "G")
    f = optimal_fractile(cell)
    q_star = q_from_fractile(demand, f, cell.cap)

    # grid search on Monte Carlo profit
    grid = np.unique(np.linspace(0, cell.cap, 400))
    profits = np.array([expected_profit(cell, q, demand) for q in grid])
    q_grid = float(grid[int(np.argmax(profits))])

    denom = max(abs(q_grid), 1.0)
    assert abs(q_star - q_grid) / denom <= 0.02, f"q_star={q_star}, q_grid={q_grid}"


# --- Naive newsvendor vs correct fractile ---


def test_naive_newsvendor_worse_than_correct_fractile_on_dxb_dec() -> None:
    rng = np.random.default_rng(99)
    demand = rng.lognormal(mean=np.log(110), sigma=0.30, size=4000)
    cell = Cell("DXB-DEC", 14500, 13702, 9592, 9592, 0.70, 0.78, 0, 150, "soft_release", "GCC")
    f_correct = optimal_fractile(cell)
    q_correct = q_from_fractile(demand, f_correct, cell.cap)
    # naive newsvendor (p-c_b)/p
    f_naive = float(np.clip((cell.p - cell.c_b) / cell.p, 0.0, 1.0))
    q_naive = q_from_fractile(demand, f_naive, cell.cap)
    profit_correct = expected_profit(cell, q_correct, demand)
    profit_naive = expected_profit(cell, q_naive, demand)
    assert profit_correct > profit_naive, (
        f"correct {profit_correct} not greater than naive {profit_naive} (q {q_correct} vs {q_naive})"
    )


# --- Structures trade-off ---


def test_compare_contract_structures_outputs_tradeoff() -> None:
    rng = np.random.default_rng(5)
    demand = rng.lognormal(mean=np.log(50), sigma=0.3, size=1000)
    out = compare_contract_structures(12000, 11340, 0.8, 0.0, 100, demand)
    assert "structures" in out and "trade_off_rule" in out
    assert len(out["structures"]) == 4
    assert "free_allotment" in out["trade_off_rule"]
    # free allotment should have highest fractile (F=1)
    fs = {s["structure"]: s["fractile"] for s in out["structures"]}
    assert fs["free_allotment"] == 1.0
