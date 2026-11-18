"""Tests for correlated scenarios, mean-CVaR LP, frontier, and controls.

Validates requested gates:
- CVaR objective matches brute-force tail average (small instance)
- frontier monotone
- risk-aware CVaR >= risk-neutral
- doubling correlation worsens tails
- independent understates tail risk vs correlated
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecast.scenarios import apply_correlated_shocks_to_baseline, generate_factor_scenarios
from optimizer.cell import Cell
from optimizer.controls import apply_exposure_caps, apply_kill_switch, effective_budget_with_reserve, kill_switch_should_release
from optimizer.cvar_lp import _portfolio_profit_vector, bruteforce_cvar, solve_mean_cvar_lp
from optimizer.frontier import compute_frontier


def _toy_cells():
    return [
        Cell("A", p=100.0, c_d=90.0, c_b=70.0, k=70.0, r=0.0, a=0.7, g=10.0, cap=50, contract_type="hard", group="G1"),
        Cell("B", p=80.0, c_d=75.0, c_b=55.0, k=55.0, r=0.5, a=0.8, g=5.0, cap=40, contract_type="soft", group="G1"),
        Cell("C", p=120.0, c_d=110.0, c_b=0.0, k=0.0, r=1.0, a=0.9, g=0.0, cap=30, contract_type="free", group="G2"),
    ]


def _toy_quantiles(mu):
    qs = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    rows = []
    for m in mu:
        rows.append({f"q_{q}": float(max(m * (0.5 + q), 0.1)) for q in qs})
    return pd.DataFrame(rows)


def test_cvar_matches_bruteforce_small_instance() -> None:
    cells = _toy_cells()
    rng = np.random.default_rng(0)
    demand = np.column_stack([
        rng.lognormal(mean=np.log(30), sigma=0.3, size=120),
        rng.lognormal(mean=np.log(25), sigma=0.35, size=120),
        rng.lognormal(mean=np.log(15), sigma=0.25, size=120),
    ])
    res = solve_mean_cvar_lp(cells, demand, budget=5000.0, alpha=0.9, beta=0.6, reserve_fraction=0.0)
    assert res.success
    scen = _portfolio_profit_vector(cells, demand, res.quantities)
    bf = bruteforce_cvar(scen, alpha=0.9)
    assert abs(res.cvar - bf) < 1e-4


def test_frontier_monotone_and_risk_aware_improves_cvar() -> None:
    cells = _toy_cells()
    rng = np.random.default_rng(1)
    demand = np.column_stack([
        rng.lognormal(mean=np.log(28), sigma=0.4, size=200),
        rng.lognormal(mean=np.log(22), sigma=0.45, size=200),
        rng.lognormal(mean=np.log(14), sigma=0.3, size=200),
    ])
    points = compute_frontier(cells, demand, budget=4500.0, betas=np.linspace(0, 1, 6), alpha=0.95, enforce_monotone=True)
    exps = [p.expected_profit for p in points]
    cvars = [p.cvar for p in points]
    assert all(exps[i] >= exps[i + 1] - 1e-6 for i in range(len(exps) - 1))
    assert all(cvars[i] <= cvars[i + 1] + 1e-6 for i in range(len(cvars) - 1))
    assert points[-1].cvar >= points[0].cvar - 1e-6


def test_correlation_and_independence_tail_effects() -> None:
    cells = _toy_cells()
    qf = _toy_quantiles([30, 25, 15])
    # correlated
    d_corr, a_corr, cd_corr, factors, betas, _ = generate_factor_scenarios(
        qf, [c.id for c in cells], [c.group for c in cells], s=800, seed=7, correlation_scale=1.0,
        base_availability=[c.a for c in cells], cell_meta=pd.DataFrame([{"cell_id": c.id, "region_group": c.group, "destination_city": c.group, "segment": "event", "product_type": "hotel", "a": c.a} for c in cells])
    )
    # independent (no shared factor contribution): zero shared scale
    d_ind, a_ind, cd_ind, _, _, _ = generate_factor_scenarios(
        qf, [c.id for c in cells], [c.group for c in cells], s=800, seed=7, correlation_scale=0.0, idio_sigma=0.18,
        base_availability=[c.a for c in cells], cell_meta=pd.DataFrame([{"cell_id": c.id, "region_group": c.group, "destination_city": c.id, "segment": "leisure", "product_type": "hotel", "a": c.a} for c in cells])
    )
    # doubled correlation via scaling shared shocks on baseline correlated demand
    d_double = apply_correlated_shocks_to_baseline(d_corr, factors, betas, scale=1.0)  # additional shared shock equivalent to doubling contribution
    # fix plan (risk-neutral on correlated) and compare tail metrics
    plan = solve_mean_cvar_lp(cells, d_corr, budget=4500.0, alpha=0.95, beta=0.0, availability=a_corr, cost_drift=cd_corr)
    q = plan.quantities
    scen_corr = _portfolio_profit_vector(cells, d_corr, q, availability=a_corr, cost_drift=cd_corr)
    scen_ind = _portfolio_profit_vector(cells, d_ind, q, availability=a_ind, cost_drift=cd_ind)
    scen_double = _portfolio_profit_vector(cells, d_double, q, availability=a_corr, cost_drift=cd_corr)
    cvar_corr = bruteforce_cvar(scen_corr, alpha=0.95)
    cvar_ind = bruteforce_cvar(scen_ind, alpha=0.95)
    cvar_double = bruteforce_cvar(scen_double, alpha=0.95)
    assert cvar_ind > cvar_corr + 1e-6  # independent understates tail risk (less negative CVaR than correlated)
    assert cvar_double < cvar_corr + 1e-6  # doubling correlation worsens tails


def test_controls_reserve_caps_and_kill_switch() -> None:
    cells = _toy_cells()
    deployable, reserve = effective_budget_with_reserve(1_000_000, 0.18)
    assert 0.17 * 1_000_000 <= reserve <= 0.19 * 1_000_000
    assert abs(deployable + reserve - 1_000_000) < 1e-6
    q = np.array([40.0, 30.0, 20.0])
    capped = apply_exposure_caps(cells, q, effective_budget=1500.0, per_destination_caps={"G1": 800.0})
    spend_g1 = cells[0].k * capped[0] + cells[1].k * capped[1]
    assert spend_g1 <= 800.0 + 1e-6
    assert kill_switch_should_release(pace_to_date=70, plan_to_date=100, days_to_release_deadline=14, threshold=0.80, lead_days=14)
    assert not kill_switch_should_release(pace_to_date=70, plan_to_date=100, days_to_release_deadline=20, threshold=0.80, lead_days=14)
    released = apply_kill_switch(np.array([10.0, 5.0]), pace_to_date=70, plan_to_date=100, days_to_release_deadline=10)
    assert np.allclose(released, 0.0)
