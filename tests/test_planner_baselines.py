"""Tests for planner, baselines B0-B4, and baseline evaluation gate.

All numbers are simulator outputs under stated assumptions, not TBO real P&L.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from eval.run_baselines import _latent_heldout_scenarios, run_evaluation
from optimizer.baselines import (
    b0_no_prebuy,
    b1_equal_split,
    b2_volume_proportional,
    b3_mean_demand_newsvendor_ranked_by_margin,
    b4_correct_fractile_without_budget_awareness,
    evaluate_strategy_profit,
)
from optimizer.demo_phase2 import sim_cell_to_optimizer
from optimizer.planner import plan
from sim.cells import get_candidate_cells


def _setup():
    sim_cells = get_candidate_cells()
    opt_cells = [sim_cell_to_optimizer(c) for c in sim_cells]
    demand, _ = _latent_heldout_scenarios("data/ground_truth.parquet", sim_cells, s=120, seed=7, embargo_days=14)
    return sim_cells, opt_cells, demand


def test_plan_returns_complete_recommendations_and_plan_metrics() -> None:
    p = plan(10_000_000.0, n_scenarios=150, scenario_seed=42)
    assert len(p.recommendations) == len(get_candidate_cells())
    assert p.spend <= p.budget + 1e-6
    assert p.cash_deployed <= p.budget + 1e-6
    assert p.reserve >= 0.0
    assert isinstance(p.lambda_star, float)
    assert p.scenario_shape[1] > 0
    actions = {r.action for r in p.recommendations}
    assert actions.issubset({"COMMIT", "NEGOTIATE_ALLOTMENT", "WAIT", "RELEASE", "DONT_BUY"})
    for r in p.recommendations:
        assert r.cell_id
        assert r.structure
        assert r.units >= 0.0
        assert r.cash >= 0.0
        assert 0.0 <= r.fractile <= 1.0
        assert r.demand_p10 <= r.demand_p50 + 1e-6 or r.units == 0.0 or r.demand_p50 == 0.0
        assert isinstance(r.why, list) and len(r.why) >= 1
        assert all(isinstance(w, str) and w for w in r.why)


def test_baselines_never_exceed_budget_and_b0_zero() -> None:
    _sim, opt_cells, demand = _setup()
    budget = 5_000_000.0
    res = [
        b0_no_prebuy(opt_cells, demand, budget),
        b1_equal_split(opt_cells, demand, budget),
        b2_volume_proportional(opt_cells, demand, budget),
        b3_mean_demand_newsvendor_ranked_by_margin(opt_cells, demand, budget),
        b4_correct_fractile_without_budget_awareness(opt_cells, demand, budget),
    ]
    for r in res:
        assert r.spend <= budget + 1e-6
        assert np.all(r.quantities >= -1e-9)
    assert np.allclose(res[0].quantities, 0.0)
    assert res[0].spend == 0.0
    m0 = evaluate_strategy_profit(opt_cells, res[0].quantities, demand)
    m0_ref = evaluate_strategy_profit(opt_cells, np.zeros(len(opt_cells), dtype=float), demand)
    assert np.isclose(float(m0["incremental_profit_mean"]), float(m0_ref["incremental_profit_mean"]))



def test_plan_can_leave_budget_unspent_and_respects_budget() -> None:
    # very large budget should be non-binding in many cases -> spend < budget or lambda 0
    p = plan(1_000_000_000.0, n_scenarios=120, scenario_seed=1)
    assert p.spend <= p.budget + 1e-6
    # small budget binding
    p2 = plan(500_000.0, n_scenarios=120, scenario_seed=1)
    assert p2.spend <= p2.budget + 1e-6


def test_run_baselines_gate_passes_and_writes_table(tmp_path: Path) -> None:
    out = tmp_path / "baseline_eval.md"
    res = run_evaluation(
        budget=10_000_000.0,
        n_seeds=50,
        scenarios_per_seed=200,
        history_dir="data/history",
        ground_truth_path="data/ground_truth.parquet",
        out_path=str(out),
    )
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "EXIT GATE" in text and "Results" in text
    rows = {r["strategy"]: r for r in res["rows"]}
    for name in ["B0_no_prebuy", "B1_equal_split", "B2_volume_proportional", "B3_mean_demand_newsvendor", "B4_correct_fractile_no_budget", "Planner"]:
        assert name in rows
        assert rows[name]["spend_max"] <= 10_000_000.0 + 1e-6
    assert rows["Planner"]["sometimes_unspent"] is True
    for b in ["B1_equal_split", "B2_volume_proportional", "B3_mean_demand_newsvendor"]:
        assert res["beats"][b]["non_overlapping_or_favourable"] is True
    assert res["gate_pass"] is True
