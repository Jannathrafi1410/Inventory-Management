"""Tests for exploration (exploration.py)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from exploration import run_exploration, thompson_sample_exploration
from sim.cells import get_candidate_cells


def test_exploration_budget_fraction_within_range() -> None:
    sim_cells = get_candidate_cells()
    res = thompson_sample_exploration(sim_cells, budget=10_000_000.0, exploration_fraction=0.03, seed=42)
    assert 0.02 <= res["exploration_fraction"] <= 0.05
    assert res["exploration_spend_inr"] <= res["exploration_budget_inr"] + 1e-3
    assert res["exploration_spend_inr"] <= 10_000_000.0 * 0.05 + 1e-3
    assert len(res["allocations"]) == len(sim_cells)


def test_exploration_run_integrates_and_writes(tmp_path: Path) -> None:
    out_md = tmp_path / "explore.md"
    out_json = tmp_path / "explore.json"
    res = run_exploration(budget=5_000_000.0, n_scenarios=80, seed=7, out_md=str(out_md), out_json=str(out_json))
    assert out_md.exists() and out_json.exists()
    exp = res["exploration"]
    assert 0.02 <= exp["exploration_fraction"] <= 0.05
    assert res["total_spend_inr"] <= 5_000_000.0 + 1e-3
    assert "thompson" in res.get("assumption", "").lower() or "Thompson" in out_md.read_text(encoding="utf-8")
