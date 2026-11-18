"""Tests for pricing optimization (pricing.py)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from pricing import PRICE_GRID, optimize_price_for_cell, run_pricing
from sim.cells import get_candidate_cells


def test_price_grid_and_argmax_and_flatness() -> None:
    assert len(PRICE_GRID) >= 10
    assert abs(float(PRICE_GRID[0]) - 0.88) < 1e-9
    assert abs(float(PRICE_GRID[-1]) - 1.16) < 1e-9
    cells = get_candidate_cells()
    c = next(x for x in cells if x.cell_id == "DXB-DEC")
    rng = np.random.default_rng(0)
    d0 = rng.lognormal(mean=np.log(max(c.base_demand_mean, 1e-6)), sigma=c.demand_sigma, size=200)
    res = optimize_price_for_cell(c, d0)
    assert res["argmax_multiplier"] >= 0.88 - 1e-9 and res["argmax_multiplier"] <= 1.16 + 1e-9
    assert "flatness" in res and "max_profit_loss_pct_vs_optimum" in res["flatness"]
    assert len(res["grid"]) == len(PRICE_GRID)


def test_pricing_run_writes_ladder(tmp_path: Path) -> None:
    out_md = tmp_path / "pricing.md"
    out_json = tmp_path / "pricing.json"
    res = run_pricing(n_scenarios=80, seed=1, out_md=str(out_md), out_json=str(out_json))
    assert out_md.exists() and out_json.exists()
    assert len(res["markdown_ladders"]) >= 1
    for lad in res["markdown_ladders"]:
        days = [r["days_to_release"] for r in lad["ladder"]]
        assert set(days) == {14, 7, 3}
    assert "ladder" in out_md.read_text(encoding="utf-8").lower()
