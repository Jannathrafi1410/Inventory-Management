"""Tests for timing policies (timing.py)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from sim.cells import get_candidate_cells
from timing import priced_wait, release_decision, run_timing, tranching_policy


def test_tranching_resizes_and_sums() -> None:
    c = next(x for x in get_candidate_cells() if x.cell_id == "DXB-NYE")
    rng = np.random.default_rng(1)
    d = rng.lognormal(mean=np.log(max(c.base_demand_mean, 1e-6)), sigma=c.demand_sigma, size=200)
    tr = tranching_policy(c, d)
    shares = [t["planned_share"] for t in tr["tranches"]]
    assert abs(sum(shares) - 1.0) < 1e-9
    assert tr["total_units"] <= c.capacity + 1e-6
    assert tr["total_cash_inr"] >= -1e-6
    cum = 0.0
    for t in tr["tranches"]:
        cum += t["resized_units"]
        assert t["cumulative_units"] >= cum - 1e-6


def test_wait_and_release_rules_and_docs(tmp_path: Path) -> None:
    c = next(x for x in get_candidate_cells() if x.cell_id == "DXB-DEC")
    rng = np.random.default_rng(2)
    d = rng.lognormal(mean=np.log(max(c.base_demand_mean, 1e-6)), sigma=c.demand_sigma, size=200)
    pw = priced_wait(c, d)
    assert pw["recommended_action"] in {"WAIT", "COMMIT"}
    assert "value_wait_inr" in pw
    rd = release_decision(c, d, q_on_hand=20.0, days_to_expiry=7)
    assert rd["recommended_action"] in {"HOLD", "RELEASE"}
    out_md = tmp_path / "timing.md"
    out_json = tmp_path / "timing.json"
    res = run_timing(n_scenarios=80, seed=3, out_md=str(out_md), out_json=str(out_json))
    assert out_md.exists() and out_json.exists()
    assert len(res["cells"]) >= 1
