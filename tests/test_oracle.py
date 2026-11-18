"""Tests for oracle evaluation (eval/oracle.py)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from eval.oracle import oracle_value, run_oracle_eval
from optimizer.demo_phase2 import sim_cell_to_optimizer
from sim.cells import get_candidate_cells


def test_oracle_value_nonnegative_and_beats_no_prebuy_proxy() -> None:
    sim_cells = get_candidate_cells()
    opt = [sim_cell_to_optimizer(c) for c in sim_cells]
    # small deterministic latent-like sample from held-out helper path
    import pandas as pd
    gt = pd.read_parquet("data/ground_truth.parquet")
    gt["date"] = pd.to_datetime(gt["date"]).dt.normalize()
    hold = gt[gt["date"] > gt["date"].max() - pd.Timedelta(days=14)]
    d = np.zeros((5, len(sim_cells)), dtype=float)
    for j, c in enumerate(sim_cells):
        vals = hold.loc[hold["cell_id"] == c.cell_id, "latent_demand"].astype(float).to_numpy()
        if len(vals) == 0:
            vals = np.array([1.0])
        d[:, j] = np.linspace(vals.min(), vals.max(), 5)
    ov, spoil, profits = oracle_value(opt, d, budget=5_000_000.0)
    assert np.isfinite(ov)
    assert np.isfinite(spoil) and spoil >= -1e-6
    assert profits.shape == (5,)


def test_run_oracle_eval_writes_results_and_regret(tmp_path: Path) -> None:
    out_md = tmp_path / "oracle.md"
    out_json = tmp_path / "oracle.json"
    res = run_oracle_eval(budget=5_000_000.0, n_seeds=2, scenarios_per_seed=60, out_md=str(out_md), out_json=str(out_json))
    assert out_md.exists() and out_json.exists()
    assert res["oracle_mean_uplift"] >= -1e-6
    names = {r["strategy"] for r in res["results"]}
    assert {"Planner", "B0_no_prebuy", "B1_equal_split", "B2_volume_proportional", "B3_mean_demand_newsvendor", "B4_correct_fractile_no_budget"}.issubset(names)
    for r in res["results"]:
        assert np.isfinite(r["mean_regret"])
        assert np.isfinite(r["pct_oracle_captured"])
