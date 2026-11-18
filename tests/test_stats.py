"""Tests for stats/CIs (eval/stats.py)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from eval.stats import paired_bootstrap_ci, run_stats_suite


def test_paired_bootstrap_ci_reproducible() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    m1, lo1, hi1 = paired_bootstrap_ci(x, n_boot=500, alpha=0.10, seed=7)
    m2, lo2, hi2 = paired_bootstrap_ci(x, n_boot=500, alpha=0.10, seed=7)
    assert (m1, lo1, hi1) == (m2, lo2, hi2)
    assert lo1 <= m1 <= hi1


def test_stats_suite_two_parameterisations(tmp_path: Path) -> None:
    out_md = tmp_path / "stats.md"
    out_json = tmp_path / "stats.json"
    res = run_stats_suite(budget=5_000_000.0, n_seeds=3, scenarios_per_seed=60, out_md=str(out_md), out_json=str(out_json))
    assert out_md.exists() and out_json.exists()
    assert len(res["results"]) == 2
    assert {r["param_set"] for r in res["results"]} == {"A_baseline", "B_alt_generator"}
    for r in res["results"]:
        assert len(r["rows"]) >= 6
        assert len(r["comparisons"]) >= 3
    assert "summary" in res and "robust_planner_beats_b1_b3_mean_both" in res["summary"]
