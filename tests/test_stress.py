"""Tests for stress suite (eval/stress.py)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from eval.stress import STRESS_CASES, run_stress_suite


def test_stress_suite_outputs_required_cases_and_metrics(tmp_path: Path) -> None:
    out_md = tmp_path / "stress.md"
    out_json = tmp_path / "stress.json"
    res = run_stress_suite(budget=5_000_000.0, n_scenarios=80, seed=42, out_md=str(out_md), out_json=str(out_json))
    assert out_md.exists() and out_json.exists()
    table = res["table"]
    names = [r["scenario"] for r in table]
    assert set(names) == set(STRESS_CASES)
    for r in table:
        for k in ["rn_expected_uplift", "rn_cvar95", "rn_cash_spoilage", "ra_expected_uplift", "ra_cvar95", "ra_cash_spoilage"]:
            assert np.isfinite(r[k])
        assert r["rn_spend"] <= 5_000_000.0 + 1e-6
        assert r["ra_spend"] <= 5_000_000.0 + 1e-6
    text = out_md.read_text(encoding="utf-8")
    assert "risk-neutral" in text.lower() and "risk-averse" in text.lower()
