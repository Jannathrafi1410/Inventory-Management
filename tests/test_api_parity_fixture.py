"""Sanity test: exported API demo fixture matches Python waterfill (parity source for Go).

Assumption: runs on deterministic demo export; simulator outputs under stated assumptions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from optimizer.demo_phase2 import build_demand_scenarios, sim_cell_to_optimizer
from optimizer.waterfill import waterfill
from sim.cells import get_candidate_cells


def test_exported_api_fixture_matches_python_waterfill() -> None:
    p = Path("data/api_demo/expected_waterfill.json")
    if not p.exists():
        # generate if missing (keeps CI/dev runnable)
        import scripts.export_api_demo  # type: ignore

        scripts.export_api_demo.main()
    data = json.loads(p.read_text(encoding="utf-8"))
    sim_cells = get_candidate_cells()
    opt = [sim_cell_to_optimizer(c) for c in sim_cells]
    demand = build_demand_scenarios(sim_cells, s=500, master_seed=42)
    for case in data["cases"]:
        res = waterfill(
            opt,
            demand,
            budget=float(case["budget"]),
            lock_days=int(case["lock_days"]),
            r_annual=float(case["r_annual"]),
            reserve_fraction=float(case["reserve_fraction"]),
            group_caps=case.get("group_caps"),
        )
        assert abs(res.lambda_star - float(case["lambda_star"])) <= 1e-6 * max(1.0, abs(res.lambda_star)) + 1e-6
        assert abs(res.spend - float(case["spend"])) <= 1e-3
        qs = [float(x) for x in res.quantities.tolist()]
        for i, (a, b) in enumerate(zip(qs, case["quantities"])):
            assert abs(a - float(b)) <= 1.0, f"case {case['name']} cell {i}: {a} vs {b}"
