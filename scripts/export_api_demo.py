"""Export deterministic demo cell/scenario dataset and Python waterfill parity fixture for Go API.

Assumption: demo scenarios are LogNormal draws per sim candidate (seeded) as in optimizer/demo_phase2;
cap = supplier_cap or capacity; scenarios stored sorted per column (float64 little-endian) for Go
binary-search quantile lookup. Simulator outputs under stated assumptions, not TBO real P&L.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from optimizer.demo_phase2 import build_demand_scenarios, sim_cell_to_optimizer
from optimizer.waterfill import waterfill
from sim.cells import get_candidate_cells


def main() -> None:
    out = Path("data/api_demo")
    out.mkdir(parents=True, exist_ok=True)
    sim_cells = get_candidate_cells()
    opt = [sim_cell_to_optimizer(c) for c in sim_cells]
    demand = build_demand_scenarios(sim_cells, s=500, master_seed=42)
    sorted_d = np.sort(np.asarray(demand, dtype=float), axis=0)

    meta = {
        "version": "demo_v1",
        "s": int(sorted_d.shape[0]),
        "n": int(sorted_d.shape[1]),
        "cell_ids": [c.id for c in opt],
        "dtype": "float64_le",
        "order": "column_major_sorted",
        "assumption": "Demo scenarios from sim candidate LogNormal draws (master_seed=42), sorted per column for empirical CDF; simulator output under stated assumptions.",
    }
    with open(out / "scenarios.bin", "wb") as f:
        for j in range(sorted_d.shape[1]):
            f.write(np.asarray(sorted_d[:, j], dtype="<f8").tobytes())
    (out / "scenarios.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    cells = []
    for c in sim_cells:
        oc = sim_cell_to_optimizer(c)
        cells.append(
            {
                "cell_id": c.cell_id,
                "destination_city": c.destination_city,
                "region_group": c.region_group,
                "destination": c.destination_city,
                "supplier_group": c.destination_city,
                "source_market": c.source_market,
                "rate_plan": c.rate_plan,
                "product_type": c.product_type,
                "segment": c.segment,
                "contract_type": c.contract_type,
                "p": oc.p,
                "c_d": oc.c_d,
                "c_b": oc.c_b,
                "k": oc.k,
                "r": oc.r,
                "a": oc.a,
                "g": oc.g,
                "cap": oc.cap,
                "group": oc.group,
            }
        )
    (out / "cells.json").write_text(
        json.dumps(
            {
                "assumption": "Derived from sim.cells.get_candidate_cells; cap=supplier_cap or capacity; destination/supplier groups default to destination_city.",
                "cells": cells,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    cases = []
    for name, budget, reserve_fraction, group_caps in [
        ("base_10m", 10_000_000.0, 0.0, None),
        ("budget_5m", 5_000_000.0, 0.0, None),
        ("budget_20m", 20_000_000.0, 0.0, None),
        ("reserve_03", 10_000_000.0, 0.03, None),
        ("group_cap_gcc", 10_000_000.0, 0.0, {"GCC": 0.4}),
    ]:
        res = waterfill(opt, demand, budget=budget, lock_days=120, r_annual=0.12, reserve_fraction=reserve_fraction, group_caps=group_caps)
        cases.append(
            {
                "name": name,
                "budget": budget,
                "lock_days": 120,
                "r_annual": 0.12,
                "reserve_fraction": reserve_fraction,
                "group_caps": group_caps,
                "lambda_star": res.lambda_star,
                "spend": res.spend,
                "quantities": [float(x) for x in res.quantities.tolist()],
                "fractiles": [float(d.fractile_used) for d in res.diagnostics],
            }
        )
    (out / "expected_waterfill.json").write_text(json.dumps({"cases": cases}, indent=2), encoding="utf-8")
    print("ok", sorted_d.shape, "cases", len(cases))


if __name__ == "__main__":
    main()
