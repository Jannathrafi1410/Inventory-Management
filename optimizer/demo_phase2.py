"""Demo: print fractiles and quantities for Phase 2 (sim candidate) cells at Rs 1 Cr budget.

Assumption: demand scenarios are LogNormal draws per cell using sim candidate base_demand_mean and
demand_sigma with deterministic seeds; carry uses 12% annual, 120-day lock. Simulator outputs only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import get_rng
from optimizer.cell import Cell
from optimizer.fractile import optimal_fractile
from optimizer.waterfill import waterfill
from sim.cells import get_candidate_cells


def sim_cell_to_optimizer(cell) -> Cell:
    cap = int(cell.supplier_cap) if cell.supplier_cap is not None else int(cell.capacity)
    return Cell(
        id=cell.cell_id,
        p=float(cell.p),
        c_d=float(cell.c_d),
        c_b=float(cell.c_b),
        k=float(cell.k),
        r=float(cell.r),
        a=float(cell.a),
        g=float(cell.g),
        cap=cap,
        contract_type=cell.contract_type,
        group=cell.region_group,
    )


def build_demand_scenarios(cells, s: int = 2500, master_seed: int = 42) -> np.ndarray:
    n = len(cells)
    demand = np.zeros((s, n), dtype=float)
    for j, c in enumerate(cells):
        rng = get_rng(master_seed, f"demo:demand:{c.cell_id}")
        demand[:, j] = rng.lognormal(mean=np.log(max(c.base_demand_mean, 1e-6)), sigma=float(c.demand_sigma))
    return demand


def main() -> None:
    sim_cells = get_candidate_cells()
    opt_cells = [sim_cell_to_optimizer(c) for c in sim_cells]
    demand = build_demand_scenarios(sim_cells, s=2500, master_seed=42)
    budget = 10_000_000.0  # Rs 1 Cr
    res = waterfill(opt_cells, demand, budget=budget, lock_days=120, r_annual=0.12)

    rows = []
    for i, c in enumerate(opt_cells):
        diag = res.diagnostics[i]
        f_uncon = optimal_fractile(c, lock_days=120, r_annual=0.12)
        rows.append(
            {
                "cell_id": c.id,
                "group": c.group,
                "contract": c.contract_type,
                "p": c.p,
                "k": c.k,
                "cap": c.cap,
                "fractile_unconstrained": round(f_uncon, 4),
                "fractile_used": round(diag.fractile_used, 4),
                "quantity": round(diag.quantity, 3),
                "spend": round(diag.spend, 2),
                "delta_m": round(diag.delta_m, 2),
                "delta_m_over_k": (None if np.isinf(diag.delta_m_over_k) else round(diag.delta_m_over_k, 6)),
                "binding_cap": diag.binding_cap,
            }
        )
    df = pd.DataFrame(rows).sort_values(["spend", "delta_m_over_k"], ascending=[False, False])
    print(f"Budget (Rs): {budget:,.0f} | lambda (shadow price): {res.lambda_star:.8f} | spend: {res.spend:,.2f}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
