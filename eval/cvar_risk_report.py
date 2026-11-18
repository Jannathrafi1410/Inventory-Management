"""Generate correlated-risk evaluation artifacts: factor scenarios, mean-CVaR frontier, and comparison.

Writes versioned scenarios to disk plus frontier chart/table comparing risk-neutral vs mean-CVaR.
Frontier LP is solved on a scenario subset (frontier_scenarios) for tractability; comparison metrics
are evaluated on the full saved scenario bundle.

Usage:
  python -m eval.cvar_risk_report --scenarios 1500 --frontier-scenarios 150 --seed 42 --out docs

All numbers are simulator outputs under stated assumptions, not TBO real P&L.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from eval.run_baselines import _build_forecast_model
from forecast.scenarios import create_and_save_factor_scenarios
from optimizer.cvar_lp import solve_mean_cvar_lp
from optimizer.demo_phase2 import sim_cell_to_optimizer
from optimizer.frontier import generate_frontier_artifacts
from sim.cells import get_candidate_cells


def _subset_bundle(bundle, n: int, seed: int = 0):
    s = int(bundle.demand.shape[0])
    n = int(max(10, min(n, s)))
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(s, size=n, replace=False))
    demand = bundle.demand[idx]
    avail = bundle.availability[idx]
    cost = bundle.cost_drift[idx]
    return demand, avail, cost, idx


def run_cvar_risk_report(
    n_scenarios: int = 1500,
    frontier_scenarios: int = 150,
    seed: int = 42,
    budget: float = 10_000_000.0,
    alpha: float = 0.95,
    beta: float = 0.5,
    reserve_fraction: float = 0.18,
    correlation_scale: float = 1.0,
    history_dir: str = "data/history",
    out_dir: str = "docs",
    scenario_dir: str = "data/scenarios",
) -> Dict[str, Any]:
    sim_cells = get_candidate_cells()
    opt_cells = [sim_cell_to_optimizer(c) for c in sim_cells]
    qmodel, cfg, feats_pool, feat_cols, cutoff, max_day = _build_forecast_model(history_dir=history_dir, embargo_days=14)
    cell_ids = [c.cell_id for c in sim_cells]
    latest = feats_pool.sort_values("decision_ts").groupby("cell_id").tail(1)
    latest = latest.set_index("cell_id").reindex(cell_ids).reset_index().rename(columns={"index": "cell_id"})
    med = feats_pool[feat_cols].median(numeric_only=True)
    for col in feat_cols:
        if col not in latest.columns:
            latest[col] = med.get(col, 0.0)
        latest[col] = pd.to_numeric(latest[col], errors="coerce")
    latest[feat_cols] = latest[feat_cols].fillna(med).fillna(0.0)
    qf = qmodel.predict_quantiles(latest[feat_cols])
    qf["cell_id"] = cell_ids

    cell_meta = pd.DataFrame(
        [
            {
                "cell_id": c.cell_id,
                "region_group": c.region_group,
                "destination_city": c.destination_city,
                "segment": c.segment,
                "product_type": c.product_type,
                "a": c.a,
            }
            for c in sim_cells
        ]
    )

    bundle = create_and_save_factor_scenarios(
        qf,
        cell_ids=cell_ids,
        cell_groups=[c.region_group for c in sim_cells],
        s=int(n_scenarios),
        seed=int(seed),
        out_dir=scenario_dir,
        version=f"factor_scenarios_s{int(n_scenarios)}_seed{int(seed)}_cs{correlation_scale}",
        correlation_scale=float(correlation_scale),
        base_availability=[c.a for c in sim_cells],
        cell_meta=cell_meta,
    )

    fd, fa, fc, _ = _subset_bundle(bundle, frontier_scenarios, seed=seed + 7)
    frontier_res = generate_frontier_artifacts(
        cells=opt_cells,
        demand=fd,
        budget=float(budget),
        out_dir=out_dir,
        betas=np.linspace(0.0, 1.0, 11),
        alpha=float(alpha),
        lock_days=120,
        r_annual=0.12,
        reserve_fraction=float(reserve_fraction),
        group_caps=None,
        availability=fa,
        cost_drift=fc,
    )

    # independent-demand baseline (shuffle per cell) on subset for tractable LP comparison
    rng = np.random.default_rng(int(seed) + 999)
    indep_d = fd.copy()
    indep_a = fa.copy()
    indep_c = fc.copy()
    for j in range(indep_d.shape[1]):
        rng.shuffle(indep_d[:, j])
        rng.shuffle(indep_a[:, j])
        rng.shuffle(indep_c[:, j])
    corr_rn = solve_mean_cvar_lp(opt_cells, fd, budget, alpha=alpha, beta=0.0, reserve_fraction=reserve_fraction, availability=fa, cost_drift=fc)
    indep_rn = solve_mean_cvar_lp(opt_cells, indep_d, budget, alpha=alpha, beta=0.0, reserve_fraction=reserve_fraction, availability=indep_a, cost_drift=indep_c)

    summary = {
        "scenario_bundle": bundle.path,
        "n_scenarios_saved": int(bundle.demand.shape[0]),
        "frontier_scenarios_used": int(fd.shape[0]),
        "seed": int(seed),
        "budget": float(budget),
        "alpha": float(alpha),
        "reserve_fraction": float(reserve_fraction),
        "correlation_scale": float(correlation_scale),
        "comparison_risk_neutral_vs_mean_cvar": frontier_res.get("comparison", {}),
        "frontier_chart": frontier_res.get("chart"),
        "tail_independent_vs_correlated_risk_neutral": {
            "correlated_cvar95": float(corr_rn.cvar),
            "independent_cvar95": float(indep_rn.cvar),
            "independent_understates_tail_risk": bool(indep_rn.cvar > corr_rn.cvar + 1e-6),
            "correlated_expected": float(corr_rn.expected_profit),
            "independent_expected": float(indep_rn.expected_profit),
        },
        "assumptions": "Simulator outputs under stated assumptions; frontier solved on subset for tractability; full correlated bundle versioned to disk.",
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cvar_risk_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Correlated risk + mean-CVaR frontier report")
    ap.add_argument("--scenarios", type=int, default=1500)
    ap.add_argument("--frontier-scenarios", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--budget", type=float, default=10_000_000.0)
    ap.add_argument("--alpha", type=float, default=0.95)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--reserve-fraction", type=float, default=0.18)
    ap.add_argument("--correlation-scale", type=float, default=1.0)
    ap.add_argument("--history", type=str, default="data/history")
    ap.add_argument("--out", type=str, default="docs")
    ap.add_argument("--scenario-dir", type=str, default="data/scenarios")
    args = ap.parse_args()
    res = run_cvar_risk_report(
        n_scenarios=args.scenarios,
        frontier_scenarios=args.frontier_scenarios,
        seed=args.seed,
        budget=args.budget,
        alpha=args.alpha,
        beta=args.beta,
        reserve_fraction=args.reserve_fraction,
        correlation_scale=args.correlation_scale,
        history_dir=args.history,
        out_dir=args.out,
        scenario_dir=args.scenario_dir,
    )
    print(
        json.dumps(
            {
                "bundle": res.get("scenario_bundle"),
                "chart": res.get("frontier_chart"),
                "indep_understates": res.get("tail_independent_vs_correlated_risk_neutral", {}).get(
                    "independent_understates_tail_risk"
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
