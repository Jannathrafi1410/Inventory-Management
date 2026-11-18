"""Statistical evaluation: paired bootstrap (CRN), 90% CIs, 50 seeds, and robustness check.

Compares strategies on same per-seed realized outcomes using common random numbers.
Checks robustness across at least two generator parameterisations. Ground truth read
only inside /eval. All numbers are simulator outputs under stated assumptions.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from config.settings import get_rng
from eval.run_baselines import _build_forecast_model
from forecast.scenarios import scenarios_for_cells
from optimizer.baselines import (
    b0_no_prebuy,
    b1_equal_split,
    b2_volume_proportional,
    b3_mean_demand_newsvendor_ranked_by_margin,
    b4_correct_fractile_without_budget_awareness,
)
from optimizer.demo_phase2 import sim_cell_to_optimizer
from optimizer.planner import candidate_funnel
from optimizer.profit import per_scenario_profit
from optimizer.waterfill import waterfill
from sim.cells import get_candidate_cells

# Assumption: paired bootstrap resamples seed-level paired outcomes with replacement (2000 resamples, seed=123) to form 90% percentile CIs.
# Assumption: parameterisation A = baseline held-out sampling (rho=0.6, demand_scale=1.0, sigma_scale=1.0); parameterisation B = stressed generator (rho=0.85, demand_scale=1.1, sigma_scale=1.25) to check robustness/overfitting to simulator defaults.
# Assumption: same forecast decisions per seed are evaluated against parameterisation-specific realized latent scenarios (CRN seed index shared).


@dataclass
class ParamSet:
    name: str
    rho: float
    demand_scale: float
    sigma_scale: float


def latent_heldout(
    ground_truth_path: str,
    sim_cells: Sequence[Any],
    s: int,
    seed: int,
    embargo_days: int = 14,
    rho: float = 0.6,
    demand_scale: float = 1.0,
    sigma_scale: float = 1.0,
) -> np.ndarray:
    """Held-out latent scenarios with configurable generator parameterisation (eval-only)."""
    gt = pd.read_parquet(ground_truth_path).copy()
    gt["date"] = pd.to_datetime(gt["date"]).dt.normalize()
    max_day = gt["date"].max()
    cutoff = max_day - pd.Timedelta(days=int(embargo_days))
    hold = gt[gt["date"] > cutoff].copy()
    if hold.empty:
        hold = gt.copy()
    cell_ids = [c.cell_id for c in sim_cells]
    groups = [c.region_group for c in sim_cells]
    rng = get_rng(int(seed), f"eval:stats:latent:{rho}:{demand_scale}:{sigma_scale}")
    unique_groups = list(dict.fromkeys(groups))
    g_index = {g: i for i, g in enumerate(unique_groups)}
    z_g = rng.standard_normal(size=(s, len(unique_groups)))
    eps = rng.standard_normal(size=(s, len(cell_ids)))
    rho_c = float(np.clip(rho, 0.0, 0.99))
    f_load = float(np.sqrt(rho_c))
    i_scale = float(np.sqrt(max(1.0 - rho_c, 1e-6)))
    demand = np.zeros((s, len(cell_ids)), dtype=float)
    for j, cid in enumerate(cell_ids):
        vals = hold.loc[hold["cell_id"] == cid, "latent_demand"].astype(float).to_numpy()
        if len(vals) == 0:
            vals = gt.loc[gt["cell_id"] == cid, "latent_demand"].astype(float).to_numpy()
        if len(vals) == 0:
            vals = np.array([1.0], dtype=float)
        base_mean = float(np.mean(vals))
        mu = base_mean * float(demand_scale)
        sdv = float(np.std(vals))
        sigma = float(np.clip((sdv / max(base_mean, 1e-6)) * float(sigma_scale), 0.05, 1.2))
        z = f_load * z_g[:, g_index[groups[j]]] + i_scale * eps[:, j]
        demand[:, j] = np.maximum(mu * np.exp(sigma * z - 0.5 * sigma * sigma), 0.0)
        draw_idx = rng.integers(0, len(vals), size=s)
        empirical = vals[draw_idx] * float(demand_scale)
        demand[:, j] = 0.7 * demand[:, j] + 0.3 * np.maximum(empirical, 0.0)
    return demand


def _profit_mean(cells, q, demand, lock_days=120, r_annual=0.12) -> float:
    d = np.asarray(demand, dtype=float)
    total = np.zeros(d.shape[0], dtype=float)
    for i, c in enumerate(cells):
        total += per_scenario_profit(c, float(max(q[i], 0.0)), d[:, i], lock_days=lock_days, r_annual=r_annual)
    return float(np.mean(total))


def paired_bootstrap_ci(diffs: np.ndarray, n_boot: int = 2000, alpha: float = 0.10, seed: int = 123) -> Tuple[float, float, float]:
    """Return (mean, lo, hi) 90% percentile CI for paired differences."""
    v = np.asarray(diffs, dtype=float)
    if len(v) == 0:
        return 0.0, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = np.mean(v[idx], axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return float(np.mean(v)), lo, hi


def _planner_quantities(sim_cells, opt_cells, feats_pool, forecast_demand, budget: float) -> np.ndarray:
    selected = candidate_funnel(sim_cells, feats_pool, budget, constraints={"lock_days": 120, "r_annual": 0.12}, top_k=18) or sim_cells
    sel_idx = [i for i, c in enumerate(sim_cells) if c.cell_id in {s.cell_id for s in selected}]
    wf_sel = waterfill([opt_cells[i] for i in sel_idx], forecast_demand[:, sel_idx], budget=float(budget), lock_days=120, r_annual=0.12, reserve_fraction=0.03)
    pq = np.zeros(len(opt_cells), dtype=float)
    for j, i in enumerate(sel_idx):
        pq[i] = wf_sel.quantities[j]
    return pq


def evaluate_paramset(
    param: ParamSet,
    budget: float,
    n_seeds: int,
    scenarios_per_seed: int,
    history_dir: str,
    ground_truth_path: str,
) -> Dict[str, Any]:
    sim_cells = get_candidate_cells()
    opt_cells = [sim_cell_to_optimizer(c) for c in sim_cells]
    qmodel, cfg, feats_pool, feat_cols, _, _ = _build_forecast_model(history_dir=history_dir, embargo_days=14)

    strategies = ["B0_no_prebuy", "B1_equal_split", "B2_volume_proportional", "B3_mean_demand_newsvendor", "B4_correct_fractile_no_budget", "Planner"]
    profits = {s: [] for s in strategies}

    for seed in range(int(n_seeds)):
        # decisions from forecast (CRN)
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
        forecast_demand, _ = scenarios_for_cells(qf, cell_ids=cell_ids, cell_groups=[c.region_group for c in sim_cells], s=scenarios_per_seed, quantiles=cfg.quantiles, intra_group_rho=0.6, seed=seed, label=f"stats:forecast:{seed}")

        heldout = latent_heldout(ground_truth_path, sim_cells, scenarios_per_seed, seed=seed, rho=param.rho, demand_scale=param.demand_scale, sigma_scale=param.sigma_scale)

        allocs = {
            "B0_no_prebuy": b0_no_prebuy(opt_cells, forecast_demand, budget).quantities,
            "B1_equal_split": b1_equal_split(opt_cells, forecast_demand, budget).quantities,
            "B2_volume_proportional": b2_volume_proportional(opt_cells, forecast_demand, budget).quantities,
            "B3_mean_demand_newsvendor": b3_mean_demand_newsvendor_ranked_by_margin(opt_cells, forecast_demand, budget).quantities,
            "B4_correct_fractile_no_budget": b4_correct_fractile_without_budget_awareness(opt_cells, forecast_demand, budget).quantities,
            "Planner": _planner_quantities(sim_cells, opt_cells, feats_pool, forecast_demand, budget),
        }
        for name, q in allocs.items():
            profits[name].append(_profit_mean(opt_cells, q, heldout))

    rows = []
    for name in strategies:
        arr = np.array(profits[name], dtype=float)
        m, lo, hi = paired_bootstrap_ci(arr - 0.0, n_boot=2000, alpha=0.10, seed=123)
        rows.append({"strategy": name, "mean": float(np.mean(arr)), "ci90_lo": lo, "ci90_hi": hi})
    # paired comparisons vs planner
    comparisons = []
    p_arr = np.array(profits["Planner"], dtype=float)
    for base in ["B1_equal_split", "B2_volume_proportional", "B3_mean_demand_newsvendor", "B0_no_prebuy", "B4_correct_fractile_no_budget"]:
        diff = p_arr - np.array(profits[base], dtype=float)
        dm, dlo, dhi = paired_bootstrap_ci(diff, n_boot=2000, alpha=0.10, seed=123)
        comparisons.append({"planner_minus": base, "mean_diff": dm, "ci90_lo": dlo, "ci90_hi": dhi, "planner_better_ci90": bool(dlo > 0)})
    return {"param_set": param.name, "rho": param.rho, "demand_scale": param.demand_scale, "sigma_scale": param.sigma_scale, "rows": rows, "comparisons": comparisons, "raw_means": {k: float(np.mean(v)) for k, v in profits.items()}}


def run_stats_suite(
    budget: float = 10_000_000.0,
    n_seeds: int = 50,
    scenarios_per_seed: int = 300,
    history_dir: str = "data/history",
    ground_truth_path: str = "data/ground_truth.parquet",
    out_md: str = "docs/stats_cis.md",
    out_json: str = "docs/stats_cis.json",
) -> Dict[str, Any]:
    """Run paired bootstrap CIs for 50 seeds across two generator parameterisations."""
    params = [
        ParamSet("A_baseline", rho=0.6, demand_scale=1.0, sigma_scale=1.0),
        ParamSet("B_alt_generator", rho=0.85, demand_scale=1.1, sigma_scale=1.25),
    ]
    results = []
    for p in params:
        results.append(evaluate_paramset(p, budget, n_seeds, scenarios_per_seed, history_dir, ground_truth_path))

    # robustness: planner beats B1-B3 in both parameterisations?
    robust = True
    checks = []
    for res in results:
        for comp in res["comparisons"]:
            if comp["planner_minus"] in {"B1_equal_split", "B2_volume_proportional", "B3_mean_demand_newsvendor"}:
                ok = bool(comp["mean_diff"] > 0)  # favorable on mean across seeds
                checks.append({"param_set": res["param_set"], **comp, "pass": ok})
                if not ok:
                    robust = False
    summary = {"robust_planner_beats_b1_b3_mean_both": robust, "checks": checks}

    out = {
        "assumption": "Simulator outputs under stated assumptions, not TBO real P&L. Paired bootstrap on CRN, 90% percentile CIs (2000 resamples). Two generator parameterisations checked to reduce overfitting to simulator defaults.",
        "budget": float(budget),
        "n_seeds": int(n_seeds),
        "scenarios_per_seed": int(scenarios_per_seed),
        "parameterisations": [p.__dict__ for p in params],
        "results": results,
        "summary": summary,
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    _write_md(out, out_md)
    return out


def _write_md(out: Dict[str, Any], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Statistical Evaluation (Paired Bootstrap, CRN, 90% CIs)",
        "",
        f"- **Note:** {out['assumption']}",
        f"- **Budget:** Rs {out['budget']:,.0f} | **Seeds:** {out['n_seeds']} | **Scenarios/seed:** {out['scenarios_per_seed']}",
        f"- **Robustness (Planner mean > B1-B3 in both parameterisations):** {out['summary']['robust_planner_beats_b1_b3_mean_both']}",
        "",
    ]
    for res in out["results"]:
        lines.append(f"## Parameterisation: {res['param_set']} (rho={res['rho']}, demand_scale={res['demand_scale']}, sigma_scale={res['sigma_scale']})")
        lines.append("")
        lines.append("| strategy | mean uplift (INR) | CI90 lo | CI90 hi |")
        lines.append("| --- | ---: | ---: | ---: |")
        for r in res["rows"]:
            lines.append(f"| {r['strategy']} | {r['mean']:.2f} | {r['ci90_lo']:.2f} | {r['ci90_hi']:.2f} |")
        lines.append("")
        lines.append("Paired differences (Planner - baseline), 90% CI:")
        lines.append("")
        lines.append("| comparison | mean diff (INR) | CI90 lo | CI90 hi | planner better (CI>0) |")
        lines.append("| --- | ---: | ---: | ---: | :---: |")
        for c in res["comparisons"]:
            lines.append(f"| Planner - {c['planner_minus']} | {c['mean_diff']:.2f} | {c['ci90_lo']:.2f} | {c['ci90_hi']:.2f} | {c['planner_better_ci90']} |")
        lines.append("")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Paired bootstrap CIs across 50 seeds and two parameterisations")
    ap.add_argument("--budget", type=float, default=10_000_000.0)
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--scenarios", type=int, default=300)
    ap.add_argument("--history", type=str, default="data/history")
    ap.add_argument("--ground-truth", type=str, default="data/ground_truth.parquet")
    ap.add_argument("--out-md", type=str, default="docs/stats_cis.md")
    ap.add_argument("--out-json", type=str, default="docs/stats_cis.json")
    args = ap.parse_args()
    run_stats_suite(budget=args.budget, n_seeds=args.seeds, scenarios_per_seed=args.scenarios, history_dir=args.history, ground_truth_path=args.ground_truth, out_md=args.out_md, out_json=args.out_json)
    print(f"wrote {args.out_md} {args.out_json}")


if __name__ == "__main__":
    main()
