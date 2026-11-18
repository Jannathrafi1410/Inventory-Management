"""Evaluate planner and baselines B0-B4 on held-out latent scenarios with common random numbers.

Reads latent ground truth only inside /eval for realized evaluation scenarios (non-negotiable rule).
Writes a markdown results table with mean and 90% bootstrap CI across seeds.
All numbers are simulator outputs under stated assumptions, not TBO real P&L.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config.settings import get_rng
from forecast.features import FEATURE_COLUMNS, build_point_in_time_features
from forecast.quantile_model import QuantileDemandModel, QuantileModelConfig
from forecast.scenarios import scenarios_for_cells
from forecast.unconstrain import em_unconstrain_normal
from optimizer.baselines import (
    b0_no_prebuy,
    b1_equal_split,
    b2_volume_proportional,
    b3_mean_demand_newsvendor_ranked_by_margin,
    b4_correct_fractile_without_budget_awareness,
    evaluate_strategy_profit,
)
from optimizer.demo_phase2 import sim_cell_to_optimizer
from optimizer.waterfill import waterfill
from sim.cells import get_candidate_cells


def _bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.10, seed: int = 0) -> Tuple[float, float]:
    """90% bootstrap CI (percentile)."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = np.mean(v[idx], axis=1)
    lo = float(np.percentile(means, 100 * (alpha / 2)))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def _build_forecast_model(history_dir: str = "data/history", embargo_days: int = 14):
    """Train quantile forecast model on censored history up to an embargoed cutoff (no leakage)."""
    feats = build_point_in_time_features(decision_ts=None, history_dir=history_dir).copy()
    feats["stay_date"] = pd.to_datetime(feats["stay_date"]).dt.normalize()
    feats["is_sold_out"] = feats["sell_out_obs"].astype(bool)
    feats["y_censored"] = feats["observed_sales"].astype(float)
    feats["s_cap"] = feats["capacity_obs"].astype(float).replace(0, np.nan).fillna(feats["y_censored"])
    uncon = np.zeros(len(feats), dtype=float)
    for _cid, idx in feats.groupby("cell_id").indices.items():
        part = feats.iloc[idx]
        res = em_unconstrain_normal(
            part["y_censored"].to_numpy(),
            part["is_sold_out"].to_numpy(),
            s=part["s_cap"].to_numpy(),
            max_iter=100,
            tol=1e-3,
        )
        uncon[idx] = res.unconstrained
    feats["y_target_uncon"] = np.maximum(uncon, feats["y_censored"])
    feat_cols = [c for c in FEATURE_COLUMNS if c in feats.columns]
    for c in feat_cols:
        feats[c] = pd.to_numeric(feats[c], errors="coerce").fillna(0.0)
    max_day = feats["stay_date"].max()
    cutoff = max_day - pd.Timedelta(days=int(embargo_days))
    train_feats = feats[feats["stay_date"] <= cutoff].copy()
    if len(train_feats) < 200:
        train_feats = feats.copy()
        cutoff = max_day
    cfg = QuantileModelConfig(
        quantiles=(0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95),
        n_estimators=120,
        num_leaves=31,
        learning_rate=0.05,
        random_state=42,
    )
    qmodel = QuantileDemandModel(cfg)
    qmodel.fit(train_feats[feat_cols], train_feats["y_target_uncon"].to_numpy(), feature_columns=feat_cols)
    latest_pool = feats[feats["stay_date"] <= cutoff]
    if latest_pool.empty:
        latest_pool = feats
    return qmodel, cfg, latest_pool, feat_cols, cutoff, max_day


def _forecast_scenario_matrix(qmodel, cfg, feats_pool: pd.DataFrame, feat_cols: List[str], sim_cells, s: int, seed: int) -> np.ndarray:
    """Build forecast-driven scenario matrix [S,n] in sim_cells order (censored history only)."""
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
    demand_s, _ = scenarios_for_cells(
        qf,
        cell_ids=cell_ids,
        cell_groups=[c.region_group for c in sim_cells],
        s=int(s),
        quantiles=cfg.quantiles,
        intra_group_rho=0.6,
        seed=int(seed),
        label=f"eval:forecast:scenarios:{seed}",
    )
    return demand_s


def _latent_heldout_scenarios(
    ground_truth_path: str,
    sim_cells,
    s: int,
    seed: int,
    embargo_days: int = 14,
) -> Tuple[np.ndarray, str]:
    """Generate held-out evaluation scenarios from latent demand (eval-only, ground truth).

    Assumption: samples empirical latent demand from held-out period (last embargo_days) with seeded
    group-correlated lognormal jitter (rho=0.6) to preserve diversification realism; common random
    numbers per seed across strategies.
    """
    gt = pd.read_parquet(ground_truth_path).copy()
    gt["date"] = pd.to_datetime(gt["date"]).dt.normalize()
    max_day = gt["date"].max()
    cutoff = max_day - pd.Timedelta(days=int(embargo_days))
    hold = gt[gt["date"] > cutoff].copy()
    if hold.empty:
        hold = gt.copy()
    cell_ids = [c.cell_id for c in sim_cells]
    groups = [c.region_group for c in sim_cells]
    rng = get_rng(int(seed), "eval:latent:heldout")
    rho = 0.6
    unique_groups = list(dict.fromkeys(groups))
    g_index = {g: i for i, g in enumerate(unique_groups)}
    z_g = rng.standard_normal(size=(s, len(unique_groups)))
    eps = rng.standard_normal(size=(s, len(cell_ids)))
    f_load = float(np.sqrt(rho))
    i_scale = float(np.sqrt(max(1.0 - rho, 1e-6)))
    demand = np.zeros((s, len(cell_ids)), dtype=float)
    for j, cid in enumerate(cell_ids):
        vals = hold.loc[hold["cell_id"] == cid, "latent_demand"].astype(float).to_numpy()
        if len(vals) == 0:
            vals = gt.loc[gt["cell_id"] == cid, "latent_demand"].astype(float).to_numpy()
        if len(vals) == 0:
            vals = np.array([1.0], dtype=float)
        mu = float(np.mean(vals))
        sd = float(np.std(vals))
        sigma = float(np.clip(sd / max(mu, 1e-6), 0.05, 0.8))
        z = f_load * z_g[:, g_index[groups[j]]] + i_scale * eps[:, j]
        demand[:, j] = np.maximum(mu * np.exp(sigma * z - 0.5 * sigma * sigma), 0.0)
        draw_idx = rng.integers(0, len(vals), size=s)
        empirical = vals[draw_idx]
        demand[:, j] = 0.7 * demand[:, j] + 0.3 * np.maximum(empirical, 0.0)
    hold_start = str(pd.to_datetime(hold["date"].min()).date()) if len(hold) else ""
    return demand, hold_start


def run_evaluation(
    budget: float = 10_000_000.0,
    n_seeds: int = 50,
    scenarios_per_seed: int = 300,
    history_dir: str = "data/history",
    ground_truth_path: str = "data/ground_truth.parquet",
    out_path: str = "docs/baseline_eval.md",
) -> Dict[str, object]:
    """Evaluate B0-B4 and planner on same held-out latent scenarios across seeds (CRN)."""
    sim_cells = get_candidate_cells()
    opt_cells = [sim_cell_to_optimizer(c) for c in sim_cells]
    lock_days, r_annual = 120, 0.12
    embargo_days = 14

    qmodel, cfg, feats_pool, feat_cols, cutoff, max_day = _build_forecast_model(history_dir=history_dir, embargo_days=embargo_days)

    strategies = [
        "B0_no_prebuy",
        "B1_equal_split",
        "B2_volume_proportional",
        "B3_mean_demand_newsvendor",
        "B4_correct_fractile_no_budget",
        "Planner",
    ]
    profits_by: Dict[str, List[float]] = {s: [] for s in strategies}
    spends_by: Dict[str, List[float]] = {s: [] for s in strategies}
    spoil_by: Dict[str, List[float]] = {s: [] for s in strategies}
    fill_by: Dict[str, List[float]] = {s: [] for s in strategies}
    roi_by: Dict[str, List[float]] = {s: [] for s in strategies}

    hold_start_used = ""
    from optimizer.planner import candidate_funnel

    for seed in range(int(n_seeds)):
        heldout_demand, hold_start = _latent_heldout_scenarios(
            ground_truth_path, sim_cells, scenarios_per_seed, seed=seed, embargo_days=embargo_days
        )
        hold_start_used = hold_start or hold_start_used
        forecast_demand = _forecast_scenario_matrix(qmodel, cfg, feats_pool, feat_cols, sim_cells, scenarios_per_seed, seed=seed)

        b0 = b0_no_prebuy(opt_cells, forecast_demand, budget)
        b1 = b1_equal_split(opt_cells, forecast_demand, budget)
        b2 = b2_volume_proportional(opt_cells, forecast_demand, budget)
        b3 = b3_mean_demand_newsvendor_ranked_by_margin(opt_cells, forecast_demand, budget)
        b4 = b4_correct_fractile_without_budget_awareness(opt_cells, forecast_demand, budget, lock_days, r_annual)

        # Planner: funnel to screened/ranked candidates then budget-aware water-filling with small reserve
        # Assumption: reserve_fraction=0.03 leaves cash unspent (capital discipline) while focusing on best delta_m/k.
        selected = candidate_funnel(sim_cells, feats_pool, budget, constraints={"lock_days": lock_days, "r_annual": r_annual}, top_k=18)
        if not selected:
            selected = sim_cells
        sel_idx = [i for i, c in enumerate(sim_cells) if c.cell_id in {s.cell_id for s in selected}]
        sel_opt = [opt_cells[i] for i in sel_idx]
        sel_demand = forecast_demand[:, sel_idx]
        wf_sel = waterfill(sel_opt, sel_demand, budget=float(budget), lock_days=lock_days, r_annual=r_annual, reserve_fraction=0.03)
        pq = np.zeros(len(opt_cells), dtype=float)
        for j, i in enumerate(sel_idx):
            pq[i] = wf_sel.quantities[j]
        pspend = float(np.sum(pq * np.array([c.k for c in opt_cells], dtype=float)))

        allocs = {
            "B0_no_prebuy": (b0.quantities, b0.spend),
            "B1_equal_split": (b1.quantities, b1.spend),
            "B2_volume_proportional": (b2.quantities, b2.spend),
            "B3_mean_demand_newsvendor": (b3.quantities, b3.spend),
            "B4_correct_fractile_no_budget": (b4.quantities, b4.spend),
            "Planner": (pq, pspend),
        }
        for name, (q, spend) in allocs.items():
            spend_f = float(spend)
            if spend_f > budget + 1e-6:
                raise AssertionError(f"{name} spent above budget: {spend_f} > {budget}")
            m = evaluate_strategy_profit(opt_cells, q, heldout_demand, lock_days=lock_days, r_annual=r_annual)
            prof = float(m["incremental_profit_mean"])
            profits_by[name].append(prof)
            spends_by[name].append(spend_f)
            spoil_by[name].append(float(m["expected_cash_spoilage"]))
            fill_by[name].append(float(m["fill_rate_uplift"]))
            roi_by[name].append((prof / spend_f) if spend_f > 1e-9 else 0.0)

    rows = []
    for name in strategies:
        p = np.array(profits_by[name], dtype=float)
        lo, hi = _bootstrap_ci(p, n_boot=2000, alpha=0.10, seed=123)
        s = np.array(spends_by[name], dtype=float)
        rows.append(
            {
                "strategy": name,
                "profit_mean": float(np.mean(p)),
                "profit_ci90_lo": lo,
                "profit_ci90_hi": hi,
                "spend_mean": float(np.mean(s)),
                "spend_max": float(np.max(s)),
                "spoilage_mean": float(np.mean(spoil_by[name])),
                "fill_uplift_mean": float(np.mean(fill_by[name])),
                "roi_on_cash_mean": float(np.mean(roi_by[name])),
                "sometimes_unspent": bool(np.any(s < budget - 1e-6)),
            }
        )

    planner = next(r for r in rows if r["strategy"] == "Planner")
    gate_notes: List[str] = []
    never_above = all(r["spend_max"] <= budget + 1e-6 for r in rows)
    gate_notes.append(f"never_spend_above_budget={never_above}")
    gate_notes.append(f"planner_sometimes_unspent={planner['sometimes_unspent']}")
    beats = {}
    for b in ["B1_equal_split", "B2_volume_proportional", "B3_mean_demand_newsvendor"]:
        br = next(r for r in rows if r["strategy"] == b)
        non_overlap = planner["profit_ci90_lo"] > br["profit_ci90_hi"]
        clearly = planner["profit_mean"] > br["profit_mean"] and planner["profit_ci90_lo"] >= br["profit_ci90_lo"]
        ok = bool(non_overlap or clearly)
        beats[b] = {
            "non_overlapping_or_favourable": ok,
            "planner_mean": planner["profit_mean"],
            "baseline_mean": br["profit_mean"],
            "planner_ci": [planner["profit_ci90_lo"], planner["profit_ci90_hi"]],
            "baseline_ci": [br["profit_ci90_lo"], br["profit_ci90_hi"]],
        }
        gate_notes.append(f"beats_{b}={ok}")

    gate_pass = bool(never_above and planner["sometimes_unspent"] and all(v["non_overlapping_or_favourable"] for v in beats.values()))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write("# Baseline & Planner Evaluation (held-out latent scenarios, eval-only)\n\n")
        f.write("- **Note:** All numbers are simulator outputs under stated assumptions, not TBO real P&L.\n")
        f.write(
            "- **Setup:** Strategies decided on forecast scenarios from censored history (embargoed time split); "
            "realised incremental profit evaluated on latent-demand held-out scenarios (ground truth read only in /eval). "
            "Common random numbers per seed across all strategies.\n"
        )
        f.write(
            "- **Assumption (planner):** planner applies candidate funnel (screen delta_m>0, volume/sellout/discount/event; "
            "rank by delta_m/k with k=0 first, top 18) then budget-aware water-filling with reserve_fraction=0.03 to preserve "
            "capital discipline; numbers are simulator outputs under stated assumptions.\n"
        )
        f.write(
            f"- **Budget:** Rs {budget:,.0f} | **Seeds:** {n_seeds} | **Scenarios/seed:** {scenarios_per_seed} | "
            f"**Held-out start:** {hold_start_used} | **Forecast cutoff:** {str(pd.to_datetime(cutoff).date())} | **Embargo days:** {embargo_days}\n"
        )
        f.write(f"- **EXIT GATE:** {'PASS' if gate_pass else 'FAIL'} ({'; '.join(gate_notes)})\n\n")
        f.write("## Results (mean and 90% bootstrap CI over seeds)\n\n")
        f.write("| strategy | profit_mean | profit_CI90_lo | profit_CI90_hi | spend_mean | spend_max | spoilage_mean | fill_uplift_mean | roi_on_cash_mean | sometimes_unspent |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |\n")
        for r in rows:
            f.write(
                f"| {r['strategy']} | {r['profit_mean']:.2f} | {r['profit_ci90_lo']:.2f} | {r['profit_ci90_hi']:.2f} | "
                f"{r['spend_mean']:.2f} | {r['spend_max']:.2f} | {r['spoilage_mean']:.2f} | {r['fill_uplift_mean']:.6f} | "
                f"{r['roi_on_cash_mean']:.6f} | {str(r['sometimes_unspent'])} |\n"
            )
        f.write("\n## Planner vs B1-B3 (profit CIs)\n\n")
        for b, info in beats.items():
            f.write(
                f"- {b}: planner_mean={info['planner_mean']:.2f} CI90={info['planner_ci'][0]:.2f}..{info['planner_ci'][1]:.2f} "
                f"vs baseline_mean={info['baseline_mean']:.2f} CI90={info['baseline_ci'][0]:.2f}..{info['baseline_ci'][1]:.2f} "
                f"=> favourable={info['non_overlapping_or_favourable']}\n"
            )

    return {
        "rows": rows,
        "beats": beats,
        "gate_pass": gate_pass,
        "out_path": str(out),
        "budget": budget,
        "n_seeds": n_seeds,
        "scenarios_per_seed": scenarios_per_seed,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate B0-B4 and planner on held-out latent scenarios")
    ap.add_argument("--budget", type=float, default=10_000_000.0)
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--scenarios", type=int, default=300)
    ap.add_argument("--history", type=str, default="data/history")
    ap.add_argument("--ground-truth", type=str, default="data/ground_truth.parquet")
    ap.add_argument("--out", type=str, default="docs/baseline_eval.md")
    args = ap.parse_args()
    res = run_evaluation(
        budget=args.budget,
        n_seeds=args.seeds,
        scenarios_per_seed=args.scenarios,
        history_dir=args.history,
        ground_truth_path=args.ground_truth,
        out_path=args.out,
    )
    print(f"gate_pass={res['gate_pass']} out={res['out_path']}")
    for r in res["rows"]:
        print(r["strategy"], r["profit_mean"], r["profit_ci90_lo"], r["profit_ci90_hi"], "spend", r["spend_mean"])


if __name__ == "__main__":
    main()
