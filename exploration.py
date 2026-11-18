"""Exploration allocation: Thompson sampling over delta_m posterior.

Allocates 2-5% of budget by Thompson sampling over posterior of delta_m for cells
with little history, using similarity-based priors shrunk toward cluster mean.
All numbers are simulator outputs under stated assumptions, not TBO real P&L.
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
from optimizer.cell import Cell
from optimizer.demo_phase2 import sim_cell_to_optimizer
from optimizer.waterfill import waterfill
from sim.cells import CandidateCell, get_candidate_cells

# Assumption: "little history" cells are those with observed history count <= median history count across cells (from censored history features).
# Assumption: delta_m posterior is Normal(mean, var) with prior mean = (1-shrink)*cell_delta_m_prior + shrink*cluster_mean_delta_m, prior var from cluster dispersion; likelihood proxy from historical observed sales/capacity variability gives n_eff observations.
# Assumption: similarity clusters defined by (region_group, segment, product_type); shrinkage weight = 0.35 toward cluster mean.
# Assumption: Thompson sampling allocates 3% of total budget (within 2-5%) across eligible low-history cash-positive cells proportional to sampled positive edge, capped by cell cap and remaining budget.
# Assumption: exploration can run alongside main allocator by reserving exploration budget first then allocating remainder with standard water-filling.


@dataclass
class Posterior:
    cell_id: str
    mean: float
    var: float
    n_eff: float
    cluster: str
    cluster_mean: float
    eligible: bool


def _history_counts(history_dir: str = "data/history") -> Dict[str, int]:
    try:
        b = pd.read_parquet(Path(history_dir) / "booking.parquet")
        if "status" in b.columns:
            b = b[b["status"] == "confirmed"]
        b = b.copy()
        b["stay_date"] = pd.to_datetime(b["checkin"]).dt.normalize()
        b["cell_id"] = b["hotel_id"].astype(str) + "|" + b["rate_plan"].astype(str)
        cnt = b.groupby("cell_id").size().to_dict()
        # map to candidate cell ids by prefix hotel+rate not directly matching; fallback: use per-candidate proxy via destination presence
        return {str(k): int(v) for k, v in cnt.items()}
    except Exception:
        return {}


def _candidate_history_proxy(sim_cells: Sequence[CandidateCell], history_dir: str = "data/history") -> Dict[str, float]:
    """Proxy history richness per candidate cell from censored booking volume by destination/rate pattern.

    Assumption: when direct cell-id mapping unavailable, use destination-level confirmed booking count as richness proxy.
    """
    try:
        b = pd.read_parquet(Path(history_dir) / "booking.parquet")
        if "status" in b.columns:
            b = b[b["status"] == "confirmed"]
        b = b.copy()
        b["destination_city"] = b["hotel_id"].astype(str).str.replace("H_", "", regex=False)
        by_dest = b.groupby("destination_city").size().to_dict()
        out = {}
        for c in sim_cells:
            base = float(by_dest.get(c.destination_city, 0.0))
            # segment/product adjustment proxy
            adj = 1.0
            if c.segment == "event":
                adj = 0.7
            if c.product_type == "air":
                adj = 0.8
            out[c.cell_id] = base * adj
        return out
    except Exception:
        return {c.cell_id: 0.0 for c in sim_cells}


def build_posteriors(
    sim_cells: Sequence[CandidateCell],
    shrinkage: float = 0.35,
    history_dir: str = "data/history",
) -> List[Posterior]:
    """Build similarity-shrunk posteriors for delta_m."""
    opt = [sim_cell_to_optimizer(c) for c in sim_cells]
    delta = np.array([float(c.delta_m()) for c in opt], dtype=float)
    clusters = [f"{c.region_group}|{c.segment}|{c.product_type}" for c in sim_cells]
    hist_proxy = _candidate_history_proxy(sim_cells, history_dir=history_dir)
    counts = np.array([hist_proxy.get(c.cell_id, 0.0) for c in sim_cells], dtype=float)
    median_cnt = float(np.median(counts)) if len(counts) else 0.0

    # cluster stats
    cluster_vals: Dict[str, List[float]] = {}
    for cl, v in zip(clusters, delta):
        cluster_vals.setdefault(cl, []).append(v)
    cluster_mean = {cl: float(np.mean(vs)) for cl, vs in cluster_vals.items()}
    cluster_var = {cl: float(np.var(vs) + 1e6) for cl, vs in cluster_vals.items()}  # floor variance in INR^2

    posteriors: List[Posterior] = []
    for i, c in enumerate(sim_cells):
        cl = clusters[i]
        cm = cluster_mean.get(cl, float(np.mean(delta)))
        prior_mean = (1.0 - shrinkage) * float(delta[i]) + shrinkage * cm
        n_eff = float(max(counts[i], 1.0))
        # likelihood precision proxy: more history => lower variance
        obs_var = float(cluster_var.get(cl, 1e7) / max(np.sqrt(n_eff), 1.0))
        prior_var = float(cluster_var.get(cl, 1e7))
        post_var = 1.0 / (1.0 / prior_var + n_eff / max(obs_var, 1e3))
        post_mean = post_var * (prior_mean / prior_var + n_eff * float(delta[i]) / max(obs_var, 1e3))
        eligible = bool(counts[i] <= median_cnt and opt[i].k > 0 and opt[i].delta_m() > 0)
        posteriors.append(Posterior(c.cell_id, float(post_mean), float(max(post_var, 1e4)), n_eff, cl, cm, eligible))
    return posteriors


def thompson_sample_exploration(
    sim_cells: Sequence[CandidateCell],
    budget: float,
    exploration_fraction: float = 0.03,
    seed: int = 42,
    history_dir: str = "data/history",
) -> Dict[str, Any]:
    """Allocate exploration budget fraction via Thompson sampling on delta_m posterior."""
    frac = float(np.clip(exploration_fraction, 0.02, 0.05))
    exp_budget = float(budget) * frac
    post = build_posteriors(sim_cells, history_dir=history_dir)
    rng = get_rng(seed, "exploration:thompson")
    opt = [sim_cell_to_optimizer(c) for c in sim_cells]
    samples = []
    for i, p in enumerate(post):
        s = float(rng.normal(p.mean, np.sqrt(p.var)))
        samples.append(s)
    samples_arr = np.array(samples, dtype=float)

    eligible_idx = [i for i, p in enumerate(post) if p.eligible]
    alloc = np.zeros(len(sim_cells), dtype=float)
    spend = 0.0
    if eligible_idx and exp_budget > 0:
        scores = np.array([max(samples_arr[i], 0.0) for i in eligible_idx], dtype=float)
        if scores.sum() <= 0:
            scores = np.ones_like(scores)
        weights = scores / scores.sum()
        cash_alloc = exp_budget * weights
        for j, i in enumerate(eligible_idx):
            c = opt[i]
            if c.k <= 0:
                continue
            q = min(float(c.cap), float(cash_alloc[j]) / float(c.k))
            alloc[i] = q
            spend += c.k * q
        # scale down if over
        if spend > exp_budget + 1e-6 and spend > 0:
            scale = exp_budget / spend
            for i in eligible_idx:
                if opt[i].k > 0:
                    alloc[i] *= scale
            spend = float(sum(opt[i].k * alloc[i] for i in range(len(opt))))
    trace = []
    for i, p in enumerate(post):
        trace.append({
            "cell_id": p.cell_id,
            "cluster": p.cluster,
            "eligible_little_history": p.eligible,
            "posterior_mean_delta_m": p.mean,
            "posterior_var": p.var,
            "thompson_sample": float(samples_arr[i]),
            "explore_units": float(alloc[i]),
            "explore_cash_inr": float(opt[i].k * alloc[i]),
        })
    return {
        "budget": float(budget),
        "exploration_fraction": frac,
        "exploration_budget_inr": exp_budget,
        "exploration_spend_inr": spend,
        "n_eligible": len(eligible_idx),
        "allocations": trace,
    }


def integrated_allocate(
    sim_cells: Sequence[CandidateCell],
    demand: np.ndarray,
    budget: float,
    exploration_fraction: float = 0.03,
    seed: int = 42,
    history_dir: str = "data/history",
) -> Dict[str, Any]:
    """Run exploration first (reserved budget) then main water-filling on remainder."""
    exp = thompson_sample_exploration(sim_cells, budget, exploration_fraction=exploration_fraction, seed=seed, history_dir=history_dir)
    exp_q = np.array([a["explore_units"] for a in exp["allocations"]], dtype=float)
    opt = [sim_cell_to_optimizer(c) for c in sim_cells]
    exp_spend = float(sum(opt[i].k * exp_q[i] for i in range(len(opt))))
    remaining_budget = max(float(budget) - exp_spend, 0.0)
    # reduce caps by explored units
    adj_cells = []
    for i, c in enumerate(opt):
        cap_rem = max(int(c.cap) - int(np.ceil(exp_q[i])), 0)
        adj_cells.append(Cell(c.id, c.p, c.c_d, c.c_b, c.k, c.r, c.a, c.g, cap_rem, c.contract_type, c.group))
    wf = waterfill(adj_cells, demand, budget=remaining_budget, lock_days=120, r_annual=0.12, reserve_fraction=0.03)
    main_q = np.array(wf.quantities, dtype=float)
    total_q = exp_q + main_q
    total_spend = float(sum(opt[i].k * total_q[i] for i in range(len(opt))))
    return {"exploration": exp, "main_spend_inr": float(wf.spend), "total_spend_inr": total_spend, "total_units": total_q.tolist(), "lambda_star": float(wf.lambda_star)}


def run_exploration(
    budget: float = 10_000_000.0,
    n_scenarios: int = 300,
    seed: int = 42,
    history_dir: str = "data/history",
    out_md: str = "docs/exploration.md",
    out_json: str = "docs/exploration.json",
) -> Dict[str, Any]:
    sim_cells = get_candidate_cells()
    # demand scenarios for integrated allocation
    qmodel, cfg, feats_pool, feat_cols, _, _ = _build_forecast_model(history_dir=history_dir, embargo_days=14)
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
    demand_s, _ = scenarios_for_cells(qf, cell_ids=cell_ids, cell_groups=[c.region_group for c in sim_cells], s=n_scenarios, quantiles=cfg.quantiles, intra_group_rho=0.6, seed=seed, label="explore:forecast")

    result = integrated_allocate(sim_cells, demand_s, budget, exploration_fraction=0.03, seed=seed, history_dir=history_dir)
    result["assumption"] = "Simulator outputs under stated assumptions, not TBO real P&L. Thompson sampling over Normal posterior of delta_m for low-history cash-positive cells; similarity clusters (region,segment,product) with shrinkage 0.35 toward cluster mean; exploration budget 3% (2-5% allowed) reserved before main water-filling."
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_md(out_md, result)
    return result


def _write_md(path: str, out: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    exp = out["exploration"]
    lines = [
        "# Exploration Allocation (Thompson Sampling on delta_m Posterior)",
        "",
        f"- **Note:** {out.get('assumption','')}",
        f"- **Total budget:** Rs {exp['budget']:,.0f} | **Exploration fraction:** {exp['exploration_fraction']*100:.1f}% | **Exploration spend:** Rs {exp['exploration_spend_inr']:,.2f} | **Eligible low-history cells:** {exp['n_eligible']}",
        f"- **Main allocator spend:** Rs {out['main_spend_inr']:,.2f} | **Total spend:** Rs {out['total_spend_inr']:,.2f} | **lambda:** {out['lambda_star']:.6f}",
        "",
        "| cell | cluster | eligible | posterior mean delta_m (INR) | posterior var | Thompson sample | explore units | explore cash (INR) |",
        "| --- | --- | :---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    rows = sorted(exp["allocations"], key=lambda r: (not r["eligible_little_history"], -r["explore_cash_inr"]))
    for r in rows:
        lines.append(f"| {r['cell_id']} | {r['cluster']} | {r['eligible_little_history']} | {r['posterior_mean_delta_m']:.2f} | {r['posterior_var']:.2f} | {r['thompson_sample']:.2f} | {r['explore_units']:.2f} | {r['explore_cash_inr']:.2f} |")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Thompson-sampling exploration allocation (2-5% budget)")
    ap.add_argument("--budget", type=float, default=10_000_000.0)
    ap.add_argument("--scenarios", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--history", type=str, default="data/history")
    ap.add_argument("--out-md", type=str, default="docs/exploration.md")
    ap.add_argument("--out-json", type=str, default="docs/exploration.json")
    args = ap.parse_args()
    run_exploration(budget=args.budget, n_scenarios=args.scenarios, seed=args.seed, history_dir=args.history, out_md=args.out_md, out_json=args.out_json)
    print(f"wrote {args.out_md} {args.out_json}")


if __name__ == "__main__":
    main()
