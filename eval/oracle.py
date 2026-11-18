"""Perfect-foresight (oracle) allocator and strategy regret analysis.

Oracle knows realized latent demand per scenario and allocates under the same
budget/cap/cash constraints and contract economics. Ground truth is read only
inside /eval. All numbers are simulator outputs under stated assumptions.
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
from optimizer.cell import Cell
from optimizer.cvar_lp import bruteforce_cvar
from optimizer.demo_phase2 import sim_cell_to_optimizer
from optimizer.planner import candidate_funnel
from optimizer.profit import per_scenario_profit
from optimizer.waterfill import waterfill
from sim.cells import get_candidate_cells

# Assumption: oracle optimizes expected incremental profit per scenario (knowing D_s) under cash budget, caps, and same economics (carry 12%/120d).
# Assumption: oracle allocation solved per scenario by greedy delta_m/k ranking on cash-positive cells (k=0 positive cells taken to cap), which is optimal for linear per-scenario dProfit with budget.
# Assumption: regret is (oracle_value - strategy_value) on same CRN held-out latent scenarios; % captured is strategy/oracle*100 (0 if oracle<=0).


@dataclass
class OracleStrategyResult:
    name: str
    mean_uplift: float
    mean_spoilage: float
    mean_spend: float
    pct_oracle_captured: float
    mean_regret: float


def latent_heldout_scenarios(
    ground_truth_path: str,
    sim_cells: Sequence[Any],
    s: int,
    seed: int,
    embargo_days: int = 14,
    rho: float = 0.6,
) -> np.ndarray:
    """Sample held-out latent demand scenarios (eval-only)."""
    gt = pd.read_parquet(ground_truth_path).copy()
    gt["date"] = pd.to_datetime(gt["date"]).dt.normalize()
    max_day = gt["date"].max()
    cutoff = max_day - pd.Timedelta(days=int(embargo_days))
    hold = gt[gt["date"] > cutoff].copy()
    if hold.empty:
        hold = gt.copy()
    cell_ids = [c.cell_id for c in sim_cells]
    groups = [c.region_group for c in sim_cells]
    rng = get_rng(int(seed), "eval:oracle:latent")
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
        mu = float(np.mean(vals))
        sdv = float(np.std(vals))
        sigma = float(np.clip(sdv / max(mu, 1e-6), 0.05, 0.8))
        z = f_load * z_g[:, g_index[groups[j]]] + i_scale * eps[:, j]
        demand[:, j] = np.maximum(mu * np.exp(sigma * z - 0.5 * sigma * sigma), 0.0)
        draw_idx = rng.integers(0, len(vals), size=s)
        empirical = vals[draw_idx]
        demand[:, j] = 0.7 * demand[:, j] + 0.3 * np.maximum(empirical, 0.0)
    return demand


def _profit_at_q(c: Cell, q: float, d: float, lock_days: int = 120, r_annual: float = 0.12) -> float:
    return float(per_scenario_profit(c, q, np.array([d], dtype=float), lock_days=lock_days, r_annual=r_annual)[0])


def oracle_allocate_scenario(
    cells: Sequence[Cell],
    d_s: np.ndarray,
    budget: float,
    lock_days: int = 120,
    r_annual: float = 0.12,
) -> np.ndarray:
    """Perfect-foresight allocation for a single realized demand vector."""
    n = len(cells)
    q = np.zeros(n, dtype=float)
    # free/zero-cash attractive cells: choose q in {0,cap} maximizing profit at known demand
    for i, c in enumerate(cells):
        if c.k <= 0.0:
            p0 = _profit_at_q(c, 0.0, float(d_s[i]), lock_days, r_annual)
            pcap = _profit_at_q(c, float(c.cap), float(d_s[i]), lock_days, r_annual)
            q[i] = float(c.cap) if pcap >= p0 else 0.0
    # cash cells: per-scenario dProfit is piecewise linear with kink at q=d -> optimal q in {0, min(d,cap), cap}
    remaining = float(budget)
    candidates: List[Tuple[float, float, int, float, float]] = []  # (gain_per_cash, gain, idx, q_target, cash)
    for i, c in enumerate(cells):
        if c.k <= 0.0 or c.cap <= 0:
            continue
        q_opt_d = float(min(max(d_s[i], 0.0), c.cap))
        options = [0.0, q_opt_d, float(c.cap)]
        best_q = 0.0
        best_pi = _profit_at_q(c, 0.0, float(d_s[i]), lock_days, r_annual)
        for qq in options:
            pi = _profit_at_q(c, qq, float(d_s[i]), lock_days, r_annual)
            if pi > best_pi + 1e-12 or (abs(pi - best_pi) <= 1e-12 and qq > best_q):
                best_pi = pi
                best_q = qq
        cash = c.k * best_q
        if best_q > 1e-9 and cash > 1e-12 and best_pi > _profit_at_q(c, 0.0, float(d_s[i]), lock_days, r_annual) + 1e-12:
            candidates.append((best_pi / cash, best_pi, i, best_q, cash))
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    for _, _, i, q_t, cash in candidates:
        if remaining <= 1e-9:
            break
        c = cells[i]
        if cash <= remaining + 1e-9:
            q[i] = q_t
            remaining -= c.k * q_t
        else:
            # partial affordable quantity: choose best among affordable breakpoints
            q_aff = remaining / c.k
            opts = [0.0, min(q_aff, float(min(max(d_s[i], 0.0), c.cap))), min(q_aff, float(c.cap))]
            opts = [min(o, q_aff) for o in opts]
            best_q = 0.0
            best_pi = _profit_at_q(c, 0.0, float(d_s[i]), lock_days, r_annual)
            for qq in opts:
                if qq < -1e-9:
                    continue
                pi = _profit_at_q(c, qq, float(d_s[i]), lock_days, r_annual)
                if pi > best_pi + 1e-12:
                    best_pi = pi
                    best_q = qq
            q[i] = best_q
            remaining -= c.k * best_q
    # budget safety
    spend = float(sum(cells[i].k * q[i] for i in range(n)))
    if spend > budget + 1e-6 and spend > 0:
        scale = budget / spend
        for i, c in enumerate(cells):
            if c.k > 0:
                q[i] *= scale
    return np.clip(q, 0.0, [float(c.cap) for c in cells])


def oracle_value(
    cells: Sequence[Cell],
    demand: np.ndarray,
    budget: float,
    lock_days: int = 120,
    r_annual: float = 0.12,
) -> Tuple[float, float, np.ndarray]:
    """Mean oracle uplift, mean oracle spoilage, per-scenario oracle profits."""
    d = np.asarray(demand, dtype=float)
    profits = np.zeros(d.shape[0], dtype=float)
    spoil = 0.0
    for s in range(d.shape[0]):
        q = oracle_allocate_scenario(cells, d[s], budget, lock_days=lock_days, r_annual=r_annual)
        for i, c in enumerate(cells):
            profits[s] += _profit_at_q(c, float(q[i]), float(d[s, i]), lock_days, r_annual)
            leftover = max(float(q[i]) - float(d[s, i]), 0.0)
            spoil += c.k * (1.0 - c.r) * leftover
    return float(np.mean(profits)), float(spoil / max(d.shape[0], 1)), profits


def _strategy_profit_metrics(cells: Sequence[Cell], q: np.ndarray, demand: np.ndarray, lock_days: int = 120, r_annual: float = 0.12):
    d = np.asarray(demand, dtype=float)
    total = np.zeros(d.shape[0], dtype=float)
    spoil = 0.0
    spend = 0.0
    for i, c in enumerate(cells):
        di = d[:, i]
        qi = float(max(q[i], 0.0))
        total += per_scenario_profit(c, qi, di, lock_days=lock_days, r_annual=r_annual)
        leftover = np.maximum(qi - di, 0.0)
        spoil += float(c.k) * (1.0 - float(c.r)) * float(np.mean(leftover))
        spend += float(c.k) * qi
    return float(np.mean(total)), float(spoil), float(spend), total


def run_oracle_eval(
    budget: float = 10_000_000.0,
    n_seeds: int = 50,
    scenarios_per_seed: int = 300,
    history_dir: str = "data/history",
    ground_truth_path: str = "data/ground_truth.parquet",
    out_md: str = "docs/oracle_results.md",
    out_json: str = "docs/oracle_results.json",
) -> Dict[str, Any]:
    """Evaluate oracle and strategies; report % oracle captured and regret."""
    sim_cells = get_candidate_cells()
    opt_cells = [sim_cell_to_optimizer(c) for c in sim_cells]
    lock_days, r_annual = 120, 0.12

    qmodel, cfg, feats_pool, feat_cols, _, _ = _build_forecast_model(history_dir=history_dir, embargo_days=14)

    strategies = ["B0_no_prebuy", "B1_equal_split", "B2_volume_proportional", "B3_mean_demand_newsvendor", "B4_correct_fractile_no_budget", "Planner"]
    uplift_by = {s: [] for s in strategies}
    spoil_by = {s: [] for s in strategies}
    spend_by = {s: [] for s in strategies}
    oracle_uplift_seeds: List[float] = []

    for seed in range(int(n_seeds)):
        heldout = latent_heldout_scenarios(ground_truth_path, sim_cells, scenarios_per_seed, seed=seed)
        # forecast scenarios for strategy decisions (CRN seed)
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
        forecast_demand, _ = scenarios_for_cells(qf, cell_ids=cell_ids, cell_groups=[c.region_group for c in sim_cells], s=scenarios_per_seed, quantiles=cfg.quantiles, intra_group_rho=0.6, seed=seed, label=f"oracle:forecast:{seed}")

        ov, _, _ = oracle_value(opt_cells, heldout, budget, lock_days=lock_days, r_annual=r_annual)
        oracle_uplift_seeds.append(ov)

        b0 = b0_no_prebuy(opt_cells, forecast_demand, budget)
        b1 = b1_equal_split(opt_cells, forecast_demand, budget)
        b2 = b2_volume_proportional(opt_cells, forecast_demand, budget)
        b3 = b3_mean_demand_newsvendor_ranked_by_margin(opt_cells, forecast_demand, budget)
        b4 = b4_correct_fractile_without_budget_awareness(opt_cells, forecast_demand, budget, lock_days, r_annual)
        selected = candidate_funnel(sim_cells, feats_pool, budget, constraints={"lock_days": lock_days, "r_annual": r_annual}, top_k=18) or sim_cells
        sel_idx = [i for i, c in enumerate(sim_cells) if c.cell_id in {s.cell_id for s in selected}]
        wf_sel = waterfill([opt_cells[i] for i in sel_idx], forecast_demand[:, sel_idx], budget=float(budget), lock_days=lock_days, r_annual=r_annual, reserve_fraction=0.03)
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
            m_u, m_sp, m_spend, _ = _strategy_profit_metrics(opt_cells, q, heldout, lock_days=lock_days, r_annual=r_annual)
            uplift_by[name].append(m_u)
            spoil_by[name].append(m_sp)
            spend_by[name].append(float(spend))

    oracle_mean = float(np.mean(oracle_uplift_seeds))
    results: List[Dict[str, Any]] = []
    for name in strategies:
        mu = float(np.mean(uplift_by[name]))
        pct = (mu / oracle_mean * 100.0) if oracle_mean > 1e-9 else 0.0
        regret = float(oracle_mean - mu)
        results.append({
            "strategy": name,
            "mean_uplift": mu,
            "mean_spoilage": float(np.mean(spoil_by[name])),
            "mean_spend": float(np.mean(spend_by[name])),
            "pct_oracle_captured": float(pct),
            "mean_regret": regret,
        })

    out = {
        "assumption": "Simulator outputs under stated assumptions, not TBO real P&L. Oracle has perfect foresight of realized latent demand per scenario (eval-only).",
        "budget": float(budget),
        "n_seeds": int(n_seeds),
        "scenarios_per_seed": int(scenarios_per_seed),
        "oracle_mean_uplift": oracle_mean,
        "results": results,
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    _write_md(out_md, out)
    return out


def _write_md(path: str, out: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Oracle (Perfect-Foresight) Evaluation",
        "",
        f"- **Note:** {out['assumption']}",
        f"- **Budget:** Rs {out['budget']:,.0f} | **Seeds:** {out['n_seeds']} | **Scenarios/seed:** {out['scenarios_per_seed']}",
        f"- **Oracle mean uplift (INR):** {out['oracle_mean_uplift']:.2f}",
        "",
        "| strategy | mean uplift (INR) | % oracle captured | mean regret (INR) | mean spoilage (INR) | mean spend (INR) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in out["results"]:
        lines.append(
            f"| {r['strategy']} | {r['mean_uplift']:.2f} | {r['pct_oracle_captured']:.2f}% | {r['mean_regret']:.2f} | {r['mean_spoilage']:.2f} | {r['mean_spend']:.2f} |"
        )
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Perfect-foresight oracle vs strategies")
    ap.add_argument("--budget", type=float, default=10_000_000.0)
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--scenarios", type=int, default=300)
    ap.add_argument("--history", type=str, default="data/history")
    ap.add_argument("--ground-truth", type=str, default="data/ground_truth.parquet")
    ap.add_argument("--out-md", type=str, default="docs/oracle_results.md")
    ap.add_argument("--out-json", type=str, default="docs/oracle_results.json")
    args = ap.parse_args()
    run_oracle_eval(budget=args.budget, n_seeds=args.seeds, scenarios_per_seed=args.scenarios, history_dir=args.history, ground_truth_path=args.ground_truth, out_md=args.out_md, out_json=args.out_json)
    print(f"wrote {args.out_md} {args.out_json}")


if __name__ == "__main__":
    main()
