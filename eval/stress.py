"""Stress testing: re-solve and evaluate under adverse scenarios.

Compares risk-neutral (water-filling) vs risk-averse (mean-CVaR LP) plans on
stressed forecast decision scenarios and realized stressed held-out latent scenarios.
All numbers are simulator outputs under stated assumptions, not TBO real P&L.
Ground truth is read only inside /eval.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config.settings import get_rng
from eval.run_baselines import _build_forecast_model
from forecast.scenarios import scenarios_for_cells
from optimizer.cell import Cell
from optimizer.cvar_lp import bruteforce_cvar, solve_mean_cvar_lp
from optimizer.demo_phase2 import sim_cell_to_optimizer
from optimizer.waterfill import waterfill
from sim.cells import CandidateCell, get_candidate_cells

# Assumption: decisions are solved on stressed embargoed forecast scenarios; realized metrics use stressed held-out latent scenarios (eval-only, last embargo window).
# Assumption: bot contamination inflates decision demand signal (forecast quantiles/scenarios) by +20% only; realized latent demand is unchanged by bots.
# Assumption: correlation x2 stress amplifies shared co-movement by doubling the common component and using higher intra-group rho (0.95) for forecast scenarios and held-out correlation.
# Assumption: volatility x1.6 scales dispersion around per-cell mean for forecast scenarios and latent sigma for held-out draws.
# Assumption: demand +/-30% scales both decision and realized demand levels multiplicatively.
# Assumption: sell-out rate shocks shift fallback availability a by -0.20/+0.10 (clipped to [0,1]) in both decision and realized economics.
# Assumption: event cancellation scales event-segment demand to 35% and mixed-segment demand to 85% in decision and realized scenarios.
# Assumption: FX +10% raises dynamic net cost c_d by 10% (cost pressure) in decision and realized economics.
# Assumption: supplier default targets the largest cash commitment (max k*cap) setting fallback availability a=0 and refund r=0 for that cell.
# Assumption: competitor price war lowers sell price p by 8% in decision and realized economics.
# Assumption: risk-averse LP is solved on up to 150 decision scenarios for tractability but all plans are evaluated on the full scenario set.

STRESS_CASES: List[str] = [
    "demand_-30pct",
    "demand_+30pct",
    "volatility_x1.6",
    "correlation_x2",
    "sellout_a_-0.20",
    "sellout_a_+0.10",
    "event_cancellation",
    "fx_+10pct",
    "supplier_default_largest",
    "competitor_price_war_-8pct",
    "bot_contamination_20pct_search",
]


@dataclass
class StressMetrics:
    """Aggregate plan metrics on a scenario set."""

    expected_uplift: float
    cvar95: float
    cash_spoilage: float
    spend: float


def latent_heldout_scenarios(
    ground_truth_path: str,
    sim_cells: Sequence[CandidateCell],
    s: int,
    seed: int,
    embargo_days: int = 14,
    demand_scale: float = 1.0,
    sigma_scale: float = 1.0,
    rho: float = 0.6,
) -> np.ndarray:
    """Sample held-out latent demand scenarios from ground truth (eval-only)."""
    gt = pd.read_parquet(ground_truth_path).copy()
    gt["date"] = pd.to_datetime(gt["date"]).dt.normalize()
    max_day = gt["date"].max()
    cutoff = max_day - pd.Timedelta(days=int(embargo_days))
    hold = gt[gt["date"] > cutoff].copy()
    if hold.empty:
        hold = gt.copy()
    cell_ids = [c.cell_id for c in sim_cells]
    groups = [c.region_group for c in sim_cells]
    rng = get_rng(int(seed), "eval:stress:latent")
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


def _forecast_quantiles(qmodel: Any, feats_pool: pd.DataFrame, feat_cols: List[str], sim_cells: Sequence[CandidateCell]) -> pd.DataFrame:
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
    return qf


def _scenarios_from_qf(
    qf: pd.DataFrame,
    sim_cells: Sequence[CandidateCell],
    s: int,
    seed: int,
    quantiles: Sequence[float],
    rho: float = 0.6,
    label: str = "stress:forecast",
) -> np.ndarray:
    demand_s, _ = scenarios_for_cells(
        qf,
        cell_ids=[c.cell_id for c in sim_cells],
        cell_groups=[c.region_group for c in sim_cells],
        s=int(s),
        quantiles=quantiles,
        intra_group_rho=float(np.clip(rho, 0.0, 0.99)),
        seed=int(seed),
        label=label,
    )
    return demand_s


def _scale_demand_qf(qf: pd.DataFrame, mult: float) -> pd.DataFrame:
    out = qf.copy()
    qcols = [c for c in out.columns if c.startswith("q_")]
    if qcols:
        out[qcols] = np.maximum(out[qcols].astype(float) * float(mult), 0.0)
    return out


def _amplify_common(d: np.ndarray, mult: float) -> np.ndarray:
    mu = np.mean(d, axis=0, keepdims=True)
    centered = d - mu
    common = np.mean(centered, axis=1, keepdims=True)
    return np.maximum(mu + common * float(mult) + (centered - common), 0.0)


def metrics_for_quantities(
    cells: Sequence[Cell],
    q: np.ndarray,
    demand: np.ndarray,
    alpha: float = 0.95,
    lock_days: int = 120,
    r_annual: float = 0.12,
) -> StressMetrics:
    """Compute expected uplift, CVaR, and cash spoilage on scenario matrix."""
    d = np.asarray(demand, dtype=float)
    s = d.shape[0]
    total = np.zeros(s, dtype=float)
    spoil = 0.0
    spend = 0.0
    for i, c in enumerate(cells):
        di = d[:, i]
        qi = float(max(q[i], 0.0))
        sold = np.minimum(di, qi)
        short = np.maximum(di - qi, 0.0)
        leftover = np.maximum(qi - di, 0.0)
        carry = float(c.carry(lock_days=lock_days, r_annual=r_annual))
        coeff_sold = c.p - c.c_b + c.k
        coeff_short = c.a * (c.p - c.c_d) - (1.0 - c.a) * c.g
        total += coeff_sold * sold + coeff_short * short - c.k * qi + c.r * c.k * leftover - carry * qi
        spoil += float(c.k) * (1.0 - float(c.r)) * float(np.mean(leftover))
        spend += float(c.k) * qi
    return StressMetrics(float(np.mean(total)), float(bruteforce_cvar(total, alpha=alpha)), float(spoil), float(spend))


def solve_risk_neutral(cells: Sequence[Cell], demand: np.ndarray, budget: float, reserve_fraction: float = 0.03) -> np.ndarray:
    """Risk-neutral water-filling allocation."""
    wf = waterfill(cells, demand, budget=float(budget), lock_days=120, r_annual=0.12, reserve_fraction=reserve_fraction)
    return np.array(wf.quantities, dtype=float)


def solve_risk_averse(
    cells: Sequence[Cell],
    demand: np.ndarray,
    budget: float,
    alpha: float = 0.95,
    beta: float = 0.5,
    reserve_fraction: float = 0.03,
    max_scenarios: int = 150,
) -> np.ndarray:
    """Risk-averse mean-CVaR allocation; solves on subset for tractability."""
    dec_d = np.asarray(demand, dtype=float)
    if dec_d.shape[0] > max_scenarios:
        idx = np.linspace(0, dec_d.shape[0] - 1, int(max_scenarios), dtype=int)
        dec_d = dec_d[idx]
    try:
        res = solve_mean_cvar_lp(
            cells,
            dec_d,
            budget=float(budget),
            alpha=alpha,
            beta=beta,
            lock_days=120,
            r_annual=0.12,
            reserve_fraction=reserve_fraction,
        )
        return np.array(res.quantities, dtype=float)
    except Exception:
        return solve_risk_neutral(cells, demand, budget, reserve_fraction=reserve_fraction)


def _apply_case(
    name: str,
    sim_cells: Sequence[CandidateCell],
    opt_base: Sequence[Cell],
    qf_base: pd.DataFrame,
    fore_base: np.ndarray,
    eval_base: np.ndarray,
    ground_truth_path: str,
    n_scenarios: int,
    seed: int,
    quantiles: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray, List[Cell], List[Cell]]:
    """Return (decision_demand, eval_demand, decision_cells, eval_cells) for a stress case."""
    dec_d = fore_base.copy()
    dec_qf = qf_base.copy()
    ev_d = eval_base.copy()
    dec_cells = [Cell(**c.__dict__) for c in opt_base]
    ev_cells = [Cell(**c.__dict__) for c in opt_base]

    if name == "demand_-30pct":
        dec_d = fore_base * 0.7
        dec_qf = _scale_demand_qf(qf_base, 0.7)
        dec_d = _scenarios_from_qf(dec_qf, sim_cells, n_scenarios, seed, quantiles, rho=0.6, label="stress:dec:-30")
        ev_d = latent_heldout_scenarios(ground_truth_path, sim_cells, n_scenarios, seed, demand_scale=0.7)
    elif name == "demand_+30pct":
        dec_d = fore_base * 1.3
        dec_qf = _scale_demand_qf(qf_base, 1.3)
        dec_d = _scenarios_from_qf(dec_qf, sim_cells, n_scenarios, seed, quantiles, rho=0.6, label="stress:dec:+30")
        ev_d = latent_heldout_scenarios(ground_truth_path, sim_cells, n_scenarios, seed, demand_scale=1.3)
    elif name == "volatility_x1.6":
        mu = np.mean(fore_base, axis=0, keepdims=True)
        dec_d = np.maximum(mu + (fore_base - mu) * 1.6, 0.0)
        ev_d = latent_heldout_scenarios(ground_truth_path, sim_cells, n_scenarios, seed, sigma_scale=1.6)
    elif name == "correlation_x2":
        dec_d = _amplify_common(fore_base, 2.0)
        dec_d = 0.5 * dec_d + 0.5 * _scenarios_from_qf(qf_base, sim_cells, n_scenarios, seed, quantiles, rho=0.95, label="stress:dec:corr2")
        ev_d = latent_heldout_scenarios(ground_truth_path, sim_cells, n_scenarios, seed, rho=0.95)
    elif name == "sellout_a_-0.20":
        dec_cells = [Cell(**{**c.__dict__, "a": float(np.clip(c.a - 0.20, 0.0, 1.0))}) for c in dec_cells]
        ev_cells = [Cell(**{**c.__dict__, "a": float(np.clip(c.a - 0.20, 0.0, 1.0))}) for c in ev_cells]
    elif name == "sellout_a_+0.10":
        dec_cells = [Cell(**{**c.__dict__, "a": float(np.clip(c.a + 0.10, 0.0, 1.0))}) for c in dec_cells]
        ev_cells = [Cell(**{**c.__dict__, "a": float(np.clip(c.a + 0.10, 0.0, 1.0))}) for c in ev_cells]
    elif name == "event_cancellation":
        qcols = [c for c in dec_qf.columns if c.startswith("q_")]
        for j, sc in enumerate(sim_cells):
            seg = str(sc.segment).lower()
            mult = 0.35 if seg == "event" else (0.85 if seg == "mixed" else 1.0)
            dec_d[:, j] *= mult
            if mult != 1.0 and qcols:
                dec_qf.loc[j, qcols] = dec_qf.loc[j, qcols].astype(float) * mult
        ev_d = latent_heldout_scenarios(ground_truth_path, sim_cells, n_scenarios, seed)
        for j, sc in enumerate(sim_cells):
            seg = str(sc.segment).lower()
            mult = 0.35 if seg == "event" else (0.85 if seg == "mixed" else 1.0)
            ev_d[:, j] *= mult
    elif name == "fx_+10pct":
        dec_cells = [Cell(**{**c.__dict__, "c_d": float(c.c_d * 1.10)}) for c in dec_cells]
        ev_cells = [Cell(**{**c.__dict__, "c_d": float(c.c_d * 1.10)}) for c in ev_cells]
    elif name == "supplier_default_largest":
        idx = int(np.argmax([c.k * c.cap for c in opt_base]))
        dec_cells[idx] = Cell(**{**dec_cells[idx].__dict__, "a": 0.0, "r": 0.0})
        ev_cells[idx] = Cell(**{**ev_cells[idx].__dict__, "a": 0.0, "r": 0.0})
    elif name == "competitor_price_war_-8pct":
        dec_cells = [Cell(**{**c.__dict__, "p": float(c.p * 0.92)}) for c in dec_cells]
        ev_cells = [Cell(**{**c.__dict__, "p": float(c.p * 0.92)}) for c in ev_cells]
    elif name == "bot_contamination_20pct_search":
        dec_qf = _scale_demand_qf(qf_base, 1.20)
        dec_d = _scenarios_from_qf(dec_qf, sim_cells, n_scenarios, seed, quantiles, rho=0.6, label="stress:dec:bot")
        ev_d = eval_base.copy()  # realized truth unchanged
        ev_cells = [Cell(**c.__dict__) for c in opt_base]
    else:
        raise ValueError(f"unknown stress case: {name}")

    dec_d = np.maximum(dec_d, 0.0)
    ev_d = np.maximum(ev_d, 0.0)
    return dec_d, ev_d, dec_cells, ev_cells


def run_stress_suite(
    budget: float = 10_000_000.0,
    n_scenarios: int = 300,
    seed: int = 42,
    alpha: float = 0.95,
    risk_beta: float = 0.5,
    reserve_fraction: float = 0.03,
    history_dir: str = "data/history",
    ground_truth_path: str = "data/ground_truth.parquet",
    out_md: str = "docs/stress_table.md",
    out_json: str = "docs/stress_results.json",
) -> Dict[str, Any]:
    """Run all stress cases and save table comparing risk-neutral vs risk-averse plans."""
    sim_cells = get_candidate_cells()
    opt_base = [sim_cell_to_optimizer(c) for c in sim_cells]
    qmodel, cfg, feats_pool, feat_cols, _, _ = _build_forecast_model(history_dir=history_dir, embargo_days=14)
    qf_base = _forecast_quantiles(qmodel, feats_pool, feat_cols, sim_cells)
    fore_base = _scenarios_from_qf(qf_base, sim_cells, n_scenarios, seed, cfg.quantiles, rho=0.6, label="stress:fore:base")
    eval_base = latent_heldout_scenarios(ground_truth_path, sim_cells, n_scenarios, seed)

    rows: List[Dict[str, Any]] = []
    for name in STRESS_CASES:
        dec_d, ev_d, dec_cells, ev_cells = _apply_case(
            name, sim_cells, opt_base, qf_base, fore_base, eval_base, ground_truth_path, n_scenarios, seed, cfg.quantiles
        )
        q_rn = solve_risk_neutral(dec_cells, dec_d, budget, reserve_fraction=reserve_fraction)
        q_ra = solve_risk_averse(dec_cells, dec_d, budget, alpha=alpha, beta=risk_beta, reserve_fraction=reserve_fraction)
        m_rn = metrics_for_quantities(ev_cells, q_rn, ev_d, alpha=alpha)
        m_ra = metrics_for_quantities(ev_cells, q_ra, ev_d, alpha=alpha)
        rows.append(
            {
                "scenario": name,
                "rn_expected_uplift": m_rn.expected_uplift,
                "rn_cvar95": m_rn.cvar95,
                "rn_cash_spoilage": m_rn.cash_spoilage,
                "rn_spend": m_rn.spend,
                "ra_expected_uplift": m_ra.expected_uplift,
                "ra_cvar95": m_ra.cvar95,
                "ra_cash_spoilage": m_ra.cash_spoilage,
                "ra_spend": m_ra.spend,
            }
        )

    assumption = (
        "Simulator outputs under stated assumptions, not TBO real P&L. "
        "Decisions on stressed forecast scenarios; realized on stressed held-out latent scenarios (eval-only); "
        "bot contamination affects decision signal only."
    )
    out = {
        "assumption": assumption,
        "budget": float(budget),
        "n_scenarios": int(n_scenarios),
        "seed": int(seed),
        "alpha": float(alpha),
        "risk_beta": float(risk_beta),
        "reserve_fraction": float(reserve_fraction),
        "cases": list(STRESS_CASES),
        "table": rows,
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    _write_md(rows, out_md, assumption, budget, n_scenarios, seed, alpha, risk_beta)
    return out


def _write_md(rows: List[Dict[str, Any]], path: str, assumption: str, budget: float, n_scenarios: int, seed: int, alpha: float, beta: float) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stress Test Results (risk-neutral vs risk-averse)",
        "",
        f"- **Note:** {assumption}",
        f"- **Budget:** Rs {budget:,.0f} | **Scenarios:** {n_scenarios} | **Seed:** {seed} | **RN:** water-filling | **RA:** mean-CVaR LP (alpha={alpha}, beta={beta})",
        "- **Cases:** demand -30%/+30%, volatility x1.6, correlation x2, sell-out a -0.20/+0.10, event cancellation, FX +10%, supplier default (largest commitment), competitor price war (p -8%), 20% bot contamination of search logs.",
        "",
        "| scenario | RN expected uplift (INR) | RN CVaR95 (INR) | RN spoilage (INR) | RA expected uplift (INR) | RA CVaR95 (INR) | RA spoilage (INR) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['scenario']} | {r['rn_expected_uplift']:.2f} | {r['rn_cvar95']:.2f} | {r['rn_cash_spoilage']:.2f} | "
            f"{r['ra_expected_uplift']:.2f} | {r['ra_cvar95']:.2f} | {r['ra_cash_spoilage']:.2f} |"
        )
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Stress suite: risk-neutral vs risk-averse plans")
    ap.add_argument("--budget", type=float, default=10_000_000.0)
    ap.add_argument("--scenarios", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=float, default=0.95)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--reserve-fraction", type=float, default=0.03)
    ap.add_argument("--history", type=str, default="data/history")
    ap.add_argument("--ground-truth", type=str, default="data/ground_truth.parquet")
    ap.add_argument("--out-md", type=str, default="docs/stress_table.md")
    ap.add_argument("--out-json", type=str, default="docs/stress_results.json")
    args = ap.parse_args()
    out = run_stress_suite(
        budget=args.budget,
        n_scenarios=args.scenarios,
        seed=args.seed,
        alpha=args.alpha,
        risk_beta=args.beta,
        reserve_fraction=args.reserve_fraction,
        history_dir=args.history,
        ground_truth_path=args.ground_truth,
        out_md=args.out_md,
        out_json=args.out_json,
    )
    print(f"wrote {args.out_md} {args.out_json} cases={len(out.get('table', []))}")


if __name__ == "__main__":
    main()
