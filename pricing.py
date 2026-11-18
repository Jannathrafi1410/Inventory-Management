"""Price optimization with demand elasticity and dynamic cost pass-through.

Computes per-cell price grid (0.88x-1.16x), re-solves q*(p) with demand shifted by
(p/p0)^epsilon and documented dynamic cost pass-through, chooses argmax, reports
flatness, and produces a markdown markdown price ladder for release-deadline +14/+7/+3.
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

from config.settings import load_assumptions
from eval.run_baselines import _build_forecast_model
from forecast.scenarios import scenarios_for_cells
from optimizer.cell import Cell
from optimizer.demo_phase2 import sim_cell_to_optimizer
from optimizer.profit import expected_profit
from sim.cells import CandidateCell, get_candidate_cells

# Assumption: segment elasticity epsilon uses config/assumptions market.elasticities where mapped (leisure=-2.5, compression/event/business proxy=-0.8) with segment-specific fallback (event->compression).
# Assumption: demand shifts as D(p)=D(p0)*(p/p0)^epsilon on scenario draws.
# Assumption: dynamic cost pass-through tau=0.30, so c_d(p)=c_d0 + tau*(p-p0) (supplier/dynamic channel partially tracks sell-price moves); c_b and k unchanged for committed structure comparison.
# Assumption: q*(p) is the unconstrained optimal quantity (critical fractile on shifted demand) capped by cell cap, evaluated at (p, c_d(p)).
# Assumption: flatness reported as max % profit loss vs optimum over grid and as % of grid prices within 1% of optimum.
# Assumption: markdown ladder at +14/+7/+3 days lowers price when marginal unit's expected margin falls below salvage value (refund value per unit for pre-bought inventory).


PRICE_GRID = np.round(np.linspace(0.88, 1.16, 15), 4)


@dataclass
class PricePoint:
    multiplier: float
    price: float
    c_d: float
    q_opt: float
    expected_profit: float


def _segment_epsilon(segment: str) -> float:
    try:
        cfg = load_assumptions()
        elas = cfg.market.elasticities
        leisure = float(elas["leisure"].value) if "leisure" in elas else -2.5
        compression = float(elas["compression"].value) if "compression" in elas else -0.8
    except Exception:
        leisure, compression = -2.5, -0.8
    s = str(segment).lower()
    if s == "leisure":
        return leisure
    if s in {"business", "event", "mixed"}:
        return compression
    return compression


def _demand_quantile(sorted_d: np.ndarray, fractile: float) -> float:
    if len(sorted_d) == 0:
        return 0.0
    f = float(np.clip(fractile, 0.0, 1.0))
    if f <= 0.0:
        return float(sorted_d[0])
    idx = int(np.ceil(f * len(sorted_d)) - 1)
    idx = max(0, min(idx, len(sorted_d) - 1))
    return float(sorted_d[idx])


def optimize_price_for_cell(
    cell: CandidateCell,
    demand_p0: np.ndarray,
    pass_through: float = 0.30,
    grid: Sequence[float] = PRICE_GRID,
    lock_days: int = 120,
    r_annual: float = 0.12,
) -> Dict[str, Any]:
    """Re-solve q*(p) over price grid for one cell and return argmax + flatness."""
    oc0 = sim_cell_to_optimizer(cell)
    eps = _segment_epsilon(cell.segment)
    d0 = np.asarray(demand_p0, dtype=float).reshape(-1)
    points: List[PricePoint] = []
    for m in grid:
        p = float(oc0.p * float(m))
        c_d = float(oc0.c_d + pass_through * (p - oc0.p))
        c_d = float(max(c_d, 0.0))
        c = Cell(oc0.id, p, c_d, oc0.c_b, oc0.k, oc0.r, oc0.a, oc0.g, oc0.cap, oc0.contract_type, oc0.group)
        dm = float(c.delta_m())
        co = float(c.c_o())
        carry = float(c.carry(lock_days=lock_days, r_annual=r_annual))
        if co == 0.0:
            f = 1.0 if dm - carry > 0.0 else 0.0
        elif dm <= 0.0 or (dm - carry) <= 0.0:
            f = 0.0
        else:
            f = float(np.clip((dm - carry) / (dm + co), 0.0, 1.0))
        shifted = np.maximum(d0 * (float(m) ** eps), 0.0)
        q_star = _demand_quantile(np.sort(shifted), f)
        q = float(min(float(c.cap), max(q_star, 0.0)))
        ep = float(expected_profit(c, q, shifted, lock_days=lock_days, r_annual=r_annual))
        points.append(PricePoint(float(m), p, c_d, q, ep))

    best = max(points, key=lambda x: x.expected_profit)
    profits = np.array([pt.expected_profit for pt in points], dtype=float)
    max_profit = float(best.expected_profit)
    losses_pct = np.where(np.abs(max_profit) > 1e-9, (max_profit - profits) / abs(max_profit) * 100.0, 0.0)
    within_1pct = float(np.mean(losses_pct <= 1.0) * 100.0)
    flatness = {
        "max_profit_inr": max_profit,
        "worst_grid_profit_inr": float(np.min(profits)),
        "max_profit_loss_pct_vs_optimum": float(np.max(losses_pct)),
        "share_grid_within_1pct_of_optimum_pct": within_1pct,
        "profit_range_inr": float(np.max(profits) - np.min(profits)),
    }
    return {
        "cell_id": cell.cell_id,
        "segment": cell.segment,
        "epsilon": eps,
        "p0": float(oc0.p),
        "argmax_multiplier": best.multiplier,
        "argmax_price": best.price,
        "argmax_q": best.q_opt,
        "argmax_expected_profit": best.expected_profit,
        "flatness": flatness,
        "grid": [pt.__dict__ for pt in points],
    }


def markdown_ladder(
    cell: CandidateCell,
    demand_p0: np.ndarray,
    q_committed: float,
    pass_through: float = 0.30,
    horizons: Sequence[int] = (14, 7, 3),
) -> Dict[str, Any]:
    """Price ladder at +14/+7/+3 days before release deadline.

    Lowers price when marginal unit's expected margin falls below salvage (refund value per unit).
    """
    oc0 = sim_cell_to_optimizer(cell)
    eps = _segment_epsilon(cell.segment)
    d0 = np.asarray(demand_p0, dtype=float).reshape(-1)
    salvage_per_unit = float(oc0.r * oc0.k)  # refund/salvage value per unsold committed unit
    rows = []
    base_mult = 1.0
    for h in horizons:
        # Assumption: as deadline approaches, effective conversion/arrival improves slightly; model as demand uplift factor (1 + 0.02*(14-h)/11).
        urgency = 1.0 + 0.02 * ((14 - int(h)) / 11.0)
        chosen = None
        for m in sorted(PRICE_GRID):
            p = float(oc0.p * float(m))
            c_d = float(max(oc0.c_d + pass_through * (p - oc0.p), 0.0))
            shifted = np.maximum(d0 * (float(m) ** eps) * urgency, 0.0)
            # expected margin on marginal committed unit: if it sells, gain (p-c_b) vs salvage; approximate sell prob of marginal unit
            if q_committed <= 1e-9:
                sell_prob_marginal = float(np.mean(shifted >= 1.0))
            else:
                sell_prob_marginal = float(np.mean(shifted >= q_committed))
            marginal_expected_margin = sell_prob_marginal * (p - oc0.c_b) + (1.0 - sell_prob_marginal) * salvage_per_unit - salvage_per_unit
            # equivalently compare (p-c_b) to salvage when sold; condition: lower price if marginal expected margin < salvage break-even => (p-c_b)*P(sell_marginal) < salvage*(P(sell_marginal)? no)
            # simpler: expected incremental margin of keeping price vs marking down; markdown if P(sell_marginal)* (p-c_b) < salvage_per_unit
            markdown_needed = sell_prob_marginal * (p - oc0.c_b) < salvage_per_unit
            candidate = {
                "days_to_release": int(h),
                "multiplier": float(m),
                "price": p,
                "sell_prob_marginal": sell_prob_marginal,
                "marginal_expected_margin_vs_salvage_inr": float(sell_prob_marginal * (p - oc0.c_b) - salvage_per_unit),
                "markdown_needed": bool(markdown_needed),
            }
            if not markdown_needed:
                chosen = candidate
                break
        if chosen is None:
            # take lowest price if all need markdown
            m = float(min(PRICE_GRID))
            p = float(oc0.p * m)
            shifted = np.maximum(d0 * (m ** eps) * urgency, 0.0)
            sell_prob_marginal = float(np.mean(shifted >= max(q_committed, 1.0)))
            chosen = {
                "days_to_release": int(h),
                "multiplier": m,
                "price": p,
                "sell_prob_marginal": sell_prob_marginal,
                "marginal_expected_margin_vs_salvage_inr": float(sell_prob_marginal * (p - oc0.c_b) - salvage_per_unit),
                "markdown_needed": True,
            }
        rows.append(chosen)
        base_mult = chosen["multiplier"]
    return {"cell_id": cell.cell_id, "q_committed": float(q_committed), "salvage_per_unit_inr": salvage_per_unit, "ladder": rows}


def run_pricing(
    budget: float = 10_000_000.0,
    n_scenarios: int = 300,
    seed: int = 42,
    history_dir: str = "data/history",
    out_md: str = "docs/pricing_ladder.md",
    out_json: str = "docs/pricing_results.json",
) -> Dict[str, Any]:
    """Run pricing optimization and markdown ladder for selected candidate cells."""
    sim_cells = get_candidate_cells()
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
    demand_s, _ = scenarios_for_cells(qf, cell_ids=cell_ids, cell_groups=[c.region_group for c in sim_cells], s=n_scenarios, quantiles=cfg.quantiles, intra_group_rho=0.6, seed=seed, label="pricing:forecast")

    # pick representative subset including anchors/large commitments
    focus_ids = {"DXB-NYE", "DXB-DEC", "AIR-BOMDXB", "SIN-F1", "MLE-DEC"}
    focus = [c for c in sim_cells if c.cell_id in focus_ids]
    if not focus:
        focus = sim_cells[:5]
    idx_map = {cid: i for i, cid in enumerate(cell_ids)}

    opt_results = []
    ladders = []
    for c in focus:
        j = idx_map[c.cell_id]
        d0 = demand_s[:, j]
        opt = optimize_price_for_cell(c, d0)
        opt_results.append(opt)
        q_commit = float(max(opt["argmax_q"], d0.mean() * 0.5))
        ladders.append(markdown_ladder(c, d0, q_commit))

    assumption = "Simulator outputs under stated assumptions, not TBO real P&L. Demand shifted by (p/p0)^epsilon with segment elasticities; dynamic cost pass-through tau=0.30 on c_d; q*(p) unconstrained fractile-capped re-solve per grid point."
    out = {"assumption": assumption, "budget_ref_inr": float(budget), "n_scenarios": int(n_scenarios), "seed": int(seed), "price_grid": [float(x) for x in PRICE_GRID], "optimizations": opt_results, "markdown_ladders": ladders}
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    _write_md(out_md, out)
    return out


def _write_md(path: str, out: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Pricing Optimization and Markdown Ladder",
        "",
        f"- **Note:** {out['assumption']}",
        f"- **Price grid:** 0.88x to 1.16x of p0 ({len(out['price_grid'])} points) | **Scenarios:** {out['n_scenarios']} | **Seed:** {out['seed']}",
        "",
        "## Optimal price per cell (argmax expected profit) and flatness",
        "",
        "| cell | segment | epsilon | p0 (INR) | argmax price (INR) | argmax q (units) | argmax E[profit] (INR) | max loss vs opt (%) | grid within 1% of opt (%) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in out["optimizations"]:
        f = r["flatness"]
        lines.append(
            f"| {r['cell_id']} | {r['segment']} | {r['epsilon']:.2f} | {r['p0']:.0f} | {r['argmax_price']:.0f} | {r['argmax_q']:.2f} | {r['argmax_expected_profit']:.2f} | {f['max_profit_loss_pct_vs_optimum']:.2f}% | {f['share_grid_within_1pct_of_optimum_pct']:.1f}% |"
        )
    lines += ["", "## Markdown ladder (release-deadline +14/+7/+3 days)", "", "Lowers price when marginal unit's expected margin falls below salvage (refund value per unit).", ""]
    for lad in out["markdown_ladders"]:
        lines.append(f"### {lad['cell_id']} (q_committed={lad['q_committed']:.1f} units, salvage={lad['salvage_per_unit_inr']:.0f} INR/unit)")
        lines.append("")
        lines.append("| days to release | price (INR) | multiplier | P(sell marginal) | margin vs salvage (INR) | markdown needed |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | :---: |")
        for row in lad["ladder"]:
            lines.append(f"| +{row['days_to_release']} | {row['price']:.0f} | {row['multiplier']:.2f} | {row['sell_prob_marginal']:.3f} | {row['marginal_expected_margin_vs_salvage_inr']:.2f} | {row['markdown_needed']} |")
        lines.append("")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Price optimization and markdown ladder")
    ap.add_argument("--budget", type=float, default=10_000_000.0)
    ap.add_argument("--scenarios", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--history", type=str, default="data/history")
    ap.add_argument("--out-md", type=str, default="docs/pricing_ladder.md")
    ap.add_argument("--out-json", type=str, default="docs/pricing_results.json")
    args = ap.parse_args()
    run_pricing(budget=args.budget, n_scenarios=args.scenarios, seed=args.seed, history_dir=args.history, out_md=args.out_md, out_json=args.out_json)
    print(f"wrote {args.out_md} {args.out_json}")


if __name__ == "__main__":
    main()
