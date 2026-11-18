"""Timing decisions: tranching, priced WAIT, and release-date logic.

Implements (a) tranching policy resized by current fractile at decision points,
(b) priced WAIT value (buy after next signal vs buy now, net of expected rate drift),
(c) release-date hold/release rule. All numbers are simulator outputs under stated
assumptions, not TBO real P&L.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from eval.run_baselines import _build_forecast_model
from forecast.scenarios import scenarios_for_cells
from optimizer.cell import Cell
from optimizer.demo_phase2 import sim_cell_to_optimizer
from optimizer.fractile import optimal_fractile
from optimizer.profit import expected_profit, per_scenario_profit
from sim.cells import CandidateCell, get_candidate_cells

# Assumption: tranching policy buys 40% at -120d, 35% at -75d, 25% at -45d of the target quantity, with each tranche re-sized by the current critical fractile at that decision point (proxy for updated forecast).
# Assumption: expected rate drift is +0.5% per 30 days on buy cost c_b/k for deferral (prices tend to rise near peak), applied as cost multiplier for later purchase.
# Assumption: next-signal demand uncertainty reduction modeled as 10% lower sigma (tighter distribution) when buying after next signal, improving decision quality.
# Assumption: release-date logic holds inventory if P(demand arrives before expiry)*margin_per_unit > refund_value_per_unit (salvage r*k).
# Assumption: P(demand arrives) approximated from scenario demand distribution as P(D_remaining >= 1 unit) given current on-hand/committed quantity context.


@dataclass
class TrancheDecision:
    days_before: int
    planned_share: float
    current_fractile: float
    resized_units: float
    cumulative_units: float
    cash_this_tranche: float


def _demand_quantile(sorted_d: np.ndarray, fractile: float) -> float:
    if len(sorted_d) == 0:
        return 0.0
    f = float(np.clip(fractile, 0.0, 1.0))
    if f <= 0.0:
        return float(sorted_d[0])
    idx = int(np.ceil(f * len(sorted_d)) - 1)
    idx = max(0, min(idx, len(sorted_d) - 1))
    return float(sorted_d[idx])


def tranching_policy(
    cell: CandidateCell,
    demand_scenarios: np.ndarray,
    schedule: Sequence[Tuple[int, float]] = ((120, 0.40), (75, 0.35), (45, 0.25)),
    lock_days: int = 120,
    r_annual: float = 0.12,
) -> Dict[str, Any]:
    """Compute tranche sizes re-sized by current fractile at each decision point.

    Target base quantity is unconstrained q* at initial fractile on base scenarios, capped.
    Each tranche buys planned_share * (F_t / F_initial) * base_target, cumulatively capped.
    """
    oc = sim_cell_to_optimizer(cell)
    d = np.asarray(demand_scenarios, dtype=float).reshape(-1)
    sorted_d = np.sort(d)
    f0 = float(optimal_fractile(oc, lock_days=lock_days, r_annual=r_annual))
    base_target = float(min(float(oc.cap), _demand_quantile(sorted_d, f0)))
    tranches: List[TrancheDecision] = []
    cum = 0.0
    cum_cash = 0.0
    for days_before, share in schedule:
        # Assumption: forecast sharpens over time (sigma reduction 2% per 30 days closer) affecting fractile estimate.
        tighten = 1.0 - 0.02 * max((120 - int(days_before)) / 30.0, 0.0)
        mu = float(np.mean(d))
        d_t = np.maximum(mu + (d - mu) * tighten, 0.0)
        sd_t = np.sort(d_t)
        f_t = float(optimal_fractile(oc, lock_days=lock_days, r_annual=r_annual))
        # use fractile-implied q at decision time
        q_t_target = float(min(float(oc.cap), _demand_quantile(sd_t, f_t)))
        resize = (f_t / f0) if f0 > 1e-9 else 1.0
        tranche_units = float(np.clip(share * resize * base_target, 0.0, max(float(oc.cap) - cum, 0.0)))
        # also respect current target at time t proportionally
        tranche_units = float(min(tranche_units, max(q_t_target - cum, 0.0), max(float(oc.cap) - cum, 0.0)))
        if tranche_units < 0:
            tranche_units = 0.0
        cash = tranche_units * float(oc.k)
        cum += tranche_units
        cum_cash += cash
        tranches.append(TrancheDecision(int(days_before), float(share), f_t, tranche_units, cum, cash))
    return {
        "cell_id": cell.cell_id,
        "base_fractile": f0,
        "base_target_units": base_target,
        "tranches": [t.__dict__ for t in tranches],
        "total_units": cum,
        "total_cash_inr": cum_cash,
    }


def priced_wait(
    cell: CandidateCell,
    demand_scenarios: np.ndarray,
    drift_per_30d: float = 0.005,
    signal_days: int = 30,
    sigma_reduction: float = 0.10,
    lock_days: int = 120,
    r_annual: float = 0.12,
) -> Dict[str, Any]:
    """Compute value of waiting for next signal vs buying now.

    value_wait = E[profit | buy after next signal] - E[profit | buy now] - drift_cost_adjustment.
    Positive => WAIT preferred under stated assumptions.
    """
    oc = sim_cell_to_optimizer(cell)
    d = np.asarray(demand_scenarios, dtype=float).reshape(-1)
    sorted_d = np.sort(d)
    f_now = float(optimal_fractile(oc, lock_days=lock_days, r_annual=r_annual))
    q_now = float(min(float(oc.cap), _demand_quantile(sorted_d, f_now)))
    e_now = float(expected_profit(oc, q_now, d, lock_days=lock_days, r_annual=r_annual))

    # buy after next signal: tighter demand info (lower sigma) but possible rate drift
    mu = float(np.mean(d))
    d_signal = np.maximum(mu + (d - mu) * (1.0 - sigma_reduction), 0.0)
    sd_signal = np.sort(d_signal)
    f_sig = float(optimal_fractile(oc, lock_days=lock_days, r_annual=r_annual))
    q_sig = float(min(float(oc.cap), _demand_quantile(sd_signal, f_sig)))
    drift_mult = 1.0 + drift_per_30d * (signal_days / 30.0)
    oc_later = Cell(oc.id, oc.p, oc.c_d, oc.c_b * drift_mult, oc.k * drift_mult, oc.r, oc.a, oc.g, oc.cap, oc.contract_type, oc.group)
    e_after = float(expected_profit(oc_later, q_sig, d_signal, lock_days=max(lock_days - signal_days, 1), r_annual=r_annual))
    value_wait = e_after - e_now
    action = "WAIT" if value_wait > 0 else "COMMIT"
    rationale = (
        f"E[profit|after next signal]={e_after:.2f} INR vs E[profit|buy now]={e_now:.2f} INR; "
        f"expected rate drift x{drift_mult:.4f} over {signal_days}d; demand sigma reduction {sigma_reduction*100:.0f}%."
    )
    return {
        "cell_id": cell.cell_id,
        "q_now_units": q_now,
        "q_after_signal_units": q_sig,
        "e_profit_now_inr": e_now,
        "e_profit_after_signal_inr": e_after,
        "value_wait_inr": float(value_wait),
        "expected_rate_drift_multiplier": drift_mult,
        "recommended_action": action,
        "rationale": rationale,
    }


def release_decision(
    cell: CandidateCell,
    demand_scenarios: np.ndarray,
    q_on_hand: float,
    days_to_expiry: int = 7,
) -> Dict[str, Any]:
    """Hold/release logic: hold if P(demand arrives)*margin > refund value.

    Assumption: P(demand arrives) = P(D >= q_on_hand marginal need proxy) approximated as P(D >= max(q_on_hand,1)) over scenarios;
    margin per unit = (p - c_b) on sale; refund value per unit = r*k (salvage/refund if released/unsold as applicable).
    """
    oc = sim_cell_to_optimizer(cell)
    d = np.asarray(demand_scenarios, dtype=float).reshape(-1)
    qh = float(max(q_on_hand, 0.0))
    if qh <= 1e-9:
        return {"cell_id": cell.cell_id, "q_on_hand": qh, "action": "RELEASE", "reason": "No inventory on hand."}
    # near expiry demand urgency: scale by small factor (assumption: remaining window shorter)
    urgency = float(np.clip(1.0 - 0.02 * max(days_to_expiry - 7, 0) / 7.0, 0.85, 1.0))
    d_eff = d * urgency
    p_arrive = float(np.mean(d_eff >= qh))
    margin_per_unit = float(oc.p - oc.c_b)
    refund_value_per_unit = float(oc.r * oc.k)
    hold_value = p_arrive * margin_per_unit
    action = "HOLD" if hold_value > refund_value_per_unit else "RELEASE"
    return {
        "cell_id": cell.cell_id,
        "q_on_hand": qh,
        "days_to_expiry": int(days_to_expiry),
        "p_demand_arrives": p_arrive,
        "margin_per_unit_inr": margin_per_unit,
        "refund_value_per_unit_inr": refund_value_per_unit,
        "hold_expected_value_inr_per_unit": hold_value,
        "recommended_action": action,
        "rule": "hold if P(demand arrives)*margin > refund value",
    }


def run_timing(
    n_scenarios: int = 300,
    seed: int = 42,
    history_dir: str = "data/history",
    out_md: str = "docs/timing_policy.md",
    out_json: str = "docs/timing_policy.json",
) -> Dict[str, Any]:
    """Run timing demo for representative cells and save docs."""
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
    demand_s, _ = scenarios_for_cells(qf, cell_ids=cell_ids, cell_groups=[c.region_group for c in sim_cells], s=n_scenarios, quantiles=cfg.quantiles, intra_group_rho=0.6, seed=seed, label="timing:forecast")
    idx_map = {cid: i for i, cid in enumerate(cell_ids)}
    focus_ids = ["DXB-NYE", "DXB-DEC", "AIR-BOMDXB"]
    focus = [c for c in sim_cells if c.cell_id in focus_ids] or sim_cells[:3]

    results = []
    for c in focus:
        d = demand_s[:, idx_map[c.cell_id]]
        tr = tranching_policy(c, d)
        pw = priced_wait(c, d)
        rd = release_decision(c, d, q_on_hand=max(tr["total_units"] * 0.4, 5.0), days_to_expiry=7)
        results.append({"tranching": tr, "priced_wait": pw, "release": rd})

    assumption = "Simulator outputs under stated assumptions, not TBO real P&L. Tranches 40/35/25% at -120/-75/-45d resized by current fractile; WAIT compares buy-now vs after-signal (sigma -10%, rate drift +0.5%/30d); release holds if P(demand arrives)*margin > refund value."
    out = {"assumption": assumption, "n_scenarios": int(n_scenarios), "seed": int(seed), "cells": results}
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    _write_md(out_md, out)
    return out


def _write_md(path: str, out: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Timing Policy (Tranching, Priced WAIT, Release)", "", f"- **Note:** {out['assumption']}", ""]
    for item in out["cells"]:
        tr = item["tranching"]
        pw = item["priced_wait"]
        rd = item["release"]
        lines.append(f"## {tr['cell_id']}")
        lines.append("")
        lines.append("### (a) Tranching policy")
        lines.append(f"- Base fractile: {tr['base_fractile']:.4f} | Base target: {tr['base_target_units']:.2f} units | Total: {tr['total_units']:.2f} units ({tr['total_cash_inr']:.0f} INR)")
        lines.append("")
        lines.append("| days before | planned share | current fractile | resized units | cumulative units | cash (INR) |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
        for t in tr["tranches"]:
            lines.append(f"| -{t['days_before']} | {t['planned_share']:.0%} | {t['current_fractile']:.4f} | {t['resized_units']:.2f} | {t['cumulative_units']:.2f} | {t['cash_this_tranche']:.0f} |")
        lines.append("")
        lines.append("### (b) Priced WAIT action")
        lines.append(f"- Recommendation: **{pw['recommended_action']}** | value_wait = {pw['value_wait_inr']:.2f} INR")
        lines.append(f"- Rationale: {pw['rationale']}")
        lines.append("")
        lines.append("### (c) Release-date logic")
        lines.append(f"- Recommendation: **{rd['recommended_action']}** | P(demand arrives)={rd['p_demand_arrives']:.3f}, margin={rd['margin_per_unit_inr']:.0f} INR/unit, refund={rd['refund_value_per_unit_inr']:.0f} INR/unit, hold value={rd['hold_expected_value_inr_per_unit']:.2f} INR/unit")
        lines.append("")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Timing policy: tranching, priced WAIT, release logic")
    ap.add_argument("--scenarios", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--history", type=str, default="data/history")
    ap.add_argument("--out-md", type=str, default="docs/timing_policy.md")
    ap.add_argument("--out-json", type=str, default="docs/timing_policy.json")
    args = ap.parse_args()
    run_timing(n_scenarios=args.scenarios, seed=args.seed, history_dir=args.history, out_md=args.out_md, out_json=args.out_json)
    print(f"wrote {args.out_md} {args.out_json}")


if __name__ == "__main__":
    main()
