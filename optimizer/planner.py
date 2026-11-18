"""Forecast-to-optimizer planner: candidate funnel -> scenarios -> water-filling -> recommendations.

Produces a complete plan with per-cell recommendations and plan-level capital metrics.
All numbers are simulator outputs under stated assumptions, not TBO real P&L.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config.settings import load_assumptions
from forecast.features import FEATURE_COLUMNS, build_point_in_time_features
from forecast.quantile_model import QuantileDemandModel, QuantileModelConfig
from forecast.scenarios import scenarios_for_cells
from forecast.unconstrain import em_unconstrain_normal
from optimizer.baselines import evaluate_strategy_profit
from optimizer.cell import Cell
from optimizer.demo_phase2 import sim_cell_to_optimizer
from optimizer.fractile import optimal_fractile
from optimizer.waterfill import WaterfillResult, waterfill
from sim.cells import CandidateCell, get_candidate_cells


@dataclass
class Recommendation:
    """Per-cell planning recommendation."""

    cell_id: str
    action: str  # COMMIT | NEGOTIATE_ALLOTMENT | WAIT | RELEASE | DONT_BUY
    structure: str
    units: float
    cash: float
    unit_net: float
    sell_price: float
    fractile: float
    delta_m: float
    c_o: float
    demand_p10: float
    demand_p50: float
    demand_p90: float
    sellout_prob: float
    why: List[str]


@dataclass
class PlanResult:
    """Complete plan output."""

    recommendations: List[Recommendation]
    lambda_star: float
    cash_deployed: float
    reserve: float
    structural_uplift: float
    capital_uplift: float
    roi_on_cash: float
    expected_cash_spoilage: float
    fill_rate_uplift: float
    budget: float
    spend: float
    candidate_ids: List[str]
    scenario_shape: Tuple[int, int]
    waterfill: WaterfillResult


def _default_risk() -> Dict[str, float]:
    try:
        cfg = load_assumptions()
        return {"alpha": float(cfg.risk.alpha.value), "beta": float(cfg.risk.beta.value)}
    except Exception:
        return {"alpha": 0.95, "beta": 0.5}


def _prepare_quantile_context(history_dir: str = "data/history", decision_ts: str | None = None) -> tuple:
    """Fit quantile demand model on censored history (point-in-time) and return context."""
    feats = build_point_in_time_features(decision_ts=decision_ts, history_dir=history_dir).copy()
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
    cfg = QuantileModelConfig(
        quantiles=(0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95),
        n_estimators=120,
        num_leaves=31,
        learning_rate=0.05,
        random_state=42,
    )
    qmodel = QuantileDemandModel(cfg)
    qmodel.fit(feats[feat_cols], feats["y_target_uncon"].to_numpy(), feature_columns=feat_cols)
    return qmodel, cfg, feats, feat_cols


def _latest_quantiles_for_cells(
    qmodel: QuantileDemandModel, feats: pd.DataFrame, feat_cols: List[str], cell_ids: List[str]
) -> pd.DataFrame:
    """Predict per-cell quantiles at latest available decision point (censored history only)."""
    latest = feats.sort_values("decision_ts").groupby("cell_id").tail(1)
    latest = latest.set_index("cell_id").reindex(cell_ids).reset_index().rename(columns={"index": "cell_id"})
    if "decision_ts" in latest.columns and latest["decision_ts"].isna().any():
        med = feats[feat_cols].median(numeric_only=True)
        for col in feat_cols:
            if col not in latest.columns:
                latest[col] = med.get(col, 0.0)
            latest[col] = pd.to_numeric(latest[col], errors="coerce")
        latest[feat_cols] = latest[feat_cols].fillna(med).fillna(0.0)
    for col in feat_cols:
        if col not in latest.columns:
            latest[col] = 0.0
        latest[col] = pd.to_numeric(latest[col], errors="coerce").fillna(0.0)
    qf = qmodel.predict_quantiles(latest[feat_cols])
    qf["cell_id"] = cell_ids
    return qf


def candidate_funnel(
    sim_cells: Sequence[CandidateCell],
    feats: pd.DataFrame,
    budget: float,
    constraints: Optional[Dict[str, Any]] = None,
    top_k: Optional[int] = None,
) -> List[CandidateCell]:
    """Candidate funnel: screen and rank cells, keep top K.

    Screens by delta_m > 0 and volume threshold and (sell-out rate or discount or event flag).
    Ranks by delta_m/k with k=0 structures first. Assumption: volume threshold = median observed
    sales >= 5 (censored) or base_demand_mean >= 10; sell-out/discount/event flags from history/params.
    """
    constraints = constraints or {}
    lock_days = int(constraints.get("lock_days", 120))
    r_annual = float(constraints.get("r_annual", 0.12))
    vol_threshold = float(constraints.get("volume_threshold", 5.0))
    vol = feats.groupby("cell_id")["observed_sales"].median() if (not feats.empty and "observed_sales" in feats) else pd.Series(dtype=float)
    sellout = feats.groupby("cell_id")["sell_out_obs"].mean() if (not feats.empty and "sell_out_obs" in feats) else pd.Series(dtype=float)
    screened: List[Tuple[Tuple[int, float, float], CandidateCell]] = []
    for c in sim_cells:
        oc = sim_cell_to_optimizer(c)
        dm = float(oc.delta_m())
        if dm <= 0.0:
            continue
        v = float(vol.get(c.cell_id, np.nan))
        volume_ok = (np.isfinite(v) and v >= vol_threshold) or (float(c.base_demand_mean) >= 10.0)
        sellout_rate = float(sellout.get(c.cell_id, np.nan))
        sellout_flag = (np.isfinite(sellout_rate) and sellout_rate >= 0.05) or (c.a < 0.95)
        discount = (c.c_d - c.c_b) / max(c.c_d, 1e-9)
        discount_flag = discount >= 0.05
        event_flag = c.segment in {"event", "mixed"}
        if not (volume_ok and (sellout_flag or discount_flag or event_flag)):
            continue
        ratio = float("inf") if oc.k <= 0.0 else float(dm / oc.k)
        priority = 1 if oc.k <= 0.0 else 0
        screened.append(((priority, ratio, dm), c))
    screened.sort(key=lambda x: (x[0][0], x[0][1], x[0][2]), reverse=True)
    ordered = [c for _, c in screened]
    if top_k is None:
        top_k = len(ordered) if ordered else 1
    top_k = int(max(1, min(int(top_k), len(ordered) if ordered else 1)))
    return ordered[:top_k] if ordered else []


def _demand_stats(demand_col: np.ndarray) -> Tuple[float, float, float]:
    d = np.asarray(demand_col, dtype=float).reshape(-1)
    if len(d) == 0:
        return 0.0, 0.0, 0.0
    p10, p50, p90 = np.percentile(d, [10, 50, 90])
    return float(p10), float(p50), float(p90)


def _sellout_prob(q: float, demand_col: np.ndarray) -> float:
    d = np.asarray(demand_col, dtype=float).reshape(-1)
    if len(d) == 0:
        return 0.0
    return float(np.mean(d > float(q)))


def _why_bullets(
    c: Cell,
    q: float,
    demand_col: np.ndarray,
    action: str,
    fractile: float,
    lock_days: int,
    lambda_star: float,
) -> List[str]:
    p10, p50, p90 = _demand_stats(demand_col)
    carry = c.carry(lock_days=lock_days, r_annual=0.12)
    dm = c.delta_m()
    co = c.c_o()
    whys: List[str] = []
    if action == "DONT_BUY":
        whys.append(f"DONT_BUY: delta_m={dm:.0f} is not attractive vs dynamic fallback/carry.")
        return whys
    if action == "WAIT":
        whys.append(f"WAIT: defer commitment (demand p10/p50/p90={p10:.0f}/{p50:.0f}/{p90:.0f}) to preserve cash flexibility.")
        if dm > 0:
            whys.append(f"Economics positive (delta_m={dm:.0f}) but not committing now under current constraints.")
        return whys
    if action == "NEGOTIATE_ALLOTMENT":
        whys.append(f"NEGOTIATE_ALLOTMENT: positive edge (delta_m={dm:.0f}) but cash-funded buy not taken now.")
        whys.append(f"Seek softer terms (lower k/higher r) given c_o={co:.0f} and demand p50={p50:.0f}.")
        return whys
    if action == "RELEASE":
        whys.append("RELEASE: do not hold/commit these units now given current economics and demand risk.")
        return whys
    if c.k == 0:
        whys.append("COMMIT: zero cash at risk (k=0, free option) so taking available cap is optimal.")
    else:
        whys.append(
            f"Sold at the {fractile*100:.0f}th demand percentile, not the mean, because cash is at risk (c_o={co:.0f}, r={c.r:.2f})."
        )
        whys.append(f"Per-unit edge delta_m={dm:.0f} net of fallback, with carry={carry:.0f} over {lock_days}d lock.")
    if lambda_star > 0 and c.k > 0:
        whys.append(f"Budget-binding allocation (lambda={lambda_star:.4f}); selected on delta_m/k={dm/max(c.k,1e-9):.4f}.")
    sp = _sellout_prob(q, demand_col)
    whys.append(f"Demand p10/p50/p90={p10:.0f}/{p50:.0f}/{p90:.0f}; sellout probability at q={q:.1f} is {sp:.1%}.")
    return whys[:5]


def plan(
    budget: float,
    constraints: Optional[Dict[str, Any]] = None,
    risk: Optional[Dict[str, Any]] = None,
    history_dir: str = "data/history",
    decision_ts: str | None = None,
    top_k: Optional[int] = None,
    n_scenarios: int = 500,
    scenario_seed: int = 42,
) -> PlanResult:
    """Run full planning pipeline and return recommendations + plan-level metrics.

    Pipeline: candidate funnel -> quantile scenario matrix -> water-filling -> recommendations.
    Assumption: uses censored history for features/forecasts; reserve defaults to 0; carry 12%/yr 120d.
    """
    constraints = constraints or {}
    _ = risk or _default_risk()
    lock_days = int(constraints.get("lock_days", 120))
    r_annual = float(constraints.get("r_annual", 0.12))
    reserve_fraction = float(constraints.get("reserve_fraction", 0.0))
    group_caps = constraints.get("group_caps")
    reserve = float(budget) * max(0.0, min(reserve_fraction, 0.95))

    sim_cells_all = get_candidate_cells()
    qmodel, cfg, feats, feat_cols = _prepare_quantile_context(history_dir=history_dir, decision_ts=decision_ts)
    selected = candidate_funnel(sim_cells_all, feats, budget, constraints=constraints, top_k=top_k or 18)
    if not selected:
        selected = sim_cells_all[:]
    selected_ids = [c.cell_id for c in selected]
    opt_cells = [sim_cell_to_optimizer(c) for c in selected]
    qf_cells = _latest_quantiles_for_cells(qmodel, feats, feat_cols, selected_ids)
    demand_s, ordered_ids = scenarios_for_cells(
        qf_cells,
        cell_ids=selected_ids,
        cell_groups=[c.region_group for c in selected],
        s=int(n_scenarios),
        quantiles=cfg.quantiles,
        intra_group_rho=0.6,
        seed=int(scenario_seed),
        label="planner:scenarios",
    )
    # Assumption: default reserve_fraction=0.03 preserves capital discipline and leaves budget unspent when marginal cells unattractive.
    eff_reserve_fraction = reserve_fraction if reserve_fraction > 0.0 else 0.03
    wf_res = waterfill(
        opt_cells,
        demand_s,
        budget=float(budget),
        lock_days=lock_days,
        r_annual=r_annual,
        reserve_fraction=eff_reserve_fraction,
        group_caps=group_caps,
    )
    reserve = float(budget) * max(0.0, min(eff_reserve_fraction, 0.95))

    eval_metrics = evaluate_strategy_profit(opt_cells, wf_res.quantities, demand_s, lock_days=lock_days, r_annual=r_annual)
    zero_q = np.array(wf_res.quantities, dtype=float, copy=True)
    cash_q = np.array(wf_res.quantities, dtype=float, copy=True)
    for i, c in enumerate(opt_cells):
        if c.k > 0:
            zero_q[i] = 0.0
        else:
            cash_q[i] = 0.0
    structural = evaluate_strategy_profit(opt_cells, zero_q, demand_s, lock_days=lock_days, r_annual=r_annual)[
        "incremental_profit_mean"
    ]
    capital = evaluate_strategy_profit(opt_cells, cash_q, demand_s, lock_days=lock_days, r_annual=r_annual)[
        "incremental_profit_mean"
    ]
    cash_deployed = float(wf_res.spend)
    roi_on_cash = (float(capital) / cash_deployed) if cash_deployed > 1e-9 else 0.0

    qty_by_id = {cid: float(wf_res.quantities[i]) for i, cid in enumerate(ordered_ids)}
    demand_by_id = {cid: demand_s[:, i] for i, cid in enumerate(ordered_ids)}
    diag_by_id = {d.cell_id: d for d in wf_res.diagnostics}
    selected_set = set(selected_ids)
    recs: List[Recommendation] = []
    for c in sim_cells_all:
        oc = sim_cell_to_optimizer(c)
        dm = float(oc.delta_m())
        co = float(oc.c_o())
        in_plan = c.cell_id in selected_set and c.cell_id in qty_by_id
        q = float(qty_by_id.get(c.cell_id, 0.0)) if in_plan else 0.0
        dcol = demand_by_id.get(c.cell_id, np.zeros(max(demand_s.shape[0], 1))) if in_plan else np.array([float(c.base_demand_mean)])
        p10, p50, p90 = _demand_stats(dcol) if in_plan else (0.0, float(c.base_demand_mean), 0.0)
        sp = _sellout_prob(q, dcol) if in_plan else float(np.clip(1.0 - c.a, 0.0, 1.0))
        fractile = float(diag_by_id[c.cell_id].fractile_used) if c.cell_id in diag_by_id else float(
            optimal_fractile(oc, lock_days=lock_days, r_annual=r_annual)
        )
        if not in_plan or q <= 1e-9:
            if dm <= 0.0:
                action = "DONT_BUY"
            elif oc.k <= 0.0:
                action = "WAIT"
            else:
                action = "NEGOTIATE_ALLOTMENT" if in_plan else "WAIT"
            why = _why_bullets(oc, 0.0, dcol, action, fractile, lock_days, wf_res.lambda_star)
            if in_plan and dm > 0 and action in {"NEGOTIATE_ALLOTMENT", "WAIT"}:
                why.insert(0, "Budget/priority left this cell unfunded; consider softer structure or wait.")
            recs.append(
                Recommendation(
                    cell_id=c.cell_id,
                    action=action,
                    structure=c.contract_type,
                    units=0.0,
                    cash=0.0,
                    unit_net=float(c.c_b),
                    sell_price=float(c.p),
                    fractile=float(np.clip(fractile, 0.0, 1.0)),
                    delta_m=dm,
                    c_o=co,
                    demand_p10=float(p10),
                    demand_p50=float(p50),
                    demand_p90=float(p90),
                    sellout_prob=float(sp),
                    why=why[:5],
                )
            )
            continue
        action = "COMMIT"
        cash = float(oc.k) * q
        why = _why_bullets(oc, q, dcol, action, fractile, lock_days, wf_res.lambda_star)
        recs.append(
            Recommendation(
                cell_id=c.cell_id,
                action=action,
                structure=c.contract_type,
                units=float(q),
                cash=float(cash),
                unit_net=float(c.c_b),
                sell_price=float(c.p),
                fractile=float(np.clip(fractile, 0.0, 1.0)),
                delta_m=dm,
                c_o=co,
                demand_p10=float(p10),
                demand_p50=float(p50),
                demand_p90=float(p90),
                sellout_prob=float(sp),
                why=why,
            )
        )

    return PlanResult(
        recommendations=recs,
        lambda_star=float(wf_res.lambda_star),
        cash_deployed=cash_deployed,
        reserve=float(reserve),
        structural_uplift=float(structural),
        capital_uplift=float(capital),
        roi_on_cash=float(roi_on_cash),
        expected_cash_spoilage=float(eval_metrics["expected_cash_spoilage"]),
        fill_rate_uplift=float(eval_metrics["fill_rate_uplift"]),
        budget=float(budget),
        spend=float(wf_res.spend),
        candidate_ids=selected_ids,
        scenario_shape=(int(demand_s.shape[0]), int(demand_s.shape[1])),
        waterfill=wf_res,
    )


def _format_inr(x: float) -> str:
    return f"Rs {x:,.2f}"


def _print_headline_table(pr: "PlanResult", budget: float) -> None:
    recs = pr.recommendations
    committed = [r for r in recs if r.action == "COMMIT"]
    top = sorted(committed, key=lambda r: r.cash, reverse=True)[:10]
    print("=== Planner Headline Results (simulator outputs under stated assumptions, not TBO real P&L) ===")
    print(f"budget={_format_inr(budget)} cash_deployed={_format_inr(pr.cash_deployed)} reserve={_format_inr(pr.reserve)} unspent={_format_inr(max(0.0,budget-pr.cash_deployed-pr.reserve))} lambda={pr.lambda_star:.6g}")
    print(f"capital_uplift={_format_inr(pr.capital_uplift)} structural_uplift={_format_inr(pr.structural_uplift)} roi_on_cash={pr.roi_on_cash:.4f} expected_cash_spoilage={_format_inr(pr.expected_cash_spoilage)} fill_rate_uplift={pr.fill_rate_uplift*100:.2f}pp")
    print("")
    print("| cell_id | action | structure | units | cash_INR | fractile | delta_m_per_unit_INR | demand_p50_units | why_summary |")
    print("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for r in top:
        why = (r.why[0] if r.why else "")
        why = why.replace("|", "/")
        print(f"| {r.cell_id} | {r.action} | {r.structure} | {r.units:.2f} | {r.cash:.2f} | {r.fractile:.3f} | {r.delta_m:.2f} | {r.demand_p50:.2f} | {why} |")
    if not top:
        print("| (none committed) | - | - | 0 | 0 | 0 | 0 | 0 | - |")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Run forecast-to-optimizer planner and print headline results table")
    ap.add_argument("--budget", type=float, default=10_000_000.0)
    ap.add_argument("--history", type=str, default="data/history")
    ap.add_argument("--scenarios", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reserve-fraction", type=float, default=0.03)
    ap.add_argument("--top-k", type=int, default=18)
    args = ap.parse_args()
    pr = plan(
        budget=args.budget,
        constraints={"reserve_fraction": args.reserve_fraction},
        history_dir=args.history,
        top_k=args.top_k,
        n_scenarios=args.scenarios,
        scenario_seed=args.seed,
    )
    _print_headline_table(pr, args.budget)


__all__ = ["Recommendation", "PlanResult", "plan", "candidate_funnel", "main"]


if __name__ == "__main__":
    main()
