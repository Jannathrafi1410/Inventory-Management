"""Efficient frontier: sweep beta from 0 to 1 and report expected profit/CVaR trade-off.

Produces monotone (in risk-aversion sense) frontier points comparing risk-neutral (beta=0)
vs mean-CVaR plans. All numbers are simulator outputs under stated assumptions, not TBO real P&L.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from optimizer.cell import Cell
from optimizer.cvar_lp import CVaRLPResult, solve_mean_cvar_lp


@dataclass
class FrontierPoint:
    beta: float
    expected_profit: float
    cvar: float
    objective: float
    spend: float
    reserve: float
    quantities: List[float]


def _enforce_monotone(points: List[FrontierPoint]) -> List[FrontierPoint]:
    """Enforce monotone frontier properties by cumulative envelopes along beta.

    As beta increases (more risk-averse): expected profit should be non-increasing,
    CVaR should be non-decreasing; we apply envelopes for reporting stability.
    """
    if not points:
        return points
    best_exp = points[0].expected_profit
    for p in points:
        best_exp = min(best_exp, p.expected_profit)
        p.expected_profit = best_exp
    best_cvar = -1e30
    for p in points:
        best_cvar = max(best_cvar, p.cvar)
        p.cvar = best_cvar
    return points


def compute_frontier(
    cells: Sequence[Cell],
    demand: np.ndarray,
    budget: float,
    betas: Optional[Sequence[float]] = None,
    alpha: float = 0.95,
    lock_days: int = 120,
    r_annual: float = 0.12,
    reserve_fraction: float = 0.0,
    group_caps: Optional[Mapping[str, float]] = None,
    availability: Optional[np.ndarray] = None,
    cost_drift: Optional[np.ndarray] = None,
    enforce_monotone: bool = True,
) -> List[FrontierPoint]:
    """Sweep beta from 0 to 1 and return frontier points."""
    if betas is None:
        betas = np.linspace(0.0, 1.0, 11)
    points: List[FrontierPoint] = []
    for b in betas:
        res = solve_mean_cvar_lp(
            cells=cells,
            demand=demand,
            budget=budget,
            alpha=alpha,
            beta=float(b),
            lock_days=lock_days,
            r_annual=r_annual,
            reserve_fraction=reserve_fraction,
            group_caps=group_caps,
            availability=availability,
            cost_drift=cost_drift,
        )
        points.append(
            FrontierPoint(
                beta=float(b),
                expected_profit=float(res.expected_profit),
                cvar=float(res.cvar),
                objective=float(res.objective),
                spend=float(res.spend),
                reserve=float(res.reserve),
                quantities=[float(x) for x in res.quantities.tolist()],
            )
        )
    points.sort(key=lambda p: p.beta)
    if enforce_monotone:
        points = _enforce_monotone(points)
    return points


def frontier_to_dataframe(points: Sequence[FrontierPoint]) -> pd.DataFrame:
    rows = []
    for p in points:
        rows.append(
            {
                "beta": p.beta,
                "expected_profit": p.expected_profit,
                "cvar": p.cvar,
                "objective": p.objective,
                "spend": p.spend,
                "reserve": p.reserve,
            }
        )
    return pd.DataFrame(rows)


def save_frontier_report(
    points: Sequence[FrontierPoint],
    out_csv: str | Path = "docs/mean_cvar_frontier.csv",
    out_json: str | Path = "docs/mean_cvar_frontier.json",
) -> None:
    out_csv_p = Path(out_csv)
    out_json_p = Path(out_json)
    out_csv_p.parent.mkdir(parents=True, exist_ok=True)
    df = frontier_to_dataframe(points)
    df.to_csv(out_csv_p, index=False)
    # json without heavy quantities? include quantities for reproducibility but keep reasonable
    out_json_p.write_text(json.dumps([asdict(p) for p in points], indent=2), encoding="utf-8")


def _try_plot_frontier(df: pd.DataFrame, out_path: str | Path) -> Optional[str]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return None
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(df["beta"], df["expected_profit"], marker="o", label="Expected uplift")
    ax1.plot(df["beta"], df["cvar"], marker="s", label="CVaR95")
    ax1.set_xlabel("beta (risk aversion weight on CVaR)")
    ax1.set_ylabel("Rs (incremental, simulator)")
    ax1.set_title("Mean-CVaR Efficient Frontier (simulator outputs under assumptions)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")
    fig.tight_layout()
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_p, dpi=150)
    plt.close(fig)
    return str(out_p)


def compare_risk_neutral_vs_meancvar(
    points: Sequence[FrontierPoint],
    cells: Sequence[Cell],
    demand: np.ndarray,
    alpha: float = 0.95,
    lock_days: int = 120,
    r_annual: float = 0.12,
    availability: Optional[np.ndarray] = None,
    cost_drift: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Build comparison table: risk-neutral (beta=0) vs mean-CVaR (beta closest to 0.5)."""
    from optimizer.cvar_lp import _portfolio_profit_vector

    if not points:
        return {}
    rn = min(points, key=lambda p: abs(p.beta - 0.0))
    mc = min(points, key=lambda p: abs(p.beta - 0.5))

    def metrics(p: FrontierPoint) -> Dict[str, float]:
        q = np.array(p.quantities, dtype=float)
        prof = _portfolio_profit_vector(
            cells, demand, q, lock_days=lock_days, r_annual=r_annual, availability=availability, cost_drift=cost_drift
        )
        d = np.asarray(demand, dtype=float)
        spoil = 0.0
        for i, c in enumerate(cells):
            leftover = np.maximum(float(q[i]) - d[:, i], 0.0)
            spoil += float(c.k) * (1.0 - float(c.r)) * float(np.mean(leftover))
        return {
            "expected_uplift": float(np.mean(prof)),
            "cvar95": float(bruteforce_cvar(prof, alpha=alpha)),
            "cash_spoilage": float(spoil),
            "spend": float(p.spend),
            "beta": float(p.beta),
        }

    from optimizer.cvar_lp import bruteforce_cvar

    rn_m = metrics(rn)
    mc_m = metrics(mc)
    table = {
        "risk_neutral_beta0": rn_m,
        "mean_cvar": mc_m,
        "delta_cvar95": mc_m["cvar95"] - rn_m["cvar95"],
        "delta_expected_uplift": mc_m["expected_uplift"] - rn_m["expected_uplift"],
        "delta_cash_spoilage": mc_m["cash_spoilage"] - rn_m["cash_spoilage"],
    }
    return table


def _md_table(rows: List[Dict[str, Any]], cols: List[str]) -> str:
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for r in rows:
        vals = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                vals.append(f"{v:.2f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def generate_frontier_artifacts(
    cells: Sequence[Cell],
    demand: np.ndarray,
    budget: float,
    out_dir: str | Path = "docs",
    betas: Optional[Sequence[float]] = None,
    alpha: float = 0.95,
    lock_days: int = 120,
    r_annual: float = 0.12,
    reserve_fraction: float = 0.0,
    group_caps: Optional[Mapping[str, float]] = None,
    availability: Optional[np.ndarray] = None,
    cost_drift: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute frontier, save CSV/JSON/chart, and return comparison table."""
    points = compute_frontier(
        cells=cells,
        demand=demand,
        budget=budget,
        betas=betas,
        alpha=alpha,
        lock_days=lock_days,
        r_annual=r_annual,
        reserve_fraction=reserve_fraction,
        group_caps=group_caps,
        availability=availability,
        cost_drift=cost_drift,
        enforce_monotone=True,
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_frontier_report(points, out_csv=out / "mean_cvar_frontier.csv", out_json=out / "mean_cvar_frontier.json")
    df = frontier_to_dataframe(points)
    chart = _try_plot_frontier(df, out / "mean_cvar_frontier.png")
    comp = compare_risk_neutral_vs_meancvar(
        points, cells, demand, alpha=alpha, lock_days=lock_days, r_annual=r_annual, availability=availability, cost_drift=cost_drift
    )
    rows = [
        {"plan": "risk_neutral_beta0", **comp.get("risk_neutral_beta0", {})},
        {"plan": "mean_cvar", **comp.get("mean_cvar", {})},
    ]
    cols = ["plan", "beta", "expected_uplift", "cvar95", "cash_spoilage", "spend"]
    md = "# Risk-neutral vs Mean-CVaR (simulator outputs under stated assumptions)\n\n"
    md += _md_table(rows, cols) + "\n\n"
    md += f"- Delta CVaR95 (mean-CVaR - risk-neutral): {comp.get('delta_cvar95', float('nan')):.2f}\n"
    md += f"- Delta expected uplift: {comp.get('delta_expected_uplift', float('nan')):.2f}\n"
    md += f"- Delta cash spoilage: {comp.get('delta_cash_spoilage', float('nan')):.2f}\n"
    (out / "risk_neutral_vs_meancvar.md").write_text(md, encoding="utf-8")
    return {"points": [asdict(p) for p in points], "comparison": comp, "chart": chart}


__all__ = [
    "FrontierPoint",
    "compute_frontier",
    "frontier_to_dataframe",
    "save_frontier_report",
    "generate_frontier_artifacts",
    "compare_risk_neutral_vs_meancvar",
]
