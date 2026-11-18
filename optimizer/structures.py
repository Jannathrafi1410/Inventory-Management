"""Contract structure comparison and trade-off rules."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List

import numpy as np

from optimizer.cell import Cell
from optimizer.fractile import optimal_fractile
from optimizer.profit import expected_profit


# Discount bands vs dynamic by structure (midpoints used for comparison; bands in assumptions.yaml).
STRUCTURE_DEFAULTS: Dict[str, Dict[str, float]] = {
    "free_allotment": {"discount_mid": 0.16, "k_factor": 0.0, "r": 1.0},
    "soft_release": {"discount_mid": 0.25, "k_factor": 1.0, "r": 0.7},
    "hard_buyout": {"discount_mid": 0.375, "k_factor": 1.0, "r": 0.0},
    "air_bulk_fare": {"discount_mid": 0.125, "k_factor": 1.0, "r": 0.3},
}


@dataclass
class StructureComparison:
    """Comparison of a single contract structure for a base cell."""

    structure: str
    discount_vs_dynamic: float
    p: float
    c_d: float
    c_b: float
    k: float
    r: float
    delta_m: float
    c_o: float
    carry: float
    fractile: float
    expected_profit_at_opt: float
    q_opt: float


def build_structure_cell(
    base_p: float,
    base_c_d: float,
    a: float,
    g: float,
    cap: int,
    group: str,
    structure: str,
    discount_vs_dynamic: float | None = None,
    cell_id: str = "CELL",
    lock_days: int = 120,
    r_annual: float = 0.12,
) -> Cell:
    """Construct a Cell for a given contract structure from base economics.

    c_b = c_d * (1 - discount_vs_dynamic); k = c_b * k_factor (0 for free allotment).
    """
    spec = STRUCTURE_DEFAULTS[structure]
    disc = float(discount_vs_dynamic) if discount_vs_dynamic is not None else float(spec["discount_mid"])
    c_b = base_c_d * (1.0 - disc)
    k_factor = float(spec["k_factor"])
    k = c_b * k_factor
    r = float(spec["r"])
    return Cell(
        id=cell_id,
        p=base_p,
        c_d=base_c_d,
        c_b=c_b,
        k=k,
        r=r,
        a=a,
        g=g,
        cap=cap,
        contract_type=structure,
        group=group,
    )


def compare_contract_structures(
    base_p: float,
    base_c_d: float,
    a: float,
    g: float,
    cap: int,
    demand: np.ndarray,
    group: str = "G",
    lock_days: int = 120,
    r_annual: float = 0.12,
    structures: List[str] | None = None,
) -> Dict[str, object]:
    """Compare all contract structures for the same base economics and return trade-off output.

    Returns dict with per-structure metrics and a `trade_off_rule` describing how many discount
    points a release clause (refundability/lower cash destruction) is worth vs hard buyout in
    terms of unconstrained critical fractile (decision aggressiveness/value proxy).
    """
    if structures is None:
        structures = ["free_allotment", "soft_release", "hard_buyout", "air_bulk_fare"]
    d = np.asarray(demand, dtype=float).reshape(-1)
    sorted_d = np.sort(d)
    results: List[StructureComparison] = []
    for st in structures:
        cell = build_structure_cell(base_p, base_c_d, a, g, cap, group, st, lock_days=lock_days, r_annual=r_annual)
        f = optimal_fractile(cell, lock_days=lock_days, r_annual=r_annual)
        # q* from fractile on empirical CDF
        if len(sorted_d) == 0:
            q_opt = 0.0
        else:
            idx = int(np.ceil(np.clip(f, 0, 1) * len(sorted_d)) - 1)
            idx = max(0, min(idx, len(sorted_d) - 1))
            q_opt = float(min(cap, sorted_d[idx]))
        ep = expected_profit(cell, q_opt, d, lock_days=lock_days, r_annual=r_annual)
        results.append(
            StructureComparison(
                structure=st,
                discount_vs_dynamic=STRUCTURE_DEFAULTS[st]["discount_mid"],
                p=cell.p,
                c_d=cell.c_d,
                c_b=cell.c_b,
                k=cell.k,
                r=cell.r,
                delta_m=cell.delta_m(),
                c_o=cell.c_o(),
                carry=cell.carry(lock_days=lock_days, r_annual=r_annual),
                fractile=f,
                expected_profit_at_opt=ep,
                q_opt=q_opt,
            )
        )

    # Trade-off rule: discount points (pp) of hard-buyout discount worth a release clause.
    # For a target structure, find hard_buyout discount d_h s.t. fractile_hard(d_h) == fractile_struct.
    # Worth in discount points = (d_h_equiv - d_hard_baseline) * 100 (positive => release clause as valuable as extra hard discount).
    by_st = {r.structure: r for r in results}
    hard_base_disc = STRUCTURE_DEFAULTS["hard_buyout"]["discount_mid"]
    trade_rules: Dict[str, str] = {}
    for st, res in by_st.items():
        if st == "hard_buyout":
            trade_rules[st] = "Baseline: hard buyout (r=0, k=c_b)."
            continue
        target_f = res.fractile
        equiv = _equivalent_hard_discount_for_fractile(
            base_p, base_c_d, a, g, target_f, lock_days=lock_days, r_annual=r_annual
        )
        if equiv is None:
            trade_rules[st] = (
                f"{st}: release/refund terms yield F={target_f:.3f} (hard buyout cannot match at any feasible discount)."
            )
        else:
            worth_pp = (equiv - hard_base_disc) * 100.0
            trade_rules[st] = (
                f"Release/refund terms of {st} (r={res.r:.2f}, k={'0' if res.k==0 else 'c_b'}) are worth ~{worth_pp:+.1f} "
                f"discount points vs hard buyout baseline ({hard_base_disc*100:.1f}%) to achieve similar service level F={target_f:.3f} "
                f"(equivalent hard-buyout discount ~{equiv*100:.1f}%)."
            )

    return {"structures": [r.__dict__ for r in results], "trade_off_rule": trade_rules}


def _equivalent_hard_discount_for_fractile(
    p: float,
    c_d: float,
    a: float,
    g: float,
    target_f: float,
    lock_days: int = 120,
    r_annual: float = 0.12,
    grid_steps: int = 2000,
) -> float | None:
    """Find hard-buyout discount in [0,0.6] whose optimal fractile is closest to target_f."""
    if target_f <= 0.0:
        return 0.0
    best_disc = None
    best_err = 1e9
    for disc in np.linspace(0.0, 0.60, grid_steps):
        c_b = c_d * (1.0 - disc)
        cell = Cell("tmp", p, c_d, c_b, c_b, 0.0, a, g, 10**9, "hard_buyout", "G")
        f = optimal_fractile(cell, lock_days=lock_days, r_annual=r_annual)
        err = abs(f - target_f)
        if err < best_err:
            best_err = err
            best_disc = float(disc)
    if best_disc is None or best_err > 0.02:
        return None
    return best_disc


__all__ = [
    "STRUCTURE_DEFAULTS",
    "StructureComparison",
    "build_structure_cell",
    "compare_contract_structures",
]
