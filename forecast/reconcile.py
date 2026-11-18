"""Cluster-to-hotel (cell-to-child) forecast reconciliation with shrinkage.

Disaggregates cluster-level forecasts to hotel/cell children such that child forecasts sum to the
cluster total. Uses shrinkage between child historical shares and equal shares for stability.

Assumption: "cluster" is identified by `cluster_col` (default: region_group/destination cluster) and
children by `child_col` (default: cell_id). Historical shares computed from censored observed demand.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


def compute_child_shares(
    history: pd.DataFrame,
    cluster_col: str = "cluster_id",
    child_col: str = "cell_id",
    y_col: str = "y_censored",
    shrinkage: float = 0.5,
) -> pd.DataFrame:
    """Compute shrinkage shares for children within clusters.

    share_i = (1-lambda)*hist_share_i + lambda*equal_share_i, normalized per cluster.
    """
    lam = float(np.clip(shrinkage, 0.0, 1.0))
    df = history[[cluster_col, child_col, y_col]].copy()
    child_tot = df.groupby([cluster_col, child_col], as_index=False)[y_col].sum()
    clus_tot = df.groupby(cluster_col, as_index=False)[y_col].sum().rename(columns={y_col: "cluster_total"})
    merged = child_tot.merge(clus_tot, on=cluster_col, how="left")
    merged["hist_share"] = np.where(merged["cluster_total"] > 0, merged[y_col] / merged["cluster_total"], 0.0)
    counts = merged.groupby(cluster_col)[child_col].transform("count")
    merged["equal_share"] = 1.0 / counts.replace(0, 1)
    merged["share"] = (1.0 - lam) * merged["hist_share"] + lam * merged["equal_share"]
    # renormalize per cluster
    share_sum = merged.groupby(cluster_col)["share"].transform("sum")
    merged["share"] = np.where(share_sum > 0, merged["share"] / share_sum, merged["equal_share"])
    return merged[[cluster_col, child_col, "share"]]


def reconcile_cluster_to_children(
    cluster_forecasts: pd.DataFrame,
    shares: pd.DataFrame,
    cluster_col: str = "cluster_id",
    child_col: str = "cell_id",
    forecast_col: str = "forecast",
    out_col: str = "forecast_child",
) -> pd.DataFrame:
    """Disaggregate cluster forecasts to children so sums match cluster totals.

    cluster_forecasts: rows per cluster (and optionally time index columns preserved).
    shares: rows per (cluster, child) with 'share' summing to 1 per cluster.
    """
    cf = cluster_forecasts.copy()
    sh = shares[[cluster_col, child_col, "share"]].copy()
    out = cf.merge(sh, on=cluster_col, how="left")
    if out["share"].isna().any():
        # distribute equally among known children if missing
        n = out.groupby(cluster_col)[child_col].transform("count").replace(0, 1)
        out["share"] = out["share"].fillna(1.0 / n)
    out[out_col] = out[forecast_col].astype(float) * out["share"].astype(float)
    # enforce exact sum-to-cluster by residual adjustment
    sums = out.groupby(out.index // 1 + 0)  # noop; adjust by merge keys below
    key_cols = [c for c in cf.columns if c != forecast_col]
    if key_cols:
        out["_sum"] = out.groupby(key_cols, dropna=False)[out_col].transform("sum")
        out["_target"] = out[forecast_col].astype(float)
        # proportional rescale
        out[out_col] = np.where(out["_sum"] > 0, out[out_col] * (out["_target"] / out["_sum"]), out[out_col])
        out = out.drop(columns=["_sum", "_target"])
    return out


def reconcile_quantiles(
    cluster_quantiles: pd.DataFrame,
    shares: pd.DataFrame,
    quantile_cols: Sequence[str],
    cluster_col: str = "cluster_id",
    child_col: str = "cell_id",
) -> pd.DataFrame:
    """Apply reconciliation independently per quantile column (sum-preserving disaggregation)."""
    base = None
    for qc in quantile_cols:
        tmp = cluster_quantiles[[c for c in cluster_quantiles.columns if c in [cluster_col] or c == qc] + [cluster_col]].copy()
        # ensure cluster_col present once
        tmp = cluster_quantiles.copy()
        tmp = tmp.rename(columns={qc: "forecast"})
        rec = reconcile_cluster_to_children(tmp[[cluster_col, "forecast"]], shares, cluster_col=cluster_col, child_col=child_col, forecast_col="forecast", out_col=qc)
        keep = [cluster_col, child_col, qc]
        rec = rec[keep]
        if base is None:
            base = rec
        else:
            base = base.merge(rec, on=[cluster_col, child_col], how="outer")
    assert base is not None
    return base


__all__ = ["compute_child_shares", "reconcile_cluster_to_children", "reconcile_quantiles"]
