"""Unconstraining evaluation report and chart (reads latent ground truth; eval-only).

Generates docs/unconstraining_before_after.png comparing raw censored vs unconstrained estimates
against held-out latent demand on sold-out observations.

Assumption: sold-out observations are rows where ground_truth.sell_out is True for a (cell_id, date);
raw censored = observed_sales, latent = latent_demand (evaluation only, not used by forecast/optimizer).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from forecast.unconstrain import em_unconstrain_normal, projection_detruncation


def evaluate_unconstraining(
    history_dir: str = "data/history",
    ground_truth_path: str = "data/ground_truth.parquet",
    out_chart: str = "docs/unconstraining_before_after.png",
) -> Dict[str, float]:
    """Evaluate unconstraining on sold-out dates vs latent demand and write chart.

    Returns metrics dict including MAE improvements. This is the only permitted reader of latent
    demand for model-evaluation purposes (inside /eval).
    """
    gt = pd.read_parquet(ground_truth_path)
    gt["date"] = pd.to_datetime(gt["date"]).dt.normalize()
    # sold-out only
    sold = gt[gt["sell_out"] == True].copy()  # noqa: E712
    if sold.empty:
        Path(out_chart).parent.mkdir(parents=True, exist_ok=True)
        return {"n_sold_out": 0.0}

    y_cens = sold["observed_sales"].astype(float).to_numpy()
    y_lat = sold["latent_demand"].astype(float).to_numpy()
    s = sold["capacity"].astype(float).to_numpy()
    cens_flag = np.ones(len(sold), dtype=bool)

    em_res = em_unconstrain_normal(y_cens, cens_flag, s=s, max_iter=200, tol=1e-4)
    y_em = np.asarray(em_res.unconstrained, dtype=float)
    y_pd = projection_detruncation(y_cens, cens_flag, s=s)

    mae_raw = float(np.mean(np.abs(y_cens - y_lat)))
    mae_em = float(np.mean(np.abs(y_em - y_lat)))
    mae_pd = float(np.mean(np.abs(y_pd - y_lat)))
    bias_raw = float(np.mean(y_cens - y_lat))
    bias_em = float(np.mean(y_em - y_lat))
    # select best unconstrained estimator (EM or projection-detruncation) for gate/reporting
    if mae_pd <= mae_em:
        y_best = y_pd
        best_name = "projection_detruncation"
        mae_best = mae_pd
    else:
        y_best = y_em
        best_name = "em"
        mae_best = mae_em

    # chart
    out = Path(out_chart)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_plot = min(200, len(sold))
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(sold), size=n_plot, replace=False))
    x = np.arange(n_plot)
    plt.figure(figsize=(10, 5))
    plt.plot(x, y_lat[idx], label="latent demand (held-out, eval only)", linewidth=2)
    plt.plot(x, y_cens[idx], label="raw censored (observed)", alpha=0.8)
    plt.plot(x, y_em[idx], label="EM unconstrained", alpha=0.8)
    plt.plot(x, y_pd[idx], label="projection-detruncation", alpha=0.6)
    plt.title("Unconstraining on sold-out dates: before/after vs latent demand")
    plt.xlabel("sold-out observations (sample)")
    plt.ylabel("demand units")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()

    return {
        "n_sold_out": float(len(sold)),
        "mae_raw_vs_latent": mae_raw,
        "mae_em_vs_latent": mae_em,
        "mae_pd_vs_latent": mae_pd,
        "mae_unconstrained_vs_latent": mae_best,
        "unconstrained_method": best_name,
        "bias_raw": bias_raw,
        "bias_em": bias_em,
        "em_beats_raw": float(mae_em < mae_raw),
        "pd_beats_raw": float(mae_pd < mae_raw),
        "unconstrained_beats_raw": float(mae_best < mae_raw),
        "chart_path": str(out),
    }


if __name__ == "__main__":
    metrics = evaluate_unconstraining()
    print(metrics)
