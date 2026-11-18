"""Scenario generation from quantile forecasts.

Includes:
- Gaussian copula with group-factor correlation (legacy, preserved API).
- Correlated factor model for demand, availability, and cost drift with seeded,
  versioned scenario bundles saved to disk.

Factor model (demand, log-space):
  log D_{i,s} = log mu_i + sum_g beta_{ig} * Z_{g,s} + eps_{i,s}

where Z_g are shared latent factors (destination/region, event cancellation, FX,
macro/geopolitical, air-capacity) and eps_i is idiosyncratic noise. The same
latent shocks induce correlated availability (city-wide sell-out) and correlated
dynamic-cost drift so tail risks are not understated.

Assumptions: mu_i is proxied from quantile forecasts (median/p50). Availability
is scenario-specific effective fallback probability in [0,1]; cost drift is a
multiplicative factor on dynamic net cost c_d. All outputs are simulator outputs
under stated assumptions, not TBO real P&L. Censored history only; no latent
ground truth is read here.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from config.settings import get_rng


DEFAULT_QUANTILES: Tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
FACTOR_NAMES: Tuple[str, ...] = (
    "destination_region",
    "event_cancellation",
    "fx",
    "macro_geopolitical",
    "air_capacity",
)


@dataclass
class CopulaConfig:
    """Gaussian copula configuration."""

    intra_group_rho: float = 0.6  # within-group correlation induced by group factor
    min_std: float = 1e-3


def _quantile_to_inverse_cdf(quantiles: Sequence[float], values: np.ndarray):
    """Build a monotone inverse-CDF (quantile) function from quantile points via linear interp."""
    q = np.asarray(list(quantiles), dtype=float)
    v = np.asarray(values, dtype=float)
    order = np.argsort(q)
    q = q[order]
    v = np.maximum(v[order], 0.0)
    v = np.maximum.accumulate(v)  # enforce monotone
    # extend tails
    q_ext = np.concatenate([[0.0], q, [1.0]])
    v_lo = max(0.0, float(v[0] - (v[1] - v[0]) * 2.0 if len(v) > 1 else v[0] * 0.5))
    v_hi = float(v[-1] + (v[-1] - v[-2]) * 3.0 if len(v) > 1 else max(v[-1] * 2.0, v[-1] + 1.0))
    v_ext = np.concatenate([[v_lo], v, [v_hi]])

    def inv(u: np.ndarray) -> np.ndarray:
        u = np.clip(np.asarray(u, dtype=float), 1e-6, 1 - 1e-6)
        return np.interp(u, q_ext, v_ext)

    return inv


def sample_scenarios_gaussian_copula(
    quantile_forecasts: pd.DataFrame,
    groups: Sequence[str],
    s: int,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    intra_group_rho: float = 0.6,
    seed: int = 42,
    label: str = "scenarios",
) -> np.ndarray:
    """Sample scenario matrix demand[S, n] from quantile forecasts using a Gaussian group-factor copula.

    - quantile_forecasts: DataFrame indexed/ordered as cells with columns q_<q> for each quantile.
    - groups: group id per cell (same order as quantile_forecasts rows/columns output mapping).
    - Returns array shape [S, n] in the same cell order as `groups`/rows.
    """
    n = len(groups)
    if len(quantile_forecasts) != n:
        raise ValueError("quantile_forecasts rows must match number of cells/groups")
    q_levels = list(quantiles)
    cols = [f"q_{q}" for q in q_levels]
    for c in cols:
        if c not in quantile_forecasts.columns:
            raise ValueError(f"Missing quantile column {c}")

    rng = get_rng(seed, label)
    rho = float(np.clip(intra_group_rho, 0.0, 0.99))
    factor_loading = float(np.sqrt(rho))
    idio_scale = float(np.sqrt(max(1.0 - rho, 1e-6)))

    # group factors Z_g ~ N(0,1) shared within group across scenarios
    unique_groups = list(dict.fromkeys(groups))
    g_index = {g: i for i, g in enumerate(unique_groups)}
    z_g = rng.standard_normal(size=(s, len(unique_groups)))

    # correlated normals per cell: X_i = sqrt(rho)*Z_{g(i)} + sqrt(1-rho)*eps_i
    eps = rng.standard_normal(size=(s, n))
    x = np.zeros((s, n), dtype=float)
    for j, g in enumerate(groups):
        x[:, j] = factor_loading * z_g[:, g_index[g]] + idio_scale * eps[:, j]
    u = stats.norm.cdf(x)

    demand = np.zeros((s, n), dtype=float)
    qvals = quantile_forecasts[cols].to_numpy(dtype=float)
    for j in range(n):
        inv = _quantile_to_inverse_cdf(q_levels, qvals[j, :])
        demand[:, j] = np.maximum(inv(u[:, j]), 0.0)
    return demand


def scenarios_for_cells(
    quantile_forecasts: pd.DataFrame,
    cell_ids: Sequence[str],
    cell_groups: Sequence[str],
    s: int = 2500,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    intra_group_rho: float = 0.6,
    seed: int = 42,
    label: str = "scenarios",
) -> Tuple[np.ndarray, List[str]]:
    """Convenience wrapper ensuring order matches cell_ids and returning (demand[S,n], cell_ids list)."""
    if "cell_id" in quantile_forecasts.columns:
        qf = quantile_forecasts.set_index("cell_id").loc[list(cell_ids)].reset_index(drop=True)
    else:
        qf = (
            quantile_forecasts.loc[list(cell_ids)].reset_index(drop=True)
            if quantile_forecasts.index.name == "cell_id"
            else quantile_forecasts.reset_index(drop=True)
        )
    demand = sample_scenarios_gaussian_copula(
        qf,
        groups=list(cell_groups),
        s=s,
        quantiles=quantiles,
        intra_group_rho=intra_group_rho,
        seed=seed,
        label=label,
    )
    return demand, list(cell_ids)


@dataclass
class FactorScenarioConfig:
    """Configuration for correlated multi-factor scenario generation.

    Assumptions (medium confidence unless noted):
    - destination/region factor captures destination-level common shocks (high).
    - event_cancellation factor stresses event-heavy cells and city sell-out.
    - fx/macro_geopolitical factors create broad correlated cost/demand drift.
    - air_capacity factor stresses air/product-linked supply.
    - correlation_scale scales all shared-factor betas (>1 => stronger correlation/tails).
    """

    s: int = 2000
    seed: int = 42
    label: str = "factor_scenarios"
    idio_sigma: float = 0.18
    correlation_scale: float = 1.0
    cost_drift_sigma: float = 0.06
    availability_shock_scale: float = 0.35
    version: Optional[str] = None
    # factor standard deviations (latent Z_g ~ N(0, sigma_g^2))
    factor_sigmas: Dict[str, float] = field(
        default_factory=lambda: {
            "destination_region": 0.22,
            "event_cancellation": 0.18,
            "fx": 0.10,
            "macro_geopolitical": 0.12,
            "air_capacity": 0.10,
        }
    )


@dataclass
class FactorScenarioBundle:
    """Versioned correlated scenario bundle saved to disk."""

    version: str
    cell_ids: List[str]
    demand: np.ndarray  # [S, n]
    availability: np.ndarray  # [S, n] effective fallback probability a_{i,s}
    cost_drift: np.ndarray  # [S, n] multiplicative drift on c_d
    factors: Dict[str, np.ndarray]  # factor draws [S]
    betas: Dict[str, np.ndarray]  # loadings per cell [n]
    path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def _median_mu(quantile_forecasts: pd.DataFrame, quantiles: Sequence[float]) -> np.ndarray:
    cols = [f"q_{q}" for q in quantiles]
    for c in cols:
        if c not in quantile_forecasts.columns:
            raise ValueError(f"Missing quantile column {c}")
    if "q_0.5" in quantile_forecasts.columns:
        mu = quantile_forecasts["q_0.5"].to_numpy(dtype=float)
    else:
        # fallback: average around central quantiles
        mu = quantile_forecasts[cols].median(axis=1).to_numpy(dtype=float)
    return np.maximum(mu, 1.0)


def _cell_meta(cell_ids: Sequence[str], cell_groups: Sequence[str], cell_meta: Optional[pd.DataFrame]) -> pd.DataFrame:
    meta = pd.DataFrame({"cell_id": list(cell_ids), "region_group": list(cell_groups)})
    if cell_meta is not None:
        m = cell_meta.copy()
        if "cell_id" in m.columns:
            m = m.set_index("cell_id")
        # align
        aligned = m.reindex(list(cell_ids))
        for col in aligned.columns:
            meta[col] = aligned[col].values
    # defaults
    if "destination_city" not in meta.columns:
        meta["destination_city"] = meta["cell_id"].astype(str).str.split("-").str[0]
    if "product_type" not in meta.columns:
        meta["product_type"] = np.where(meta["cell_id"].astype(str).str.startswith("AIR"), "air", "hotel")
    if "segment" not in meta.columns:
        meta["segment"] = "leisure"
    if "a" not in meta.columns:
        meta["a"] = 0.8
    return meta


def _build_factor_loadings(meta: pd.DataFrame, correlation_scale: float = 1.0) -> Dict[str, np.ndarray]:
    """Build per-cell factor loadings beta_{ig} from cell metadata.

    Assumption: destination/region groups share strong common shocks; event segments load on
    event_cancellation; air cells load on air_capacity and FX/macro; all load on macro/FX.
    """
    n = len(meta)
    scale = float(max(correlation_scale, 0.0))
    region = meta["region_group"].astype(str).values
    dest = meta["destination_city"].astype(str).values
    seg = meta["segment"].astype(str).values
    prod = meta["product_type"].astype(str).values

    # destination/region: within same region, stronger for same destination
    beta_dest = np.full(n, 0.55 * scale, dtype=float)
    # event cancellation: high for event cells, medium for mixed
    beta_event = np.where(seg == "event", 0.75, np.where(seg == "mixed", 0.35, 0.10)) * scale
    # fx: broad, higher for non-domestic sensitivity proxy (all moderate)
    beta_fx = np.full(n, 0.30 * scale, dtype=float)
    # macro/geopolitical: broad
    beta_macro = np.full(n, 0.28 * scale, dtype=float)
    # air capacity: air products and high-dependency flows
    beta_air = np.where(prod == "air", 0.80, 0.10) * scale
    # normalize destination factor to be identical within (region,destination) key via later grouping? keep as loading
    # strengthen same-destination co-movement
    for d in np.unique(dest):
        idx = np.where(dest == d)[0]
        if len(idx) > 1:
            beta_dest[idx] = 0.70 * scale
    for r in np.unique(region):
        idx = np.where(region == r)[0]
        if len(idx) > 0:
            beta_dest[idx] = np.maximum(beta_dest[idx], 0.45 * scale)

    return {
        "destination_region": beta_dest,
        "event_cancellation": beta_event,
        "fx": beta_fx,
        "macro_geopolitical": beta_macro,
        "air_capacity": beta_air,
    }


def generate_factor_scenarios(
    quantile_forecasts: pd.DataFrame,
    cell_ids: Sequence[str],
    cell_groups: Sequence[str],
    s: int = 2000,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    seed: int = 42,
    label: str = "factor_scenarios",
    correlation_scale: float = 1.0,
    idio_sigma: float = 0.18,
    cost_drift_sigma: float = 0.06,
    availability_shock_scale: float = 0.35,
    base_availability: Optional[Sequence[float]] = None,
    cell_meta: Optional[pd.DataFrame] = None,
    factor_sigmas: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray], pd.DataFrame]:
    """Generate correlated demand/availability/cost-drift scenarios via multi-factor log model.

    Returns (demand[S,n], availability[S,n], cost_drift[S,n], factor_draws, betas, meta_aligned).
    - demand: non-negative latent demand scenarios.
    - availability: effective fallback probability in [0,1], correlated (city-wide sell-out shocks).
    - cost_drift: multiplicative factor on dynamic net cost c_d (>1 means cost pressure).
    """
    if "cell_id" in quantile_forecasts.columns:
        qf = quantile_forecasts.set_index("cell_id").loc[list(cell_ids)].reset_index(drop=True)
    else:
        qf = (
            quantile_forecasts.loc[list(cell_ids)].reset_index(drop=True)
            if quantile_forecasts.index.name == "cell_id"
            else quantile_forecasts.reset_index(drop=True)
        )
    if len(qf) != len(cell_ids):
        raise ValueError("quantile_forecasts rows must match number of cells")

    s = int(max(1, s))
    rng = get_rng(int(seed), label)
    meta = _cell_meta(cell_ids, cell_groups, cell_meta)
    if base_availability is not None:
        meta["a"] = np.asarray(list(base_availability), dtype=float)
    a0 = np.clip(meta["a"].to_numpy(dtype=float), 0.0, 1.0)

    sigmas = {
        "destination_region": 0.22,
        "event_cancellation": 0.18,
        "fx": 0.10,
        "macro_geopolitical": 0.12,
        "air_capacity": 0.10,
    }
    if factor_sigmas:
        sigmas.update({k: float(v) for k, v in factor_sigmas.items()})
    betas = _build_factor_loadings(meta, correlation_scale=correlation_scale)

    # latent factor draws Z_{g,s}
    factors: Dict[str, np.ndarray] = {}
    for g in FACTOR_NAMES:
        factors[g] = rng.standard_normal(size=s) * float(sigmas.get(g, 0.1))

    mu = _median_mu(qf, quantiles)
    log_mu = np.log(np.maximum(mu, 1e-6))
    eps = rng.standard_normal(size=(s, len(cell_ids))) * float(max(idio_sigma, 1e-6))

    log_d = np.tile(log_mu.reshape(1, -1), (s, 1)) + eps
    for g in FACTOR_NAMES:
        log_d += factors[g].reshape(-1, 1) * betas[g].reshape(1, -1)
    demand = np.exp(log_d)
    demand = np.maximum(demand, 0.0)

    # correlated availability shock: positive demand/event/air stress reduces fallback availability (city-wide sell-out)
    avail_shock = (
        0.55 * factors["destination_region"].reshape(-1, 1) * betas["destination_region"].reshape(1, -1)
        + 0.90 * factors["event_cancellation"].reshape(-1, 1) * betas["event_cancellation"].reshape(1, -1)
        + 0.70 * factors["air_capacity"].reshape(-1, 1) * betas["air_capacity"].reshape(1, -1)
    )
    avail_noise = rng.standard_normal(size=(s, len(cell_ids))) * 0.08
    availability = a0.reshape(1, -1) - float(availability_shock_scale) * avail_shock + avail_noise
    availability = np.clip(availability, 0.0, 1.0)

    # correlated dynamic-cost drift: FX/macro/air-capacity pressure increases c_d
    cost_shock = (
        0.60 * factors["fx"].reshape(-1, 1)
        + 0.55 * factors["macro_geopolitical"].reshape(-1, 1)
        + 0.45 * factors["air_capacity"].reshape(-1, 1) * betas["air_capacity"].reshape(1, -1)
        + 0.20 * factors["destination_region"].reshape(-1, 1)
    )
    cost_noise = rng.standard_normal(size=(s, len(cell_ids))) * 0.02
    cost_drift = np.exp(float(max(cost_drift_sigma, 0.0)) * cost_shock + cost_noise)
    cost_drift = np.clip(cost_drift, 0.7, 1.6)

    return demand, availability, cost_drift, factors, betas, meta


def save_scenario_bundle(bundle: FactorScenarioBundle, out_dir: str | Path = "data/scenarios") -> str:
    """Save versioned scenario bundle (npz + json manifest) to disk and return manifest path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    npz_path = out / f"{bundle.version}.npz"
    np.savez_compressed(
        npz_path,
        demand=bundle.demand,
        availability=bundle.availability,
        cost_drift=bundle.cost_drift,
        cell_ids=np.array(bundle.cell_ids, dtype=object),
        factor_destination_region=bundle.factors.get("destination_region", np.array([])),
        factor_event_cancellation=bundle.factors.get("event_cancellation", np.array([])),
        factor_fx=bundle.factors.get("fx", np.array([])),
        factor_macro_geopolitical=bundle.factors.get("macro_geopolitical", np.array([])),
        factor_air_capacity=bundle.factors.get("air_capacity", np.array([])),
        beta_destination_region=bundle.betas.get("destination_region", np.array([])),
        beta_event_cancellation=bundle.betas.get("event_cancellation", np.array([])),
        beta_fx=bundle.betas.get("fx", np.array([])),
        beta_macro_geopolitical=bundle.betas.get("macro_geopolitical", np.array([])),
        beta_air_capacity=bundle.betas.get("air_capacity", np.array([])),
    )
    manifest = {
        "version": bundle.version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "npz": str(npz_path).replace("\\", "/"),
        "n_scenarios": int(bundle.demand.shape[0]),
        "n_cells": int(bundle.demand.shape[1]),
        "cell_ids": list(bundle.cell_ids),
        "metadata": bundle.metadata,
    }
    manifest_path = out / f"{bundle.version}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    bundle.path = str(manifest_path)
    return str(manifest_path)


def load_scenario_bundle(version_or_manifest: str | Path, out_dir: str | Path = "data/scenarios") -> FactorScenarioBundle:
    """Load a saved scenario bundle from manifest path or version id."""
    p = Path(version_or_manifest)
    if p.suffix != ".json":
        p = Path(out_dir) / f"{p.name}.manifest.json"
    manifest = json.loads(p.read_text(encoding="utf-8"))
    npz = np.load(manifest["npz"], allow_pickle=True)
    factors = {g: npz[f"factor_{g}"] for g in FACTOR_NAMES if f"factor_{g}" in npz}
    betas = {g: npz[f"beta_{g}"] for g in FACTOR_NAMES if f"beta_{g}" in npz}
    return FactorScenarioBundle(
        version=str(manifest["version"]),
        cell_ids=[str(x) for x in npz["cell_ids"].tolist()],
        demand=npz["demand"],
        availability=npz["availability"],
        cost_drift=npz["cost_drift"],
        factors=factors,
        betas=betas,
        path=str(p),
        metadata=manifest.get("metadata", {}),
    )


def create_and_save_factor_scenarios(
    quantile_forecasts: pd.DataFrame,
    cell_ids: Sequence[str],
    cell_groups: Sequence[str],
    s: int = 2000,
    seed: int = 42,
    out_dir: str | Path = "data/scenarios",
    version: Optional[str] = None,
    correlation_scale: float = 1.0,
    base_availability: Optional[Sequence[float]] = None,
    cell_meta: Optional[pd.DataFrame] = None,
    label: str = "factor_scenarios",
    **kwargs: Any,
) -> FactorScenarioBundle:
    """Generate 1,000-5,000 seeded correlated scenarios and save a versioned bundle to disk."""
    s = int(np.clip(int(s), 1, 20000))
    demand, availability, cost_drift, factors, betas, meta = generate_factor_scenarios(
        quantile_forecasts,
        cell_ids=cell_ids,
        cell_groups=cell_groups,
        s=s,
        seed=seed,
        label=label,
        correlation_scale=correlation_scale,
        base_availability=base_availability,
        cell_meta=cell_meta,
        **kwargs,
    )
    ver = version or f"scen_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_s{s}_seed{seed}_cs{correlation_scale}"
    bundle = FactorScenarioBundle(
        version=ver,
        cell_ids=list(cell_ids),
        demand=demand,
        availability=availability,
        cost_drift=cost_drift,
        factors=factors,
        betas=betas,
        metadata={
            "seed": int(seed),
            "label": label,
            "s": int(s),
            "correlation_scale": float(correlation_scale),
            "factor_model": "log D_i = log mu_i + sum_g beta_ig*Z_g + eps_i",
            "assumptions": "simulator outputs under stated assumptions; correlated availability/cost drift included",
            "groups": list(cell_groups),
        },
    )
    save_scenario_bundle(bundle, out_dir=out_dir)
    return bundle


def apply_correlated_shocks_to_baseline(
    baseline_demand: np.ndarray,
    factors: Mapping[str, np.ndarray],
    betas: Mapping[str, np.ndarray],
    scale: float = 1.0,
) -> np.ndarray:
    """Apply factor shocks to a baseline demand matrix (for stress tests e.g. doubling correlation).

    Assumption: multiplicative log-space scaling of shared components; used to construct
    controlled experiments (independent vs correlated vs doubled-correlation tails).
    """
    d = np.maximum(np.asarray(baseline_demand, dtype=float), 1e-6)
    log_d = np.log(d)
    for g, z in factors.items():
        if g in betas:
            log_d = log_d + float(scale) * np.asarray(z).reshape(-1, 1) * np.asarray(betas[g]).reshape(1, -1)
    return np.maximum(np.exp(log_d), 0.0)


__all__ = [
    "sample_scenarios_gaussian_copula",
    "scenarios_for_cells",
    "CopulaConfig",
    "DEFAULT_QUANTILES",
    "FACTOR_NAMES",
    "FactorScenarioConfig",
    "FactorScenarioBundle",
    "generate_factor_scenarios",
    "create_and_save_factor_scenarios",
    "save_scenario_bundle",
    "load_scenario_bundle",
    "apply_correlated_shocks_to_baseline",
]
