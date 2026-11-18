"""Typed configuration loader and deterministic seeding utilities.

Assumption: missing optional YAML keys fall back to documented defaults in this module;
all loaded assumption keys must carry value/source/confidence metadata.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np
import yaml


@dataclass(frozen=True)
class AssumptionValue:
    """A single assumption with provenance metadata."""

    value: Any
    source: str
    confidence: str


@dataclass(frozen=True)
class ContractAssumption:
    """Contract-structure assumptions."""

    discount_vs_dynamic: AssumptionValue
    k: AssumptionValue
    r: AssumptionValue
    supplier_cap: AssumptionValue | None = None
    salvage_note: AssumptionValue | None = None


@dataclass(frozen=True)
class CarryAssumptions:
    annual_rate: AssumptionValue
    lock_days: AssumptionValue


@dataclass(frozen=True)
class MarketAssumptions:
    dynamic_margin: AssumptionValue
    elasticities: Dict[str, AssumptionValue]


@dataclass(frozen=True)
class CancellationAssumptions:
    lead_time_dependent_probabilities: Dict[str, AssumptionValue]


@dataclass(frozen=True)
class SimulatorAssumptions:
    number_of_cells: AssumptionValue
    scenarios: AssumptionValue
    group_sigma: AssumptionValue


@dataclass(frozen=True)
class RiskAssumptions:
    alpha: AssumptionValue
    beta: AssumptionValue


@dataclass(frozen=True)
class AppConfig:
    contracts: Dict[str, ContractAssumption]
    carry: CarryAssumptions
    market: MarketAssumptions
    cancellation: CancellationAssumptions
    simulator: SimulatorAssumptions
    risk: RiskAssumptions
    path: Path


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "assumptions.yaml"
_MASTER_SEED_MAX = 2**32


def _require_mapping(data: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise TypeError(f"Expected mapping for {context}, got {type(data)!r}")
    return data


def _parse_assumption(node: Any, context: str) -> AssumptionValue:
    mapping = _require_mapping(node, context)
    for key in ("value", "source", "confidence"):
        if key not in mapping:
            raise KeyError(f"Missing '{key}' in assumption at {context}")
    confidence = str(mapping["confidence"]).lower()
    if confidence not in {"high", "medium", "low"}:
        raise ValueError(f"Invalid confidence '{confidence}' at {context}")
    return AssumptionValue(
        value=mapping["value"],
        source=str(mapping["source"]),
        confidence=confidence,
    )


def _parse_contract(name: str, node: Any) -> ContractAssumption:
    mapping = _require_mapping(node, f"contracts.{name}")
    supplier_cap = (
        _parse_assumption(mapping["supplier_cap"], f"contracts.{name}.supplier_cap")
        if "supplier_cap" in mapping
        else None
    )
    salvage_note = (
        _parse_assumption(mapping["salvage_note"], f"contracts.{name}.salvage_note")
        if "salvage_note" in mapping
        else None
    )
    return ContractAssumption(
        discount_vs_dynamic=_parse_assumption(
            mapping["discount_vs_dynamic"], f"contracts.{name}.discount_vs_dynamic"
        ),
        k=_parse_assumption(mapping["k"], f"contracts.{name}.k"),
        r=_parse_assumption(mapping["r"], f"contracts.{name}.r"),
        supplier_cap=supplier_cap,
        salvage_note=salvage_note,
    )


def load_assumptions(path: str | Path | None = None) -> AppConfig:
    """Load and validate assumptions YAML into typed dataclasses."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    raw_map = _require_mapping(raw, "root")

    contracts_raw = _require_mapping(raw_map.get("contracts", {}), "contracts")
    contracts = {name: _parse_contract(name, node) for name, node in contracts_raw.items()}

    carry_raw = _require_mapping(raw_map.get("carry", {}), "carry")
    carry = CarryAssumptions(
        annual_rate=_parse_assumption(carry_raw["annual_rate"], "carry.annual_rate"),
        lock_days=_parse_assumption(carry_raw["lock_days"], "carry.lock_days"),
    )

    market_raw = _require_mapping(raw_map.get("market", {}), "market")
    elasticities_raw = _require_mapping(market_raw.get("elasticities", {}), "market.elasticities")
    market = MarketAssumptions(
        dynamic_margin=_parse_assumption(market_raw["dynamic_margin"], "market.dynamic_margin"),
        elasticities={
            k: _parse_assumption(v, f"market.elasticities.{k}") for k, v in elasticities_raw.items()
        },
    )

    cancellation_raw = _require_mapping(raw_map.get("cancellation", {}), "cancellation")
    ltd = _require_mapping(
        cancellation_raw.get("lead_time_dependent_probabilities", {}),
        "cancellation.lead_time_dependent_probabilities",
    )
    buckets_raw = ltd.get("by_lead_time_bucket_days", ltd)
    buckets = _require_mapping(
        buckets_raw, "cancellation.lead_time_dependent_probabilities.by_lead_time_bucket_days"
    )
    cancellation = CancellationAssumptions(
        lead_time_dependent_probabilities={
            k: _parse_assumption(v, f"cancellation.lead_time_dependent_probabilities.{k}")
            for k, v in buckets.items()
        }
    )

    sim_raw = _require_mapping(raw_map.get("simulator", {}), "simulator")
    simulator = SimulatorAssumptions(
        number_of_cells=_parse_assumption(sim_raw["number_of_cells"], "simulator.number_of_cells"),
        scenarios=_parse_assumption(sim_raw["scenarios"], "simulator.scenarios"),
        group_sigma=_parse_assumption(sim_raw["group_sigma"], "simulator.group_sigma"),
    )

    risk_raw = _require_mapping(raw_map.get("risk", {}), "risk")
    risk = RiskAssumptions(
        alpha=_parse_assumption(risk_raw["alpha"], "risk.alpha"),
        beta=_parse_assumption(risk_raw["beta"], "risk.beta"),
    )

    return AppConfig(
        contracts=contracts,
        carry=carry,
        market=market,
        cancellation=cancellation,
        simulator=simulator,
        risk=risk,
        path=config_path,
    )


def stable_label_seed(master_seed: int, label: str) -> int:
    """Derive a stable 32-bit seed from (master_seed, label)."""
    payload = f"{int(master_seed)}::{label}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") % _MASTER_SEED_MAX


def get_rng(master_seed: int, label: str) -> np.random.Generator:
    """Return a deterministic, label-specific numpy Generator.

    Same (master_seed, label) -> identical stream. Different labels -> independent streams.
    """
    seed = stable_label_seed(master_seed, label)
    return np.random.default_rng(seed)


__all__ = [
    "AppConfig",
    "AssumptionValue",
    "ContractAssumption",
    "CarryAssumptions",
    "MarketAssumptions",
    "CancellationAssumptions",
    "SimulatorAssumptions",
    "RiskAssumptions",
    "DEFAULT_CONFIG_PATH",
    "load_assumptions",
    "stable_label_seed",
    "get_rng",
]
