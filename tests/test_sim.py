"""Tests for synthetic data generator: schema, censoring, economics, determinism, and isolation.

Assumptions: simulator outputs are under stated assumptions, not TBO real P&L. Tests use a fixed
seed and generated tables in temp dirs; sell-out rate "intended range" interpreted as 10-40% overall
(consistent with fallback availability targets ~0.55-0.87 in candidate cells).
"""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sim.generator import generate_synthetic_data
from sim.schema import SCHEMAS


def _hash_dir_parquet(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*.parquet")):
        rel = str(p.relative_to(root)).encode("utf-8")
        h.update(rel)
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory) -> dict[str, pd.DataFrame]:
    out = tmp_path_factory.mktemp("sim_out")
    tables = generate_synthetic_data(master_seed=123, out_dir=out)
    # also keep path via attribute-like access not needed; tests read from returned dfs and files
    tables["_out_dir"] = out  # type: ignore[assignment]
    return tables


def test_required_tables_and_columns_exist(generated: dict) -> None:
    for table, cols in SCHEMAS.items():
        assert table in generated, f"missing table {table}"
        df = generated[table]
        for col in cols:
            assert col in df.columns, f"{table} missing column {col}"
    assert "ground_truth" in generated
    gt = generated["ground_truth"]
    assert {"latent_demand", "capacity", "observed_sales", "dynamic_margin", "fallback_availability"}.issubset(
        gt.columns
    )


def test_history_never_contains_values_above_capacity(generated: dict) -> None:
    gt = generated["ground_truth"]
    # observed/censored sales must never exceed capacity and never exceed latent demand
    assert (gt["observed_sales"] <= gt["capacity"]).all()
    assert (gt["observed_sales"] <= np.floor(gt["latent_demand"] + 1e-9)).all()
    # inventory commits are censored to observed sales
    commits = generated["inventory_commit"]
    assert (commits["units_sold"] <= commits["units"]).all()
    # quotes rooms_left consistent with non-negative and sell-out flag
    quotes = generated["quote"]
    assert (quotes["rooms_left"] >= 0).all()


def test_mean_dynamic_margin_within_target(generated: dict) -> None:
    gt = generated["ground_truth"]
    mean_margin = float(gt["dynamic_margin"].mean())
    assert abs(mean_margin - 0.055) <= 0.003, f"mean dynamic margin {mean_margin:.6f} out of 5.5% +/- 0.3pp"


def test_sell_out_rate_in_intended_range(generated: dict) -> None:
    gt = generated["ground_truth"]
    sell_out_rate = float(gt["sell_out"].mean())
    # intended range for this simulator mix (anchors target a 0.5-0.87 => sell-out 13-50%); assert broad sane band
    assert 0.10 <= sell_out_rate <= 0.45, f"sell-out rate {sell_out_rate:.4f} outside intended range"


def test_cross_cell_correlation_within_group_higher_than_across(generated: dict) -> None:
    gt = generated["ground_truth"].copy()
    # pivot latent demand by (date) x (cell_id) within same region groups vs across
    pivot = gt.pivot_table(index="date", columns="cell_id", values="latent_demand")
    cell_to_group = gt.drop_duplicates("cell_id").set_index("cell_id")["region_group"].to_dict()
    cells = list(pivot.columns)
    within = []
    across = []
    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            a, b = cells[i], cells[j]
            s = pivot[[a, b]].dropna()
            if len(s) < 10:
                continue
            corr = float(s[a].corr(s[b]))
            if np.isnan(corr):
                continue
            if cell_to_group[a] == cell_to_group[b]:
                within.append(corr)
            else:
                across.append(corr)
    assert within and across, "insufficient pairs for correlation test"
    assert np.mean(within) > np.mean(across) + 0.05, (
        f"within-group corr {np.mean(within):.3f} not clearly higher than across-group {np.mean(across):.3f}"
    )


def test_regeneration_same_seed_is_byte_identical(tmp_path: Path) -> None:
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    generate_synthetic_data(master_seed=77, out_dir=out1)
    generate_synthetic_data(master_seed=77, out_dir=out2)
    assert _hash_dir_parquet(out1) == _hash_dir_parquet(out2)


def test_ground_truth_never_imported_by_forecast_or_optimizer() -> None:
    # Enforce isolation: forecast/ and optimizer/ source must not reference ground_truth.
    root = Path(__file__).resolve().parents[1]
    offenders: list[str] = []
    for pkg in ("forecast", "optimizer"):
        pkg_dir = root / pkg
        if not pkg_dir.exists():
            continue
        for py in pkg_dir.rglob("*.py"):
            text = py.read_text(encoding="utf-8", errors="ignore")
            if "ground_truth" in text:
                offenders.append(str(py.relative_to(root)))
    assert not offenders, f"ground_truth referenced in model/optimizer code: {offenders}"
