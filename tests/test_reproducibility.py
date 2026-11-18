"""Reproducibility tests for deterministic seeding and config loading."""

from __future__ import annotations

import numpy as np

from config.settings import get_rng, load_assumptions


def test_same_master_seed_and_label_identical_arrays() -> None:
    rng_a = get_rng(master_seed=42, label="sim:demand")
    rng_b = get_rng(master_seed=42, label="sim:demand")
    a = rng_a.normal(size=10)
    b = rng_b.normal(size=10)
    assert np.array_equal(a, b)


def test_different_labels_give_different_arrays() -> None:
    rng_a = get_rng(master_seed=42, label="sim:demand")
    rng_b = get_rng(master_seed=42, label="sim:shock")
    a = rng_a.normal(size=10)
    b = rng_b.normal(size=10)
    assert not np.array_equal(a, b)


def test_loading_config_twice_gives_identical_objects() -> None:
    cfg1 = load_assumptions()
    cfg2 = load_assumptions()
    assert cfg1 == cfg2
    assert cfg1.contracts["free_allotment"].k.value == 0.0
    assert cfg1.contracts["free_allotment"].r.value == 1.0
    assert cfg1.carry.annual_rate.value == 0.12
    assert cfg1.carry.lock_days.value == 120
    assert cfg1.simulator.scenarios.value == 2500
    assert cfg1.risk.alpha.value == 0.95
    assert cfg1.risk.beta.value == 0.5
