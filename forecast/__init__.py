"""Forecasting package (baselines, splits, demand distribution engine).

Submodules: baseline, cancellations, conformal, features, quantile_model, reconcile,
            scenarios, splits, unconstrain.
Import submodules directly (e.g. from forecast.splits import ...) to avoid heavy transitive imports.
Censored history only for training/model code; latent evaluation data must never be read here.
"""
