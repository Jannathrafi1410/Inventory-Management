"""Quantile demand forecasting with LightGBM.

Trains one global LightGBM model per quantile on an unconstrained (censored-adjusted) target.
Optionally weights rows by economic sensitivity (delta_m + c_o). Predicts quantiles and can
produce scenario-ready distributions.

Assumption: features are point-in-time and censored-safe; target for training is the unconstrained
demand estimate (not latent ground truth). Missing feature handling delegated to LightGBM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# LightGBM is imported lazily inside fit() to avoid hanging at module-import time
# on systems where the native DLL initialisation blocks (Windows DLL-load issue).
lgb = None  # will be set on first use

DEFAULT_QUANTILES: Tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)


def _get_lgb():
    """Return lightgbm module, importing it on first call."""
    global lgb
    if lgb is None:
        import lightgbm as _lgb  # noqa: PLC0415
        lgb = _lgb
    return lgb


@dataclass
class QuantileModelConfig:
    """Configuration for quantile LightGBM models."""

    quantiles: Tuple[float, ...] = DEFAULT_QUANTILES
    num_leaves: int = 31
    learning_rate: float = 0.05
    n_estimators: int = 200
    min_child_samples: int = 10
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    random_state: int = 42
    use_economic_weights: bool = False


class QuantileDemandModel:
    """Global LightGBM quantile regression model (one booster per quantile)."""

    def __init__(self, config: Optional[QuantileModelConfig] = None) -> None:
        self.config = config or QuantileModelConfig()
        self.models: Dict[float, object] = {}
        self.feature_columns: List[str] = []
        self.fitted: bool = False

    @staticmethod
    def _economic_weights(df: pd.DataFrame) -> np.ndarray:
        """Row weights = delta_m + c_o when available, else 1."""
        if {"p", "c_b", "c_d", "a", "g", "k", "r"}.issubset(df.columns):
            delta_m = (df["p"] - df["c_b"]) - df["a"] * (df["p"] - df["c_d"]) + (1.0 - df["a"]) * df["g"]
            c_o = df["k"] * (1.0 - df["r"])
            w = (delta_m + c_o).astype(float).to_numpy()
            w = np.clip(w, 1e-3, None)
            # normalize to mean 1 for stable training
            return w / np.mean(w)
        return np.ones(len(df), dtype=float)

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray | pd.Series,
        feature_columns: Optional[Sequence[str]] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "QuantileDemandModel":
        """Fit one model per quantile."""
        lgb = _get_lgb()
        if lgb is None:
            raise ImportError("lightgbm is required for QuantileDemandModel")
        if feature_columns is None:
            feature_columns = [c for c in X.columns if np.issubdtype(X[c].dtype, np.number)]
        self.feature_columns = list(feature_columns)
        Xf = X[self.feature_columns].fillna(0.0)
        y_arr = np.asarray(y, dtype=float)
        if sample_weight is None and self.config.use_economic_weights:
            sample_weight = self._economic_weights(X)
        self.models = {}
        for q in self.config.quantiles:
            params = dict(
                objective="quantile",
                alpha=float(q),
                num_leaves=int(self.config.num_leaves),
                learning_rate=float(self.config.learning_rate),
                n_estimators=int(self.config.n_estimators),
                min_child_samples=int(self.config.min_child_samples),
                subsample=float(self.config.subsample),
                colsample_bytree=float(self.config.colsample_bytree),
                random_state=int(self.config.random_state),
                verbosity=-1,
                force_col_wise=True,
            )
            model = lgb.LGBMRegressor(**params)
            if sample_weight is not None:
                model.fit(Xf, y_arr, sample_weight=sample_weight)
            else:
                model.fit(Xf, y_arr)
            self.models[float(q)] = model
        self.fitted = True
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame with columns q_<quantile> for each fitted quantile."""
        if not self.fitted:
            raise RuntimeError("Model not fitted")
        Xf = X[self.feature_columns].fillna(0.0) if self.feature_columns else X.fillna(0.0)
        out = pd.DataFrame(index=X.index)
        for q, model in sorted(self.models.items()):
            preds = np.asarray(model.predict(Xf), dtype=float)
            out[f"q_{q}"] = np.maximum(preds, 0.0)
        # enforce monotonicity across quantiles per row (isotonic-like cumulative fix)
        cols = [f"q_{q}" for q in sorted(self.models.keys())]
        arr = out[cols].to_numpy(dtype=float)
        arr = np.maximum.accumulate(arr, axis=1)
        out[cols] = arr
        return out

    def predict(self, X: pd.DataFrame, quantile: float = 0.5) -> np.ndarray:
        """Predict a single quantile (default median)."""
        qdf = self.predict_quantiles(X)
        col = f"q_{quantile}"
        if col not in qdf.columns:
            # nearest available
            qs = sorted(self.models.keys())
            nearest = min(qs, key=lambda x: abs(x - quantile))
            col = f"q_{nearest}"
        return qdf[col].to_numpy(dtype=float)


__all__ = ["QuantileDemandModel", "QuantileModelConfig", "DEFAULT_QUANTILES"]
