"""LightGBM frequency / severity / Tweedie models with insurance-appropriate objectives."""

from __future__ import annotations

import lightgbm as lgb
import pandas as pd

from .config import N_JOBS, SEED
from .features import CATEGORICAL, FEATURES, add_features, to_lgbm_frame

# Pricing sense-check: more bonus-malus (worse claims record) must never lower frequency.
MONOTONE = {"BonusMalus": 1}

_COMMON = dict(
    n_estimators=3000,
    subsample=0.8,
    subsample_freq=1,
    n_jobs=N_JOBS,
    random_state=SEED,
    verbose=-1,
)
FREQ_DEFAULTS = dict(
    _COMMON,
    objective="poisson",
    learning_rate=0.05,
    num_leaves=16,
    min_child_samples=300,
    colsample_bytree=0.8,
    reg_lambda=5.0,
    monotone_constraints=[MONOTONE.get(c, 0) for c in FEATURES],
)
SEV_DEFAULTS = dict(
    _COMMON,
    objective="gamma",
    learning_rate=0.03,
    num_leaves=8,
    min_child_samples=100,
    colsample_bytree=0.8,
    reg_lambda=10.0,
)
TWEEDIE_DEFAULTS = dict(
    _COMMON,
    objective="tweedie",
    tweedie_variance_power=1.5,
    learning_rate=0.05,
    num_leaves=16,
    min_child_samples=300,
    colsample_bytree=0.8,
    reg_lambda=5.0,
)


def prepare(df: pd.DataFrame, categories: dict) -> pd.DataFrame:
    """Raw policy rows -> LightGBM feature matrix."""
    return to_lgbm_frame(add_features(df), categories)


def freq_xyw(df: pd.DataFrame, categories: dict):
    """Frequency target = claims per year of exposure, weighted by exposure."""
    return prepare(df, categories), df["ClaimNb"] / df["Exposure"], df["Exposure"]


def sev_xyw(df: pd.DataFrame, categories: dict):
    """Severity target = average cost per claim, claims-only, weighted by claim count."""
    c = df[df["ClaimNb"] > 0]
    return prepare(c, categories), c["ClaimAmount"] / c["ClaimNb"], c["ClaimNb"]


def tweedie_xyw(df: pd.DataFrame, categories: dict):
    """Pure-premium target = loss per year of exposure, weighted by exposure."""
    return prepare(df, categories), df["ClaimAmount"] / df["Exposure"], df["Exposure"]


def fit_lgbm(xtr, ytr, wtr, xva, yva, wva, params: dict, patience: int = 100):
    """Fit with early stopping on the validation set (uses the objective's own metric)."""
    model = lgb.LGBMRegressor(**params)
    model.fit(
        xtr,
        ytr,
        sample_weight=wtr,
        eval_set=[(xva, yva)],
        eval_sample_weight=[wva],
        categorical_feature=CATEGORICAL,
        callbacks=[lgb.early_stopping(patience, verbose=False)],
    )
    return model
