"""Feature engineering shared by the GLMs, the LightGBM models and the API."""

from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "LogDensity"]
CATEGORICAL = ["Area", "VehBrand", "VehGas", "Region"]
FEATURES = NUMERIC + CATEGORICAL

DRIVAGE_BINS = [17, 20, 25, 30, 40, 50, 60, 70, 100]
DRIVAGE_LABELS = ["18-20", "21-25", "26-30", "31-40", "41-50", "51-60", "61-70", "71+"]
VEHAGE_BINS = [-1, 0, 2, 5, 10, 15, 100]
VEHAGE_LABELS = ["0", "1-2", "3-5", "6-10", "11-15", "16+"]

# Poisson GLM for claim counts (an exposure offset is passed separately)
FREQ_FORMULA = (
    "ClaimNb ~ C(DrivAgeBand, Treatment('41-50')) + C(VehAgeBand, Treatment('6-10'))"
    " + C(VehPower) + BonusMalus + LogDensity + C(Area) + C(VehGas) + C(VehBrand) + C(Region)"
)
# Gamma GLM for average claim cost (leaner: severity has far less signal)
SEV_FORMULA = (
    "AvgClaim ~ C(DrivAgeBand, Treatment('41-50')) + C(VehAgeBand, Treatment('6-10'))"
    " + BonusMalus + LogDensity + C(Area) + C(VehGas)"
)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clip outliers and add derived columns. Needs raw columns incl. ``Density``."""
    out = df.copy()
    out["VehPower"] = out["VehPower"].clip(4, 12)
    out["VehAge"] = out["VehAge"].clip(0, 20)
    out["DrivAge"] = out["DrivAge"].clip(18, 90)
    out["BonusMalus"] = out["BonusMalus"].clip(50, 150)
    out["LogDensity"] = np.log1p(out["Density"])
    return out


def add_glm_bands(df: pd.DataFrame) -> pd.DataFrame:
    """Add the age bands used by the GLMs (expects output of :func:`add_features`)."""
    out = df.copy()
    out["DrivAgeBand"] = pd.cut(out["DrivAge"], DRIVAGE_BINS, labels=DRIVAGE_LABELS).astype(str)
    out["VehAgeBand"] = pd.cut(out["VehAge"], VEHAGE_BINS, labels=VEHAGE_LABELS).astype(str)
    return out


def learn_categories(df: pd.DataFrame) -> dict[str, list[str]]:
    """Category levels seen in training; stored with the model so scoring is consistent."""
    return {c: sorted(df[c].astype(str).unique()) for c in CATEGORICAL}


def to_lgbm_frame(df: pd.DataFrame, categories: dict[str, list[str]]) -> pd.DataFrame:
    """Model matrix for LightGBM: native pandas categoricals with fixed levels."""
    x = df[FEATURES].copy()
    for col in CATEGORICAL:
        x[col] = pd.Categorical(x[col].astype(str), categories=categories[col])
    return x
