"""Load, clean, validate and split the freMTPL2 motor datasets.

Source: French motor third-party liability portfolio (freMTPL2freq / freMTPL2sev),
published on OpenML (ids 41214 and 41215) from the CASdatasets R package.

Set ``MOTOR_PRICING_SYNTHETIC=1`` to use a small synthetic portfolio with the same
schema (used by the tests / CI so that nothing needs the network).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_DIR, PROCESSED_DIR, SEED

FREQ_OPENML_ID = 41214
SEV_OPENML_ID = 41215
CLAIM_CAP = 200_000.0  # cap on total claim amount per policy (extreme-loss truncation)
MAX_CLAIMS = 4

BRANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B10", "B11", "B12", "B13", "B14"]
REGIONS = [
    "R11", "R21", "R22", "R23", "R24", "R25", "R26", "R31", "R41", "R42", "R43",
    "R52", "R53", "R54", "R72", "R73", "R74", "R82", "R83", "R91", "R93", "R94",
]  # fmt: skip


# --------------------------------------------------------------------------- #
# Raw data
# --------------------------------------------------------------------------- #
def fetch_openml_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download (and cache under data/openml_cache) the raw frequency & severity tables."""
    from sklearn.datasets import fetch_openml

    cache = str(DATA_DIR / "openml_cache")
    freq = fetch_openml(data_id=FREQ_OPENML_ID, as_frame=True, data_home=cache).data
    sev = fetch_openml(data_id=SEV_OPENML_ID, as_frame=True, data_home=cache).data
    return freq, sev


def make_synthetic(n: int = 60_000, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic portfolio with the raw freMTPL2 schema and a known risk structure."""
    rng = np.random.default_rng(seed)
    bonus = np.clip(50 + rng.gamma(1.5, 22, n).round(), 50, 230)
    drv_age = np.clip(rng.normal(45, 14, n).round(), 18, 100)
    veh_age = np.minimum(rng.gamma(2.5, 3, n).astype(int), 100)
    area = rng.choice(list("ABCDEF"), n)
    gas = rng.choice(["Diesel", "Regular"], n)
    freq = pd.DataFrame(
        {
            "IDpol": np.arange(1, n + 1, dtype=float),
            "ClaimNb": 0.0,
            "Exposure": np.where(rng.random(n) < 0.4, 1.0, rng.uniform(0.02, 1.0, n)),
            "Area": area,
            "VehPower": rng.integers(4, 16, n).astype(float),
            "VehAge": veh_age.astype(float),
            "DrivAge": drv_age,
            "BonusMalus": bonus,
            "VehBrand": rng.choice(BRANDS, n),
            "VehGas": gas,
            "Density": np.exp(rng.normal(6, 1.8, n)).round().clip(1, 27_000),
            "Region": rng.choice(REGIONS, n),
        }
    )
    area_eff = pd.Series(area).map(dict(zip("ABCDEF", np.linspace(-0.2, 0.3, 6), strict=True))).to_numpy()
    log_rate = (
        -2.9
        + area_eff
        + 0.012 * (bonus - 70)
        + 0.7 * (drv_age < 25)
        + 0.25 * (drv_age < 30)
        - 0.15 * (drv_age > 60)
        + 0.2 * (veh_age < 2)
        + 0.05 * np.log1p(freq["Density"].to_numpy())
    )
    freq["ClaimNb"] = rng.poisson(np.exp(log_rate) * freq["Exposure"].to_numpy()).astype(float)

    mean_sev = 1800 * np.exp(0.25 * (drv_age < 25) + 0.15 * (gas == "Diesel"))
    rows = np.repeat(np.arange(n), freq["ClaimNb"].astype(int).to_numpy())
    shape = 1.5
    amounts = rng.gamma(shape, mean_sev[rows] / shape)
    sev = pd.DataFrame({"IDpol": freq["IDpol"].to_numpy()[rows], "ClaimAmount": amounts})
    return freq, sev


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def build_dataset(freq: pd.DataFrame, sev: pd.DataFrame) -> pd.DataFrame:
    """Merge frequency and severity tables and apply documented cleaning rules."""
    freq = freq.copy()
    freq["IDpol"] = freq["IDpol"].astype(int)
    sev = sev.copy()
    sev["IDpol"] = sev["IDpol"].astype(int)

    total = sev.groupby("IDpol", as_index=False)["ClaimAmount"].sum()
    df = freq.merge(total, on="IDpol", how="left")
    df["ClaimAmount"] = df["ClaimAmount"].fillna(0.0).astype(float)

    # Rule 1: claims reported but no amount recorded -> treat as no claim.
    df.loc[(df["ClaimNb"] > 0) & (df["ClaimAmount"] == 0), "ClaimNb"] = 0
    # Rule 2: an amount exists but no claim counted -> at least one claim happened.
    df.loc[(df["ClaimNb"] == 0) & (df["ClaimAmount"] > 0), "ClaimNb"] = 1
    # Rule 3: truncate extreme counts / exposures / losses.
    df["ClaimNb"] = df["ClaimNb"].clip(upper=MAX_CLAIMS).astype(int)
    df["Exposure"] = df["Exposure"].clip(lower=1e-3, upper=1.0).astype(float)
    df["ClaimAmount"] = df["ClaimAmount"].clip(upper=CLAIM_CAP)

    for col in ["VehPower", "VehAge", "DrivAge", "BonusMalus"]:
        df[col] = df[col].astype(int)
    df["Density"] = df["Density"].astype(float)
    for col in ["Area", "VehBrand", "VehGas", "Region"]:
        # OpenML ships these categoricals with literal quotes ('Diesel'); strip them.
        df[col] = df[col].astype(str).str.strip().str.strip("'\"").str.strip().astype(object)

    cols = [
        "IDpol", "ClaimNb", "Exposure", "ClaimAmount",
        "Area", "VehPower", "VehAge", "DrivAge", "BonusMalus",
        "VehBrand", "VehGas", "Density", "Region",
    ]  # fmt: skip
    return df[cols].reset_index(drop=True)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Schema + sanity checks (raises ``pandera.errors.SchemaError`` on failure)."""
    import pandera.pandas as pa

    schema = pa.DataFrameSchema(
        {
            "IDpol": pa.Column(int, unique=True),
            "ClaimNb": pa.Column(int, pa.Check.in_range(0, MAX_CLAIMS)),
            "Exposure": pa.Column(float, pa.Check.in_range(0, 1.0)),
            "ClaimAmount": pa.Column(float, pa.Check.in_range(0, CLAIM_CAP)),
            "Area": pa.Column(str, pa.Check.isin(list("ABCDEF"))),
            "VehPower": pa.Column(int, pa.Check.ge(0)),
            "VehAge": pa.Column(int, pa.Check.ge(0)),
            "DrivAge": pa.Column(int, pa.Check.in_range(18, 120)),
            "BonusMalus": pa.Column(int, pa.Check.in_range(50, 350)),
            "VehBrand": pa.Column(str),
            "VehGas": pa.Column(str, pa.Check.isin(["Diesel", "Regular"])),
            "Density": pa.Column(float, pa.Check.gt(0)),
            "Region": pa.Column(str),
        },
        checks=[
            pa.Check(lambda d: ((d["ClaimNb"] > 0) == (d["ClaimAmount"] > 0)).all(),
                     error="ClaimNb and ClaimAmount must agree on whether a claim happened"),
        ],
        coerce=True,
    )  # fmt: skip
    return schema.validate(df)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def load_data(use_cache: bool = True, synthetic: bool | None = None) -> pd.DataFrame:
    """Return the cleaned, validated policy-level dataset (one row per policy)."""
    if synthetic is None:
        synthetic = os.environ.get("MOTOR_PRICING_SYNTHETIC") == "1"
    if synthetic:
        return validate(build_dataset(*make_synthetic()))

    cache = PROCESSED_DIR / "freMTPL2_clean.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)

    df = validate(build_dataset(*fetch_openml_raw()))
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


def split_data(
    df: pd.DataFrame,
    valid_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified (on claim / no claim) train / validation / test split.

    freMTPL2 has no date column and exactly one row per policy, so there is neither a
    time axis nor repeated policyholders to group on; a stratified random split is the
    appropriate choice here. With real data you would split by underwriting year.
    """
    strat = (df["ClaimNb"] > 0).astype(int)
    train_valid, test = train_test_split(df, test_size=test_size, stratify=strat, random_state=seed)
    strat_tv = (train_valid["ClaimNb"] > 0).astype(int)
    train, valid = train_test_split(
        train_valid,
        test_size=valid_size / (1 - test_size),
        stratify=strat_tv,
        random_state=seed,
    )
    return (
        train.reset_index(drop=True),
        valid.reset_index(drop=True),
        test.reset_index(drop=True),
    )
