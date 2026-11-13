import pandas as pd
import pytest

from motor_pricing.data import CLAIM_CAP, MAX_CLAIMS, build_dataset, make_synthetic, validate


def test_cleaning_invariants(portfolio):
    assert portfolio["IDpol"].is_unique
    assert portfolio["ClaimNb"].between(0, MAX_CLAIMS).all()
    assert portfolio["Exposure"].between(0, 1).all()
    assert (portfolio["ClaimAmount"] <= CLAIM_CAP).all()
    # claim count and claim amount must agree on whether a claim happened
    assert ((portfolio["ClaimNb"] > 0) == (portfolio["ClaimAmount"] > 0)).all()


def test_inconsistent_raw_rows_are_repaired():
    freq, sev = make_synthetic(n=500, seed=1)
    freq.loc[0, "ClaimNb"] = 2.0  # claims reported, no amount recorded
    sev = sev[sev["IDpol"] != freq.loc[0, "IDpol"]]
    freq.loc[1, "ClaimNb"] = 0.0  # amount recorded, no claim counted
    sev = pd.concat([sev, pd.DataFrame({"IDpol": [freq.loc[1, "IDpol"]], "ClaimAmount": [900.0]})])
    out = build_dataset(freq, sev).set_index("IDpol")
    assert out.loc[int(freq.loc[0, "IDpol"]), "ClaimNb"] == 0
    assert out.loc[int(freq.loc[1, "IDpol"]), "ClaimNb"] == 1


def test_validate_rejects_bad_exposure(portfolio):
    bad = portfolio.copy()
    bad.loc[0, "Exposure"] = 3.0
    with pytest.raises(Exception):  # noqa: B017 - pandera SchemaError
        validate(bad)


def test_split_is_disjoint_and_complete(portfolio, splits):
    train, valid, test = splits
    ids = [set(s["IDpol"]) for s in splits]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert len(train) + len(valid) + len(test) == len(portfolio)
    rates = [(s["ClaimNb"] > 0).mean() for s in splits]
    assert max(rates) - min(rates) < 0.01  # stratified


def test_quoted_categoricals_from_openml_are_cleaned():
    """OpenML returns categoricals such as 'Diesel' with literal quote characters."""
    freq, sev = make_synthetic(n=2_000)
    for col in ["Area", "VehBrand", "VehGas", "Region"]:
        freq[col] = "'" + freq[col].astype(str) + "'"
    df = build_dataset(freq, sev)
    assert set(df["VehGas"]) <= {"Diesel", "Regular"}
    assert not df["Area"].str.contains("'").any()
    validate(df)
