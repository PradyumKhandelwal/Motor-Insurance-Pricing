import joblib
import pytest
from fastapi.testclient import TestClient

from motor_pricing.api import app
from motor_pricing.features import FEATURES, learn_categories, to_lgbm_frame
from motor_pricing.models import FREQ_DEFAULTS, SEV_DEFAULTS
from motor_pricing.scoring import predict_components
from motor_pricing.train import train_artifact

FAST = {"n_estimators": 40, "learning_rate": 0.1}


@pytest.fixture(scope="module")
def artifact(splits):
    train, valid, _ = splits
    return train_artifact(
        train, valid, {**FREQ_DEFAULTS, **FAST, "min_child_samples": 50}, {**SEV_DEFAULTS, **FAST}
    )


def test_feature_matrix_has_fixed_levels(splits):
    train, _, test = splits
    cats = learn_categories(train)
    x = to_lgbm_frame(test.assign(LogDensity=0.0), cats)
    assert list(x.columns) == FEATURES
    assert list(x["Area"].cat.categories) == cats["Area"]


def test_predictions_are_positive_and_sensible(artifact, splits):
    scored = predict_components(artifact, splits[2])
    assert (scored > 0).all().all()
    assert scored["frequency"].mean() == pytest.approx(
        splits[2]["ClaimNb"].sum() / splits[2]["Exposure"].sum(), rel=0.35
    )


def test_bonus_malus_monotonic_constraint(artifact, splits):
    base = splits[2].iloc[[0]].copy()
    prices = []
    for bm in (50, 80, 110, 140, 170):
        row = base.copy()
        row["BonusMalus"] = bm
        prices.append(predict_components(artifact, row)["frequency"].iloc[0])
    assert prices == sorted(prices)


def test_api_quote_roundtrip(artifact, tmp_path, monkeypatch):
    path = tmp_path / "artifact.joblib"
    joblib.dump(artifact, path)
    monkeypatch.setenv("MODEL_PATH", str(path))
    payload = {
        "VehPower": 6, "VehAge": 5, "DrivAge": 35, "BonusMalus": 60, "Density": 1500,
        "Area": "C", "VehBrand": "B12", "VehGas": "Regular", "Region": "R24",
    }  # fmt: skip
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
        r = client.post("/quote", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert body["pure_premium"] > 0
        assert body["pure_premium"] == pytest.approx(
            body["expected_claim_frequency"] * body["expected_claim_severity"], rel=0.02
        )
        assert client.post("/quote", json={**payload, "DrivAge": 12}).status_code == 422
        assert client.post("/quote", json={**payload, "Region": "R99"}).status_code == 422


def test_api_without_model_returns_503(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    with TestClient(app) as client:
        assert client.get("/health").json()["model_loaded"] is False
        payload = {
            "VehPower": 6, "VehAge": 5, "DrivAge": 35, "BonusMalus": 60, "Density": 1500,
            "Area": "C", "VehBrand": "B12", "VehGas": "Regular", "Region": "R24",
        }  # fmt: skip
        assert client.post("/quote", json=payload).status_code == 503
