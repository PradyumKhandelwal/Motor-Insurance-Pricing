"""FastAPI service returning a technical (pure-premium) price for one policy.

Run:  uvicorn motor_pricing.api:app --reload
Docs: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from .config import MODELS_DIR
from .scoring import predict_components


class Policy(BaseModel):
    VehPower: int = Field(..., ge=4, le=15, examples=[6], description="Vehicle power class")
    VehAge: int = Field(..., ge=0, le=100, examples=[5], description="Vehicle age in years")
    DrivAge: int = Field(..., ge=18, le=100, examples=[35], description="Driver age in years")
    BonusMalus: int = Field(..., ge=50, le=230, examples=[60], description="Bonus-malus (100 = no discount)")
    Density: float = Field(..., gt=0, examples=[1500], description="Inhabitants per km2 of the city")
    Area: Literal["A", "B", "C", "D", "E", "F"] = Field(..., examples=["C"])
    VehBrand: str = Field(..., examples=["B12"])
    VehGas: Literal["Diesel", "Regular"] = Field(..., examples=["Regular"])
    Region: str = Field(..., examples=["R24"])


class Quote(BaseModel):
    expected_claim_frequency: float
    expected_claim_severity: float
    pure_premium: float
    trained_at: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = Path(os.environ.get("MODEL_PATH", MODELS_DIR / "pricing_artifact.joblib"))
    app.state.artifact = joblib.load(path) if path.exists() else None
    yield


app = FastAPI(title="Motor pricing API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health(request: Request) -> dict:
    return {"status": "ok", "model_loaded": request.app.state.artifact is not None}


@app.post("/quote", response_model=Quote)
def quote(policy: Policy, request: Request) -> Quote:
    artifact = request.app.state.artifact
    if artifact is None:
        raise HTTPException(503, "No model loaded - run `python -m motor_pricing.train` first.")
    for field in ("VehBrand", "Region"):
        if getattr(policy, field) not in artifact["categories"][field]:
            allowed = artifact["categories"][field]
            raise HTTPException(422, f"Unknown {field}={getattr(policy, field)!r}; use one of {allowed}")

    scored = predict_components(artifact, pd.DataFrame([policy.model_dump()])).iloc[0]
    return Quote(
        expected_claim_frequency=round(float(scored["frequency"]), 5),
        expected_claim_severity=round(float(scored["severity"]), 2),
        pure_premium=round(float(scored["pure_premium"]), 2),
        trained_at=artifact["metadata"]["trained_at"],
    )
