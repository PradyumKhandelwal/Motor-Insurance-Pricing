"""Package trained models into one artifact and score new policies with it."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from .models import prepare


def build_artifact(freq_model, sev_model, categories: dict, metrics: dict | None = None) -> dict:
    return {
        "freq": freq_model,
        "sev": sev_model,
        "categories": categories,
        "metadata": {
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "metrics": metrics or {},
        },
    }


def predict_components(artifact: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Annual claim frequency, average severity and pure premium for each policy."""
    x = prepare(df, artifact["categories"])
    freq = artifact["freq"].predict(x)
    sev = artifact["sev"].predict(x)
    return pd.DataFrame(
        {"frequency": freq, "severity": sev, "pure_premium": freq * sev}, index=df.index
    )
