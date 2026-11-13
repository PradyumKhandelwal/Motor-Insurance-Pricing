"""Insurance-style model evaluation: deviance, Gini / Lorenz, lift and calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_gamma_deviance, mean_poisson_deviance, mean_tweedie_deviance

TWEEDIE_POWER = 1.5


def mean_deviance(kind, y_true, y_pred, weight=None, power: float = TWEEDIE_POWER) -> float:
    """Exposure/claim-weighted mean deviance. ``kind``: poisson | gamma | tweedie."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-9, None)
    w = None if weight is None else np.asarray(weight, dtype=float)
    if kind == "poisson":
        return float(mean_poisson_deviance(y_true, y_pred, sample_weight=w))
    if kind == "gamma":
        return float(mean_gamma_deviance(y_true, y_pred, sample_weight=w))
    if kind == "tweedie":
        return float(mean_tweedie_deviance(y_true, y_pred, sample_weight=w, power=power))
    raise ValueError(f"unknown deviance kind: {kind}")


def deviance_skill(kind, y_true, y_pred, weight, null_value: float) -> float:
    """Share of the null-model deviance explained (a.k.a. D^2): 1 - dev / dev_null."""
    null = np.full(len(np.asarray(y_true)), null_value)
    return 1.0 - mean_deviance(kind, y_true, y_pred, weight) / mean_deviance(
        kind, y_true, null, weight
    )


# --------------------------------------------------------------------------- #
# Ranking power
# --------------------------------------------------------------------------- #
def lorenz_curve(loss, pred_rate, exposure) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative exposure vs cumulative loss, ordered by increasing predicted rate."""
    loss, pred_rate, exposure = (np.asarray(a, dtype=float) for a in (loss, pred_rate, exposure))
    order = np.argsort(pred_rate, kind="stable")
    cum_loss = np.concatenate([[0.0], np.cumsum(loss[order])])
    cum_expo = np.concatenate([[0.0], np.cumsum(exposure[order])])
    return cum_expo / cum_expo[-1], cum_loss / cum_loss[-1]


def gini(loss, pred_rate, exposure) -> float:
    """Exposure-weighted Gini index: 1 - 2 * area under the Lorenz curve."""
    x, y = lorenz_curve(loss, pred_rate, exposure)
    area = np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2)
    return float(1.0 - 2.0 * area)


def normalised_gini(loss, pred_rate, exposure) -> float:
    """Gini divided by the best achievable Gini (ranking by the realised loss itself)."""
    loss, exposure = np.asarray(loss, dtype=float), np.asarray(exposure, dtype=float)
    return gini(loss, pred_rate, exposure) / gini(loss, loss / exposure, exposure)


# --------------------------------------------------------------------------- #
# Lift & calibration tables
# --------------------------------------------------------------------------- #
def _exposure_bins(sort_key, exposure, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Order policies by ``sort_key`` and cut into bins of equal exposure."""
    order = np.argsort(sort_key, kind="stable")
    expo = np.asarray(exposure, dtype=float)[order]
    mid = (np.cumsum(expo) - expo / 2) / expo.sum()
    bins = np.minimum((mid * n_bins).astype(int), n_bins - 1)
    return order, bins


def lift_table(loss, exposure, pred_rate, n_bins: int = 10) -> pd.DataFrame:
    """Actual vs predicted loss rate in equal-exposure bins of predicted rate."""
    loss, exposure, pred_rate = (np.asarray(a, dtype=float) for a in (loss, exposure, pred_rate))
    order, bins = _exposure_bins(pred_rate, exposure, n_bins)
    d = pd.DataFrame(
        {
            "bin": bins,
            "exposure": exposure[order],
            "loss": loss[order],
            "predicted": (pred_rate * exposure)[order],
        }
    )
    g = d.groupby("bin").sum()
    return pd.DataFrame(
        {
            "exposure": g["exposure"],
            "actual_rate": g["loss"] / g["exposure"],
            "predicted_rate": g["predicted"] / g["exposure"],
        }
    )


def double_lift_table(loss, exposure, pred_a, pred_b, n_bins: int = 10) -> pd.DataFrame:
    """Double-lift chart data: bins sorted by ratio pred_b / pred_a (challenger / base)."""
    loss, exposure, pred_a, pred_b = (
        np.asarray(a, dtype=float) for a in (loss, exposure, pred_a, pred_b)
    )
    order, bins = _exposure_bins(pred_b / pred_a, exposure, n_bins)
    d = pd.DataFrame(
        {
            "bin": bins,
            "exposure": exposure[order],
            "loss": loss[order],
            "a": (pred_a * exposure)[order],
            "b": (pred_b * exposure)[order],
        }
    )
    g = d.groupby("bin").sum()
    out = pd.DataFrame(
        {
            "actual": g["loss"] / g["exposure"],
            "base_model": g["a"] / g["exposure"],
            "challenger": g["b"] / g["exposure"],
        }
    )
    out["base_vs_actual"] = out["base_model"] / out["actual"]
    out["challenger_vs_actual"] = out["challenger"] / out["actual"]
    return out


def segment_calibration(segment, loss, exposure, predictions: dict) -> pd.DataFrame:
    """Actual vs predicted loss rate per segment (e.g. driver-age band)."""
    exposure = np.asarray(exposure, dtype=float)
    d = pd.DataFrame(
        {"segment": list(segment), "loss": np.asarray(loss, dtype=float), "expo": exposure}
    )
    for name, pred in predictions.items():
        d[name] = np.asarray(pred, dtype=float) * exposure
    g = d.groupby("segment", observed=True, sort=False).sum()
    out = pd.DataFrame({"exposure_share": g["expo"] / g["expo"].sum(), "actual": g["loss"] / g["expo"]})
    for name in predictions:
        out[name] = g[name] / g["expo"]
    return out


# --------------------------------------------------------------------------- #
# One-call comparison of several models
# --------------------------------------------------------------------------- #
def null_rates(train: pd.DataFrame) -> dict[str, float]:
    """Portfolio-average frequency / severity / pure premium (the 'no model' benchmark)."""
    claims = train["ClaimNb"] > 0
    return {
        "frequency": train["ClaimNb"].sum() / train["Exposure"].sum(),
        "severity": train.loc[claims, "ClaimAmount"].sum() / train.loc[claims, "ClaimNb"].sum(),
        "pure_premium": train["ClaimAmount"].sum() / train["Exposure"].sum(),
    }


def scorecard(test: pd.DataFrame, preds: dict[str, pd.DataFrame], null: dict[str, float]):
    """Deviance (lower is better) and Gini (higher is better) for each model on the test set.

    ``preds[name]`` is a DataFrame aligned row-by-row with ``test`` holding any of the
    columns ``frequency``, ``severity``, ``pure_premium`` (annual rates per policy).
    """
    expo = test["Exposure"].to_numpy()
    claims = (test["ClaimNb"] > 0).to_numpy()
    y_freq = (test["ClaimNb"] / test["Exposure"]).to_numpy()
    y_sev = (test["ClaimAmount"] / test["ClaimNb"].replace(0, np.nan)).to_numpy()[claims]
    w_sev = test["ClaimNb"].to_numpy()[claims]
    y_pp = (test["ClaimAmount"] / test["Exposure"]).to_numpy()
    loss = test["ClaimAmount"].to_numpy()

    null_df = pd.DataFrame({k: np.full(len(test), v) for k, v in null.items()})
    rows = {}
    for name, p in {"null (portfolio average)": null_df, **preds}.items():
        row = {}
        if "frequency" in p:
            row["freq_poisson_dev"] = mean_deviance("poisson", y_freq, p["frequency"], expo)
        if "severity" in p:
            row["sev_gamma_dev"] = mean_deviance(
                "gamma", y_sev, np.asarray(p["severity"])[claims], w_sev
            )
        row["pp_tweedie_dev"] = mean_deviance("tweedie", y_pp, p["pure_premium"], expo)
        row["gini"] = gini(loss, p["pure_premium"], expo)
        row["gini_vs_best_possible"] = normalised_gini(loss, p["pure_premium"], expo)
        rows[name] = row
    return pd.DataFrame(rows).T
