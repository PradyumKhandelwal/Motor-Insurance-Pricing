"""Train the production frequency & severity models and save one artifact.

Usage:  python -m motor_pricing.train
(The notebooks do the same with tuned hyper-parameters; this is the CI-friendly path.)
"""

from __future__ import annotations

import argparse

import joblib

from .config import MODELS_DIR
from .data import load_data, split_data
from .evaluation import mean_deviance
from .features import learn_categories
from .models import FREQ_DEFAULTS, SEV_DEFAULTS, fit_lgbm, freq_xyw, sev_xyw
from .scoring import build_artifact


def train_artifact(train, valid, freq_params=None, sev_params=None) -> dict:
    categories = learn_categories(train)
    freq_params = freq_params or FREQ_DEFAULTS
    sev_params = sev_params or SEV_DEFAULTS

    f_tr, f_va = freq_xyw(train, categories), freq_xyw(valid, categories)
    freq_model = fit_lgbm(*f_tr, *f_va, freq_params)
    s_tr, s_va = sev_xyw(train, categories), sev_xyw(valid, categories)
    sev_model = fit_lgbm(*s_tr, *s_va, sev_params)

    metrics = {
        "valid_freq_poisson_deviance": mean_deviance(
            "poisson", f_va[1], freq_model.predict(f_va[0]), f_va[2]
        ),
        "valid_sev_gamma_deviance": mean_deviance(
            "gamma", s_va[1], sev_model.predict(s_va[0]), s_va[2]
        ),
    }
    return build_artifact(freq_model, sev_model, categories, metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(MODELS_DIR / "pricing_artifact.joblib"))
    args = parser.parse_args()

    df = load_data()
    train, valid, _ = split_data(df)
    artifact = train_artifact(train, valid)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.out)
    print(f"saved {args.out}\nvalidation metrics: {artifact['metadata']['metrics']}")


if __name__ == "__main__":
    main()
