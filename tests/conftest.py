import os

os.environ.setdefault("MOTOR_PRICING_SYNTHETIC", "1")

import pytest  # noqa: E402

from motor_pricing.data import build_dataset, make_synthetic, split_data  # noqa: E402


@pytest.fixture(scope="session")
def portfolio():
    return build_dataset(*make_synthetic(n=8_000, seed=7))


@pytest.fixture(scope="session")
def splits(portfolio):
    return split_data(portfolio)
