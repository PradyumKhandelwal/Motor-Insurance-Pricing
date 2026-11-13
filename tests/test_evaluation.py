import numpy as np
import pytest

from motor_pricing.evaluation import (
    double_lift_table,
    gini,
    lift_table,
    mean_deviance,
    normalised_gini,
)


@pytest.fixture()
def toy():
    rng = np.random.default_rng(0)
    n = 20_000
    expo = rng.uniform(0.1, 1, n)
    true_rate = rng.gamma(2, 50, n)
    loss = rng.poisson(true_rate * expo).astype(float)
    return loss, expo, true_rate


def test_gini_rewards_good_ranking(toy):
    loss, expo, true_rate = toy
    random_rate = np.random.default_rng(1).random(len(loss))
    assert gini(loss, true_rate, expo) > gini(loss, random_rate, expo) + 0.2
    assert abs(gini(loss, random_rate, expo)) < 0.05
    assert 0 < normalised_gini(loss, true_rate, expo) <= 1.0


def test_deviance_zero_for_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0])
    for kind in ("poisson", "gamma", "tweedie"):
        assert mean_deviance(kind, y, y) == pytest.approx(0, abs=1e-9)
    assert mean_deviance("poisson", y, y * 2) > 0


def test_lift_table_is_monotone_for_true_rates(toy):
    loss, expo, true_rate = toy
    t = lift_table(loss, expo, true_rate, n_bins=10)
    assert len(t) == 10
    assert t["actual_rate"].is_monotonic_increasing
    assert t["exposure"].sum() == pytest.approx(expo.sum())


def test_double_lift_prefers_the_better_model(toy):
    loss, expo, true_rate = toy
    noisy = true_rate * np.random.default_rng(2).lognormal(0, 0.8, len(loss))
    t = double_lift_table(loss, expo, pred_a=noisy, pred_b=true_rate)
    top = t.iloc[-1]  # where challenger >> base
    assert abs(top["challenger_vs_actual"] - 1) < abs(top["base_vs_actual"] - 1)


def test_scorecard_ranks_true_model_first(splits):
    from motor_pricing.evaluation import null_rates, scorecard

    train, _, test = splits
    null = null_rates(train)
    flat = test[["Exposure"]].assign(
        frequency=null["frequency"], severity=null["severity"], pure_premium=null["pure_premium"]
    )
    card = scorecard(test, {"flat": flat}, null)
    assert set(card.index) == {"null (portfolio average)", "flat"}
    assert card.loc["flat", "freq_poisson_dev"] == pytest.approx(
        card.loc["null (portfolio average)", "freq_poisson_dev"]
    )
