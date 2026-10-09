"""Validate risk measures, portfolio revaluation and the Kupiec backtest."""
import numpy as np
from scipy.stats import norm

from mcsde import (OptionPosition, Portfolio, delta_normal_var, expected_shortfall,
                   full_revaluation_losses, kupiec_test, rolling_var_backtest,
                   value_at_risk)


def test_var_and_es_for_standard_normal_losses():
    L = np.random.default_rng(0).standard_normal(1_000_000)
    a = 0.99
    assert abs(value_at_risk(L, a) - norm.ppf(a)) < 0.01
    assert abs(expected_shortfall(L, a) - norm.pdf(norm.ppf(a)) / (1 - a)) < 0.02
    assert expected_shortfall(L, a) >= value_at_risk(L, a)


def test_portfolio_delta_and_zero_loss_without_move():
    pf = Portfolio(shares=100, options=[OptionPosition(-100, 100, 0.5)])
    assert 0 < pf.delta(100.0) < 100
    loss = full_revaluation_losses(pf, 100.0, np.array([100.0]), h=0.0)
    assert abs(loss[0]) < 1e-9


def test_delta_normal_matches_full_revaluation_for_linear_portfolio():
    pf = Portfolio(shares=1000, options=[], sigma=0.2)
    h, S0 = 10 / 252, 100.0
    Z = np.random.default_rng(1).standard_normal(1_000_000)
    S_h = S0 * np.exp(0.2 * np.sqrt(h) * Z)
    full = value_at_risk(full_revaluation_losses(pf, S0, S_h, h))
    assert abs(full / delta_normal_var(pf, S0, 0.2, h) - 1) < 0.05


def test_kupiec_accepts_correct_and_rejects_wrong_model():
    rng = np.random.default_rng(2)
    good = rng.random(1000) < 0.01
    bad = rng.random(1000) < 0.04
    assert not kupiec_test(good, 0.99).reject()
    assert kupiec_test(bad, 0.99).reject()


def test_rolling_backtest_on_iid_normal_returns():
    r = np.random.default_rng(3).normal(0, 0.01, 2250)
    var, exc = rolling_var_backtest(r, window=250, method="normal")
    assert len(var) == len(exc) == 2000
    assert not kupiec_test(exc).reject(0.01)
