"""Consistency checks for the closed-form and semi-analytical prices."""
import numpy as np

from mcsde import (black_scholes, bs_delta, bs_gamma, bs_vega, heston_price,
                   merton_price)

S0, K, T, R, SIG = 100.0, 100.0, 1.0, 0.03, 0.2
HESTON = dict(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)


def test_black_scholes_put_call_parity():
    c = black_scholes(S0, K, T, R, SIG, "call")
    p = black_scholes(S0, K, T, R, SIG, "put")
    assert abs(c - p - (S0 - K * np.exp(-R * T))) < 1e-12


def test_black_scholes_greeks_match_finite_differences():
    h = 1e-4
    fd_delta = (black_scholes(S0 + h, K, T, R, SIG) - black_scholes(S0 - h, K, T, R, SIG)) / (2 * h)
    fd_gamma = (bs_delta(S0 + h, K, T, R, SIG) - bs_delta(S0 - h, K, T, R, SIG)) / (2 * h)
    fd_vega = (black_scholes(S0, K, T, R, SIG + h) - black_scholes(S0, K, T, R, SIG - h)) / (2 * h)
    assert abs(fd_delta - bs_delta(S0, K, T, R, SIG)) < 1e-6
    assert abs(fd_gamma - bs_gamma(S0, K, T, R, SIG)) < 1e-6
    assert abs(fd_vega - bs_vega(S0, K, T, R, SIG)) < 1e-5


def test_merton_reduces_to_black_scholes_without_jumps():
    assert abs(merton_price(S0, K, T, R, SIG, 0.0, -0.1, 0.15) - black_scholes(S0, K, T, R, SIG)) < 1e-10


def test_heston_reduces_to_black_scholes_for_vanishing_vol_of_vol():
    p = heston_price(S0, K, T, R, v0=0.04, kappa=2.0, theta=0.04, xi=1e-4, rho=0.0)
    assert abs(p - black_scholes(S0, K, T, R, 0.2)) < 1e-3


def test_heston_put_call_parity():
    c = heston_price(S0, 90, T, R, **HESTON, kind="call")
    p = heston_price(S0, 90, T, R, **HESTON, kind="put")
    assert abs(c - p - (S0 - 90 * np.exp(-R * T))) < 1e-8


def test_heston_negative_correlation_creates_skew():
    """With rho < 0, deep out-of-the-money puts are more expensive than under Black-Scholes."""
    put = heston_price(S0, 80, T, R, **HESTON, kind="put")
    assert put > black_scholes(S0, 80, T, R, 0.2, "put")
