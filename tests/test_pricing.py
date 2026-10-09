"""Validate Monte Carlo prices and Greeks against the analytical benchmarks."""
import numpy as np

from mcsde import (GeometricBrownianMotion, Heston, MertonJumpDiffusion,
                   black_scholes, bs_delta, delta_finite_difference,
                   delta_likelihood_ratio, delta_pathwise, heston_price,
                   mc_price, mc_price_control_variate, merton_price,
                   payoff_asian, payoff_european, payoff_up_and_out_call, simulate)

S0, K, T, R, SIG = 100.0, 100.0, 1.0, 0.03, 0.2


def _gbm_paths(n=200_000, antithetic=False, seed=10, steps=50):
    gbm = GeometricBrownianMotion(x0=S0, mu=R, sigma=SIG)
    return simulate(gbm, T, steps, n, "milstein", np.random.default_rng(seed), antithetic)


def _inside(result, reference, slack=0.0):
    lo, hi = result.ci95
    return lo - slack <= reference <= hi + slack


def test_mc_european_call_matches_black_scholes():
    p = _gbm_paths()
    assert _inside(mc_price(p, payoff_european(p, K), R, T), black_scholes(S0, K, T, R, SIG), 0.03)


def test_mc_heston_matches_fourier_price():
    S, _ = Heston(s0=S0, mu=R, xi=0.5).simulate(T, 200, 200_000, np.random.default_rng(11))
    ref = heston_price(S0, K, T, R, v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
    assert _inside(mc_price(S, payoff_european(S, K), R, T), ref, 0.05)


def test_mc_merton_matches_series_price():
    m = MertonJumpDiffusion(s0=S0, mu=R)
    P = m.simulate(T, 1, 400_000, np.random.default_rng(12))
    ref = merton_price(S0, K, T, R, m.sigma, m.lam, m.mu_j, m.delta)
    assert _inside(mc_price(P, payoff_european(P, K), R, T), ref, 0.01)


def test_variance_reduction_lowers_standard_error():
    plain, anti = _gbm_paths(100_000), _gbm_paths(100_000, antithetic=True)
    se_plain = mc_price(plain, payoff_european(plain, K), R, T).std_error
    se_anti = mc_price(anti, payoff_european(anti, K), R, T, antithetic=True).std_error
    se_cv = mc_price_control_variate(plain, payoff_european(plain, K), R, T, S0).std_error
    assert se_anti < se_plain
    assert se_cv < 0.5 * se_plain


def test_exotic_payoff_ordering():
    """Averaging and knock-out features make the option cheaper than the vanilla call."""
    p = _gbm_paths(100_000, steps=100)
    vanilla = mc_price(p, payoff_european(p, K), R, T).price
    asian = mc_price(p, payoff_asian(p, K), R, T).price
    barrier = mc_price(p, payoff_up_and_out_call(p, K, 130), R, T).price
    assert asian < vanilla and barrier < vanilla


def test_delta_estimators_agree_with_black_scholes():
    ref = bs_delta(S0, K, T, R, SIG)
    Z = np.random.default_rng(13).standard_normal(400_000)
    ST = S0 * np.exp((R - 0.5 * SIG**2) * T + SIG * np.sqrt(T) * Z)
    assert _inside(delta_pathwise(S0, K, T, R, ST), ref)
    assert _inside(delta_likelihood_ratio(S0, K, T, R, SIG, Z), ref)

    def price(s):  # common random numbers
        ST_s = s * np.exp((R - 0.5 * SIG**2) * T + SIG * np.sqrt(T) * Z)
        return np.exp(-R * T) * np.maximum(ST_s - K, 0).mean()

    assert abs(delta_finite_difference(price, S0) - ref) < 0.005
