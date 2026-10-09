"""Validate the simulated processes against their exact moments."""
import numpy as np

from mcsde import (CIR, GeometricBrownianMotion, Heston, MertonJumpDiffusion,
                   OrnsteinUhlenbeck, simulate)

N = 100_000


def test_gbm_mean_and_variance():
    gbm = GeometricBrownianMotion(x0=100, mu=0.05, sigma=0.2)
    ST = simulate(gbm, 1.0, 100, N, rng=np.random.default_rng(1))[:, -1]
    assert abs(ST.mean() - gbm.mean(1.0)) < 4 * np.sqrt(gbm.variance(1.0) / N)
    assert abs(ST.var() / gbm.variance(1.0) - 1) < 0.05


def test_ornstein_uhlenbeck_moments():
    ou = OrnsteinUhlenbeck(x0=0.05, kappa=1.0, theta=0.03, sigma=0.01)
    rT = simulate(ou, 2.0, 200, N, rng=np.random.default_rng(2))[:, -1]
    assert abs(rT.mean() - ou.mean(2.0)) < 4 * np.sqrt(ou.variance(2.0) / N)
    assert abs(rT.var() / ou.variance(2.0) - 1) < 0.05


def test_cir_moments_and_feller():
    cir = CIR(x0=0.03, kappa=1.5, theta=0.04, sigma=0.1)
    assert cir.feller_condition()
    rT = simulate(cir, 1.0, 200, N, rng=np.random.default_rng(3))[:, -1]
    assert abs(rT.mean() - cir.mean(1.0)) < 4 * np.sqrt(cir.variance(1.0) / N)
    assert abs(rT.var() / cir.variance(1.0) - 1) < 0.05


def test_heston_discounted_price_is_martingale():
    h = Heston(s0=100, mu=0.03)
    S, v = h.simulate(1.0, 100, N, rng=np.random.default_rng(4))
    assert S.shape == v.shape == (N, 101)
    assert abs(S[:, -1].mean() - 100 * np.exp(0.03)) < 4 * S[:, -1].std() / np.sqrt(N)


def test_merton_mean_is_compensated():
    m = MertonJumpDiffusion(s0=100, mu=0.03, lam=1.0, mu_j=-0.1, delta=0.2)
    ST = m.simulate(1.0, 50, N, rng=np.random.default_rng(5))[:, -1]
    assert abs(ST.mean() - m.mean(1.0)) < 4 * ST.std() / np.sqrt(N)
