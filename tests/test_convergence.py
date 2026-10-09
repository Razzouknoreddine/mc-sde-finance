"""Check the empirical convergence orders against theory."""
from mcsde import GeometricBrownianMotion, strong_order, weak_order

GBM = GeometricBrownianMotion(x0=1.0, mu=0.05, sigma=0.4)


def test_euler_strong_order_one_half():
    order, *_ = strong_order(GBM, scheme="euler", n_paths=10_000)
    assert 0.4 < order < 0.65


def test_milstein_strong_order_one():
    order, *_ = strong_order(GBM, scheme="milstein", n_paths=10_000)
    assert 0.85 < order < 1.15


def test_euler_weak_order_one():
    order, *_ = weak_order(GBM)
    assert 0.95 < order < 1.05
