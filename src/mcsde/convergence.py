"""Empirical convergence analysis of the discretisation schemes.

The geometric Brownian motion is used as test case because its exact
solution is known pathwise (strong error) and its moments are known in
closed form (weak error).
"""
from __future__ import annotations

import numpy as np

from .processes import GeometricBrownianMotion
from .schemes import SCHEMES


def strong_error(gbm: GeometricBrownianMotion, T: float, n_steps: int, n_paths: int,
                 scheme: str, rng: np.random.Generator) -> float:
    """E|X_T - X_T^h|, comparing exact and numerical solution on the same Brownian path."""
    dW = rng.standard_normal((n_paths, n_steps)) * np.sqrt(T / n_steps)
    X = SCHEMES[scheme](gbm, T, dW)
    exact = gbm.exact(T, dW.sum(axis=1))
    return float(np.mean(np.abs(exact - X[:, -1])))


def strong_order(gbm: GeometricBrownianMotion, T: float = 1.0,
                 steps: tuple[int, ...] = (8, 16, 32, 64, 128, 256),
                 n_paths: int = 20_000, scheme: str = "euler",
                 seed: int = 0) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares slope of log(error) against log(dt) = empirical strong order."""
    rng = np.random.default_rng(seed)
    dts = np.array([T / n for n in steps])
    errs = np.array([strong_error(gbm, T, n, n_paths, scheme, rng) for n in steps])
    slope = np.polyfit(np.log(dts), np.log(errs), 1)[0]
    return float(slope), dts, errs


def weak_error_second_moment(gbm: GeometricBrownianMotion, T: float, n_steps: int) -> float:
    """|E[X_T^2] - E[(X_T^h)^2]| for the Euler scheme, computed exactly.

    For Euler applied to GBM, E[X_{n+1}^2] = E[X_n^2] ((1 + mu h)^2 + sigma^2 h),
    so the weak error can be evaluated without Monte Carlo noise.
    """
    h = T / n_steps
    euler = gbm.x0**2 * ((1 + gbm.mu * h) ** 2 + gbm.sigma**2 * h) ** n_steps
    exact = gbm.x0**2 * np.exp((2 * gbm.mu + gbm.sigma**2) * T)
    return float(abs(exact - euler))


def weak_order(gbm: GeometricBrownianMotion, T: float = 1.0,
               steps: tuple[int, ...] = (8, 16, 32, 64, 128, 256)) -> tuple[float, np.ndarray, np.ndarray]:
    """Empirical weak order of the Euler scheme (theory: 1)."""
    dts = np.array([T / n for n in steps])
    errs = np.array([weak_error_second_moment(gbm, T, n) for n in steps])
    slope = np.polyfit(np.log(dts), np.log(errs), 1)[0]
    return float(slope), dts, errs
