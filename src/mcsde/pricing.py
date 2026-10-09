"""Monte Carlo pricing of derivatives, variance reduction and Greeks."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MCResult:
    """Monte Carlo estimate with its standard error."""

    price: float
    std_error: float

    @property
    def ci95(self) -> tuple[float, float]:
        return self.price - 1.96 * self.std_error, self.price + 1.96 * self.std_error


def _estimate(samples: np.ndarray) -> MCResult:
    return MCResult(float(samples.mean()), float(samples.std(ddof=1) / np.sqrt(len(samples))))


# ----------------------------------------------------------------- payoffs
def payoff_european(paths: np.ndarray, K: float, kind: str = "call") -> np.ndarray:
    ST = paths[:, -1]
    return np.maximum(ST - K, 0.0) if kind == "call" else np.maximum(K - ST, 0.0)


def payoff_asian(paths: np.ndarray, K: float, kind: str = "call") -> np.ndarray:
    """Arithmetic average over all monitoring dates after t = 0."""
    A = paths[:, 1:].mean(axis=1)
    return np.maximum(A - K, 0.0) if kind == "call" else np.maximum(K - A, 0.0)


def payoff_up_and_out_call(paths: np.ndarray, K: float, barrier: float) -> np.ndarray:
    """Discretely monitored up-and-out call."""
    alive = paths.max(axis=1) < barrier
    return np.where(alive, np.maximum(paths[:, -1] - K, 0.0), 0.0)


# ----------------------------------------------------------------- pricing
def mc_price(paths: np.ndarray, payoff: np.ndarray, r: float, T: float,
             antithetic: bool = False) -> MCResult:
    """Discounted expected payoff under the risk-neutral measure.

    With antithetic=True each path is averaged with its mirrored path first,
    so that the standard error accounts for the pairing correctly.
    """
    disc = np.exp(-r * T) * payoff
    if antithetic:
        half = len(disc) // 2
        disc = 0.5 * (disc[:half] + disc[half:2 * half])
    return _estimate(disc)


def mc_price_control_variate(paths: np.ndarray, payoff: np.ndarray, r: float, T: float,
                             S0: float) -> MCResult:
    """Control variate e^{-rT} S_T, whose expectation S0 is known exactly."""
    Y = np.exp(-r * T) * payoff
    X = np.exp(-r * T) * paths[:, -1]
    c = np.cov(Y, X)
    beta = c[0, 1] / c[1, 1]
    return _estimate(Y - beta * (X - S0))


# ------------------------------------------------------------------ Greeks
def delta_pathwise(S0: float, K: float, T: float, r: float, ST: np.ndarray) -> MCResult:
    """Pathwise derivative estimator of the call delta under GBM:
    d/dS0 e^{-rT}(S_T - K)^+ = e^{-rT} 1{S_T > K} S_T / S0."""
    return _estimate(np.exp(-r * T) * (ST > K) * ST / S0)


def delta_likelihood_ratio(S0: float, K: float, T: float, r: float, sigma: float,
                           Z: np.ndarray, kind: str = "call") -> MCResult:
    """Likelihood ratio estimator of delta under GBM (works for discontinuous payoffs).

    Z are the standard normal draws generating S_T.
    """
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    pay = np.maximum(ST - K, 0.0) if kind == "call" else np.maximum(K - ST, 0.0)
    return _estimate(np.exp(-r * T) * pay * Z / (S0 * sigma * np.sqrt(T)))


def delta_finite_difference(price_fn, S0: float, h: float = 0.01) -> float:
    """Central finite difference with common random numbers.

    price_fn(S0) must reuse the same random numbers for every call.
    """
    return (price_fn(S0 * (1 + h)) - price_fn(S0 * (1 - h))) / (2 * S0 * h)
