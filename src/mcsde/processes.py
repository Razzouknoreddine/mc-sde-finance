"""Stochastic processes defined by SDEs.

One-dimensional diffusions are written as

    dX_t = mu(t, X_t) dt + sigma(t, X_t) dW_t

and expose their drift, diffusion and the derivative of the diffusion with
respect to x (needed by the Milstein scheme). Where closed-form moments are
known they are provided, so the simulation can be validated in the tests.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SDE:
    """Base class for one-dimensional SDEs."""

    x0: float

    def drift(self, t: float, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def diffusion(self, t: float, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def diffusion_dx(self, t: float, x: np.ndarray) -> np.ndarray:
        """Derivative d sigma / dx, required by the Milstein scheme."""
        raise NotImplementedError


@dataclass
class GeometricBrownianMotion(SDE):
    """dS = mu S dt + sigma S dW   (Black-Scholes model)."""

    mu: float = 0.05
    sigma: float = 0.2

    def drift(self, t, x):
        return self.mu * x

    def diffusion(self, t, x):
        return self.sigma * x

    def diffusion_dx(self, t, x):
        return self.sigma * np.ones_like(x)

    def exact(self, T: float, W_T: np.ndarray) -> np.ndarray:
        """Exact solution S_T for a given Brownian terminal value W_T."""
        return self.x0 * np.exp((self.mu - 0.5 * self.sigma**2) * T + self.sigma * W_T)

    def mean(self, T: float) -> float:
        return self.x0 * np.exp(self.mu * T)

    def variance(self, T: float) -> float:
        return self.x0**2 * np.exp(2 * self.mu * T) * (np.exp(self.sigma**2 * T) - 1)


@dataclass
class OrnsteinUhlenbeck(SDE):
    """dr = kappa (theta - r) dt + sigma dW   (Vasicek short-rate model)."""

    kappa: float = 1.0
    theta: float = 0.03
    sigma: float = 0.01

    def drift(self, t, x):
        return self.kappa * (self.theta - x)

    def diffusion(self, t, x):
        return self.sigma * np.ones_like(x)

    def diffusion_dx(self, t, x):
        return np.zeros_like(x)

    def mean(self, T: float) -> float:
        return self.theta + (self.x0 - self.theta) * np.exp(-self.kappa * T)

    def variance(self, T: float) -> float:
        return self.sigma**2 / (2 * self.kappa) * (1 - np.exp(-2 * self.kappa * T))


@dataclass
class CIR(SDE):
    """dr = kappa (theta - r) dt + sigma sqrt(r) dW   (Cox-Ingersoll-Ross).

    The discretisation evaluates the coefficients at max(r, 0)
    ("full truncation"), which keeps the scheme well defined.
    """

    kappa: float = 1.5
    theta: float = 0.04
    sigma: float = 0.1

    def drift(self, t, x):
        return self.kappa * (self.theta - np.maximum(x, 0.0))

    def diffusion(self, t, x):
        return self.sigma * np.sqrt(np.maximum(x, 0.0))

    def diffusion_dx(self, t, x):
        return 0.5 * self.sigma / np.sqrt(np.maximum(x, 1e-12))

    def feller_condition(self) -> bool:
        """2 kappa theta >= sigma^2 implies the process stays strictly positive."""
        return 2 * self.kappa * self.theta >= self.sigma**2

    def mean(self, T: float) -> float:
        return self.theta + (self.x0 - self.theta) * np.exp(-self.kappa * T)

    def variance(self, T: float) -> float:
        e = np.exp(-self.kappa * T)
        return (self.x0 * self.sigma**2 / self.kappa * (e - e**2)
                + self.theta * self.sigma**2 / (2 * self.kappa) * (1 - e) ** 2)


@dataclass
class Heston:
    """Heston stochastic volatility model:

        dS = mu S dt + sqrt(v) S dW1
        dv = kappa (theta - v) dt + xi sqrt(v) dW2,     d<W1, W2>_t = rho dt
    """

    s0: float = 100.0
    v0: float = 0.04
    mu: float = 0.05
    kappa: float = 2.0
    theta: float = 0.04
    xi: float = 0.3
    rho: float = -0.7

    def simulate(self, T: float, n_steps: int, n_paths: int,
                 rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Full-truncation Euler for v and log-Euler for S.

        Returns (S, v), each with shape (n_paths, n_steps + 1).
        """
        rng = rng or np.random.default_rng()
        dt = T / n_steps
        S = np.empty((n_paths, n_steps + 1))
        v = np.empty((n_paths, n_steps + 1))
        S[:, 0], v[:, 0] = self.s0, self.v0
        for i in range(n_steps):
            z1 = rng.standard_normal(n_paths)
            z2 = self.rho * z1 + np.sqrt(1 - self.rho**2) * rng.standard_normal(n_paths)
            vp = np.maximum(v[:, i], 0.0)
            S[:, i + 1] = S[:, i] * np.exp((self.mu - 0.5 * vp) * dt + np.sqrt(vp * dt) * z1)
            v[:, i + 1] = v[:, i] + self.kappa * (self.theta - vp) * dt + self.xi * np.sqrt(vp * dt) * z2
        return S, v


@dataclass
class MertonJumpDiffusion:
    """Merton jump-diffusion model:

        dS / S_- = (mu - lam * k) dt + sigma dW + (J - 1) dN

    with N a Poisson process of intensity lam, log J ~ N(mu_j, delta^2)
    and k = E[J - 1] = exp(mu_j + delta^2 / 2) - 1, so that E[S_T] = S0 e^{mu T}.
    """

    s0: float = 100.0
    mu: float = 0.05
    sigma: float = 0.15
    lam: float = 0.5
    mu_j: float = -0.1
    delta: float = 0.15

    @property
    def k(self) -> float:
        return np.exp(self.mu_j + 0.5 * self.delta**2) - 1

    def simulate(self, T: float, n_steps: int, n_paths: int,
                 rng: np.random.Generator | None = None) -> np.ndarray:
        """Exact simulation of log S on the time grid. Shape (n_paths, n_steps + 1)."""
        rng = rng or np.random.default_rng()
        dt = T / n_steps
        drift = (self.mu - self.lam * self.k - 0.5 * self.sigma**2) * dt
        N = rng.poisson(self.lam * dt, size=(n_paths, n_steps))
        jumps = self.mu_j * N + self.delta * np.sqrt(N) * rng.standard_normal((n_paths, n_steps))
        dlog = drift + self.sigma * np.sqrt(dt) * rng.standard_normal((n_paths, n_steps)) + jumps
        logS = np.log(self.s0) + np.concatenate([np.zeros((n_paths, 1)), np.cumsum(dlog, axis=1)], axis=1)
        return np.exp(logS)

    def mean(self, T: float) -> float:
        return self.s0 * np.exp(self.mu * T)
