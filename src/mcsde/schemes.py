"""Time discretisation schemes for one-dimensional SDEs."""
from __future__ import annotations

import numpy as np

from .processes import SDE


def brownian_increments(T: float, n_steps: int, n_paths: int,
                        rng: np.random.Generator | None = None,
                        antithetic: bool = False) -> np.ndarray:
    """Brownian increments dW with shape (n_paths, n_steps).

    With antithetic=True the second half of the paths uses -dW of the first
    half (antithetic variates, a variance reduction technique).
    """
    rng = rng or np.random.default_rng()
    dt = T / n_steps
    if antithetic:
        half = (n_paths + 1) // 2
        dW = rng.standard_normal((half, n_steps)) * np.sqrt(dt)
        return np.vstack([dW, -dW])[:n_paths]
    return rng.standard_normal((n_paths, n_steps)) * np.sqrt(dt)


def euler_maruyama(sde: SDE, T: float, dW: np.ndarray) -> np.ndarray:
    """X_{n+1} = X_n + mu dt + sigma dW.

    Strong order 1/2, weak order 1 (under the usual smoothness assumptions).
    """
    n_paths, n_steps = dW.shape
    dt = T / n_steps
    X = np.empty((n_paths, n_steps + 1))
    X[:, 0] = sde.x0
    for i in range(n_steps):
        t, x = i * dt, X[:, i]
        X[:, i + 1] = x + sde.drift(t, x) * dt + sde.diffusion(t, x) * dW[:, i]
    return X


def milstein(sde: SDE, T: float, dW: np.ndarray) -> np.ndarray:
    """Euler-Maruyama plus the correction 1/2 sigma sigma' (dW^2 - dt).

    Strong order 1, weak order 1.
    """
    n_paths, n_steps = dW.shape
    dt = T / n_steps
    X = np.empty((n_paths, n_steps + 1))
    X[:, 0] = sde.x0
    for i in range(n_steps):
        t, x = i * dt, X[:, i]
        sig = sde.diffusion(t, x)
        X[:, i + 1] = (x + sde.drift(t, x) * dt + sig * dW[:, i]
                       + 0.5 * sig * sde.diffusion_dx(t, x) * (dW[:, i] ** 2 - dt))
    return X


SCHEMES = {"euler": euler_maruyama, "milstein": milstein}


def simulate(sde: SDE, T: float, n_steps: int, n_paths: int, scheme: str = "euler",
             rng: np.random.Generator | None = None, antithetic: bool = False) -> np.ndarray:
    """Simulate paths of an SDE with the chosen scheme ("euler" or "milstein")."""
    dW = brownian_increments(T, n_steps, n_paths, rng, antithetic)
    return SCHEMES[scheme](sde, T, dW)
