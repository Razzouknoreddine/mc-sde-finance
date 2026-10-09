"""Closed-form and semi-analytical benchmark prices.

These reference values are used to validate the Monte Carlo estimators:
- Black-Scholes price and Greeks,
- Merton (1976) jump-diffusion price as a Poisson-weighted series,
- Heston (1993) price via Fourier inversion of the characteristic function.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm, poisson


def _d1_d2(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)


def black_scholes(S0, K, T, r, sigma, kind: str = "call"):
    """Black-Scholes price of a European call or put."""
    d1, d2 = _d1_d2(S0, K, T, r, sigma)
    if kind == "call":
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)


def bs_delta(S0, K, T, r, sigma, kind: str = "call"):
    d1, _ = _d1_d2(S0, K, T, r, sigma)
    return norm.cdf(d1) if kind == "call" else norm.cdf(d1) - 1


def bs_gamma(S0, K, T, r, sigma):
    d1, _ = _d1_d2(S0, K, T, r, sigma)
    return norm.pdf(d1) / (S0 * sigma * np.sqrt(T))


def bs_vega(S0, K, T, r, sigma):
    d1, _ = _d1_d2(S0, K, T, r, sigma)
    return S0 * norm.pdf(d1) * np.sqrt(T)


def merton_price(S0, K, T, r, sigma, lam, mu_j, delta, kind: str = "call",
                 n_terms: int = 60) -> float:
    """Merton jump-diffusion price: Black-Scholes prices weighted by Poisson probabilities."""
    k = np.exp(mu_j + 0.5 * delta**2) - 1
    lam_p = lam * (1 + k)
    price = 0.0
    for n in range(n_terms):
        sigma_n = np.sqrt(sigma**2 + n * delta**2 / T)
        r_n = r - lam * k + n * np.log(1 + k) / T
        weight = poisson.pmf(n, lam_p * T)
        price += weight * black_scholes(S0, K, T, r_n, sigma_n, kind)
    return float(price)


def heston_charfn(u, T, S0, r, v0, kappa, theta, xi, rho):
    """Characteristic function E[exp(i u log S_T)] under the risk-neutral measure.

    Uses the "little trap" formulation (Albrecher et al., 2007), which is
    numerically stable for long maturities.
    """
    iu = 1j * u
    b = kappa - rho * xi * iu
    d = np.sqrt(b**2 + xi**2 * (iu + u**2))
    g = (b - d) / (b + d)
    e = np.exp(-d * T)
    C = r * iu * T + kappa * theta / xi**2 * ((b - d) * T - 2 * np.log((1 - g * e) / (1 - g)))
    D = (b - d) / xi**2 * (1 - e) / (1 - g * e)
    return np.exp(C + D * v0 + iu * np.log(S0))


def heston_price(S0, K, T, r, v0, kappa, theta, xi, rho, kind: str = "call") -> float:
    """Heston price of a European option via Fourier inversion (Gil-Pelaez)."""
    args = (T, S0, r, v0, kappa, theta, xi, rho)
    lnK = np.log(K)
    phi_mi = heston_charfn(-1j, *args)  # = S0 e^{rT}

    def p1_integrand(u):
        return (np.exp(-1j * u * lnK) * heston_charfn(u - 1j, *args) / (1j * u * phi_mi)).real

    def p2_integrand(u):
        return (np.exp(-1j * u * lnK) * heston_charfn(u, *args) / (1j * u)).real

    P1 = 0.5 + quad(p1_integrand, 1e-8, 200, limit=500)[0] / np.pi
    P2 = 0.5 + quad(p2_integrand, 1e-8, 200, limit=500)[0] / np.pi
    call = S0 * P1 - K * np.exp(-r * T) * P2
    if kind == "call":
        return float(call)
    return float(call - S0 + K * np.exp(-r * T))
