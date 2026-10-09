"""Risk measures, portfolio revaluation and VaR backtesting."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2, norm

from .analytic import black_scholes, bs_delta


# ---------------------------------------------------------- risk measures
def value_at_risk(losses: np.ndarray, alpha: float = 0.99) -> float:
    """VaR_alpha = alpha-quantile of the loss distribution (losses are positive)."""
    return float(np.quantile(losses, alpha))


def expected_shortfall(losses: np.ndarray, alpha: float = 0.99) -> float:
    """ES_alpha = E[L | L >= VaR_alpha], a coherent risk measure."""
    var = value_at_risk(losses, alpha)
    return float(losses[losses >= var].mean())


# ------------------------------------------------------ option portfolio
@dataclass
class OptionPosition:
    """Position of `quantity` European options on one underlying (negative = short)."""

    quantity: float
    K: float
    T: float
    kind: str = "call"


@dataclass
class Portfolio:
    """Stock holding plus European options, valued with Black-Scholes."""

    shares: float
    options: list[OptionPosition]
    r: float = 0.03
    sigma: float = 0.2

    def value(self, S: np.ndarray, t: float = 0.0) -> np.ndarray:
        """Full revaluation of the portfolio at time t for spot prices S."""
        V = self.shares * np.asarray(S, dtype=float)
        for o in self.options:
            V = V + o.quantity * black_scholes(S, o.K, o.T - t, self.r, self.sigma, o.kind)
        return V

    def delta(self, S0: float) -> float:
        return self.shares + sum(o.quantity * bs_delta(S0, o.K, o.T, self.r, self.sigma, o.kind)
                                 for o in self.options)


def full_revaluation_losses(portfolio: Portfolio, S0: float, S_h: np.ndarray, h: float) -> np.ndarray:
    """Losses over horizon h when every scenario S_h is revalued exactly."""
    return portfolio.value(S0) - portfolio.value(S_h, t=h)


def delta_normal_var(portfolio: Portfolio, S0: float, sigma: float, h: float,
                     alpha: float = 0.99) -> float:
    """Linear (delta-normal) VaR approximation: |Delta| S0 sigma sqrt(h) z_alpha."""
    return float(abs(portfolio.delta(S0)) * S0 * sigma * np.sqrt(h) * norm.ppf(alpha))


# ------------------------------------------------------------ backtesting
@dataclass
class KupiecResult:
    n_obs: int
    n_exceptions: int
    expected: float
    lr_stat: float
    p_value: float

    def reject(self, level: float = 0.05) -> bool:
        return self.p_value < level


def kupiec_test(exceptions: np.ndarray, alpha: float = 0.99) -> KupiecResult:
    """Kupiec (1995) proportion-of-failures test for a VaR model.

    H0: the exception probability equals 1 - alpha.
    """
    exceptions = np.asarray(exceptions, dtype=bool)
    n, x = len(exceptions), int(exceptions.sum())
    p = 1 - alpha
    p_hat = x / n

    def loglik(q):
        q = min(max(q, 1e-12), 1 - 1e-12)
        return (n - x) * np.log(1 - q) + x * np.log(q)

    lr = -2 * (loglik(p) - loglik(p_hat))
    return KupiecResult(n, x, n * p, float(lr), float(1 - chi2.cdf(lr, df=1)))


def rolling_var_backtest(returns: np.ndarray, window: int = 250, alpha: float = 0.99,
                         method: str = "normal") -> tuple[np.ndarray, np.ndarray]:
    """One-day-ahead VaR forecasts from a rolling window, and the exception indicators.

    method = "normal": Gaussian VaR from the window's mean and standard deviation.
    method = "historical": empirical quantile of the window (historical simulation).
    """
    losses = -np.asarray(returns)
    var = np.empty(len(losses) - window)
    for t in range(window, len(losses)):
        w = losses[t - window:t]
        if method == "normal":
            var[t - window] = w.mean() + w.std(ddof=1) * norm.ppf(alpha)
        else:
            var[t - window] = np.quantile(w, alpha)
    exceptions = losses[window:] > var
    return var, exceptions
