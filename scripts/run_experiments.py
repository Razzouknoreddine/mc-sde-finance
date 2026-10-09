"""Reproduce all results of the project.

Writes the figures to docs/figures/ and a results table to docs/results.md.
Usage:  python scripts/run_experiments.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mcsde import (CIR, GeometricBrownianMotion, Heston, MertonJumpDiffusion,
                   OptionPosition, OrnsteinUhlenbeck, Portfolio, black_scholes,
                   delta_normal_var, expected_shortfall, full_revaluation_losses,
                   heston_price, kupiec_test, mc_price, mc_price_control_variate,
                   merton_price, payoff_european, rolling_var_backtest, simulate,
                   strong_order, value_at_risk, weak_order)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(2026)

# Categorical palette in fixed order (blue, orange, aqua, yellow, magenta)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, MUTED = "#1f1f1e", "#6b6b66"
plt.rcParams.update({
    "figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": "#e6e5df",
    "grid.linewidth": 0.6, "lines.linewidth": 2, "legend.frameon": False,
    "axes.titleweight": "bold", "axes.titlesize": 11,
})
S0, K, T, R = 100.0, 100.0, 1.0, 0.03
HP = dict(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
results = []


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------ 1. sample paths
def figure_paths():
    t = np.linspace(0, 1, 253)
    data = [
        ("GBM", simulate(GeometricBrownianMotion(100, 0.05, 0.2), 1, 252, 8, rng=rng)),
        ("Heston", Heston(mu=0.05, xi=0.5).simulate(1, 252, 8, rng)[0]),
        ("Merton jump diffusion", MertonJumpDiffusion(mu=0.05, lam=3).simulate(1, 252, 8, rng)),
        ("Vasicek short rate", simulate(OrnsteinUhlenbeck(0.05, 2.0, 0.03, 0.01), 1, 252, 8, rng=rng)),
        ("CIR short rate", simulate(CIR(0.01, 1.5, 0.04, 0.1), 1, 252, 8, rng=rng)),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.2))
    for ax, (title, X), c in zip(axes, data, C):
        ax.plot(t, X.T, color=c, lw=0.9, alpha=0.75)
        ax.set_title(title)
        ax.set_xlabel("t (years)")
    save(fig, "paths.png")


# --------------------------------------------------------- 2. convergence
def figure_convergence():
    gbm = GeometricBrownianMotion(1.0, 0.05, 0.4)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    orders = {}
    for scheme, c in (("euler", C[0]), ("milstein", C[1])):
        order, dts, errs = strong_order(gbm, scheme=scheme)
        orders[f"strong_{scheme}"] = order
        axes[0].loglog(dts, errs, "o-", color=c, ms=6, label=f"{scheme.title()} (slope {order:.2f})")
    axes[0].loglog(dts, 0.12 * dts**0.5, ":", color=MUTED, lw=1.2, label="reference dt^0.5")
    axes[0].loglog(dts, 0.06 * dts, "--", color=MUTED, lw=1.2, label="reference dt^1")
    axes[0].set(title="Strong error  E|X_T - X_T^h|", xlabel="step size dt")
    axes[0].legend(fontsize=8)

    order, dts, errs = weak_order(gbm)
    orders["weak_euler"] = order
    axes[1].loglog(dts, errs, "o-", color=C[0], ms=6, label=f"Euler (slope {order:.2f})")
    axes[1].loglog(dts, errs[0] / dts[0] * dts, "--", color=MUTED, lw=1.2, label="reference dt^1")
    axes[1].set(title="Weak error  |E[X_T^2] - E[(X_T^h)^2]|", xlabel="step size dt")
    axes[1].legend(fontsize=8)
    save(fig, "convergence.png")
    return orders


# ------------------------------------------- 3. Monte Carlo vs benchmarks
def figure_pricing():
    m = MertonJumpDiffusion(s0=S0, mu=R)
    refs = {
        "Black-Scholes": black_scholes(S0, K, T, R, 0.2),
        "Heston": heston_price(S0, K, T, R, **HP),
        "Merton": merton_price(S0, K, T, R, m.sigma, m.lam, m.mu_j, m.delta),
    }
    ns = np.unique(np.logspace(3, 5.3, 9).astype(int))
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    final = {}
    for ax, (name, ref), c in zip(axes, refs.items(), C):
        est, se = [], []
        for n in ns:
            if name == "Black-Scholes":
                P = simulate(GeometricBrownianMotion(S0, R, 0.2), T, 50, n, "milstein", rng)
            elif name == "Heston":
                P, _ = Heston(s0=S0, mu=R, **HP).simulate(T, 100, n, rng)
            else:
                P = m.simulate(T, 1, n, rng)
            res = mc_price(P, payoff_european(P, K), R, T)
            est.append(res.price); se.append(res.std_error)
        est, se = np.array(est), np.array(se)
        ax.fill_between(ns, est - 1.96 * se, est + 1.96 * se, color=c, alpha=0.18, lw=0, label="95% CI")
        ax.semilogx(ns, est, "o-", color=c, ms=5, label="Monte Carlo")
        ax.axhline(ref, color=INK, ls="--", lw=1.2, label=f"analytic {ref:.3f}")
        ax.set(title=f"{name}: ATM call", xlabel="number of paths")
        ax.legend(fontsize=8)
        final[name] = (ref, est[-1], se[-1])
    axes[0].set_ylabel("price")
    save(fig, "pricing.png")
    return final


# ----------------------------------------------------- 4. variance reduction
def figure_variance_reduction():
    gbm = GeometricBrownianMotion(S0, R, 0.2)
    n = 50_000
    plain = simulate(gbm, T, 50, n, "milstein", rng)
    anti = simulate(gbm, T, 50, n, "milstein", rng, antithetic=True)
    se = {
        "plain MC": mc_price(plain, payoff_european(plain, K), R, T).std_error,
        "antithetic": mc_price(anti, payoff_european(anti, K), R, T, antithetic=True).std_error,
        "control variate": mc_price_control_variate(plain, payoff_european(plain, K), R, T, S0).std_error,
    }
    fig, ax = plt.subplots(figsize=(6, 3.4))
    names = list(se)
    ax.barh(names, [se[k] for k in names], color=C[0], height=0.5)
    for i, k in enumerate(names):
        ax.text(se[k], i, f"  {se[k]:.4f}  (x{(se['plain MC'] / se[k])**2:.1f} efficiency)",
                va="center", color=INK, fontsize=9)
    ax.invert_yaxis()
    ax.set(title=f"Standard error of the ATM call price ({n:,} paths)", xlabel="standard error")
    ax.set_xlim(0, se["plain MC"] * 1.9)
    ax.grid(axis="y", visible=False)
    save(fig, "variance_reduction.png")
    return se


# ------------------------------------------------- 5. portfolio risk (VaR/ES)
def figure_risk():
    """Delta-hedged short call book: 10-day risk under three models with equal variance."""
    h, n = 10 / 252, 400_000
    pf = Portfolio(shares=0, options=[OptionPosition(-10_000, 100, 0.5)], r=R, sigma=0.2)
    pf.shares = -pf.delta(S0)  # delta hedge
    models = {
        "GBM": simulate(GeometricBrownianMotion(S0, R, 0.2), h, 10, n, rng=rng)[:, -1],
        "Heston": Heston(s0=S0, mu=R, v0=0.04, kappa=2, theta=0.04, xi=0.8, rho=-0.7).simulate(h, 10, n, rng)[0][:, -1],
        "Merton": MertonJumpDiffusion(s0=S0, mu=R, sigma=0.15, lam=0.5, mu_j=-0.1, delta=0.15).simulate(h, 10, n, rng)[:, -1],
    }
    fig, ax = plt.subplots(figsize=(8, 4))
    table = {}
    for (name, S_h), c in zip(models.items(), C):
        L = full_revaluation_losses(pf, S0, S_h, h)
        table[name] = (value_at_risk(L), expected_shortfall(L))
        ax.hist(L / 1e3, bins=300, density=True, histtype="step", color=c, lw=1.6, label=name)
    ax.set_yscale("log")
    ax.set_xlim(-5, 120)
    ax.set(title="10-day loss distribution of a delta-hedged short call book",
           xlabel="loss (thousand EUR)", ylabel="density (log scale)")
    ax.legend()
    save(fig, "risk.png")
    dn = delta_normal_var(pf, S0, 0.2, h)
    return table, dn


# --------------------------------------------------------- 6. VaR backtest
def figure_backtest():
    """Daily returns from a Heston model (volatility clustering), 99% VaR backtest."""
    years = 10
    S, _ = Heston(s0=100, mu=0.05, v0=0.04, kappa=3.0, theta=0.04, xi=1.0, rho=-0.7).simulate(
        years, 252 * years, 1, np.random.default_rng(7))
    ret = np.diff(np.log(S[0]))
    out = {}
    fig, ax = plt.subplots(figsize=(11, 3.8))
    days = np.arange(250, len(ret))
    ax.plot(days, -ret[250:] * 100, color="#b9b8b0", lw=0.7, label="daily loss")
    for (method, label), c in zip((("normal", "Gaussian VaR"), ("historical", "historical simulation VaR")), C):
        var, exc = rolling_var_backtest(ret, window=250, method=method)
        k = kupiec_test(exc)
        out[label] = k
        ax.plot(days, var * 100, color=c, lw=1.4, label=f"{label}: {k.n_exceptions} exceptions (p = {k.p_value:.3f})")
    ax.set(title=f"99% one-day VaR backtest (expected exceptions: {out['Gaussian VaR'].expected:.0f})",
           xlabel="trading day", ylabel="loss (%)")
    ax.legend(fontsize=8, loc="upper left")
    save(fig, "backtest.png")
    return out


if __name__ == "__main__":
    figure_paths()
    orders = figure_convergence()
    prices = figure_pricing()
    se = figure_variance_reduction()
    risk, dn = figure_risk()
    bt = figure_backtest()

    lines = ["# Results", "", "Generated by `scripts/run_experiments.py`.", "",
             "## Convergence orders", "", "| quantity | empirical | theory |", "|---|---|---|",
             f"| strong order Euler | {orders['strong_euler']:.2f} | 0.5 |",
             f"| strong order Milstein | {orders['strong_milstein']:.2f} | 1.0 |",
             f"| weak order Euler | {orders['weak_euler']:.2f} | 1.0 |", "",
             "## ATM call prices (S0 = K = 100, T = 1, r = 3%)", "",
             "| model | analytic | Monte Carlo | std. error |", "|---|---|---|---|"]
    lines += [f"| {k} | {a:.4f} | {m:.4f} | {s:.4f} |" for k, (a, m, s) in prices.items()]
    lines += ["", "## Variance reduction (50,000 paths)", "", "| method | std. error | efficiency gain |", "|---|---|---|"]
    lines += [f"| {k} | {v:.4f} | {(se['plain MC'] / v)**2:.1f}x |" for k, v in se.items()]
    lines += ["", "## 10-day risk of a delta-hedged short call book (10,000 calls)", "",
              "| model | VaR 99% (EUR) | ES 99% (EUR) |", "|---|---|---|"]
    lines += [f"| {k} | {v:,.0f} | {e:,.0f} |" for k, (v, e) in risk.items()]
    lines += [f"| delta-normal approximation | {dn:,.0f} | - |", "",
              "## VaR backtest (Kupiec test, 99%)", "",
              "| method | exceptions | expected | p-value | rejected at 5% |", "|---|---|---|---|---|"]
    lines += [f"| {k} | {v.n_exceptions} | {v.expected:.1f} | {v.p_value:.3f} | {'yes' if v.reject() else 'no'} |"
              for k, v in bt.items()]
    (ROOT / "docs" / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
