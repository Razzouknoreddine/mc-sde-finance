# Project Plan

**Title:** Monte Carlo Simulation of Stochastic Differential Equations for Pricing and Risk Measurement of Financial Derivatives
**Scope:** Software project, 6 ECTS (about 180 hours), M.Sc. Mathematics, University of Augsburg
**Author:** Noreddine Razzouk

## Objectives

1. Implement a modular Python library for simulating SDEs used in finance.
2. Verify the numerical schemes empirically against convergence theory.
3. Price European and path-dependent options by Monte Carlo and validate the estimators against analytical benchmarks.
4. Quantify market risk (VaR, ES) of an option portfolio and study model risk.
5. Backtest VaR models statistically.
6. Follow good software engineering practice: tests, continuous integration, documentation, reproducibility.

## Work packages and timeline

| # | Work package | Deliverable | Effort |
|---|---|---|---|
| 1 | Project setup | repository, packaging, CI pipeline | 10 h |
| 2 | Theory: Itô calculus, SDEs, discretisation | notes for the report | 25 h |
| 3 | Stochastic processes | `processes.py` + moment tests | 25 h |
| 4 | Discretisation and convergence | `schemes.py`, `convergence.py` + order tests | 25 h |
| 5 | Analytical benchmarks | `analytic.py` (Black-Scholes, Heston, Merton) | 20 h |
| 6 | Monte Carlo pricing and Greeks | `pricing.py` + validation tests | 25 h |
| 7 | Risk measurement and backtesting | `risk.py` + tests | 20 h |
| 8 | Experiments and figures | `run_experiments.py`, `docs/results.md` | 10 h |
| 9 | Written report and presentation | report (about 15 pages), slides | 20 h |
|   | **Total** | | **180 h** |

## Report outline

1. Introduction and motivation
2. Stochastic processes in finance (GBM, Vasicek, CIR, Heston, Merton)
3. Numerical solution of SDEs: Euler-Maruyama, Milstein, strong and weak convergence
4. Monte Carlo methods: estimator, error, variance reduction, Greeks
5. Analytical benchmarks: Black-Scholes, Heston via Fourier inversion, Merton series
6. Risk measurement: VaR, ES, full revaluation vs delta-normal, model risk
7. Backtesting: Kupiec test, results
8. Software design and validation strategy
9. Results and discussion
10. Conclusion and outlook

## Validation strategy

Every component is tested against an independent reference:

- simulated moments vs closed-form moments,
- empirical convergence orders vs theoretical orders,
- Monte Carlo prices vs analytical prices (within the 95% confidence interval),
- semi-analytical formulas vs known limit cases (Heston and Merton reduce to Black-Scholes),
- risk measures vs closed-form values for the normal distribution,
- statistical tests vs data with known exception probability.
