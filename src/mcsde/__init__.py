"""mcsde: Monte Carlo simulation of stochastic differential equations
for pricing and risk measurement of financial derivatives."""

from .processes import (CIR, SDE, GeometricBrownianMotion, Heston,
                        MertonJumpDiffusion, OrnsteinUhlenbeck)
from .schemes import brownian_increments, euler_maruyama, milstein, simulate
from .convergence import strong_order, weak_order

__version__ = "0.3.0"
