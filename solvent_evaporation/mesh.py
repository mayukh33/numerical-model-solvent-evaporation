"""Landau-immobilised grids and the dx-dependent operators.  Requires FiPy."""

import numpy as np
import fipy as fp


class ScaledMesh:
    """Liquid grid on eta in [0, 1] and gas grid on [1, scaled_b]."""

    def __init__(self, dx, scaled_b):
        self.dx = dx
        self.nx = np.ceil(1.0 / dx).astype(np.int64)
        self.scaled_b = scaled_b

        # dx MUST be passed: Grid1D defaults it to 1.0, which rescales the
        # domain to [0, nx] while every stencil below still divides by dx.
        self.liquid = fp.Grid1D(nx=self.nx, dx=dx)
        self.gas = fp.Grid1D(nx=self.nx, dx=dx) + 1.0

        self.liquid_x = self.liquid.cellCenters.value[0]
        self.gas_x = self.gas.cellCenters.value[0]

    def gas_integral(self, rho_g):
        return np.float64(np.sum(rho_g.value) * self.dx)

    def gas_gradient_right(self, rho_g):
        return (0.0 - rho_g.value[-1]) / (self.dx / 2.0)

    def boundary_coefficient(self, D):
        return 2.0 * D / self.dx
