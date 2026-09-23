"""Multicomponent solvent evaporation into still air, by the method of lines.

Everything is in scaled coordinates: lengths over the initial film thickness,
time as Song et al.'s t_hat = t D_ref/delta0^2, and a case set by the
dimensionless groups gamma_i, alpha_i, lam_i, nu_i and b_hat rather than by any
dimensional property.  The mesh step and the time step are both fixed and both
given by the caller.  Needs only NumPy and SciPy.

mesh        Grid: uniform finite-volume cells
diffusion   Maxwell-Stefan / multicomponent Vignes -> volume-frame Fick matrix
mixture     Mixture: the dimensionless groups, Raoult vapour, ideal solution
model       EvaporationModel: state layout, interface closure, right-hand side
compute     compute(): fixed-step backward Euler, post-processed into a Result
"""

from .mesh import Grid
from .mixture import Mixture
from .model import EvaporationModel
from .compute import Result, compute

__all__ = ["Grid", "Mixture", "EvaporationModel",
           "Result", "compute"]
