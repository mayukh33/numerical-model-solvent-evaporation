"""Multicomponent solvent evaporation into still air, by the method of lines."""

from .mesh import Grid
from .mixture import Mixture
from .model import EvaporationModel
from .compute import Result, compute

__all__ = ["Grid", "Mixture", "EvaporationModel",
           "Result", "compute"]
