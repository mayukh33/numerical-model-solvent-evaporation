"""1D two-component solvent evaporation with a moving interface.

components      Component and ComponentLibrary types (no data; you supply it)
thermodynamics  Mixture: reference state and the equation of state
mesh            ScaledMesh: the eta grids and dx-dependent operators
fields          ComponentField: FiPy variables and equations, one component
local_flux      LocalFluxBalance: interfacial balance -> v0, uN
global_flux     GlobalFluxBalance: integrated balance -> ddelta_dt
solver          EvaporationSolver: the time loop

Deliberately imports nothing: mesh, fields and solver need FiPy, the rest do not.
"""

__all__ = ["components", "thermodynamics", "mesh", "fields", "local_flux",
           "global_flux", "solver"]
