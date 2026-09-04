"""FiPy variables and equations for one component.  Requires FiPy."""

import numpy as np
import fipy as fp


class ComponentField:
    """Liquid and gas fields, boundary constraints and PDEs for one component."""

    def __init__(self, mesh, delta_var, Lg_var, v_liq_face, v_gas_face,
                 eff_Dl_cv, liquid_ini=1.0, gas_ini=0.0):
        # liquid_ini / gas_ini: scalar or array of length mesh.nx, in the scaled
        # units the solver transports.  gas_ini = 0.0 starts the gas empty.
        self.mesh = mesh

        self.uN_var = fp.Variable(value=1.0)
        self.v0_var = fp.Variable(value=0.0)

        self.rho_l = fp.CellVariable(mesh=mesh.liquid, value=liquid_ini,
                                     hasOld=True)
        self.rho_g = fp.CellVariable(mesh=mesh.gas, value=gas_ini, hasOld=True)

        self.rho_l.constrain(self.uN_var, mesh.liquid.facesRight)
        self.rho_g.constrain(self.v0_var, mesh.gas.facesLeft)
        self.rho_g.constrain(0.0, mesh.gas.facesRight)

        self.Dg_eff_cv = fp.CellVariable(mesh=mesh.gas, value=1.0)

        self.eq_l = (fp.TransientTerm(coeff=delta_var) +
                     fp.UpwindConvectionTerm(coeff=v_liq_face) ==
                     fp.DiffusionTerm(coeff=eff_Dl_cv.arithmeticFaceValue))
        self.eq_g = (fp.TransientTerm(coeff=Lg_var) +
                     fp.UpwindConvectionTerm(coeff=v_gas_face) ==
                     fp.DiffusionTerm(coeff=self.Dg_eff_cv.arithmeticFaceValue))

    def update_old(self):
        self.rho_l.updateOld()
        self.rho_g.updateOld()

    def set_gas_diffusivity(self, Dg_eff):
        self.Dg_eff_cv.setValue(np.float64(Dg_eff))

    def set_interface(self, v0, uN):
        self.v0_var.setValue(np.float64(v0))
        self.uN_var.setValue(np.float64(uN))

    def solve(self, dt):
        self.eq_l.solve(var=self.rho_l, dt=dt)
        self.eq_g.solve(var=self.rho_g, dt=dt)

    def gas_integral(self):
        return self.mesh.gas_integral(self.rho_g)

    def gas_gradient_right(self):
        return self.mesh.gas_gradient_right(self.rho_g)
