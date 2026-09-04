"""Time loop and Picard iteration"""

from dataclasses import dataclass

import numpy as np
import fipy as fp

from .fields import ComponentField
from .global_flux import GlobalFluxBalance
from .local_flux import LocalFluxBalance


def _per_component(value):
    # A 2-tuple means the components differ; a scalar or array applies to both.
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(
                f"per-component initial condition must be a 2-tuple "
                f"(for_A, for_B), got {len(value)} entries")
        return value
    return value, value


@dataclass
class State:
    """Solver state after one completed step."""

    step: int
    t: float
    delta: float
    ddelta_dt: float
    err: float
    rho_l_A: np.ndarray
    rho_l_B: np.ndarray
    rho_g_A: np.ndarray
    rho_g_B: np.ndarray
    rho_l: np.ndarray        # scaled mixture density
    rhol: np.ndarray         # unscaled mixture density, kg/m^3


class EvaporationSolver:
    """Advances the interface position and both species fields in time."""

    def __init__(self, mixture, mesh, liquid_ini=1.0, gas_ini=0.0,
                 inner_tol=1e-13):
        self.mixture = mixture
        self.mesh = mesh
        self.inner_tol = inner_tol

        self.delta = 1.0
        self.ddelta_dt = 0.0

        self.delta_var = fp.CellVariable(mesh=mesh.liquid, value=self.delta,
                                         hasOld=True)
        self.Lg_var = fp.CellVariable(mesh=mesh.gas,
                                      value=mesh.scaled_b - self.delta,
                                      hasOld=True)
        self.eff_Dl_cv = fp.CellVariable(mesh=mesh.liquid, value=1.0)
        self.v_liq_face = fp.FaceVariable(mesh=mesh.liquid, rank=1)
        self.v_gas_face = fp.FaceVariable(mesh=mesh.gas, rank=1)

        liquid_A, liquid_B = _per_component(liquid_ini)
        gas_A, gas_B = _per_component(gas_ini)

        shared = (self.delta_var, self.Lg_var, self.v_liq_face,
                  self.v_gas_face, self.eff_Dl_cv)
        self.field_A = ComponentField(mesh, *shared,
                                      liquid_ini=liquid_A, gas_ini=gas_A)
        self.field_B = ComponentField(mesh, *shared,
                                      liquid_ini=liquid_B, gas_ini=gas_B)

        self.local = LocalFluxBalance(mixture, mesh)
        self.glob = GlobalFluxBalance(mixture)

    def _iterate(self, delta_pred, ddelta_dt_actual, dt,
                 previous_Ig_A, previous_Ig_B):
        # One Picard sweep: solve all four PDEs at delta_pred, return the rate.
        m, mesh = self.mixture, self.mesh

        self.delta_var.setValue(delta_pred)
        Lg_pred = mesh.scaled_b - delta_pred
        self.Lg_var.setValue(Lg_pred)

        rho_l_A = self.field_A.rho_l.value
        rho_l_B = self.field_B.rho_l.value
        rho_g_A = self.field_A.rho_g.value
        rho_g_B = self.field_B.rho_g.value

        wt_A, wt_B = m.weight_fractions(rho_l_A, rho_l_B)
        _, scaled_rhol = m.scaled_density(wt_A, wt_B)
        x_A, x_B = m.mole_fractions(wt_A, wt_B)
        scaled_Ml = m.scaled_molar_mass(x_A, x_B)

        eff_Dl = m.effective_liquid_diffusivity(x_A, x_B, delta_pred)
        Dg_eff_A = m.gas_diffusivity(m.alpha_A, Lg_pred)
        Dg_eff_B = m.gas_diffusivity(m.alpha_B, Lg_pred)

        C_liq_bnd = mesh.boundary_coefficient(eff_Dl[-1])
        K_part = m.partition_coefficient(scaled_rhol[-1], scaled_Ml[-1])

        v0_A, uN_A = self.local.solve(m.gamma_A, Dg_eff_A, rho_l_A[-1],
                                      rho_g_A[0], C_liq_bnd, K_part,
                                      ddelta_dt_actual, scaled_rhol[-1])
        v0_B, uN_B = self.local.solve(m.gamma_B, Dg_eff_B, rho_l_B[-1],
                                      rho_g_B[0], C_liq_bnd, K_part,
                                      ddelta_dt_actual, scaled_rhol[-1])

        self.eff_Dl_cv.setValue(np.float64(eff_Dl))
        self.field_A.set_gas_diffusivity(Dg_eff_A)
        self.field_B.set_gas_diffusivity(Dg_eff_B)

        k = m.stefan_factor(scaled_rhol[-1])
        self.v_liq_face.setValue(-mesh.liquid.faceCenters * ddelta_dt_actual)
        self.v_gas_face.setValue(ddelta_dt_actual
                                 * (mesh.gas.faceCenters - mesh.scaled_b + k))

        self.field_A.set_interface(v0_A, uN_A)
        self.field_B.set_interface(v0_B, uN_B)

        self.field_A.solve(dt)
        self.field_B.solve(dt)

        wt_A_new, wt_B_new = m.weight_fractions(self.field_A.rho_l.value,
                                                self.field_B.rho_l.value)
        rhol_new, scaled_rhol_new = m.scaled_density(wt_A_new, wt_B_new)

        Ig_A = self.field_A.gas_integral()
        Ig_B = self.field_B.gas_integral()
        dIg_A_dt = (Ig_A - previous_Ig_A) / dt
        dIg_B_dt = (Ig_B - previous_Ig_B) / dt

        rate = self.glob.ddelta_dt(
            Lg_pred, scaled_rhol_new[-1],
            self.field_A.gas_gradient_right(), dIg_A_dt, Ig_A,
            self.field_B.gas_gradient_right(), dIg_B_dt, Ig_B)

        return rate, Ig_A, Ig_B, rhol_new, scaled_rhol_new

    def initial_state(self):
        """The t = 0 state, before any step."""
        wt_A0, wt_B0 = self.mixture.weight_fractions(self.field_A.rho_l.value,
                                                     self.field_B.rho_l.value)
        rhol_0, scaled_rhol_0 = self.mixture.scaled_density(wt_A0, wt_B0)
        return State(step=0, t=0.0, delta=self.delta,
                     ddelta_dt=self.ddelta_dt, err=0.0,
                     rho_l_A=self.field_A.rho_l.value,
                     rho_l_B=self.field_B.rho_l.value,
                     rho_g_A=self.field_A.rho_g.value,
                     rho_g_B=self.field_B.rho_g.value,
                     rho_l=scaled_rhol_0, rhol=rhol_0)

    def run(self, final_time, dt):
        """Advance to final_time, yielding a State after every completed step."""
        previous_Ig_A = self.field_A.gas_integral()
        previous_Ig_B = self.field_B.gas_integral()

        step = 1
        t_ = dt

        while t_ <= final_time:
            err = 1
            delta_old = self.delta
            ddelta_dt_old = self.ddelta_dt

            self.delta_var.updateOld()
            self.Lg_var.updateOld()
            self.field_A.update_old()
            self.field_B.update_old()

            delta_pred = delta_old + ddelta_dt_old * dt
            ddelta_dt_actual = ddelta_dt_old

            while err > self.inner_tol:
                (ddelta_dt_actual, Ig_A_new, Ig_B_new, rhol_new,
                 scaled_rhol_new) = self._iterate(
                    delta_pred, ddelta_dt_actual, dt,
                    previous_Ig_A, previous_Ig_B)

                delta_next = delta_old + ddelta_dt_actual * dt
                err = abs(delta_next - delta_pred)
                if err < self.inner_tol:
                    delta_pred = delta_next
                    break
                delta_pred = delta_next

            self.ddelta_dt = ddelta_dt_actual
            previous_Ig_A = Ig_A_new
            previous_Ig_B = Ig_B_new
            self.delta = delta_old + (self.ddelta_dt * dt)

            yield State(step=step, t=t_, delta=self.delta,
                        ddelta_dt=self.ddelta_dt, err=err,
                        rho_l_A=self.field_A.rho_l.value,
                        rho_l_B=self.field_B.rho_l.value,
                        rho_g_A=self.field_A.rho_g.value,
                        rho_g_B=self.field_B.rho_g.value,
                        rho_l=scaled_rhol_new, rhol=rhol_new)
            t_ += dt
            step += 1
