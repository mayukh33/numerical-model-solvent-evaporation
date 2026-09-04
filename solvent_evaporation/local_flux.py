"""Local (interfacial) flux balance.  No FiPy."""


class LocalFluxBalance:
    """Interfacial balance for one component, solved for the interface values."""

    def __init__(self, mixture, mesh):
        self.mixture = mixture
        self.mesh = mesh

    def solve(self, gamma, Dg_eff, rho_l_bnd, rho_g_bnd,
              C_liq_bnd, K_part, ddelta_dt, scaled_rhol_int):
        # Liquid-side flux = gas-side flux + Stefan convection, tied by Raoult.
        # The ddelta_dt term has a pole, so a large |ddelta_dt| is ill-conditioned.
        C_gas_bnd = self.mesh.boundary_coefficient(Dg_eff)
        numerator = C_liq_bnd * rho_l_bnd + gamma * C_gas_bnd * rho_g_bnd
        denominator = (gamma * C_gas_bnd + C_liq_bnd * K_part
                       + ddelta_dt * (K_part - gamma * self.mixture.const_rho
                                      * scaled_rhol_int))
        v0 = numerator / denominator
        return v0, K_part * v0
