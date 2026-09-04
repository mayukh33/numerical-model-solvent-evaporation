"""Mixture reference state and the per-cell equation of state"""


class Mixture:
    """Binary mixture of components A and B at a fixed initial composition."""

    def __init__(self, component_A, component_B, wt_frac_A, *,
                 R, T, rho_gas, D_ab=None, D_ba=None, library=None):
        if D_ab is None or D_ba is None:
            if library is None:
                raise ValueError(
                    "Mixture needs liquid diffusivities: pass library=<your "
                    "ComponentLibrary>, or both D_ref and D_bm explicitly.")
            D_ab, D_ba = library.binary_diffusivities(
                component_A.name, component_B.name)

        self.comp_A = component_A
        self.comp_B = component_B
        self.wt_frac_A = wt_frac_A
        self.wt_frac_B = 1.0 - wt_frac_A
        self.D_ab = D_ab
        self.D_ba = D_ba

        self.rhol_0 = 1.0 / (self.wt_frac_A / component_A.rho_pure
                             + self.wt_frac_B / component_B.rho_pure)
        self.rhol_A_ini = self.wt_frac_A * self.rhol_0
        self.rhol_B_ini = self.wt_frac_B * self.rhol_0

        self.x_A_ini, self.x_B_ini = self.mole_fractions(self.wt_frac_A,
                                                          self.wt_frac_B)
        self.Ml0 = (self.x_A_ini * component_A.M) + (self.x_B_ini * component_B.M)

        self.const_rho = self.rhol_0 / rho_gas
        self.omega0_A = self.rhol_A_ini / self.rhol_0
        self.omega0_B = self.rhol_B_ini / self.rhol_0
        self.gamma_A = component_A.gamma(self.Ml0, self.rhol_0, R, T)
        self.gamma_B = component_B.gamma(self.Ml0, self.rhol_0, R, T)
        self.alpha_A = component_A.alpha(D_ab)
        self.alpha_B = component_B.alpha(D_ab)

    def weight_fractions(self, scaled_l_A, scaled_l_B):
        # From the RATIO of the transported fields, which is what the diffusion
        # problem actually determines.
        rho_A = scaled_l_A * self.rhol_A_ini
        rho_B = scaled_l_B * self.rhol_B_ini
        wt_A = rho_A / (rho_A + rho_B)
        return wt_A, 1.0 - wt_A

    def scaled_density(self, wt_A, wt_B):
        rhol = 1.0 / (wt_A / self.comp_A.rho_pure + wt_B / self.comp_B.rho_pure)
        return rhol, rhol / self.rhol_0

    def mole_fractions(self, wt_A, wt_B):
        Ma, Mb = self.comp_A.M, self.comp_B.M
        x_A = (wt_A / Ma) / (wt_A / Ma + wt_B / Mb)
        return x_A, 1.0 - x_A

    def scaled_molar_mass(self, x_A, x_B):
        Ml = (x_A * self.comp_A.M) + (x_B * self.comp_B.M)
        return Ml / self.Ml0

    def effective_liquid_diffusivity(self, x_A, x_B, delta):
        # Vignes, scaled.  The 1/delta is the Landau transform.
        Dl = (self.D_ab ** x_B) * (self.D_ba ** x_A)
        return (Dl / self.D_ab) / delta

    def gas_diffusivity(self, alpha, Lg):
        return (1.0 / alpha) / Lg

    def partition_coefficient(self, scaled_rhol_int, scaled_Ml_int):
        return scaled_rhol_int / scaled_Ml_int

    def stefan_factor(self, scaled_rhol_int):
        return 1.0 - self.const_rho * scaled_rhol_int
