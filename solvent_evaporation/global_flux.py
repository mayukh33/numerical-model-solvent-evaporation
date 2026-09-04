"""Global (domain-integrated) flux balance."""


class GlobalFluxBalance:
    """Interface recession rate from the gas-layer integral balance."""

    def __init__(self, mixture):
        self.mixture = mixture

    def rate_terms(self, omega0, gamma, alpha, Lg, grad_gas_right, dIg_dt):
        term1 = (omega0 * gamma / (alpha * Lg)) * grad_gas_right
        term2 = omega0 * gamma * Lg * dIg_dt
        return term1, term2

    def denominator(self, scaled_rhol_int, Ig_A, Ig_B):
        m = self.mixture
        return (scaled_rhol_int
                - (m.omega0_A * m.gamma_A * Ig_A)
                - (m.omega0_B * m.gamma_B * Ig_B))

    def ddelta_dt(self, Lg, scaled_rhol_int,
                  grad_A, dIg_A_dt, Ig_A,
                  grad_B, dIg_B_dt, Ig_B):
        m = self.mixture
        term1_A, term2_A = self.rate_terms(m.omega0_A, m.gamma_A, m.alpha_A,
                                           Lg, grad_A, dIg_A_dt)
        term1_B, term2_B = self.rate_terms(m.omega0_B, m.gamma_B, m.alpha_B,
                                           Lg, grad_B, dIg_B_dt)
        denom = self.denominator(scaled_rhol_int, Ig_A, Ig_B)
        return (term1_A - term2_A + term1_B - term2_B) / denom
