"""The solvent mixture, as dimensionless groups:

    gamma_i = p_sat,i v_i / (R T)      Raoult partition,   Song et al. Eq. (26)
    theta_i = v_i P / (R T)            liquid over gas molar volume
    alpha_i = D_ref / Dg_i             liquid over gas diffusivity,    Eq. (31)
    nu_i    = v_i / v_ref              molar volumes; only ratios enter
    D0_ij   = D0_ij / D_ref            Vignes pair diffusivities,       Eq. (2)

v_i is the pure molar volume M_i/rho_pure,i, and D_ref sets the time scale (Song
et al. use D0_MB, their Eq. 41).

Both phases are written in the VOLUME-average frame, so the gas composition
enters as mole fractions: x_i = psi_i/theta_i, and the inert (air) fraction at
the interface is 1 - sum_i psi_i/theta_i.  Song et al. use the mass-average
velocity in the gas instead, which needs lam_i = rho_pure,i/rho_gas and a
constant total gas density; theta_i needs only P, T and the pure molar volume,
and no gas density at all.  The derivation is in VOLUME_FRAME.tex.

Note gamma_i/theta_i = p_sat,i/P, so the Raoult jump is x_i^(g) = (p_sat,i/P)
x_i^(l): gamma and theta are not independent of the vapour pressures, only a
convenient pair to carry.

Each group is per component here, where Song et al. define one number apiece:
they carry a single volatile species and take the mixture's properties.  gamma
differs in substance as well -- their Eq. (26) uses the initial MIXTURE molar
volume where this uses the pure one, so their methanol gamma is 3.83e-4 against
2.79e-4 here.  The pure form is the one the per-component Raoult jump
psi_i = gamma_i x_i needs, since psi is measured against each component's own
pure density.

The solution is ideal, as Song et al. take methanol/1-butanol, so every activity
coefficient is one and the thermodynamic factor is the identity.  Mixing is ideal
in volume too (Alsoy & Duda 1999): phi sums to one, and vapour is measured the
same way, psi_i = V_i rho_g,i, so both phases carry one unit.
"""

import numpy

from . import diffusion


class Mixture:
    """N solvents, given by the dimensionless groups above.

    gamma, theta, alpha and nu are length-N; D0 is (N, N) with D0[i, j] the
    diffusivity of i infinitely dilute in j over D_ref (diagonal unused).

    Order the components with the least volatile LAST.  fill_last rebuilds that
    one from sum(phi) = 1, leaving the other N-1 as the unknowns of the interface
    solve; the reconstructed component is the enriched one.  Evaporation depletes
    the unknowns, so their surface values sit below the bulk: with one volatile
    component that gives brentq the bracket [0, phi_bulk], and with more the
    bounded solve searches the unit box (see EvaporationModel.interface).  Put
    the least volatile among the N-1 and its surface value lies above the bulk,
    outside the bracket: the solve returns an evaporation rate about ten times
    too fast and no error, so the constructor checks.  (Alsoy & Duda 1999's last
    component is the polymer, gamma = 0, so the question never arose.)
    """

    def __init__(self, gamma, alpha, theta, nu, D0):
        self.gamma = numpy.asarray(gamma, dtype=float)
        self.alpha = numpy.asarray(alpha, dtype=float)
        self.theta = numpy.asarray(theta, dtype=float)
        self.nu = numpy.asarray(nu, dtype=float)
        self.D0 = numpy.asarray(D0, dtype=float)
        self.n = self.gamma.size

        if self.n > 1 and self.gamma[-1] > self.gamma[:-1].min():
            least = int(self.gamma.argmin())
            raise ValueError(
                f"the last component must be the least volatile, but gamma = "
                f"{self.gamma.tolist()} puts the smallest at index {least}. "
                f"Reorder every argument -- gamma, alpha, theta, nu, the rows "
                f"and columns of D0, and the initial phi0 -- so component "
                f"{least} comes last.")

    def fill_last(self, partial):
        """Append the reconstructed last component, so the fractions sum to one.

        Takes the (N-1,) interface values or an (N-1, m) profile.
        """
        partial = numpy.asarray(partial, dtype=float)
        last = 1.0 - partial.sum(axis=0)
        return numpy.concatenate((partial, last.reshape((1,) + partial.shape[1:])))

    def mole_fractions(self, phi):
        """Mole fractions at one composition: phi_i/nu_i, normalised.

        Not clipped -- a used-up component's small negative must pass through
        smoothly, since clipping kinks Raoult's law where the solve works.
        """
        c = phi / self.nu
        return c / c.sum()

    def vapour_fraction(self, phi):
        """Raoult's law (Song et al. Eq. 8): psi_i = gamma_i x_i."""
        return self.gamma * self.mole_fractions(phi)

    def fick_matrix(self, phi):
        """Volume-frame Fick matrix over D_ref, (N-1, N-1), at phi (N,)."""
        return diffusion.fick_matrix(phi, self.nu, self.D0)
