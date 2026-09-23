"""Dimensionless groups gamma, theta, alpha, nu, D0; see VOLUME_FRAME.tex."""

import numpy

from . import diffusion


class Mixture:
    """N solvents as dimensionless groups; the least volatile must be LAST."""

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
        """Append the reconstructed last component, so phi sums to one."""
        # phi_N = 1 - sum_{i<N} phi_i
        partial = numpy.asarray(partial, dtype=float)
        last = 1.0 - partial.sum(axis=0)
        return numpy.concatenate((partial, last.reshape((1,) + partial.shape[1:])))

    def mole_fractions(self, phi):
        """Mole fractions phi_i/nu_i, normalised; not clipped."""
        # x_i = (phi_i/nu_i) / sum_k (phi_k/nu_k)
        c = phi / self.nu
        return c / c.sum()

    def vapour_fraction(self, phi):
        """Raoult's law: psi_i = gamma_i x_i."""
        # psi_i^s = gamma_i x_i^(l),  gamma_i = p_sat,i v_i / (R T)
        return self.gamma * self.mole_fractions(phi)

    def fick_matrix(self, phi):
        """Volume-frame Fick matrix over D_ref, (N-1, N-1), at phi (N,)."""
        return diffusion.fick_matrix(phi, self.nu, self.D0)
