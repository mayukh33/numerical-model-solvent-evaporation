import numpy
from . import diffusion


class Mixture:
    """N solvents as dimensionless groups; any component order is accepted."""

    def __init__(self, gamma, alpha, theta, nu, D0):
        self.gamma = numpy.asarray(gamma, dtype=float)
        self.alpha = numpy.asarray(alpha, dtype=float)
        self.theta = numpy.asarray(theta, dtype=float)
        self.nu = numpy.asarray(nu, dtype=float)
        self.D0 = numpy.asarray(D0, dtype=float)
        self.n = self.gamma.size

    def fill_last(self, partial):
        """Append the reconstructed last component, so phi sums to one."""
        # phi_N = 1 - sum_{i<N} phi_i
        partial = numpy.asarray(partial, dtype=float)
        last = 1.0 - partial.sum(axis=0)
        return numpy.concatenate((partial, last.reshape((1,) + partial.shape[1:])))

    def mole_fractions(self, phi):
        """Mole fractions at phi (N,), normalised over components; not clipped."""
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
