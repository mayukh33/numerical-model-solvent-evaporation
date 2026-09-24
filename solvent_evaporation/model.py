"""Governing equations in scaled coordinates; see VOLUME_FRAME.tex."""

import numpy
from scipy.optimize import brentq, least_squares

from .mesh import Grid


class EvaporationModel:
    """One evaporating film: mixture groups, initial state, geometry, meshes."""

    def __init__(self, mixture, phi0, *, b_hat,
                 psi_ambient=None, liquid_grid=None, gas_grid=None):
        N = mixture.n
        phi0 = numpy.asarray(phi0, dtype=float)

        self.mixture = mixture
        self.phi0 = phi0 / phi0.sum()
        self.b_hat = float(b_hat)
        self.L_hat0 = self.b_hat - 1.0
        self.psi_ambient = (numpy.zeros(N) if psi_ambient is None
                            else numpy.asarray(psi_ambient, dtype=float))

        if liquid_grid is None:
           raise ValueError("Provide a valid liquid grid")
        else:
           self.liquid_grid = liquid_grid
        if gas_grid is None:
           raise ValueError("Provide a valid gas grid")
        else:
            self.gas_grid = gas_grid

        self.liquid_end = (N - 1) * self.liquid_grid.n
        self.delta_index = self.liquid_end + N * self.gas_grid.n
        self.size = self.delta_index + 1

    # --- state ----------------------------------------------------------------

    def initial_state(self):
        """Uniform liquid, ambient gas, delta_hat = 1."""
        nl, ng = self.liquid_grid.n, self.gas_grid.n
        return numpy.concatenate((
            numpy.repeat(self.phi0[:-1], nl),
            numpy.repeat(self.psi_ambient, ng) * self.L_hat0,
            [1.0]))

    def fields(self, y):
        """Unpack: phi (N, n_liquid), psi (N, n_gas), delta_hat, L_hat."""
        # stored u_i = delta_hat phi_i, w_i = L_hat psi_i; L_hat = b_hat - delta_hat
        N = self.mixture.n
        delta_hat = y[self.delta_index]
        L_hat = self.b_hat - delta_hat
        u = y[:self.liquid_end].reshape(N - 1, self.liquid_grid.n)
        w = y[self.liquid_end:self.delta_index].reshape(N, self.gas_grid.n)
        return self.mixture.fill_last(u / delta_hat), w / L_hat, delta_hat, L_hat

    # --- interface -------------------------------------------------------------

    def interface(self, phi_last, psi_first, delta_hat, L_hat):
        """Solve the jump balance for the interface composition."""
        n = self.mixture.n - 1
        h_l = 0.5 * delta_hat * self.liquid_grid.step
        h_g = 0.5 * L_hat * self.gas_grid.step
        args = (phi_last, psi_first, h_l, h_g)

        if n == 1 and phi_last[0] > 0.0:
            try:
                root = brentq(lambda x: self.interface_residual(
                    numpy.array([x]), *args)[0], 0.0, phi_last[0], xtol=1e-12)
                return self.interface_state(numpy.array([root]), psi_first, h_g)
            except ValueError:
                # No sign change: bracket lost; the bounded solve handles it.
                pass

        solved = least_squares(self.interface_residual,
                               numpy.clip(phi_last[:n], 0.0, 1.0), args=args,
                               bounds=(0.0, 1.0),
                               xtol=1e-10, ftol=1e-10, gtol=1e-15)
        # least_squares can report success on a bound short of the root
        step = numpy.linalg.lstsq(solved.jac, solved.fun, rcond=None)[0]
        if numpy.abs(step).max() > 1e-8 or solved.x.sum() > 1.0:
            raise RuntimeError(
                f"the interface solve found no root (phi_s = {solved.x}, "
                f"residual {numpy.abs(solved.fun).max():.3g})")
        return self.interface_state(solved.x, psi_first, h_g)

    def interface_state(self, phi_s_n, psi_first, h_g):
        """(phi_s, psi_s, flux n_i, Stefan draft, ddelta_hat/dt_hat) at phi_s_n."""
        m = self.mixture
        phi_s = m.fill_last(phi_s_n)
        # psi_i^s = gamma_i x_i^(l)(phi^s): Raoult at the interface
        psi_s = m.vapour_fraction(phi_s)
        # jt_i = -(1/alpha_i) dpsi_i/dz_hat -> (psi_i^s - psi_i,1) / (alpha_i h_g)
        j = (psi_s - psi_first) / (m.alpha * h_g)
        # x_a^s = 1 - sum_i psi_i^s/theta_i: gas mole fractions are xg_i = psi_i/theta_i
        air = 1.0 - (psi_s / m.theta).sum()
        if air <= 0.0:
            raise RuntimeError(
                f"interfacial vapour fills the gas (sum xg_i = "
                f"{(psi_s / m.theta).sum():.4g}): the film is at or above its "
                f"boiling point, which this model excludes.")
        # s = (sum_i jt_i/theta_i) / x_a^s: draft set by the insoluble air
        slip = (j / m.theta).sum() / air
        # n_i = psi_i^s s + jt_i: gas side of the jump
        flux = psi_s * slip + j
        # ddelta_hat/dt_hat = -sum_i n_i: the film loses what leaves it
        return phi_s, psi_s, flux, slip, -flux.sum()

    def interface_residual(self, phi_s_n, phi_last, psi_first, h_l, h_g):
        """Jump balance, liquid side minus gas side, components 1..N-1."""
        m = self.mixture
        n = m.n - 1
        phi_s, _, flux, _, ddelta_dt = self.interface_state(
            phi_s_n, psi_first, h_g)
        D = m.fick_matrix(0.5 * (phi_s + phi_last))
        # j_i = -[D] dphi/dz_hat -> [D](phi_i,nl - phi_i^s) / h_l
        liquid_flux = numpy.matmul(D, phi_last[:n] - phi_s_n) / h_l
        # r_i = j_i - phi_i^s delta' - n_i = 0: liquid side minus gas side
        return liquid_flux - phi_s_n * ddelta_dt - flux[:n]

    # right hand side of the ode
    def rhs(self, t, y):
        """d(state)/dt_hat.  t does not appear: nothing is driven externally."""
        m = self.mixture
        n = m.n - 1
        liquid, gas = self.liquid_grid, self.gas_grid
        phi, psi, delta_hat, L_hat = self.fields(y)
        _, _, flux, slip, ddelta_dt = self.interface(
            phi[:, -1], psi[:, 0], delta_hat, L_hat)
        dL_dt = -ddelta_dt          # the outer edge is fixed in the laboratory

        # d(delta_hat phi_i)/dt_hat = -d/deta [j_i - eta delta' phi_i], F_0 = 0, F_nl = n_i
        phi_face = liquid.face_values(phi)
        gradient = liquid.face_gradient(phi[:n], delta_hat)
        F = numpy.empty((n, liquid.n + 1))
        F[:, 0] = 0.0
        for f in range(liquid.n - 1):
            # -[D](phibar_f) dphi/dz_hat|_f
            diffusion = -numpy.matmul(m.fick_matrix(phi_face[:, f]),
                                      gradient[:, f])
            # eta_f delta' phibar_f: the Landau term from the moving interface
            grid_motion = liquid.faces[f + 1] * ddelta_dt * phi_face[:n, f]
            F[:, f + 1] = diffusion - grid_motion
        F[:, -1] = flux[:n]

        # d(L_hat psi_i)/dt_hat = -d/deta [(s - (eta-1) L') psi_i + jt_i], G_0 = n_i
        psi_face = gas.face_values(psi)
        G = numpy.empty((m.n, gas.n + 1))
        G[:, 0] = flux
        # G_f = [s - (eta_f - 1) L'] psibar_f - (1/alpha_i) dpsi/dz_hat|_f
        G[:, 1:-1] = ((slip - (gas.faces[1:-1] - gas.start) * dL_dt) * psi_face
                      - gas.face_gradient(psi, L_hat) / m.alpha[:, None])
        # eta = 2: the same face flux, one-sided against the held ambient psi
        G[:, -1] = ((slip - dL_dt) * self.psi_ambient
                    - (self.psi_ambient - psi[:, -1])
                    / (m.alpha * 0.5 * L_hat * gas.step))

        return numpy.concatenate((liquid.divergence(F).ravel(),
                                  gas.divergence(G).ravel(),
                                  [ddelta_dt]))
