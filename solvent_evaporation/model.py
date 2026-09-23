"""Governing equations in scaled coordinates: state, interface, derivative.

Lengths over the initial film thickness, t_hat = t D_ref/delta0^2 (Song et al.
Eq. 41).  Liquid: volume fraction phi_i.  Vapour: psi_i = V_i rho_g,i.  Film on
0 <= z_hat <= delta_hat, gas above it to b_hat, isothermal.

Two VOLUMES are in play, and they must not be confused: they are built from the
partial volumes of DIFFERENT phases and differ by the constant theta_i.

    phi_i                 liquid volume fraction; sums to 1
    xg_i = psi_i/theta_i  gas volume fraction, which for an ideal gas is the
                          mole fraction; sums to 1 with air
    psi_i = theta_i xg_i  liquid-equivalent volume, the STORAGE variable;
                          obeys no sum rule at all (sum psi ~ 1e-4)

psi_i pairs a LIQUID volume V_i with a GAS density rho_g,i deliberately: that is
what gives phi and psi one shared unit, so a single flux n_i closes both sides of
the interface and a single budget covers both phases.  But the gas COMPOSITION
is xg_i, not psi_i -- the frame and the inert fraction are built from xg_i.

Two fluxes, kept distinct likewise: j_i is the LIQUID volume flux and jt_i the
GAS-side flux, the latter measured in psi units so that both are velocities.

Liquid, eta = z/delta, u_i = delta_hat phi_i (Alsoy & Duda 1998 Eq. 20 in
conservative form), with j_i = -[D] dphi/dz_hat:

    du_i/dt_hat = -d/deta ( j_i - eta ddelta_hat/dt_hat phi_i )

Gas, eta = 1 + (z - delta)/L so 1 <= eta <= 2 (Song et al. Eq. 15), with
w_i = L_hat psi_i and jt_i = -(1/alpha_i) dpsi_i/dz_hat:

    dw_i/dt_hat = -d/deta ( (s - (eta - 1) dL_hat/dt_hat) psi_i
                            - (1/(alpha_i L_hat)) dpsi_i/deta )

The outer edge is held in the laboratory, so the gas layer grows as the film
recedes: L_hat = b_hat - delta_hat.  A grid point at fixed eta then moves at
ddelta_hat/dt_hat + (eta - 1) dL_hat/dt_hat = (2 - eta) ddelta_hat/dt_hat, which
is their Eq. (20).

Both phases are written in the VOLUME-average frame, so the gas composition is
mole fractions, xg_i = psi_i/theta_i.  Song et al. instead use the mass-average
velocity and close the draft with a total mass balance (their Eqs. 4-5); this
model uses air insolubility, which needs no gas density.  See VOLUME_FRAME.tex.

Interface: Raoult, psi_i^s = gamma_i xl_i, with xl_i the LIQUID mole fraction
(Song et al. Eq. 8); air is insoluble, so (1 - sum xg_i^s) s = sum jt_i/theta_i;
and the jump balance (Song et al. Eq. 10; Alsoy & Duda 1999 Eq. 6)

    j_i - phi_i^s ddelta_hat/dt_hat = n_i = psi_i^s s + jt_i

sums to ddelta_hat/dt_hat = -sum_i n_i, which reduces to Alsoy & Duda 1998's
binary thickness equation when only one component evaporates.

State: delta_hat phi_i (N-1 blocks), L_hat psi_i (N blocks), delta_hat, then the
volume of each component gone through the outer edge.
"""

import numpy
from scipy.optimize import brentq, least_squares

from .mesh import Grid


class EvaporationModel:
    """One evaporating film: mixture groups, initial state, geometry, meshes.

    b_hat is the outer gas boundary over the initial film thickness (2 in Song
    et al. Eq. 15), fixed in the laboratory, so the gas layer is b_hat -
    delta_hat and grows as the film recedes; psi_ambient is the vapour at that
    edge, zero in their Eq. (7).
    """

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

        self.liquid_grid = Grid(60) if liquid_grid is None else liquid_grid
        self.gas_grid = Grid(120, start=1.0) if gas_grid is None else gas_grid

        self.liquid_end = (N - 1) * self.liquid_grid.n
        self.delta_index = self.liquid_end + N * self.gas_grid.n
        self.size = self.delta_index + 1 + N

    # --- state ----------------------------------------------------------------

    def initial_state(self):
        """Uniform liquid, ambient gas, delta_hat = 1 (Alsoy & Duda 1998
        Eqs. 24-25)."""
        nl, ng = self.liquid_grid.n, self.gas_grid.n
        return numpy.concatenate((
            numpy.repeat(self.phi0[:-1], nl),
            numpy.repeat(self.psi_ambient, ng) * self.L_hat0,
            [1.0],
            numpy.zeros(self.mixture.n)))

    def fields(self, y):
        """Unpack: phi (N, n_liquid), psi (N, n_gas), delta_hat, L_hat."""
        N = self.mixture.n
        delta_hat = y[self.delta_index]
        L_hat = self.b_hat - delta_hat
        u = y[:self.liquid_end].reshape(N - 1, self.liquid_grid.n)
        w = y[self.liquid_end:self.delta_index].reshape(N, self.gas_grid.n)
        return self.mixture.fill_last(u / delta_hat), w / L_hat, delta_hat, L_hat

    def conserved_volume(self, y, fields=None):
        """Volume of each component: film + vapour + gone.  Constant in time, so
        its drift is the conservation error (Song et al. Eq. 12)."""
        phi, psi, delta_hat, L_hat = self.fields(y) if fields is None else fields
        in_film = delta_hat * self.liquid_grid.step * phi.sum(axis=1)
        in_gas = L_hat * self.gas_grid.step * psi.sum(axis=1)
        gone = y[self.delta_index + 1:]
        return in_film + in_gas + gone

    # --- interface -------------------------------------------------------------

    def interface(self, phi_last, psi_first, delta_hat, L_hat):
        """Solve the jump balance for the interface composition.

        Evaporation depletes the surface, so the root lies between zero and the
        cell value: the residual is positive at zero (the liquid supplies a
        stripped surface, nothing evaporates) and negative at the cell value
        (evaporation outruns a supply that has stopped).

        With one volatile component that bracket makes it a scalar root-find,
        which brentq closes to 1e-12 in a handful of evaluations -- worth doing,
        since this solve is most of the cost of every rhs call.

        With more it is a bounded least-squares started at zero, closed to
        1e-10.  Its box 0 <= phi_i <= 1 is per component: it keeps each one off
        the second root above phi = 1, which solves the algebra but is not a
        composition, but it does not impose sum_i phi_i <= 1, so for N >= 3 the
        reconstructed last component can in principle come back negative.  No
        run so far has done so; read Result.phi_interface if a ternary case
        turns strange.
        """
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
                # No sign change: a vapour value has gone slightly negative, so
                # a stripped surface evaporates rather than condenses and the
                # bracket is lost.  The bounded solve below still handles it.
                pass

        solved = least_squares(self.interface_residual, numpy.zeros(n), args=args,
                               bounds=(0.0, 1.0),
                               xtol=1e-10, ftol=1e-10, gtol=1e-15)
        return self.interface_state(solved.x, psi_first, h_g)

    def interface_state(self, phi_s_n, psi_first, h_g):
        """(phi_s, psi_s, flux n_i, Stefan draft, ddelta_hat/dt_hat) at phi_s_n."""
        m = self.mixture
        phi_s = m.fill_last(phi_s_n)
        psi_s = m.vapour_fraction(phi_s)
        j = (psi_s - psi_first) / (m.alpha * h_g)
        # Volume frame: the gas composition is mole fractions, xg_i =
        # psi_i/theta_i, so `air` is the inert mole fraction left over.  The
        # mass frame put lam_i here instead and needed a gas density for it.
        air = 1.0 - (psi_s / m.theta).sum()
        if air <= 0.0:
            raise RuntimeError(
                f"interfacial vapour fills the gas (sum xg_i = "
                f"{(psi_s / m.theta).sum():.4g}): the film is at or above its "
                f"boiling point, which this model excludes.")
        slip = (j / m.theta).sum() / air
        flux = psi_s * slip + j
        return phi_s, psi_s, flux, slip, -flux.sum()

    def interface_residual(self, phi_s_n, phi_last, psi_first, h_l, h_g):
        """Jump balance, liquid side minus gas side, components 1..N-1."""
        m = self.mixture
        n = m.n - 1
        phi_s, _, flux, _, ddelta_dt = self.interface_state(
            phi_s_n, psi_first, h_g)
        D = m.fick_matrix(0.5 * (phi_s + phi_last))
        liquid_flux = D @ (phi_last[:n] - phi_s_n) / h_l
        return liquid_flux - phi_s_n * ddelta_dt - flux[:n]

    # --- derivative -------------------------------------------------------------

    def rhs(self, t, y):
        """d(state)/dt_hat.  t does not appear: nothing is driven externally."""
        m = self.mixture
        n = m.n - 1
        liquid, gas = self.liquid_grid, self.gas_grid
        phi, psi, delta_hat, L_hat = self.fields(y)
        _, _, flux, slip, ddelta_dt = self.interface(
            phi[:, -1], psi[:, 0], delta_hat, L_hat)
        dL_dt = -ddelta_dt          # the outer edge is fixed in the laboratory

        # Liquid: no flux at the substrate, n_i at the interface.
        phi_face = liquid.face_values(phi)
        gradient = liquid.face_gradient(phi[:n], delta_hat)
        F = numpy.empty((n, liquid.n + 1))
        F[:, 0] = 0.0
        for f in range(liquid.n - 1):
            diffusion = -m.fick_matrix(phi_face[:, f]) @ gradient[:, f]
            grid_motion = liquid.faces[f + 1] * ddelta_dt * phi_face[:n, f]
            F[:, f + 1] = diffusion - grid_motion
        F[:, -1] = flux[:n]

        # Gas: each vapour rides the draft and diffuses at 1/alpha_i.
        psi_face = gas.face_values(psi)
        G = numpy.empty((m.n, gas.n + 1))
        G[:, 0] = flux
        G[:, 1:-1] = ((slip - (gas.faces[1:-1] - gas.start) * dL_dt) * psi_face
                      - gas.face_gradient(psi, L_hat) / m.alpha[:, None])
        G[:, -1] = ((slip - dL_dt) * self.psi_ambient
                    - (self.psi_ambient - psi[:, -1])
                    / (m.alpha * 0.5 * L_hat * gas.step))

        return numpy.concatenate((liquid.divergence(F).ravel(),
                                  gas.divergence(G).ravel(),
                                  [ddelta_dt],
                                  G[:, -1]))
