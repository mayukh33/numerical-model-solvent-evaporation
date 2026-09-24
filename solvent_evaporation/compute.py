"""Fixed-step backward Euler in scaled time; the step never adapts."""

import time
from dataclasses import dataclass

import numpy


def show_progress(k, n_steps, t_hat, delta_hat, started):
    """One line, rewritten in place: bar, position, and time left."""
    done = k / n_steps
    elapsed = time.perf_counter() - started
    filled = int(30 * done)
    print(f"\r[{'#' * filled}{'.' * (30 - filled)}] {done:5.1%}  "
          f"t_hat {t_hat:.4g}  delta_hat {delta_hat:.5f}  "
          f"{elapsed:5.0f}s elapsed",
          end="", flush=True)


@dataclass
class Result:
    """Simulation output, all scaled.  Profiles are [component, cell, time]."""

    model: object
    t: numpy.ndarray               # t_hat
    delta: numpy.ndarray           # delta_hat
    phi: numpy.ndarray             # liquid volume fractions
    psi: numpy.ndarray             # vapour, liquid-equivalent
    phi_interface: numpy.ndarray
    flux: numpy.ndarray            # evaporative flux n_i

    @property
    def eta_liquid(self):
        return self.model.liquid_grid.centers

    @property
    def eta_gas(self):
        return self.model.gas_grid.centers


def jacobian(model, t, y, eps=1e-7):
    """df/dy by forward differences; eps is floored by rhs noise."""
    # J[:, i] = (f(y + h e_i) - f(y)) / h,  h = eps |y_i|
    f0 = model.rhs(t, y)
    J = numpy.empty((y.size, y.size))
    for i in range(y.size):
        h = eps * abs(y[i])
        if h == 0.0:
            h = eps
        perturbed = y.copy()
        perturbed[i] = y[i] + h
        h = perturbed[i] - y[i]
        J[:, i] = (model.rhs(t, perturbed) - f0) / h
    return J


def backward_euler_step(model, t, y, h, newton_tol, max_newton):
    """The state at t + h, by Newton with a frozen Jacobian, re-formed if stale."""
    # y_next = y + h f(t + h, y_next); Newton solves (I - h J) c = -r for c
    matrix = numpy.eye(y.size) - h * jacobian(model, t + h, y)

    y_next = y.copy()
    previous = numpy.inf
    for _ in range(max_newton):
        try:
            # r = y_next - y - h f(t + h, y_next)
            residual = y_next - y - h * model.rhs(t + h, y_next)
        except (RuntimeError, numpy.linalg.LinAlgError) as reason:
            raise RuntimeError(
                f"the step to t_hat = {t + h:.6g} with h = {h:.3g} left the "
                f"model's range ({reason}). Use a smaller dt.") from reason
        correction = numpy.linalg.solve(matrix, -residual)
        y_next = y_next + correction
        size = numpy.max(numpy.abs(correction))
        if size <= newton_tol:
            return y_next
        # a correction that failed to halve means the frozen Jacobian is stale
        if size > 0.5 * previous:
            # converging too slowly: the frozen Jacobian has gone stale
            matrix = numpy.eye(y.size) - h * jacobian(model, t + h, y_next)
        previous = size

    raise RuntimeError(
        f"Newton did not converge at t_hat = {t + h:.6g} with h = {h:.3g}; "
        f"the last correction was {numpy.max(numpy.abs(correction)):.3g}. "
        f"Use a smaller dt.")


def compute(model, t_end, dt, store_period=1, progress=0, newton_tol=1e-10,
            max_newton=1000):
    """March from t_hat = 0 to t_end in steps of exactly dt."""
    n_steps = numpy.max((1, numpy.round(t_end / dt))).astype(numpy.int64)

    # Worked out up front so states is allocated once at its final size.
    keep = numpy.unique(numpy.append(numpy.arange(0, n_steps + 1, store_period),
                                     n_steps))
    states = numpy.empty((model.size, keep.size))

    y = model.initial_state()
    states[:, 0] = y
    kept = 1

    started = time.perf_counter()
    try:
        for k in range(1, n_steps + 1):
            # (k - 1) * dt, not a running sum, which drifts over many steps.
            y = backward_euler_step(model, (k - 1) * dt, y, dt,
                                    newton_tol, max_newton)
            if kept < keep.size and k == keep[kept]:
                states[:, kept] = y
                kept += 1
            if progress and k % progress == 0:
                show_progress(k, n_steps, k * dt, y[model.delta_index], started)
    except KeyboardInterrupt:
        # Ctrl-C keeps what has been stored instead of losing the run.
        print(f"\ninterrupted at step {k} of {n_steps} (t_hat = {k * dt:.6g}); "
              f"returning {kept} stored times, to t_hat = "
              f"{keep[kept - 1] * dt:.6g}")
    else:
        if progress:
            print()

    return post_process(model, keep[:kept] * dt, states[:, :kept])


def post_process(model, t, states):
    """Split each stored state into profiles and interface values."""
    m = model.mixture
    nt = t.size
    delta = numpy.empty(nt)
    phi = numpy.empty((m.n, model.liquid_grid.n, nt))
    psi = numpy.empty((m.n, model.gas_grid.n, nt))
    phi_interface = numpy.empty((m.n, nt))
    flux = numpy.empty((m.n, nt))

    for c in range(nt):
        fields = model.fields(states[:, c])
        phi[..., c], psi[..., c], delta[c], L_hat = fields
        phi_interface[:, c], _, flux[:, c], _, _ = model.interface(
            phi[:, -1, c], psi[:, 0, c], delta[c], L_hat)

    return Result(model=model, t=t, delta=delta, phi=phi, psi=psi,
                  phi_interface=phi_interface, flux=flux)
