"""Uniform finite-volume cells on [start, start + 1]; start = 1 for the gas."""

import numpy


class Grid:
    """n uniform cells; give `n` or `step`, which rounds to whole cells."""

    def __init__(self, n=None, step=None, start=0.0):
        if step is not None:
            n = numpy.max((2, numpy.round(1.0 / step))).astype(numpy.int64)
        if n is None:
            raise TypeError(
                "give the grid a cell count `n` or a mesh step `step`")

        self.n = n
        self.start = float(start)
        self.step = 1.0 / n
        self.faces = self.start + numpy.linspace(0.0, 1.0, n + 1)
        self.centers = 0.5 * (self.faces[:-1] + self.faces[1:])

    def face_values(self, cells):
        """Cell values (..., n) averaged onto the interior faces."""
        # phibar_f = (phi_f + phi_{f+1}) / 2
        return 0.5 * (cells[..., :-1] + cells[..., 1:])

    def face_gradient(self, cells, length):
        """d(cells)/dz_hat on the interior faces, for a domain of that length."""
        # dphi/dz_hat|_f = (phi_{f+1} - phi_f) / (length * deta)
        return numpy.diff(cells, axis=-1) / (length * self.step)

    def divergence(self, face_flux):
        """Minus the flux difference, per cell width."""
        # du_c/dt_hat = -(F_c - F_{c-1}) / deta
        return -numpy.diff(face_flux, axis=-1) / self.step
