"""Uniform finite-volume cells on an interval of unit length.

After the moving-boundary transform Song et al. carry one coordinate across both
phases: 0 to 1 in the liquid, 1 to 2 in the gas (their Eqs. 14-15).  `start`
places a grid on either.  Alsoy & Duda 1998 grade theirs toward the free surface
instead (their Eq. 29, Fig. 2) to resolve the steep gradient there; with a fixed
step that resolution has to be bought with cells everywhere.

The grid owns the stencils that depend on its spacing, so both halves of
model.rhs share one definition of each.
"""

import numpy


class Grid:
    """n uniform cells on [start, start + 1]; start = 1 for the gas phase.

    Give the cell count as `n` or the mesh step as `step`; `step` wins, and is
    rounded to a whole number of cells, so `step=0.3` becomes three cells of
    1/3.  Read `self.step` back for the width actually used.  `start` only
    labels the cells -- spacing, and everything derived from it, is the same
    either way.  The step in scaled height is `step` times delta_hat in the
    liquid and times L_hat in the gas.
    """

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
        return 0.5 * (cells[..., :-1] + cells[..., 1:])

    def face_gradient(self, cells, length):
        """d(cells)/dz_hat on the interior faces, for a domain of that length."""
        return numpy.diff(cells, axis=-1) / (length * self.step)

    def divergence(self, face_flux):
        """Minus the flux difference, per cell width."""
        return -numpy.diff(face_flux, axis=-1) / self.step
