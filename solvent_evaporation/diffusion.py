"""Liquid diffusion as a Fick matrix in the volume-average frame, one
composition at a time: phi is (N,), the matrix is (N-1, N-1).

Multicomponent Vignes (Wesselingh & Krishna), Song et al. Eq. (2) for N = 2:

    D_ij = D0_ji^x_i D0_ij^x_j prod_{k != i, j} (D0_ik D0_jk)^(x_k/2)

Inverted to a Fick matrix in the molar-average frame (Taylor & Krishna 1993),
with no thermodynamic factor because the solution is ideal:

    J* = -c_t [B]^-1 grad x ,
    B_ii = x_i/D_iN + sum_{k != i} x_k/D_ik ,   B_ij = -x_i (1/D_ij - 1/D_iN)

then moved to the volume-average frame, where the liquid is at rest (Alsoy &
Duda 1999 Eq. 2), and onto volume-fraction gradients:

    j = -[D] grad phi ,   [D] = diag(nu) T c_t B^-1 X ,
    T_ik = delta_ik - c_i (nu_k - nu_N) ,   X_jl = d x_j / d phi_l
"""

import numpy


def maxwell_stefan_diffusivities(x, D0):
    """Vignes diffusivities D_ij, (N, N), at mole fractions x.  Diagonal unused."""
    N = x.size
    D = numpy.ones((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            D[i, j] = D0[j, i] ** x[i] * D0[i, j] ** x[j]
            for k in range(N):
                if k != i and k != j:
                    D[i, j] *= (D0[i, k] * D0[j, k]) ** (0.5 * x[k])
    return D


def fick_matrix(phi, nu, D0):
    """Volume-frame Fick matrix over D_ref, (N-1, N-1), at volume fractions phi."""
    N = phi.size
    n = N - 1
    c = phi / nu
    c_t = c.sum()
    x = c / c_t
    inv_D = 1.0 / maxwell_stefan_diffusivities(x, D0)

    B = numpy.empty((n, n))
    for i in range(n):
        B[i, i] = x[i] * inv_D[i, N - 1] + sum(
            x[k] * inv_D[i, k] for k in range(N) if k != i)
        for j in range(n):
            if j != i:
                B[i, j] = -x[i] * (inv_D[i, j] - inv_D[i, N - 1])

    T = numpy.eye(n) - numpy.outer(c[:n], nu[:n] - nu[-1])
    X = (numpy.diag(1.0 / nu[:n]) / c_t
         - numpy.outer(c[:n] / c_t ** 2, 1.0 / nu[:n] - 1.0 / nu[-1]))
    return numpy.diag(nu[:n]) @ T @ (c_t * numpy.linalg.inv(B)) @ X
