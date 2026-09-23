"""Multicomponent Vignes diffusivities, inverted to a volume-frame Fick matrix."""

import numpy


def maxwell_stefan_diffusivities(x, D0):
    """Vignes diffusivities D_ij, (N, N), at mole fractions x.  Diagonal unused."""
    # D_ij = D0_ji^x_i D0_ij^x_j prod_{k != i,j} (D0_ik D0_jk)^(x_k/2)
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
    # j = -[D] grad phi ,  [D] = diag(nu) T c_t B^-1 X
    N = phi.size
    n = N - 1
    c = phi / nu
    c_t = c.sum()
    x = c / c_t
    inv_D = 1.0 / maxwell_stefan_diffusivities(x, D0)

    # B_ii = x_i/D_iN + sum_{k != i} x_k/D_ik ;  B_ij = -x_i (1/D_ij - 1/D_iN)
    B = numpy.empty((n, n))
    for i in range(n):
        B[i, i] = x[i] * inv_D[i, N - 1] + sum(
            x[k] * inv_D[i, k] for k in range(N) if k != i)
        for j in range(n):
            if j != i:
                B[i, j] = -x[i] * (inv_D[i, j] - inv_D[i, N - 1])

    # T_ik = delta_ik - c_i (nu_k - nu_N): molar frame -> volume frame
    T = numpy.eye(n) - numpy.outer(c[:n], nu[:n] - nu[-1])
    # X_jl = dx_j/dphi_l: grad x -> grad phi
    X = (numpy.diag(1.0 / nu[:n]) / c_t
         - numpy.outer(c[:n] / c_t ** 2, 1.0 / nu[:n] - 1.0 / nu[-1]))
    # [D] = diag(nu) T c_t B^-1 X, accumulated left to right
    fick = numpy.matmul(numpy.diag(nu[:n]), T)
    fick = numpy.matmul(fick, c_t * numpy.linalg.inv(B))
    return numpy.matmul(fick, X)
