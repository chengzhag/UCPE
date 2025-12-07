#!/usr/bin/env python3

"""Some codes to figure out if the 3x3 matrix inverse could be optimized
"""

from timeit import timeit

from numba import jit

import numpy as np

# from tests.helpers.benchmarking import check_close, mae, mse
# from tests.helpers.timer import time_func_loop


# Helper function for determinant
def vdet(A):
    detA = np.zeros_like(A[0, 0])
    detA = (
        A[0, 0] * (A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1])
        - A[0, 1] * (A[2, 2] * A[1, 0] - A[2, 0] * A[1, 2])
        + A[0, 2] * (A[1, 0] * A[2, 1] - A[2, 0] * A[1, 1])
    )
    return detA


# Another function for computing inverse
# using the Cayley-Hamilton corollary
def finv(A):
    detA = vdet(A)
    I1 = np.einsum("ii...", A)
    I2 = -0.5 * (np.einsum("ik...,ki...", A, A) - I1**2)
    Asq = np.einsum("ik...,kj...->ij...", A, A)
    eye = np.zeros_like(A)
    eye[0, 0] = 1.0
    eye[1, 1] = 1.0
    eye[2, 2] = 1.0
    return 1.0 / detA * (Asq - I1 * A + I2 * eye)


# Hard coded inverse
def hdinv(A):
    invA = np.zeros_like(A)
    detA = vdet(A)

    invA[0, 0] = (-A[1, 2] * A[2, 1] + A[1, 1] * A[2, 2]) / detA
    invA[1, 0] = (A[1, 2] * A[2, 0] - A[1, 0] * A[2, 2]) / detA
    invA[2, 0] = (-A[1, 1] * A[2, 0] + A[1, 0] * A[2, 1]) / detA
    invA[0, 1] = (A[0, 2] * A[2, 1] - A[0, 1] * A[2, 2]) / detA
    invA[1, 1] = (-A[0, 2] * A[2, 0] + A[0, 0] * A[2, 2]) / detA
    invA[2, 1] = (A[0, 1] * A[2, 0] - A[0, 0] * A[2, 1]) / detA
    invA[0, 2] = (-A[0, 2] * A[1, 1] + A[0, 1] * A[1, 2]) / detA
    invA[1, 2] = (A[0, 2] * A[1, 0] - A[0, 0] * A[1, 2]) / detA
    invA[2, 2] = (-A[0, 1] * A[1, 0] + A[0, 0] * A[1, 1]) / detA
    return invA


@jit("float64[:,:](float64[:,:])", cache=True, nopython=True, nogil=True)
def fast_inverse(A):
    inv = np.empty_like(A)
    a = A[0, 0]
    b = A[0, 1]
    c = A[0, 2]
    d = A[1, 0]
    e = A[1, 1]
    f = A[1, 2]
    g = A[2, 0]
    h = A[2, 1]
    i = A[2, 2]

    inv[0, 0] = e * i - f * h
    inv[1, 0] = -(d * i - f * g)
    inv[2, 0] = d * h - e * g
    inv_det = 1 / (a * inv[0, 0] + b * inv[1, 0] + c * inv[2, 0])

    inv[0, 0] *= inv_det
    inv[0, 1] = -inv_det * (b * i - c * h)
    inv[0, 2] = inv_det * (b * f - c * e)
    inv[1, 0] *= inv_det
    inv[1, 1] = inv_det * (a * i - c * g)
    inv[1, 2] = -inv_det * (a * f - c * d)
    inv[2, 0] *= inv_det
    inv[2, 1] = -inv_det * (a * h - b * g)
    inv[2, 2] = inv_det * (a * e - b * d)
    return inv


@jit(cache=True, nopython=True, nogil=True)
def vecinv(A):
    invA = np.zeros_like(A)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            invA[i, j, :, :] = fast_inverse(A[i, j, :, :])
    return invA


if __name__ == "__main__":
    from numpy.linalg import inv as npinv  # noqa

    np.random.seed(0)

    F = np.random.random((3, 3, 1000, 4))
    F2 = np.einsum("ij...->...ij", F)

    print("single")
    print(
        "npinv", timeit("npinv(F2[0, 0, :, :])", globals=globals(), number=100)
    )
    print(
        "hdinv", timeit("hdinv(F[:, :, 0, 0])", globals=globals(), number=1000)
    )
    # for single, npinv is 500 times faster

    print("batch")
    print("npinv", timeit("npinv(F2)", globals=globals(), number=100))
    print("hdinv", timeit("hdinv(F)", globals=globals(), number=100))
    print("vecinv", timeit("vecinv(F2)", globals=globals(), number=100))
    # for batch hdinv is 70 times faster

    # %timeit hdinv(F) # 371 µs ± 27 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
    # %timeit finv(F) # 5.35 ms ± 661 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    # %timeit npinv(F2) # 79.2 ms ± 12.9 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
