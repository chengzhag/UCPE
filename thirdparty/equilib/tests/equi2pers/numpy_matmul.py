#!/usr/bin/env python3

"""A couple matmul functions in numpy

Input:
- m: pixel grid
- G: "cam-to-world" rotation matrix
- R: batched rotation matrices as a tensor

Mid:
- C: array of size (batch, 3, 3)
- m: array of size (batch, height, width, 3, 1)

Output:
- M: array of size (batch, height, width, 3, 1)

"""

import numpy as np

from tests.equi2pers.numpy_prep import example
from tests.helpers.benchmarking import check_close, mae, mse
from tests.helpers.timer import time_func_loop

np.random.seed(0)

"""R, G
"""


def RG_baseline_v1(R, G):
    C = R @ G
    return C


def RG_einsum(R, G):
    C = np.einsum("bik,kj->bij", R, G, optimize=True)
    return C


"""C = R @ G, m
"""


def Cm_naive_v1(C, m):
    batch_size, height, width, _, _ = m.shape
    M = np.empty_like(m)
    for b in range(batch_size):
        for h in range(height):
            for w in range(width):
                for r in range(3):
                    M[b, h, w, r, ...] = (
                        C[b, r, 0] * m[b, h, w, 0, 0]
                        + C[b, r, 1] * m[b, h, w, 1, 0]
                        + C[b, r, 2] * m[b, h, w, 2, 0]
                    )
    return M


def Cm_naive_v2(C, m):
    batch_size, height, width, _, _ = m.shape
    M = np.empty_like(m)
    for b in range(batch_size):
        for h in range(height):
            for w in range(width):
                M[b, h, w, ...] = C[b, ...] @ m[b, h, w, ...]
    return M


def Cm_baseline_v1(C, m):
    M = np.matmul(C[:, np.newaxis, np.newaxis, ...], m)
    return M


def Cm_baseline_v2(C, m):
    M = C[:, np.newaxis, np.newaxis, ...] @ m
    return M


def Cm_einsum_v1(C, m):
    """Testing out einsum, but this method is the slowest, but the closest to baseline"""
    M = np.einsum("bik,bhwkj->bhwij", C, m, optimize=True)
    # M = np.einsum("bik,b...kj->b...ij", C, m, optimize=True)
    return M


def Cm_einsum_v2(C, m):
    """This method seems like the fastest, with a little loss of *accuracy*"""
    batch_size = m.shape[0]
    M = np.empty_like(m)
    for b in range(batch_size):
        M[b, ...] = np.einsum(
            "ik,...kj->...ij", C[b, ...], m[b, ...], optimize=True
        )
    return M


def Cm_loop_batch_v1(C, m):
    batch_size, height, width, _, _ = m.shape
    M = np.empty_like(m)
    for b in range(batch_size):
        M[b, ...] = np.matmul(
            C[b, ...], m[b, ...].reshape(height * width, 3, 1)
        ).reshape(height, width, 3, 1)
    return M


def Cm_loop_batch_v2(C, m):
    batch_size = m.shape[0]
    M = np.empty_like(m)
    for b in range(batch_size):
        M[b, ...] = C[b, ...] @ m[b, ...]
    return M


def Cm_trasposed(C, m):
    # FIXME: need to trasnpose the arrays, might take up some time
    CT = C.transpose((0, 2, 1)).copy()
    mT = m.transpose((0, 1, 2, 4, 3)).copy()

    MT = mT @ CT[:, np.newaxis, np.newaxis, ...]
    M = MT.transpose((0, 1, 2, 4, 3)).copy()
    return M


"""R, G, m
"""


def naive_v1(R, G, m):
    batch_size, height, width, _, _ = m.shape
    M = np.empty_like(m)
    for b in range(batch_size):
        for h in range(height):
            for w in range(width):
                M[b, h, w, ...] = R[b, ...] @ G @ m[b, h, w, ...]
    return M


def baseline_v1(R, G, m):
    """Slower than np.matmul"""
    M = R[:, np.newaxis, np.newaxis, ...] @ G @ m
    return M


def baseline_v2(R, G, m):
    """Fastest out of basic numpy function"""
    M = np.matmul(np.matmul(R, G)[:, np.newaxis, np.newaxis, ...], m)
    return M


def baseline_v3(R, G, m):
    """Over all, slower than v2"""
    batch_size = m.shape[0]
    M = np.empty_like(m)
    for b in range(batch_size):
        M[b, ...] = np.matmul(np.matmul(R[b, ...], G), m[b, ...])
    return M


def einsum_v1(R, G, m):
    """Becomes slower when batch size is high, always slower than v2"""
    M = np.einsum("bik,kj,bhwjl->bhwil", R, G, m, optimize=True)
    return M


def einsum_v2(R, G, m):
    """Slower than v3 but way faster than v1"""
    batch_size = m.shape[0]
    M = np.empty_like(m)
    for b in range(batch_size):
        M[b, ...] = np.einsum(
            "ik,kj,...jl->...il", R[b, ...], G, m[b, ...], optimize=True
        )
    return M


def einsum_v3(R, G, m):
    """Fastest!!!!!!"""
    C = np.einsum("bik,kj->bij", R, G, optimize=True)
    batch_size = m.shape[0]
    M = np.empty_like(m)
    for b in range(batch_size):
        M[b, ...] = np.einsum(
            "ik,...kj->...ij", C[b, ...], m[b, ...], optimize=True
        )
    return M


"""Benchmarking

"""


def gen_rand_rots(batch: int):
    rots = []
    for b in range(batch):
        # create random rots in radians
        rot = {
            "roll": 2 * np.pi * np.random.random_sample() - np.pi,
            "pitch": np.pi * np.random.random_sample() - np.pi / 2,
            "yaw": 2 * np.pi * np.random.random_sample() - np.pi,
        }
        rots.append(rot)
    return rots


DATA = [
    {
        "rots": gen_rand_rots(1),
        "height": 256,
        "width": 512,
        "batch": 1,
        "dtype": np.dtype(np.float32),
    },
    {
        "rots": gen_rand_rots(4),
        "height": 32,
        "width": 64,
        "batch": 4,
        "dtype": np.dtype(np.float32),
    },
    {
        "rots": gen_rand_rots(4),
        "height": 64,
        "width": 128,
        "batch": 4,
        "dtype": np.dtype(np.float32),
    },
    {
        "rots": gen_rand_rots(4),
        "height": 64,
        "width": 128,
        "batch": 4,
        "dtype": np.dtype(np.float64),
    },
    {
        "rots": gen_rand_rots(4),
        "height": 128,
        "width": 256,
        "batch": 4,
        "dtype": np.dtype(np.float32),
    },
    {
        "rots": gen_rand_rots(4),
        "height": 256,
        "width": 512,
        "batch": 4,
        "dtype": np.dtype(np.float32),
    },
    {
        "rots": gen_rand_rots(16),
        "height": 256,
        "width": 512,
        "batch": 16,
        "dtype": np.dtype(np.float32),
    },
    # {
    #     "rots": gen_rand_rots(32),
    #     "height": 256,
    #     "width": 512,
    #     "batch": 32,
    #     "dtype": np.dtype(np.float32)
    # },
]


def bench_time():
    num = 100

    test_functions = [
        # naive_v1,
        baseline_v1,
        baseline_v2,
        baseline_v3,
        einsum_v1,
        einsum_v2,
        einsum_v3,
    ]

    for i, data in enumerate(DATA):
        print()
        print(f"TEST {i+1}")
        print(f"batch size: {data['batch']}")
        print(
            f"height/width/type: {data['height']}/{data['width']}/{data['dtype']}"
        )

        # make data
        m, G, R = example(**data)

        args = {"R": R, "G": G, "m": m}

        for func in test_functions:
            time_func_loop(func=func, func_args=args, num=num)


def compare_accuracy():
    data = DATA[3]
    m, G, R = example(**data)
    args = {"R": R, "G": G, "m": m}

    GT = naive_v1(**args)
    einsum = einsum_v3(**args)
    matmul = baseline_v2(**args)

    print("GT vs einsum")
    print("are close?", check_close(GT, einsum))
    print("are equal?", check_close(GT, einsum))
    print("MSE:", mse(GT, einsum))
    print("MAE:", mae(GT, einsum))

    print("GT vs matmul")
    print("are close?", check_close(GT, matmul))
    print("are equal?", check_close(GT, matmul))
    print("MSE:", mse(GT, matmul))
    print("MAE:", mae(GT, matmul))


"""Results:

## Time

```
TEST 1
batch size: 1
height/width/type: 256/512/float32
Func: baseline_v1       0.003471
Func: baseline_v2       0.003107
Func: baseline_v3       0.003196
Func: einsum_v1 0.001808
Func: einsum_v2 0.001583
Func: einsum_v3 0.001078

TEST 2
batch size: 4
height/width/type: 32/64/float32
Func: baseline_v1       0.000351
Func: baseline_v2       0.000347
Func: baseline_v3       0.000221
Func: einsum_v1 0.000734
Func: einsum_v2 0.000558
Func: einsum_v3 0.000352

TEST 3
batch size: 4
height/width/type: 64/128/float32
Func: baseline_v1       0.000792
Func: baseline_v2       0.000789
Func: baseline_v3       0.000815
Func: einsum_v1 0.002518
Func: einsum_v2 0.000684
Func: einsum_v3 0.000450

TEST 4
batch size: 4
height/width/type: 64/128/float64
Func: baseline_v1       0.000796
Func: baseline_v2       0.000795
Func: baseline_v3       0.000823
Func: einsum_v1 0.002524
Func: einsum_v2 0.000731
Func: einsum_v3 0.000489

TEST 5
batch size: 4
height/width/type: 128/256/float32
Func: baseline_v1       0.003117
Func: baseline_v2       0.003087
Func: baseline_v3       0.003159
Func: einsum_v1 0.009830
Func: einsum_v2 0.001720
Func: einsum_v3 0.000911

TEST 6
batch size: 4
height/width/type: 256/512/float32
Func: baseline_v1       0.012879
Func: baseline_v2       0.012377
Func: baseline_v3       0.012704
Func: einsum_v1 0.039047
Func: einsum_v2 0.005223
Func: einsum_v3 0.001276

TEST 7
batch size: 16
height/width/type: 256/512/float32
Func: baseline_v1       0.050168
Func: baseline_v2       0.049829
Func: baseline_v3       0.050522
Func: einsum_v1 0.156020
Func: einsum_v2 0.024179
Func: einsum_v3 0.006772
```

## Accuracy:

```
GT vs einsum
are close? True
are equal? True
MSE: 6.343245421748295e-30
MAE: 9.819691956327718e-16

GT vs matmul
are close? True
are equal? True
MSE: 0.0
MAE: 0.0
```

"""


if __name__ == "__main__":
    bench_time()
    compare_accuracy()
