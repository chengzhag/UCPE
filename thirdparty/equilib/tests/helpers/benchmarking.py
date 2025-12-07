#!/usr/bin/env python3

"""Useful benchmarking functions"""

from typing import Any, Dict, Union, Callable

import numpy as np

import torch

from .timer import func_timer, printable_time

# aliases
ArrayLike = Union[np.ndarray, torch.Tensor]


def _check_arrays(a1: ArrayLike, a2: ArrayLike) -> bool:
    return type(a1) == type(a2)


def mse(a1: ArrayLike, a2: ArrayLike) -> float:
    """Mean square error"""
    assert _check_arrays(a1, a2)
    if isinstance(a1, np.ndarray):
        e = ((a1 - a2) ** 2).mean(axis=None)
    elif torch.is_tensor(a1):
        e = torch.mean((a1 - a2) ** 2)
    else:
        raise ValueError
    return float(e)


def mae(a1: ArrayLike, a2: ArrayLike) -> float:
    """Mean absolute error"""
    assert _check_arrays(a1, a2)
    if isinstance(a1, np.ndarray):
        e = abs(a1 - a2).mean(axis=None)
    elif torch.is_tensor(a1):
        e = torch.mean(torch.abs(a1 - a2))
    else:
        raise ValueError
    return float(e)


def check_equal(a1: ArrayLike, a2: ArrayLike) -> bool:
    """Check if the arrays are equal"""
    assert _check_arrays(a1, a2)
    if isinstance(a1, np.ndarray):
        return np.array_equal(a1, a2)
    elif torch.is_tensor(a1):
        return torch.equal(a1, a2)
    else:
        raise ValueError


def check_close(
    a1: ArrayLike,
    a2: ArrayLike,
    rtol: float = 1e-05,  # 1e-04
    atol: float = 1e-08,  # 1e-06
) -> bool:
    """Check if the arrays are close

    | a1 - a2 | <= atol + rtol * | a2 |

    FIXME: setup a decent checker using tolerance
    - generally want to use `rtol` since the precision of numbers
      calculations is very much finite, larger numbers will always
      be less precise than smaller ones
    - the only tim eyou use `atol` is for numbers that are so close
      to zsero that rounding erros are liable to be larger than the
      number itself
    """
    assert _check_arrays(a1, a2)
    if isinstance(a1, np.ndarray):
        return np.allclose(a1, a2, rtol=rtol, atol=atol)
    elif torch.is_tensor(a1):
        return torch.allclose(a1, a2, rtol=rtol, atol=atol)
    else:
        raise ValueError


def how_many_closes(
    a1: ArrayLike, a2: ArrayLike, rtol: float = 1e-05, atol: float = 1e-08
) -> float:
    """Returns ratio of 'closes' (counted by `isclose`)"""
    assert _check_arrays(a1, a2)
    if isinstance(a1, np.ndarray):
        return np.mean(np.isclose(a1, a2, rtol=rtol, atol=atol))
    elif torch.is_tensor(a1):
        return float(
            torch.mean(
                torch.isclose(a1, a2, rtol=rtol, atol=atol).type(torch.float32)
            )
        )
    else:
        raise ValueError


def check_similar(a1: ArrayLike, a2: ArrayLike, tol: float = 1e-08) -> bool:
    """When a1 and a2 are close to zero"""
    assert _check_arrays(a1, a2)
    d = a1 - a2
    if isinstance(a1, np.ndarray):
        z = np.zeros_like(d)
        return np.allclose(d, z, atol=tol)
    elif torch.is_tensor(a1):
        z = torch.zeros_like(d)
        return torch.allclose(d, z, atol=tol)
    else:
        raise ValueError


def compare_methods(
    func1: Callable[[], Any],
    func2: Callable[[], Any],
    data_func: Callable[[], Any],
    data_kwargs: Dict[str, Any],
) -> None:
    """Compare Performances of the two sampling functions"""
    func1_name = func1.__name__
    func2_name = func2.__name__
    func1 = func_timer(func1, ret=True, verbose=False)
    func2 = func_timer(func2, ret=True, verbose=False)

    # run the methods
    data = data_func(**data_kwargs)
    out1, time1 = func1(data)
    out2, time2 = func2(data)

    # compare results
    time1 = printable_time(func1_name, time1)
    time2 = printable_time(func2_name, time2)
    print("are close?", check_close(out1, out2))
    print("are equal?", check_close(out1, out2))
    print("MSE:", mse(out1, out2))
    print("MAE:", mae(out1, out2))
