#!/usr/bin/env python3

"""Profilers for execution time"""

from functools import partial, update_wrapper
import time
from typing import Any, Callable, Dict

__all__ = ["func_timer", "printable_time", "wrapped_partial"]


def wrapped_partial(
    func: Callable[[], Any], *args, **kwargs
) -> Callable[[], Any]:
    """Useful wrapper for creating partials

    It preserves `__name__` that's needed in `func_timer`
    """
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


def printable_time(name: str, time: float, prec: int = 6) -> str:
    # FIXME: cleaner way of specifying precision?
    # base = "Func:{}:\t{:0." + str(prec) + "f}"
    # s = base.format(name, time)
    return f"Func: {name}\t" + format(time, f".{str(prec)}f")


def func_timer(
    func: Callable[[], Any],
    *,
    ret: bool = False,
    verbose: bool = True,
    prec: int = 6,
) -> Callable[[], Any]:
    """Decorator that reports the execution time when called

    params:
    - func (def/method): function (passby)
    - ret (bool): returns the time in float
    - verbose (bool): prints time with precision
    - prec (int): precison for verbose

    return:
    - results (Any): return value(s) from the function

    If `ret=True`, the function will return the time in float as
    a tuple `(result, time)`.
    """

    def wrap(*args, **kwargs):
        tic = time.perf_counter()
        results = func(*args, **kwargs)
        toc = time.perf_counter()
        if verbose:
            print(printable_time(func.__name__, toc - tic, prec=prec))
        if ret:
            return results, toc - tic
        return results

    return wrap


def time_func_loop(
    func: Callable[[], Any],
    func_args: Dict[str, Any],
    *,
    num: int = 100,
    prec: int = 6,
) -> None:
    times = []
    name = func.__name__
    timed_func = func_timer(func, ret=True, verbose=False)

    for _ in range(num):
        _, t = timed_func(**func_args)
        times.append(t)

    avg = sum(times) / num
    print(printable_time(name, avg, prec=prec))
