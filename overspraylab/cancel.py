"""Cooperative cancellation of a computation.

A running thread cannot be killed: the long steps call `check_cancel()` between two operations and
the computation stops there. Outside `cancellable` (exports, previews) the check does nothing.
A long step added to the pipeline should call it, otherwise it delays the next computation.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable


class Cancelled(BaseException):
    """The computation was replaced. Not an `Exception`, so that no fallback handler swallows it."""


_IS_CANCELLED: ContextVar[Callable[[], bool] | None] = ContextVar("overspraylab_cancel", default=None)


@contextmanager
def cancellable(is_cancelled: Callable[[], bool] | None):
    token = _IS_CANCELLED.set(is_cancelled)
    try:
        yield
    finally:
        _IS_CANCELLED.reset(token)


def check_cancel() -> None:
    f = _IS_CANCELLED.get()
    if f is not None and f():
        raise Cancelled
