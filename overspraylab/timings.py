"""Optional timing of a computation, function by function (preference "Log computation times").

The functions named in `instrument` are wrapped once; a wrapper only measures while a `Recorder`
is active in its context (`recording`), so the cost is nil when the option is off. The threads of
`pipeline.run_threads` and of the layer jobs copy the context, so they report to the same recorder;
other threads pass it on explicitly (`current` / `use`). Layer jobs run in separate processes
(OVERSPRAYLAB_PARALLEL=process) are timed as a whole only.
"""
from __future__ import annotations

import contextvars
import functools
import threading
import time
from contextlib import contextmanager

_REC: contextvars.ContextVar = contextvars.ContextVar("overspraylab_timings", default=None)


class Recorder:
    """Calls and summed wall time of each function, with the first start and the last end in
    seconds from the start of the computation. Times of nested functions are included in the
    time of the functions that call them; functions run in parallel add up."""

    def __init__(self):
        self.t0 = time.perf_counter()
        self.lock = threading.Lock()
        self.funcs: dict[str, list] = {}

    def add(self, name: str, start: float, end: float) -> None:
        with self.lock:
            f = self.funcs.get(name)
            if f is None:
                self.funcs[name] = [1, end - start, start - self.t0, end - self.t0]
            else:
                f[0] += 1
                f[1] += end - start
                f[2] = min(f[2], start - self.t0)
                f[3] = max(f[3], end - self.t0)

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def result(self) -> dict:
        with self.lock:
            items = sorted(self.funcs.items(), key=lambda kv: kv[1][2])
        return {n: {"calls": c, "total": round(s, 3), "from": round(a, 3), "to": round(b, 3)}
                for n, (c, s, a, b) in items}


def timed(name: str, f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        rec = _REC.get()
        if rec is None:
            return f(*args, **kwargs)
        t = time.perf_counter()
        try:
            return f(*args, **kwargs)
        finally:
            rec.add(name, t, time.perf_counter())
    return wrapper


def instrument(namespace: dict, names) -> None:
    """Wraps the functions `names` of a module namespace (module globals: the calls made inside the
    module and through `module.name` both go through the wrapper)."""
    for n in names:
        f = namespace.get(n)
        if callable(f) and not hasattr(f, "__wrapped__"):
            namespace[n] = timed(n, f)


def current() -> Recorder | None:
    return _REC.get()


@contextmanager
def use(rec: Recorder | None):
    """Reports to `rec` (taken with `current` in another thread) inside the block."""
    tok = _REC.set(rec)
    try:
        yield rec
    finally:
        _REC.reset(tok)


@contextmanager
def recording(on: bool):
    """A new Recorder for the block when `on`, else None (nothing measured)."""
    if not on:
        yield None
        return
    with use(Recorder()) as rec:
        yield rec
