import os
from io import StringIO

import pytest

from myia.debug import traceback as myia_tr
from myia.info import DebugInherit
from myia.parser import MyiaSyntaxError
from myia.utils import InferenceError, MultiTrace, Profiler, resolve_tracers

if os.environ.get("BUCHE"):
    from debug import do_inject  # noqa
    from debug.butest import *  # noqa


_context_managers = []


def pytest_addoption(parser):
    parser.addoption(
        "--gpu",
        action="store_true",
        dest="gpu",
        default=False,
        help="Enable GPU tests",
    )
    parser.addoption(
        "-T",
        action="append",
        dest="tracer",
        default=None,
        help="Set a Myia tracer",
    )
    parser.addoption(
        "--mprof",
        action="store_true",
        dest="mprof",
        default=False,
        help="Use the Myia profiler",
    )
    parser.addoption(
        "--trace-nodes",
        action="store_true",
        dest="trace_nodes",
        default=False,
        help="Save trace when creating Myia nodes",
    )
    parser.addoption(
        "-D",
        action="store_true",
        dest="do_inject",
        default=False,
        help="Import Myia debug functions",
    )


def pytest_configure(config):
    listener_pairs = []
    if config.option.usepdb:
        os.environ["MYIA_PYTEST_USE_PDB"] = "1"
    if config.option.do_inject:
        from debug import do_inject  # noqa
    if config.option.tracer:
        for tracer in config.option.tracer:
            listener_pairs += resolve_tracers(tracer)
    if config.option.mprof:
        listener_pairs.append((Profiler, ()))
    if config.option.trace_nodes:
        _context_managers.append(((lambda: DebugInherit(save_trace=True)), ()))

    def _make_trace_listeners():
        listeners = [fn(*args) for fn, args in listener_pairs]
        return MultiTrace(*listeners)

    if listener_pairs:
        _context_managers.append((_make_trace_listeners, ()))


class StringIOTTY(StringIO):
    def isatty(self):
        return True


def myia_repr_failure(self, excinfo):
    exc = excinfo.value
    if isinstance(exc, InferenceError):
        s = StringIOTTY()
        myia_tr.print_inference_error(exc, file=s)
        return s.getvalue()
    elif isinstance(exc, MyiaSyntaxError):
        s = StringIOTTY()
        myia_tr.print_myia_syntax_error(exc, file=s)
        return s.getvalue()
    else:
        return self._repr_failure(excinfo)


def pytest_collection_modifyitems(config, items):
    # Here we replace repr_failure on all the items so that they display a nice
    # error on InferenceError. This is very hacky but it works and I don't have
    # any more time to waste with pytest's nonsense.
    for item in items:
        typ = type(item)
        if not hasattr(typ, "_repr_failure"):
            typ._repr_failure = typ.repr_failure
            typ.repr_failure = myia_repr_failure


# That hook may have been defined in debug.butest
_prev = globals().get("pytest_runtest_setup", None)


def pytest_runtest_setup(item):
    if _prev is not None:
        _prev(item)

    gpu = any(mark for mark in item.iter_markers(name="gpu"))
    if gpu and not item.config.option.gpu:
        pytest.skip("GPU tests are not enabled. Use --gpu to enable them.")

    item._ctxms = [fn(*args) for fn, args in _context_managers]
    for cm in item._ctxms:
        cm.__enter__()


def pytest_runtest_teardown(item):
    if hasattr(item, "_ctxms"):
        for cm in item._ctxms:
            cm.__exit__(None, None, None)
