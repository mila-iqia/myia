
import os
from io import StringIO

from myia.debug import traceback as myia_tr
from myia.info import DebugInherit
from myia.parser import MyiaSyntaxError
from myia.utils import InferenceError

if os.environ.get('BUCHE'):
    from debug import do_inject  # noqa
    from debug.butest import *  # noqa


_tracer_class = None
_trace_nodes = False


def pytest_addoption(parser):
    parser.addoption('--gpu', action='store_true', dest="gpu",
                     default=False, help="enable gpu tests")
    parser.addoption('-T', action='store', dest="tracer",
                     default=None, help="set a Myia tracer")
    parser.addoption('--trace-nodes', action='store_true', dest="trace_nodes",
                     default=False, help="save trace when creating Myia nodes")


def pytest_configure(config):
    global _tracer_class
    global _trace_nodes
    if not config.option.gpu:
        setattr(config.option, 'markexpr', 'not gpu')
    if config.option.tracer:
        modname, field = config.option.tracer.rsplit('.', 1)
        mod = __import__(modname, fromlist=[field])
        _tracer_class = getattr(mod, field)
    if config.option.trace_nodes:
        _trace_nodes = DebugInherit(save_trace=True)


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
        if not hasattr(typ, '_repr_failure'):
            typ._repr_failure = typ.repr_failure
            typ.repr_failure = myia_repr_failure


# That hook may have been defined in debug.butest
_prev = globals().get('pytest_runtest_setup', None)


def pytest_runtest_setup(item):
    if _prev is not None:
        _prev(item)
    if _tracer_class:
        item._tracer = _tracer_class()
        item._tracer.__enter__()
    if _trace_nodes:
        _trace_nodes.__enter__()


def pytest_runtest_teardown(item):
    if hasattr(item, '_tracer'):
        item._tracer.__exit__(None, None, None)
    if _trace_nodes:
        _trace_nodes.__exit__()
