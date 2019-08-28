
import os
from io import StringIO

from myia.debug import traceback as myia_tr
from myia.parser import MyiaSyntaxError
from myia.utils import InferenceError

if os.environ.get('BUCHE'):
    from debug import do_inject  # noqa
    from debug.butest import *  # noqa


def pytest_addoption(parser):
    parser.addoption('--gpu', action='store_true', dest="gpu",
                     default=False, help="enable gpu tests")


def pytest_configure(config):
    if not config.option.gpu:
        setattr(config.option, 'markexpr', 'not gpu')


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
