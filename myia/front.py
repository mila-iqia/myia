
from typing import Tuple as TupleT
import inspect
import textwrap
import ast
from .parse import Parser, Locator, get_global_parse_env, parse_function
from .stx import Symbol, ParseEnv, _Assign, gsym
from .interpret import evaluate2, wrap_globals
from .lib import Pending


# def make_error_function(data):
#     def _f(*args, **kwargs):
#         raise Exception(
#             f"Function {data['name']} is for internal use only."
#         )
#     _f.data = data
#     return _f


# def myia(fn):
#     # This is probably broken.
#     _, genv = parse_function(fn)
#     gbindings = genv.bindings
#     glob = fn.__globals__
#     bdings = {k: make_error_function({"name": k, "ast": v, "globals": glob})
#               for k, v in gbindings.items()}
#     glob.update(bindings)
#     fsym = Symbol(fn.__name__, namespace='global')
#     fn.data = bindings[fsym].data
#     fn.associates = bindings
#     return fn


class MyiaFunction:
    def __init__(self, **options):
        fn = options['fn']
        sym, genv = parse_function(fn)
        glob = fn.__globals__
        self.globals = wrap_globals(glob)
        for k in genv.bindings.keys():
            print(k.namespace)
        self.lbda = genv[sym]
        self.fn = evaluate2(self.lbda, self.globals)

    def __call__(self, *args):
        return self.fn(*args)


def myia(fn):
    return MyiaFunction(fn=fn)
