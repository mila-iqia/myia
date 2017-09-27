
from typing import Tuple as TupleT
import inspect
import textwrap
import ast
from .parse import Parser, Locator, parse_function
from .stx import Symbol, _Assign
from .interpret import evaluate


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
        ctrl = options.get('controller', None)
        self.lbda = parse_function(fn)
        self.fn = evaluate(self.lbda, controller=ctrl)

    def __call__(self, *args):
        return self.fn(*args)

    def configure(self, **config):
        return self.fn.configure(**config)


def myia(fn, **options):
    return MyiaFunction(fn=fn, **options)
