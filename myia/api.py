"""Myia's API functions."""

from .abstract.to_abstract import to_abstract
from .infer.infnode import infer_graph
from .parser import parse
from .utils.info import enable_debug


class CheckedFunction:
    """Decorates a function to turn on type checking with Myia."""

    def __init__(self, fn, args=(), graph=None):
        self.fn = fn
        self.args = args
        self.graph = graph
        if graph is None:
            with enable_debug():
                self.graph = parse(self.fn)
        self.cache = {}

    def __call__(self, *args):
        """Typecheck the function and then execute it."""
        args = (*self.args, *args)
        types = tuple(to_abstract(arg) for arg in args)
        result = self.cache.get(types, None)
        if result is None:
            try:
                with enable_debug():
                    _ = infer_graph(self.graph, types)
            except Exception as exc:
                result = exc
            else:
                result = True
            finally:
                self.cache[types] = result

        if result is True:
            return self.fn(*args)
        else:
            raise result

    def __myia__(self):
        """Return the underlying object for Myia to interpret."""
        return self.fn

    def __get__(self, obj, objtype):
        return CheckedFunction(
            fn=self.fn, args=(*self.args, obj), graph=self.graph
        )


checked = CheckedFunction
