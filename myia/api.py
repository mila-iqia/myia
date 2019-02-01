"""User-friendly interfaces to Myia machinery."""

import inspect

from .abstract import MyiaTypeError, from_value, broaden
from .pipeline import standard_pipeline
from .utils import as_frozen


#################
# Top-level API #
#################


class MyiaFunction:
    """Represents a function compiled by Myia.

    MyiaFunction will compile the original function for every combination of
    argument types and shapes it is given (as well as their values,
    optionally).

    Attributes:
        fn: The root function to compile.
        specialize_values: Set of arguments for which we should specialize the
            function based on their values (list of argument names).

    """

    def __init__(self, fn, specialize_values=[]):
        """Initialize a MyiaFunction."""
        self.fn = fn
        self.specialize_values = set(specialize_values)
        self._cache = {}
        # self.pip = standard_pipeline.make()

    def specialize(self, args):
        """Specialize on the types of the given arguments.

        Returns a Pipeline. If the argument types were seen before, returns a
        cached version.
        """
        self.pip = standard_pipeline.make()
        inf = self.pip.resources.inferrer

        argnames = inspect.getfullargspec(self.fn).args
        n1 = len(argnames)
        n2 = len(args)
        if n1 != n2:
            raise MyiaTypeError(
                f'Wrong number of arguments: expected {n1}, got {n2}'
            )

        argspec = tuple(from_value(arg) for arg in args)
        argspec = tuple(broaden(arg, None)
                         if name not in self.specialize_values else arg
                         for arg, name in zip(argspec, argnames))

        if argspec not in self._cache:
            self._cache[argspec] = self.pip(
                input=self.fn,
                argspec=argspec
            )
        return self._cache[argspec]

    def compile(self, args):
        """Returns a function specialized for the given args."""
        return self.specialize(args)['output']

    def __call__(self, *args):
        """Call the function on the given args."""
        return self.compile(args)(*args)


def myia(fn=None, *, specialize_values=[]):
    """Create a function using Myia's runtime.

    `@myia` can be used as a simple decorator. If custom options are needed,
    they can be provided as keyword arguments:

        @myia
        def myfun(x, y):
            return x + y

        @myia(specialize_values=['cond'])
        def myfun2(cond, x, y):
            return x if cond else y

    Arguments:
        fn: The Python function to convert.
        specialize_values: Set of arguments for which we should specialize the
            function based on their values (list of argument names).
    """
    if fn is None:
        def deco(fn):
            return MyiaFunction(fn, specialize_values)
        return deco
    else:
        return MyiaFunction(fn, specialize_values)
