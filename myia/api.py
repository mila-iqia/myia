"""User-friendly interfaces to Myia machinery."""

import numpy as np
import inspect

from myia import dtype

from .abstract import MyiaTypeError, from_value
from .pipeline import standard_pipeline
from .utils import keyword_decorator, overload
from .abstract import TYPE, ArrayWrapper, AbstractTuple, \
    AbstractList, AbstractArray, AbstractScalar, AbstractClassBase
from .compile.backends import load_backend, Backend


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

    def __init__(self, fn, specialize_values=[], return_backend=False,
                 backend=None, backend_options=None):
        """Initialize a MyiaFunction."""
        self.fn = fn
        self.specialize_values = set(specialize_values)

        self.pip = standard_pipeline.configure({
        pip = standard_pipeline.configure({
            'wrap.return_backend': return_backend
        })
        backend = load_backend(backend, backend_options)
        self.pip = backend.configure(pip)
        self._cache = {}

    def specialize(self, args):
        """Specialize on the types of the given arguments.

        Returns a Pipeline. If the argument types were seen before, returns a
        cached version.
        """
        argnames = inspect.getfullargspec(self.fn).args
        n1 = len(argnames)
        n2 = len(args)
        if n1 != n2:
            raise MyiaTypeError(
                f'Wrong number of arguments: expected {n1}, got {n2}'
            )

        argspec = tuple(
            from_value(
                arg,
                broaden=name not in self.specialize_values)
            for arg, name in zip(args, argnames)
        )

        if argspec not in self._cache:
            self._cache[argspec] = self.pip.run(
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


@keyword_decorator
def myia(fn, *, specialize_values=[], backend=None, backend_options=None,
         return_backend=False):
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
        backend: the backend to use for compilation
        backend_options: backend-specific options.
        return_backend: return backend values (avoids copies to CPU).
    """
    return MyiaFunction(fn, specialize_values, backend=backend,
                        backend_options=backend_options,
                        return_backend=return_backend)


######################################################################
# Converts args to initialize model on target accelerator hardware   #
# Note: not to be conflated with weight initialization distributions #
######################################################################

@overload(bootstrap=True)
def _convert_arg_init(self, arg, orig_t: AbstractTuple, backend):
    oe = orig_t.elements
    return tuple(self(x, o, backend) for x, o in zip(arg, oe))


@overload  # noqa: F811
def _convert_arg_init(self, arg, orig_t: AbstractList, backend):
    ot = orig_t.element
    return [self(x, ot, backend) for x in arg]


@overload  # noqa: F811
def _convert_arg_init(self, arg, orig_t: AbstractClassBase, backend):
    arg = tuple(getattr(arg, attr) for attr in orig_t.attributes)
    oe = list(orig_t.attributes.values())
    return orig_t.constructor(*(self(x, o, backend) for x, o in zip(arg, oe)))


@overload  # noqa: F811
def _convert_arg_init(self, arg, orig_t: AbstractArray, backend):
    et = orig_t.element
    assert isinstance(et, AbstractScalar)
    et = et.values[TYPE]
    assert issubclass(et, dtype.Number)
    if isinstance(arg, np.ndarray):
        arg = ArrayWrapper(
            backend.from_numpy(arg), arg.dtype, arg.shape, backend
        )
    return arg


@overload  # noqa: F811
def _convert_arg_init(self, arg, orig_t: AbstractScalar, backend):
    return arg


#############################################
# Move model to target accelerator hardware #
#############################################

def to_device(model, backend, backend_options=None):
    """Move model to target accelerator hardware (using selected backend)."""
    if not isinstance(backend, Backend):
        backend = load_backend(backend, backend_options)
    model = _convert_arg_init(
        model,
        from_value(model, broaden=True),
        backend
    )
    return model
