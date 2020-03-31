"""User-friendly interfaces to Myia machinery."""

import inspect
import os

from .abstract import ABSENT, find_aliases, from_value
from .compile.backends import Backend, load_backend
from .compile.utils import BackendValue
from .pipeline import standard_pipeline
from .simplify_types import to_canonical
from .utils import (
    MultiTrace,
    MyiaInputTypeError,
    MyiaTypeError,
    keyword_decorator,
    resolve_tracers,
)

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

    def __init__(
        self,
        fn,
        specialize_values=[],
        return_backend=False,
        backend=None,
        backend_options=None,
        alias_tracker=None,
        use_universe=False,
        tracer=ABSENT,
        pipeline=standard_pipeline,
    ):
        """Initialize a MyiaFunction."""
        # Change this once relay becomes the default backend.
        if use_universe and backend != "relay":
            raise RuntimeError(  # pragma: no cover
                "Universe is only supported for the relay backend."
            )
        self.fn = fn
        self.alias_tracker = alias_tracker
        self.specialize_values = set(specialize_values)
        self.pip = pipeline.configure(
            {
                "resources.universal": use_universe,
                "resources.backend.name": backend,
                "resources.backend.options": backend_options,
                "resources.return_backend": return_backend,
            }
        )
        self._cache = {}
        self.latest = None
        if tracer is ABSENT:
            mt = os.environ.get("MYIATRACER")
            if mt:
                pairs = resolve_tracers(mt)
                tracers = [fn(*args) for fn, args in pairs]
                tracer = MultiTrace(*tracers)
            else:
                tracer = None
        self.tracer = tracer

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
                f"Wrong number of arguments: expected {n1}, got {n2}"
            )

        alias_map, aid_to_paths = find_aliases(args, self.alias_tracker)
        argspec = tuple(
            from_value(
                arg,
                broaden=name not in self.specialize_values,
                alias_map=alias_map,
            )
            for arg, name in zip(args, argnames)
        )

        if argspec not in self._cache:
            if self.tracer:
                self.tracer.__enter__()
            self._cache[argspec] = self.pip.run(
                input=self.fn,
                argspec=argspec,
                aliasspec=(self.alias_tracker, aid_to_paths),
            )
            if self.tracer:
                self.tracer.__exit__(None, None, None)
        return self._cache[argspec]

    def compile(self, args):
        """Returns a function specialized for the given args."""
        self.latest = self.specialize(args)["output"]
        return self.latest

    def __call__(self, *args):
        """Call the function on the given args."""
        if self.latest:
            try:
                return self.latest(*args)
            except MyiaInputTypeError:
                pass
        return self.compile(args)(*args)

    def to_device(self, v, *, broaden=True, vm_t=None, orig_t=None):
        """Move value to the function's accelerator hardware."""
        backr = self.pip.steps["resources"].keywords["backend"].keywords
        return to_device(
            v,
            backr["name"],
            backr["options"],
            broaden=broaden,
            vm_t=vm_t,
            orig_t=orig_t,
        )


@keyword_decorator
def myia(
    fn,
    *,
    specialize_values=[],
    backend=None,
    backend_options=None,
    return_backend=False,
    alias_tracker=None,
    use_universe=False,
    pipeline=standard_pipeline,
):
    """Create a function using Myia's runtime.

    `@myia` can be used as a simple decorator. If custom options are needed,
    they can be provided as keyword arguments::

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
        alias_tracker: function to indicate aliases
            (see :func:`myia.abstract.find_aliases`)
        use_universe: Enable use of sequential code (experimental)

    """
    return MyiaFunction(
        fn,
        specialize_values,
        backend=backend,
        backend_options=backend_options,
        return_backend=return_backend,
        alias_tracker=alias_tracker,
        use_universe=use_universe,
        pipeline=pipeline,
    )


#############################################
# Move value to target accelerator hardware #
#############################################


def to_device(
    value,
    backend,
    backend_options=None,
    *,
    broaden=True,
    orig_t=None,
    vm_t=None,
):
    """Move value to target accelerator hardware (using selected backend)."""
    if not isinstance(backend, Backend):
        backend = load_backend(backend, backend_options)
    if orig_t is None:
        orig_t = from_value(value, broaden=broaden)
    value = to_canonical(value, orig_t)
    if vm_t is None:
        vm_t = from_value(value, broaden=broaden)
    value = backend.to_backend_value(value, vm_t)
    return BackendValue(value, orig_t, vm_t, backend)


__all__ = ["MyiaFunction", "myia", "to_device"]
