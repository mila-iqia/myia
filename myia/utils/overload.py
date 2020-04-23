"""Utilities to overload functions for multiple types."""


import inspect
from types import FunctionType

from .misc import MISSING, keyword_decorator


class TypeMap(dict):
    """Map types to handlers or values.

    Mapping a type to a value also (lazily) maps all of its subclasses to the
    same value, unless they have a mapping of their own.

    TypeMap should ideally not be updated after it is used, because updates may
    make some cached associations invalid.
    """

    def register(self, *obj_ts):
        """Decorator to register a handler to the given types."""

        def deco(handler):
            for obj_t in obj_ts:
                self[obj_t] = handler
            return handler

        return deco

    def __missing__(self, obj_t):
        """Get the handler for the given type."""
        handler = None
        to_set = []

        for cls in type.mro(obj_t):
            handler = super().get(cls, None)
            if handler is not None:
                for cls2 in to_set:
                    self[cls2] = handler
                break
            to_set.append(cls)

        if handler is not None:
            return handler
        elif hasattr(self, "_key_error"):
            raise self._key_error(obj_t)
        else:  # pragma: no cover
            raise KeyError(obj_t)


def _fresh(t):
    return type(t.__name__, (t,), {})


class Overload:
    """Overloaded function.

    A function can be added with the `register` method. One of its parameters
    should be annotated with a type, but only one, and every registered
    function should annotate the same parameter.

    Arguments:
        bootstrap: Whether to bind the first argument to the OverloadCall
            object. Forced to True if initial_state or postprocess is not
            None.
        bind_to: Binds the first argument to the given object.
        wrapper: A function to use as the entry point. In addition to all
            normal arguments, it will receive as its first argument the
            function to dispatch to.
        initial_state: A function returning the initial state, or None if
            there is no state.
        postprocess: A function to call on the return value. It is not called
            after recursive calls.
        mixins: A list of Overload instances that contribute functions to this
            Overload.
        name: Optional name for the Overload. If not provided, it will be
            gotten automatically from the first registered function or wrapper.
    """

    def __init__(
        self,
        *,
        bootstrap=False,
        bind_to=None,
        wrapper=None,
        initial_state=None,
        postprocess=None,
        mixins=[],
        name=None,
    ):
        """Initialize an Overload."""
        self.bind_to = bind_to
        self._wrapper = wrapper
        self.state = None
        self.initial_state = initial_state
        self.postprocess = postprocess
        self.bootstrap = bool(
            bootstrap or self.initial_state or self.postprocess
        )
        self.name = name
        _map = {}
        self.which = None
        for mixin in mixins:
            if self.which is None:
                self.which = mixin.which
            else:
                assert mixin.which == self.which
            _map.update(mixin._uncached_map)
        self.map = TypeMap(_map)
        self.map._key_error = lambda key: TypeError(
            f"No overloaded method in {self} for {key}"
        )
        self._uncached_map = _map
        self.ocls = _fresh(OverloadCall)

    def _set_attrs_from(self, fn, wrapper=False):
        if self.name is None:
            self.name = f"{fn.__module__}.{fn.__qualname__}"
            self.__doc__ = fn.__doc__
            self.__name__ = fn.__name__
            self.__qualname__ = fn.__qualname__
            self.__module__ = fn.__module__
            sign = inspect.signature(fn)
            params = list(sign.parameters.values())
            if wrapper:
                params = params[1:]
            if self.bootstrap or self.bind_to is not None:
                params = params[1:]
            params = [
                p.replace(annotation=inspect.Parameter.empty) for p in params
            ]
            self.__signature__ = sign.replace(parameters=params)

    def compile(self):
        """Finalize this overload."""
        cls = type(self)
        if self.name is not None:
            name = self.__name__

            # Place __real_call__ into __call__, renamed as the entry function
            cls.__call__ = rename_function(cls.__real_call__, f"{name}.entry")

            # Use the proper dispatch function
            method_name = "__xcall"
            if self.bootstrap or self.bind_to:
                method_name += "_bind"
            if self._wrapper is not None:
                method_name += "_wrap"
            method_name += "__"
            callfn = getattr(self.ocls, method_name)
            self.ocls.__call__ = rename_function(callfn, f"{name}.dispatch")

            # Rename the wrapper
            if self._wrapper:
                self._wrapper = rename_function(
                    self._wrapper, f"{name}.wrapper"
                )

            # Rename the mapped functions
            self.map.update(
                {
                    t: rename_function(fn, f"{name}[{t.__name__}]")
                    for t, fn in self.map.items()
                }
            )
        else:
            cls.__call__ = cls.__real_call__

    def wrapper(self, wrapper):
        """Set a wrapper function."""
        if self._wrapper is not None:
            raise TypeError(f"wrapper for {self} is already set")
        self._wrapper = wrapper
        self._set_attrs_from(wrapper, wrapper=True)
        return self

    def register(self, fn):
        """Register a function."""
        ann = fn.__annotations__
        if len(ann) != 1:
            raise Exception("Only one parameter may be annotated.")
        argnames = inspect.getfullargspec(fn).args
        for i, name in enumerate(argnames):
            t = ann.get(name, None)
            if t is not None:
                if isinstance(t, str):
                    t = eval(t)
                if self.which is None:
                    self.which = i
                elif self.which != i:
                    raise Exception(
                        "Annotation must always be on same parameter"
                    )
                break

        ts = t if isinstance(t, tuple) else (t,)

        for t in ts:
            self.map[t] = fn
            self._uncached_map[t] = fn

        self._set_attrs_from(fn)
        return self

    def copy(self, wrapper=MISSING, initial_state=None, postprocess=None):
        """Create a copy of this Overload.

        New functions can be registered to the copy without affecting the
        original.
        """
        return _fresh(Overload)(
            bootstrap=self.bootstrap,
            bind_to=self.bind_to,
            wrapper=self._wrapper if wrapper is MISSING else wrapper,
            mixins=[self],
            initial_state=initial_state or self.initial_state,
            postprocess=postprocess or self.postprocess,
        )

    def variant(
        self, fn=None, *, wrapper=MISSING, initial_state=None, postprocess=None
    ):
        """Decorator to create a variant of this Overload.

        New functions can be registered to the variant without affecting the
        original.
        """
        ov = self.copy(wrapper, initial_state, postprocess)
        if fn is None:
            return ov.register
        else:
            ov.register(fn)
            return ov

    def __get__(self, obj, cls):
        return self.ocls(
            map=self.map,
            state=self.initial_state() if self.initial_state else None,
            which=self.which,
            wrapper=self._wrapper,
            bootstrap=self.bootstrap,
            bind_to=obj,
        )

    def __getitem__(self, t):
        assert not self.bootstrap and self.bind_to is None
        return self.map[t]

    def __call__(self, *args, **kwargs):
        """Compile the overloaded function and then call it."""
        self.compile()
        return self(*args, **kwargs)

    def __real_call__(self, *args, **kwargs):
        """Call the overloaded function."""
        ovc = self.__get__(self.bind_to, None)
        res = ovc(*args, **kwargs)
        if self.postprocess:
            res = self.postprocess(res)
        return res

    def __repr__(self):
        return f"<Overload {self.name or hex(id(self))}>"


class OverloadCall:
    """Context for an Overload call."""

    def __init__(self, map, state, which, wrapper, bootstrap, bind_to):
        """Initialize an OverloadCall."""
        self.map = map
        self.state = state
        self.which = which
        self.wrapper = wrapper
        self.bind_to = self if bootstrap else bind_to

    def __getitem__(self, t):
        return self.map[t].__get__(self)

    def __xcall_bind_wrap__(self, *args, **kwargs):
        main = args[self.which - 1]
        method = self.map[type(main)]
        return self.wrapper(method, self.bind_to, *args, **kwargs)

    def __xcall_bind__(self, *args, **kwargs):
        main = args[self.which - 1]
        method = self.map[type(main)]
        return method(self.bind_to, *args, **kwargs)

    def __xcall_wrap__(self, *args, **kwargs):
        main = args[self.which]
        method = self.map[type(main)]
        return self.wrapper(method, *args, **kwargs)

    def __xcall__(self, *args, **kwargs):
        main = args[self.which]
        method = self.map[type(main)]
        return method(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        fself = self.bind_to
        if fself is not None:
            args = (fself,) + args

        main = args[self.which]
        method = self.map[type(main)]

        if self.wrapper is None:
            return method(*args, **kwargs)
        else:
            return self.wrapper(method, *args, **kwargs)


def _find_overload(fn, bootstrap, initial_state, postprocess):
    mod = __import__(fn.__module__, fromlist="_")
    dispatch = getattr(mod, fn.__name__, None)
    if dispatch is None:
        dispatch = _fresh(Overload)(
            bootstrap=bootstrap,
            initial_state=initial_state,
            postprocess=postprocess,
        )
    else:  # pragma: no cover
        assert bootstrap is False
        assert initial_state is None
        assert postprocess is None
    if not isinstance(dispatch, Overload):  # pragma: no cover
        raise TypeError("@overload requires Overload instance")
    return dispatch


@keyword_decorator
def overload(fn, *, bootstrap=False, initial_state=None, postprocess=None):
    """Overload a function.

    Overloading is based on the function name.

    The decorated function should have one parameter annotated with a type.
    Any parameter can be annotated, but only one, and every overloading of a
    function should annotate the same parameter.

    The decorator optionally takes keyword arguments, *only* on the first
    use.

    Arguments:
        fn: The function to register.
        bootstrap: Whether to bootstrap this function so that it receives
            itself as its first argument. Useful for recursive functions.
        initial_state: A function with no arguments that returns the initial
            state for top level calls to the overloaded function, or None
            if there is no initial state.
        postprocess: A function to transform the result. Not called on the
            results of recursive calls.

    """
    dispatch = _find_overload(fn, bootstrap, initial_state, postprocess)
    return dispatch.register(fn)


@keyword_decorator
def overload_wrapper(
    wrapper, *, bootstrap=False, initial_state=None, postprocess=None
):
    """Overload a function using the decorated function as a wrapper.

    The wrapper is the entry point for the function and receives as its
    first argument the method to dispatch to, and then the arguments to
    give to that method.

    Arguments:
        wrapper: Function to wrap the dispatch with.
        bootstrap: Whether to bootstrap this function so that it receives
            itself as its first argument. Useful for recursive functions.
        initial_state: A function with no arguments that returns the initial
            state for top level calls to the overloaded function, or None
            if there is no initial state.
        postprocess: A function to transform the result. Not called on the
            results of recursive calls.

    """
    dispatch = _find_overload(wrapper, bootstrap, initial_state, postprocess)
    return dispatch.wrapper(wrapper)


overload.wrapper = overload_wrapper


def rename_function(fn, newname):
    """Create a copy of the function with a different name."""
    co = fn.__code__
    newcode = type(co)(
        co.co_argcount,
        co.co_kwonlyargcount,
        co.co_nlocals,
        co.co_stacksize,
        co.co_flags,
        co.co_code,
        co.co_consts,
        co.co_names,
        co.co_varnames,
        co.co_filename,
        newname,
        co.co_firstlineno,
        co.co_lnotab,
        co.co_freevars,
        co.co_cellvars,
    )
    return FunctionType(
        newcode, fn.__globals__, newname, fn.__defaults__, fn.__closure__
    )


__consolidate__ = True
__all__ = ["Overload", "TypeMap", "overload", "overload_wrapper"]
