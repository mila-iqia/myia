"""Utilities to overload functions for multiple types."""


import inspect

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
        else:
            raise KeyError(obj_t)


class Overload:
    """Overloaded function.

    A function can be added with the `register` method. One of its parameters
    should be annotated with a type, but only one, and every registered
    function should annotate the same parameter.

    Arguments:
        bind_to: Binds the first argument to the given object.
        wrapper: A function to use as the entry point. In addition to all
            normal arguments, it will receive as its first argument the
            function to dispatch to.
        initial_state: A function returning the initial state, or None if
            there is no state.
        mixins: A list of Overload instances that contribute functions to this
            Overload.
    """

    def __init__(self,
                 *,
                 bind_to=None,
                 wrapper=None,
                 initial_state=None,
                 mixins=[],
                 name=None,
                 _parent=None):
        """Initialize an Overload."""
        if bind_to is True:
            bind_to = self
        elif bind_to is False:
            bind_to = None
        self.__self__ = bind_to
        self._parent = _parent
        self._wrapper = wrapper
        self.state = None
        self.initial_state = initial_state
        self.name = name
        if _parent:
            assert _parent.which is not None
            self.map = _parent.map
            self._uncached_map = _parent._uncached_map
            self.which = _parent.which
            self._wrapper = _parent._wrapper
            return
        _map = {}
        self.which = None
        for mixin in mixins:
            if self.which is None:
                self.which = mixin.which
            else:
                assert mixin.which == self.which
            _map.update(mixin._uncached_map)
        self.map = TypeMap(_map)
        self._uncached_map = _map

    def _set_attrs_from(self, fn, wrapper=False):
        if self.name is None:
            self.name = f'{fn.__module__}.{fn.__qualname__}'
            self.__doc__ = fn.__doc__
            self.__name__ = fn.__name__
            self.__qualname__ = fn.__qualname__
            self.__module__ = fn.__module__
            sign = inspect.signature(fn)
            params = list(sign.parameters.values())
            if wrapper:
                params = params[1:]
            if self.__self__ is not None:
                params = params[1:]
            params = [p.replace(annotation=inspect.Parameter.empty)
                      for p in params]
            self.__signature__ = sign.replace(parameters=params)

    def wrapper(self, wrapper):
        """Set a wrapper function."""
        if self._wrapper is not None:
            raise TypeError(f'wrapper for {self} is already set')
        self._wrapper = wrapper
        self._set_attrs_from(wrapper, wrapper=True)
        return self

    def register(self, fn):
        """Register a function."""
        if self._parent:  # pragma: no cover
            raise Exception('Cannot register a function on derived Overload')
        ann = fn.__annotations__
        if len(ann) != 1:
            raise Exception('Only one parameter may be annotated.')
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
                        'Annotation must always be on same parameter'
                    )
                break

        ts = t if isinstance(t, tuple) else (t,)

        for t in ts:
            self.map[t] = fn
            self._uncached_map[t] = fn

        self._set_attrs_from(fn)
        return self

    def copy(self, wrapper=MISSING, initial_state=None):
        """Create a copy of this Overload.

        New functions can be registered to the copy without affecting the
        original.
        """
        fself = self.__self__
        bootstrap = True if fself is self else fself
        return Overload(
            bind_to=bootstrap,
            wrapper=self._wrapper if wrapper is MISSING else wrapper,
            mixins=[self],
            initial_state=initial_state or self.initial_state
        )

    def variant(self, fn=None, *, wrapper=MISSING, initial_state=None):
        """Decorator to create a variant of this Overload.

        New functions can be registered to the variant without affecting the
        original.
        """
        ov = self.copy(wrapper, initial_state)
        if fn is None:
            return ov.register
        else:
            ov.register(fn)
            return ov

    def variant_wrapper(self, wrapper=MISSING, *, initial_state=None):
        """Decorator to create a variant of this Overload with a new wrapper.

        New functions can be registered to the variant without affecting the
        original.
        """
        ov = self.copy(wrapper=None, initial_state=initial_state)
        if wrapper is MISSING:
            return ov.wrapper
        else:
            return ov.wrapper(wrapper)

    def __get__(self, obj, cls):
        return Overload(bind_to=obj, _parent=self, name=self.name)

    def __getitem__(self, t):
        if self.__self__:
            return self.map[t].__get__(self.__self__)
        else:
            return self.map[t]

    def __call__(self, *args, **kwargs):
        """Call the overloaded function."""
        if self.initial_state is not None:
            ov = Overload(bind_to=True, _parent=self)
            ov.state = self.initial_state()
            return ov(*args, **kwargs)

        fself = self.__self__
        if fself is not None:
            args = (fself,) + args

        main = args[self.which]

        try:
            method = self.map[type(main)]
        except KeyError:
            method = None

        if self._wrapper is None:
            if method is None:
                raise TypeError(
                    f'No overloaded method in {self} for {type(main)}'
                )
            else:
                return method(*args, **kwargs)
        else:
            return self._wrapper(method, *args, **kwargs)

    def __repr__(self):
        return f'<Overload {self.name or hex(id(self))}>'


def _find_overload(fn, bootstrap, initial_state):
    mod = __import__(fn.__module__, fromlist='_')
    dispatch = getattr(mod, fn.__name__, None)
    if dispatch is None:
        dispatch = Overload(bind_to=bootstrap, initial_state=initial_state)
    else:  # pragma: no cover
        assert bootstrap is False
        assert initial_state is None
    if not isinstance(dispatch, Overload):  # pragma: no cover
        raise TypeError('@overload requires Overload instance')
    return dispatch


@keyword_decorator
def overload(fn, *, bootstrap=False, initial_state=None):
    """Overload a function.

    Overloading is based on the function name.

    The decorated function should have one parameter annotated with a type.
    Any parameter can be annotated, but only one, and every overloading of a
    function should annotate the same parameter.

    The decorator optionally takes keyword arguments, *only* on the first
    use.

    Arguments:
        bootstrap: Whether to bootstrap this function so that it receives
            itself as its first argument. Useful for recursive functions.
        initial_state: A function with no arguments that returns the initial
            state for top level calls to the overloaded function, or None
            if there is no initial state.
    """
    dispatch = _find_overload(fn, bootstrap, initial_state)
    return dispatch.register(fn)


@keyword_decorator
def overload_wrapper(wrapper, *, bootstrap=False, initial_state=None):
    """Overload a function using the decorated function as a wrapper.

    The wrapper is the entry point for the function and receives as its
    first argument the method to dispatch to, and then the arguments to
    give to that method.

    Arguments:
        bootstrap: Whether to bootstrap this function so that it receives
            itself as its first argument. Useful for recursive functions.
        initial_state: A function with no arguments that returns the initial
            state for top level calls to the overloaded function, or None
            if there is no initial state.
    """
    dispatch = _find_overload(wrapper, bootstrap, initial_state)
    return dispatch.wrapper(wrapper)


overload.wrapper = overload_wrapper
