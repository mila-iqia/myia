"""Compilation backends."""

import importlib
import os
import urllib
import weakref

import pkg_resources

from ... import abstract, xtype


class UnknownBackend(Exception):
    """Indicates that the backend name is not recognized."""


class LoadingError(Exception):
    """Indicates that there was an error loading the backend.

    This can happen because of missing dependencies.  There should be
    a chained exception with the original error that comes with it.
    """


class BackendLoader:
    """Utility class to load a backend."""

    __slots__ = ("load_options", "load_backend")

    def __init__(self, load_fn, defaults_fn):
        """Create a backend loader from given functions.

        :param load_fn: function(backend_options): must take
          a dictionary of valid backend options and return a new instance
          of backend. Used to effectively load the backend
          if not already in cache.
        :param defaults_fn: function(**backend_options): must check
          backend options and return a dictionary with valid options.
          Used to cache loaded backends.
        """
        self.load_options = defaults_fn
        self.load_backend = load_fn

    @classmethod
    def loader_callable_from_pkg(cls, pkg):
        """Return a function that creates a new backend loader.

        :param pkg: module name (example myia.compile.backends.relay).
            Module must provide 2 functions:
            - `load_options` for `__init__`'s `default_fn` parameter
            - `load_backend` for `__init__`'s `load_fn` parameter

        :return: a callable (with no arguments) that will generate
            and return a BackendLoader object.
        """

        def loader():
            module = importlib.import_module(pkg)
            load_options = getattr(module, "load_options")
            load_backend = getattr(module, "load_backend")
            return cls(load_fn=load_backend, defaults_fn=load_options)

        return loader

    @classmethod
    def loader_callable_from_functions(cls, load_fn, defaults_fn):
        """Return a function that creates a new backend loader.

        for more details about load_fn and defaults_fn.
        :return: a callable (with no arguments) that will generate
        and return a BackendLoader object.
        """

        def loader():
            return cls(load_fn=load_fn, defaults_fn=defaults_fn)

        return loader


def collect_backend_plugins():
    """Collect backend plugins.

    Look for entry points in namespace "myia.backend".
    Each entry point must be a backend module.
    From a backend module we must be able to import two functions:

    - `load_options(**backend_options)`: must check backend options and return
      a dictionary with valid options. Used to cache loaded backends.
    - `load_backend(backend_options)`: must take
      a dictionary of valid backend options and return a new instance
      of backend. Used to effectively load the backend
      if not already in cache.

    :return: a dictionary mapping a backend name to a loader function
        to generate BackendLoader instances.
    """
    return {
        entry_point.name: BackendLoader.loader_callable_from_pkg(
            entry_point.module_name
        )
        for entry_point in pkg_resources.iter_entry_points("myia.backend")
    }


_backends = collect_backend_plugins()

_active_backends = weakref.WeakValueDictionary()


def get_default():
    """Returns the default backend.

    This is fetched from the MYIA_BACKEND environement variable or
    from the built-in defaults.

    The syntax for specifiying a backend is
    'name?option1=value1&option2=value2' when name is the name of the
    backend and option1 is a valid keyword option for that backend.
    This is strongly inspired by HTTP query syntax except you don't
    need to urlencode values.

    As an example here is a string for pytorch on the GPU

        pytorch?target=cuda:0

    """
    backend, opts = parse_default()
    assert backend is not None
    return load_backend(backend, opts)


def parse_default():
    """Parses the default backend.

    Returns name and options from the environement or builtin default.
    See the documentation of get_default() for the backend string syntax.
    """
    backend_spec = os.environ.get("MYIA_BACKEND", "pytorch")
    backend, *opts = backend_spec.split("?", maxsplit=1)
    if len(opts) == 1:
        opts = urllib.parse.parse_qs(
            opts[0],
            keep_blank_values=True,
            strict_parsing=True,
            errors="strict",
        )
        for k in opts:
            assert len(opts[k]) == 1
            opts[k] = opts[k][0]
    else:
        assert len(opts) == 0
        opts = {}
    return backend, opts


def load_backend(name, options=None):
    """Load the named backend.

    Returns the backend class registered for the name.

    If you pass None as the name, this will load the default backend.
    See the documenation for get_default() for more information.

    Raises:
        UnknownBackend: The name is not recognized.
        LoadingError: There was an error loading the backend.

    """
    if name is None:
        assert options is None
        return get_default()
    if options is None:
        options = {}
    if name not in _backends:
        raise UnknownBackend(name)
    backend_loader = _backends[name]()
    options = backend_loader.load_options(**options)
    key = (name, tuple(sorted(list(options.items()))))
    res = _active_backends.get(key, None)
    if res is None:
        try:
            res = backend_loader.load_backend(options)
            _active_backends[key] = res
        except Exception as e:
            raise LoadingError(name) from e
    return res


def register_backend(name, load_fn, defaults_fn):
    """Register a new backend.

    This is to allow third party libraries to register their own
    backends if loaded by the user.  Built-in backends are
    preregistered.

    Arguments:
        name (str): Name of the backend, must be unique
        load_fn: function that will load the backend.  This must
                 return a callable that will take keyword arguemnts
                 for options.
        defaults_fn: function that takes the same default arguments as
                     load_fn and maps them to canonical and/or default
                     values.

    """
    assert name not in _backends
    _backends[name] = BackendLoader.loader_callable_from_functions(
        load_fn, defaults_fn
    )


class Backend:
    """This is a class interface that all backends must implement."""

    def compile(self, graph, argspec, outspec):
        """Compile the group of graphs rooted at `graph`.

        This function takes in a fully typed graph cluster rooted at
        `graph` with a manager and must return a callable that accepts
        arguments of the same type and number as the root graph.
        """
        raise NotImplementedError("compile")

    def from_backend_value(self, v, t):
        """Convert a backend value to an intermediate value."""
        raise NotImplementedError("from_backend_value")

    def to_backend_value(self, v, t):
        """Convert an intermediate value to a backend value."""
        raise NotImplementedError("to_backend_value")


class Converter:
    """Converts values between representations for backends."""

    def convert_array(self, v, t):
        """Converts array values."""
        raise NotImplementedError("convert_numpy")

    def convert_scalar(self, v, t):
        """Convert numeric scalars."""
        raise NotImplementedError("convert_scalar")

    def convert_nil(self, v, t):
        """Convert Nil values."""
        raise NotImplementedError("convert_nil")

    def convert_dead(self, v, t):
        """Convert dead values."""
        raise NotImplementedError("convert_dead")

    def convert_bool(self, v, t):
        """Convert boolean values."""
        raise NotImplementedError("convert_bool")

    def convert_env(self, v, t):
        """Convert a grad env."""
        raise NotImplementedError("convert_env")

    def convert_universe(self, v, t):
        """Convert a Universe."""
        raise NotImplementedError("convert_universe")

    def convert_handle(self, v, t):
        """Convert a Handle."""
        raise NotImplementedError("convert_handle")

    def convert_tuple(self, v, t):
        """Convert a tuple."""
        raise NotImplementedError("convert_tuple")

    def convert_tagged(self, v, t):
        """Convert a union value."""
        raise NotImplementedError("convert_tagged")

    def convert_type(self, v, t):
        """Convert a type value."""
        raise NotImplementedError("convert_type")

    def convert_random_state(self, v, t):
        """Convert a random state value."""
        raise NotImplementedError("convert_type")

    def __call__(self, v, t):
        """Convert a value."""
        if v is abstract.DEAD:
            return self.convert_dead(v, t)
        elif isinstance(t, abstract.AbstractArray):
            return self.convert_array(v, t)
        elif isinstance(t, abstract.AbstractRandomState):
            return self.convert_random_state(v, t)
        elif isinstance(t, abstract.AbstractScalar):
            if issubclass(t.xtype(), xtype.Number):
                return self.convert_scalar(v, t.xtype())
            elif issubclass(t.xtype(), xtype.Bool):
                return self.convert_bool(v, t.xtype())
            elif issubclass(t.xtype(), xtype.Nil):
                return self.convert_nil(v, t.xtype())
            elif issubclass(t.xtype(), xtype.EnvType):
                return self.convert_env(v, t.xtype())
            elif issubclass(t.xtype(), xtype.UniverseType):
                return self.convert_universe(v, t.xtype())
            else:
                raise NotImplementedError(f"convert for scalar {t.xtype()}")
        elif isinstance(t, abstract.AbstractTuple):
            return self.convert_tuple(v, t)
        elif isinstance(t, abstract.AbstractTaggedUnion):
            return self.convert_tagged(v, t)
        elif isinstance(t, abstract.AbstractType):
            return self.convert_type(v, t)
        elif isinstance(t, abstract.AbstractHandle):
            return self.convert_handle(v, t)
        else:
            raise NotImplementedError(f"convert for {t}")
