"""Compilation backends."""

import importlib
import os
import urllib

from ... import abstract, xtype
from ...utils import TaggedValue


class UnknownBackend(Exception):
    """Indicates that the backend name is not recognized."""


class LoadingError(Exception):
    """Indicates that there was an error loading the backend.

    This can happen because of missing dependencies.  There should be
    a chained exception with the original error that comes with it.
    """


def import_load(pkg, name):
    """Helper function for simple backends.

    This will return a callable that will load a module, retrieve a
    object from its namespace and return that.

    """
    def loader():
        mod = importlib.import_module(pkg)
        return getattr(mod, name)
    return loader


_backends = {
    'nnvm': import_load('myia.compile.backends.nnvm', 'NNVMBackend'),
    'relay': import_load('myia.compile.backends.relay', 'RelayBackend'),
    'pytorch': import_load('myia.compile.backends.pytorch', 'PyTorchBackend'),
}

_active_backends = {}


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
    backend_spec = os.environ.get('MYIA_BACKEND', 'nnvm')
    backend, *opts = backend_spec.split('?', maxsplit=1)
    if len(opts) == 1:
        opts = urllib.parse.parse_qs(opts[0], keep_blank_values=True,
                                     strict_parsing=True, errors='strict')
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
    key = (name, tuple(sorted(list(options.items()))))
    res = _active_backends.get(key, None)
    if res is None:
        try:
            res = _backends[name]()(**options)
            _active_backends[key] = res
        except Exception as e:
            raise LoadingError(name) from e
    return res


def register_backend(name, load_fn):
    """Register a new backend.

    This is to allow third party libraries to register their own
    backends if loaded by the user.  Built-in backends are
    preregistered.

    """
    assert name not in _backends
    _backends[name] = load_fn


class Backend:
    """This is a class interface that all backends must implement."""

    def compile(self, graph, argspec, outspec, pipeline):
        """Compile the group of graphs rooted at `graph`.

        This function takes in a fully typed graph cluster rooted at
        `graph` with a manager and must return a callable that accepts
        arguments of the same type and number as the root graph.
        """
        raise NotImplementedError('compile')

    def from_numpy(self, a):
        """Convert a numpy ndarray to a backend value."""
        raise NotImplementedError("from_numpy")

    def to_numpy(self, v):
        """Convert a backend value to a numpy.ndarray."""
        raise NotImplementedError("to_numpy")

    def from_scalar(self, s, t):
        """Convert the python scalar to a backend value with explicit type."""
        raise NotImplementedError('from_scalar')

    def to_scalar(self, v):
        """Convert the backend value to a python scalar."""
        raise NotImplementedError('to_scalar')

    def empty_env(self):
        """An empty grad environment for the backend."""
        return ()

    def from_backend_value(self, v, t):
        """Convert a backend value to an intermediate value."""
        if isinstance(t, abstract.AbstractScalar):
            return self.to_scalar(v)
        elif isinstance(t, abstract.AbstractArray):
            res = self.to_numpy(v)
            # Some backends will use 1d instead of 0d for internal reasons.
            if res.shape != t.values[abstract.SHAPE]:
                res = res.reshape(t.values[abstract.SHAPE])
            return res
        elif isinstance(t, abstract.AbstractTuple):
            return tuple(self.from_backend_value(ve, te)
                         for ve, te in zip(v, t.elements))
        elif isinstance(t, abstract.AbstractTaggedUnion):
            return TaggedValue(v.tag, self.from_backend_value(
                v.value, t.options.get(v.tag)))
        else:
            raise NotImplementedError(f"Don't know what to do for {t}")

    def to_backend_value(self, v, t):
        """Convert an intermediate value to a backend value."""
        from ..utils import BackendValue
        if isinstance(v, BackendValue):
            assert v.backend is self
            return v.value
        if (isinstance(t, (abstract.AbstractError, abstract.AbstractType))
                or v is abstract.DEAD):
            return None
        if isinstance(t, abstract.AbstractArray):
            return self.from_numpy(v)
        elif isinstance(t, abstract.AbstractScalar):
            if issubclass(t.values[abstract.TYPE],
                          (xtype.Number, xtype.Bool, xtype.Nil)):
                return self.from_scalar(v, t.values[abstract.TYPE])
            elif issubclass(t.values[abstract.TYPE], xtype.EnvType):
                assert len(v._contents) == 0
                return self.empty_env()
            else:
                raise NotImplementedError(f'from_value for {t}')
        elif isinstance(t, abstract.AbstractTuple):
            return tuple(self.to_backend_value(v, t)
                         for v, t in zip(v, t.elements))
        elif isinstance(t, abstract.AbstractTaggedUnion):
            real_t = t.options.get(v.tag)
            return TaggedValue(v.tag, self.to_backend_value(v.value, real_t))
        else:
            raise NotImplementedError(f'from_value for {t}')
