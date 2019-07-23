"""Compilation backends."""

import importlib
import os
import urllib

from ... import dtype, abstract


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
    backend, opts = parse_env()
    if backend is None:
        for backend in _backends.keys():
            try:
                return load_backend(backend)
            except LoadingError:
                pass
        raise LoadingError("No backends available.")
    return load_backend(backend, opts)


def parse_env():
    """Parses the environement-specified backend (if any)

    Returns name and options from the environement. See the
    documentation of get_default() for the backend string syntax.
    """
    backend_spec = os.environ.get('MYIA_BACKEND', None)
    if backend_spec is None:
        return None, {}
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
    try:
        res = _backends[name]()(**options)
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

    def from_dlpack(self, dlp):
        """Convert a value from a DLpack PyCapsule to a backend value."""
        raise NotImplementedError('from_dlpack')

    def to_dlpack(self, v):
        """Convert a backend-specific tensor to a DLpack PyCapsule."""
        raise NotImplementedError('to_dlpack')

    def empty_env(self):
        """An empty grad environment for the backend."""
        return ()

    def convert_value(self, v, t):
        """Convert a value to the appropriate backend representation."""
        if isinstance(t, abstract.AbstractError) or v is abstract.DEAD:
            return None
        elif isinstance(t, abstract.AbstractScalar):
            if issubclass(t.values[abstract.TYPE],
                          (dtype.Number, dtype.Bool, dtype.Nil)):
                return self.from_scalar(v, t.values[abstract.TYPE])
            elif issubclass(t.values[abstract.TYPE], dtype.EnvType):
                assert len(v._contents) == 0
                return self.empty_env()
            else:
                raise NotImplementedError(f'convert_value for {t}')
        elif isinstance(t, abstract.AbstractTuple):
            return tuple(self.convert_value(v, t)
                         for v, t in zip(v, t.elements))
        else:
            raise NotImplementedError(f'convert_value for {t}')

    def check_array(self, v, t):
        """Check array value for type and element dtype.

        This must raise exceptions describing the problem encountered.
        """
        raise NotImplementedError('check_array')

    def configure(self, pip):
        """Configure the pipeline for the backend needs."""
        return pip.configure({'backend': self})
