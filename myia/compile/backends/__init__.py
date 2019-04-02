"""Compilation backends."""

import importlib
from ... import dtype
from ...dtype import ismyiatype


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

# This is used as the default backend and options if None is provided
default_name = 'nnvm'


def load_backend(name):
    """Load the named backend.

    Returns the backend class registered for the name.

    Raises:
        UnknownBackend: The name is not recognized.
        LoadingError: There was an error loading the backend.

    """
    if name is None:
        name = default_name
    if name not in _backends:
        raise UnknownBackend(name)
    try:
        res = _backends[name]()
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

    def convert_value(self, v, t):
        """Convert a value to the appropriate backend representation."""
        if ismyiatype(t, dtype.Number):
            return self.from_scalar(v, t)
        elif ismyiatype(t, dtype.Tuple):
            return tuple(self.convert_value(v, t)
                         for v, t in zip(v, t.elements))
        elif ismyiatype(t, dtype.Array):
            return self.from_numpy(v)
        else:
            raise NotImplementedError(f'convert_value for {t}')

    def check_array(self, v, t):
        """Check array value for type and element dtype.

        This must raise exceptions describing the problem encountered.
        """
        raise NotImplementedError('check_array')
