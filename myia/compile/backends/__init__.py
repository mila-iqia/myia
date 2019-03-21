import importlib


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
}

# This is used as the default backend and options if None is provided
default_name = 'nnvm'


def load_backend(name):
    """Load the named backend.

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

    def compile(self, graph):
        """Compile the group of graphs rooted at `graph`.

        This function takes in a fully typed graph cluster rooted at
        `graph` with a manager and must return a callable that accepts
        arguments of the same type and number as the root graph.
        """
        raise NotImplementedError('compile')

    def from_numpy(self, a):
        """
        Convert a numpy ndarray to a backend-appropriate value.
        """
        raise NotImplementedError("from_numpy")

    def to_numpy(self, v):
        """
        Convert a backlend value to a numpy.ndarray.
        """
        raise NotImplementedError("to_numpy")

    def from_dlpack(self, dlp):
        """
        Convert a value from a DLpack PyCapsule to a backend-appropriate value.
        """
        raise NotImplementedError('from_dlpack')

    def to_dlpack(self, v):
        """
        Convert a backend-specific tensor to a DLpack PyCapsule.
        """
        raise NotImplementedError('to_dlpack')
