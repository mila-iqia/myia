"""Frontends."""

import importlib


class UnknownFrontend(Exception):
    """Indicates that the frontend name is not recognized."""


class FrontendLoadingError(Exception):
    """Indicates that there was an error loading the frontend.

    This can happen because of missing dependencies.  There should be
    a chained exception with the original error that comes with it.
    """


def import_load(pkg, name):
    """Helper function for simple frontends.

    This will return a callable that will load a module, retrieve a
    object from its namespace and return that.

    """
    def loader():
        mod = importlib.import_module(pkg)
        return getattr(mod, name)
    return loader


_frontends = {
    'numpy': import_load('myia.frontends.numpy', 'NumpyFrontend'),
    'pytorch': import_load('myia.frontends.pytorch', 'PyTorchFrontend'),
}


def load_frontend(name, options=None):
    """Load the named frontend.

    Returns the frontend class registered for the name.

    Raises:
        UnknownFrontend: The name is not recognized.
        FrontendLoadingError: There was an error loading the frontend.

    """
    if name is None:
        name = 'numpy'
    if options is None:
        options = {}
    if name not in _frontends:
        raise UnknownFrontend(name)
    try:
        res = _frontends[name]()(**options)
    except Exception as e:
        raise FrontendLoadingError(name) from e
    return res


def register_frontend(name, load_fn):
    """Register a new frontend.

    This is to allow third party libraries to register their own
    frontends if loaded by the user.  Built-in frontends are
    preregistered.

    """
    assert name not in _frontends
    _frontends[name] = load_fn


class Frontend:
    """This is a class interface that all frontends must implement."""

    def configure(self, pip):
        """Additional configuration of pipeline for Frontend."""
        return NotImplementedError("configure")
