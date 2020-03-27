"""Frontends."""

import importlib


class UnknownFrontend(Exception):
    """Indicates that the frontend name is not recognized."""


class FrontendLoadingError(Exception):
    """Indicates that there was an error loading the frontend.

    This can happen because of missing dependencies.  There should be
    a chained exception with the original error that comes with it.
    """


def import_mod(pkg):
    """Helper function for simple frontends.

    This will return a callable that will load a module.
    """

    def loader():
        importlib.import_module(pkg)

    return loader


_frontends = {"pytorch": import_mod("myia.frontends.pytorch")}


def activate_frontend(name):
    """Activates the named frontend.

    Raises:
        UnknownFrontend: The name is not recognized.
        FrontendLoadingError: There was an error loading the frontend.

    """
    if name not in _frontends:
        raise UnknownFrontend(name)
    try:
        _frontends[name]()
    except Exception as e:
        raise FrontendLoadingError(name) from e
