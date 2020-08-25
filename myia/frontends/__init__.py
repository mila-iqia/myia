"""Frontends."""

import importlib

import pkg_resources


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


def collect_frontend_plugins():
    """Collect frontend plugins.

    Look for entry points in namespace "myia.frontend".
    Each entry point must be a frontend module.

    :return: a dictionary mapping a frontend name to a loader function
        to import frontend module.
    """
    return {
        entry_point.name: import_mod(entry_point.module_name)
        for entry_point in pkg_resources.iter_entry_points("myia.frontend")
    }


_frontends = collect_frontend_plugins()


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
