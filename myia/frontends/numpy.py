"""Numpy Frontend."""

from . import Frontend


class NumpyFrontend(Frontend):
    """Frontend to run using Numpy.

    Frontend options:


    """

    def __init__(self):
        """Create a Numpy frontend."""
        pass

    def configure(self, pip):
        """Additional configuration of pipeline for Numpy frontend."""
        # Current pipeline for Numpy frontend doesn't need any additional
        # configuration, so 'configure' just returns the pipeline (pip).
        return pip
