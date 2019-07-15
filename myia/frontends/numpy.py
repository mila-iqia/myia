"""Numpy Frontend."""

# import numpy as np

from ..abstract.infer import to_abstract
from ..api import _convert_arg_init
from ..pipeline.steps import convert_arg
from ..pipeline.steps import convert_result

from . import Frontend


class NumpyFrontend(Frontend):
    """Frontend to run using Numpy.

    Frontend options:


    """

    def __init__(self):
        """Create a Numpy frontend."""
        pass

    to_abstract = staticmethod(to_abstract)
    _convert_arg_init = staticmethod(_convert_arg_init)
    convert_result = staticmethod(convert_result)
    convert_arg = staticmethod(convert_arg)

    def configure(self, pip):
        """Additional configuration of pipeline for Numpy frontend."""
        # Current pipeline for Numpy frontend doesn't need any additional
        # configuration, so 'configure' just returns the pipeline (pip).
        return pip
