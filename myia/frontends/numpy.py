"""Numpy Frontend."""

#import numpy

from ..abstract.infer import to_abstract
from ..pipeline.steps import convert_result
from ..pipeline.steps import convert_arg

from . import Frontend

class NumpyFrontend(Frontend):
    """Frontend to run using Numpy.

    Frontend options:
        

    """
    def __init__(self):
        pass

    to_abstract = staticmethod(to_abstract)
    convert_result = staticmethod(convert_result)
    convert_arg = staticmethod(convert_arg)

    def configure(self, pip):
        return pip