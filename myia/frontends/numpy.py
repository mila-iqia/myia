"""Numpy Frontend."""

#import numpy

from ..abstract.infer import to_abstract

from . import Frontend

class NumpyFrontend(Frontend):
    """Frontend to run using Numpy.

    Frontend options:
        

    """
    def __init__(self):
        pass

    to_abstract = staticmethod(to_abstract)