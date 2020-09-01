"""Myia Pytorch backend.

   isort:skip_file
"""
__version__ = "0.1.0"

from .pytorch import load_backend, load_options
from myia.testing.multitest import register_backend_testing


register_backend_testing("pytorch", "cpu", {"device": "cpu"})
register_backend_testing("pytorch", "gpu", {"device": "cuda"}, "pytorch-cuda")
