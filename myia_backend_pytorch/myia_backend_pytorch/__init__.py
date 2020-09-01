__version__ = "0.1.0"

from myia.testing.multitest import register_backend_testing

from .pytorch import load_backend, load_options

register_backend_testing("pytorch", "cpu", {"device": "cpu"})
register_backend_testing("pytorch", "gpu", {"device": "cuda"}, "pytorch-cuda")
